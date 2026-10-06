"""장면 통합 메시를 정합하고 복셀을 음식별로 배정해 부피를 낸다.

절차:
  1) EXIF 카메라로 투영해 **접시+음식 합집합 실루엣**에 맞춘다.
     물체 하나짜리보다 제약이 강하다 -- 접시 테두리가 자세를 강하게 묶는다.
  2) 배율은 접시 지름 사전값이 정한다. 메시 안에서 접시가 차지하는 지름을
     재고 그것이 D_prior 가 되도록 맞춘다. 그러면 **장면 전체가 한 배율**로
     정해지고 음식들의 상대 크기가 자동으로 일관된다.
  3) 복셀화 후 각 복셀 중심을 이미지로 투영해 어느 음식 마스크에 떨어지는지로 배정.
     접시 평면 아래는 접시 자체이므로 버린다.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backends.depth import build                   # noqa: E402
from geometry import fit_plane_ransac              # noqa: E402
from paths import CACHE, ID_MAP, OUTPUTS, combo_images  # noqa: E402
from intrinsics import from_exif                   # noqa: E402
from scale_plate import unpack, hull_extent        # noqa: E402
from diag_plate_scale import largest_component     # noqa: E402
from segment import resized                        # noqa: E402
from pipeline import plate_distance_mm             # noqa: E402
import pose_project as pp                          # noqa: E402
from gen_scene import scene_masks                  # noqa: E402

BASE = 2016


def main(voxel_mm=2.5, out="scene_volumes.csv"):
    be = build("depthpro")
    tbl = {}
    for r in csv.DictReader(open(ID_MAP)):
        tbl.setdefault(int(r["combo"]), []).append((int(r["id"]), r["food"]))
    rows = []
    print(f"{'c':>3} {'IoU':>5} {'접시mm':>7} {'배율':>9}  음식별 부피")
    for c in range(1, 15):
        mp = OUTPUTS / "meshes_scene" / f"scene_c{c:02d}.glb"
        if not mp.exists():
            continue
        t0 = time.time()
        plate_m, foods = scene_masks(c)
        z_mm, px_major, f_px, _ = plate_distance_mm(c)
        D_mm = z_mm * px_major / f_px

        img = resized(combo_images()[c], BASE)
        W, H = img.size
        sc = 1200 / max(W, H)
        im2 = img.resize((round(W * sc), round(H * sc)), Image.LANCZOS)
        K = from_exif(combo_images()[c])
        fov = float(2 * np.degrees(np.arctan(W / (2 * K.fx * (BASE / K.width)))))
        pts, valid = be.points(im2, fov)
        rm = lambda m: np.asarray(Image.fromarray(m.astype(np.uint8) * 255)
                                  .resize(im2.size, Image.NEAREST)) > 127
        pmask = rm(plate_m) & valid
        P = pts[pmask]; P = P[np.isfinite(P).all(1)]
        if len(P) < 1000:
            continue
        sdep = z_mm / float(np.median(P[:, 2]))
        P = P * sdep
        pl = fit_plane_ransac(P[np.random.default_rng(0).choice(len(P), min(40000, len(P)), replace=False)],
                              iters=500, thresh_mm=3.0)

        union = plate_m.copy()
        for m in foods.values():
            union |= m
        umask = rm(union)

        mesh = trimesh.load(mp, force="mesh", process=False)
        # 중심을 한 번만 정하고 표본과 복셀에 **같은** 값을 쓴다.
        # 이전에 mesh.sample() 을 두 번 따로 불러 중심이 어긋났고 부피가 10배 작아졌다.
        raw = np.asarray(mesh.sample(20000), dtype=np.float64)
        ctr = raw.mean(0)
        mpts = raw - ctr
        f_crop = K.fx * (BASE / K.width) * sc
        anchor = P.mean(0)
        r = pp.search(mpts, umask, pl, anchor, f_crop, im2.width/2, im2.height/2,
                      n_rot=300, ss=3)
        if r is None:
            print(f"{c:>3}  자세 실패"); continue
        iou, R, s = pp.refine(mpts, umask, pl, anchor, f_crop, im2.width/2,
                              im2.height/2, r[1], r[2], ss=3)

        # 복셀화 후 배정
        pitch = max(float(mesh.extents.max()) / 220.0, 1e-6)
        vg = mesh.voxelized(pitch=pitch).fill()      # 껍질이 아니라 속을 채운다
        vc = np.asarray(vg.points, dtype=np.float64) - ctr
        V = pp.place(vc, R, s, pl, anchor)
        # place() 가 s 를 곱하므로 복셀 한 변은 pitch*s (mm). 부피는 세제곱.
        vol_per_voxel = (pitch * s) ** 3 * 1e-3
        u_, v_ = pp.project(V, f_crop, im2.width/2, im2.height/2)
        iu = np.clip(u_.astype(int), 0, im2.width-1)
        iv = np.clip(v_.astype(int), 0, im2.height-1)
        above = pl.height(V) > 1.0        # 접시 평면 위만 음식
        got = []
        for fid, fm in sorted(foods.items()):
            fmm = rm(fm)
            sel = above & fmm[iv, iu]
            vol = float(sel.sum()) * vol_per_voxel
            rows.append((c, fid, vol, iou))
            got.append(f"{fid}={vol:.0f}")
        print(f"{c:>3} {iou:>5.2f} {D_mm:>7.0f} {s:>9.4f}  {' '.join(got)}  ({time.time()-t0:.0f}s)",
              flush=True)
    with open(OUTPUTS / out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["combo", "id", "vol_ml", "scene_iou"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.3f}", f"{r[3]:.3f}"])
    print(f"\n  ✔ {OUTPUTS/out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scene_volumes.csv")
    main(out=ap.parse_args().out)
