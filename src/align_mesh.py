"""생성 메시를 실측 기하에 정합해 부피를 낸다. rho 의 두 버그를 함께 없앤다.

앞선 rho 구현에 버그가 두 개 있었다.

  (1) 적분 축이 틀렸다. Hunyuan3D 는 물체를 자기 정준 좌표계로 내놓는데
      그 '위' 가 어디인지 모른 채 z 축을 위로 가정했다. bbox 를 보면 쿠키가
      (1.73, 0.26, 1.99) 로 얇은 축이 y 이고, 셀러리는 긴 축이 z, 당근은 x 라
      물체마다 다르다. 축을 바꿔 재보면 바나나 rho 가 0.48/0.81/0.36 으로
      2 배씩 흔들렸다.

  (2) 그릇 문제를 원리적으로 못 고친다. 생성기에 음식 마스크만 넣었으므로
      메시는 음식만 있고 볼은 없다. 파스타가 932 mL 로 나온 진짜 원인은
      '숨은 형상' 이 아니라 '기준면을 접시로 잡은 것' 인데 rho 는 거기에
      손댈 수 없다.

두 문제가 같은 뿌리다 -- 생성 메시가 장면 안에서 **어떤 자세로 얼마나 크게**
놓이는지를 정하지 않았다. 그래서 정합으로 그걸 정한다.

  자세  : 메시를 여러 방향으로 돌려보고, 접시 평면에 투영한 **실루엣이 실측과
          가장 잘 겹치는 방향**을 고른다 (24개 축정렬 회전 x 면내 36각도).
          렌더-앤-컴페어이며, 깊이 모델의 절대값이 아니라 마스크에 맞춘다.
  크기  : 그 자세에서 메시 발자국 면적이 실측 발자국 면적과 같아지도록 s 를 정한다.
          면적은 길이^2 이므로 s = sqrt(A_obs / A_mesh).
  부피  : V = (메시 자체 부피) x s^3.

이러면 접시 평면을 바닥으로 쓰지 않으므로 **그릇 항목이 자동으로 해결된다.**
볼 안의 파스타든 접시 위의 스테이크든, 완전한 메시를 실측 실루엣 크기에 맞추는
같은 연산이다.
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

from backends.depth import build                          # noqa: E402
from geometry import fit_plane_ransac, heightfield_volume  # noqa: E402
from paths import CACHE, ID_MAP, OUTPUTS, combo_images     # noqa: E402
from intrinsics import from_exif                           # noqa: E402
from scale_plate import hull_extent, unpack, PRIORS_MM     # noqa: E402
from diag_plate_scale import largest_component             # noqa: E402
from segment import resized                                # noqa: E402
from diag_crop import crop_box                             # noqa: E402
from pipeline import plate_distance_mm                     # noqa: E402
import pose_project as pp                                  # noqa: E402

BASE = 2016
GRID = 96          # 실루엣 비교 격자
N_SAMPLE = 60000   # 메시 표면 샘플 수


def axis_rotations():
    """24개 축정렬 회전 (정육면체 대칭군). 자세 탐색의 거친 격자."""
    out = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for sx in (1, -1):
            for sy in (1, -1):
                M = np.zeros((3, 3))
                M[0, perm[0]] = sx
                M[1, perm[1]] = sy
                M[2, perm[2]] = 1.0
                if np.linalg.det(M) < 0:
                    M[2, perm[2]] = -1.0
                out.append(M)
    return out


def inplane(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rasterize(a, b, grid=GRID):
    """평면좌표 점들을 정규화 격자 실루엣으로. 중심·크기 정규화 후 비교하기 위함."""
    a = a - np.median(a); b = b - np.median(b)
    # max 로 정규화하면 점 하나가 전체 크기를 좌우한다. 98 백분위를 쓴다.
    r = max(np.percentile(np.abs(a), 98), np.percentile(np.abs(b), 98)) + 1e-9
    ia = np.clip(((a / r + 1) * 0.5 * (grid - 1)).astype(int), 0, grid - 1)
    ib = np.clip(((b / r + 1) * 0.5 * (grid - 1)).astype(int), 0, grid - 1)
    m = np.zeros((grid, grid), bool)
    m[ia, ib] = True
    from scipy import ndimage
    return ndimage.binary_fill_holes(ndimage.binary_closing(m, np.ones((3, 3))))


def observed(be, cidx, fid, expand=4.0, out_px=1036, grid_mm=1.0):
    """실측: 크롭 깊이로 음식의 미터 발자국·높이장·부피를 구한다 (pipeline 과 동일)."""
    seg = np.load(CACHE / f"seg_merged_combo_{cidx:02d}.npz", allow_pickle=True)
    fm_full = unpack(seg, f"food_{fid}")
    if len(fm_full) == 0:
        return None
    food = largest_component(fm_full[0])
    z_mm, _, _, plate = plate_distance_mm(cidx)

    img = resized(combo_images()[cidx], BASE)
    W, H = img.size
    x0, y0, x1, y1 = crop_box(food, expand, W, H)
    sub = img.crop((x0, y0, x1, y1))
    cw, ch = sub.size
    sc = out_px / max(cw, ch)
    sub_r = sub.resize((max(16, round(cw * sc)), max(16, round(ch * sc))), Image.LANCZOS)

    K = from_exif(combo_images()[cidx])
    f_base = K.fx * (BASE / K.width)
    fov_x = float(2 * np.degrees(np.arctan(cw / (2 * f_base))))
    pts, valid = be.points(sub_r, fov_x)

    def remap(m):
        mm = Image.fromarray(m[y0:y1, x0:x1].astype(np.uint8) * 255).resize(sub_r.size, Image.NEAREST)
        return np.asarray(mm) > 127

    fm = remap(food) & valid
    ring = remap(plate) & valid & ~fm
    if ring.sum() < 800 or fm.sum() < 200:
        return None
    zr = pts[..., 2][ring]; zr = zr[np.isfinite(zr) & (zr > 0)]
    if len(zr) < 200:
        return None
    s = z_mm / float(np.median(zr))

    P = pts[ring] * s; P = P[np.isfinite(P).all(1)]
    P = P[np.random.default_rng(0).choice(len(P), min(40000, len(P)), replace=False)]
    pl = fit_plane_ransac(P, iters=500, thresh_mm=3.0)
    F = pts[fm] * s; F = F[np.isfinite(F).all(1)]
    vol_hf, info = heightfield_volume(F, pl, grid_mm=grid_mm)
    u, v = pl.basis()
    return {"F": F, "plane": pl, "u": u, "v": v, "vol_hf": vol_hf,
            "A_cm2": info["footprint_cm2"], "z_mm": z_mm,
            "sil": rasterize(F @ u, F @ v),
            # 투영 자세탐색에 필요한 것들: 크롭 좌표의 마스크와 그 좌표계의 내부 파라미터
            "mask_crop": remap(food), "f_crop": f_base * sc,
            "cx": sub_r.size[0] / 2.0, "cy": sub_r.size[1] / 2.0}


def best_pose(mesh_pts, obs_sil, n_angle=36):
    """실루엣 IoU 를 최대화하는 자세를 찾는다. 정규화 실루엣이라 크기와 무관하다."""
    best = (-1.0, None)
    for R0 in axis_rotations():
        Q = mesh_pts @ R0.T
        for k in range(n_angle):
            R = inplane(2 * np.pi * k / n_angle) @ R0
            P = mesh_pts @ R.T
            sil = rasterize(P[:, 0], P[:, 1])
            inter = np.logical_and(sil, obs_sil).sum()
            union = np.logical_or(sil, obs_sil).sum()
            iou = inter / union if union else 0.0
            if iou > best[0]:
                best = (iou, R)
    return best


def footprint_area(P2, grid_mm=1.0):
    """평면좌표 점들의 발자국 면적 (mm^2). 볼록껍질이 아니라 래스터라 오목도 반영."""
    a, b = P2[:, 0], P2[:, 1]
    ia = np.floor((a - a.min()) / grid_mm).astype(int)
    ib = np.floor((b - b.min()) / grid_mm).astype(int)
    m = np.zeros((ia.max() + 1, ib.max() + 1), bool)
    m[ia, ib] = True
    from scipy import ndimage
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, np.ones((3, 3))))
    return float(m.sum()) * grid_mm * grid_mm


def umeyama(X, Y):
    """대응점 X->Y 의 유사변환 (s, R, t) 를 닫힌 형태로 푼다. Umeyama 1991."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    S = (Yc.T @ Xc) / len(X)
    U, D, Vt = np.linalg.svd(S)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1.0
    R = U @ W @ Vt
    var = (Xc ** 2).sum() / len(X)
    c = float(np.trace(np.diag(D) @ W) / var) if var > 0 else 1.0
    t = my - c * (R @ mx)
    return c, R, t


def scale_icp(mesh_pts, obs_pts, c0, R0, t0, iters=25, trim=0.85):
    """관측점(보이는 면)을 메시 표면에 붙이는 스케일 포함 ICP.

    발자국 '면적' 으로 크기를 맞추면 안 되는 이유: 비스듬히 본 물체의 실루엣은
    옆면까지 지지평면에 번져서 실제 접촉 발자국보다 크다. 사과에서 특히 심하다.
    그래서 2D 면적이 아니라 **3D 점을 메시 표면에 직접** 맞춘다.

    관측은 보이는 면뿐이고 메시는 완전하므로 대응은 관측->메시 한 방향으로만 건다.
    상위 15% 잔차는 잘라내 세그멘테이션 경계 잡음에 끌려가지 않게 한다.
    """
    from scipy.spatial import cKDTree
    c, R, t = c0, R0, t0
    prev = np.inf
    for _ in range(iters):
        M = (c * (mesh_pts @ R.T)) + t
        tree = cKDTree(M)
        d, idx = tree.query(obs_pts, workers=-1)
        k = max(50, int(len(d) * trim))
        keep = np.argsort(d)[:k]
        rms = float(np.sqrt((d[keep] ** 2).mean()))
        # 대응된 메시 점을 원좌표로 되돌려 Umeyama 를 푼다
        src = mesh_pts[idx[keep]]
        c, R, t = umeyama(src, obs_pts[keep])
        if abs(prev - rms) < 1e-3 * max(1.0, rms):
            break
        prev = rms
    return c, R, t, rms


def combo_table():
    out = {}
    for r in csv.DictReader(open(ID_MAP)):
        out.setdefault(int(r["combo"]), []).append(
            (int(r["id"]), r["food"], r["container"] or None))
    return out


def find_mesh(cidx, fid, slug, seed=0):
    sfx = "" if seed == 0 else f"_s{seed}"
    for pat in (f"c{cidx:02d}_{fid:02d}_{slug}{sfx}.glb",
                f"c{cidx:02d}_{fid:02d}_{slug}.glb", f"{fid:02d}_{slug}.glb"):
        p = OUTPUTS / "meshes" / pat
        if p.exists():
            return p
    return None


def main(backend="moge_v3", shard=0, nshards=1, resume=True,
         expand=4.0, mesh_seed=0):
    """백엔드별로 별도 CSV 에 쓴다. 여러 깊이 모델의 산포가 곧 관측된 sigma 다."""
    be = build(backend)
    table = combo_table()
    items = [(c, fid, slug, cont) for c in sorted(table) for fid, slug, cont in table[c]]
    mine = [it for i, it in enumerate(items) if i % nshards == shard]

    suffix = "" if backend == "moge_v3" else f"_{backend}"
    if expand != 4.0:
        suffix += f"_e{expand:g}"
    if mesh_seed:
        suffix += f"_s{mesh_seed}"
    out_csv = OUTPUTS / f"align{suffix}_shard{shard}.csv"
    done = set()
    if resume and out_csv.exists():
        for r in csv.DictReader(open(out_csv)):
            done.add((int(r["combo"]), int(r["id"])))
        print(f"[shard {shard}] 이어하기: {len(done)}개 완료", flush=True)
    todo = [it for it in mine if (it[0], it[1]) not in done]
    print(f"[shard {shard}] 내 몫 {len(mine)}, 남은 {len(todo)}", flush=True)
    if not todo:
        return

    write_header = not out_csv.exists()
    f = open(out_csv, "a", buffering=1)
    w = csv.writer(f)
    if write_header:
        w.writerow(["combo", "id", "slug", "container", "vol_heightfield_ml",
                    "vol_mesh_ml", "A_obs_cm2", "A_mesh_units2", "scale", "sil_iou", "fit"])

    for k, (cidx, fid, slug, container) in enumerate(todo, 1):
        t0 = time.time()
        mp = find_mesh(cidx, fid, slug, mesh_seed)
        if mp is None:
            print(f"[shard {shard}] {k}/{len(todo)} c{cidx:02d} id{fid} {slug}: 메시 없음", flush=True)
            continue
        ob = observed(be, cidx, fid, expand=expand)
        if ob is None:
            print(f"[shard {shard}] {k}/{len(todo)} c{cidx:02d} id{fid} {slug}: 실측 실패", flush=True)
            continue

        mesh = trimesh.load(mp, force="mesh", process=False)
        # seed 필수. 이것만 시드가 없어서 (나머지 난수원은 전부 고정돼 있다)
        # 같은 입력을 두 번 돌리면 메시 부피가 항목당 최대 17% 흔들렸다.
        # 표면 12000점 추출이 달라지면 투영 실루엣이 달라지고 자세·배율이 따라 흔들린다.
        pts = np.asarray(mesh.sample(12000, seed=0), dtype=np.float64)
        pts -= pts.mean(axis=0)

        # 자세·크기는 '실제 카메라로 투영한 실루엣' 을 SAM 3 마스크에 맞춰 정한다.
        # 접시 평면에 투영한 발자국끼리 비교하던 이전 방식은 정의가 어긋나 있었다
        # (관측=보이는 윗면의 그림자 / 메시=물체 전체의 그림자). 그 때문에 당근이
        # 4337 mL 로 터졌고, 투영 방식으로 바꾸니 11 mL 가 되었다.
        m_crop = ob["mask_crop"]
        r = pp.search(pts, m_crop, ob["plane"], ob["F"].mean(0),
                      ob["f_crop"], ob["cx"], ob["cy"], n_rot=300, ss=3)
        if r is None:
            print(f"[shard {shard}] {k}/{len(todo)} c{cidx:02d} id{fid} {slug}: 자세탐색 실패",
                  flush=True)
            continue
        iou, Rf, c = pp.refine(pts, m_crop, ob["plane"], ob["F"].mean(0),
                               ob["f_crop"], ob["cx"], ob["cy"], r[1], r[2], ss=3)
        v_mesh_ml = float(abs(mesh.volume)) * c ** 3 * 1e-3
        A_mesh = float("nan")

        # 정합 잔차: 관측점이 놓인 메시 표면과 얼마나 떨어져 있는가 (신뢰도 지표)
        from scipy.spatial import cKDTree
        M = pp.place(pts, Rf, c, ob["plane"], ob["F"].mean(0))
        F = ob["F"]
        if len(F) > 20000:
            F = F[np.random.default_rng(0).choice(len(F), 20000, replace=False)]
        d, _ = cKDTree(M).query(F, workers=-1)
        obj_mm = float(np.ptp(M, axis=0).max())
        fit = float(np.sqrt((np.sort(d)[:int(0.85 * len(d))] ** 2).mean())) / max(obj_mm, 1e-6)

        w.writerow([cidx, fid, slug, container or "", f"{ob['vol_hf']:.3f}",
                    f"{v_mesh_ml:.3f}", f"{ob['A_cm2']:.2f}", f"{A_mesh:.4f}",
                    f"{c:.5f}", f"{iou:.3f}", f"{fit:.4f}"])
        print(f"[shard {shard}] {k}/{len(todo)} c{cidx:02d} id{fid:02d} {slug:<18} "
              f"높이장={ob['vol_hf']:8.1f}mL  메시정합={v_mesh_ml:8.1f}mL  "
              f"IoU={iou:.2f} 잔차={fit:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    f.close()
    print(f"[shard {shard}] 완료 -> {out_csv}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="moge_v3")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--expand", type=float, default=4.0)
    ap.add_argument("--mesh-seed", type=int, default=0)
    a = ap.parse_args()
    main(a.backend, a.shard, a.nshards, True, a.expand, a.mesh_seed)
