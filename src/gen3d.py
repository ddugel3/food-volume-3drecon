"""생성형 3D 로 '숨은 부피 비율' rho 를 구한다.

왜 필요한가. 우리 부피는 깊이맵의 **보이는 윗면**을 접시 평면 위에서 적분한 값이다.
이 값은 진짜 부피와 두 가지 이유로 다르다.

  과대 방향 -- 둥근 물체는 옆면이 접시 쪽으로 말려 들어가는데, 윗면 적분은 그 처마
              아래를 꽉 찬 것으로 센다. 반지름 r 인 구를 평면에 놓고 적분하면
              (5/3)pi r^3 이 나오지만 진짜 부피는 (4/3)pi r^3 이라 rho = 0.8 이다.
  과소/과대  -- 그릇에 담긴 음식은 바닥이 접시 평면이 아니라 그릇 안쪽 곡면이다.
              접시 평면을 바닥으로 잡으면 그릇 높이만큼의 빈 공간까지 부피가 된다.
              실제로 파스타가 932 mL 로 나왔다.

rho = V(완전한 메시) / V(그 메시를 같은 카메라에서 본 윗면 적분) 로 정의하면
분자와 분모에서 스케일이 소거된다. 즉 **생성 모델이 물체의 실제 크기를 몰라도 된다.**
생성 모델에게는 그것이 잘하는 '형상' 만 묻고, 크기는 접시가 담당한다.

핵심은 분모를 **실측과 똑같은 알고리즘**으로 계산하는 것이다. 그래야 적분 방식의
계통 오차까지 비율에서 함께 소거된다.

SAM 3D Objects 를 1순위로 쓰려 했으나 VRAM 32GB 이상을 요구해 4090(24GB)에 올라가지
않는다. Hunyuan3D 2.1 은 shape 만이면 10GB 라 여유가 있고, CVPR2025 상위 3팀이
공통으로 쓴 백본이기도 하다.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "external/Hunyuan3D-2.1/hy3dshape"))

from paths import CACHE, ID_MAP, OUTPUTS, combo_images   # noqa: E402
from scale_plate import unpack                           # noqa: E402
from diag_plate_scale import largest_component           # noqa: E402
from segment import resized                              # noqa: E402
from diag_crop import crop_box                           # noqa: E402

BASE = 2016
MESH_DIR = None


def combo_table():
    out = {}
    for r in csv.DictReader(open(ID_MAP)):
        out.setdefault(int(r["combo"]), []).append(
            (int(r["id"]), r["food"], r["container"] or None))
    return out


def cutout(cidx, food_key, expand=1.35, size=768):
    """생성기 입력: 배경을 지운 RGBA 크롭. 물체가 프레임을 채우게 한다."""
    seg = np.load(CACHE / f"seg_merged_combo_{cidx:02d}.npz", allow_pickle=True)
    m = unpack(seg, food_key)
    if len(m) == 0:
        return None
    mask = largest_component(m[0])
    img = resized(combo_images()[cidx], BASE).convert("RGB")
    W, H = img.size
    x0, y0, x1, y1 = crop_box(mask, expand, W, H)
    rgb = np.asarray(img)[y0:y1, x0:x1]
    a = (mask[y0:y1, x0:x1] * 255).astype(np.uint8)
    rgba = np.dstack([rgb, a])
    out = Image.fromarray(rgba, "RGBA")
    s = size / max(out.size)
    return out.resize((max(8, round(out.width * s)), max(8, round(out.height * s))),
                      Image.LANCZOS)


def top_surface_volume(mesh, n_grid=256):
    """메시를 '위에서 본 윗면 적분' 부피로 잰다 -- 실측 파이프라인과 같은 정의.

    메시를 그 자체의 지지평면(최저점이 닿는 z=min) 위에 놓고, xy 균일 격자에
    표면 최고 z 를 올려 적분한다. 우리가 깊이맵에 하는 것과 동일한 연산이다.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces)
    z0 = v[:, 2].min()
    x0, y0 = v[:, 0].min(), v[:, 1].min()
    x1, y1 = v[:, 0].max(), v[:, 1].max()
    dx = (x1 - x0) / n_grid
    dy = (y1 - y0) / n_grid
    if dx <= 0 or dy <= 0:
        return 0.0
    # 삼각형 정점을 격자에 뿌려 셀별 최고 높이를 취한다 (표면 샘플링으로 보강)
    pts, _ = mesh.sample(400000, return_index=True) if hasattr(mesh, "sample") else (v, None)
    pts = np.asarray(pts, dtype=np.float64)
    ix = np.clip(((pts[:, 0] - x0) / dx).astype(int), 0, n_grid - 1)
    iy = np.clip(((pts[:, 1] - y0) / dy).astype(int), 0, n_grid - 1)
    hf = np.full((n_grid, n_grid), -np.inf)
    np.maximum.at(hf, (ix, iy), pts[:, 2] - z0)
    occ = np.isfinite(hf)
    from scipy import ndimage
    filled = ndimage.binary_fill_holes(occ)
    h = np.where(occ, hf, 0.0)
    # 발자국 내부 빈 셀은 이웃 평균으로 (실측 파이프라인과 동일한 처리)
    holes = filled & ~occ
    if holes.any():
        k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
        known = occ.copy(); work = h.copy()
        for _ in range(64):
            rem = holes & ~known
            if not rem.any():
                break
            s_ = ndimage.convolve(np.where(known, work, 0.0), k, mode="constant")
            c_ = ndimage.convolve(known.astype(float), k, mode="constant")
            upd = rem & (c_ > 0)
            work[upd] = s_[upd] / c_[upd]; known |= upd
        h = np.where(known, work, 0.0)
    np.clip(h, 0.0, None, out=h)
    return float(h.sum()) * dx * dy


def main(steps=30, octree=320, seed=0, shard=0, nshards=1, resume=True):
    """샤딩 실행. 항목끼리 완전히 독립이므로 GPU 당 프로세스 하나로 반씩 나눠 돌린다.

    CUDA_VISIBLE_DEVICES 로 각 워커에 GPU 를 하나씩 고정한다. 생성 1회가 8.2GB 라
    24GB 카드에 하나씩 올리면 서로 간섭이 없다. 결과는 샤드별 CSV 로 따로 쓰고
    나중에 합친다 -- 한 워커가 죽어도 나머지는 살고, 재실행하면 이어서 한다.
    """
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    mesh_dir = OUTPUTS / "meshes"; mesh_dir.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "qc").mkdir(parents=True, exist_ok=True)
    table = combo_table()

    # (combo, id, slug, container) 를 평탄화한 뒤 순서대로 샤드에 배분한다.
    # 브로콜리(id 5)는 콤보 2 와 6 에 각각 있으므로 35 개다.
    items = [(c, fid, slug, cont)
             for c in sorted(table) for fid, slug, cont in table[c]]
    mine = [it for i, it in enumerate(items) if i % nshards == shard]

    # 시드를 파일명에 넣지 않으면 다중 시드 실행이 서로를 덮어쓰고,
    # 이어하기 로직이 "이미 끝났다" 고 판단해 전부 건너뛴다.
    sfx = "" if seed == 0 else f"_s{seed}"
    out_csv = OUTPUTS / f"rho{sfx}_shard{shard}.csv"
    done = set()
    if resume and out_csv.exists():
        for r in csv.DictReader(open(out_csv)):
            done.add((int(r["combo"]), int(r["id"])))
        print(f"[shard {shard}] 이어하기: 이미 {len(done)}개 완료")

    todo = [it for it in mine if (it[0], it[1]) not in done]
    print(f"[shard {shard}] 전체 {len(items)} 중 내 몫 {len(mine)}, 남은 {len(todo)}")
    if not todo:
        return

    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")

    write_header = not out_csv.exists()
    f = open(out_csv, "a", buffering=1)          # 줄 단위 플러시 -> 중간에 죽어도 보존
    w = csv.writer(f)
    if write_header:
        w.writerow(["combo", "id", "slug", "container", "v_full", "v_top", "rho"])

    import time
    for k, (cidx, fid, slug, container) in enumerate(todo, 1):
        t0 = time.time()
        img = cutout(cidx, f"food_{fid}")
        if img is None:
            print(f"[shard {shard}] {k}/{len(todo)} combo{cidx} id{fid} {slug}: 마스크 없음")
            continue
        img.save(OUTPUTS / "qc" / f"cut_{cidx:02d}_{fid:02d}.png")
        g = torch.Generator("cuda").manual_seed(seed)
        mesh = pipe(image=img, num_inference_steps=steps, octree_resolution=octree,
                    generator=g, output_type="trimesh")[0]
        mesh.export(mesh_dir / f"c{cidx:02d}_{fid:02d}_{slug}{sfx}.glb")
        v_full = float(abs(mesh.volume))
        v_top = top_surface_volume(mesh)
        rho = v_full / v_top if v_top > 0 else float("nan")
        w.writerow([cidx, fid, slug, container or "", f"{v_full:.6f}",
                    f"{v_top:.6f}", f"{rho:.6f}"])
        print(f"[shard {shard}] {k}/{len(todo)} combo{cidx:02d} id{fid:02d} {slug:<18} "
              f"rho={rho:.3f}  ({time.time()-t0:.0f}s)")
    f.close()
    print(f"[shard {shard}] 완료 -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--octree", type=int, default=320)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.steps, a.octree, a.seed, a.shard, a.nshards)
