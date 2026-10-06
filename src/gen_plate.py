"""접시/그릇 자체를 생성 복원해 지지면 프로파일을 얻는다.

측정만으로는 한계가 있다. 음식이 접시를 많이 가리면 보이는 고리가 테두리뿐이라
안쪽 우물을 복원할 수 없다(스테이크에서 지지면이 무너져 부피가 198 -> 337 mL 로
폭주했다). 가려진 부분을 채우는 것은 생성 모델의 일이다.

우리가 이미 확립한 원칙과 일치한다 -- 생성 모델은 크기는 모르지만 형상은 안다.
접시 지름은 픽셀에서 재고(자세 불변 볼록껍질 장축), 생성 메시에서는 **무차원
프로파일** h(rho)/D 만 가져온다. 그 둘을 곱하면 실제 우물 깊이가 나온다.
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
sys.path.insert(0, str(HERE.parent / "external/Hunyuan3D-2.1/hy3dshape"))
from paths import CACHE, OUTPUTS, combo_images   # noqa: E402
from scale_plate import unpack                   # noqa: E402
from diag_plate_scale import largest_component   # noqa: E402
from segment import resized                      # noqa: E402
from diag_crop import crop_box                   # noqa: E402

BASE = 2016


def cutout(cidx, ref="ref_plate", expand=1.15, size=768):
    seg = np.load(CACHE / f"seg_merged_combo_{cidx:02d}.npz", allow_pickle=True)
    m = unpack(seg, ref)
    if len(m) == 0:
        return None
    mask = largest_component(m[0])
    img = resized(combo_images()[cidx], BASE).convert("RGB")
    W, H = img.size
    x0, y0, x1, y1 = crop_box(mask, expand, W, H)
    rgb = np.asarray(img)[y0:y1, x0:x1]
    a = (mask[y0:y1, x0:x1] * 255).astype(np.uint8)
    out = Image.fromarray(np.dstack([rgb, a]), "RGBA")
    s = size / max(out.size)
    return out.resize((round(out.width * s), round(out.height * s)), Image.LANCZOS)


def profile_from_mesh(mesh, n_bins=32):
    """회전 대칭 가정 하에 메시의 안쪽 면 프로파일을 뽑는다.

    반환: (rho/R, h/D) 무차원 프로파일. 접시를 '위가 열린' 방향으로 정렬한 뒤
    각 반경 구간에서 **가장 낮은** 표면(= 안쪽 바닥)을 취한다.
    """
    P = np.asarray(mesh.sample(200000), dtype=np.float64)
    P -= P.mean(0)
    # 접시는 납작하므로 가장 얇은 축이 법선이다
    ext = np.ptp(P, axis=0)
    ax = int(np.argmin(ext))
    others = [i for i in range(3) if i != ax]
    z = P[:, ax]
    rho = np.hypot(P[:, others[0]], P[:, others[1]])
    R = float(np.percentile(rho, 99))
    if R <= 0:
        return None
    # 위가 열린 방향으로: 바깥 테두리가 안쪽보다 높아야 한다
    inner = z[rho < 0.3 * R]
    outer = z[rho > 0.8 * R]
    if len(inner) < 50 or len(outer) < 50:
        return None
    if np.median(outer) < np.median(inner):
        z = -z
    edges = np.linspace(0, R, n_bins + 1)
    rs, hs = [], []
    for i in range(n_bins):
        sel = (rho >= edges[i]) & (rho < edges[i + 1])
        if sel.sum() >= 100:
            rs.append(0.5 * (edges[i] + edges[i + 1]) / R)
            hs.append(float(np.percentile(z[sel], 15)))   # 안쪽 면 = 낮은 쪽
    if len(rs) < 5:
        return None
    hs = np.array(hs) - min(hs)
    return np.array(rs), hs / (2 * R)      # 지름으로 정규화


def main(steps=30, octree=320):
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
    md = OUTPUTS / "meshes_plate"; md.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'c':>3} {'우물깊이/지름':>13} {'r=0.3':>7} {'r=0.6':>7} {'r=0.9':>7}")
    for c in range(1, 15):
        img = cutout(c)
        if img is None:
            print(f"{c:>3}  마스크 없음"); continue
        t0 = time.time()
        mesh = pipe(image=img, num_inference_steps=steps, octree_resolution=octree,
                    output_type="trimesh")[0]
        mesh.export(md / f"plate_c{c:02d}.glb")
        pr = profile_from_mesh(mesh)
        if pr is None:
            print(f"{c:>3}  프로파일 실패 ({time.time()-t0:.0f}s)"); continue
        rs, hs = pr
        g = lambda x: float(np.interp(x, rs, hs))
        rows.append((c, float(hs.max()), g(0.3), g(0.6), g(0.9)))
        print(f"{c:>3} {hs.max():>13.4f} {g(0.3):>7.4f} {g(0.6):>7.4f} {g(0.9):>7.4f} "
              f"({time.time()-t0:.0f}s)")
        np.savez(OUTPUTS / f"plate_profile_c{c:02d}.npz", rs=rs, hs=hs)
    with open(OUTPUTS / "plate_profiles.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["combo", "depth_over_D", "h03", "h06", "h09"])
        for r in rows:
            w.writerow([r[0]] + [f"{x:.5f}" for x in r[1:]])
    print(f"\n  ✔ {OUTPUTS/'plate_profiles.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--octree", type=int, default=320)
    a = ap.parse_args()
    main(a.steps, a.octree)
