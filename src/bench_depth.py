"""깊이 백엔드를 GT 없이 채점한다.

대회 GT 는 봉인했고 MetaFood3D dev set 은 아직 없다. 그래도 두 가지 지표로
모델을 줄 세울 수 있다. 둘 다 정답을 보지 않는다.

지표 1 — 브로콜리 교차일치도 (정확하지만 표본 1개)
  food id 5 는 콤보 2 와 콤보 6 에 같은 물체로 등장한다. 정답이 하나이므로
  두 장면에서 잰 스케일 무관 종횡비(높이/가로)가 같아야 한다.
  불일치 배수가 낮을수록 좋다. 이론적 최소는 1.00.

지표 2 — 접시 평면성 (표본 14개)
  접시 바닥은 평면이다. 접시 링 점들을 평면에 맞췄을 때 남는 잔차를
  접시 지름으로 나눈 값이 작아야 한다. 깊이가 휘거나 계단지면 커진다.
  이건 절대 스케일과 무관하고 14 장면 전부에서 잴 수 있다.

두 지표 모두 '절대 스케일이 맞는가' 가 아니라 '형상이 맞는가' 를 본다.
절대 스케일은 어차피 접시가 담당하므로 깊이 모델에게 물을 것은 형상뿐이다.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends.depth import build, REGISTRY              # noqa: E402
from geometry import fit_plane_ransac                   # noqa: E402
from paths import CACHE, ID_MAP, combo_images           # noqa: E402
from intrinsics import from_exif                        # noqa: E402
from scale_plate import hull_extent, unpack             # noqa: E402
from diag_plate_scale import largest_component          # noqa: E402
from segment import resized                             # noqa: E402
from diag_crop import crop_box                          # noqa: E402

BASE = 2016


def run_region(be, cidx, food_key, expand, out_px=1036):
    """한 장면의 한 물체를 크롭해 추론하고, 스케일 무관 형상량을 잰다."""
    seg = np.load(CACHE / f"seg_gdino_sam2_combo_{cidx:02d}.npz", allow_pickle=True)
    img = resized(combo_images()[cidx], BASE)
    W, H = img.size
    plate = largest_component(unpack(seg, "ref_plate")[0])
    food = largest_component(unpack(seg, food_key)[0]) if food_key else None

    if expand is None:
        x0, y0, x1, y1 = 0, 0, W, H
    else:
        x0, y0, x1, y1 = crop_box(food if food is not None else plate, expand, W, H)

    sub = img.crop((x0, y0, x1, y1))
    cw, ch = sub.size
    s = out_px / max(cw, ch)
    sub_r = sub.resize((max(16, round(cw * s)), max(16, round(ch * s))), Image.LANCZOS)

    K = from_exif(combo_images()[cidx])
    f_base = K.fx * (BASE / K.width)
    fov_x = float(2 * np.degrees(np.arctan(cw / (2 * f_base))))

    pts, valid = be.points(sub_r, fov_x)

    def remap(m):
        mm = Image.fromarray(m[y0:y1, x0:x1].astype(np.uint8) * 255).resize(sub_r.size, Image.NEAREST)
        return np.asarray(mm) > 127

    pm = remap(plate) & valid
    fm = (remap(food) & valid) if food is not None else np.zeros_like(pm)
    ring = pm & ~fm
    if ring.sum() < 1200:
        return None

    P = pts[ring]; P = P[np.isfinite(P).all(1)]
    if len(P) < 800:
        return None
    P = P[np.random.default_rng(0).choice(len(P), min(40000, len(P)), replace=False)]
    scale_hint = max(1e-6, 0.004 * float(np.median(np.abs(P[:, 2]))))
    pl = fit_plane_ransac(P, iters=500, thresh_mm=scale_hint)
    u, v = pl.basis()
    ring_major, _ = hull_extent(P @ u, P @ v)
    resid = pl.height(P)
    planarity = float(np.percentile(resid, 90) - np.percentile(resid, 10)) / max(ring_major, 1e-9)

    out = {"planarity": planarity, "ring_major": ring_major}
    if food is not None and fm.sum() > 250:
        F = pts[fm]; F = F[np.isfinite(F).all(1)]
        if len(F) > 200:
            lat, _ = hull_extent(F @ u, F @ v)
            h = float(np.percentile(pl.height(F), 98))
            out.update({"lat": lat, "h": h, "aspect": h / max(lat, 1e-9)})
    return out


def bench(names, expand):
    print(f"=== 깊이 백엔드 채점 (크롭 x{expand if expand else '전체'}) ===\n")
    print(f"{'백엔드':>14} {'브로콜리 불일치':>15} {'콤보2 종횡비':>13} {'콤보6 종횡비':>13} "
          f"{'접시평면성 중앙값':>17} {'실패':>5}")
    for nm in names:
        try:
            be = build(nm)
        except Exception as e:
            print(f"{nm:>14}  로드 실패 {type(e).__name__}: {str(e)[:60]}"); continue
        try:
            a = run_region(be, 2, "food_5", expand)
            b = run_region(be, 6, "food_5", expand)
            plan, fail = [], 0
            for c in range(1, 15):
                try:
                    r = run_region(be, c, None, None)   # 접시 평면성은 전체 프레임에서
                    if r: plan.append(r["planarity"])
                    else: fail += 1
                except Exception:
                    fail += 1
            if a and b and "aspect" in a and "aspect" in b:
                dis = max(a["aspect"], b["aspect"]) / max(1e-9, min(a["aspect"], b["aspect"]))
                print(f"{nm:>14} {dis:>15.2f} {a['aspect']:>13.3f} {b['aspect']:>13.3f} "
                      f"{np.median(plan):>17.4f} {fail:>5}")
            else:
                print(f"{nm:>14} {'측정불가':>15} {'-':>13} {'-':>13} "
                      f"{(np.median(plan) if plan else float('nan')):>17.4f} {fail:>5}")
        finally:
            del be
            import torch, gc
            gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default=",".join(sorted(REGISTRY)))
    ap.add_argument("--expand", type=float, default=4.0)
    a = ap.parse_args()
    bench([s for s in a.backends.split(",") if s], a.expand)
