"""파이프라인 전체를 GT 없이 검증하는 유일한 방법: 같은 브로콜리, 다른 두 장면.

대회 데이터 설명에 "note that the same broccoli appears twice" 라고 적혀 있고,
food index 5 가 콤보 2 와 콤보 6 에 공유된다. 같은 물체이므로 정답이 하나다.
제출도 값 하나만 낸다.

그러므로 두 장면에서 독립적으로 계산한 부피가 **얼마나 어긋나는지가 곧
우리 파이프라인의 실제 오차 하한**이다. GT 도, 사전분포도, 리더보드도 필요 없다.
서로 다른 접시, 서로 다른 카메라 자세, 서로 다른 가림 상태에서 같은 답이 나와야 한다.

각 단계의 중간값(발자국 면적, 최고 높이, 부피)을 나란히 찍어서
어긋남이 스케일에서 오는지 형상에서 오는지도 함께 본다.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import fit_plane_ransac, heightfield_volume   # noqa: E402
from paths import CACHE                                     # noqa: E402
from scale_plate import PRIORS_MM, hull_extent, unpack      # noqa: E402
from diag_plate_scale import largest_component              # noqa: E402


def analyze(cidx, food_key, ver="v3", variant="exif", plate_mm=None):
    seg = np.load(CACHE / f"seg_gdino_sam2_combo_{cidx:02d}.npz", allow_pickle=True)
    dep = np.load(CACHE / f"moge{ver}_combo_{cidx:02d}.npz")
    pm, mv = dep[f"points_{variant}"], dep[f"mask_{variant}"]
    plate_mm = plate_mm or PRIORS_MM["dinner_plate"][0]

    # --- 접시 -> 평면 + alpha ---
    pmask = unpack(seg, "ref_plate")
    pmask = largest_component(pmask[0])
    pts = pm[pmask & mv]; pts = pts[np.isfinite(pts).all(axis=1)]
    z = pts[:, 2]; med = np.median(z); mad = np.median(np.abs(z - med)) + 1e-6
    pts = pts[np.abs(z - med) < 6 * mad]
    sub = pts[np.random.default_rng(0).choice(len(pts), min(60000, len(pts)), replace=False)]
    pl = fit_plane_ransac(sub, iters=800, thresh_mm=4.0)
    u, v = pl.basis()
    major, minor = hull_extent(sub @ u, sub @ v)
    alpha = plate_mm / major

    # --- 음식 -> alpha 적용 후 높이장 적분 ---
    fmask = unpack(seg, food_key)
    fm = largest_component(fmask[0]) & mv
    fp = pm[fm]; fp = fp[np.isfinite(fp).all(axis=1)]
    fp_s = fp * alpha
    pl_s = type(pl)(n=pl.n, d=pl.d * alpha)     # 평면도 같이 스케일
    vol_ml, info = heightfield_volume(fp_s, pl_s, grid_mm=1.0)
    h = pl_s.height(fp_s)
    return {
        "alpha": alpha, "plate_major_raw_mm": major, "plate_ratio": minor / major,
        "footprint_cm2": info["footprint_cm2"], "h_p98_mm": float(np.percentile(h, 98)),
        "h_mean_mm": float(np.clip(h, 0, None).mean()), "vol_ml": vol_ml,
        "n_pix": int(fm.sum()),
    }


def main():
    print("=== 같은 브로콜리(id 5), 두 장면 교차검증 ===\n")
    for ver in ("v3", "v2"):
        a = analyze(2, "food_5", ver)
        b = analyze(6, "food_5", ver)
        print(f"--- MoGe {ver} ---")
        keys = ["alpha", "plate_major_raw_mm", "plate_ratio", "n_pix",
                "footprint_cm2", "h_p98_mm", "h_mean_mm", "vol_ml"]
        print(f"{'항목':>20} {'콤보2':>12} {'콤보6':>12} {'비율':>9}")
        for k in keys:
            r = b[k] / a[k] if a[k] else float("nan")
            print(f"{k:>20} {a[k]:>12.2f} {b[k]:>12.2f} {r:>9.2f}")
        lr = abs(np.log(b["vol_ml"] / a["vol_ml"]))
        print(f"\n  ★ 부피 {a['vol_ml']:.1f} mL vs {b['vol_ml']:.1f} mL   "
              f"|로그비| {lr:.3f}  ->  {100*(np.exp(lr)-1):.0f}% 불일치")
        print(f"    (같은 물체이므로 이 불일치가 파이프라인 오차의 하한이다)\n")


if __name__ == "__main__":
    main()
