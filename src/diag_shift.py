"""MoGe 오차가 곱셈(스케일)인가 덧셈(깊이 시프트)인가.

브로콜리 교차검증에서 발자국 면적은 맞고 높이만 6배 틀렸다. alpha 를 곱하기 전
원시 높이는 두 장면 모두 36~39mm 로 실제 브로콜리와 맞았다. 이건 곱셈 오차의
증상이 아니다 -- 전체를 k 배 하면 높이도 k 배로 같이 틀려야 한다.

덧셈 오차 Z -> Z + c 라면 정확히 이 증상이 나온다.
  기복(relief) dZ 는 보존된다.
  가로 X = (u-cx)/f * Z 는 (Z+c)/Z 배로 부푼다. 깊이가 얕은 곳일수록 더 부푼다.
  그리고 원은 타원으로 일그러진다 <- 이게 우리가 쓸 검출기다.

그래서 순차로 푼다.
  1) 접시가 다시 '원'이 되게 하는 시프트 t 를 찾는다 (형상 복원, 크기와 무관)
  2) 그 상태에서 지름이 265mm 가 되게 하는 배율 s 를 정한다 (크기 확정)

1) 이 성공하면 원형성이 0.90 -> 1.00 근처로 올라가야 하고,
복원된 카메라 거리가 접시 265mm 로 역산한 ~350mm 와 맞아야 한다.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import fit_plane_ransac                    # noqa: E402
from paths import CACHE, combo_images                    # noqa: E402
from intrinsics import from_exif                         # noqa: E402
from scale_plate import PRIORS_MM, hull_extent, unpack   # noqa: E402
from diag_plate_scale import largest_component           # noqa: E402


def plate_pixels(cidx, ver="v3", variant="exif", max_pts=40000):
    """접시 마스크의 (u, v, Z) 와 리사이즈 반영 내부 파라미터."""
    seg = np.load(CACHE / f"seg_gdino_sam2_combo_{cidx:02d}.npz", allow_pickle=True)
    dep = np.load(CACHE / f"moge{ver}_combo_{cidx:02d}.npz")
    pm, mv = dep[f"points_{variant}"], dep[f"mask_{variant}"]
    m = largest_component(unpack(seg, "ref_plate")[0]) & mv & np.isfinite(pm).all(-1)
    vs, us = np.nonzero(m)
    Z = pm[..., 2][m]
    med = np.median(Z); mad = np.median(np.abs(Z - med)) + 1e-6
    keep = np.abs(Z - med) < 6 * mad
    us, vs, Z = us[keep], vs[keep], Z[keep]
    if len(Z) > max_pts:
        i = np.random.default_rng(0).choice(len(Z), max_pts, replace=False)
        us, vs, Z = us[i], vs[i], Z[i]

    K = from_exif(combo_images()[cidx])
    H, W = pm.shape[:2]
    f = K.fx * (W / K.width)
    return us.astype(float), vs.astype(float), Z.astype(float), f, W / 2.0, H / 2.0


def circularity(t, us, vs, Z, f, cx, cy):
    """깊이를 t 만큼 민 뒤 접시를 평면에 펴서 잰 단축/장축 비."""
    Zt = Z + t
    if Zt.min() <= 1.0:
        return 0.0
    P = np.stack([(us - cx) / f * Zt, (vs - cy) / f * Zt, Zt], axis=-1)
    pl = fit_plane_ransac(P, iters=250, thresh_mm=max(1.0, 0.004 * np.median(Zt)))
    u_, v_ = pl.basis()
    major, minor = hull_extent(P @ u_, P @ v_)
    return minor / major if major > 0 else 0.0


def solve(cidx, ver="v3", D=None):
    D = D or PRIORS_MM["dinner_plate"][0]
    us, vs, Z, f, cx, cy = plate_pixels(cidx, ver)
    z0 = float(np.median(Z))

    base = circularity(0.0, us, vs, Z, f, cx, cy)
    # 시프트는 음수 쪽(장면을 카메라 쪽으로 당김)이 예상된다. z0 의 -95% ~ +200% 를 훑는다.
    res = minimize_scalar(lambda t: -circularity(t, us, vs, Z, f, cx, cy),
                          bounds=(-0.95 * z0, 2.0 * z0), method="bounded",
                          options={"xatol": z0 * 1e-3})
    t = float(res.x); best = -res.fun

    Zt = Z + t
    P = np.stack([(us - cx) / f * Zt, (vs - cy) / f * Zt, Zt], axis=-1)
    pl = fit_plane_ransac(P, iters=600, thresh_mm=max(1.0, 0.004 * np.median(Zt)))
    u_, v_ = pl.basis()
    major, _ = hull_extent(P @ u_, P @ v_)
    s = D / major
    return {"z0": z0, "t": t, "z_corr": (z0 + t) * s, "circ0": base, "circ": best,
            "s": s, "major_after_shift": major}


def main(ver="v3"):
    print(f"=== 깊이 시프트 t 를 접시 원형성으로 추정 (MoGe {ver}) ===")
    print(f"{'combo':>6} {'MoGe z0':>9} {'시프트t':>9} {'보정거리':>9} "
          f"{'원형성전':>9} {'원형성후':>9} {'배율s':>8}")
    rows = []
    for c in range(1, 15):
        try:
            r = solve(c, ver)
        except Exception as e:
            print(f"{c:>6}  실패 {type(e).__name__}: {e}"); continue
        rows.append((c, r))
        print(f"{c:>6} {r['z0']:>9.0f} {r['t']:>9.0f} {r['z_corr']:>9.0f} "
              f"{r['circ0']:>9.3f} {r['circ']:>9.3f} {r['s']:>8.3f}")
    c0 = np.array([r["circ0"] for _, r in rows]); c1 = np.array([r["circ"] for _, r in rows])
    zc = np.array([r["z_corr"] for _, r in rows])
    print(f"\n  원형성: 보정 전 중앙값 {np.median(c0):.3f} -> 보정 후 {np.median(c1):.3f}")
    print(f"  보정 후 카메라 거리: 중앙값 {np.median(zc):.0f}mm  범위 {zc.min():.0f}~{zc.max():.0f}")
    print(f"    (접시 265mm 픽셀 역산으로 얻은 값은 중앙값 348mm, 범위 243~568 이었다)")


if __name__ == "__main__":
    main()
