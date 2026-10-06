"""시프트 t 의 목적함수 지형을 직접 훑는다.

앞선 최적화가 콤보 6 에서 t=-63 을 찾고 멈췄는데, 물리적으로 기대되는 값은
-(1516-243) = -1273 근처다. 지역해에 빠졌는지, 아니면 원형성이라는 목적함수 자체가
그 방향에 대해 평평한지 확인해야 한다.

두 가지를 같이 본다.
  원형성  minor/major  -- 1.0 이어야 한다
  지름    major        -- 265mm 가 되는 t 가 물리적 정답이다 (배율 s=1 가정)

'지름 = 265mm 가 되는 t' 는 단조로운 강한 제약이다. 원형성이 평평하더라도
이쪽은 해가 유일하게 잡힌다. 두 곡선이 같은 t 를 가리키면 모델이 맞은 것이고,
서로 다른 t 를 가리키면 순수 시프트 모델이 부족하다는 뜻이다.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import fit_plane_ransac                 # noqa: E402
from scale_plate import hull_extent                   # noqa: E402
from diag_shift import plate_pixels                   # noqa: E402

D = 265.0


def probe(t, us, vs, Z, f, cx, cy):
    Zt = Z + t
    if Zt.min() <= 1.0:
        return None
    P = np.stack([(us - cx) / f * Zt, (vs - cy) / f * Zt, Zt], axis=-1)
    pl = fit_plane_ransac(P, iters=200, thresh_mm=max(0.5, 0.004 * np.median(Zt)))
    u_, v_ = pl.basis()
    major, minor = hull_extent(P @ u_, P @ v_)
    h = pl.height(P)
    flat = float(np.percentile(h, 98) - np.percentile(h, 2))
    return major, minor / major, flat


def scan(cidx, ver="v3", n=26):
    us, vs, Z, f, cx, cy = plate_pixels(cidx, ver, max_pts=12000)
    z0 = float(np.median(Z))
    ts = np.linspace(-0.93 * z0, 0.3 * z0, n)
    print(f"\n=== combo {cidx}  (MoGe z0 = {z0:.0f} mm) ===")
    print(f"{'t':>8} {'거리z0+t':>9} {'지름mm':>9} {'원형성':>8} {'테두리mm':>9}")
    best_d, best_c = None, None
    for t in ts:
        r = probe(t, us, vs, Z, f, cx, cy)
        if r is None:
            continue
        major, circ, flat = r
        mark = ""
        if best_d is None or abs(major - D) < abs(best_d[1] - D):
            best_d = (t, major); mark += " <-지름"
        if best_c is None or circ > best_c[1]:
            best_c = (t, circ); mark += " <-원형"
        print(f"{t:>8.0f} {z0+t:>9.0f} {major:>9.1f} {circ:>8.3f} {flat:>9.1f}{mark}")
    print(f"  지름 265mm 에 가장 가까운 t = {best_d[0]:.0f}  (거리 {z0+best_d[0]:.0f}mm)")
    print(f"  원형성 최대 t          = {best_c[0]:.0f}  (거리 {z0+best_c[0]:.0f}mm, "
          f"원형성 {best_c[1]:.3f})")


if __name__ == "__main__":
    for c in (2, 6, 1):
        scan(c)
