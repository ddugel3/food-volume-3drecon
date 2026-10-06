"""접시를 자로 써서 MoGe 의 절대 스케일이 얼마나 틀렸는지 측정한다.

이 대회에서 접시는 유일하게 신뢰할 만한 절대 크기 단서다. 핵심은 **어떻게 재느냐**다.

CVPR2025 3위 팀(PSHS)은 접시의 픽셀 바운딩박스 대각선을 썼고 MAPE 0.46 으로 최하위였다.
픽셀 공간에서 재면 접시가 기울어 보일수록 짧게 나오고, 그 단축률이 자세에 따라 제멋대로
바뀐다. 우리는 그렇게 하지 않는다.

원형 접시의 테두리는 3D 공간에서 원이다. 그 원을 **자기 자신의 평면 좌표계로 투영하면
보는 각도와 무관하게 다시 원**이 된다. 그러므로
  1) 접시 마스크의 3D 점에 RANSAC 평면을 맞추고
  2) 그 평면 좌표계로 점을 옮긴 뒤
  3) 볼록껍질의 최대 폭(장축)과 최소 폭(단축)을 재면
자세 불변인 지름이 나온다. 장축/단축 비는 공짜 품질검사다 -- 원형 접시라면 1에 가까워야
하고, 크게 벗어나면 깊이가 평면을 왜곡시켰거나 접시가 실제로 타원형이라는 뜻이다.

여기서 나오는 숫자가 곧 스케일 보정계수다.
  alpha = (실제 접시 지름 사전분포) / (MoGe 가 말하는 지름)
부피는 alpha^3 으로 움직인다.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import fit_plane_ransac   # noqa: E402
from paths import CACHE                 # noqa: E402

# 공개 통계 기반 식기 치수 사전분포 (mm). 평균과 표준편차 둘 다 필요하다 --
# 로그공간 융합에서 가중치가 1/sigma^2 이기 때문이다.
PRIORS_MM = {
    "dinner_plate": (265.0, 20.0),   # 일반 디너 플레이트 24~29cm
    "salad_plate": (210.0, 15.0),
    "bowl": (165.0, 25.0),
    "cup": (85.0, 12.0),
    "fork": (195.0, 12.0),
    "knife": (220.0, 15.0),
    "spoon": (175.0, 12.0),
}


def unpack(z, key):
    ms = z[f"{key}__mshape"]
    if int(ms[0]) == 0:
        return np.zeros((0, 0, 0), bool)
    packed = z[f"{key}__masks"]
    n, h, w = (int(x) for x in ms)
    return np.unpackbits(packed, axis=-1)[..., :w].reshape(n, h, w).astype(bool)


def hull_extent(a, b):
    """평면 좌표 (a,b) 볼록껍질의 최대 폭과 최소 폭(회전 캘리퍼)."""
    from scipy.spatial import ConvexHull
    pts = np.column_stack([a, b])
    if len(pts) > 20000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 20000, replace=False)]
    h = ConvexHull(pts)
    v = pts[h.vertices]
    d = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=-1)
    major = float(d.max())
    # 최소 폭: 각 껍질 변 방향에 수직으로 투영했을 때의 폭 중 최소
    best = np.inf
    for i in range(len(v)):
        e = v[(i + 1) % len(v)] - v[i]
        n = np.linalg.norm(e)
        if n < 1e-9:
            continue
        nvec = np.array([-e[1], e[0]]) / n
        proj = v @ nvec
        best = min(best, float(proj.max() - proj.min()))
    return major, best


def measure(cidx, ver="v3", variant="exif", ref="ref_plate"):
    seg = np.load(CACHE / f"seg_gdino_sam2_combo_{cidx:02d}.npz", allow_pickle=True)
    dep = np.load(CACHE / f"moge{ver}_combo_{cidx:02d}.npz")
    masks = unpack(seg, ref)
    if len(masks) == 0:
        return None
    m = masks[0] & dep[f"mask_{variant}"]
    pts = dep[f"points_{variant}"][m]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 2000:
        return None

    sub = pts[np.random.default_rng(0).choice(len(pts), min(60000, len(pts)), replace=False)]
    pl = fit_plane_ransac(sub, iters=600, thresh_mm=4.0)
    u, v = pl.basis()
    a, b, h = sub @ u, sub @ v, pl.height(sub)
    major, minor = hull_extent(a, b)
    return {
        "major_mm": major, "minor_mm": minor, "ratio": minor / major,
        "flatness_mm": float(np.percentile(h, 97.5) - np.percentile(h, 2.5)),
        "depth_mm": float(np.median(sub[:, 2])), "n": len(pts),
    }


def main(ver, variant):
    print(f"=== MoGe {ver} ({variant}) 가 보는 접시 크기 ===")
    print(f"{'combo':>6} {'장축mm':>9} {'단축mm':>9} {'단/장':>7} {'평탄도mm':>9} {'거리mm':>8}")
    rows = []
    for cidx in range(1, 15):
        r = measure(cidx, ver, variant)
        if r is None:
            print(f"{cidx:>6}  (접시 마스크 없음/점 부족)"); continue
        rows.append((cidx, r))
        print(f"{cidx:>6} {r['major_mm']:>9.1f} {r['minor_mm']:>9.1f} {r['ratio']:>7.3f} "
              f"{r['flatness_mm']:>9.1f} {r['depth_mm']:>8.0f}")

    maj = np.array([r["major_mm"] for _, r in rows])
    mu, sd = PRIORS_MM["dinner_plate"]
    alpha = mu / maj
    la = np.log(alpha)
    print(f"\n  MoGe 측정 접시 장축: 중앙값 {np.median(maj):.1f}mm  "
          f"범위 {maj.min():.1f}~{maj.max():.1f}mm")
    print(f"  디너 플레이트 사전분포: {mu:.0f} ± {sd:.0f} mm")
    print(f"  ★ 길이 보정계수 alpha 중앙값 {np.median(alpha):.3f} "
          f"(로그 표준편차 {la.std():.3f})")
    print(f"  ★ 부피 보정계수 alpha^3 중앙값 {np.median(alpha)**3:.3f} "
          f"-> 보정 안 하면 부피가 {100*(1/np.median(alpha)**3 - 1):+.0f}% 어긋난다")
    print(f"\n  단/장 비 중앙값 {np.median([r['ratio'] for _,r in rows]):.3f} "
          f"(원형 접시를 평면 좌표에서 재면 1에 가까워야 한다)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", default="v3"); ap.add_argument("--variant", default="exif")
    a = ap.parse_args()
    main(a.ver, a.variant)
