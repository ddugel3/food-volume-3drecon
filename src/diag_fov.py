"""EXIF 화각을 주입했을 때 복원 기하가 실제로 얼마나 바뀌는가.

'초점거리가 16% 다르면 부피가 세제곱으로 58% 다르다'는 말은 물체 전체가 상사확대될
때만 맞다. MoGe 는 fov_x 를 주면 투영만 다시 하고 깊이 예측 자체는 크게 안 바뀔 수
있으므로, 가로세로(X, Y)는 바뀌고 깊이(Z)는 덜 바뀌는 비대칭 변화가 일어난다.
그러면 부피는 세제곱이 아니라 대략 (가로배율^2 x 높이배율) 로 간다.

그래서 가정하지 않고 직접 잰다. 각 장면에서
  - 하단 중앙 영역(대개 접시/식탁)에 평면을 맞추고
  - 그 평면 좌표계에서 같은 이미지 영역이 몇 mm 로 펼쳐지는지(가로 배율)
  - 평면으로부터의 높이 산포가 몇 배인지(높이 배율)
를 두 조건에서 비교한다. 부피 배율 = 가로배율^2 x 높이배율.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import fit_plane_ransac      # noqa: E402
from paths import CACHE                    # noqa: E402


def region_stats(pts, mask):
    """유효점에 평면을 맞추고 (평면상 가로 산포, 높이 산포) 를 mm 로 반환."""
    p = pts[mask]
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 500:
        return None
    sub = p[np.random.default_rng(0).choice(len(p), min(40000, len(p)), replace=False)]
    pl = fit_plane_ransac(sub, iters=400, thresh_mm=3.0)
    u, v = pl.basis()
    a, b, h = sub @ u, sub @ v, pl.height(sub)
    lateral = float(np.hypot(a.std(), b.std()))
    height = float(np.percentile(h, 95) - np.percentile(h, 5))
    return lateral, height, float(np.median(sub[:, 2]))


def main(ver):
    print(f"=== MoGe {ver}: EXIF 화각 주입이 기하를 얼마나 바꾸는가 ===")
    print(f"{'combo':>6} {'가로배율':>9} {'높이배율':>9} {'깊이배율':>9} {'부피배율':>9}")
    ratios = []
    for f in sorted(CACHE.glob(f"moge{ver}_combo_*.npz")):
        z = np.load(f)
        idx = int(f.stem.split("_")[-1])
        h, w = z["points_exif"].shape[:2]
        # 하단 중앙 60% x 하단 절반 -- 접시와 식탁이 있을 확률이 높은 영역
        box = np.zeros((h, w), bool)
        box[h // 2:, int(0.2 * w):int(0.8 * w)] = True

        out = {}
        for tag in ("exif", "self"):
            m = box & z[f"mask_{tag}"]
            out[tag] = region_stats(z[f"points_{tag}"], m)
        if out["exif"] is None or out["self"] is None:
            print(f"{idx:>6}  (유효점 부족)"); continue

        lat = out["exif"][0] / out["self"][0]
        hei = out["exif"][1] / out["self"][1]
        dep = out["exif"][2] / out["self"][2]
        vol = lat * lat * hei
        ratios.append((lat, hei, dep, vol))
        print(f"{idx:>6} {lat:>9.3f} {hei:>9.3f} {dep:>9.3f} {vol:>9.3f}")

    r = np.array(ratios)
    lv = np.log(r[:, 3])
    print(f"\n  가로 배율 중앙값 {np.median(r[:,0]):.3f}   높이 배율 중앙값 {np.median(r[:,1]):.3f}")
    print(f"  깊이 배율 중앙값 {np.median(r[:,2]):.3f}")
    print(f"  ★ 부피 배율 중앙값 {np.median(r[:,3]):.3f}  "
          f"(= EXIF 를 안 쓰면 부피가 {100*(1/np.median(r[:,3])-1):+.1f}% 어긋난다)")
    print(f"  로그 부피비 표준편차 {lv.std():.3f} -> 장면별 산포 {100*(np.exp(lv.std())-1):.1f}%")


if __name__ == "__main__":
    for ver in ("v2", "v3"):
        if list(CACHE.glob(f"moge{ver}_combo_*.npz")):
            main(ver); print()
