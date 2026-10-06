"""접시 크기가 이상한 원인이 세그멘테이션인가 깊이인가를 가른다.

가르는 방법: **픽셀 공간 접시 지름**을 같이 잰다. 이건 깊이와 무관하게 마스크만으로
정해진다. 14장의 촬영 구도가 비슷하다면 픽셀 지름도 비슷해야 한다.

  - 픽셀 지름이 비슷한데 mm 지름이 크게 흔들린다  -> 깊이 모델의 절대 스케일 문제
  - 픽셀 지름부터 흔들린다                        -> 마스크가 접시 밖으로 샌 것

추가로, 접시가 실제 265mm 라고 가정했을 때 카메라 거리가 얼마여야 하는지를 역산한다.
휴대폰으로 식탁 위 접시를 찍으면 보통 30~60cm 다. 역산 거리가 그 범위에 들어오는데
MoGe 가 훨씬 먼 거리를 말한다면, 틀린 쪽은 MoGe 다.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import fit_plane_ransac                    # noqa: E402
from paths import CACHE, combo_images                    # noqa: E402
from intrinsics import from_exif                         # noqa: E402
from scale_plate import PRIORS_MM, hull_extent, unpack   # noqa: E402


def largest_component(m):
    from scipy import ndimage
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def main(ver="v3", variant="exif"):
    imgs = combo_images()
    print(f"{'combo':>6} {'접시px':>8} {'접시mm':>9} {'거리mm':>8} {'역산거리mm':>11} "
          f"{'MoGe/역산':>10} {'단/장':>7}")
    rows = []
    for cidx in range(1, 15):
        seg = np.load(CACHE / f"seg_gdino_sam2_combo_{cidx:02d}.npz", allow_pickle=True)
        dep = np.load(CACHE / f"moge{ver}_combo_{cidx:02d}.npz")
        masks = unpack(seg, "ref_plate")
        if len(masks) == 0:
            print(f"{cidx:>6}  접시 마스크 없음"); continue
        m = largest_component(masks[0])

        # --- 픽셀 공간 지름 (깊이 무관) ---
        ys, xs = np.nonzero(m)
        px_major, px_minor = hull_extent(xs.astype(float), ys.astype(float))

        # --- mm 지름 (깊이 의존) ---
        mm = m & dep[f"mask_{variant}"]
        pts = dep[f"points_{variant}"][mm]
        pts = pts[np.isfinite(pts).all(axis=1)]
        # 마스크가 새어 먼 배경을 물었을 때를 대비해 깊이 이상치를 자른다
        z = pts[:, 2]; med = np.median(z); mad = np.median(np.abs(z - med)) + 1e-6
        pts = pts[np.abs(z - med) < 6 * mad]
        sub = pts[np.random.default_rng(0).choice(len(pts), min(60000, len(pts)), replace=False)]
        pl = fit_plane_ransac(sub, iters=600, thresh_mm=4.0)
        u, v = pl.basis()
        mm_major, mm_minor = hull_extent(sub @ u, sub @ v)
        depth = float(np.median(sub[:, 2]))

        # --- 접시가 265mm 라면 카메라는 얼마나 떨어져 있어야 하는가 ---
        # 이미지 리사이즈를 반영한 픽셀 초점거리
        K = from_exif(imgs[cidx])
        H, W = (int(x) for x in dep["rgb_shape"][:2])
        f_px = K.fx * (W / K.width)
        # 접시는 기울어 보이므로 장축(=단축이 아닌 쪽)이 실제 지름의 투영에 가깝다.
        implied = PRIORS_MM["dinner_plate"][0] * f_px / px_major
        rows.append((cidx, px_major, mm_major, depth, implied, mm_minor / mm_major))
        print(f"{cidx:>6} {px_major:>8.0f} {mm_major:>9.1f} {depth:>8.0f} {implied:>11.0f} "
              f"{depth/implied:>10.2f} {mm_minor/mm_major:>7.3f}")

    a = np.array([[r[1], r[2], r[3], r[4]] for r in rows], float)
    px, mmv, dep_, imp = a.T
    print(f"\n  픽셀 지름:  중앙값 {np.median(px):.0f}px  변동계수 {px.std()/px.mean()*100:.1f}%")
    print(f"  mm  지름:  중앙값 {np.median(mmv):.0f}mm  변동계수 {mmv.std()/mmv.mean()*100:.1f}%")
    print(f"  -> 픽셀은 {px.std()/px.mean()*100:.0f}% 밖에 안 흔들리는데 mm 는 "
          f"{mmv.std()/mmv.mean()*100:.0f}% 흔들린다.")
    print(f"\n  접시 265mm 가정시 역산 카메라 거리: 중앙값 {np.median(imp):.0f}mm "
          f"(범위 {imp.min():.0f}~{imp.max():.0f})")
    print(f"  MoGe 가 말하는 거리:                중앙값 {np.median(dep_):.0f}mm "
          f"(범위 {dep_.min():.0f}~{dep_.max():.0f})")
    r = dep_ / imp
    print(f"  ★ MoGe 거리 / 역산 거리 = 중앙값 {np.median(r):.2f}배 "
          f"(로그 표준편차 {np.log(r).std():.3f})")
    print(f"  ★ 부피로는 {np.median(r)**3:.1f}배 오차 -- 절대 스케일 앵커로 쓸 수 없다")


if __name__ == "__main__":
    main()
