"""MoGe 의 오차가 '순수한 스케일 오차'인가?

이 질문이 설계 전체를 가른다.
  - 순수 스케일 오차라면 -> 접시로 alpha 하나만 구해 곱하면 형상이 그대로 맞는다.
                            깊이 모델은 상대 기하 센서로 계속 쓸 수 있다.
  - 형상까지 일그러졌다면 -> 접시로도 못 고친다. 깊이 경로를 통째로 버리고
                            생성형 3D 로 가야 한다.

GT 를 안 보고 판정하는 방법: **크기를 아는 물체의 두 번째 치수**를 본다.
접시의 지름으로 alpha 를 맞췄으니, 지름은 정의상 맞는다. 하지만 접시의 '테두리 높이'는
alpha 를 맞추는 데 쓰지 않은 독립 정보다. 실제 접시 테두리는 10~25mm 다.
alpha 를 곱한 뒤 테두리 높이가 그 범위로 모이면, 스케일만 틀렸고 형상은 맞다는 뜻이다.

두 번째 독립 검사: alpha 를 곱한 뒤 음식 높이가 물리적으로 말이 되는가.
베이글 약 50mm, 연어 스테이크 약 30mm, 핫도그 약 40mm 다.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import fit_plane_ransac                    # noqa: E402
from paths import CACHE, ID_MAP                          # noqa: E402
from scale_plate import PRIORS_MM, hull_extent, unpack   # noqa: E402
from diag_plate_scale import largest_component           # noqa: E402
import csv


def combo_foods():
    out = {}
    for r in csv.DictReader(open(ID_MAP)):
        out.setdefault(int(r["combo"]), []).append((int(r["id"]), r["food"]))
    return out


def main(ver="v3", variant="exif"):
    foods = combo_foods()
    D = PRIORS_MM["dinner_plate"][0]
    print(f"{'combo':>6} {'alpha':>7} {'테두리높이mm':>13} {'단/장':>7}   음식 높이(mm, alpha 적용)")
    rims, ratios, food_rows = [], [], []

    for cidx in range(1, 15):
        seg = np.load(CACHE / f"seg_gdino_sam2_combo_{cidx:02d}.npz", allow_pickle=True)
        dep = np.load(CACHE / f"moge{ver}_combo_{cidx:02d}.npz")
        pm, mv = dep[f"points_{variant}"], dep[f"mask_{variant}"]

        pmask = unpack(seg, "ref_plate")
        if len(pmask) == 0:
            continue
        pmask = largest_component(pmask[0])
        pts = pm[pmask & mv]
        pts = pts[np.isfinite(pts).all(axis=1)]
        z = pts[:, 2]; med = np.median(z); mad = np.median(np.abs(z - med)) + 1e-6
        pts = pts[np.abs(z - med) < 6 * mad]
        sub = pts[np.random.default_rng(0).choice(len(pts), min(60000, len(pts)), replace=False)]

        pl = fit_plane_ransac(sub, iters=600, thresh_mm=4.0)
        u, v = pl.basis()
        major, minor = hull_extent(sub @ u, sub @ v)
        alpha = D / major

        # 접시 테두리 높이 -- alpha 를 맞추는 데 안 쓴 독립 치수
        h = pl.height(sub)
        rim = float(np.percentile(h, 99) - np.percentile(h, 1)) * alpha
        rims.append(rim); ratios.append(minor / major)

        # 음식 높이 (같은 접시 평면 기준)
        fh = []
        for fid, slug in foods[cidx]:
            fm = unpack(seg, f"food_{fid}")
            if len(fm) == 0:
                continue
            m = largest_component(fm[0]) & mv
            fp = pm[m]; fp = fp[np.isfinite(fp).all(axis=1)]
            if len(fp) < 200:
                continue
            hh = pl.height(fp)
            top = float(np.percentile(hh, 98)) * alpha
            fh.append(f"{slug[:12]}={top:.0f}")
            food_rows.append((slug, top))
        print(f"{cidx:>6} {alpha:>7.3f} {rim:>13.1f} {minor/major:>7.3f}   {' '.join(fh)}")

    r = np.array(rims)
    print(f"\n  접시 테두리 높이(alpha 적용): 중앙값 {np.median(r):.1f}mm  "
          f"범위 {r.min():.1f}~{r.max():.1f}  변동계수 {r.std()/r.mean()*100:.0f}%")
    print(f"    실제 접시 테두리는 10~25mm 다.")
    print(f"    보정 전에는 이 값이 {np.median(r)/np.median([D/ m for m in [1]]):.0f}... "
          f"장면마다 20~160mm 로 흩어져 있었다.")
    print(f"  접시 원형성(단/장): 중앙값 {np.median(ratios):.3f}  "
          f"최소 {min(ratios):.3f}  (원형이면 1.0)")
    print(f"\n  음식 높이 분포(alpha 적용): 중앙값 {np.median([t for _,t in food_rows]):.0f}mm  "
          f"범위 {min(t for _,t in food_rows):.0f}~{max(t for _,t in food_rows):.0f}mm")


if __name__ == "__main__":
    main()
