"""여러 자를 융합해 장면 스케일을 정한다.

왜 필요한가. 오차를 장면별로 분해하면 같은 콤보의 항목들이 **함께** 밀린다
(콤보 5 는 셋 다 -0.2 대, 항목 내 산포 0.03). 장면별로 배율 하나씩만 고쳐주면
MAPE 가 0.198 -> 0.115 로 떨어진다. 즉 남은 오차의 절반이 **접시 지름 사전지식**이다.

접시 하나에 의존하는 대신 식기를 함께 쓴다. 디너 포크는 19~21cm 로 접시(21~30cm)보다
분산이 작은 자다.

핵심 기법: **평면 좌표계에서 재기.**
픽셀 길이를 그냥 쓰면 안 된다. 포크가 카메라 쪽으로 기울수록 짧아 보이기 때문이다
(실측 포크/접시 픽셀 비가 0.432~0.750 으로 74% 흔들렸다. 실제 비는 0.736 근처여야 한다).
접시는 원형이라 볼록껍질 장축이 자세 불변이지만 포크는 축이 하나뿐이라 그렇지 않다.
깊이맵으로 지지평면을 얻어 그 좌표계로 점을 옮기면 **단축이 사라진다.**
평면의 방향만 쓰므로 깊이의 절대 스케일은 필요 없다.

그러면 비 r = (포크 평면길이)/(접시 평면지름) 가 스케일 무관하게 측정된다.
사전분포 D ~ 접시지름, L ~ 포크길이 에 제약 L = r*D 를 걸고 로그공간에서 풀면
두 사전지식이 서로를 보정한 D 가 나온다.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backends.depth import build                  # noqa: E402
from geometry import fit_plane_ransac             # noqa: E402
from paths import CACHE, OUTPUTS, combo_images    # noqa: E402
from intrinsics import from_exif                  # noqa: E402
from scale_plate import hull_extent, unpack       # noqa: E402
from diag_plate_scale import largest_component    # noqa: E402
from segment import resized                       # noqa: E402

BASE = 2016
# (평균 mm, 로그 표준편차). 식기가 접시보다 표준화되어 있어 분산이 작다.
PRIORS = {"plate": (265.0, 0.075), "fork": (195.0, 0.060), "knife": (220.0, 0.068)}


def measure_inplane(cidx, be, seg_prefix="seg_refs", out_px=1400):
    """전체 프레임 깊이로 지지평면을 얻고, 참조물들의 **평면상 최대 길이**를 잰다."""
    f = CACHE / f"{seg_prefix}_combo_{cidx:02d}.npz"
    if not f.exists():
        f = CACHE / f"seg_merged_combo_{cidx:02d}.npz"
    seg = np.load(f, allow_pickle=True)
    img = resized(combo_images()[cidx], BASE)
    W, H = img.size
    sc = out_px / max(W, H)
    im2 = img.resize((round(W * sc), round(H * sc)), Image.LANCZOS)
    K = from_exif(combo_images()[cidx])
    fov = float(2 * np.degrees(np.arctan(W / (2 * K.fx * (BASE / K.width)))))
    pts, valid = be.points(im2, fov)

    def rm(m):
        return np.asarray(Image.fromarray(m.astype(np.uint8) * 255)
                          .resize(im2.size, Image.NEAREST)) > 127

    pm = unpack(seg, "ref_plate")
    if len(pm) == 0:
        return None
    plate = rm(largest_component(pm[0])) & valid
    P = pts[plate]; P = P[np.isfinite(P).all(1)]
    if len(P) < 1000:
        return None
    P = P[np.random.default_rng(0).choice(len(P), min(40000, len(P)), replace=False)]
    pl = fit_plane_ransac(P, iters=500, thresh_mm=max(1e-6, 0.004 * abs(np.median(P[:, 2]))))
    u, v = pl.basis()
    out = {}
    for ref in ("plate", "fork", "knife"):
        mm = unpack(seg, f"ref_{ref}")
        if len(mm) == 0:
            continue
        m = rm(largest_component(mm[0])) & valid
        if m.sum() < 200:
            continue
        Q = pts[m]; Q = Q[np.isfinite(Q).all(1)]
        if len(Q) < 200:
            continue
        if len(Q) > 30000:
            Q = Q[np.random.default_rng(1).choice(len(Q), 30000, replace=False)]
        maj, _ = hull_extent(Q @ u, Q @ v)
        out[ref] = float(maj)
    return out or None


def fuse_scale(meas):
    """평면상 길이들과 사전분포로 접시 지름을 추정한다.

    측정은 임의 스케일 s 에 대해 len_i = s * L_i 이므로, 각 자에 대해
      log s_i = log(len_i) - log(prior_i)
    이고 이들을 1/sigma^2 로 가중평균한다. 그 s 로 접시 지름을 되돌린다.
    """
    ls, ws = [], []
    for ref, (mu, sig) in PRIORS.items():
        if ref in meas and meas[ref] > 0:
            ls.append(np.log(meas[ref]) - np.log(mu))
            ws.append(1.0 / sig ** 2)
    if not ls:
        return None, None
    ls, ws = np.array(ls), np.array(ws)
    log_s = float(np.sum(ls * ws) / np.sum(ws))
    sig_p = float(np.sqrt(1.0 / np.sum(ws)))
    D = float(meas["plate"] / np.exp(log_s)) if "plate" in meas else None
    return D, sig_p


def main(backend="depthpro"):
    be = build(backend)
    rows = []
    print(f"{'c':>3} {'접시':>8} {'포크':>8} {'나이프':>8} {'포크/접시':>9} "
          f"{'융합 접시mm':>11} {'sigma':>7}")
    for c in range(1, 15):
        m = measure_inplane(c, be)
        if m is None:
            print(f"{c:>3}  측정 실패"); continue
        D, sg = fuse_scale(m)
        fr = m.get("fork", np.nan) / m["plate"] if "plate" in m else np.nan
        rows.append((c, D, sg, fr, m))
        print(f"{c:>3} {m.get('plate',np.nan):>8.1f} {m.get('fork',np.nan):>8.1f} "
              f"{m.get('knife',np.nan):>8.1f} {fr:>9.3f} {(D or 0):>11.1f} {(sg or 0):>7.3f}")
    fr = np.array([r[3] for r in rows], float)
    print(f"\n  평면 보정 후 포크/접시 비: 중앙 {np.nanmedian(fr):.3f} "
          f"변동계수 {np.nanstd(fr)/np.nanmean(fr)*100:.0f}%  (픽셀 기준이던 이전 18%)")
    print(f"  이론 비 (195/265) = 0.736")
    with open(OUTPUTS / "rulers.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["combo", "plate_mm", "sigma_log", "fork_over_plate"])
        for c, D, sg, r, _ in rows:
            w.writerow([c, f"{D:.2f}" if D else "", f"{sg:.4f}" if sg else "", f"{r:.4f}"])
    print(f"  ✔ {OUTPUTS/'rulers.csv'}")


if __name__ == "__main__":
    main()
