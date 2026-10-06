"""end-to-end 부피 추정: 34개 id -> mL.

측정으로 확정된 세 가지 사실 위에 세운다.

(1) 깊이 모델의 절대 스케일은 못 쓴다. 접시 지름으로 재보면 거리 2.27배,
    부피 11.7배 어긋나고 장면마다 제멋대로다 (src/diag_plate_scale.py).
(2) 깊이 모델의 상대 기하는 쓸 만하다. 접시 대비 음식의 가로 비가 픽셀 비와 맞는다.
(3) 전체 프레임에서는 작은 물체의 기복이 뭉개진다. 물체별로 크롭해서 다시 돌리면
    같은 물체의 교차장면 종횡비 불일치가 6.29배 -> 1.10배로 떨어진다 (src/diag_crop.py).

그래서 역할을 이렇게 나눈다.

  미터 스케일  <- 접시. 깊이 모델을 전혀 쓰지 않는다.
      원형 접시를 비스듬히 봐도 상의 타원의 '장축'은 기울기 방향에 수직인 지름의
      상이라 단축되지 않는다. 따라서 z = D_true * f_px / P_px 로 카메라 거리가 바로 나온다.
      이게 PSHS(픽셀 바운딩박스, MAPE 0.46)와 다른 점이다 -- 박스가 아니라 마스크
      볼록껍질의 장축을 쓰고, 그 장축이 자세 불변이라는 기하적 근거가 있다.

  형상       <- 물체별 크롭 깊이. 크롭 안에서 접시 링의 중앙 깊이가 z 와 맞도록
                s = z / median(Z_ring) 배 하면 크롭의 임의 스케일이 미터로 고정된다.
                이때 깊이 모델에게 요구하는 것은 '링과 음식의 상대 깊이' 뿐이다.

  부피       <- 접시 평면 좌표계 균일 격자에 높이장을 올려 적분 (src/geometry.py).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends.depth import build                        # noqa: E402
from geometry import fit_plane_ransac, heightfield_volume  # noqa: E402
from paths import CACHE, ID_MAP, OUTPUTS, SUBMISSIONS, combo_images  # noqa: E402
from intrinsics import from_exif                        # noqa: E402
from scale_plate import PRIORS_MM, hull_extent, unpack  # noqa: E402
from diag_plate_scale import largest_component          # noqa: E402
from segment import resized                             # noqa: E402
from diag_crop import crop_box                          # noqa: E402

BASE = 2016
SEG = "merged"   # 세그멘테이션 캐시 접두사 (merged / sam3 / gdino_sam2)


def combo_table():
    out = {}
    for r in csv.DictReader(open(ID_MAP)):
        out.setdefault(int(r["combo"]), []).append(
            (int(r["id"]), r["food"], r["container"] or None))
    return out


PLATE_MM_BY_COMBO = None


def _load_plate_priors():
    """Qwen 이 장면별로 추정한 접시 지름을 읽는다. 없으면 전역 265mm.

    14장 전부에 디너 플레이트 265mm 를 박아 넣는 것이 큰 오차원이었다.
    Qwen 판정은 대부분 250~280mm 로 우리 가정과 맞지만 **콤보 4 만 210mm** 다.
    그 장면(치즈케이크·딸기·라즈베리)은 길이 26%, 부피 2.01배 과대추정된다는 뜻이고,
    실제로 Qwen 이 그 셋을 전부 too_large 로 판정했다.

    접시 지름은 기하로는 풀 수 없고 사전지식이 필요한 유일한 지점이다.
    대회 설명 자체가 "implicit cues and prior knowledge 로 스케일을 추론하라" 고 한다.
    """
    global PLATE_MM_BY_COMBO
    if PLATE_MM_BY_COMBO is not None:
        return PLATE_MM_BY_COMBO
    PLATE_MM_BY_COMBO = {}
    p = OUTPUTS / "qwen_qc_shard0.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            if r["task"] != "plate" or not r["num"]:
                continue
            try:
                cm = float(r["num"])
            except ValueError:
                continue
            # 상식 범위를 벗어나는 판정은 버린다 (사이드 접시 18cm ~ 대형 플래터 34cm)
            if 18.0 <= cm <= 34.0:
                PLATE_MM_BY_COMBO[int(r["combo"])] = cm * 10.0
    return PLATE_MM_BY_COMBO


def plate_distance_mm(cidx, ref="ref_plate", D_mm=None):
    """접시 마스크의 픽셀 장축으로 카메라 거리를 구한다. 깊이 모델을 쓰지 않는다."""
    if D_mm is None:
        D_mm = _load_plate_priors().get(cidx, PRIORS_MM["dinner_plate"][0])
    seg = np.load(CACHE / f"seg_{SEG}_combo_{cidx:02d}.npz", allow_pickle=True)
    m = largest_component(unpack(seg, ref)[0])
    ys, xs = np.nonzero(m)
    px_major, _ = hull_extent(xs.astype(float), ys.astype(float))
    K = from_exif(combo_images()[cidx])
    f_px = K.fx * (BASE / K.width)
    return D_mm * f_px / px_major, px_major, f_px, m


def food_volume(be, cidx, food_key, plate_mask, z_mm, expand=4.0, out_px=1036, grid_mm=1.0):
    seg = np.load(CACHE / f"seg_{SEG}_combo_{cidx:02d}.npz", allow_pickle=True)
    fm_full = unpack(seg, food_key)
    if len(fm_full) == 0:
        return None, "검출 없음"
    food = largest_component(fm_full[0])

    img = resized(combo_images()[cidx], BASE)
    W, H = img.size
    x0, y0, x1, y1 = crop_box(food, expand, W, H)
    sub = img.crop((x0, y0, x1, y1))
    cw, ch = sub.size
    sc = out_px / max(cw, ch)
    sub_r = sub.resize((max(16, round(cw * sc)), max(16, round(ch * sc))), Image.LANCZOS)

    K = from_exif(combo_images()[cidx])
    f_base = K.fx * (BASE / K.width)
    fov_x = float(2 * np.degrees(np.arctan(cw / (2 * f_base))))

    pts, valid = be.points(sub_r, fov_x)

    def remap(m):
        mm = Image.fromarray(m[y0:y1, x0:x1].astype(np.uint8) * 255).resize(sub_r.size, Image.NEAREST)
        return np.asarray(mm) > 127

    fm = remap(food) & valid
    ring = remap(plate_mask) & valid & ~fm
    if ring.sum() < 800 or fm.sum() < 200:
        return None, f"점 부족 (링 {int(ring.sum())}, 음식 {int(fm.sum())})"

    # --- 크롭의 임의 스케일을 미터로 고정: 접시 링이 z_mm 에 오도록 ---
    zr = pts[..., 2][ring]
    zr = zr[np.isfinite(zr) & (zr > 0)]
    if len(zr) < 200:
        return None, "링 깊이 없음"
    s = z_mm / float(np.median(zr))

    P = pts[ring] * s
    P = P[np.isfinite(P).all(1)]
    P = P[np.random.default_rng(0).choice(len(P), min(40000, len(P)), replace=False)]
    pl = fit_plane_ransac(P, iters=500, thresh_mm=3.0)

    F = pts[fm] * s
    F = F[np.isfinite(F).all(1)]
    vol_ml, info = heightfield_volume(F, pl, grid_mm=grid_mm)
    h = pl.height(F)
    return vol_ml, {
        "footprint_cm2": info["footprint_cm2"],
        "h_p98_mm": float(np.percentile(h, 98)),
        "crop_px": cw, "scale": s, "n_food_px": int(fm.sum()),
    }


def run(backend="moge_v3", expand=4.0, tag="baseline"):
    be = build(backend)
    table = combo_table()
    rows, notes = {}, []
    for cidx in sorted(table):
        z_mm, px_major, f_px, plate = plate_distance_mm(cidx)
        print(f"\n--- combo {cidx:02d}  접시 {px_major:.0f}px -> 거리 {z_mm:.0f}mm ---")
        for fid, slug, container in table[cidx]:
            v, info = food_volume(be, cidx, f"food_{fid}", plate, z_mm, expand)
            if v is None:
                print(f"    id{fid:>3} {slug:<18} 실패: {info}")
                notes.append((fid, slug, info)); continue
            rows.setdefault(fid, []).append(v)
            print(f"    id{fid:>3} {slug:<18} {v:8.1f} mL   "
                  f"발자국 {info['footprint_cm2']:5.1f}cm²  최고 {info['h_p98_mm']:5.1f}mm"
                  + (f"  [{container}]" if container else ""))

    # id 5(브로콜리)는 두 장면에 나온다 -> 로그평균으로 합친다 (곱셈적 양이므로)
    final = {}
    for fid, vs in rows.items():
        final[fid] = float(np.exp(np.mean(np.log(vs)))) if len(vs) > 1 else vs[0]
        if len(vs) > 1:
            print(f"\n  id{fid} 두 장면 {[f'{x:.1f}' for x in vs]} -> 로그평균 {final[fid]:.1f} mL "
                  f"(불일치 {max(vs)/min(vs):.2f}배)")

    missing = [i for i in range(1, 35) if i not in final]
    if missing:
        med = float(np.median(list(final.values())))
        print(f"\n  결측 {len(missing)}개 {missing} -> 중앙값 {med:.1f} mL 로 채움")
        for i in missing:
            final[i] = med

    SUBMISSIONS.mkdir(exist_ok=True)
    out = SUBMISSIONS / f"{tag}_{backend}.csv"
    with open(out, "w") as f:
        f.write("id,predicted\n")
        for i in range(1, 35):
            f.write(f"{i},{final[i]:.2f}\n")
    v = np.array([final[i] for i in range(1, 35)])
    print(f"\n  ✔ {out}")
    print(f"  부피 분포: 중앙값 {np.median(v):.1f} mL  범위 {v.min():.1f}~{v.max():.1f}  "
          f"(로그 표준편차 {np.log(v).std():.2f})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="moge_v3")
    ap.add_argument("--expand", type=float, default=4.0)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--seg", default="merged")
    a = ap.parse_args()
    SEG = a.seg
    globals()["SEG"] = a.seg
    run(a.backend, a.expand, a.tag)
