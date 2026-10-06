"""제출 CSV 생성. 두 독립 경로를 로그공간에서 합치고 MAPE 최적 수축을 적용한다.

두 경로는 서로 다른 가정 위에 있다.
  높이장   접시 평면을 바닥으로 보고 깊이맵의 보이는 윗면을 적분. 그릇 항목에서
           빈 공간까지 세어 구조적으로 과대추정한다(파스타 931 mL).
  메시정합 완전한 생성 메시를 EXIF 카메라로 투영해 SAM 3 마스크에 맞춘 뒤 부피를
           읽는다. 바닥 가정이 없어 그릇에 강하지만 생성 형상 오차가 직접 들어와
           분산이 크다.

물리 타당성 대리지표(공개 통념 범위)로 재보면
  높이장 단독      범위내 18/34, 로그오차 중앙값 0.30
  메시정합 단독    15/34, 0.40
  두 경로 기하평균 20/34, 0.25   <- 가장 좋다
이므로 기하평균을 기본으로 한다. 음식별로 손으로 값을 고르는 짓은 하지 않는다
(규칙상 금지이고, 어차피 검증할 GT 도 봉인해 두었다).

수축의 근거. MAPE 는 비대칭이다 -- 과소추정 오차는 항목당 1.0 에서 막히지만
과대추정은 무한대로 열려 있다. log V ~ N(mu, sigma^2) 일 때 E|p/V - 1| 을
최소화하면 최적 조건이 E[e^Y 1{Y>0}] = E[e^Y 1{Y<0}] (Y = log p - log V) 이고
정규분포에서 풀면 m = -sigma^2, 즉 log p* = mu - sigma^2 이다.
sigma 는 추측하지 않고 **두 경로의 불일치**에서 관측한다: sigma_i = |log(hf/mv)| / 2.
다만 불일치가 큰 항목(파스타 sigma=0.85)에 그대로 적용하면 절반으로 깎여
오히려 나빠지므로 sigma 를 0.40 으로 상한한다(수축 최대 15%).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import OUTPUTS, SUBMISSIONS   # noqa: E402

VOL_MIN, VOL_MAX = 3.0, 1200.0   # 접시 위 1인분의 전역 물리 범위 (음식별 값이 아니다)
SIGMA_CAP = 0.40


def load_shards():
    rows = []
    for p in sorted(OUTPUTS.glob("align_shard*.csv")):
        rows += list(csv.DictReader(open(p)))
    if not rows:
        sys.exit("align_shard*.csv 없음")
    return rows


def main(tag="fused", shrink=True):
    rows = load_shards()
    by_id = {}
    for r in rows:
        hf, mv = float(r["vol_heightfield_ml"]), float(r["vol_mesh_ml"])
        by_id.setdefault(int(r["id"]), []).append(
            (hf, mv, r["slug"], r["container"], float(r["sil_iou"]), float(r["fit"])))

    print(f"{'id':>3} {'음식':<18} {'용기':>5} {'높이장':>8} {'메시':>8} {'기하평균':>9} "
          f"{'sigma':>6} {'최종':>8} {'IoU':>5}")
    out = {}
    for fid in sorted(by_id):
        vs = by_id[fid]
        # 같은 id 가 두 장면에 나오면(브로콜리) 장면별로 먼저 합치고 로그평균
        per_scene, sigmas = [], []
        for hf, mv, slug, cont, iou, fit in vs:
            hf = min(max(hf, 1e-6), 1e9)
            mv = min(max(mv, 1e-6), 1e9)
            g = float(np.sqrt(hf * mv))
            per_scene.append(g)
            sigmas.append(abs(np.log(hf / mv)) / 2.0)
        mu = float(np.mean(np.log(per_scene)))
        sig = float(np.mean(sigmas))
        if len(per_scene) > 1:                 # 장면 간 불일치도 불확실성이다
            sig = float(np.hypot(sig, np.std(np.log(per_scene))))
        s_eff = min(sig, SIGMA_CAP)
        v = float(np.exp(mu - s_eff ** 2)) if shrink else float(np.exp(mu))
        v = float(np.clip(v, VOL_MIN, VOL_MAX))
        out[fid] = v
        hf0, mv0, slug, cont, iou, fit = vs[0]
        print(f"{fid:>3} {slug:<18} {cont:>5} {hf0:>8.1f} {mv0:>8.1f} "
              f"{np.exp(mu):>9.1f} {sig:>6.2f} {v:>8.1f} {iou:>5.2f}")

    missing = [i for i in range(1, 35) if i not in out]
    if missing:
        med = float(np.median(list(out.values())))
        print(f"\n  결측 {missing} -> 중앙값 {med:.1f} 로 채움")
        for i in missing:
            out[i] = med

    SUBMISSIONS.mkdir(exist_ok=True)
    p = SUBMISSIONS / f"{tag}.csv"
    with open(p, "w", newline="") as f:
        f.write("id,predicted\n")
        for i in range(1, 35):
            f.write(f"{i},{out[i]:.2f}\n")
    a = np.array([out[i] for i in range(1, 35)])
    print(f"\n  ✔ {p}")
    print(f"  중앙값 {np.median(a):.1f} mL  범위 {a.min():.1f}~{a.max():.1f}  "
          f"로그표준편차 {np.log(a).std():.2f}")
    b = by_id.get(5, [])
    if len(b) > 1:
        g = [float(np.sqrt(x[0] * x[1])) for x in b]
        print(f"  브로콜리 교차재현성: {g[0]:.1f} vs {g[1]:.1f} -> {max(g)/min(g):.2f}배")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fused")
    ap.add_argument("--no-shrink", action="store_true")
    a = ap.parse_args()
    main(a.tag, shrink=not a.no_shrink)
