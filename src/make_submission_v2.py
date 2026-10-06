"""최종 제출본: 네 갈래 증거를 로그공간에서 강건하게 합친다.

증거 목록과 각각의 성격.

  깊이 백엔드 4종 x 2경로   moge_v3 / moge_v2 / depthpro / da_v2_metric 각각에서
      (높이장, 메시정합) 을 얻는다. 같은 알고리즘에 입력만 바꾼 반복측정이므로
      이들의 흩어짐이 **관측된 sigma** 다. 중앙값 0.19(약 21%).
      개별 백엔드가 통째로 실패하는 일이 있다(피자가 da_v2_metric 에서 5.8 mL,
      나머지 셋은 130 근처). 그래서 평균이 아니라 **로그 중앙값**으로 모은다.

  Qwen 독립 추정             기하와 완전히 독립인 증거다. 34/35 에서 값을 냈고
      24건 plausible / 6건 too_large / 5건 too_small 로 판정했다.
      벤치마크에서 VLM 단독(GPT-5.2)이 MAPE 0.34 였으므로 앙상블 멤버로 쓸 값이 있다.
      결정적으로 우리 두 경로가 크게 갈릴 때 어느 쪽인지 골라준다 --
      연어(22.8 vs 159.5 -> Qwen 160: 메시), 살사(35.5 vs 1.5 -> Qwen 35: 높이장).
      즉 '그릇이면 메시' 같은 단순 규칙으로는 안 되고 항목별 판별이 필요하다.

합치는 방식. 세 증거(높이장 중앙값, 메시정합 중앙값, Qwen)의 **로그 중앙값**을 쓴다.
평균이 아니라 중앙값인 이유는 셋 중 하나가 통째로 틀리는 일이 잦기 때문이다 --
중앙값은 그 하나를 자동으로 무시한다. 세 값이 모일 때만 좁아지고, 하나가 튀면
나머지 둘이 답을 정한다. 항목별 손질은 하지 않는다.

수축은 넣지 않는다. 이전 버전에서 sigma 를 '두 경로 불일치의 절반' 으로 잘못 관측해
적용했다가 물리 타당성이 20/34 -> 19/34 로 나빠졌다. 이제 sigma 를 제대로 관측했지만,
그 sigma 는 '깊이 모델 선택의 불확실성' 이지 '최종 추정치의 사후분포 폭' 이 아니다.
전제가 다르므로 그대로 쓰지 않고, sigma 는 기록만 해서 분석에 남긴다.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import OUTPUTS, SUBMISSIONS   # noqa: E402

BACKENDS = ["moge_v3", "moge_v2", "depthpro", "da_v2_metric"]
VOL_MIN, VOL_MAX = 2.0, 1200.0


def load_backend(b):
    suffix = "" if b == "moge_v3" else f"_{b}"
    rows = []
    for p in sorted(OUTPUTS.glob(f"align{suffix}_shard*.csv")):
        rows += list(csv.DictReader(open(p)))
    return rows


def load_qwen():
    p = OUTPUTS / "qwen_qc_shard0.csv"
    if not p.exists():
        return {}, {}
    est, verd = {}, {}
    for r in csv.DictReader(open(p)):
        if r["task"] != "volume":
            continue
        try:
            v = float(r["num"])
        except (TypeError, ValueError):
            continue
        if v > 0:
            est.setdefault(int(r["id"]), []).append(v)
            verd[int(r["id"])] = r["verdict"]
    return {k: float(np.exp(np.mean(np.log(v)))) for k, v in est.items()}, verd


def logmedian(vals):
    vals = [v for v in vals if v and np.isfinite(v) and v > 0]
    return float(np.exp(np.median(np.log(vals)))) if vals else None


def fuse(h, m, q, verdict, w_qwen=1.0):
    """기하 두 경로와 Qwen 을 로그공간 가중평균으로 합친다.

    이전에는 세 값의 로그 **중앙값** 을 썼는데, 그것이 리더보드에서 드러난 실패의
    원인이었다. public(10개) 점수가 세 버전에서 0.50/0.50/0.51 로 꿈쩍하지 않았는데,
    그 10개 중 9개에서 Qwen 이 무시되고 있었다.

    이유는 이렇다. 세 값의 중앙값은 '기하 두 경로가 서로 일치하면 Qwen 이 자동으로
    이상치가 되어 버려지는' 구조다. 그런데 기하 두 경로는 독립이 아니다 -- 같은
    세그멘테이션 마스크, 같은 접시 스케일, 같은 크롭 위에 서 있다. 공통 원인으로
    함께 틀리면 둘이 사이좋게 일치하고, 진짜 독립인 Qwen 만 소수로 몰려 버려진다.
    **상관된 다수를 신뢰하는 잘못된 집계**였다.

    실제로 private(24개, 그릇·연어처럼 두 경로가 갈리는 항목이 많다)은 0.39 -> 0.29 로
    좋아졌는데 public(두 경로가 함께 틀리는 항목)은 그대로였다. 증상이 진단과 일치한다.

    그래서 가중평균으로 바꾼다. 기하는 두 경로가 하나의 증거원이므로 합쳐서 무게 1,
    Qwen 은 독립 증거원이므로 무게 w_qwen 을 준다. Qwen 이 too_large/too_small 로
    적극 판정한 항목은 그 무게를 두 배로 올린다 -- 단순히 값이 다른 것과
    '틀렸다고 명시한 것' 은 정보량이 다르다.
    """
    geo = [x for x in (h, m) if x and np.isfinite(x) and x > 0]
    if not geo:
        return q
    g = float(np.exp(np.mean(np.log(geo))))
    if not q or not np.isfinite(q) or q <= 0:
        return g
    w = w_qwen * (2.0 if verdict in ("too_large", "too_small") else 1.0)
    return float(np.exp((np.log(g) + w * np.log(q)) / (1.0 + w)))


def main(tag="v2_fused", w_qwen=1.0):
    global W_QWEN
    W_QWEN = w_qwen
    hf_all, mv_all, slugs, conts = {}, {}, {}, {}
    for b in BACKENDS:
        for r in load_backend(b):
            fid = int(r["id"])
            hf, mv = float(r["vol_heightfield_ml"]), float(r["vol_mesh_ml"])
            if hf > 0:
                hf_all.setdefault(fid, []).append(hf)
            if mv > 0:
                mv_all.setdefault(fid, []).append(mv)
            slugs[fid] = r["slug"]
            conts[fid] = r["container"]
    qwen, verd = load_qwen()

    print(f"{'id':>3} {'음식':<18} {'용기':>5} {'높이장':>8} {'메시':>8} {'Qwen':>7} "
          f"{'sigma':>6} {'최종':>8}  {'Qwen판정':<11}")
    out, rows = {}, []
    for fid in sorted(set(hf_all) | set(mv_all)):
        h = logmedian(hf_all.get(fid, []))
        m = logmedian(mv_all.get(fid, []))
        q = qwen.get(fid)
        v = fuse(h, m, q, verd.get(fid, ""), w_qwen=W_QWEN)
        # sigma 는 백엔드 간 산포(같은 알고리즘, 입력만 다름)로 관측한다
        allv = [x for x in hf_all.get(fid, []) + mv_all.get(fid, []) if x > 0]
        sig = float(np.std(np.log(allv))) if len(allv) > 1 else float("nan")
        v = float(np.clip(v, VOL_MIN, VOL_MAX))
        out[fid] = v
        rows.append((fid, slugs.get(fid, ""), conts.get(fid, ""), h, m, q, sig, v,
                     verd.get(fid, "")))
        print(f"{fid:>3} {slugs.get(fid,''):<18} {conts.get(fid,''):>5} "
              f"{(h or 0):>8.1f} {(m or 0):>8.1f} {(q or 0):>7.0f} {sig:>6.2f} "
              f"{v:>8.1f}  {verd.get(fid,''):<11}")

    missing = [i for i in range(1, 35) if i not in out]
    if missing:
        med = float(np.median(list(out.values())))
        print(f"\n  결측 {missing} -> {med:.1f}")
        for i in missing:
            out[i] = med

    SUBMISSIONS.mkdir(exist_ok=True)
    p = SUBMISSIONS / f"{tag}.csv"
    with open(p, "w", newline="") as f:
        f.write("id,predicted\n")
        for i in range(1, 35):
            f.write(f"{i},{out[i]:.2f}\n")
    # 분석용 상세본도 남긴다
    d = OUTPUTS / f"{tag}_detail.csv"
    with open(d, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "slug", "container", "heightfield", "mesh", "qwen",
                    "sigma_log", "final", "qwen_verdict"])
        for r in rows:
            w.writerow([r[0], r[1], r[2],
                        f"{r[3]:.2f}" if r[3] else "", f"{r[4]:.2f}" if r[4] else "",
                        f"{r[5]:.0f}" if r[5] else "",
                        f"{r[6]:.3f}" if np.isfinite(r[6]) else "",
                        f"{r[7]:.2f}", r[8]])
    a = np.array([out[i] for i in range(1, 35)])
    print(f"\n  ✔ {p}")
    print(f"  ✔ {d}")
    print(f"  중앙값 {np.median(a):.1f} mL  범위 {a.min():.1f}~{a.max():.1f}")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2_fused")
    ap.add_argument("--w-qwen", type=float, default=1.0,
                    help="Qwen 증거의 무게. 0 이면 기하만, 1 이면 기하와 동등")
    a = ap.parse_args()
    main(a.tag, a.w_qwen)
