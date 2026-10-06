"""제출 후보 격자 생성. 검증 데이터가 없으므로 유망한 조합을 다 만들어 두고
리더보드(하루 1회)로 하나씩 판정한다.

축:
  w_qwen   0(기하만) / 0.5 / 1 / 2 / 4 / inf(Qwen만)
           역분산 유도로는 (sigma_geo/sigma_qwen)^2 = (0.5/0.35)^2 ~ 2 가 이론값이다.
  shrink   MAPE 최적 수축 exp(-sigma_p^2) 적용 여부
  backends 어떤 깊이 백엔드를 쓸지
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fuse import fuse                        # noqa: E402
from paths import OUTPUTS, SUBMISSIONS       # noqa: E402

ALL_BACKENDS = ["moge_v3", "moge_v2", "depthpro", "da_v2_metric"]
VOL_MIN, VOL_MAX = 2.0, 1200.0


ALIGN_GLOB = None     # None 이면 아래 기본 규칙. 크롭 배율 변형을 명시적으로 고를 때 쓴다.


def load_backend(b):
    """정합 결과(경로 A 높이장 + 경로 B 메시)를 읽는다.

    주의: 기본 규칙 f"align_{b}_shard*.csv" 는 크롭 배율 변형을 **잡지 못한다**.
    align_depthpro_e5_shard0.csv 는 'align_depthpro_shard*' 에 매칭되지 않아서,
    e5 라고 이름 붙인 제출이 실제로는 메시 경로만 e4 를 쓰고 있었다.
    형상 테이블(v_k)은 shape_depthpro_e5.0.csv 로 e5 였으므로 두 경로의 크롭
    배율이 서로 달랐다. ALIGN_GLOB 로 명시한다.
    """
    suffix = "" if b == "moge_v3" else f"_{b}"
    pat = ALIGN_GLOB or f"align{suffix}_shard*.csv"
    rows = []
    for p in sorted(OUTPUTS.glob(pat)):
        rows += list(csv.DictReader(open(p)))
    return rows


def load_qwen():
    p = OUTPUTS / "qwen_qc_shard0.csv"
    est, verd = {}, {}
    if not p.exists():
        return est, verd
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
    return ({k: float(np.exp(np.mean(np.log(v)))) for k, v in est.items()}, verd)


SHAPE_GLOB = "shape_*.csv"
CONT_GLOB = None      # 용기값만 따로 어디서 가져올지. None 이면 SHAPE_GLOB 과 같다.


def load_shape_table():
    """형상 계열 k 와 용기 모델 부피. apply_shape.py 가 만든다.

    용기값은 출처를 따로 지정할 수 있다. 테두리 측정본(--rim)과 Qwen 스칼라본을
    한 통에 넣고 중앙값을 내면 둘이 섞여서 어느 쪽 효과인지 알 수 없다.
    """
    out = {}
    cont_only = set()
    if CONT_GLOB:
        cont_only = {q.name for q in OUTPUTS.glob(CONT_GLOB)}
    for p in sorted(OUTPUTS.glob(SHAPE_GLOB)) + \
            [q for q in sorted(OUTPUTS.glob(CONT_GLOB or SHAPE_GLOB))
             if q.name not in {r.name for r in OUTPUTS.glob(SHAPE_GLOB)}]:
        for r in csv.DictReader(open(p)):
            fid = int(r["id"])
            d = out.setdefault(fid, {"vk": [], "vc": [], "shape": r["shape"]})
            if not (cont_only and p.name in cont_only
                    and p.name not in {q.name for q in OUTPUTS.glob(SHAPE_GLOB)}):
                try:
                    vk = float(r["v_k"])
                    if vk > 0:
                        d["vk"].append(vk)
                except (TypeError, ValueError):
                    pass
            if r.get("v_container") and (not cont_only or p.name in cont_only):
                try:
                    vc = float(r["v_container"])
                    if vc > 0:
                        d["vc"].append(vc)
                except (TypeError, ValueError):
                    pass
    return out


def gather(backends):
    hf, mv, slug, cont = {}, {}, {}, {}
    for b in backends:
        for r in load_backend(b):
            fid = int(r["id"])
            h, m = float(r["vol_heightfield_ml"]), float(r["vol_mesh_ml"])
            if h > 0:
                hf.setdefault(fid, []).append(h)
            if m > 0:
                mv.setdefault(fid, []).append(m)
            slug[fid] = r["slug"]; cont[fid] = r["container"]
    return hf, mv, slug, cont


def logmed(v):
    v = [x for x in v if x and np.isfinite(x) and x > 0]
    return float(np.exp(np.median(np.log(v)))) if v else None


def build(backends, w_qwen, shrink, tag, use_shape=False, use_container=False):
    hf, mv, slug, cont = gather(backends)
    qwen, verd = load_qwen()
    shp = load_shape_table() if (use_shape or use_container) else {}
    out = {}
    for fid in sorted(set(hf) | set(mv) | set(qwen)):
        h, m = logmed(hf.get(fid, [])), logmed(mv.get(fid, []))
        allv = [x for x in hf.get(fid, []) + mv.get(fid, []) if x > 0]
        spread = float(np.std(np.log(allv))) if len(allv) > 1 else None
        q = qwen.get(fid)
        # 형상 계열 k 를 적용한 높이장 값으로 교체 (해석해 기반 보정)
        if use_shape and fid in shp and shp[fid]["vk"]:
            h = logmed(shp[fid]["vk"])
        # 용기 안쪽 모델이 있으면 그것이 그 항목의 답이다 (바닥 가정이 유일하게 옳다)
        if use_container and fid in shp and shp[fid]["vc"]:
            vc = logmed(shp[fid]["vc"])
            out[fid] = float(np.clip(vc, VOL_MIN, VOL_MAX))
            continue
        if w_qwen == float("inf"):
            v = q if q else (logmed([x for x in (h, m) if x]) or 0)
        elif w_qwen == 0:
            v, _ = fuse(h, m, None, backend_spread=spread, shrink=shrink)
        else:
            v, _ = fuse(h, m, q, verd.get(fid, ""), backend_spread=spread,
                        w_qwen_scale=w_qwen, shrink=shrink)
        if v:
            out[fid] = float(np.clip(v, VOL_MIN, VOL_MAX))
    miss = [i for i in range(1, 35) if i not in out]
    if miss:
        med = float(np.median(list(out.values()))) if out else 100.0
        for i in miss:
            out[i] = med
    SUBMISSIONS.mkdir(exist_ok=True)
    p = SUBMISSIONS / f"{tag}.csv"
    with open(p, "w") as f:
        f.write("id,predicted\n")
        for i in range(1, 35):
            f.write(f"{i},{out[i]:.2f}\n")
    a = np.array([out[i] for i in range(1, 35)])
    return p, np.median(a), a.min(), a.max(), len(miss)


def main():
    combos = []
    for w in (0.15, 0.25, 0.35, 0.5):
        combos.append((["depthpro"], w, False, True, True, f"e5_w{w:g}"))
    print(f"{'태그':<26} {'중앙값':>8} {'최소':>7} {'최대':>8} {'결측':>4}")
    for bks, w, sh, us, uc, tag in combos:
        p, med, lo, hi, miss = build(bks, w, sh, tag, us, uc)
        print(f"{tag:<26} {med:>8.1f} {lo:>7.1f} {hi:>8.1f} {miss:>4}")


if __name__ == "__main__":
    main()
