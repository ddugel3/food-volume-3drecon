"""융합 규칙이 실제로 일을 하는가. 유도한 규칙 vs 단순 대안들.

"식은 그럴듯한데 결국 CSV 만들어서 점수 본 것 아니냐"에 답하려면
같은 입력에 규칙만 바꿔 끼워봐야 한다. 채점은 보류 GT 로 한다(사후 분석).
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_grid as G                                      # noqa: E402
from fuse import fuse                                      # noqa: E402
from paths import OUTPUTS                                  # noqa: E402

GT = {int(r["id"]): float(r["gt_ml"])
      for r in csv.DictReader(open(OUTPUTS / "_HELD_OUT_gt_volumes.csv"))}


def collect():
    G.ALIGN_GLOB = "align_depthpro_e5_shard*.csv"
    G.SHAPE_GLOB = "shape_depthpro_e5.0.csv"
    G.CONT_GLOB = "shape_depthpro_e5_rim.csv"
    hf, mv, slug, cont = G.gather(["depthpro"])
    qwen, verd = G.load_qwen()
    shp = G.load_shape_table()
    items = {}
    for fid in sorted(set(hf) | set(mv) | set(qwen)):
        if fid not in GT:
            continue
        h = G.logmed(shp[fid]["vk"]) if fid in shp and shp[fid]["vk"] else G.logmed(hf.get(fid, []))
        m = G.logmed(mv.get(fid, []))
        allv = [x for x in hf.get(fid, []) + mv.get(fid, []) if x > 0]
        items[fid] = {"h": h, "m": m, "q": qwen.get(fid), "verd": verd.get(fid, ""),
                      "spread": float(np.std(np.log(allv))) if len(allv) > 1 else None,
                      "vc": G.logmed(shp[fid]["vc"]) if fid in shp and shp[fid]["vc"] else None}
    return items


def mape(pred):
    return float(np.mean([abs(pred[i] - GT[i]) / GT[i] for i in pred]))


RULES = {}


def rule(name):
    def deco(f):
        RULES[name] = f
        return f
    return deco


@rule("높이장 단독")
def _(d): return d["h"]


@rule("메시 단독")
def _(d): return d["m"]


@rule("Qwen 단독")
def _(d): return d["q"]


@rule("산술평균 (선형공간)")
def _(d):
    v = [x for x in (d["h"], d["m"], d["q"]) if x]
    return float(np.mean(v)) if v else None


@rule("기하평균 (무게 없음)")
def _(d):
    v = [x for x in (d["h"], d["m"], d["q"]) if x]
    return float(np.exp(np.mean(np.log(v)))) if v else None


@rule("중앙값 (로그공간)")
def _(d):
    v = [x for x in (d["h"], d["m"], d["q"]) if x]
    return float(np.exp(np.median(np.log(v)))) if v else None


@rule("역분산 · w=1 (이론 형태)")
def _(d): return fuse(d["h"], d["m"], d["q"], d["verd"], d["spread"],
                      w_qwen_scale=1.0, shrink=False)[0]


@rule("역분산 · w=2 (이론 무게)")
def _(d): return fuse(d["h"], d["m"], d["q"], d["verd"], d["spread"],
                      w_qwen_scale=2.0, shrink=False)[0]


@rule("역분산 · w=0.25 (실사용)")
def _(d): return fuse(d["h"], d["m"], d["q"], d["verd"], d["spread"],
                      w_qwen_scale=0.25, shrink=False)[0]


@rule("역분산 · w=0.25 + 수축")
def _(d): return fuse(d["h"], d["m"], d["q"], d["verd"], d["spread"],
                      w_qwen_scale=0.25, shrink=True)[0]


def main():
    items = collect()
    print(f"  항목 {len(items)}개 · 용기 4개는 모든 규칙에서 동일하게 용기모델 값 사용\n")
    print(f"  {'규칙':<26} {'MAPE':>7}")
    out = []
    for name, f in RULES.items():
        pred = {}
        for fid, d in items.items():
            v = d["vc"] if d["vc"] else f(d)
            if v and np.isfinite(v):
                pred[fid] = float(np.clip(v, G.VOL_MIN, G.VOL_MAX))
        if len(pred) < 30:
            print(f"  {name:<26} {'계산불가':>7} (n={len(pred)})")
            continue
        out.append((name, mape(pred), len(pred)))
        print(f"  {name:<26} {mape(pred):>7.4f}  n={len(pred)}")
    best = min(out, key=lambda r: r[1])
    print(f"\n  최저: {best[0]}  {best[1]:.4f}")


if __name__ == "__main__":
    main()
