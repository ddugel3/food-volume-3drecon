"""제출 CSV 를 보류해 둔 GT 로 채점한다. 예측에는 절대 쓰지 않는다."""
import csv, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import OUTPUTS, SUBMISSIONS

GT = {int(r["id"]): float(r["gt_ml"])
      for r in csv.DictReader(open(OUTPUTS / "_HELD_OUT_gt_volumes.csv"))}
CONT = {15, 24, 25, 31}


def score(path):
    pred = {int(r["id"]): float(r["predicted"]) for r in csv.DictReader(open(path))}
    ape = {i: abs(pred[i] - GT[i]) / GT[i] for i in sorted(GT) if i in pred}
    all_ = np.mean(list(ape.values()))
    c = np.mean([v for i, v in ape.items() if i in CONT])
    o = np.mean([v for i, v in ape.items() if i not in CONT])
    return all_, c, o, ape


if __name__ == "__main__":
    rows = []
    for a in sys.argv[1:]:
        p = Path(a) if Path(a).exists() else SUBMISSIONS / a
        all_, c, o, ape = score(p)
        rows.append((p.name, all_, c, o, ape))
    print(f"{'제출':<28} {'MAPE':>7} {'용기4':>7} {'나머지30':>9}")
    for n, a_, c, o, _ in rows:
        print(f"{n:<28} {a_:>7.4f} {c:>7.4f} {o:>9.4f}")
    if len(rows) == 2:
        print(f"\n{'id':>3} {'항목':<18} {'GT':>8} " +
              " ".join(f"{r[0][:14]:>15}" for r in rows))
        import csv as _c
        names = {int(r["id"]): r["slug"] for r in _c.DictReader(
            open(OUTPUTS / "e5_w0.25_detail.csv"))} if (OUTPUTS / "e5_w0.25_detail.csv").exists() else {}
        for i in sorted(CONT):
            print(f"{i:>3} {names.get(i,''):<18} {GT[i]:>8.1f} " +
                  " ".join(f"{r[4][i]:>15.3f}" for r in rows))
