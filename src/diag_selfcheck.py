"""GT 없이 쓸 수 있는 자체 점검 지표들.

"하이퍼파라미터를 GT 보고 맞춘 것 아니냐"에 대한 방어는 두 가지여야 한다.
(1) 코드가 GT 를 읽지 않는다 -> scripts/run_blind.py 로 강제 검증.
(2) 선택 자체가 GT 없이 가능했다 -> 이 파일.

여기서 쓰는 신호는 전부 대회 중에도 쓸 수 있었던 것들이다.

  cross_scene   브로콜리(id 5)는 콤보 2 와 6 에 **같은 물체**로 나온다.
                두 측정이 얼마나 어긋나는지는 GT 가 필요 없다.
                다만 반복 물체가 하나뿐이라 n=1 이다. 전체 설정 순위와의
                Spearman 은 0.06 으로, 정확도의 일반 대리지표가 **아니다**.
                크롭 배율처럼 한 축을 훑을 때의 안정성 판정에만 쓴다.
  iou           투영 실루엣과 마스크의 겹침. 자세탐색이 스스로 내는 점수.
  spread        같은 항목에 대한 경로 A/B 의 로그 산포. 융합의 sigma 로 이미 쓴다.
"""
import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import OUTPUTS                                  # noqa: E402

REPEATED = 5     # 브로콜리. 콤보 2 와 6 에 같은 물체로 등장한다.


def cross_scene(pattern="shape_*.csv", col="v_k"):
    """같은 물체를 두 장면에서 잰 값의 |log| 불일치. 낮을수록 안정적."""
    out = []
    for f in sorted(glob.glob(str(OUTPUTS / pattern))):
        vals = []
        for r in csv.DictReader(open(f)):
            if int(r["id"]) != REPEATED:
                continue
            try:
                v = float(r[col])
            except (TypeError, ValueError):
                continue
            if v > 0:
                vals.append(v)
        if len(vals) == 2:
            out.append((os.path.basename(f), float(abs(np.log(vals[0] / vals[1])))))
    return sorted(out, key=lambda r: r[1])


def align_quality(pattern="align*_shard*.csv"):
    """정합의 내부 품질. sil_iou 는 높을수록, fit 잔차는 낮을수록 좋다. GT 무관."""
    out = []
    for f in sorted(glob.glob(str(OUTPUTS / pattern))):
        iou, res = [], []
        for r in csv.DictReader(open(f)):
            for k, acc in (("sil_iou", iou), ("fit", res)):
                try:
                    acc.append(float(r[k]))
                except (TypeError, ValueError, KeyError):
                    pass
        if iou:
            out.append((os.path.basename(f), float(np.mean(iou)),
                        float(np.mean(res)) if res else float("nan"), len(iou)))
    return sorted(out, key=lambda r: -r[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="shape_*.csv")
    a = ap.parse_args()
    print("== 같은 물체 두 장면 불일치 (GT 무관, n=1) ==")
    for n, d in cross_scene(a.pattern):
        print(f"  {n:<34} |log| {d:>6.3f}")
    print("\n== 정합 내부 품질 (GT 무관) ==")
    for n, i, r, c in align_quality():
        print(f"  {n:<34} IoU {i:>5.3f}  잔차 {r:>6.3f}  n={c}")


if __name__ == "__main__":
    main()
