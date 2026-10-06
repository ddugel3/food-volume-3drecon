"""세그멘테이션 최종본: SAM 3 우선 + 동의어 탐색 + 충돌 해소 + GDINO 보충.

측정에 근거한 설계다.

  SAM 3 를 1순위로 둔다. 35행 중 28행에서 점수가 올랐고(0.44~0.88 -> 0.90~0.98),
  같은 개념의 인스턴스를 서로 밀어내므로 GDINO 가 못 푼 중복이 사라졌다.
  콤보 4 라즈베리가 딸기와 IoU 1.00 이던 것이 0.00 으로 분리되고 면적도
  0.29% < 0.94% 로 물리적 순서가 맞았다.

  대신 SAM 3 는 5개를 통째로 놓친다(에너지바, 마늘빵, pb&j, 치킨윙, 그리고 GDINO 에서도
  저점수였던 것들과 겹친다). 이건 가중치가 아니라 **어휘** 문제다. 근거:
  `chicken wing` 은 임계 0.25 에서 피자를 0.59 로 오검출하지만
  `fried chicken` 으로 바꾸면 같은 임계에서 진짜 치킨을 0.52 로 잡는다.
  이 음식들이 3D 스캔용 모형처럼 생긴 실물이라 부위 수준 어휘가 잘 안 붙는다.

  그래서 음식마다 **동의어 목록**을 두고, 이미 배정된 마스크와 겹치지 않는
  첫 번째 후보를 받는다. 충돌 인지 탐색이라 중복이 구조적으로 막힌다.

  그래도 비면 GDINO+SAM2.1 캐시로 메운다. 두 모델이 대부분 IoU 0.93~0.99 로
  일치하므로 보충분만 다른 출처인 것은 문제가 되지 않는다.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CACHE, ID_MAP, OUTPUTS, combo_images   # noqa: E402
from scale_plate import unpack                           # noqa: E402
from diag_plate_scale import largest_component           # noqa: E402
from segment import PHRASE, REFERENCES, resized          # noqa: E402

# 1순위 표현이 실패했을 때 시도할 대체 표현. 외형 서술이 부위 이름보다 잘 붙는다.
SYNONYMS = {
    "energy_bar": ["energy bar", "granola bar", "cereal bar", "protein bar",
                   "rectangular snack bar"],
    "garlic_bread": ["garlic bread", "toasted bread", "slice of bread", "bread"],
    "pbj": ["peanut butter and jelly sandwich", "thick sandwich", "square sandwich",
            "sandwich", "slice of bread"],
    "chicken_wing": ["chicken wing", "fried chicken", "breaded chicken",
                     "piece of fried chicken", "chicken drumstick"],
    "fried_egg": ["fried egg", "sunny side up egg", "egg"],
    "cream_cheese": ["cream cheese", "spread of cream cheese", "white spread"],
    "salsa": ["salsa", "tomato salsa", "red sauce in a cup"],
    "guacamole": ["guacamole", "green dip", "avocado dip"],
    "mashed_potatoes": ["mashed potatoes", "bowl of mashed potatoes", "white puree"],
    "celery": ["celery stick", "celery", "green vegetable stick"],
    "sausage": ["sausage", "sausage link", "cooked sausage"],
    "quesadilla": ["quesadilla", "folded tortilla", "tortilla"],
    "biscuit": ["biscuit", "scone", "bread roll"],
}

MAX_OVERLAP = 0.30   # 이미 배정된 음식 마스크와 이 이상 겹치면 거부


def combo_table():
    out = {}
    for r in csv.DictReader(open(ID_MAP)):
        out.setdefault(int(r["combo"]), []).append(
            (int(r["id"]), r["food"], r["container"] or None))
    return out


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum()) / u if u else 0.0


def phrases_for(slug):
    base = SYNONYMS.get(slug)
    if base:
        return base
    return [PHRASE.get(slug, slug.replace("_", " "))]


def pack(masks):
    return (np.packbits(masks, axis=-1) if len(masks) else np.zeros((0, 0, 0), np.uint8))


def run(long_side=2016, threshold=0.35, device="cuda"):
    from segment import Sam3Backend
    be = Sam3Backend(device=device, threshold=threshold)
    table = combo_table()

    for cidx, path in sorted(combo_images().items()):
        img = resized(path, long_side)
        gd = np.load(CACHE / f"seg_gdino_sam2_combo_{cidx:02d}.npz", allow_pickle=True)
        store, taken, report = {}, [], []

        # --- 참조물 먼저 (충돌 검사 대상 아님) ---
        for r in REFERENCES:
            m, s, b = be(img, r)
            store[f"ref_{r}__masks"] = pack(m)
            store[f"ref_{r}__mshape"] = np.array(m.shape if len(m) else (0, img.height, img.width))
            store[f"ref_{r}__scores"] = s
            store[f"ref_{r}__boxes"] = b
            store[f"ref_{r}__phrase"] = r
            store[f"ref_{r}__source"] = "sam3"
            if len(s) == 0:                       # SAM3 가 놓치면 GDINO 로 보충
                gm = unpack(gd, f"ref_{r}")
                if len(gm):
                    store[f"ref_{r}__masks"] = pack(gm)
                    store[f"ref_{r}__mshape"] = np.array(gm.shape)
                    store[f"ref_{r}__scores"] = gd[f"ref_{r}__scores"]
                    store[f"ref_{r}__boxes"] = gd[f"ref_{r}__boxes"]
                    store[f"ref_{r}__source"] = "gdino"

        # --- 음식: 동의어를 순서대로 시도하고 충돌하면 다음으로 ---
        for fid, slug, container in table[cidx]:
            key = f"food_{fid}"
            chosen = None
            for ph in phrases_for(slug):
                m, s, b = be(img, ph)
                if len(s) == 0:
                    continue
                for j in range(len(s)):                 # 점수 순서대로 후보 검사
                    cand = largest_component(m[j])
                    if cand.sum() < 400:
                        continue
                    if all(iou(cand, t) < MAX_OVERLAP for t in taken):
                        chosen = (cand, float(s[j]), b[j], ph, "sam3", j)
                        break
                if chosen:
                    break
            if chosen is None:                          # GDINO 보충
                gm = unpack(gd, key)
                for j in range(len(gm)):
                    cand = largest_component(gm[j])
                    if cand.sum() < 400:
                        continue
                    if all(iou(cand, t) < MAX_OVERLAP for t in taken):
                        chosen = (cand, float(gd[f"{key}__scores"][j]),
                                  gd[f"{key}__boxes"][j], str(gd[f"{key}__phrase"]), "gdino", j)
                        break
            if chosen is None:
                report.append(f"{key:10s} {slug:<18} 실패 (모든 후보 충돌/부재)")
                store[f"{key}__masks"] = pack(np.zeros((0, 0, 0), bool))
                store[f"{key}__mshape"] = np.array((0, img.height, img.width))
                store[f"{key}__scores"] = np.zeros((0,), np.float32)
                store[f"{key}__boxes"] = np.zeros((0, 4), np.float32)
                store[f"{key}__phrase"] = ""
                store[f"{key}__source"] = "none"
                continue
            cand, sc, bx, ph, src, rank = chosen
            taken.append(cand)
            store[f"{key}__masks"] = pack(cand[None])
            store[f"{key}__mshape"] = np.array((1,) + cand.shape)
            store[f"{key}__scores"] = np.array([sc], np.float32)
            store[f"{key}__boxes"] = np.array([bx], np.float32)
            store[f"{key}__phrase"] = ph
            store[f"{key}__source"] = src
            report.append(f"{key:10s} {slug:<18} {src:6s} '{ph}' #{rank} "
                          f"점수={sc:.2f} 면적={cand.sum()/cand.size*100:5.2f}%")

        np.savez_compressed(CACHE / f"seg_merged_combo_{cidx:02d}.npz",
                            img_shape=np.array((img.height, img.width)), **store)
        print(f"\n--- combo {cidx:02d} ---")
        for r in report:
            print("   ", r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--long-side", type=int, default=2016)
    ap.add_argument("--threshold", type=float, default=0.35)
    a = ap.parse_args()
    run(a.long_side, a.threshold)
