"""참조물(식기) 세그멘테이션을 다시 한다. 음식에 썼던 충돌 인지 탐색을 그대로 적용한다.

현재 상태가 망가져 있다. 콤보 1 에서 ref_fork / ref_knife / ref_spoon 이 전부 629px 로
**같은 영역**을 잡았다. 프롬프트를 하나씩 독립으로 돌렸기 때문이고, 음식에서 딸기와
라즈베리가 IoU 1.000 으로 겹쳤던 것과 같은 실패다. 그래서 포크/접시 픽셀 비가
0.432~0.750 으로 74% 흔들린다 -- 실제 비(디너포크 19.5 / 디너플레이트 26.5 = 0.736)와
맞지 않으니 자로 쓸 수 없다.

식기를 제대로 재야 하는 이유: 디너 포크는 19~21cm 로 접시(21~30cm)보다 **분산이 작은
자**다. 지금 장면별 스케일 오차(|로그오차| 중앙 0.054, 부피로 0.162)가 최종 MAPE 0.198 의
대부분을 차지하므로, 자를 하나 더 얻으면 그만큼 줄어든다.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import CACHE, combo_images     # noqa: E402
from scale_plate import unpack            # noqa: E402
from diag_plate_scale import largest_component  # noqa: E402
from segment import Sam3Backend, resized  # noqa: E402

# 참조물별 동의어. 음식에서 'chicken wing' -> 'fried chicken' 으로 바꾸니 잡혔던 것처럼,
# 외형 서술이 명칭보다 잘 붙는 경우가 많다.
REF_PHRASES = {
    "plate": ["plate", "dinner plate", "round plate"],
    "bowl": ["bowl", "small bowl", "cereal bowl"],
    "cup": ["cup", "small sauce cup", "ramekin"],
    "fork": ["fork", "dinner fork", "metal fork", "table fork"],
    "knife": ["knife", "table knife", "dinner knife", "butter knife"],
    "spoon": ["spoon", "teaspoon", "tablespoon", "metal spoon"],
    "napkin": ["napkin", "paper napkin"],
}
ORDER = ["plate", "bowl", "cup", "fork", "knife", "spoon", "napkin"]
MAX_OVERLAP = 0.25


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum()) / u if u else 0.0


def run(long_side=2016, threshold=0.35, device="cuda"):
    be = Sam3Backend(device=device, threshold=threshold)
    for cidx, path in sorted(combo_images().items()):
        img = resized(path, long_side)
        old = np.load(CACHE / f"seg_merged_combo_{cidx:02d}.npz", allow_pickle=True)
        store = {k: old[k] for k in old.files if k.startswith("food_") or k == "img_shape"}
        taken, report = [], []
        # 음식 마스크는 이미 확정됐으므로 참조물이 그것들을 침범하지 않게 막는다
        for k in old.files:
            if k.endswith("__mshape") and k.startswith("food_"):
                m = unpack(old, k[:-len("__mshape")])
                if len(m):
                    taken.append(largest_component(m[0]))

        for ref in ORDER:
            chosen = None
            for ph in REF_PHRASES[ref]:
                m, s, b = be(img, ph)
                if len(s) == 0:
                    continue
                for j in range(len(s)):
                    cand = largest_component(m[j])
                    if cand.sum() < 300:
                        continue
                    if all(iou(cand, t) < MAX_OVERLAP for t in taken):
                        chosen = (cand, float(s[j]), b[j], ph, j)
                        break
                if chosen:
                    break
            key = f"ref_{ref}"
            if chosen is None:
                store[f"{key}__masks"] = np.zeros((0, 0, 0), np.uint8)
                store[f"{key}__mshape"] = np.array((0, img.height, img.width))
                store[f"{key}__scores"] = np.zeros((0,), np.float32)
                store[f"{key}__boxes"] = np.zeros((0, 4), np.float32)
                store[f"{key}__phrase"] = ""
                report.append(f"{key:<12} 실패")
                continue
            cand, sc, bx, ph, rank = chosen
            taken.append(cand)
            store[f"{key}__masks"] = np.packbits(cand[None], axis=-1)
            store[f"{key}__mshape"] = np.array((1,) + cand.shape)
            store[f"{key}__scores"] = np.array([sc], np.float32)
            store[f"{key}__boxes"] = np.array([bx], np.float32)
            store[f"{key}__phrase"] = ph
            ys, xs = np.nonzero(cand)
            report.append(f"{key:<12} '{ph}' #{rank} 점수={sc:.2f} "
                          f"px={cand.sum():>7} bbox={xs.max()-xs.min()}x{ys.max()-ys.min()}")

        np.savez_compressed(CACHE / f"seg_refs_combo_{cidx:02d}.npz", **store)
        print(f"\n--- combo {cidx:02d} ---", flush=True)
        for r in report:
            print("   ", r, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.35)
    run(threshold=ap.parse_args().threshold)
