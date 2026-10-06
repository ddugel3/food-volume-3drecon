"""Qwen 에게 부피가 아니라 **구조 파라미터**를 묻는다.

왜 바꾸는가. 접시에서 이미 성공한 패턴이 있다 -- "카메라가 몇 cm 떨어졌냐" 를 묻지 않고
"어떤 접시이고 지름이 얼마냐" 를 물은 뒤 거리는 픽셀에서 계산했다. 그 수정이
public 을 0.50 -> 0.25 로 끌어내린 큰 몫이었다.

반면 "이 라즈베리 몇 mL 냐" 는 원칙을 어긴다. 모델은 사진 속 이 물체를 재는 게 아니라
"라즈베리는 보통 2 mL" 라는 기억을 꺼낸다. 그래서 작은 것들을 일괄로 낮춰 잡고,
리더보드에서 Qwen 무게를 올릴수록 public 이 단조 악화(0.25 -> 0.34)한 원인이 된다.
반대로 private 은 단조 개선(0.34 -> 0.24)인데, 거기엔 기하가 구조적으로 실패하는
그릇 항목이 몰려 있다.

그래서 크기가 아니라 **무차원 형상**만 묻는다. 크기는 전부 우리가 픽셀에서 잰다.

  shape   물체의 형상 계열. 윗면 적분 V_int = ∫h dA 와 진짜 부피의 비 k 가
          계열마다 해석적으로 정해진다(shape_prior.py 참조).
          평면에 놓인 구는 k=0.80, 누운 원통 0.88, 도넛 0.88, 슬래브 1.00 이다.
          '이건 구형인가 슬래브인가' 는 분류 문제라 VLM 이 잘한다.

  용기    테두리 지름은 우리가 잰다(자세 불변인 볼록껍질 장축). 못 보는 것은
          **안쪽 깊이와 단면 형상**뿐이므로 그것만 묻는다. 깊이도 절대값이 아니라
          지름 대비 비율로 받아 무차원을 유지한다.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import ID_MAP, OUTPUTS      # noqa: E402
from qwen_qc import Qwen, overlay_image, label   # noqa: E402

PROMPT = """The region outlined in bright cyan is one serving of "{food}" in a real photo.
I will measure its size from pixels myself. Do NOT estimate any volume or length.
Only classify its SHAPE, which is scale-free.

Choose the shape family that best describes how the food occupies space:
  slab          flat with roughly constant thickness (toast, cheese block, cookie, sandwich)
  wedge         triangular/tapered prism (cake slice, pizza slice, folded tortilla)
  dome          rounded on top, flat on the bottom (a mound, a scoop, half a sphere)
  sphere        rounded on top AND curving back in near the plate (whole fruit, a berry)
  cylinder      elongated round bar lying on its side (sausage, hot dog, carrot, banana)
  torus         ring with a hole (bagel, donut)
  irregular     porous or knobbly with gaps (broccoli floret, chicken wing, curly pasta)

Also state whether it sits in a container (bowl / cup / ramekin) rather than directly
on the plate. If it does, give the container's interior shape and its depth as a
FRACTION of its rim diameter (a shallow sauce cup is about 0.35, a cereal bowl about 0.5).

Think in one sentence, then output the JSON on the last line.
JSON schema:
{{"shape": one of the seven words above,
 "in_container": true or false,
 "container_kind": one of "none","sauce_cup","ramekin","small_bowl","deep_bowl",
 "container_profile": one of "none","hemispherical","conical","cylindrical",
 "depth_over_diameter": number (0 if none),
 "fill_level": one of "none","below_rim","level_with_rim","mounded_above_rim",
 "confidence": 0.0-1.0}}"""


def combo_table():
    out = []
    for r in csv.DictReader(open(ID_MAP)):
        out.append((int(r["combo"]), int(r["id"]), r["food"], r["container"] or ""))
    return out


def main(resume=True):
    q = Qwen()
    out_csv = OUTPUTS / "qwen_shape.csv"
    done = set()
    if resume and out_csv.exists():
        for r in csv.DictReader(open(out_csv)):
            done.add((int(r["combo"]), int(r["id"])))
        print(f"이어하기 {len(done)}건", flush=True)
    f = open(out_csv, "a", buffering=1)
    w = csv.writer(f)
    if not done:
        w.writerow(["combo", "id", "slug", "gt_container", "shape", "in_container",
                    "container_kind", "container_profile", "depth_over_diameter",
                    "fill_level", "confidence", "raw"])
    items = [it for it in combo_table() if (it[0], it[1]) not in done]
    print(f"남은 {len(items)}건", flush=True)
    for k, (cidx, fid, slug, cont) in enumerate(items, 1):
        t0 = time.time()
        try:
            im = overlay_image(cidx, fid)
            if im is None:
                continue
            im = label(im, slug)
            txt = q.ask([im], PROMPT.format(food=slug.replace("_", " ")))
            d = q.parse_json(txt) or {}
            w.writerow([cidx, fid, slug, cont, d.get("shape", "?"),
                        d.get("in_container", ""), d.get("container_kind", ""),
                        d.get("container_profile", ""), d.get("depth_over_diameter", ""),
                        d.get("fill_level", ""), d.get("confidence", ""), txt[:4000]])
            print(f"[{k}/{len(items)}] c{cidx:02d} id{fid:02d} {slug:<18} "
                  f"{d.get('shape'):<10} 용기={d.get('in_container')} "
                  f"{d.get('container_kind','')} d/D={d.get('depth_over_diameter','')} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[{k}/{len(items)}] c{cidx} id{fid} 실패 {type(e).__name__}: {e}", flush=True)
    f.close()
    print(f"✔ {out_csv}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    main()
