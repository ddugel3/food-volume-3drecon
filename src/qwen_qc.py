"""Qwen3.8-27B 로 파이프라인 중간 산출물을 독립 판정한다.

왜 필요한가. 지금 34개 항목 중 어느 숫자를 믿을지 판단할 근거가 부족하다.
내부 지표는 이 실패를 못 잡는다 -- 살사는 실루엣 IoU 0.80 으로 멀쩡해 보이는데
부피는 1.4 mL 로 실제(30~60)의 40분의 1 이다. 컵 안 소스라 마스크가 표면만 잡고
메시도 얇은 원반이라 '실루엣은 맞고 두께가 없는' 상태인데, 실루엣 지표로는
원리적으로 못 걸러낸다. 기하와 **독립인** 판정자가 필요하다.

네 가지를 묻는다.

  plate  접시 종류와 지름 (장면당 1회, 14회)
         파이프라인 전체가 '디너 플레이트 265mm' 가정 위에 서 있다. 샐러드 플레이트
         210mm 라면 길이 21%, 부피 47% 가 달라진다. 기하로는 풀 수 없고 사전지식이
         필요한 유일한 지점이라 여기가 최대 단일 오차원이다.

  mask   마스크가 그 음식만 정확히 덮는지 (34회)
  mesh   생성된 3D 형상이 그 음식처럼 보이는지 (34회)
  volume 우리 숫자가 그 크기의 음식에 타당한지 + 모델 자체 추정치 (34회)

volume 을 묻는 근거: 벤치마크에서 GPT-5.2 단독 VLM 이 MAPE 0.34 로 기하 기반
2위(SGPS 0.31)와 3위(PSHS 0.46) 사이였다. VLM 의 부피 감각 자체가 쓸 만하다.

모든 답은 엄격한 JSON 으로 받는다. 파싱 실패는 그대로 기록해 은폐하지 않는다.
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paths import CACHE, ID_MAP, OUTPUTS, combo_images   # noqa: E402
from scale_plate import unpack                           # noqa: E402
from diag_plate_scale import largest_component           # noqa: E402
from segment import resized                              # noqa: E402
from diag_crop import crop_box                           # noqa: E402

MODEL = "Qwen/Qwen3.8-27B-FP8"
BASE = 2016
QCDIR = OUTPUTS / "qwen_qc"


def combo_table():
    out = {}
    for r in csv.DictReader(open(ID_MAP)):
        out.setdefault(int(r["combo"]), []).append(
            (int(r["id"]), r["food"], r["container"] or None))
    return out


def load_volumes():
    """지금까지의 부피 추정치 (align.csv). 없으면 빈 dict."""
    p = OUTPUTS / "align.csv"
    if not p.exists():
        return {}
    return {int(r["id"]): (float(r["vol_heightfield_ml"]), float(r["vol_mesh_ml"]))
            for r in csv.DictReader(open(p))}


# ---------------------------------------------------------------- 이미지 만들기
def overlay_image(cidx, fid, expand=2.2, size=896, fill=False):
    """원본 크롭 + 마스크 **윤곽선만**. 색을 덮지 않는다.

    처음에는 마스크를 빨강 38% 로 덧칠했는데, 그것이 판정자를 속였다.
    콤보 6 브로콜리를 "붉은 소스 덩어리", 콤보 12 샌드위치를 "케이크 조각" 이라고
    했는데 둘 다 내 덧칠 때문이었다 -- 초록 브로콜리를 빨갛게 칠해놓고
    "이게 브로콜리냐" 고 물은 셈이다. 음식 판별에서 색은 결정적 증거이므로
    가려서는 안 된다. 윤곽선만 그리고 원본 색을 그대로 남긴다.
    """
    seg = np.load(CACHE / f"seg_merged_combo_{cidx:02d}.npz", allow_pickle=True)
    mm = unpack(seg, f"food_{fid}")
    if len(mm) == 0:
        return None
    mask = largest_component(mm[0])
    img = resized(combo_images()[cidx], BASE).convert("RGB")
    W, H = img.size
    x0, y0, x1, y1 = crop_box(mask, expand, W, H)
    arr = np.asarray(img)[y0:y1, x0:x1].astype(np.float32)
    m = mask[y0:y1, x0:x1]
    from scipy import ndimage
    thick = max(3, int(0.004 * max(arr.shape)))
    edge = ndimage.binary_dilation(m, np.ones((thick, thick))) ^ \
        ndimage.binary_erosion(m, np.ones((thick, thick)))
    if fill:
        arr[m] = 0.85 * arr[m] + 0.15 * np.array([255, 40, 40], np.float32)
    arr[edge] = np.array([0, 255, 255], np.float32)      # 시안: 음식 색과 겹치지 않는다
    out = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    s = size / max(out.size)
    return out.resize((round(out.width * s), round(out.height * s)), Image.LANCZOS)


def mesh_views(path, size=384, n=3):
    """메시를 세 방향에서 깊이 음영 점군으로 렌더해 가로로 붙인다.

    오프스크린 GL 없이 numpy 투영만으로 그린다. '이게 바나나처럼 보이나' 를
    묻는 데는 이 정도면 충분하고, 헤드리스 서버에서 의존성이 늘지 않는다.
    """
    import trimesh
    mesh = trimesh.load(path, force="mesh", process=False)
    P = np.asarray(mesh.sample(120000), dtype=np.float64)
    P -= P.mean(0)
    P /= (np.abs(P).max() + 1e-9)
    tiles = []
    for k in range(n):
        th = 2 * np.pi * k / n
        c, s_ = np.cos(th), np.sin(th)
        R = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
        Q = P @ R.T
        u = ((Q[:, 0] * 0.45 + 0.5) * (size - 1)).astype(int)
        v = ((-Q[:, 1] * 0.45 + 0.5) * (size - 1)).astype(int)
        d = Q[:, 2]
        img = np.full((size, size), np.inf)
        np.minimum.at(img, (v, u), -d)
        vis = np.isfinite(img)
        g = np.zeros((size, size), np.uint8)
        if vis.any():
            z = img[vis]
            g[vis] = (235 - 165 * (z - z.min()) / (np.ptp(z) + 1e-9)).astype(np.uint8)
        tiles.append(np.dstack([g, g, g]))
    return Image.fromarray(np.concatenate(tiles, axis=1))


def label(img, text):
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        f = ImageFont.load_default()
    d.rectangle([0, 0, img.width, 30], fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 255, 255), font=f)
    return img


# ---------------------------------------------------------------- 모델
def _fix_fp8_skip_patterns(model_id):
    """FP8 스킵 패턴이 gate_proj 까지 잡아먹는 문제를 고친다.

    체크포인트의 modules_to_not_convert 에는 MoE 라우터인 `...mlp.gate` 가 들어 있다.
    그런데 transformers 의 should_convert_module 은
        re.match(f"{key}", full_name)
    로 **앵커 없는 접두사 매칭**을 해서 `...mlp.gate` 가 `...mlp.gate_proj` 에도 걸린다.
    결과적으로 64개 레이어의 gate_proj 가 전부 FP8 변환에서 빠지고, FP8 원시 가중치가
    bf16 Linear 에 그대로 들어가 출력이 깨진다(로드 리포트에 weight_scale_inv 가
    UNEXPECTED 로 찍히는 것이 그 증거다).

    라우터만 정확히 제외되도록 패턴 끝에 종료 앵커 $ 를 붙인다.
    """
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    qc = getattr(cfg, "quantization_config", None)
    if qc is None:
        return cfg
    pats = qc["modules_to_not_convert"] if isinstance(qc, dict) else \
        getattr(qc, "modules_to_not_convert", None)
    if not pats:
        return cfg
    fixed, n = [], 0
    for k in pats:
        if k.endswith(".gate") or k.endswith("_gate"):
            fixed.append(k + "$"); n += 1
        else:
            fixed.append(k)
    if isinstance(qc, dict):
        qc["modules_to_not_convert"] = fixed
    else:
        qc.modules_to_not_convert = fixed
    print(f"  FP8 스킵 패턴 {n}개에 종료 앵커 추가 (gate_proj 오탐 방지)", flush=True)
    return cfg


class Qwen:
    def __init__(self, model=MODEL, max_new=1800):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(model)
        cfg = _fix_fp8_skip_patterns(model)
        self.m = AutoModelForImageTextToText.from_pretrained(
            model, config=cfg, dtype="auto", device_map="auto").eval()
        self.max_new = max_new

    @torch.no_grad()
    def ask(self, images, prompt, max_new=None):
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt})
        msgs = [{"role": "user", "content": content}]
        inputs = self.proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.m.device)
        out = self.m.generate(**inputs, max_new_tokens=max_new or self.max_new,
                              do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.proc.decode(gen, skip_special_tokens=True).strip()

    @staticmethod
    def parse_json(txt):
        """가장 마지막의 유효한 JSON 객체를 취한다.

        모델이 추론 과정을 먼저 길게 쓰고 마지막에 답을 내므로, 앞쪽에 등장하는
        중괄호 조각에 속으면 안 된다. 뒤에서부터 후보를 훑는다.
        """
        cands = []
        depth = 0
        start = None
        for i, ch in enumerate(txt):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    cands.append(txt[start:i + 1])
                    start = None
        for c in reversed(cands):
            try:
                d = json.loads(c)
                if isinstance(d, dict) and d:
                    return d
            except Exception:
                continue
        return None


# ---------------------------------------------------------------- 프롬프트
P_PLATE = """You are looking at a photo of a food scene on a table.
Identify the main dish-ware the food sits on or in.
Think briefly, then end your reply with the JSON object on the last line.
JSON schema:
{"type": one of "dinner_plate","salad_plate","side_plate","oval_platter","bowl","tray",
 "shape": one of "round","oval",
 "diameter_cm": number (the real-world longest diameter you believe it is),
 "confidence": 0.0-1.0,
 "reason": "short"}"""

P_MASK = """The region outlined in bright cyan is a segmentation mask that is supposed to
cover exactly one food item: "{food}".
Judge the mask. Think briefly, then end your reply with the JSON object on the last line.
JSON schema:
{{"verdict": one of "good","includes_extra","misses_part","wrong_object",
 "coverage": 0.0-1.0 (fraction of the real {food} that the mask covers),
 "reason": "short"}}"""

P_MESH = """These are three views of a 3D shape that was generated from a photo of "{food}".
Judge whether the 3D shape plausibly represents a "{food}".
Think briefly, then end your reply with the JSON object on the last line.
JSON schema:
{{"verdict": one of "good","rough_but_ok","wrong_shape",
 "reason": "short"}}"""

P_VOL = """The region outlined in bright cyan is one serving of "{food}" in a real photo.
Our geometric pipeline estimated its volume as {vol:.0f} mL.
Judge by RELATIVE size against the plate/cutlery. Do NOT count pixels or do arithmetic.
Keep reasoning to two sentences at most, then output the JSON.
JSON schema:
{{"verdict": one of "plausible","too_small","too_large",
 "my_estimate_ml": number (your own best estimate of the volume in mL),
 "confidence": 0.0-1.0,
 "reason": "short"}}"""


# ---------------------------------------------------------------- 실행
def main(tasks="plate,mask,mesh,volume", shard=0, nshards=1, resume=True):
    QCDIR.mkdir(parents=True, exist_ok=True)
    tasks = set(t.strip() for t in tasks.split(",") if t.strip())
    table = combo_table()
    vols = load_volumes()
    q = Qwen()

    out_csv = OUTPUTS / f"qwen_qc_shard{shard}.csv"
    done = set()
    if resume and out_csv.exists():
        for r in csv.DictReader(open(out_csv)):
            done.add((r["task"], int(r["combo"]), int(r["id"])))
        print(f"[shard {shard}] 이어하기 {len(done)}건", flush=True)
    f = open(out_csv, "a", buffering=1)
    w = csv.writer(f)
    if not done:
        w.writerow(["task", "combo", "id", "slug", "verdict", "num", "confidence",
                    "reason", "raw"])

    jobs = []
    if "plate" in tasks:
        jobs += [("plate", c, 0, "") for c in sorted(table)]
    items = [(c, fid, slug) for c in sorted(table) for fid, slug, _ in table[c]]
    for t in ("mask", "mesh", "volume"):
        if t in tasks:
            jobs += [(t, c, fid, slug) for c, fid, slug in items]
    mine = [j for i, j in enumerate(jobs) if i % nshards == shard]
    todo = [j for j in mine if (j[0], j[1], j[2]) not in done]
    print(f"[shard {shard}] 내 몫 {len(mine)}, 남은 {len(todo)}", flush=True)

    for k, (task, cidx, fid, slug) in enumerate(todo, 1):
        t0 = time.time()
        try:
            if task == "plate":
                im = resized(combo_images()[cidx], 1024).convert("RGB")
                txt = q.ask([im], P_PLATE)
                d = q.parse_json(txt) or {}
                w.writerow([task, cidx, 0, "", d.get("type", "?"),
                            d.get("diameter_cm", ""), d.get("confidence", ""),
                            str(d.get("reason", ""))[:120], txt[:6000]])
                print(f"[{shard}] {k}/{len(todo)} plate c{cidx:02d}: "
                      f"{d.get('type')} {d.get('diameter_cm')}cm "
                      f"conf={d.get('confidence')} ({time.time()-t0:.0f}s)", flush=True)
                continue

            if task in ("mask", "volume"):
                im = overlay_image(cidx, fid)
                if im is None:
                    continue
                im = label(im, f"{slug}")
                im.save(QCDIR / f"ov_c{cidx:02d}_{fid:02d}.jpg", quality=90)
                if task == "mask":
                    txt = q.ask([im], P_MASK.format(food=slug.replace("_", " ")))
                    d = q.parse_json(txt) or {}
                    w.writerow([task, cidx, fid, slug, d.get("verdict", "?"),
                                d.get("coverage", ""), "", str(d.get("reason", ""))[:120],
                                txt[:6000]])
                    print(f"[{shard}] {k}/{len(todo)} mask c{cidx:02d} id{fid:02d} "
                          f"{slug:<16} {d.get('verdict')} cov={d.get('coverage')} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                else:
                    v = vols.get(fid, (float("nan"), float("nan")))[1]
                    txt = q.ask([im], P_VOL.format(food=slug.replace("_", " "), vol=v))
                    d = q.parse_json(txt) or {}
                    w.writerow([task, cidx, fid, slug, d.get("verdict", "?"),
                                d.get("my_estimate_ml", ""), d.get("confidence", ""),
                                str(d.get("reason", ""))[:120], txt[:6000]])
                    print(f"[{shard}] {k}/{len(todo)} vol c{cidx:02d} id{fid:02d} "
                          f"{slug:<16} 우리={v:.0f} Qwen={d.get('my_estimate_ml')} "
                          f"{d.get('verdict')} ({time.time()-t0:.0f}s)", flush=True)
                continue

            if task == "mesh":
                mp = None
                for pat in (f"c{cidx:02d}_{fid:02d}_{slug}.glb", f"{fid:02d}_{slug}.glb"):
                    p = OUTPUTS / "meshes" / pat
                    if p.exists():
                        mp = p
                        break
                if mp is None:
                    continue
                im = label(mesh_views(mp), f"{slug} (3 views)")
                im.save(QCDIR / f"mesh_c{cidx:02d}_{fid:02d}.jpg", quality=90)
                txt = q.ask([im], P_MESH.format(food=slug.replace("_", " ")))
                d = q.parse_json(txt) or {}
                w.writerow([task, cidx, fid, slug, d.get("verdict", "?"), "", "",
                            str(d.get("reason", ""))[:120], txt[:6000]])
                print(f"[{shard}] {k}/{len(todo)} mesh c{cidx:02d} id{fid:02d} "
                      f"{slug:<16} {d.get('verdict')} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[{shard}] {k}/{len(todo)} {task} c{cidx} id{fid} 실패: "
                  f"{type(e).__name__}: {e}", flush=True)
    f.close()
    print(f"[shard {shard}] 완료 -> {out_csv}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="plate,mask,mesh,volume")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    main(a.tasks, a.shard, a.nshards)
