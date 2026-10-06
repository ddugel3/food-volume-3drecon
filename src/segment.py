"""음식과 스케일 참조물을 텍스트 프롬프트만으로 분리한다.

손으로 클릭해 마스크를 만들지 않는 것이 설계 제약이다. Kaggle 기본 규칙 4b 가
"Submissions may not use or incorporate information from hand labeling or human
prediction of the test data records" 라고 못박고 있어서, 사람이 찍은 점/박스로
마스크를 만드는 경로는 규칙 위반 소지가 있다. 텍스트 프롬프트는 대회가 배포한
콤보 표(음식 이름)를 쓰는 것이라 안전하다.

백엔드 두 가지를 같은 인터페이스로 둔다.

  sam3        facebook/sam3. 텍스트 개념 프롬프트 하나로 해당 개념의 모든 인스턴스를
              직접 분할(PCS)한다. 멀티푸드에 가장 알맞지만 현재 게이트가 걸려 있다.
  gdino_sam2  Grounding DINO 로 텍스트->박스, SAM 2.1 로 박스->마스크. 고전적인
              Grounded-SAM 조합이고 두 저장소 모두 열려 있다.

음식 이름을 이미 알고 있다는 점을 최대한 쓴다. 일반명사 "food" 보다
"slice of pizza", "hot dog" 처럼 구체적인 명사구가 개방어휘 검출기에서 훨씬 잘 잡힌다.

해상도는 깊이 캐시(MoGe, 긴 변 2016px)와 정확히 일치시킨다.
마스크와 포인트맵을 픽셀 단위로 겹쳐 쓸 것이기 때문이다.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CACHE, ID_MAP, OUTPUTS, combo_images   # noqa: E402

PHRASE = {
    "bagel": "bagel", "cream_cheese": "cream cheese",
    "breaded_fish": "breaded fish fillet", "lemon": "lemon wedge",
    "broccoli": "broccoli", "burger": "hamburger", "hotdog": "hot dog",
    "cheesecake": "slice of cheesecake", "strawberry": "strawberry",
    "raspberry": "raspberry", "energy_bar": "energy bar",
    "cheddar_cheese": "block of cheddar cheese", "banana": "banana",
    "grilled_salmon": "grilled salmon fillet", "pasta": "pasta",
    "garlic_bread": "garlic bread", "pbj": "peanut butter and jelly sandwich",
    "carrot_stick": "carrot stick", "apple": "apple", "celery": "celery stick",
    "pizza": "slice of pizza", "chicken_wing": "chicken wing",
    "quesadilla": "quesadilla", "guacamole": "guacamole", "salsa": "salsa",
    "roast_chicken_leg": "roast chicken leg", "biscuit": "biscuit",
    "sandwich": "sandwich", "cookie": "cookie", "steak": "steak",
    "mashed_potatoes": "mashed potatoes", "toast": "slice of toast",
    "sausage": "sausage", "fried_egg": "fried egg",
}

# 이 대회의 유일한 절대 크기 단서. 접시가 1순위, 식기가 백업이다.
REFERENCES = ["plate", "bowl", "cup", "fork", "spoon", "knife"]

MAX_INSTANCES = 6   # 프롬프트당 보관할 인스턴스 수. 병합 여부는 뒤 단계에서 결정한다.


def combo_table():
    out = {}
    for r in csv.DictReader(open(ID_MAP)):
        out.setdefault(int(r["combo"]), []).append(
            (int(r["id"]), r["food"], r["container"] or None))
    return out


def resized(path: Path, long_side: int) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = long_side / max(w, h)
        return im.resize((round(w * s), round(h * s)), Image.LANCZOS) if s < 1 else im.copy()


class GDinoSam2:
    """Grounding DINO (텍스트->박스) + SAM 2.1 (박스->마스크)."""

    def __init__(self, device="cuda", box_thresh=0.25, text_thresh=0.25):
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                                  Sam2Model, Sam2Processor)
        self.device, self.box_thresh, self.text_thresh = device, box_thresh, text_thresh
        self.gp = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self.gm = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base").to(device).eval()
        self.sp = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
        self.sm = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(device).eval()

    @torch.no_grad()
    def __call__(self, img: Image.Image, phrase: str):
        # Grounding DINO 는 소문자 + 마침표로 끝나는 텍스트를 기대한다.
        text = phrase.lower().rstrip(".") + "."
        inp = self.gp(images=img, text=text, return_tensors="pt").to(self.device)
        out = self.gm(**inp)
        det = self.gp.post_process_grounded_object_detection(
            out, inp["input_ids"], threshold=self.box_thresh,
            text_threshold=self.text_thresh, target_sizes=[(img.height, img.width)])[0]
        boxes = det["boxes"].float().cpu().numpy()
        scores = det["scores"].float().cpu().numpy()
        if len(scores) == 0:
            return np.zeros((0, img.height, img.width), bool), scores, boxes
        order = np.argsort(-scores)[:MAX_INSTANCES]
        boxes, scores = boxes[order], scores[order]

        si = self.sp(images=img, input_boxes=[boxes.tolist()], return_tensors="pt").to(self.device)
        so = self.sm(**si, multimask_output=False)
        masks = self.sp.post_process_masks(so.pred_masks, si["original_sizes"])[0]
        masks = masks.squeeze(1).bool().cpu().numpy()
        return masks, scores, boxes


class Sam3Backend:
    """facebook/sam3 개념 프롬프트 세그멘테이션."""

    def __init__(self, device="cuda", threshold=0.35):
        from transformers import Sam3Model, Sam3Processor
        self.device, self.threshold = device, threshold
        self.p = Sam3Processor.from_pretrained("facebook/sam3")
        self.m = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to(device).eval()

    @torch.no_grad()
    def __call__(self, img: Image.Image, phrase: str):
        inp = self.p(images=img, text=phrase, return_tensors="pt").to(self.device)
        out = self.m(**inp)
        res = self.p.post_process_instance_segmentation(
            out, threshold=self.threshold, mask_threshold=0.5,
            target_sizes=[(img.height, img.width)])[0]
        masks = res["masks"].cpu().numpy().astype(bool)
        scores = res["scores"].float().cpu().numpy()
        boxes = res.get("boxes")
        boxes = boxes.float().cpu().numpy() if boxes is not None else np.zeros((len(scores), 4))
        if len(scores):
            o = np.argsort(-scores)[:MAX_INSTANCES]
            masks, scores, boxes = masks[o], scores[o], boxes[o]
        return masks, scores, boxes


def qc_overlay(img: Image.Image, entries, path: Path):
    """마스크를 눈으로 검증할 오버레이. 자동 파이프라인은 반드시 눈으로 한 번 확인한다."""
    from PIL import ImageDraw, ImageFont
    base = img.copy().convert("RGB")
    arr = np.asarray(base).astype(np.float32)
    palette = [(255, 80, 80), (80, 200, 120), (90, 150, 255), (255, 200, 60),
               (220, 110, 240), (60, 220, 220), (255, 140, 60), (160, 160, 160)]
    draw_items = []
    for i, (key, phrase, masks, scores) in enumerate(entries):
        if len(masks) == 0:
            continue
        col = np.array(palette[i % len(palette)], np.float32)
        m = masks[0]
        arr[m] = 0.55 * arr[m] + 0.45 * col
        ys, xs = np.nonzero(m)
        draw_items.append((key, phrase, scores[0], int(xs.mean()), int(ys.mean()), palette[i % len(palette)]))
    out = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
    except Exception:
        font = ImageFont.load_default()
    for key, phrase, sc, x, y, col in draw_items:
        t = f"{key} {sc:.2f}"
        d.text((x + 2, y + 2), t, fill=(0, 0, 0), font=font)
        d.text((x, y), t, fill=col, font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.resize((out.width // 2, out.height // 2), Image.LANCZOS).save(path, quality=88)


def run(backend_name="gdino_sam2", long_side=2016, device="cuda"):
    backend = {"gdino_sam2": GDinoSam2, "sam3": Sam3Backend}[backend_name](device=device)
    table = combo_table()
    CACHE.mkdir(exist_ok=True)

    for cidx, path in sorted(combo_images().items()):
        img = resized(path, long_side)
        prompts = [(f"food_{fid}", PHRASE.get(slug, slug.replace("_", " ")))
                   for fid, slug, _ in table[cidx]]
        prompts += [(f"ref_{r}", r) for r in REFERENCES]

        store, entries, lines = {}, [], []
        for key, phrase in prompts:
            masks, scores, boxes = backend(img, phrase)
            store[f"{key}__masks"] = (np.packbits(masks, axis=-1) if len(masks)
                                      else np.zeros((0, 0, 0), np.uint8))
            store[f"{key}__mshape"] = np.array(masks.shape if len(masks)
                                               else (0, img.height, img.width))
            store[f"{key}__scores"] = scores
            store[f"{key}__boxes"] = boxes
            store[f"{key}__phrase"] = phrase
            entries.append((key, phrase, masks, scores))
            if len(scores):
                px = masks[0].sum() / (img.height * img.width) * 100
                lines.append(f"{key:12s} {phrase:32s} n={len(scores)} "
                             f"top={scores[0]:.2f} 면적={px:5.2f}%")
            else:
                lines.append(f"{key:12s} {phrase:32s} 검출 없음")

        np.savez_compressed(CACHE / f"seg_{backend_name}_combo_{cidx:02d}.npz",
                            img_shape=np.array((img.height, img.width)), **store)
        qc_overlay(img, entries, OUTPUTS / "qc" / f"seg_{backend_name}_combo_{cidx:02d}.jpg")
        print(f"\n--- combo {cidx:02d} ({path.name}) ---")
        for ln in lines:
            print("   ", ln)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gdino_sam2", choices=["gdino_sam2", "sam3"])
    ap.add_argument("--long-side", type=int, default=2016)
    a = ap.parse_args()
    run(a.backend, a.long_side)
