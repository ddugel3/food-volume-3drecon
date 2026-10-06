"""용기 마스크가 실제로 뭘 잡았는지 눈으로 본다. R_rim 이 접시만큼 크게 나와서."""
import csv, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import CACHE, ID_MAP, OUTPUTS, combo_images
from scale_plate import unpack
from diag_plate_scale import largest_component
from segment import resized

BASE = 2016
WANT = {15, 24, 25, 31}
tbl = [(int(r["combo"]), int(r["id"]), r["food"]) for r in csv.DictReader(open(ID_MAP))]
tiles = []
for cidx, fid, food in tbl:
    if fid not in WANT:
        continue
    seg = np.load(CACHE / f"seg_merged_combo_{cidx:02d}.npz", allow_pickle=True)
    img = resized(combo_images()[cidx], BASE).convert("RGB")
    ov = img.copy(); d = ImageDraw.Draw(ov)
    def outline(m, col, w=7):
        from skimage import measure as skm
        for c in skm.find_contours(m.astype(float), 0.5):
            d.line([(float(x), float(y)) for y, x in c] + [(float(c[0][1]), float(c[0][0]))],
                   fill=col, width=w)
    fm = unpack(seg, f"food_{fid}")
    if len(fm): outline(largest_component(fm[0]), (0, 220, 255))
    for ref, col in (("ref_bowl", (255, 170, 0)), ("ref_cup", (255, 60, 120)),
                     ("ref_plate", (140, 140, 150))):
        cm = unpack(seg, ref)
        if len(cm): outline(largest_component(cm[0]), col, 5 if ref == "ref_plate" else 7)
    d.text((30, 30), f"{fid} {food}  cyan=food amber=bowl pink=cup grey=plate", fill=(255, 255, 255))
    tiles.append(ov.resize((760, int(760 * ov.size[1] / ov.size[0]))))
w = max(t.size[0] for t in tiles); h = sum(t.size[1] for t in tiles)
sheet = Image.new("RGB", (w, h), (18, 20, 24)); y = 0
for t in tiles: sheet.paste(t, (0, y)); y += t.size[1]
p = OUTPUTS / "rim_masks.png"; sheet.save(p); print(p, sheet.size)
