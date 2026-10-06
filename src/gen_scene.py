"""장면 통합 복원. CVPR2025 1위 MDMS 가 쓴 방식이다.

지금까지 우리는 물체를 **하나씩 따로** 복원했다. 그러면 같은 접시 위 음식들의
크기가 서로 모순될 수 있다 -- 버거와 핫도그가 각자 다른 스케일로 맞춰진다.
접시와 모든 음식을 한 메시로 만들면 그 안에서 상대 크기가 자동으로 일관되고,
접시가 함께 들어 있으므로 접시 지름 하나가 장면 전체를 정한다.

절차:
  1) 접시 + 모든 음식 마스크의 합집합으로 컷아웃 -> Hunyuan3D 로 단일 메시
  2) EXIF 카메라로 투영해 **합집합 실루엣**에 맞춘다 (물체 하나보다 훨씬 제약이 강하다)
  3) 복셀화한 뒤 각 복셀을 이미지로 투영해 어느 음식 마스크에 떨어지는지로 배정
     (접시 평면 아래 복셀은 접시 자체이므로 버린다)

MDMS 는 3) 을 k-means 로 했지만 우리는 2D 마스크가 이미 정확하므로(35/35 good)
투영 배정이 더 곧다.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "external/Hunyuan3D-2.1/hy3dshape"))
from paths import CACHE, ID_MAP, OUTPUTS, combo_images   # noqa: E402
from scale_plate import unpack                           # noqa: E402
from diag_plate_scale import largest_component           # noqa: E402
from segment import resized                              # noqa: E402
from diag_crop import crop_box                           # noqa: E402

BASE = 2016


def scene_masks(cidx):
    seg = np.load(CACHE / f"seg_merged_combo_{cidx:02d}.npz", allow_pickle=True)
    plate = largest_component(unpack(seg, "ref_plate")[0])
    foods = {}
    for k in seg.files:
        if k.startswith("food_") and k.endswith("__mshape"):
            fid = int(k[len("food_"):-len("__mshape")])
            m = unpack(seg, f"food_{fid}")
            if len(m):
                foods[fid] = largest_component(m[0])
    return plate, foods


def scene_cutout(cidx, expand=1.08, size=896):
    plate, foods = scene_masks(cidx)
    union = plate.copy()
    for m in foods.values():
        union |= m
    img = resized(combo_images()[cidx], BASE).convert("RGB")
    W, H = img.size
    x0, y0, x1, y1 = crop_box(union, expand, W, H)
    rgb = np.asarray(img)[y0:y1, x0:x1]
    a = (union[y0:y1, x0:x1] * 255).astype(np.uint8)
    out = Image.fromarray(np.dstack([rgb, a]), "RGBA")
    s = size / max(out.size)
    return out.resize((round(out.width * s), round(out.height * s)), Image.LANCZOS), \
        (x0, y0, x1, y1), union


def main(steps=30, octree=352):
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
    md = OUTPUTS / "meshes_scene"; md.mkdir(parents=True, exist_ok=True)
    for c in range(1, 15):
        img, box, union = scene_cutout(c)
        t0 = time.time()
        mesh = pipe(image=img, num_inference_steps=steps, octree_resolution=octree,
                    output_type="trimesh")[0]
        p = md / f"scene_c{c:02d}.glb"
        mesh.export(p)
        img.save(OUTPUTS / "qc" / f"scenecut_c{c:02d}.png")
        np.savez(OUTPUTS / f"scene_box_c{c:02d}.npz", box=np.array(box))
        print(f"  c{c:02d} 정점 {len(mesh.vertices):>8} 면 {len(mesh.faces):>8} "
              f"부피 {abs(mesh.volume):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  ✔ {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--octree", type=int, default=352)
    a = ap.parse_args()
    main(a.steps, a.octree)
