"""물체 주변을 크롭해서 MoGe 를 다시 돌리면 기복이 살아나는가.

증상: 같은 브로콜리인데 콤보 2 에서는 가로 50mm/높이 39mm(종횡비 0.77),
콤보 6 에서는 가로 218mm/높이 36mm(종횡비 0.165) 로 복원된다.
가로는 두 장면 모두 접시 대비 비율이 맞는데 높이만 4.5배 어긋난다.

가설: 전체 프레임을 한 번에 넣으면 작은 물체에 배정되는 ViT 토큰이 너무 적어
기복이 뭉개진다. 콤보 6 의 브로콜리는 240x196px = 대략 15x12 토큰뿐이다.

검사: 물체 bbox 를 3배로 넓혀(접시 바닥이 함께 들어오도록) 크롭한 뒤 다시 추론하고,
**스케일과 무관한 종횡비 h/sqrt(A)** 를 두 장면에서 비교한다.
크롭은 주점을 중심에서 벗어나게 하므로 복원이 약간 회전하지만, 높이를 그 자리에서
맞춘 평면 기준으로 재므로 회전은 상관없다. 비율만 보므로 절대 스케일도 상관없다.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import fit_plane_ransac                 # noqa: E402
from paths import CACHE, OUTPUTS, combo_images        # noqa: E402
from intrinsics import from_exif                      # noqa: E402
from scale_plate import hull_extent, unpack           # noqa: E402
from diag_plate_scale import largest_component        # noqa: E402
from segment import resized                           # noqa: E402
from depth_moge import load_model                     # noqa: E402


def crop_box(mask, expand, W, H):
    ys, xs = np.nonzero(mask)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) * expand / 2
    x0, x1 = int(max(0, cx - half)), int(min(W, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(H, cy + half))
    return x0, y0, x1, y1


@torch.no_grad()
def measure(model, cidx, food_key, expand, out_px=1036, base=2016):
    seg = np.load(CACHE / f"seg_gdino_sam2_combo_{cidx:02d}.npz", allow_pickle=True)
    img = resized(combo_images()[cidx], base)
    W, H = img.size
    food = largest_component(unpack(seg, food_key)[0])
    plate = largest_component(unpack(seg, "ref_plate")[0])

    if expand is None:                      # 전체 프레임 (대조군)
        x0, y0, x1, y1 = 0, 0, W, H
    else:
        x0, y0, x1, y1 = crop_box(food, expand, W, H)

    sub = img.crop((x0, y0, x1, y1))
    cw, ch = sub.size
    s = out_px / max(cw, ch)
    sub_r = sub.resize((max(8, round(cw * s)), max(8, round(ch * s))), Image.LANCZOS)

    K = from_exif(combo_images()[cidx])
    f_base = K.fx * (base / K.width)                       # base 해상도에서의 초점거리
    fov_x = 2 * np.degrees(np.arctan(cw / (2 * f_base)))   # 크롭의 수평 화각

    t = torch.from_numpy(np.asarray(sub_r).copy()).permute(2, 0, 1).float().div_(255).cuda()
    o = model.infer(t, fov_x=float(fov_x))
    pts = o["points"].float().cpu().numpy()
    valid = o["mask"].cpu().numpy() if "mask" in o else np.isfinite(pts).all(-1)

    # 마스크를 크롭 -> 리사이즈 좌표로 옮긴다
    def remap(m):
        mm = Image.fromarray(m[y0:y1, x0:x1].astype(np.uint8) * 255)
        mm = mm.resize(sub_r.size, Image.NEAREST)
        return np.asarray(mm) > 127

    fm, pm_ = remap(food) & valid, remap(plate) & valid
    ring = pm_ & ~fm
    if ring.sum() < 1500 or fm.sum() < 300:
        return None

    P = pts[ring]; P = P[np.isfinite(P).all(1)]
    P = P[np.random.default_rng(0).choice(len(P), min(40000, len(P)), replace=False)]
    pl = fit_plane_ransac(P, iters=500, thresh_mm=max(1e-4, 0.004 * np.median(P[:, 2])))

    F = pts[fm]; F = F[np.isfinite(F).all(1)]
    u, v = pl.basis()
    lat, _ = hull_extent(F @ u, F @ v)
    h = float(np.percentile(pl.height(F), 98))
    return {"lat": lat, "h": h, "aspect": h / lat, "crop_px": cw,
            "fov": fov_x, "n": int(fm.sum())}


def main():
    model = load_model("v3")
    print("=== 같은 브로콜리(id 5): 크롭 배율에 따른 종횡비 h/가로 ===")
    print("   (스케일 무관 지표. 두 장면이 같아야 정상. 실제 브로콜리는 대략 1.0)")
    print(f"\n{'크롭':>8} {'콤보2 종횡비':>13} {'콤보6 종횡비':>13} {'불일치배수':>11}"
          f"   {'콤보2 크롭px':>12} {'콤보6 크롭px':>12}")
    for expand in (None, 6.0, 4.0, 3.0, 2.0, 1.5):
        a = measure(model, 2, "food_5", expand)
        b = measure(model, 6, "food_5", expand)
        if a is None or b is None:
            print(f"{str(expand):>8}  측정 실패"); continue
        r = max(a["aspect"], b["aspect"]) / min(a["aspect"], b["aspect"])
        tag = "전체" if expand is None else f"x{expand}"
        print(f"{tag:>8} {a['aspect']:>13.3f} {b['aspect']:>13.3f} {r:>11.2f}"
              f"   {a['crop_px']:>12} {b['crop_px']:>12}")


if __name__ == "__main__":
    main()
