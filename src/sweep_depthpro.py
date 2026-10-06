"""Depth Pro 설정 탐색. SOTA 인데 우리가 제대로 못 쓰고 있는지 확인한다.

Depth Pro 는 1536x1536 고정 입력에서 동작하도록 설계됐고 '고주파 디테일 보존' 을
대표 주장으로 내세운다. 우리는 크롭을 1036px 로 넣고 있었고, 게다가 초점거리 환산에
도/라디안 혼동 버그가 있어 35건 중 16건이 죽어 있었다. 버그를 고쳤으니 이제
설정 자체가 문제인지 본다.

축 세 가지를 훑는다.
  해상도   768 / 1036 / 1536      (모델 고유 해상도에 맞추면 좋아지는가)
  초점거리 EXIF 주입 / 자체 추정   (우리 f_px 를 강제하는 것이 이득인가)
  크롭배율 3 / 4 / 6              (물체가 프레임을 얼마나 채워야 하는가)

판정은 GT 없이 두 지표로 한다.
  브로콜리 교차일치도  같은 물체가 콤보 2 와 6 에서 얼마나 같게 나오는가 (정확하지만 표본 1)
  접시 평면성          접시 바닥은 평면이다. 평면 잔차 / 접시 지름 (표본 14)
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backends.depth import HFDepthBackend, unproject      # noqa: E402
from geometry import fit_plane_ransac                     # noqa: E402
from paths import CACHE, OUTPUTS, combo_images            # noqa: E402
from intrinsics import from_exif                          # noqa: E402
from scale_plate import hull_extent, unpack               # noqa: E402
from diag_plate_scale import largest_component            # noqa: E402
from segment import resized                               # noqa: E402
from diag_crop import crop_box                            # noqa: E402

BASE = 2016


class DepthProCfg(HFDepthBackend):
    def __init__(self, use_exif_focal=True, device="cuda"):
        super().__init__("apple/DepthPro-hf", "depthpro", device)
        self.use_exif_focal = use_exif_focal

    @torch.no_grad()
    def points(self, img, fov_x_deg):
        inp = self.proc(images=img, return_tensors="pt").to(self.device)
        out = self.m(**inp)
        post = self.proc.post_process_depth_estimation(
            out, target_sizes=[(img.height, img.width)])[0]
        d = post["predicted_depth"].float().cpu().numpy()
        f_theirs = post.get("focal_length")
        f_theirs = float(f_theirs) if f_theirs is not None else None
        if self.use_exif_focal and f_theirs and f_theirs > 0:
            f_ours = img.width / (2 * np.tan(np.radians(fov_x_deg) / 2))
            d = d * (f_ours / f_theirs)
            fov_use = fov_x_deg
        else:
            fov_use = float(2 * np.degrees(np.arctan(img.width / (2 * f_theirs)))) \
                if f_theirs else fov_x_deg
        valid = np.isfinite(d) & (d > 1e-6)
        d = np.where(valid, d, np.nan)
        return unproject(d, fov_use), valid


def region(be, cidx, fid, expand, out_px):
    seg = np.load(CACHE / f"seg_merged_combo_{cidx:02d}.npz", allow_pickle=True)
    plate = largest_component(unpack(seg, "ref_plate")[0])
    food = largest_component(unpack(seg, f"food_{fid}")[0]) if fid else None
    img = resized(combo_images()[cidx], BASE)
    W, H = img.size
    if fid:
        x0, y0, x1, y1 = crop_box(food, expand, W, H)
    else:
        x0, y0, x1, y1 = 0, 0, W, H
    sub = img.crop((x0, y0, x1, y1))
    cw, ch = sub.size
    s = out_px / max(cw, ch)
    sub_r = sub.resize((max(16, round(cw * s)), max(16, round(ch * s))), Image.LANCZOS)
    K = from_exif(combo_images()[cidx])
    fov = float(2 * np.degrees(np.arctan(cw / (2 * K.fx * (BASE / K.width)))))
    pts, valid = be.points(sub_r, fov)

    def rm(m):
        return np.asarray(Image.fromarray(m[y0:y1, x0:x1].astype(np.uint8) * 255)
                          .resize(sub_r.size, Image.NEAREST)) > 127

    pm = rm(plate) & valid
    fm = (rm(food) & valid) if fid else np.zeros_like(pm)
    ring = pm & ~fm
    if ring.sum() < 800:
        return None
    P = pts[ring]; P = P[np.isfinite(P).all(1)]
    if len(P) < 500:
        return None
    P = P[np.random.default_rng(0).choice(len(P), min(30000, len(P)), replace=False)]
    pl = fit_plane_ransac(P, iters=400, thresh_mm=max(1e-6, 0.004 * abs(np.median(P[:, 2]))))
    u, v = pl.basis()
    maj, _ = hull_extent(P @ u, P @ v)
    h = pl.height(P)
    flat = float(np.percentile(h, 90) - np.percentile(h, 10)) / max(maj, 1e-9)
    out = {"planarity": flat}
    if fid and fm.sum() > 200:
        F = pts[fm]; F = F[np.isfinite(F).all(1)]
        if len(F) > 200:
            lat, _ = hull_extent(F @ u, F @ v)
            out["aspect"] = float(np.percentile(pl.height(F), 98)) / max(lat, 1e-9)
    return out


def main(out_csv):
    rows = []
    for use_exif in (True, False):
        be = DepthProCfg(use_exif_focal=use_exif)
        for out_px in (768, 1036, 1536):
            for expand in (3.0, 4.0, 6.0):
                t0 = time.time()
                try:
                    a = region(be, 2, 5, expand, out_px)
                    b = region(be, 6, 5, expand, out_px)
                    plan = []
                    for c in range(1, 15):
                        r = region(be, c, None, None, out_px)
                        if r:
                            plan.append(r["planarity"])
                    if a and b and "aspect" in a and "aspect" in b:
                        dis = max(a["aspect"], b["aspect"]) / max(1e-9, min(a["aspect"], b["aspect"]))
                    else:
                        dis = float("nan")
                    pm = float(np.median(plan)) if plan else float("nan")
                    rows.append((use_exif, out_px, expand, dis, pm, len(plan)))
                    print(f"  EXIF={use_exif}  {out_px}px  x{expand}  "
                          f"브로콜리불일치={dis:.2f}  평면성={pm:.4f}  "
                          f"접시성공={len(plan)}/14  ({time.time()-t0:.0f}s)", flush=True)
                except Exception as e:
                    print(f"  EXIF={use_exif} {out_px} x{expand} 실패 {type(e).__name__}: {e}",
                          flush=True)
        del be
        import gc
        gc.collect(); torch.cuda.empty_cache()
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["use_exif_focal", "out_px", "expand", "broccoli_disagree",
                    "plate_planarity", "plate_ok"])
        w.writerows(rows)
    print(f"  ✔ {out_csv}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTPUTS / "sweep_depthpro.csv"))
    main(ap.parse_args().out)
