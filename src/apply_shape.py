"""형상 계열 k 와 용기 안쪽 모델을 적용해 항목별 부피를 다시 계산한다.

두 갈래를 만든다.

  k 보정        접시 위 단순 물체. V = k(shape) * V_int.
                k 는 해석해라 손으로 맞춘 값이 아니다(shape_prior.py 참조).
                리더보드에서 균일 수축이 public 을 0.25 -> 0.20 으로 개선했는데
                그 편향의 정체가 이 k 이고, 균일 대신 형상별로 주면 더 정확하다.

  용기 모델     그릇/컵. 바닥을 접시 평면이 아니라 **용기 안쪽 면**으로 잡는다.
                테두리 반지름은 우리가 픽셀에서 재고(자세 불변 볼록껍질 장축),
                안쪽 깊이 비율과 단면 형상만 Qwen 이 준다.
                지금까지 파스타 931 mL(접시 평면부터 셈) / 살사 1.3 mL(표면만 얇게)
                로 양쪽 극단이 나온 것이 전부 '바닥을 몰라서' 였다.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backends.depth import build                          # noqa: E402
from geometry import fit_plane_ransac, heightfield_volume  # noqa: E402
from paths import CACHE, ID_MAP, OUTPUTS, combo_images     # noqa: E402
from intrinsics import from_exif                           # noqa: E402
from scale_plate import hull_extent, unpack                # noqa: E402
from diag_plate_scale import largest_component             # noqa: E402
from segment import resized                                # noqa: E402
from diag_crop import crop_box                             # noqa: E402
from pipeline import plate_distance_mm                     # noqa: E402
from shape_prior import (fill_factor, container_volume,     # noqa: E402
                         interior_solid_volume, measured_depth)

BASE = 2016


def _convex_hull(mask, small=504):
    """마스크의 볼록껍질. 축소해서 계산한 뒤 되돌린다(전해상도는 느리다)."""
    from skimage.morphology import convex_hull_image
    h, w = mask.shape
    m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255)
                   .resize((small, max(8, round(small * h / w))), Image.NEAREST)) > 127
    if not m.any():
        return np.zeros_like(mask)
    hull = convex_hull_image(m)
    return np.asarray(Image.fromarray(hull.astype(np.uint8) * 255)
                      .resize((w, h), Image.NEAREST)) > 127


CONTAINER_REFS = ["ref_bowl", "ref_cup"]


def load_shapes():
    p = OUTPUTS / "qwen_shape.csv"
    out = {}
    if not p.exists():
        return out
    for r in csv.DictReader(open(p)):
        try:
            dr = float(r["depth_over_diameter"]) if r["depth_over_diameter"] else 0.0
        except ValueError:
            dr = 0.0
        out[(int(r["combo"]), int(r["id"]))] = {
            "shape": r["shape"], "in_container": str(r["in_container"]).lower() == "true",
            "profile": r["container_profile"], "depth_ratio": dr,
            "kind": r["container_kind"], "fill": r["fill_level"]}
    return out


def measure(be, cidx, fid, expand=4.0, out_px=1036):
    """크롭 깊이에서 음식 점군과 접시 평면, 그리고 용기 마스크를 가져온다."""
    seg = np.load(CACHE / f"seg_merged_combo_{cidx:02d}.npz", allow_pickle=True)
    fm_full = unpack(seg, f"food_{fid}")
    if len(fm_full) == 0:
        return None
    food = largest_component(fm_full[0])
    z_mm, _, _, plate = plate_distance_mm(cidx)
    img = resized(combo_images()[cidx], BASE)
    W, H = img.size
    x0, y0, x1, y1 = crop_box(food, expand, W, H)
    sub = img.crop((x0, y0, x1, y1))
    cw, ch = sub.size
    sc = out_px / max(cw, ch)
    sub_r = sub.resize((max(16, round(cw * sc)), max(16, round(ch * sc))), Image.LANCZOS)
    K = from_exif(combo_images()[cidx])
    fov = float(2 * np.degrees(np.arctan(cw / (2 * K.fx * (BASE / K.width)))))
    pts, valid = be.points(sub_r, fov)

    def rm(m):
        return np.asarray(Image.fromarray(m[y0:y1, x0:x1].astype(np.uint8) * 255)
                          .resize(sub_r.size, Image.NEAREST)) > 127

    fmask = rm(food) & valid
    ring = rm(plate) & valid & ~fmask
    if ring.sum() < 800 or fmask.sum() < 200:
        return None
    zr = pts[..., 2][ring]; zr = zr[np.isfinite(zr) & (zr > 0)]
    if len(zr) < 200:
        return None
    s = z_mm / float(np.median(zr))
    P = pts[ring] * s; P = P[np.isfinite(P).all(1)]
    P = P[np.random.default_rng(0).choice(len(P), min(40000, len(P)), replace=False)]
    pl = fit_plane_ransac(P, iters=500, thresh_mm=3.0)
    F = pts[fmask] * s; F = F[np.isfinite(F).all(1)]
    v_int, info = heightfield_volume(F, pl, grid_mm=1.0)

    # 용기 마스크가 있으면 테두리 정보도 만든다
    cont = None
    C_pts = None
    for ref in CONTAINER_REFS:
        cm = unpack(seg, ref)
        if len(cm) == 0:
            continue
        c0 = largest_component(cm[0])
        # 이 음식을 담고 있는 용기인가.
        # 교집합으로 판정하면 안 된다 -- SAM 3 는 용기와 음식을 서로 겹치지 않게
        # 분할하므로 용기 마스크는 '음식에 가려지지 않은 그릇 부분' 만 잡는다.
        # 실측 교집합이 음식의 0~6% 밖에 안 돼서 네 항목 전부 탈락했다.
        # 올바른 판정은 포함 관계다: 음식이 용기의 볼록껍질 안에 들어 있는가.
        hull = _convex_hull(c0)
        if (hull & food).sum() < 0.6 * food.sum():
            continue
        cmask = rm(c0) & valid & ~fmask
        if cmask.sum() < 500:
            continue
        C = pts[cmask] * s; C = C[np.isfinite(C).all(1)]
        u, v = pl.basis()
        maj, _ = hull_extent(C @ u, C @ v)
        # 이 용기가 **접시 위에** 놓였는가. 접시 평면을 기준으로 쓸 수 있는지가
        # 여기서 갈린다(으깬감자 컵은 접시가 아니라 식탁에 따로 놓여 있다).
        # 교집합으로 보면 안 된다 -- SAM 은 접시와 그릇을 서로 겹치지 않게 자른다.
        # 그릇이 접시의 볼록껍질 **안에** 들어 있는지를 본다.
        ch_c, ch_p = _convex_hull(c0), _convex_hull(plate)
        on_plate = float((ch_c & ch_p).sum()) / max(1.0, ch_c.sum())
        cont = {"R_mm": maj / 2.0, "center": (float((C @ u).mean()), float((C @ v).mean())),
                "ref": ref, "on_plate": on_plate}
        C_pts = C
        break
    u, v = pl.basis()
    return {"v_int": v_int, "F": F, "plane": pl, "u": u, "v": v,
            "A_cm2": info["footprint_cm2"], "cont": cont, "C_pts": C_pts}


def main(backend="moge_v3", out_name=None, expand=4.0, out_px=1036, rim=False):
    be = build(backend)
    shapes = load_shapes()
    rows = []
    tbl = [(int(r["combo"]), int(r["id"]), r["food"], r["container"] or "")
           for r in csv.DictReader(open(ID_MAP))]
    print(f"{'c':>3} {'id':>3} {'음식':<18} {'shape':<10} {'V_int':>8} {'k보정':>8} "
          f"{'용기모델':>9}  비고")
    for cidx, fid, slug, gt_cont in tbl:
        m = measure(be, cidx, fid, expand=expand, out_px=out_px)
        if m is None:
            print(f"{cidx:>3} {fid:>3} {slug:<18} 측정 실패")
            continue
        s = shapes.get((cidx, fid), {})
        shape = s.get("shape", "")
        k = fill_factor(shape)
        v_k = m["v_int"] * k
        v_cont, note = None, ""
        # 용기 여부는 **대회가 준 표**를 따른다. Qwen 판정만 믿으면 안 된다.
        # Qwen 은 크림치즈(id 2)도 소스컵이라고 봤는데 대회 표에는 용기 항목이
        # 15/24/25/31 네 개뿐이다. 그 오적용 하나가 크림치즈를 30.3 -> 47.4 로
        # 만들었다. 표는 GT 가 아니라 제공된 메타데이터이므로 쓰는 데 문제가 없다.
        is_container = bool(gt_cont)
        if is_container and m["cont"] and (rim or s.get("depth_ratio", 0) > 0.05):
            # 충전선 평면을 음식 표면의 **경계** 점들로 맞춘다.
            # 접시 평면을 쓰면 그릇 높이만큼 통째로 어긋난다(파스타 두께 218mm 사고).
            u, v = m["u"], m["v"]
            a = m["F"] @ u; b = m["F"] @ v
            a0, b0 = a.mean(), b.mean()
            rho = np.hypot(a - a0, b - b0)
            edge = rho >= np.percentile(rho, 85)      # 바깥 15% 링 = 그릇과 만나는 경계
            if edge.sum() >= 200:
                fill = fit_plane_ransac(m["F"][edge], iters=300, thresh_mm=3.0)
            else:
                fill = m["plane"]
            h_fill = fill.height(m["F"])
            if rim and m.get("C_pts") is not None:
                # 깊이도 단면도 묻지 않는다. 테두리를 잰다.
                hp = float(np.median(m["plane"].height(m["F"][edge])))
                H, ri = measured_depth(m["plane"].height(m["C_pts"]), hp,
                                       m["cont"].get("on_plate", 0.0))
                mound = container_volume(a, b, h_fill, 1e-9, "cylindrical")
                if H and mound is not None:
                    R = float(np.sqrt(m["A_cm2"] * 100.0 / np.pi))
                    v_cont = mound + interior_solid_volume(R, H, "hemispherical") * 1e-3
                    note = (f"측정 {ri['mode']} 테두리 {ri['h_rim']:.0f}mm "
                            f"여유 {ri['freeboard']:.0f}mm H={H:.0f}mm")
                else:
                    note = "테두리 측정 실패"
            else:
                v_cont = container_volume(a, b, h_fill, s["depth_ratio"],
                                          s.get("profile") or "hemispherical")
                note = (f"{s.get('kind','')} d/D={s['depth_ratio']} "
                        f"봉우리 {np.clip(h_fill,0,None).max():.0f}mm")
        elif is_container:
            note = "용기 마스크 없음"
        rows.append((cidx, fid, slug, gt_cont, shape, k, m["v_int"], v_k, v_cont))
        print(f"{cidx:>3} {fid:>3} {slug:<18} {shape:<10} {m['v_int']:>8.1f} {v_k:>8.1f} "
              f"{(v_cont if v_cont else float('nan')):>9.1f}  {note}")
    out = OUTPUTS / (out_name or f"shape_{backend}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["combo", "id", "slug", "gt_container", "shape", "k",
                    "v_int", "v_k", "v_container"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], r[4], f"{r[5]:.3f}",
                        f"{r[6]:.3f}", f"{r[7]:.3f}",
                        f"{r[8]:.3f}" if r[8] else ""])
    print(f"\n  ✔ {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="moge_v3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--expand", type=float, default=4.0)
    ap.add_argument("--out-px", type=int, default=1036)
    ap.add_argument("--rim", action="store_true",
                    help="용기 깊이를 Qwen 에게 묻지 않고 테두리에서 잰다")
    a = ap.parse_args()
    main(a.backend, a.out, a.expand, a.out_px, a.rim)
