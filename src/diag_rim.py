"""용기의 테두리를 '물어보지 않고 재는' 실험.

지금 용기 모델은 안쪽 깊이 H = depth_ratio * 2R 과 단면 profile 을 전부 Qwen 에게
받는다. 그런데 그릇 테두리는 사진에 보인다. 깊이맵에 이미 용기 점군 C 가 잡히고
있는데(apply_shape.measure) 반지름만 쓰고 **높이를 버리고 있었다**.

여기서 재는 것:
  h_rim     접시 평면 위로 솟은 용기 테두리 높이 (mm)  -> 그릇이 얼마나 깊은가
  h_fill    음식 표면 경계(충전선)의 접시 평면 기준 높이
  r(h)      높이별 용기 반지름 -> 단면이 원뿔인지 원통인지 **측정**
그리고 이걸로 계산한 부피를 Qwen 스칼라 버전, 접시기준 높이장과 나란히 놓는다.
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from apply_shape import measure, load_shapes, CONTAINER_REFS   # noqa: E402
from backends.depth import build                                # noqa: E402
from paths import ID_MAP, OUTPUTS                               # noqa: E402
from geometry import fit_plane_ransac                           # noqa: E402
from shape_prior import interior_solid_volume, container_volume # noqa: E402

GT = {int(r["id"]): float(r["gt_ml"])
      for r in csv.DictReader(open(OUTPUTS / "_HELD_OUT_gt_volumes.csv"))}


def rim_profile(m, nbin=14):
    """용기 점군에서 높이별 최대 반지름 r(h) 와 테두리 높이 h_rim 을 잰다."""
    C = m.get("C_pts")
    if C is None or len(C) < 400:
        return None
    pl, u, v = m["plane"], m["u"], m["v"]
    h = pl.height(C)
    cu, cv = m["cont"]["center"]
    r = np.hypot(C @ u - cu, C @ v - cv)
    ok = np.isfinite(h) & np.isfinite(r)
    h, r = h[ok], r[ok]
    if len(h) < 400:
        return None
    # 이상치 제거: 높이 상하위 1%
    lo, hi = np.percentile(h, [1, 99])
    sel = (h >= lo) & (h <= hi)
    h, r = h[sel], r[sel]
    h_rim = float(np.percentile(h, 95))
    edges = np.linspace(0.0, max(h_rim, 1.0), nbin + 1)
    prof = []
    for i in range(nbin):
        s = (h >= edges[i]) & (h < edges[i + 1])
        if s.sum() >= 25:
            prof.append((float(0.5 * (edges[i] + edges[i + 1])),
                         float(np.percentile(r[s], 95)), int(s.sum())))
    return {"h_rim": h_rim, "prof": prof, "n": len(h)}


def interior_measured(prof, R_fill, h_fill, h_rim):
    """측정한 r(h) 로 충전선 아래 안쪽 고체를 적분한다.

    용기 점군은 **바깥벽**이라 안쪽보다 조금 크다. 그래서 절대 반지름을 그대로
    쓰지 않고 **모양(테이퍼)만** 가져온다: 충전선 높이에서 우리가 직접 잰
    R_fill 에 맞도록 프로파일을 정규화한다. 절대 크기는 측정, 테이퍼도 측정.
    """
    if not prof or h_fill <= 0:
        return None, None
    hs = np.array([p[0] for p in prof]); rs = np.array([p[1] for p in prof])
    if len(hs) < 3:
        return None, None
    r_at_fill = float(np.interp(min(h_fill, hs[-1]), hs, rs))
    if r_at_fill <= 1e-6:
        return None, None
    g = R_fill / r_at_fill                      # 바깥벽 -> 충전선 등가반지름 보정
    zz = np.linspace(0.0, h_fill, 200)
    rr = np.interp(zz, hs, rs, left=rs[0], right=rs[-1]) * g
    V = float(np.trapezoid(np.pi * rr * rr, zz))
    taper = float(rr[0] / rr[-1]) if rr[-1] > 0 else 1.0
    return V, taper


def main(backend="depthpro", expand=5.0):
    be = build(backend)
    shapes = load_shapes()
    tbl = [(int(r["combo"]), int(r["id"]), r["food"], r["container"] or "")
           for r in csv.DictReader(open(ID_MAP))]
    rows = []
    print(f"{'id':>3} {'음식':<17} {'h_rim':>6} {'R_rim':>6} {'h_fill':>7} {'R':>6} "
          f"{'H_vlm':>6} {'H_meas':>7} {'prof':<9} {'V_int':>7} {'V_qwen':>7} {'V_H':>7} "
          f"{'V_cap':>7} {'h_ext':>6} {'free':>6} {'H_free':>7} {'V_free':>7} {'onP':>5} {'V_best':>7} {'GT':>7}")
    for cidx, fid, food, gt_cont in tbl:
        if not gt_cont:
            continue
        m = measure(be, cidx, fid, expand=expand)
        if m is None or not m.get("cont"):
            print(f"{fid:>3} {food:<17} 용기 점군 없음")
            continue
        s = shapes.get((cidx, fid), {})
        rp = rim_profile(m)
        # 충전선 평면: 음식 발자국 바깥 15% 링에 평면 적합 (apply_shape 와 동일)
        pl, u, v = m["plane"], m["u"], m["v"]
        F = m["F"]
        cu = np.array([(F @ u).mean(), (F @ v).mean()])
        rad = np.hypot(F @ u - cu[0], F @ v - cu[1])
        edge = rad >= np.percentile(rad, 85)
        fill = fit_plane_ransac(F[edge], iters=300, thresh_mm=3.0) if edge.sum() >= 60 else pl
        h_fill_plane = float(np.median(pl.height(F[edge])))   # 접시 기준 충전선 높이
        h_f = fill.height(F)
        A_mm2 = m["A_cm2"] * 100.0
        R = float(np.sqrt(A_mm2 / np.pi))
        dr = float(s.get("depth_ratio", 0.0))
        H_vlm = dr * 2.0 * R
        v_qwen = container_volume(F @ u, F @ v, h_f, dr, s.get("profile")) if dr > 0.05 else None
        mound = float(np.clip(h_f, 0, None).sum() * 0.0)  # 아래에서 격자로 다시
        # 봉우리는 container_volume 과 같은 격자 방식으로
        from shape_prior import container_volume as _cv
        v_mound_only = _cv(F @ u, F @ v, h_f, 1e-9, "cylindrical")
        h_rim = rp["h_rim"] if rp else float("nan")
        Vin, taper = interior_measured(rp["prof"], R, h_fill_plane, h_rim) if rp else (None, None)
        v_meas = (v_mound_only + Vin * 1e-3) if (Vin is not None and v_mound_only) else None
        # --- 물어보지 않고 잰 깊이 ---------------------------------------
        # 그릇 바닥은 테두리 아래 h_rim 만큼 내려간 곳. 굽/두께를 10% 로 두면
        # 음식이 잠긴 깊이는 H_meas = h_fill - 0.10*h_rim 이다. 전부 측정값이다.
        H_meas = h_fill_plane - 0.10 * h_rim if rp else float("nan")
        R_rim = m["cont"]["R_mm"]
        prof_q = (s.get("profile") or "hemispherical")
        v_H = None
        if np.isfinite(H_meas) and H_meas > 2.0 and v_mound_only:
            v_H = v_mound_only + interior_solid_volume(R, H_meas, prof_q) * 1e-3
        # --- 기준면을 아예 안 쓰는 측정 -----------------------------------
        # 으깬감자 컵은 접시 위가 아니라 식탁에 따로 놓여 있다(rim_masks.png).
        # 접시 평면을 기준으로 삼으면 그 항목만 무너진다. 그런데 우리가 필요한 두 양은
        # 둘 다 **높이의 차이**라 기준면 오프셋이 상쇄된다.
        #   h_ext     = 용기 자체의 높이 (점군 높이 p95 - p5)
        #   freeboard = 테두리에서 음식 표면까지 남은 빈 높이
        #   H_free    = 0.90 * h_ext - freeboard      (굽/바닥두께 10%)
        h_ext = free = H_free = float("nan")
        if rp is not None:
            hC = pl.height(m["C_pts"])
            hC = hC[np.isfinite(hC)]
            h_ext = float(np.percentile(hC, 95) - np.percentile(hC, 5))
            free = float(np.percentile(hC, 95) - h_fill_plane)
            H_free = 0.90 * h_ext - free
        v_free = None
        if np.isfinite(H_free) and H_free > 2.0 and v_mound_only:
            v_free = v_mound_only + interior_solid_volume(R, H_free, "hemispherical") * 1e-3
        # --- 두 기준을 마스크로 갈라 쓴다 ---------------------------------
        onp = m["cont"].get("on_plate", 0.0)
        H_best = H_meas if onp >= 0.5 else H_free
        v_best = None
        if np.isfinite(H_best) and H_best > 2.0 and v_mound_only:
            v_best = v_mound_only + interior_solid_volume(R, H_best, "hemispherical") * 1e-3
        v_cyl = v_cap = None
        if np.isfinite(H_meas) and H_meas > 2.0 and v_mound_only:
            v_cyl = v_mound_only + interior_solid_volume(R, H_meas, "cylindrical") * 1e-3
            v_cap = v_mound_only + interior_solid_volume(R, H_meas, "hemispherical") * 1e-3
        gt = GT.get(fid, float("nan"))
        print(f"{fid:>3} {food:<17} {h_rim:>6.1f} {R_rim:>6.1f} {h_fill_plane:>7.1f} {R:>6.1f} "
              f"{H_vlm:>6.1f} {H_meas:>7.1f} {str(prof_q)[:9]:<9} "
              f"{m['v_int']:>7.1f} {(v_qwen or 0):>7.1f} {(v_H or 0):>7.1f} "
              f"{(v_cap or 0):>7.1f} {h_ext:>6.1f} {free:>6.1f} {H_free:>7.1f} "
              f"{(v_free or 0):>7.1f} {onp:>5.2f} {(v_best or 0):>7.1f} {gt:>7.1f}")
        rows.append({"combo": cidx, "id": fid, "food": food, "h_rim": h_rim,
                     "h_fill": h_fill_plane, "R": R, "H_vlm": H_vlm,
                     "profile": s.get("profile"), "taper": taper,
                     "h_rim2": h_rim, "R_rim": R_rim, "H_meas": H_meas,
                     "v_int": m["v_int"], "v_qwen": v_qwen, "v_meas": v_meas,
                     "v_H": v_H, "v_cyl": v_cyl, "v_cap": v_cap,
                     "h_ext": h_ext, "freeboard": free, "H_free": H_free,
                     "v_free": v_free, "on_plate": onp, "v_best": v_best, "gt": gt})
    if rows:
        p = OUTPUTS / "rim_probe.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        for key in ("v_int", "v_qwen", "v_meas", "v_H", "v_cyl", "v_cap", "v_free", "v_best"):
            v = [abs(r[key] - r["gt"]) / r["gt"] for r in rows if r.get(key)]
            if v:
                print(f"  {key:<7} MAPE {np.mean(v):.3f}  n={len(v)}")
        print(f"  -> {p}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="depthpro")
    ap.add_argument("--expand", type=float, default=5.0)
    a = ap.parse_args()
    main(a.backend, a.expand)
