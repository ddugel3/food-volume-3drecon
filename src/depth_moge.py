"""MoGe (v2 / v3) 로 14장의 메트릭 포인트맵을 뽑는다.

MoGe 를 첫 깊이 모델로 고른 이유:
  - 깊이가 아니라 **메트릭 포인트맵**을 바로 준다. 우리는 어차피 역투영해서
    포인트클라우드로 만들 것이므로 중간 단계가 하나 줄고, 모델이 내부적으로
    쓴 내부 파라미터와 우리 역투영이 어긋날 여지가 없다.
  - `fov_x` 를 인자로 받는다. EXIF 에서 얻은 수평 화각(67.55도)을 그대로 넣어
    "초점거리를 추정시키지 않고 알려주는" 경로를 만들 수 있다.
  - FOV 를 주지 않으면 스스로 추정한다. 그 값을 EXIF 와 비교하면 이미지별로
    모델을 얼마나 믿을지 판단하는 **무료 진단 지표**가 생긴다.

두 모델 x 두 조건(EXIF FOV 주입 / 자체 추정) = 이미지당 4회 추론.
이미지가 14장뿐이라 전부 돌려도 비용이 무의미하다. 불일치는 버리는 게 아니라
스케일 융합(D6)과 MAPE 최적 수축(F3)에 넣을 sigma 의 재료다.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intrinsics import from_exif          # noqa: E402
from paths import CACHE, combo_images     # noqa: E402

CHECKPOINTS = {"v2": "Ruicheng/moge-2-vitl-normal", "v3": "Ruicheng/moge-3-vitl"}


def load_model(ver: str, device="cuda"):
    mod = __import__(f"moge.model.{ver}", fromlist=["MoGeModel"])
    model = mod.MoGeModel.from_pretrained(CHECKPOINTS[ver]).to(device).eval()
    return model


def load_image(path: Path, long_side: int):
    """긴 변을 long_side 로 줄인다. 리사이즈는 화각을 바꾸지 않으므로 fov_x 는 그대로 유효하다."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = long_side / max(w, h)
        if s < 1.0:
            im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
        return np.asarray(im)


@torch.no_grad()
def run(ver: str, long_side: int = 2016, device="cuda"):
    model = load_model(ver, device)
    CACHE.mkdir(exist_ok=True)
    rows = []

    for idx, path in sorted(combo_images().items()):
        K = from_exif(path)
        rgb = load_image(path, long_side)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255).to(device)

        out = {}
        for tag, fov in (("exif", K.hfov_deg), ("self", None)):
            t0 = time.time()
            o = model.infer(t, fov_x=fov)
            dt = time.time() - t0
            pts = o["points"].float().cpu().numpy()          # (H, W, 3), 미터
            msk = o["mask"].cpu().numpy() if "mask" in o else np.isfinite(pts).all(-1)
            Ki = o["intrinsics"].float().cpu().numpy()        # 정규화 내부 파라미터
            out[f"points_{tag}"] = (pts * 1000.0).astype(np.float32)   # mm
            out[f"mask_{tag}"] = msk
            out[f"K_{tag}"] = Ki
            # MoGe 의 정규화 내부 파라미터는 fx 가 이미지 폭으로 나뉜 값이다.
            fov_pred = 2 * np.degrees(np.arctan(0.5 / float(Ki[0, 0])))
            out[f"fov_{tag}"] = fov_pred
            if tag == "self":
                rows.append((idx, K.hfov_deg, fov_pred, dt))

        np.savez_compressed(CACHE / f"moge{ver}_combo_{idx:02d}.npz",
                            exif_fov=K.hfov_deg, exif_fx_px=K.fx,
                            rgb_shape=np.array(rgb.shape), **out)
        print(f"  combo {idx:2d}  EXIF FOV {K.hfov_deg:6.2f}°  "
              f"MoGe{ver} 자체추정 {out['fov_self']:6.2f}°  "
              f"차이 {out['fov_self'] - K.hfov_deg:+6.2f}°  ({rows[-1][3]:.1f}s)")

    e = np.array([r[1] for r in rows]); m = np.array([r[2] for r in rows])
    # 화각 차이를 초점거리 로그 비율로 환산한다. f ∝ 1/tan(fov/2).
    lr = np.log(np.tan(np.radians(e / 2)) / np.tan(np.radians(m / 2)))
    print(f"\n  MoGe{ver} 자체추정 FOV vs EXIF")
    print(f"    평균 차이 {np.mean(m - e):+.2f}°   |차이| 중앙값 {np.median(np.abs(m - e)):.2f}°")
    print(f"    초점거리 로그비 평균 {lr.mean():+.4f}  표준편차 {lr.std():.4f}")
    print(f"    -> 부피로는 평균 {100*(np.exp(3*lr.mean())-1):+.1f}% 편향, "
          f"산포 {100*(np.exp(3*lr.std())-1):.1f}%")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", default="v3", choices=["v2", "v3"])
    ap.add_argument("--long-side", type=int, default=2016)
    a = ap.parse_args()
    print(f"=== MoGe {a.ver} ({CHECKPOINTS[a.ver]}), 긴 변 {a.long_side}px ===")
    run(a.ver, a.long_side)
