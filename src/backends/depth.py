"""깊이 백엔드 공통 인터페이스.

설계 원칙 하나: **모든 백엔드는 같은 카메라를 쓴다.**

EXIF 로 f_px 를 ±0.5% 로 알고 있으므로, 깊이만 내놓는 모델(Depth Pro, Depth Anything)도
우리가 직접 역투영해서 포인트맵으로 바꾼다. 모델이 제 나름의 초점거리를 추정해 쓰게
두면 비교가 오염된다 -- 어떤 모델이 좋아 보이는 게 깊이가 좋아서인지 초점거리를 잘
찍어서인지 구분할 수 없게 된다. 카메라를 고정해야 깊이만 비교된다.

포인트맵의 단위는 백엔드마다 다르다(미터, 임의 단위). 하류에서 쓰는 양은
크롭 안의 **스케일 무관 형상**(높이/가로 종횡비)이므로 단위는 문제되지 않는다.
다만 종횡비는 예측된 '절대 깊이'에 의존한다 -- 가로 = (u-cx)/f * Z 이고 기복은 dZ 라서,
Z 를 k 배 크게 찍으면 가로만 k 배가 되고 종횡비가 1/k 로 눌린다.
그래서 종횡비 일치도가 곧 절대 깊이 품질의 대리 지표가 된다.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def unproject(depth: np.ndarray, fov_x_deg: float) -> np.ndarray:
    """깊이맵 -> 포인트맵. 주점은 중심, 화각은 주어진 값."""
    h, w = depth.shape
    f = w / (2.0 * np.tan(np.radians(fov_x_deg) / 2.0))
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    z = depth.astype(np.float64)
    return np.stack([(us - w / 2) / f * z, (vs - h / 2) / f * z, z], axis=-1)


class DepthBackend:
    name = "base"
    unit = "?"

    def points(self, img: Image.Image, fov_x_deg: float):
        """-> (points HxWx3, valid HxW). 실패 시 valid 가 전부 False."""
        raise NotImplementedError


class MoGeBackend(DepthBackend):
    """포인트맵을 직접 예측한다. fov_x 를 인자로 받으므로 카메라 고정이 자연스럽다."""

    def __init__(self, ver="v3", device="cuda"):
        mod = __import__(f"moge.model.{ver}", fromlist=["MoGeModel"])
        ckpt = {"v2": "Ruicheng/moge-2-vitl-normal", "v3": "Ruicheng/moge-3-vitl"}[ver]
        self.m = mod.MoGeModel.from_pretrained(ckpt).to(device).eval()
        self.name, self.unit, self.device = f"moge_{ver}", "m", device

    @torch.no_grad()
    def points(self, img, fov_x_deg):
        t = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div_(255).to(self.device)
        o = self.m.infer(t, fov_x=float(fov_x_deg))
        p = o["points"].float().cpu().numpy()
        v = o["mask"].cpu().numpy() if "mask" in o else np.isfinite(p).all(-1)
        return p, v & np.isfinite(p).all(-1)


class HFDepthBackend(DepthBackend):
    """transformers 의 깊이 추정 모델 공통 래퍼 (Depth Pro, Depth Anything 등)."""

    def __init__(self, repo, name, device="cuda", dtype=torch.float32):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.proc = AutoImageProcessor.from_pretrained(repo)
        self.m = AutoModelForDepthEstimation.from_pretrained(repo, dtype=dtype).to(device).eval()
        self.name, self.unit, self.device = name, "m", device

    @torch.no_grad()
    def points(self, img, fov_x_deg):
        inp = self.proc(images=img, return_tensors="pt").to(self.device)
        out = self.m(**inp)
        post = self.proc.post_process_depth_estimation(
            out, target_sizes=[(img.height, img.width)])[0]
        d = post["predicted_depth"].float().cpu().numpy()
        # Depth Pro 는 자체 추정 초점거리로 미터 깊이를 만든다. 우리 초점거리로 환산한다.
        # 주의: field_of_view 는 **도(degree)** 단위다. 라디안으로 착각하고
        # tan(fov/2) 를 계산하면 tan(30 rad) 처럼 엉뚱한 값이 나오고, 부호가 뒤집히면
        # 깊이 전체가 음수가 되어 유효점이 0 개가 된다(35건 중 16건이 이렇게 죽었다).
        # focal_length 를 직접 주므로 화각을 거칠 이유도 없다.
        f_theirs = post.get("focal_length")
        if f_theirs is not None:
            f_theirs = float(f_theirs)
        else:
            fov_deg = post.get("field_of_view")
            if fov_deg is not None:
                f_theirs = float(img.width / (2 * np.tan(np.radians(float(fov_deg)) / 2)))
        if f_theirs and f_theirs > 0:
            f_ours = img.width / (2 * np.tan(np.radians(fov_x_deg) / 2))
            d = d * (f_ours / f_theirs)
        valid = np.isfinite(d) & (d > 1e-6)
        d = np.where(valid, d, np.nan)
        p = unproject(d, fov_x_deg)
        return p, valid


REGISTRY = {
    "moge_v3":      lambda: MoGeBackend("v3"),
    "moge_v2":      lambda: MoGeBackend("v2"),
    "depthpro":     lambda: HFDepthBackend("apple/DepthPro-hf", "depthpro"),
    "da_v2_metric": lambda: HFDepthBackend(
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", "da_v2_metric"),
    "da_v2_rel":    lambda: HFDepthBackend(
        "depth-anything/Depth-Anything-V2-Large-hf", "da_v2_rel"),
}


def build(name: str) -> DepthBackend:
    if name not in REGISTRY:
        raise KeyError(f"모르는 백엔드 {name}. 가능: {sorted(REGISTRY)}")
    return REGISTRY[name]()
