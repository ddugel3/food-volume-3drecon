"""깊이 단계의 계약과 백엔드들.

계약: 이미지 + (알고 있다면) 수평 화각 -> 카메라 좌표계 포인트맵(mm) + 유효 마스크.

왜 '깊이'가 아니라 '포인트맵'을 계약으로 잡는가:
어차피 역투영해서 3D 로 쓸 것이고, 모델마다 내부적으로 가정한 초점거리가 다르면
우리가 밖에서 역투영할 때 어긋난다. 포인트맵으로 통일하면 그 불일치가 사라진다.
MoGe 는 원래 포인트맵을 내고, DepthPro / DepthAnything 은 깊이를 내므로
**EXIF 초점거리로 우리가 역투영**한다 (모델 자체 추정 초점거리는 기록만 해둔다).

metric 플래그의 의미: 모델이 절대 미터 스케일을 주장하는가. 우리는 어차피 절대
스케일을 접시로 다시 잡으므로 이 플래그는 진단용이다 -- 측정해보니 MoGe 의
절대 스케일은 실제 대비 2.3배 틀렸다(부피 11.7배).
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import unproject   # noqa: E402


@dataclass
class DepthResult:
    points_mm: np.ndarray      # (H, W, 3) 카메라 좌표, mm
    valid: np.ndarray          # (H, W) bool
    metric: bool               # 모델이 절대 스케일을 주장하는가
    focal_px_pred: float | None  # 모델이 스스로 추정한 초점거리 (있으면)
    backend: str


class MoGeBackend:
    """Ruicheng/moge-{2,3}. 포인트맵을 직접 낸다. fov_x 주입 가능."""

    def __init__(self, ver="v3", device="cuda"):
        self.ver, self.device = ver, device
        self.name = f"moge_{ver}"
        mod = __import__(f"moge.model.{ver}", fromlist=["MoGeModel"])
        ckpt = {"v2": "Ruicheng/moge-2-vitl-normal", "v3": "Ruicheng/moge-3-vitl"}[ver]
        self.model = mod.MoGeModel.from_pretrained(ckpt).to(device).eval()

    @torch.no_grad()
    def infer(self, img: Image.Image, fov_x_deg=None) -> DepthResult:
        t = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div_(255).to(self.device)
        o = self.model.infer(t, fov_x=fov_x_deg)
        pts = o["points"].float().cpu().numpy() * 1000.0
        valid = o["mask"].cpu().numpy() if "mask" in o else np.isfinite(pts).all(-1)
        K = o["intrinsics"].float().cpu().numpy()
        return DepthResult(pts, valid, True, float(K[0, 0]) * img.width, self.name)


class DepthProBackend:
    """apple/DepthPro-hf. 1536^2 다중스케일 패치 피라미드 -- 미세 기복에 강하다는 것이 주장.

    깊이를 내므로 EXIF 초점거리로 우리가 역투영한다.
    """

    def __init__(self, device="cuda", dtype=torch.float16):
        from transformers import DepthProForDepthEstimation, DepthProImageProcessor
        self.name, self.device = "depth_pro", device
        self.proc = DepthProImageProcessor.from_pretrained("apple/DepthPro-hf")
        self.model = DepthProForDepthEstimation.from_pretrained(
            "apple/DepthPro-hf", dtype=dtype).to(device).eval()

    @torch.no_grad()
    def infer(self, img: Image.Image, fov_x_deg=None) -> DepthResult:
        w, h = img.size
        inp = self.proc(images=img, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        post = self.proc.post_process_depth_estimation(out, target_sizes=[(h, w)])[0]
        depth_m = post["predicted_depth"].float().cpu().numpy()
        f_pred = post.get("focal_length")
        f_pred = float(np.asarray(f_pred.cpu() if hasattr(f_pred, "cpu") else f_pred).ravel()[0]) \
            if f_pred is not None else None
        # EXIF 화각으로 우리가 역투영한다. 모델 추정 초점거리는 진단용으로만 남긴다.
        f = (w / 2) / np.tan(np.radians(fov_x_deg / 2)) if fov_x_deg else (f_pred or w)
        K = ((f, 0, w / 2), (0, f, h / 2), (0, 0, 1))
        d = depth_m * 1000.0
        valid = np.isfinite(d) & (d > 0)
        pts = unproject(np.where(valid, d, np.nan), K)
        return DepthResult(pts, valid, True, f_pred, self.name)


class DepthAnythingV2MetricBackend:
    """depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf. 실내 메트릭 변형."""

    def __init__(self, device="cuda"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        rid = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
        self.name, self.device = "depth_anything_v2_metric", device
        self.proc = AutoImageProcessor.from_pretrained(rid)
        self.model = AutoModelForDepthEstimation.from_pretrained(rid).to(device).eval()

    @torch.no_grad()
    def infer(self, img: Image.Image, fov_x_deg=None) -> DepthResult:
        w, h = img.size
        inp = self.proc(images=img, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        post = self.proc.post_process_depth_estimation(out, target_sizes=[(h, w)])[0]
        d = post["predicted_depth"].float().cpu().numpy() * 1000.0
        f = (w / 2) / np.tan(np.radians(fov_x_deg / 2)) if fov_x_deg else w
        K = ((f, 0, w / 2), (0, f, h / 2), (0, 0, 1))
        valid = np.isfinite(d) & (d > 0)
        return DepthResult(unproject(np.where(valid, d, np.nan), K), valid, True, None, self.name)


_REGISTRY = {
    "moge_v3": lambda: MoGeBackend("v3"),
    "moge_v2": lambda: MoGeBackend("v2"),
    "depth_pro": DepthProBackend,
    "depth_anything_v2_metric": DepthAnythingV2MetricBackend,
}


def get_backend(name, cache={}):
    if name not in cache:
        if name not in _REGISTRY:
            raise KeyError(f"모르는 백엔드 {name}. 가능: {list(_REGISTRY)}")
        cache[name] = _REGISTRY[name]()
    return cache[name]


def available():
    return list(_REGISTRY)
