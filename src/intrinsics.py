"""EXIF 에서 카메라 내부 파라미터를 복원한다.

벤치마크 논문과 대회 설명은 "명시적 물리 참조물도, 카메라 내부 파라미터도 주어지지
않는다"고 전제하지만, 배포된 14장의 JPEG 에는 EXIF 가 그대로 살아 있다.
전부 iPhone 13 Pro 메인 광각(5.7mm, f/1.5, 35mm 환산 26mm, 4032x3024)이다.

이걸 쓰는 이유는 단순히 편해서가 아니다. 단안 부피 추정에서 오차의 지배항은
길이 스케일이고(V = s^3 V_u, 길이 10% -> 부피 33%), 그 스케일의 첫 단계가
픽셀 -> 각도 변환, 즉 초점거리다. 초점거리를 깊이 모델에 "추정시키는" 대신
"알려주면" 오차원 하나가 통째로 사라진다. Metric3D v2 처럼 내부 파라미터를
canonical camera transformation 으로 소비하는 모델에서는 이게 그대로 성능이 된다.

두 가지 방식으로 f_px 를 구해 교차검증한다.

  (A) 센서 물리 크기 기준
      iPhone 13 Pro 메인 센서는 1.9um 픽셀의 12MP(4032x3024) 이므로
      센서폭 = 4032 * 1.9um = 7.661mm
      f_px  = f_mm / 센서폭 * width

  (B) EXIF 의 FocalLengthIn35mmFilm(=26mm) 기준
      크롭팩터 = 26 / 5.7 = 4.561
      35mm 환산의 표준 관례는 '대각' 기준이므로 대각으로 환산한다.
      센서대각 = 43.267 / 크롭팩터,  f_px = f_mm / 센서대각 * 이미지대각

둘의 차이가 곧 우리가 감수하는 불확실도다. 실측상 약 1% 이내이고,
이는 부피로 약 3% 에 해당한다. 이 값을 sigma_log_f 로 남겨
스케일 융합(D6)과 MAPE 최적 수축(F3)에 그대로 넘긴다.
"""
from dataclasses import dataclass
from math import atan, degrees, hypot
from pathlib import Path

from PIL import ExifTags, Image

# iPhone 13 Pro 메인 광각 센서. 픽셀 피치 1.9um (Apple 공개 사양).
IPHONE_13_PRO_PIXEL_PITCH_UM = 1.9
FULL_FRAME_DIAGONAL_MM = hypot(36.0, 24.0)  # 43.267


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    source: str
    sigma_log_f: float  # 추정 방식 간 불일치에서 얻은 로그 스케일 불확실도

    @property
    def K(self):
        return ((self.fx, 0.0, self.cx), (0.0, self.fy, self.cy), (0.0, 0.0, 1.0))

    @property
    def hfov_deg(self):
        return 2 * degrees(atan(self.width / (2 * self.fx)))

    @property
    def vfov_deg(self):
        return 2 * degrees(atan(self.height / (2 * self.fy)))


def _exif(path: Path) -> dict:
    with Image.open(path) as im:
        ex = im.getexif()
        merged = dict(ex)
        try:
            merged.update(dict(ex.get_ifd(0x8769)))  # ExifIFD
        except Exception:
            pass
        size = (im.width, im.height)
    out = {ExifTags.TAGS.get(k, str(k)): v for k, v in merged.items()}
    out["_size"] = size
    return out


def from_exif(path: Path) -> Intrinsics:
    ex = _exif(path)
    w, h = ex["_size"]
    f_mm = float(ex["FocalLength"])

    # (A) 센서 물리 크기 기준
    sensor_w_mm = w * IPHONE_13_PRO_PIXEL_PITCH_UM / 1000.0
    f_px_sensor = f_mm / sensor_w_mm * w

    # (B) 35mm 환산 기준 (대각)
    f35 = ex.get("FocalLengthIn35mmFilm")
    if f35:
        crop = float(f35) / f_mm
        sensor_diag_mm = FULL_FRAME_DIAGONAL_MM / crop
        f_px_35 = f_mm / sensor_diag_mm * hypot(w, h)
    else:
        f_px_35 = f_px_sensor

    # 두 추정의 기하평균을 쓴다. 스케일은 곱셈적이라 로그공간 평균이 맞다.
    f_px = (f_px_sensor * f_px_35) ** 0.5
    # 불일치의 절반을 1시그마로 본다 (두 추정이 대략 +-1sigma 라고 가정).
    sigma_log_f = abs(0.5 * (f_px_sensor / f_px_35 - 1.0))

    model = str(ex.get("Model", "?"))
    if "iPhone 13 Pro" not in model:
        # 센서 피치 가정이 깨지므로 35mm 환산만 믿는다.
        f_px, sigma_log_f = f_px_35, 0.03

    # 주점은 중심으로 둔다. EXIF 에 주점 정보는 없다.
    return Intrinsics(
        width=w, height=h, fx=f_px, fy=f_px, cx=w / 2.0, cy=h / 2.0,
        source=f"EXIF {model} {f_mm}mm (sensor {f_px_sensor:.1f}px / 35mm {f_px_35:.1f}px)",
        sigma_log_f=sigma_log_f,
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paths import combo_images

    print(f"{'combo':>6} {'해상도':>12} {'f_px':>9} {'HFOV':>7} {'VFOV':>7} {'sigma_log_f':>12}")
    fs = []
    for idx, p in sorted(combo_images().items()):
        k = from_exif(p)
        fs.append(k.fx)
        print(f"{idx:>6} {k.width:>5}x{k.height:<6} {k.fx:>9.1f} "
              f"{k.hfov_deg:>6.2f}° {k.vfov_deg:>6.2f}° {k.sigma_log_f:>12.4f}")
    print(f"\n14장 f_px: 최소 {min(fs):.1f} / 최대 {max(fs):.1f} "
          f"(전부 같은 렌즈면 동일해야 한다)")
    k = from_exif(sorted(combo_images().values())[0])
    print(f"출처: {k.source}")
    print(f"\n부피로 환산한 초점거리 불확실도: 약 {3 * k.sigma_log_f * 100:.1f}%  (V ∝ s^3)")
