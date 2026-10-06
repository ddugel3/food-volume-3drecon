"""형상 계열별 충전율 k 와 용기 안쪽 부피 모델.

우리가 재는 값은 V_int = ∫ h dA 다 -- 지지평면 위로의 보이는 윗면을 적분한 것.
진짜 부피와의 비 k = V_true / V_int 는 **형상 계열마다 해석적으로 정해진다.**
크기와 무관한 순수 기하 상수이므로 손으로 맞춘 값이 아니다.

  slab      두께가 일정한 판. V_int = A·t = V_true                     -> k = 1.00
  wedge     삼각기둥/쐐기. 위에서 본 적분이 곧 부피                      -> k = 1.00
  dome      z = sqrt(r²-ρ²). ∫ = (2/3)πr³ = 반구 부피                  -> k = 1.00
  sphere    평면에 놓인 구. z = r + sqrt(r²-ρ²) 를 반지름 r 원판에 적분하면
            πr³ + (2/3)πr³ = (5/3)πr³ 인데 진짜는 (4/3)πr³            -> k = 0.80
            (처마 아래를 꽉 찬 것으로 세기 때문이다)
  cylinder  옆으로 누운 원통. ∫ = L(2r² + πr²/2), 진짜는 πr²L
            k = π/(2+π/2)                                            -> k = 0.88
  torus     도넛. 구멍을 마스크가 제외한다고 보면 k = 2π/(4+π)          -> k = 0.88
            (Hunyuan3D 메시로 잰 베이글 값 0.88 과 정확히 일치했다)
  irregular 다공성/울퉁불퉁(브로콜리 송이, 치킨윙). 표면 사이 빈틈이 있으므로
            위 값들보다 낮다. 해석해가 없어 0.75 로 둔다 -- 이 하나만 경험값이다.

리더보드에서 균일 수축(값을 10~20% 낮추는 조작)이 public 을 0.25 -> 0.20 으로
개선했는데, 그 편향의 정체가 바로 이 k 다. 균일 수축 대신 형상별 k 를 쓰면
같은 이득을 물리적 근거로 얻고, 물체마다 다르게 적용되니 더 정확하다.
"""
import numpy as np

K = {
    "slab": 1.00,
    "wedge": 1.00,
    "dome": 1.00,
    "sphere": 0.80,
    "cylinder": 0.88,
    "torus": 0.88,
    "irregular": 0.75,
}
K_DEFAULT = 0.90     # 분류 실패 시. 위 값들의 중앙 부근.


def fill_factor(shape):
    return K.get((shape or "").strip().lower(), K_DEFAULT)


# ---------------------------------------------------------------- 용기 안쪽
def interior_depth_profile(profile, R, H):
    """테두리 반지름 R, 안쪽 깊이 H 인 용기의 안쪽 면 z(rho). 테두리 평면이 z=0, 아래가 음수.

    hemispherical  구면 캡. 테두리에서 z=0, 중심에서 z=-H 가 되도록 구 반지름을 잡는다.
                   a = (R² + H²) / (2H),  z(ρ) = -H + a - sqrt(a² - ρ²)
    conical        원뿔. z(ρ) = -H (1 - ρ/R)
    cylindrical    평바닥. z(ρ) = -H
    """
    profile = (profile or "").strip().lower()
    if profile == "cylindrical":
        return lambda rho: np.full_like(np.asarray(rho, float), -H)
    if profile == "conical":
        return lambda rho: -H * (1.0 - np.clip(np.asarray(rho, float) / max(R, 1e-9), 0, 1))
    a = (R * R + H * H) / (2.0 * max(H, 1e-9))          # hemispherical (기본)
    return lambda rho: -H + a - np.sqrt(np.maximum(a * a - np.asarray(rho, float) ** 2, 0.0))


def interior_solid_volume(R, H, profile="hemispherical"):
    """테두리 반지름 R, 깊이 H 인 용기 안쪽 고체의 부피 (mm^3). 해석해.

      cylindrical    pi R^2 H
      conical        (1/3) pi R^2 H                (바닥 중심이 꼭짓점)
      hemispherical  구면 캡. 구 반지름 a = (R^2+H^2)/(2H) 일 때
                     V = pi H (3R^2 + H^2) / 6
    """
    profile = (profile or "").strip().lower()
    if profile == "cylindrical":
        return np.pi * R * R * H
    if profile == "conical":
        return np.pi * R * R * H / 3.0
    return np.pi * H * (3.0 * R * R + H * H) / 6.0


def container_volume(food_a, food_b, food_h_fill, depth_ratio,
                     profile="hemispherical", grid_mm=1.0):
    """용기에 담긴 음식의 부피 (mL). 충전선(fill line) 기준으로 푼다.

    이전 구현은 두 군데가 틀렸다.
      (1) 높이를 **접시 평면** 기준으로 재면서 용기 모델은 **테두리 평면** 기준이라고
          가정했다. 그릇 높이만큼 통째로 어긋나 파스타 두께가 218 mm 로 나왔다.
      (2) 반지름을 용기 마스크 볼록껍질의 절반으로 잡았는데 그건 그릇 **바깥** 지름이다.
          파스타 볼이 지름 297 mm 로 나왔다(실제 시리얼 볼은 150~180 mm).

    바로잡은 관점: 음식 표면의 **경계**가 곧 충전선이고, 그 반지름이 곧 그 높이에서의
    용기 안쪽 반지름이다. 둘 다 우리가 직접 잰다. Qwen 은 무차원 깊이비만 준다.

      V = (충전선 위로 솟은 봉우리)  +  (충전선 아래 용기 안쪽 고체)
          = ∫ max(h,0) dA           +  interior_solid_volume(R, H)
      R = 음식 발자국의 등가 반지름,  H = depth_ratio * 2R

    food_h_fill 은 **충전선 평면 기준** 높이여야 한다(호출부에서 그 평면을 맞춘다).
    """
    a = np.asarray(food_a, float); b = np.asarray(food_b, float)
    h = np.asarray(food_h_fill, float)
    ia = np.floor((a - a.min()) / grid_mm).astype(int)
    ib = np.floor((b - b.min()) / grid_mm).astype(int)
    na, nb = ia.max() + 1, ib.max() + 1
    if na * nb > 20_000_000:
        return None
    top = np.full((na, nb), -np.inf)
    np.maximum.at(top, (ia, ib), h)
    occ = np.isfinite(top)
    if occ.sum() < 20:
        return None
    A_mm2 = float(occ.sum()) * grid_mm * grid_mm
    mound = float(np.clip(np.where(occ, top, 0.0), 0.0, None).sum()) * grid_mm * grid_mm
    R = float(np.sqrt(A_mm2 / np.pi))              # 발자국의 등가 반지름
    H = float(depth_ratio) * 2.0 * R
    if H <= 0:
        return None
    return (mound + interior_solid_volume(R, H, profile)) * 1e-3


def measured_depth(h_container, h_fill_plate, on_plate, base_frac=0.10):
    """용기 안쪽 깊이를 **물어보지 않고 잰다**.

    지금까지는 H = depth_ratio * 2R 로, depth_ratio 를 Qwen 이 줬다. 그런데 그릇
    테두리는 사진에 보이고 깊이맵에 이미 용기 점군이 잡혀 있었다(반지름만 쓰고
    높이는 버리고 있었다). 두 경우로 갈린다.

      그릇이 접시 위에 있으면   접시 평면이 옳은 기준이다.
                              H = h_fill - base_frac * h_rim
      접시 위가 아니면          접시 평면은 틀린 기준이다(으깬감자 컵은 식탁에 따로
                              놓여 있어 H 가 0.7 mm 로 나왔다). 이때는 **높이의 차이**
                              만 쓴다. 기준면 오프셋이 상쇄된다.
                              H = (1-base_frac) * h_ext - freeboard
                              h_ext = 용기 점군 높이 p95-p5,  freeboard = p95 - h_fill

    on_plate 는 용기 볼록껍질이 접시 볼록껍질 안에 든 비율. 교집합으로 보면 안 된다
    (SAM 은 접시와 그릇을 겹치지 않게 자른다).
    """
    h = np.asarray(h_container, float)
    h = h[np.isfinite(h)]
    if len(h) < 200:
        return None, {}
    top = float(np.percentile(h, 95))
    ext = top - float(np.percentile(h, 5))
    free = top - float(h_fill_plate)
    if on_plate >= 0.5:
        H = float(h_fill_plate) - base_frac * top
        mode = "plate"
    else:
        H = (1.0 - base_frac) * ext - free
        mode = "free"
    info = {"h_rim": top, "h_ext": ext, "freeboard": free, "mode": mode, "H": H}
    return (H if H > 2.0 else None), info
