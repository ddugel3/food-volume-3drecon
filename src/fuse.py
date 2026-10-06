"""증거 융합: 역분산 로그공간 가중평균 + MAPE 최적 수축.

이전 구현은 무게를 손으로 정했다(기하 1, Qwen 1, 판정 있으면 2). 근거가 없었다.
제대로 유도하면 무게는 각 증거원의 sigma 가 정한다.

  로그정규 가정에서 독립 추정치 log v_i ~ N(log V, sigma_i^2) 들의
    사후평균   mu    = sum(log v_i / sigma_i^2) / sum(1 / sigma_i^2)
    사후분산   s_p^2 = 1 / sum(1 / sigma_i^2)
    MAPE 최적  log p* = mu - s_p^2

  MAPE 최적점이 사후평균이 아니라 그보다 낮은 이유: MAPE 는 비대칭이다.
  과소추정 오차는 항목당 1.0 에서 막히지만 과대추정은 무한대로 열려 있다.
  E|p/V - 1| 을 최소화하는 조건 E[e^Y 1{Y>0}] = E[e^Y 1{Y<0}] (Y = log p - log V)를
  정규분포에서 풀면 정확히 m = -sigma^2 가 나온다.

sigma 를 어떻게 잡는가가 실질적 문제다.

  기하 경로  높이장과 메시정합은 **독립이 아니다**. 같은 세그멘테이션 마스크,
             같은 접시 스케일, 같은 크롭 위에 서 있다. 그래서 둘을 하나의 증거원으로
             묶고(기하평균), sigma_geo 는 (a) 두 경로 불일치와 (b) 깊이 백엔드 간
             산포를 합쳐 잡는다. 다만 공통 계통오차는 둘 다 못 잡으므로 하한이 있다.
  Qwen       기하와 완전히 독립이다. 벤치마크에서 VLM 단독(GPT-5.2)이 MAPE 0.34 였고
             MAPE ~ sigma 이므로 sigma_qwen ~ 0.35 를 기본으로 둔다.

리더보드 public 이 우리 기하 위주 버전에서 0.50 이었으므로 sigma_geo ~ 0.5 가 관측값에
가깝다. 그러면 Qwen 의 무게가 (0.5/0.35)^2 = 2.0 배가 되어야 한다 -- 제가 손으로 준
1.0 은 Qwen 을 절반으로 과소평가한 셈이었다.

모든 sigma 는 인자로 노출한다. 검증 데이터가 없으므로 정답을 알 수 없고,
리더보드가 유일한 판정자이기 때문이다.
"""
import numpy as np

SIGMA_GEO_FLOOR = 0.30   # 공통 계통오차 하한. 두 경로가 완전히 일치해도 이만큼은 모른다.
SIGMA_QWEN = 0.35        # VLM 단독 MAPE 0.34 (벤치마크 GPT-5.2) 에서 온 값


def _clean(x):
    return x if (x is not None and np.isfinite(x) and x > 0) else None


def sigma_geo(h, m, backend_spread=None, floor=SIGMA_GEO_FLOOR):
    """기하 증거원의 sigma. 두 경로 불일치와 백엔드 산포를 직교 합성한다."""
    parts = [floor]
    h, m = _clean(h), _clean(m)
    if h and m:
        parts.append(abs(np.log(h / m)) / 2.0)   # 불일치의 절반을 1시그마로
    if backend_spread and np.isfinite(backend_spread):
        parts.append(float(backend_spread))
    return float(np.sqrt(np.sum(np.square(parts))))


def fuse(h, m, q, verdict="", backend_spread=None,
         sigma_qwen=SIGMA_QWEN, geo_floor=SIGMA_GEO_FLOOR,
         w_qwen_scale=1.0, shrink=True):
    """-> (예측값 mL, 사후 sigma). w_qwen_scale 로 Qwen 무게를 배율 조정한다."""
    h, m, q = _clean(h), _clean(m), _clean(q)
    geo = [x for x in (h, m) if x]
    srcs = []
    if geo:
        g = float(np.exp(np.mean(np.log(geo))))
        srcs.append((np.log(g), sigma_geo(h, m, backend_spread, geo_floor)))
    if q:
        # 판정이 too_large/too_small 이면 그 항목에 대해 Qwen 이 더 확신한다는 뜻이므로
        # sigma 를 줄인다(무게 증가). 값이 다른 것과 '틀렸다고 명시한 것' 은 정보량이 다르다.
        sq = sigma_qwen * (0.75 if verdict in ("too_large", "too_small") else 1.0)
        sq = sq / max(w_qwen_scale, 1e-6) ** 0.5
        srcs.append((np.log(q), sq))
    if not srcs:
        return None, None
    wsum = sum(1.0 / s ** 2 for _, s in srcs)
    mu = sum(lv / s ** 2 for lv, s in srcs) / wsum
    s_p2 = 1.0 / wsum
    return float(np.exp(mu - (s_p2 if shrink else 0.0))), float(np.sqrt(s_p2))


# ---------------------------------------------------------------- 형상 일관성 검사
# 형상 계열별로 '평균 높이 / 등가 지름' 이 들어야 할 범위. 해석적 하한·상한이다.
#   누운 원통  단면이 반원이므로 평균높이 ~ 지름의 0.39, 여유를 두어 0.20~0.60
#   슬래브     얇은 판. 0.05~0.35
#   구/돔      0.30~0.80 / 0.25~0.75
SHAPE_HW = {"slab": (0.05, 0.35), "wedge": (0.08, 0.40), "dome": (0.25, 0.75),
            "sphere": (0.30, 0.80), "cylinder": (0.20, 0.60), "torus": (0.15, 0.55),
            "irregular": (0.15, 0.70)}


def shape_consistent(vol_ml, area_cm2, shape):
    """추정 부피가 그 형상 계열에서 물리적으로 가능한가.

    실루엣 IoU(오차와 상관 -0.24)나 정합 잔차(-0.17)로는 실패를 못 걸렀는데
    이 검사는 걸린다. 물리 제약이기 때문이다 -- 누운 당근의 평균 높이가 지름의
    0.03 배일 수는 없다. 실측:
      메시 경로  범위내 평균 APE 0.276 / 범위밖 0.464
      Qwen      범위내 0.238 / 범위밖 0.527
    """
    if not vol_ml or not area_cm2 or area_cm2 <= 0 or vol_ml <= 0:
        return True
    h_mean = vol_ml * 1000.0 / (area_cm2 * 100.0)          # mm
    w_eq = 2.0 * np.sqrt(area_cm2 * 100.0 / np.pi)         # mm
    lo, hi = SHAPE_HW.get((shape or "").strip().lower(), (0.05, 0.90))
    return lo <= (h_mean / w_eq) <= hi


def fuse_gated(h, m, q, verdict="", backend_spread=None, area_cm2=None, shape="",
               sigma_qwen=SIGMA_QWEN, geo_floor=SIGMA_GEO_FLOOR,
               w_qwen_scale=1.0, shrink=True, penalty=2.5):
    """형상 일관성을 위반한 증거원의 sigma 를 키워 자동으로 무게를 뺀다.

    증거를 버리지 않고 sigma 만 키우는 이유: 위반이 곧 오답은 아니고,
    다른 증거가 전부 위반이면 그래도 무언가는 써야 하기 때문이다.
    역분산 융합이므로 sigma 를 p 배 하면 무게가 1/p^2 로 준다.
    """
    h_, m_, q_ = _clean(h), _clean(m), _clean(q)
    srcs = []
    geo = [x for x in (h_, m_) if x]
    if geo:
        g = float(np.exp(np.mean(np.log(geo))))
        sg = sigma_geo(h_, m_, backend_spread, geo_floor)
        # 기하 두 경로 중 위반한 쪽이 있으면 그만큼 신뢰를 낮춘다
        bad = sum(0 if shape_consistent(x, area_cm2, shape) else 1 for x in geo)
        sg *= penalty ** (bad / max(len(geo), 1))
        srcs.append((np.log(g), sg))
    if q_:
        sq = sigma_qwen * (0.75 if verdict in ("too_large", "too_small") else 1.0)
        sq = sq / max(w_qwen_scale, 1e-6) ** 0.5
        if not shape_consistent(q_, area_cm2, shape):
            sq *= penalty
        srcs.append((np.log(q_), sq))
    if not srcs:
        return None, None
    wsum = sum(1.0 / s ** 2 for _, s in srcs)
    mu = sum(lv / s ** 2 for lv, s in srcs) / wsum
    s_p2 = 1.0 / wsum
    return float(np.exp(mu - (s_p2 if shrink else 0.0))), float(np.sqrt(s_p2))
