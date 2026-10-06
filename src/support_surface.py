"""지지면을 평면이 아니라 **측정된 회전 프로파일**로 잡는다.

왜 바꾸는가. 접시는 평평하지 않다. 깊이맵으로 반경별 높이를 재보면 실제로 오목하다.
  콤보 6: 안쪽 -24.2mm, 바깥 -0.7mm -> 우물 깊이 23.5mm
  콤보 3: 17.6mm,  콤보 9: 12.0mm,  콤보 13: 10.2mm
그런데 음식 높이는 10~60mm 다. 쿠키 6.8mm, 튀긴계란 11.5mm, 케사디야 12.8mm 처럼
얇은 항목에서는 지지면이 10~20mm 어긋나면 부피가 두 배로 틀리거나 음수가 된다.
RANSAC 평면은 테두리와 우물 바닥 사이 어딘가에 걸리므로, 음식이 우물 안에 있으면
높이가 통째로 깎인다. **용기 4개가 아니라 34개 전부에 영향을 준다.**

해법은 단순하다. 접시는 회전 대칭이므로 지지면을 h(rho) 로 두고,
음식에 가려지지 않은 고리 영역에서 측정한 뒤 안쪽으로 외삽한다.
생성 모델도 VLM 도 필요 없다 -- 이미 측정된 값을 쓰는 것뿐이다.

볼과 컵에도 그대로 통한다. 음식 위로 드러난 안쪽 벽이 프로파일을 주고,
바닥은 그 추세를 외삽한다. Qwen 에게 깊이비를 묻던 것을 측정이 대신한다.
"""
import numpy as np


def isotonic_profile(rho, h, n_bins=28, min_pts=30):
    """반경별 높이 프로파일을 **단조 증가** 제약으로 맞춘다.

    접시와 그릇의 안쪽 면은 바깥으로 갈수록 높아진다(우물 -> 테두리). 이 물리
    제약을 걸면 음식에 가린 안쪽으로의 외삽이 안정된다 -- 단순 보간은 가장 안쪽
    관측값을 그대로 유지하는데, 그 값이 이미 우물 바깥이면 지지면이 통째로 뜬다.
    실제로 스테이크가 198 -> 337 mL 로 폭주한 원인이었다.
    """
    from sklearn.isotonic import IsotonicRegression
    edges = np.linspace(0.0, float(np.percentile(rho, 99)), n_bins + 1)
    cs, hs, ws = [], [], []
    for i in range(n_bins):
        sel = (rho >= edges[i]) & (rho < edges[i + 1])
        if sel.sum() >= min_pts:
            cs.append(0.5 * (edges[i] + edges[i + 1]))
            hs.append(float(np.median(h[sel])))
            ws.append(float(sel.sum()))
    if len(cs) < 3:
        return None, None
    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    hh = ir.fit_transform(np.array(cs), np.array(hs), sample_weight=np.array(ws))
    return np.array(cs), np.asarray(hh, float)


class RadialSupport:
    """회전 대칭 지지면. 기준 평면 위로의 높이를 반경의 함수로 준다."""

    def __init__(self, plane, center_ab, rho, h, r_max, n_bins=24, min_pts=40):
        self.plane, self.center = plane, center_ab
        self.r_max = r_max
        # 반경 구간별 중앙값으로 프로파일을 만든다 (이상치에 끌리지 않게)
        rc, hc = isotonic_profile(rho, h, n_bins=n_bins, min_pts=min_pts)
        self.rc, self.hc = (rc, hc) if rc is not None else (np.array([]), np.array([]))
        self.ok = rc is not None

    def height_of_surface(self, rho):
        """반경 rho 에서의 지지면 높이. 관측 구간 밖은 가장자리 값으로 외삽한다.

        안쪽(음식에 가린 영역)으로의 외삽은 가장 안쪽 관측 구간의 값을 유지한다.
        추세를 선형 외삽하면 접시 중앙이 비현실적으로 깊어질 수 있어 보수적으로 간다.
        """
        if not self.ok:
            return np.zeros_like(np.asarray(rho, float))
        return np.interp(np.asarray(rho, float), self.rc, self.hc,
                         left=self.hc[0], right=self.hc[-1])

    def food_height(self, pts):
        """음식 점들의 '지지면 위로의' 높이. 평면 기준이 아니라 굽은 면 기준이다."""
        u, v = self.plane.basis()
        a, b = pts @ u, pts @ v
        rho = np.hypot(a - self.center[0], b - self.center[1])
        return self.plane.height(pts) - self.height_of_surface(rho)


def build_support(plane, ring_pts, n_bins=24, center=None):
    """음식에 가려지지 않은 접시/용기 고리 점들로 지지면을 만든다.

    center 를 반드시 **전체 프레임의 접시 마스크**에서 계산해 넘겨야 한다.
    크롭이 접시를 잘라내면 고리가 비대칭이 되고, 그 중앙값을 중심으로 쓰면
    반경이 통째로 어긋난다.
    """
    u, v = plane.basis()
    a, b = ring_pts @ u, ring_pts @ v
    a0, b0 = center if center is not None else (float(np.median(a)), float(np.median(b)))
    rho = np.hypot(a - a0, b - b0)
    h = plane.height(ring_pts)
    r_max = float(np.percentile(rho, 99))
    return RadialSupport(plane, (a0, b0), rho, h, r_max, n_bins=n_bins)


class HybridSupport:
    """보이는 곳은 측정, 가려진 안쪽만 생성 프로파일로 채우는 지지면.

    앞선 세 번의 실패에서 배운 것:
      - 단순 보간 프로파일     MAPE 0.498  (평면 0.481)
      - 단조 제약 프로파일     0.498
      - 생성 접시 프로파일     1.137  <- 전부 과대. 지지면을 너무 깊게 잡았다.

    공통 원인은 **지지면의 절대 위치**였다. 우물 모양을 알아도 '평면이 그 우물의
    어디에 걸려 있는지' 를 모르면 오프셋이 통째로 틀린다. 생성 프로파일을 테두리에서
    0 으로 놓았는데 RANSAC 평면은 실제로는 우물 바닥 근처에 맞춰지므로, 우물 깊이를
    **중복으로** 빼서 튀긴계란이 50 -> 131 mL 로 부풀었다.

    그래서 오프셋을 밖에서 정하지 않고 **링 점들이 정하게** 한다.
      h_support(rho) = c + prof(rho),   c = mean(h_ring - prof(rho_ring))
    관측 반경 범위 안에서는 측정 프로파일을 그대로 쓰고, 음식에 가려 관측이 없는
    안쪽만 생성 프로파일의 **모양**을 빌려 경계에서 이어 붙인다.
    측정할 수 있는 곳은 측정하고, 못 보는 곳만 생성에 맡긴다.
    """

    def __init__(self, plane, center_ab, rho_ring, h_ring, gen_rs=None, gen_hs_mm=None,
                 n_bins=24, min_pts=30):
        self.plane, self.center = plane, center_ab
        rc, hc = isotonic_profile(rho_ring, h_ring, n_bins=n_bins, min_pts=min_pts)
        self.ok = rc is not None
        if not self.ok:
            return
        self.rc, self.hc = rc, hc
        self.r_lo = float(rc[0])
        self.inner = None
        if gen_rs is not None and gen_hs_mm is not None and len(gen_rs) >= 3:
            # 생성 프로파일을 관측 최내곽에서 이어 붙인다 (모양만 빌린다)
            g_at_lo = float(np.interp(self.r_lo, gen_rs, gen_hs_mm))
            self.inner = (np.asarray(gen_rs, float),
                          np.asarray(gen_hs_mm, float) - g_at_lo + float(hc[0]))

    def height_of_surface(self, rho):
        rho = np.asarray(rho, float)
        out = np.interp(rho, self.rc, self.hc, left=self.hc[0], right=self.hc[-1])
        if self.inner is not None:
            m = rho < self.r_lo
            if m.any():
                out[m] = np.interp(rho[m], self.inner[0], self.inner[1],
                                   left=self.inner[1][0], right=self.inner[1][-1])
        return out

    def food_height(self, pts):
        u, v = self.plane.basis()
        rho = np.hypot(pts @ u - self.center[0], pts @ v - self.center[1])
        return self.plane.height(pts) - self.height_of_surface(rho)
