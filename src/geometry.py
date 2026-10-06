"""깊이맵 -> 포인트클라우드 -> 지지평면 -> 높이장 적분 -> 부피(mL).

여기가 파이프라인의 기하 핵심이다. 설계 판단 세 가지를 먼저 적어둔다.

1) 픽셀 공간에서 적분하지 않는다.
   흔한 구현은 "픽셀마다 높이 x 픽셀 면적"을 더하고 원근 축소를 z^2/(fx*fy) 로
   보정한다. 이 근사는 표면이 카메라 광축에 수직일 때만 맞고, 접시가 기울어
   보이는 실제 식탁 사진에서는 계통 편차를 만든다. CVPR2025 3위 팀(PSHS)이
   픽셀 공간 휴리스틱으로 MAPE 0.46 을 받은 것과 같은 종류의 함정이다.
   대신 점들을 **지지평면 좌표계로 옮겨 균일 격자에 재샘플링**한 뒤 적분한다.
   격자가 균일하므로 셀 면적이 상수이고 원근 보정이 애초에 필요 없어진다.

2) 높이장은 셀당 최대값을 쓴다.
   단안 깊이는 음식의 '윗면'만 본다. 한 셀에 여러 점이 떨어지면 그중 가장 높은
   점이 윗면이다. 평균을 쓰면 경계에서 접시 쪽 점들에 끌려 내려간다.

3) 반환은 mL 이고 내부 계산은 mm 로 한다.
   대회 제공 calculate_volumes.py 가 `volume_ml = mesh.volume * 1e-3` 이므로
   GT 메시는 mm 단위이고 제출 단위는 mL(=cm^3) 다. 여기서도 맞춘다.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Plane:
    """n·x + d = 0. n 은 단위법선이며 카메라 쪽(음식이 있는 쪽)을 향한다."""
    n: np.ndarray
    d: float

    def height(self, pts: np.ndarray) -> np.ndarray:
        """평면 위쪽을 양수로 하는 부호 있는 거리."""
        return pts @ self.n + self.d

    def basis(self):
        """평면 위의 정규직교 접선축 (u, v)."""
        a = np.array([1.0, 0.0, 0.0])
        if abs(self.n @ a) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        u = np.cross(self.n, a); u /= np.linalg.norm(u)
        v = np.cross(self.n, u)
        return u, v


def unproject(depth_mm: np.ndarray, K) -> np.ndarray:
    """깊이맵(mm) -> (H, W, 3) 카메라 좌표 포인트맵(mm)."""
    fx, _, cx = K[0]
    _, fy, cy = K[1]
    h, w = depth_mm.shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    z = depth_mm.astype(np.float64)
    return np.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z], axis=-1)


def fit_plane_ransac(pts: np.ndarray, iters: int = 2000, thresh_mm: float = 2.0,
                     rng: np.random.Generator | None = None) -> Plane:
    """RANSAC 평면 피팅 후 인라이어로 최소제곱 재적합.

    접시 표면은 대체로 평면이지만 테두리 융기와 음식 그림자로 아웃라이어가 섞인다.
    RANSAC 으로 대충 잡고 인라이어 전체에 SVD 를 다시 돌려 법선을 다듬는다.
    """
    rng = rng or np.random.default_rng(0)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 3:
        raise ValueError("평면을 맞출 점이 3개 미만이다")

    best_n, best_d, best_cnt = None, 0.0, -1
    for _ in range(iters):
        i = rng.choice(len(pts), 3, replace=False)
        p0, p1, p2 = pts[i]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -(n @ p0)
        cnt = int((np.abs(pts @ n + d) < thresh_mm).sum())
        if cnt > best_cnt:
            best_n, best_d, best_cnt = n, d, cnt

    inl = np.abs(pts @ best_n + best_d) < thresh_mm
    q = pts[inl]
    c = q.mean(axis=0)
    _, _, vt = np.linalg.svd(q - c, full_matrices=False)
    n = vt[-1] / np.linalg.norm(vt[-1])
    d = -(n @ c)

    # 법선이 카메라 쪽(-z)을 향하게 뒤집는다. 카메라는 원점, 물체는 z>0 이므로
    # 접시 위쪽 방향은 카메라를 향하는 -z 성분을 가진다.
    if n[2] > 0:
        n, d = -n, -d
    return Plane(n=n, d=float(d))


def heightfield_volume(pts_mm: np.ndarray, plane: Plane, grid_mm: float = 1.0,
                       fill_holes: bool = True):
    """평면 좌표계 균일 격자에 높이장을 만들고 적분한다.

    Returns (volume_ml, info) -- info 에 격자 크기, 점유 셀 수, 최대 높이 등.
    """
    pts = np.asarray(pts_mm, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return 0.0, {"reason": "빈 점군"}

    u, v = plane.basis()
    a = pts @ u
    b = pts @ v
    hgt = plane.height(pts)

    a0, b0 = a.min(), b.min()
    ia = np.floor((a - a0) / grid_mm).astype(np.int64)
    ib = np.floor((b - b0) / grid_mm).astype(np.int64)
    na, nb = int(ia.max()) + 1, int(ib.max()) + 1
    if na * nb > 40_000_000:
        raise ValueError(f"격자가 너무 큽니다 ({na}x{nb}). grid_mm 을 키우세요.")

    # 셀당 최대 높이 (윗면)
    hf = np.full((na, nb), -np.inf)
    np.maximum.at(hf, (ia, ib), hgt)
    occupied = np.isfinite(hf)

    if fill_holes:
        hf, occupied = _fill_footprint(hf, occupied)

    h = np.where(occupied, hf, 0.0)
    np.clip(h, 0.0, None, out=h)          # 평면 아래는 음식이 아니다
    vol_mm3 = float(h.sum()) * grid_mm * grid_mm
    return vol_mm3 * 1e-3, {
        "grid": (na, nb),
        "grid_mm": grid_mm,
        "cells": int(occupied.sum()),
        "h_max_mm": float(h.max()) if occupied.any() else 0.0,
        "footprint_cm2": float(occupied.sum()) * grid_mm ** 2 / 100.0,
    }


def _fill_footprint(hf: np.ndarray, occ: np.ndarray):
    """발자국 내부의 빈 셀을 이웃 평균으로 메운다.

    원근 때문에 카메라에서 먼 쪽은 표면 샘플이 성기다. 발자국 내부에 빈 셀이
    남으면 그만큼 부피가 깎이므로 반드시 메워야 한다. 발자국 경계는 건드리지 않는다.
    """
    from scipy import ndimage

    filled = ndimage.binary_fill_holes(occ)
    holes = filled & ~occ
    if not holes.any():
        return hf, occ

    work = np.where(occ, hf, 0.0)
    known = occ.copy()
    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
    for _ in range(64):
        rem = holes & ~known
        if not rem.any():
            break
        s = ndimage.convolve(np.where(known, work, 0.0), k, mode="constant")
        c = ndimage.convolve(known.astype(np.float64), k, mode="constant")
        upd = rem & (c > 0)
        work[upd] = s[upd] / c[upd]
        known |= upd
    return np.where(known, work, -np.inf), known
