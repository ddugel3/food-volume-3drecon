"""메시 자세·크기를 '카메라로 투영해 마스크와 맞추기' 로 정한다 (render-and-compare).

이전 방식의 결함: 접시 평면에 투영한 **발자국** 실루엣으로 비교했다. 그런데
  관측 발자국 = 보이는 윗면만 평면에 번진 그림자
  메시 발자국 = 물체 전체의 그림자
로 정의가 다르다. 비스듬히 본 물체일수록 어긋나고, 실제로 바나나 IoU 0.42,
체다 0.54, 당근 0.56 처럼 자세를 못 찾아 부피가 터졌다(당근 4337 mL).

여기서는 우리가 가장 신뢰하는 증거인 **SAM 3 의 2D 마스크**에 직접 맞춘다.
내부 파라미터를 EXIF 로 알고 있으므로 메시를 실제 카메라로 투영할 수 있다.

스케일이 결정되는 원리:
  투영 실루엣의 크기는 (물체 크기 / 물체 거리) 로 정해진다. 거리는 접시로 이미
  알고 있으므로(z_mm), 실루엣 크기가 곧 물체 크기다. 그래서 ICP 처럼 3D 기복에
  의존하지 않는다 -- 살사처럼 관측면이 거의 평평해서 ICP 가 스케일을 못 정하는
  축퇴 상황에서도 2D 크기는 항상 정보를 준다.

물리 제약: 음식은 지지평면에 놓여 있다. 그래서 회전마다 '가장 낮은 점이 평면에
닿도록' 평행이동을 정한다. 이것이 자유도를 하나 줄이고 자세를 물리적으로 만든다.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def so3_grid(n=600, seed=0):
    """SO(3) 준균일 표본. 24개 축정렬 회전으로는 너무 성기다."""
    return Rotation.random(n, random_state=seed).as_matrix()


def inplane(theta):
    """지지평면 안에서의 회전. 물체가 놓인 채 제자리에서 도는 자유도 하나."""
    c, s_ = np.cos(theta), np.sin(theta)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def rasterize_uv(u, v, H, W, ss=2):
    """투영점을 이미지 격자에 채운다. ss 배 다운샘플로 구멍을 줄인다."""
    # mask[::ss] 는 올림이므로 여기서도 올림을 써야 모양이 맞는다
    h, w = -(-H // ss), -(-W // ss)
    iu = np.clip((u / ss).astype(np.int32), 0, w - 1)
    iv = np.clip((v / ss).astype(np.int32), 0, h - 1)
    m = np.zeros((h, w), bool)
    m[iv, iu] = True
    from scipy import ndimage
    m = ndimage.binary_closing(m, np.ones((3, 3)))
    return ndimage.binary_fill_holes(m)


def project(pts, f, cx, cy):
    z = pts[:, 2]
    ok = z > 1e-6
    return f * pts[ok, 0] / z[ok] + cx, f * pts[ok, 1] / z[ok] + cy


def place(pts, R, s, plane, anchor):
    """회전 R, 배율 s 로 놓되 (1) 중심을 관측 중심에 (2) 최저점이 평면에 닿게."""
    P = s * (pts @ R.T)
    P = P - P.mean(0) + anchor
    h = plane.height(P)
    return P - plane.n * h.min()          # n 은 위쪽(카메라쪽)을 향한다


def search(pts, mask, plane, anchor, f, cx, cy, n_rot=600, ss=2, iters=3, seed=0):
    """회전 격자를 훑고, 각 회전마다 투영 면적이 마스크 면적과 같아지도록 s 를 맞춘다.

    면적은 크기의 제곱이므로 s <- s * sqrt(A_mask / A_proj) 로 두세 번이면 수렴한다.
    """
    from scipy import ndimage
    H, W = mask.shape
    tgt = ndimage.binary_fill_holes(mask)[::ss, ::ss]
    A_t = int(tgt.sum())
    if A_t < 20:
        return None
    # 초기 배율: 물체 최대폭이 마스크 최대폭에 대응한다고 보고 잡는다
    span_px = max(np.ptp(np.nonzero(mask)[1]), np.ptp(np.nonzero(mask)[0]))
    span_mm = span_px / f * float(anchor[2])
    s0 = span_mm / max(np.ptp(pts, axis=0).max(), 1e-9)

    best = (-1.0, None, None)
    for R in so3_grid(n_rot, seed):
        s = s0
        for _ in range(iters):
            P = place(pts, R, s, plane, anchor)
            u, v = project(P, f, cx, cy)
            sil = rasterize_uv(u, v, H, W, ss)
            A_p = int(sil.sum())
            if A_p < 10:
                break
            s *= float(np.sqrt(A_t / A_p))
        else:
            inter = np.logical_and(sil, tgt).sum()
            union = np.logical_or(sil, tgt).sum()
            iou = inter / union if union else 0.0
            if iou > best[0]:
                best = (iou, R, s)
    return best


def refine(pts, mask, plane, anchor, f, cx, cy, R, s, ss=2,
           schedule=((30.0, 120), (12.0, 96), (5.0, 72), (2.0, 48)), seed=1):
    """거친 격자에서 찾은 자세를 반경을 좁혀가며 다단계로 정련한다.

    이전 구현의 약점: SO(3) 무작위 300개면 정답까지 대략 30~40도인데 정련 반경이
    8도 하나뿐이라 그 사이에 구멍이 있었다. 30 -> 12 -> 5 -> 2도로 좁혀 내려가면
    거친 격자 간격부터 미세 조정까지 끊김 없이 덮인다.

    안정 자세(볼록껍질 지지면) 기반 매개화도 시험했지만 더 나빴다 -- 이 데이터의
    음식은 볼 안에 담기거나(파스타) 컵에 들어 있거나(살사) 반으로 잘려 기대어
    있어서(사과) '평평한 바닥에 스스로 놓였다' 는 가정이 성립하지 않는다.
    측정값: 8개 중 2개만 개선, 파스타 IoU 0.83->0.28.
    """
    from scipy import ndimage
    H, W = mask.shape
    tgt = ndimage.binary_fill_holes(mask)[::ss, ::ss]
    A_t = int(tgt.sum())
    rng = np.random.default_rng(seed)

    def score(Rc, sc):
        for _ in range(3):
            P = place(pts, Rc, sc, plane, anchor)
            u, v = project(P, f, cx, cy)
            sil = rasterize_uv(u, v, H, W, ss)
            a = int(sil.sum())
            if a < 10:
                return -1.0, sc
            sc *= float(np.sqrt(A_t / a))
        inter = np.logical_and(sil, tgt).sum()
        union = np.logical_or(sil, tgt).sum()
        return (inter / union if union else 0.0), sc

    best_iou, best_R, best_s = score(R, s)[0], R, s
    for sigma_deg, n in schedule:
        improved = True
        while improved:
            improved = False
            for _ in range(n):
                Rc = Rotation.from_rotvec(
                    rng.normal(0, np.radians(sigma_deg), 3)).as_matrix() @ best_R
                i2, s2 = score(Rc, best_s)
                if i2 > best_iou:
                    best_iou, best_R, best_s = i2, Rc, s2
                    improved = True
            if not improved:
                break
    return best_iou, best_R, best_s


# ---------------------------------------------------------------- 안정 자세 기반 탐색
def stable_poses(mesh, max_poses=16):
    """볼록껍질의 안정 지지면들을 열거한다. 각각 물체가 실제로 놓일 수 있는 자세다.

    SO(3) 무작위 표본 300개는 대부분 물체를 공중에 세운 불가능한 자세라 낭비이고,
    정답 근처는 30~40도 간격으로만 훑게 된다. 음식은 접시 위에 안정하게 놓여
    있으므로 후보를 '안정 지지면 x 면내 회전' 으로 줄이면 전부 유효한 자세가 되고
    같은 계산량으로 훨씬 촘촘히 훑을 수 있다.

    반환: [(R, prob)] -- R 은 메시 좌표를 '+z 가 위' 인 좌표로 보내는 회전.
    """
    import trimesh
    try:
        T, P = trimesh.poses.compute_stable_poses(mesh.convex_hull, n_samples=1,
                                                  threshold=0.005)
    except Exception:
        return []
    out = [(np.asarray(t)[:3, :3], float(p)) for t, p in zip(T[:max_poses], P[:max_poses])]
    return out


def search_stable(mesh, pts, mask, plane, anchor, f, cx, cy,
                  n_angle=72, ss=3, iters=3, refine_step=2.0):
    """안정 자세 x 면내 회전으로 훑고, 최적점 주변을 잘게 정련한다.

    plane.n 은 카메라 쪽(위)을 향하므로, '메시의 +z 가 위' 인 자세를
    장면 좌표로 옮기려면 +z 를 plane.n 에 맞추는 회전을 한 번 더 곱한다.
    """
    from scipy import ndimage
    sp = stable_poses(mesh)
    if not sp:
        return None
    # +z 를 평면 법선에 맞추는 회전
    z = np.array([0.0, 0.0, 1.0])
    n = plane.n / np.linalg.norm(plane.n)
    v = np.cross(z, n); c = float(z @ n)
    if np.linalg.norm(v) < 1e-9:
        A = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        A = np.eye(3) + K + K @ K * (1.0 / (1.0 + c))

    H, W = mask.shape
    tgt = ndimage.binary_fill_holes(mask)[::ss, ::ss]
    A_t = int(tgt.sum())
    if A_t < 20:
        return None
    span_px = max(np.ptp(np.nonzero(mask)[1]), np.ptp(np.nonzero(mask)[0]))
    s0 = (span_px / f * float(anchor[2])) / max(np.ptp(pts, axis=0).max(), 1e-9)

    def score(R, s):
        for _ in range(iters):
            P = place(pts, R, s, plane, anchor)
            u, v_ = project(P, f, cx, cy)
            sil = rasterize_uv(u, v_, H, W, ss)
            a = int(sil.sum())
            if a < 10:
                return -1.0, s
            s *= float(np.sqrt(A_t / a))
        inter = np.logical_and(sil, tgt).sum()
        union = np.logical_or(sil, tgt).sum()
        return (inter / union if union else 0.0), s

    best = (-1.0, None, None, 0.0)
    for Rs, prob in sp:
        for k in range(n_angle):
            R = A @ inplane(2 * np.pi * k / n_angle) @ Rs
            iou, s = score(R, s0)
            if iou > best[0]:
                best = (iou, R, s, prob)
    if best[1] is None:
        return None
    # 최적점 주변 면내 각도를 잘게 (거친 격자 간격의 절반 범위)
    iou, R, s, prob = best
    half = np.radians(360.0 / n_angle / 2)
    for d in np.linspace(-half, half, int(2 * half / np.radians(refine_step)) + 1):
        Rc = A @ inplane(d) @ np.linalg.solve(A, R)
        i2, s2 = score(Rc, s)
        if i2 > iou:
            iou, R, s = i2, Rc, s2
    return iou, R, s, prob
