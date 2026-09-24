"""ステージ4: 床面推定と定数オフセット。

床の高さの補正は、シーケンス全体で 1 つの定数（区間モードでも区間ごとの定数）とする。
フレーム毎に変わる補正は入れない（入れると接地判定やロック値に歪みが混ざる）。
"""
from dataclasses import dataclass, field

import numpy as np

from . import quat


def horizontal_speed(points, fps):
    """(T, ...) の点の水平速度 [単位/s]。"""
    v = np.gradient(np.asarray(points, np.float64), axis=0) * fps if len(points) > 1 else \
        np.zeros_like(points)
    return np.linalg.norm(v[..., [0, 2]], axis=-1)


def fit_plane_ransac(points, iterations, inlier_dist, rng, max_tilt_deg=90.0):
    """RANSAC で平面を当てはめ、インライアで最小二乗し直す。法線は +Y 側を向ける。

    傾きが max_tilt_deg を超える候補（床ではありえない平面）は最初から採用しない。
    戻り値: (法線, インライアの bool, インライアの広がり = 平面内の 2 番目の主軸方向の標準偏差)
    """
    best = None
    n_pts = len(points)
    min_ny = np.cos(np.deg2rad(max_tilt_deg))
    for _ in range(int(iterations)):
        a, b, c = points[rng.choice(n_pts, 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9 or abs(n[1]) / norm < min_ny:
            continue
        n /= norm
        inl = np.abs((points - a) @ n) < inlier_dist
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 3:
        best = np.ones(n_pts, bool)
    p = points[best]
    centroid = p.mean(0)
    _, sv, vt = np.linalg.svd(p - centroid, full_matrices=False)
    normal = vt[-1]
    if normal[1] < 0:
        normal = -normal
    spread = float(sv[1] / np.sqrt(len(p))) if len(sv) > 1 else 0.0
    return normal, best, spread


@dataclass
class FloorResult:
    rotation: np.ndarray                 # (3, 3) 床の傾き補正（＋向きの反転）
    offset: np.ndarray                   # (T, 3) 回転後に足す量（上下は定数・区間定数、水平は定数）
    normal: np.ndarray                   # 推定した床の法線（補正前）
    tilt_deg: float                      # 推定した床の傾き
    num_points: int
    num_inliers: int
    fallback: bool                       # 候補点が足りず、傾きを推定しなかった
    tilt_applied: bool = False           # 傾きの補正を実際に掛けたか
    spread: float = 0.0                  # 候補点の平面内の広がり [m]
    segment_heights: list = field(default_factory=list)

    def apply(self, kin):
        return kin.transformed(self.rotation, self.offset)


def _cosine_blend_curve(values, bounds, T, blend):
    """区間ごとの定数 values を、境界の前後 blend フレームでコサイン補間してつなぐ。"""
    curve = np.empty(T)
    for (s, e), v in zip(bounds, values):
        curve[s:e] = v
    half = blend / 2.0
    for i in range(1, len(bounds)):
        b = bounds[i][0]
        v0, v1 = values[i - 1], values[i]
        lo, hi = int(max(0, np.floor(b - half))), int(min(T, np.ceil(b + half)))
        t = (np.arange(lo, hi) - (b - half)) / max(blend, 1e-9)
        curve[lo:hi] = v0 + (v1 - v0) * (0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1)))
    return curve


def estimate_floor(kin, fps, cfg):
    """かかと・つま先の低速点から床を推定し、床が y=0・法線が +Y になる変換を返す。"""
    pts = kin.contact_points.reshape(len(kin.contact_points), -1, 3)   # (T, 4, 3)
    T = len(pts)
    speed = horizontal_speed(pts, fps)
    low = speed < float(cfg.low_speed_m_per_s)
    cand = pts[low]
    frame_of = np.nonzero(low)[0]
    rng = np.random.default_rng(int(cfg.seed))

    R = np.eye(3)
    normal = np.array([0.0, 1.0, 0.0])
    inliers = np.ones(len(cand), bool)
    fallback = len(cand) < int(cfg.min_points)
    tilt, spread, tilt_applied = 0.0, 0.0, False
    if not fallback:
        normal, inliers, spread = fit_plane_ransac(cand, cfg.ransac_iterations,
                                                   cfg.ransac_inlier_m, rng, cfg.max_tilt_deg)
        tilt = float(np.rad2deg(np.arccos(np.clip(normal[1], -1, 1))))
        # 候補点がほぼ一直線に並ぶ（例: まっすぐ歩くだけ）と、その線まわりの傾きは決まらない。
        # 平面内に十分な広がりがあるときだけ傾きを補正する
        if cfg.align_tilt and tilt <= float(cfg.max_tilt_deg) and \
                spread >= float(cfg.min_spread_m):
            R = quat.to_matrix(quat.from_two_vectors(normal, [0.0, 1.0, 0.0]))
            tilt_applied = True
    if cfg.flip_facing:
        R = np.diag([-1.0, 1.0, -1.0]) @ R   # Y 軸まわりに 180 度

    if fallback:
        # 低速点が足りない: 全フレームの接地点の低いほう 10% を床とみなす
        heights = (pts.reshape(-1, 3) @ R.T)[:, 1]
        heights = heights[heights <= np.percentile(heights, 10)]
    else:
        heights = (cand[inliers] @ R.T)[:, 1]

    # 上下: 中央値から求める定数（平均値は外れ値に引っ張られるので使わない）
    y_off = np.full(T, -float(np.median(heights)))
    seg_heights = []
    if cfg.segment_mode and not fallback:
        # 区間ごとの定数。床の高さが途中で変わると全体の RANSAC のインライアは片方の高さにしか
        # 乗らないので、各区間ではその区間の低速点すべての中央値を使う
        cand_h = (cand @ R.T)[:, 1]
        seg_len = max(1, int(round(float(cfg.segment_sec) * fps)))
        bounds = [(s, min(T, s + seg_len)) for s in range(0, T, seg_len)]
        values = []
        for s, e in bounds:
            m = (frame_of >= s) & (frame_of < e)
            values.append(-float(np.median(cand_h[m])) if m.sum() >= 5 else y_off[0])
        blend = max(1.0, float(cfg.segment_blend_sec)) * fps
        y_off = _cosine_blend_curve(values, bounds, T, blend)
        seg_heights = [-v for v in values]

    # 水平: 骨盤の水平位置の原点（定数）
    pelvis = kin.root_pos @ R.T
    origin = str(cfg.horizontal_origin)
    if origin == 'first_frame':
        xz = pelvis[0, [0, 2]]
    elif origin == 'mean':
        xz = pelvis[:, [0, 2]].mean(0)
    elif origin == 'none':
        xz = np.zeros(2)
    else:
        raise ValueError(f'floor.horizontal_origin が不正です: {origin}')
    offset = np.zeros((T, 3))
    offset[:, 0], offset[:, 1], offset[:, 2] = -xz[0], y_off, -xz[1]
    return FloorResult(R, offset, normal, tilt, len(cand), int(inliers.sum()), fallback,
                       tilt_applied, spread, seg_heights)
