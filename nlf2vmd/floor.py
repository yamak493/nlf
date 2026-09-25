"""ステージ4: 床面推定と定数オフセット。

床の高さの補正は、シーケンス全体で 1 つの定数（区間モードでも区間ごとの定数）とする。
フレーム毎に変わる補正は入れない（入れると接地判定やロック値に歪みが混ざる）。
"""
from dataclasses import dataclass, field

import numpy as np

from . import quat
from .body_model import ANKLES


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


def floor_vectors(points, low, rest_points):
    """同じフレームの低速の接地点どうしを結ぶベクトルと、足裏が床に平らに着いているときのその上下成分。

    points: (T, 2 足, 2 [かかと, つま先], 3) / low: (T, 2, 2) bool / rest_points: (2, 2, 3) 初期姿勢
    （足裏が床に平ら）での接地点。かかと→つま先（その足の 2 点が低速）と、左足→右足の中点（4 点とも低速）。
    体全体の平行移動は差で消えるので、単眼推定の位置のずれ（視線に沿った奥行きのずれなど）が入らない。
    戻り値: (ベクトル (N, 3), 上下成分の目標値 (N,))
    """
    rest = np.asarray(rest_points, np.float64)
    vecs, targets = [], []
    for foot in range(2):
        m = low[:, foot].all(-1)
        vecs.append(points[m, foot, 1] - points[m, foot, 0])
        targets.append(np.full(m.sum(), rest[foot, 1, 1] - rest[foot, 0, 1]))
    m = low.reshape(len(low), -1).all(-1)
    mid, rest_mid = points[m].mean(2), rest.mean(1)
    vecs.append(mid[:, 1] - mid[:, 0])
    targets.append(np.full(m.sum(), rest_mid[1, 1] - rest_mid[0, 1]))
    return np.concatenate(vecs).reshape(-1, 3), np.concatenate(targets)


def sole_normals(glob_rot, low, rest_points):
    """(N, 3) 接地中（その足の 2 点が低速）の足裏の上向き。初期姿勢の足裏の上向きを足首の大域回転で回したもの。

    glob_rot: (T, 2 足, 3, 3) 足首の大域回転。初期姿勢の足裏の上向きは、+Y からかかと→つま先の成分を除いたもの。
    """
    rest = np.asarray(rest_points, np.float64)
    out = []
    for foot in range(2):
        d = rest[foot, 1] - rest[foot, 0]
        d /= max(np.linalg.norm(d), 1e-9)
        up = np.array([0.0, 1.0, 0.0]) - d[1] * d
        up /= np.linalg.norm(up)
        out.append(glob_rot[low[:, foot].all(-1), foot] @ up)
    return np.concatenate(out).reshape(-1, 3)


def _tukey_weights(r, c=4.685, min_scale=np.sin(np.deg2rad(1.0))):
    """Tukey の重み。スケールは残差の絶対値の中央値から求める（揺れの大きさに合わせて外れ値の幅が決まる）。"""
    scale = max(1.4826 * float(np.median(np.abs(r))), min_scale) if len(r) else min_scale
    u = np.asarray(r) / (c * scale)
    return np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)


def fit_tilt_vectors(vecs, targets, min_spread, normals=None, normal_weight=0.0, iterations=20):
    """床と平行なはずのベクトルから床の法線を求める（外れたものの重みを下げながら最小二乗）。

    法線を n ∝ (a, 1, b) とおくと、条件 n·v = c（c: 平らな足裏での上下成分）は a, b について線形になる。
    接地中の足裏の上向き u（normals）も、弱い重み normal_weight [m]（その長さのベクトル 1 本分）で
    「n ∝ u」の条件として加える。足の向きの推定はベクトルより不確かなので、ベクトルで決まる方向はベクトルが
    優先され、ベクトルでは決まらない方向（例: 横向きのまま横へ歩き、かかと→つま先がすべて同じ向き）を補う。
    足裏の向きを使わないとき（normal_weight=0）は、ベクトルの水平方向の広がり（主軸ごとの二乗平均平方根）が
    2 方向とも min_spread 以上なら a, b の両方を、1 方向だけならその方向の傾きだけを求める
    （もう 1 方向の傾きは 0 のまま = 補正しない）。
    外れたもの（つま先立ち・片足を上げて止めている等）は、角度の残差に Tukey の重みを掛けて除く（反復重み付き
    最小二乗）。スケールは残差の中央値から求めるので、足の向きの推定の揺れが大きくても正しいベクトルを捨てすぎない
    （揺れより狭い固定のしきい値で除くと、残ったベクトルが少なく偏り、傾きの推定が数度ばらつく）。
    戻り値: (法線, ベクトルの重みが 0 でない bool, 'full' | 'line' | 'none',
            水平方向の広がり (2,) [小さいほう, 大きいほう], 重みが 0 でない足裏の向きの数)
    """
    vecs, targets = np.asarray(vecs, np.float64), np.asarray(targets, np.float64)
    normals = np.zeros((0, 3)) if normals is None or normal_weight <= 0 else         np.asarray(normals, np.float64)
    normals = normals[normals[:, 1] > 0.5]      # 60 度以上傾いた足裏は床に着いていない
    wv, wn = np.ones(len(vecs)), np.ones(len(normals))
    g = np.zeros(2)
    mode, spread = 'none', np.zeros(2)
    length = np.maximum(np.linalg.norm(vecs, axis=1), 1e-9)
    slopes = normals[:, [0, 2]] / normals[:, 1:2]
    for _ in range(int(iterations)):
        H = vecs[:, [0, 2]] * wv[:, None]
        if (wv > 0).sum() >= 3:
            w, U = np.linalg.eigh(H.T @ H / wv.sum())
            spread = np.sqrt(np.maximum(w, 0.0))
        rhs = (targets * np.sqrt(1.0 + g @ g) - vecs[:, 1]) * wv
        if (wn > 0).sum() >= 3:
            wn2 = np.repeat(normal_weight * wn, 2)
            A = np.concatenate([H, wn2[:, None] * np.tile(np.eye(2), (len(normals), 1))])
            b = np.concatenate([rhs, wn2 * slopes.reshape(-1)])
            g = np.linalg.lstsq(A, b, rcond=None)[0]
            mode = 'full'
        elif (wv > 0).sum() >= 3 and spread[0] >= min_spread:
            g = np.linalg.lstsq(H, rhs, rcond=None)[0]
            mode = 'full'
        elif (wv > 0).sum() >= 3 and spread[1] >= min_spread:
            e = U[:, 1]
            g = np.linalg.lstsq((H @ e)[:, None], rhs, rcond=None)[0][0] * e
            mode = 'line'
        else:
            mode, g = 'none', np.zeros(2)
            break
        n = np.array([g[0], 1.0, g[1]]) / np.sqrt(1.0 + g @ g)
        wv = _tukey_weights((vecs @ n - targets) / length)
        wn = _tukey_weights(np.linalg.norm(np.cross(normals, n), axis=1))
    n = np.array([g[0], 1.0, g[1]]) / np.sqrt(1.0 + g @ g)
    return n, wv > 0, mode, spread, int((wn > 0).sum())


@dataclass
class FloorResult:
    rotation: np.ndarray                 # (3, 3) 床の傾き補正（＋向きの反転）
    offset: np.ndarray                   # (T, 3) 回転後に足す量（上下は定数・区間定数、水平は定数）
    normal: np.ndarray                   # 推定した床の法線（補正前）
    tilt_deg: float                      # 推定した床の傾き
    num_points: int
    num_inliers: int                     # plane: 平面のインライアの点数 / vectors: 使ったベクトルの数
    fallback: bool                       # 候補点が足りず、傾きを推定しなかった
    tilt_applied: bool = False           # 傾きの補正を実際に掛けたか（1 方向だけの補正も含む）
    spread: float = 0.0                  # 候補点の平面内の広がり [m]（vectors: ベクトルの広がりの小さいほう）
    tilt_mode: str = 'none'              # full（床全体）| line（1 方向の傾きだけ）| none
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


def estimate_floor(kin, fps, cfg, rest_points=None):
    """かかと・つま先の低速点から床を推定し、床が y=0・法線が +Y になる変換を返す。

    rest_points: (2 足, 2 [かかと, つま先], 3) 初期姿勢での接地点（tilt_method=vectors で、足裏が平らなときの
    かかと→つま先の上下成分に使う）。None なら同じ高さとみなす。
    """
    T = len(kin.contact_points)
    pts = kin.contact_points.reshape(T, -1, 3)   # (T, 4, 3)
    speed = horizontal_speed(pts, fps)
    low = speed < float(cfg.low_speed_m_per_s)
    cand = pts[low]
    frame_of = np.nonzero(low)[0]
    rng = np.random.default_rng(int(cfg.seed))

    R = np.eye(3)
    normal = np.array([0.0, 1.0, 0.0])
    inliers = np.ones(len(cand), bool)
    fallback = len(cand) < int(cfg.min_points)
    tilt, spread, tilt_mode, num_inliers = 0.0, 0.0, 'none', len(cand)
    method = str(cfg.tilt_method)
    if not fallback and method == 'vectors':
        rest = np.zeros((2, 2, 3)) if rest_points is None else rest_points
        low4 = low.reshape(T, 2, 2)
        vecs, targets = floor_vectors(kin.contact_points, low4, rest)
        normals = sole_normals(kin.glob_rot[:, ANKLES], low4, rest)
        normal, vec_inl, mode, vec_spread, _ = fit_tilt_vectors(
            vecs, targets, cfg.min_vector_spread_m, normals, float(cfg.sole_normal_weight_m))
        tilt, spread = float(np.rad2deg(np.arccos(np.clip(normal[1], -1, 1)))), vec_spread[0]
        num_inliers = int(vec_inl.sum())
        if cfg.align_tilt and mode != 'none' and tilt <= float(cfg.max_tilt_deg):
            R = quat.to_matrix(quat.from_two_vectors(normal, [0.0, 1.0, 0.0]))
            tilt_mode = mode
    elif not fallback and method == 'plane':
        normal, inliers, spread = fit_plane_ransac(cand, cfg.ransac_iterations,
                                                   cfg.ransac_inlier_m, rng, cfg.max_tilt_deg)
        num_inliers = int(inliers.sum())
        tilt = float(np.rad2deg(np.arccos(np.clip(normal[1], -1, 1))))
        # 候補点がほぼ一直線に並ぶ（例: まっすぐ歩くだけ）と、その線まわりの傾きは決まらない。
        # 平面内に十分な広がりがあるときだけ傾きを補正する
        if cfg.align_tilt and tilt <= float(cfg.max_tilt_deg) and \
                spread >= float(cfg.min_spread_m):
            R = quat.to_matrix(quat.from_two_vectors(normal, [0.0, 1.0, 0.0]))
            tilt_mode = 'full'
    elif not fallback:
        raise ValueError(f'floor.tilt_method は vectors か plane です: {method}')
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
    return FloorResult(R, offset, normal, tilt, len(cand), num_inliers, fallback,
                       tilt_mode != 'none', spread, tilt_mode, seg_heights)
