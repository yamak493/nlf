"""ステージ6c: 前後の傾きの補正（全身の重心が、接地している足の上に来るように体を起こす）。

単眼動画からの推定では、体がカメラに向かって前後に傾く角度がほとんど決まらない。カメラの方へ 10 度
傾けても、画像上では体の縦の長さが 1.5% 縮むだけだからである（左右の傾きは画像上でそのまま見えるので
正しく推定される）。床の傾き（ステージ4）は足のかかと・つま先から求めるので、足が床に平らに着いていても、
その上の体全体がカメラの前後に傾いたまま残る。

体の縦のライン（骨盤→首など）をフレームごとに鉛直に直すと、お辞儀や前傾姿勢などの本当の傾きまで消える。
そこで、姿勢ではなく次の物理的な条件を使う。

* 人は倒れずに立っているので、全身の重心は、接地している足（支持基底面）の上にある。お辞儀で上半身が
  前に倒れるときも、腰が後ろへ引かれて重心は足の上に残る
* 歩く・踊るなどの速い動きでは重心が足の上から一時的に外れるが、数秒の平均では足の上に戻る
  （重心と足圧中心の差は重心の水平加速度に比例するので、時間平均すると速度の変化分しか残らない）

各フレームで、全身の重心（関節の位置と体の部位の質量比から求める）と、接地している足の支持点との
カメラの奥行き方向の差を求め、数秒の窓（ガウシアンの重み）でこの差が最も小さくなる補正角を探す。

* 片足接地: 重心がその足の支持点（かかと・つま先のうち床に着いている点）の中心に来るのが最もよい
* 両足接地: 重心は両足の支持点の間のどこにあってもよい（どちらの足に体重を掛けているかは分からない）。
  支持点の範囲から外れた量を差とし、両足の中心へは弱い重みでだけ寄せる
* 差が outlier_m を超えるフレーム（接地判定の誤り・踏み切りなど）は、差を頭打ちにして影響を抑える
* 初期姿勢（直立）での重心と支持点の中心の前後の差を基準（差 0）とする
補正角は格子の上で探すので、頭打ちで損失が凸でなくても最もよい角度が見つかる。

補正は、足首から下（足の位置と向き・接地点）はそのままにして、骨盤から上の体を傾ける（骨盤から上の
大域回転をカメラの左右の軸まわりに回し、骨盤を奥行き方向に平行移動する）。脚は MMD の足ＩＫで付いてくる。
上下の位置（画像上で見えている量）と、左右の傾きは変えない。
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import filters, quat
from .body_model import Kinematics

HIPS = (1, 2)
KNEES = (4, 5)
FIXED = (7, 8, 10, 11)       # 足首・足先（足の位置と向きは変えない）
UPPER = tuple(j for j in range(24) if j not in HIPS + KNEES + FIXED)   # 骨盤から上

# 体の部位の質量比と重心の位置（de Leva 1996, 男性）。(近位の関節, 遠位の関節, 質量比, 近位からの割合)
# 関節の番号 -1 は左右の股関節の中点。体幹は上部・中部・下部に分け、首→背骨2→背骨1→股関節の中点で近似する
SEGMENTS = [
    (15, 15, 0.0694, 0.0),                                        # 頭・首（頭の関節）
    (12, 6, 0.1596, 0.2999), (6, 3, 0.1633, 0.4502), (3, -1, 0.1117, 0.6115),   # 体幹
    *[seg for side in range(2) for seg in (
        (16 + side, 18 + side, 0.0271, 0.5772),                   # 上腕
        (18 + side, 20 + side, 0.0162, 0.4574),                   # 前腕
        (20 + side, 22 + side, 0.0061, 0.7900),                   # 手
        (1 + side, 4 + side, 0.1416, 0.4095),                     # 大腿
        (4 + side, 7 + side, 0.0433, 0.4459),                     # 下腿
        (7 + side, 10 + side, 0.0137, 0.5000))],                  # 足
]


def com_weights(num_joints=24):
    """(J,) 全身の重心 = Σ 重み × 関節位置 となる関節ごとの重み（合計 1）。"""
    w = np.zeros(num_joints)

    def add(j, m):
        if j < 0:
            w[list(HIPS)] += 0.5 * m
        else:
            w[j] += m

    for a, b, m, t in SEGMENTS:
        add(a, m * (1.0 - t))
        add(b, m * t)
    return w / w.sum()


def body_com(joints):
    """(..., J, 3) の関節位置から全身の重心 (..., 3)。"""
    return np.einsum('...jc,j->...c', np.asarray(joints, np.float64), com_weights())


def horizontal_depth_direction(axis):
    """奥行きの軸（カメラの光軸）を水平面に投影した単位ベクトル。カメラが真下を向くときは Z 軸。"""
    d = np.array([axis[0], 0.0, axis[2]], np.float64)
    n = np.linalg.norm(d)
    return d / n if n > 1e-3 else np.array([0.0, 0.0, 1.0])


def rotation_about(axis, angles):
    """(T, 3, 3) 単位ベクトル axis まわりに angles (T,) [rad] 回す回転。"""
    return quat.to_matrix(quat.from_rotvec(np.asarray(angles, np.float64)[:, None] * axis))


def rest_offset(rest):
    """初期姿勢（直立）での、足の支持点の中心から全身の重心への水平のずれ (3,) [m]（体の座標: +Z が正面）。"""
    sup = np.asarray(rest.points, np.float64).mean(axis=(0, 1, 2))
    off = body_com(rest.joints) - sup
    off[1] = 0.0
    return off


def support(contact_points, flags, band):
    """接地している足の支持点。contact_points: (T, 2 足, 2 [かかと, つま先], 3) / flags: (T, 2)。

    その足のかかと・つま先のうち、その足の最下点から band 以内の点（かかとを上げていればつま先だけ）を支持点とする。
    戻り値: (点ごとに使うか (T, 2, 2) bool, 足ごとの支持点の中心 (T, 2, 3))
    """
    pts = np.asarray(contact_points, np.float64)
    h = pts[..., 1]
    use = (h <= h.min(-1, keepdims=True) + band) & np.asarray(flags, bool)[..., None]
    cnt = np.maximum(use.sum(-1), 1)[..., None]
    center = (pts * use[..., None]).sum(2) / cnt
    return use, center


@dataclass
class LeanResult:
    angle: np.ndarray        # (T,) 補正角 [rad]（+ で上半身をカメラの方へ倒す）
    shift: np.ndarray        # (T, 3) 骨盤の平行移動
    direction: np.ndarray    # (3,) カメラの奥行き方向（水平、カメラへ向かう向き）
    axis: np.ndarray         # (3,) 回転の軸（水平、カメラの左右）
    before_deg: np.ndarray   # (T,) 支持点の中心から重心への線の奥行き方向の傾き（補正前、支持なしは nan）
    after_deg: np.ndarray    # (T,) 同（補正後）
    supported: np.ndarray    # (T,) bool 補正角を求めるのに使ったフレーム
    enabled: bool
    clamped: bool = False    # 補正角が探す範囲（max_deg）の端に当たった

    def apply(self, kin):
        """骨盤から上を回して骨盤を平行移動する。足首から下（位置・向き・接地点）は変えない。

        股関節は骨盤と一緒に動き、膝は股関節の移動の半分だけ動かす（脚の位置は重心の計算にだけ使い、
        MMD では足ＩＫで付いてくる）。大腿・下腿の大域回転は変えない。
        """
        if not self.enabled:
            return kin
        R = rotation_about(self.axis, self.angle)
        G = np.array(kin.glob_rot, copy=True)
        P = np.array(kin.joints, copy=True)
        p0 = kin.joints[:, 0]
        moved = list(UPPER) + list(HIPS)
        P[:, moved] = (p0 + self.shift)[:, None] + np.einsum(
            'tab,tjb->tja', R, kin.joints[:, moved] - p0[:, None])
        for knee, hip in zip(KNEES, HIPS):
            P[:, knee] += 0.5 * (P[:, hip] - kin.joints[:, hip])
        G[:, list(UPPER)] = np.einsum('tab,tjbc->tjac', R, kin.glob_rot[:, list(UPPER)])
        return Kinematics(G, P, kin.contact_points)


def pivot_height(kin, fps):
    """(T,) 骨盤を平行移動する量（高さ × tanθ）に使う骨盤の高さ。σ 0.5 秒のガウシアンでならす。

    歩行の 1 歩ごとの骨盤の上下（倒立振子）まで追うと、平行移動が 1 歩ごとに揺れ、接地している足が骨盤に対して
    前後に動いて見える（接地判定の速度に混ざる）。しゃがむなどのゆっくりした高さの変化には付いていく。
    """
    return filters.gaussian_time(np.asarray(kin.joints[:, 0, 1], np.float64), 0.5 * fps)


def com_along(kin, d, height):
    """補正角 θ で apply したときの重心の奥行き方向の位置 = c0 + c1 tanθ + c2 cosθ + c3 sinθ の係数 (T, 4)。

    骨盤の平行移動は height × tanθ（height は pivot_height）、骨盤から上は骨盤まわりの回転、膝は股関節の
    移動の半分。
    """
    w = com_weights()
    P = np.asarray(kin.joints, np.float64)
    p0 = P[:, 0]
    x = P - p0[:, None]                                   # 骨盤からの相対位置
    moved = list(UPPER) + list(HIPS)
    xd, xy = x @ d, x[..., 1]
    w_moved, w_knee = w[moved].sum(), w[list(KNEES)].sum()
    rot_d = xd[:, moved] @ w[moved] + 0.5 * sum(w[k] * xd[:, h] for k, h in zip(KNEES, HIPS))
    rot_y = xy[:, moved] @ w[moved] + 0.5 * sum(w[k] * xy[:, h] for k, h in zip(KNEES, HIPS))
    fixed = list(FIXED) + list(KNEES)
    c0 = (P[:, fixed] @ d) @ w[fixed] + w_moved * (p0 @ d) \
        - 0.5 * sum(w[k] * xd[:, h] for k, h in zip(KNEES, HIPS))
    c1 = (w_moved + 0.5 * w_knee) * np.asarray(height, np.float64)
    return np.stack([c0, c1, rot_d, rot_y], axis=1)


def _evaluate(coef, theta):
    """coef: (T, 4) / theta: (G,) → (T, G) 重心の奥行き方向の位置。"""
    basis = np.stack([np.ones_like(theta), np.tan(theta), np.cos(theta), np.sin(theta)])
    return coef @ basis


def _refine(cost, grid):
    """(T, G) の損失の最小を格子の点から放物線で補間する。戻り値 (T,) の角度と、端に当たったか (T,)。"""
    idx = np.argmin(cost, axis=1)
    step = grid[1] - grid[0] if len(grid) > 1 else 0.0
    theta = grid[idx]
    inner = (idx > 0) & (idx < len(grid) - 1)
    t = np.flatnonzero(inner)
    if len(t):
        a, b, c = cost[t, idx[t] - 1], cost[t, idx[t]], cost[t, idx[t] + 1]
        den = a - 2.0 * b + c
        off = np.where(den > 1e-12, 0.5 * (a - c) / np.where(den > 1e-12, den, 1.0), 0.0)
        theta[t] += np.clip(off, -0.5, 0.5) * step
    return theta, ~inner & (len(grid) > 1)


def estimate_lean(kin, contact, depth_axis, rest, fps, unit, cfg, flight=None):
    """ステージ6c。kin: 床 y=0 の座標（ステージ6a の接地の拘束の後、MMD 単位）/ contact: ステージ6 の接地判定。

    rest: 体型の初期姿勢の情報（重心と支持点の基準のずれに使う）/ unit: スケール係数。
    flight: (T,) bool ジャンプとして残した滞空のフレーム（ステージ6a）。支持が無いので使わない。
    接地判定から漏れた短い区間（歩行の踏み替え・接地の始まりと終わり）は、前後の支持点の中心を直線でつないだ
    点を支持点とする（足圧中心は後ろの足から前の足へ移る）。漏れたフレームを除くと、重心が足から最も離れる
    接地の端が抜けて平均が偏る（合成の歩行で 3 度前後）。
    """
    T = len(kin.joints)
    d = horizontal_depth_direction(depth_axis)
    up = np.array([0.0, 1.0, 0.0])
    axis = np.cross(up, d)                    # この軸まわりの + の回転で、上向きが d の向きへ倒れる
    zeros = np.zeros(T)
    nan = np.full(T, np.nan)

    flags = np.asarray(contact.flags, bool)
    n_feet = flags.sum(1)
    supported = n_feet > 0
    if not cfg.enabled or supported.sum() < max(3, int(round(fps))):
        return LeanResult(zeros, np.zeros((T, 3)), d, axis, nan, nan, supported, False)

    use, centers = support(kin.contact_points, flags, float(cfg.support_band_m) * unit)
    # 足の中心（接地している足の平均）と、支持点の範囲（接地している全点の奥行きの最小・最大）
    center = (centers @ d * flags).sum(1) / np.maximum(n_feet, 1)
    along = kin.contact_points @ d
    lo = np.where(use, along, np.inf).min(axis=(1, 2))
    hi = np.where(use, along, -np.inf).max(axis=(1, 2))
    # 接地判定から漏れた短い区間は、前後の支持点の中心を直線でつなぐ
    t = np.arange(T)
    center = np.interp(t, t[supported], center[supported])
    filled = np.zeros(T, bool)
    max_gap = float(cfg.fill_gap_sec) * fps
    for s, e in filters.runs(~supported):
        if 0 < s and e < T - 1 and e - s + 1 <= max_gap:
            filled[s:e + 1] = True
    if flight is not None:
        filled &= ~np.asarray(flight, bool)
    used = supported | filled
    # 基準のずれ: 初期姿勢での「支持点の中心 → 重心」を骨盤の左右の向きに合わせて回したもの
    fwd = kin.glob_rot[:, 0, :, 2] * [1.0, 0.0, 1.0]
    fwd /= np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9)
    left = np.cross(up, fwd)
    off = rest_offset(rest) * unit
    target = (off[0] * left + off[2] * fwd) @ d
    center, lo, hi = center + target, lo + target, hi + target

    height = pivot_height(kin, fps)
    coef = com_along(kin, d, height)
    step = np.deg2rad(0.25)
    limit = np.deg2rad(float(cfg.max_deg))
    grid = np.linspace(-limit, limit, 2 * int(np.ceil(limit / step)) + 1)
    com = _evaluate(coef, grid)                                      # (T, G)
    diff = com - center[:, None]
    double = (n_feet >= 2)[:, None]
    with np.errstate(invalid='ignore'):
        outside = np.maximum(np.maximum(lo[:, None] - com, com - hi[:, None]), 0.0)
    lam = float(cfg.double_support_center_weight)
    cost = np.where(double, outside ** 2 + lam * diff ** 2, diff ** 2)
    cost = np.minimum(cost, (float(cfg.outlier_m) * unit) ** 2) * used[:, None]

    total = cost[used].mean(0)                                       # (G,) シーケンス全体
    sigma = float(cfg.window_sec) * fps
    if sigma > 0:
        # 窓の中に支持のあるフレームが無い所では、全体の平均がそのまま効く
        local = gaussian_filter1d(cost, sigma, axis=0, mode='constant')
        angle, edge = _refine(local + 0.05 * total[None], grid)
        angle = filters.gaussian_time(angle, 0.5 * fps)
    else:
        a, e = _refine(total[None], grid)
        angle, edge = np.full(T, a[0]), np.full(T, e[0])
    angle = np.clip(angle, -limit, limit)

    shift = (height * np.tan(angle))[:, None] * d
    com_height = body_com(kin.joints)[:, 1]
    com0 = coef @ np.array([1.0, 0.0, 1.0, 0.0])
    com1 = (coef * np.stack([np.ones(T), np.tan(angle), np.cos(angle), np.sin(angle)], 1)).sum(1)
    before = np.where(used, np.rad2deg(np.arctan2(com0 - center, com_height)), np.nan)
    after = np.where(used, np.rad2deg(np.arctan2(com1 - center, com_height)), np.nan)
    return LeanResult(angle, shift, d, axis, before, after, used, True, bool(edge[used].any()))
