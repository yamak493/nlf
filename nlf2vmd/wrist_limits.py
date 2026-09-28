"""ステージ9c: 手首の可動域（前腕の回内・回外と、手首の曲げを人の関節の範囲に収める）。

NLF・MediaPipe の推定やステージ9b の補正で、手が人の関節では届かない向きになることがある（手のひらの表裏を取り違えて
前腕が 180 度近くひねれ、前腕の下側に手の甲が来る等）。前腕（ひじ）に対する手の向きを

  * 前腕のひねり（回外が +、回内が −）: ひじ のひねり + 手首 のひねり（どちらも前腕の軸まわり。手首の関節そのものは
    ひねれないので、手の前腕に対するひねりは前腕の回内・回外）
  * 手首の曲げ（掌屈が +・背屈が −、橈屈が +・尺屈が −）: 手首 の残りの回転（曲げ）を、ひねった後の手の座標系で
    測ったもの（手のひら側へ曲げるのが掌屈。手のひらがどちらを向いていても同じ）

に分け、それぞれを可動域（既定は日本整形外科学会の参考可動域）に収める。曲げは、掌屈・背屈と橈屈・尺屈を半軸にした
4 つの 1/4 楕円の中に収める（斜めに曲げたときも角の所まで曲がらない）。ひじ の回転は変えず、手首 の回転だけを直す
（捩りボーンのあるモデルでは、あとで前腕のひねりは 手捩 に移る）。

中立（ひねり 0・曲げ 0）は初期姿勢の手首（SMPL の T ポーズ・MMD の A ポーズとも、手のひらが下〜体の側を向いて親指が前。
解剖学の中立 = ひじを 90 度曲げたとき親指が上）。手のひらの向きは、前腕の軸・正面（+Z）から左右の鏡像で決める
（ステージ9 のリターゲットも、初期姿勢の手首をこの向きとみなしている）。

時間方向の扱い:
  * 範囲の外のひねりは、上限・下限のどちらへ戻すかで 170 度以上違う。前のフレームの値に近いほうを優先し
    （範囲の外を通って裏側へ回った推定で、上限と下限の間を行き来しない）、範囲に戻ったフレームで推定の値に戻す
  * 1 フレームの変化を max_speed_deg_per_s までにする（上限 → 下限へ移るときは、手のひらを中立の側から返す。
    範囲は凸なので、速さを抑えた途中の値も範囲の中）。変化がそれより遅いフレームは変えない
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import maximum_filter1d

from . import quat
from .filters import rate_limit
from .skeleton import SIDES

FRONT = np.array([0.0, 0.0, 1.0])      # 内部座標の正面（初期姿勢の親指の向き）
JUMP_WEIGHT = 0.5                      # 範囲の外のひねりを戻す先を選ぶとき、前のフレームからの跳びに掛ける重み
_EPS = 1e-9


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _wrap(x):
    return (np.asarray(x, np.float64) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class HandAxes:
    """初期姿勢の手の軸（内部座標の単位ベクトル）。"""
    forearm: np.ndarray    # 前腕の軸（ひじ → 手首）
    palm: np.ndarray       # 手のひら側の法線（前腕の軸に垂直）
    thumb: np.ndarray      # 親指側（前腕の軸・手のひらの法線に垂直）
    sign: float            # 前腕の軸まわりの回転の向き → 回外が + になる符号

    @property
    def flexion(self):
        """この軸まわりの + の回転で、手が手のひら側へ曲がる（掌屈）。"""
        return np.cross(self.forearm, self.palm)

    @property
    def radial(self):
        """この軸まわりの + の回転で、手が親指側へ曲がる（橈屈）。"""
        return np.cross(self.forearm, self.thumb)


def hand_axes(elbow, wrist, side):
    """初期姿勢のひじ・手首の位置（内部座標）と腕（0 = 左 / 1 = 右）から HandAxes。

    親指は正面（前腕の軸に垂直にしたもの）、手のひらの法線は 左手 = 前腕 × 親指 ・右手 = 親指 × 前腕
    （左腕を +X へ水平に伸ばすと、親指が前（+Z）で手のひらが下（−Y））。回外は手のひらを親指側へ返す向き。
    """
    a = _unit(np.asarray(wrist, np.float64) - np.asarray(elbow, np.float64))
    thumb = FRONT - (FRONT @ a) * a
    if np.linalg.norm(thumb) < 1e-6:           # 前腕が前後を向いている初期姿勢（ほぼ無い）
        thumb = np.array([0.0, 1.0, 0.0]) - a[1] * a
    thumb = _unit(thumb)
    side_sign = -1.0 if side == 0 else 1.0
    palm = _unit(side_sign * np.cross(thumb, a))
    # 前腕の軸まわりの +θ は、手のひらの法線を (前腕 × 法線) の側へ回す。回外は法線を親指側へ回す向き
    return HandAxes(a, palm, thumb, float(np.sign(np.cross(a, palm) @ thumb)))


def _split_twist(q, axis):
    """q = 曲げ · ひねり（ひねりは axis まわり）の (ひねりの角 [rad]（−π〜π）, 曲げ (T, 4))。"""
    q = quat.normalize(q)
    theta = _wrap(2.0 * np.arctan2(q[..., :3] @ axis, q[..., 3]))
    swing = quat.mul(q, quat.conj(quat.from_rotvec(theta[..., None] * axis)))
    return theta, swing


def _decompose(elbow_q, wrist_q, ax):
    """(ひじのひねり [rad], 回外 [rad], 掌屈 [rad], 橈屈 [rad]) それぞれ (T,)。"""
    a = ax.forearm
    t_elbow, _ = _split_twist(elbow_q, a)
    t_wrist, swing = _split_twist(wrist_q, a)
    # 曲げを、手首のひねりの後の手の座標系で測る（手首 = 曲げ · ひねり = ひねり · (ひねり⁻¹ · 曲げ · ひねり)）
    v = quat.rotate(quat.from_rotvec(-t_wrist[:, None] * a), quat.to_rotvec(swing))
    return t_elbow, ax.sign * _wrap(t_elbow + t_wrist), v @ ax.flexion, v @ ax.radial


def wrist_angles(elbow_q, wrist_q, ax):
    """(T, 3) [度] 回外（+）/回内（−）・掌屈（+）/背屈（−）・橈屈（+）/尺屈（−）。elbow_q・wrist_q はひじ・手首の
    ローカル回転 (T, 4)（親に対する回転。どちらも初期姿勢で単位回転）。"""
    _, sup, flex, rad = _decompose(np.asarray(elbow_q, np.float64), np.asarray(wrist_q, np.float64), ax)
    return np.rad2deg(np.stack([sup, flex, rad], axis=-1))


def _limits(cfg, margin_deg=0.0):
    """ラジアンの (回内, 回外, 掌屈, 背屈, 橈屈, 尺屈)。"""
    names = ('pronation_deg', 'supination_deg', 'flexion_deg', 'extension_deg', 'radial_deg', 'ulnar_deg')
    return tuple(np.deg2rad(max(float(getattr(cfg, n)) + float(margin_deg), 1e-3)) for n in names)


def _swing_excess(flex, rad, lim):
    """曲げ (flex, rad) が、4 つの 1/4 楕円の範囲の何倍の所にあるか（1 以下なら範囲の中）。"""
    _, _, f_pos, f_neg, r_pos, r_neg = lim
    F = np.where(flex >= 0.0, f_pos, f_neg)
    R = np.where(rad >= 0.0, r_pos, r_neg)
    return np.hypot(flex / F, rad / R)


def within_limits(angles_deg, cfg, margin_deg=0.0):
    """(T,) wrist_angles の値が、可動域（各上限に margin_deg を足したもの）に入っているか。"""
    lim = _limits(cfg, margin_deg)
    sup, flex, rad = np.deg2rad(np.moveaxis(np.asarray(angles_deg, np.float64), -1, 0))
    tol = 1e-7
    return (sup >= -lim[0] - tol) & (sup <= lim[1] + tol) & (_swing_excess(flex, rad, lim) <= 1.0 + tol)


def _clamp_twist(sup, lo, hi):
    """回外の角の列 (T,)（−π〜π）を [lo, hi] に収める。範囲の外のフレームは、上限（hi へ戻す）と下限（lo へ進める）の
    うち、戻す角 + JUMP_WEIGHT × 前のフレームからの跳び が小さいほうにする。"""
    out = np.empty_like(sup)
    prev = None
    for t, x in enumerate(sup):
        if lo <= x <= hi:
            out[t] = prev = x
            continue
        to_hi = (x - hi) % (2.0 * np.pi)
        to_lo = (lo - x) % (2.0 * np.pi)
        if prev is not None:
            to_hi += JUMP_WEIGHT * abs(hi - prev)
            to_lo += JUMP_WEIGHT * abs(lo - prev)
        out[t] = prev = hi if to_hi <= to_lo else lo
    return out


def _changed(sup, flex, rad, sup_c, flex_c, rad_c):
    return (np.abs(_wrap(sup_c - sup)) > _EPS) | (np.abs(flex_c - flex) > _EPS) \
        | (np.abs(rad_c - rad) > _EPS)


def clamp_wrist(elbow_q, wrist_q, ax, cfg, fps):
    """手首のローカル回転 (T, 4) を可動域に収めたものと、直したフレーム (T,) bool。ひじの回転は変えない。"""
    elbow_q = np.asarray(elbow_q, np.float64)
    wrist_q = np.asarray(wrist_q, np.float64)
    t_elbow, sup, flex, rad = _decompose(elbow_q, wrist_q, ax)
    lim = _limits(cfg)
    sup_c = _clamp_twist(sup, -lim[0], lim[1])
    scale = 1.0 / np.maximum(_swing_excess(flex, rad, lim), 1.0)
    flex_c, rad_c = flex * scale, rad * scale
    changed = _changed(sup, flex, rad, sup_c, flex_c, rad_c)
    speed = float(cfg.max_speed_deg_per_s)
    if speed > 0 and len(sup) > 1 and changed.any():
        # 速さを抑えるのは、範囲に収めたフレームと、そこから上限 → 下限へ移り切れる所まで（範囲の中を動いている
        # だけのフレームは、速くてもそのまま）
        step = np.deg2rad(speed) / fps
        reach = int(np.ceil((lim[0] + lim[1]) / step)) + 1
        near = maximum_filter1d(changed.astype(np.uint8), 2 * reach + 1, mode='constant') > 0
        sup_c = np.where(near, rate_limit(sup_c[:, None], step)[:, 0], sup_c)
        bend = rate_limit(np.stack([flex_c, rad_c], axis=-1), step)
        flex_c, rad_c = np.where(near, bend[:, 0], flex_c), np.where(near, bend[:, 1], rad_c)
        changed = _changed(sup, flex, rad, sup_c, flex_c, rad_c)
    if not changed.any():
        return wrist_q.copy(), changed
    # 手首 = (ひじのひねりを戻す) · (前腕のひねり) · (ひねった後の手の座標系での曲げ)
    a = ax.forearm
    twist = quat.from_rotvec((ax.sign * sup_c - t_elbow)[:, None] * a)
    bend = quat.from_rotvec(flex_c[:, None] * ax.flexion + rad_c[:, None] * ax.radial)
    out = np.where(changed[:, None], quat.mul(twist, bend), wrist_q)
    return quat.make_continuous(out), changed


@dataclass
class WristLimitResult:
    enabled: bool
    angles_before: np.ndarray    # (T, 2 腕, 3) [度] 回外・掌屈・橈屈（wrist_angles）
    angles_after: np.ndarray
    changed: np.ndarray          # (T, 2) 直したフレーム
    correction_deg: np.ndarray   # (T, 2) 手首のローカル回転に掛けた補正の角度 [度]
    info: dict = field(default_factory=dict)


def limit_wrists(skel, local, cfg, fps):
    """ステージ9c。local（ボーン名 → ローカル回転 (T, 4)。捩りボーンに分ける前）の 手首 を可動域に収めた dict と
    WristLimitResult を返す。ひじ・手首のキーの無い腕はそのまま。"""
    T = len(next(iter(local.values()))) if local else 0
    before = np.full((T, 2, 3), np.nan)
    after = before.copy()
    changed = np.zeros((T, 2), bool)
    corr = np.zeros((T, 2))
    out = dict(local)
    for side, s in enumerate(SIDES):
        e, w = s + 'ひじ', s + '手首'
        if e not in local or w not in local or not (skel.has(e) and skel.has(w)):
            continue
        ax = hand_axes(skel.internal(e), skel.internal(w), side)
        before[:, side] = wrist_angles(local[e], local[w], ax)
        if cfg.enabled and T:
            out[w], changed[:, side] = clamp_wrist(local[e], local[w], ax, cfg, fps)
            corr[:, side] = np.rad2deg(quat.angle_between(out[w], local[w]))
        after[:, side] = wrist_angles(local[e], out[w], ax)
    outside = ~within_limits(np.nan_to_num(before), cfg) & np.isfinite(before).all(-1)
    info = dict(enabled=bool(cfg.enabled),
                out_of_range_frames=[int(outside[:, s].sum()) for s in range(2)],
                changed_frames=[int(changed[:, s].sum()) for s in range(2)],
                max_correction_deg=[round(float(corr[:, s].max(initial=0.0)), 2) for s in range(2)],
                limits_deg={k: float(getattr(cfg, k)) for k in (
                    'pronation_deg', 'supination_deg', 'flexion_deg', 'extension_deg', 'radial_deg',
                    'ulnar_deg')})
    return out, WristLimitResult(bool(cfg.enabled), before, after, changed, corr, info)
