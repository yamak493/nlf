"""ステージ8: センターの安定化と「届く高さ」へのクランプ。

水平移動はセンター、上下移動はグルーブに分けて出力する。センターをフレーム毎に持ち上げる
処理はしない（脚が届かない問題は、センターを下げる方向の補正だけで解決する）。

届く高さは、体を支えている脚（supporting_legs: 接地している脚と、足裏が床の近くにある脚）だけで判定する。
宙にある遊脚（前へ蹴り上げた脚・ジャンプ中に伸ばした脚）の足ＩＫに届かなくても、MMD の IK が脚を伸ばしきって
足が少し手前で止まるだけなので、体全体を下げない（下げると、まっすぐな脚を前へ蹴り上げるたびに体が沈み、軸足の
膝が曲がる。骨盤を下げても縮むのは距離の鉛直の成分だけなので、脚が水平に近いほど下げ幅が大きくなる）。
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

from . import filters
from .skeleton import SIDES


@dataclass
class ReachGeometry:
    """届く高さの判定に使う MMD モデルの寸法（内部座標・MMD 単位）。"""
    lower: np.ndarray        # (3,) 下半身の初期位置
    hips: np.ndarray         # (2, 3) 左右の足（股関節）の初期位置
    ik: np.ndarray           # (2, 3) 左右の足ＩＫの初期位置
    leg_length: np.ndarray   # (2,) 足→ひざ→足首の長さ

    @classmethod
    def from_skeleton(cls, skel):
        return cls(skel.internal('下半身'),
                   np.stack([skel.internal(s + '足') for s in SIDES]),
                   np.stack([skel.internal(s + '足ＩＫ') for s in SIDES]),
                   np.array([skel.leg_length(0), skel.leg_length(1)]))

    def hip_positions(self, center_delta, lower_rot):
        """(T, 2, 3) 各フレームの股関節の位置。lower_rot は下半身の大域回転 (T, 3, 3)。"""
        rel = self.hips - self.lower
        return (self.lower + center_delta[:, None]
                + np.einsum('tab,sb->tsa', lower_rot, rel))

    def distances(self, center_delta, lower_rot, ik_delta):
        hip = self.hip_positions(center_delta, lower_rot)
        return np.linalg.norm(hip - (self.ik + ik_delta), axis=-1)   # (T, 2)

    def overextended(self, center_delta, lower_rot, ik_delta, ratio, tol=1e-6, legs=None):
        """(T,) bool 脚が伸び切っているフレーム。legs: (T, 2) bool 判定する脚（None なら両脚）。"""
        d = self.distances(center_delta, lower_rot, ik_delta)
        over = d > ratio * self.leg_length + tol
        if legs is not None:
            over &= np.asarray(legs, bool)
        return over.any(axis=1)


def supporting_legs(flags, sole_heights, near, margin):
    """(T, 2) bool 届く高さの判定に使う脚（体を支えている脚）。

    flags: (T, 2) 接地判定（前後 margin フレームにも広げる。足ＩＫの境界のブレンドの間も足は床の近くにある）/
    sole_heights: (T, 2) 足ＩＫでのモデルの足裏の床からの高さ（locked.mmd_sole_heights）。near より低い脚も使う
    （接地判定から漏れても床の上にある足は、届かないと浮いて見える）。
    """
    flags = np.asarray(flags, bool)
    if margin > 0 and len(flags):
        flags = maximum_filter1d(flags.astype(np.uint8), 2 * int(margin) + 1, axis=0,
                                 mode='constant').astype(bool)
    return flags | (np.asarray(sole_heights, np.float64) < float(near))


@dataclass
class CenterResult:
    delta: np.ndarray            # (T, 3) 最終的なセンター＋グルーブの差分（内部座標）
    smoothed: np.ndarray         # (T, 3) クランプ前の差分
    raw: np.ndarray              # (T, 3) 処理前の差分
    correction_raw: np.ndarray   # (T,) フレーム毎に求めた補正量（≤ 0）
    correction: np.ndarray       # (T,) 移動最小値 → ガウシアン後の補正量（≤ 0）
    exceed_before: int
    exceed_after: int
    mode: str
    legs: np.ndarray = None      # (T, 2) bool 届く高さの判定に使った脚（None なら両脚）


def _axis_one_euro(p, fps, unit, cfg, depth_reconstructed):
    ax = cfg.axis_one_euro
    # 奥行きをステージ6b で求め直した後は、加速度の項ですでになめらか。ここで強く平滑化すると、足ＩＫを
    # 固定した位置と骨盤がずれる（前後のステップが小さくなり、膝の曲がりが変わる）
    z = ax.z if depth_reconstructed else ax.z_raw
    mc = np.array([ax.x.min_cutoff, ax.y.min_cutoff, z.min_cutoff])
    beta = np.array([ax.x.beta, ax.y.beta, z.beta])
    return filters.one_euro(p / unit, fps, mc, beta, cfg.d_cutoff, cfg.zero_phase) * unit


def _hermite(p0, p1, v0, v1, n):
    # p0 → p1 を n+1 等分した内側 n 点（端点は含まない）。v は 1 区間あたりの速度
    s = np.arange(1, n + 1) / (n + 1.0)
    h00, h10 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s
    h01, h11 = -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    L = n + 1.0
    return (h00[:, None] * p0 + h10[:, None] * v0 * L + h01[:, None] * p1
            + h11[:, None] * v1 * L)


def pelvis_from_contacts(pelvis, ankles, ik_target, contact, fps, unit, cfg):
    """モードB: 接地中は「ロックした足ＩＫ位置 + 姿勢から求めた足首→骨盤ベクトル」で骨盤を求める。"""
    T = len(pelvis)
    acc = np.zeros((T, 3))
    cnt = np.zeros(T)
    for foot in range(2):
        m = contact.flags[:, foot]
        acc[m] += ik_target[m, foot] + (pelvis[m] - ankles[m, foot])
        cnt[m] += 1
    known = cnt > 0
    if not known.any():
        return pelvis.copy()
    est = np.where(known[:, None], acc / np.maximum(cnt, 1)[:, None], np.nan)
    offset = est - pelvis     # 姿勢由来の骨盤との差
    vel = np.gradient(pelvis, axis=0)
    g = float(cfg.mode_b.gravity_m_per_s2) * unit
    max_flight = float(cfg.mode_b.max_flight_sec) * fps
    for s, e in filters.runs(~known):
        n = e - s + 1
        if s == 0 or e == T - 1:
            # 端は最寄りの差を保ったまま姿勢由来の値を使う
            ref = e + 1 if s == 0 else s - 1
            est[s:e + 1] = pelvis[s:e + 1] + offset[ref]
            continue
        a, b = s - 1, e + 1
        # 水平は 3 次（エルミート）補間
        est[s:e + 1, [0, 2]] = _hermite(est[a, [0, 2]], est[b, [0, 2]],
                                        vel[a, [0, 2]], vel[b, [0, 2]], n)
        if n <= max_flight:
            # 上下は重力加速度の放物線（両端の値を通る）
            tau = np.arange(1, n + 1) / fps
            D = (n + 1) / fps
            est[s:e + 1, 1] = (est[a, 1] + (est[b, 1] - est[a, 1]) * tau / D
                               + 0.5 * g * tau * (D - tau))
        else:
            w = np.arange(1, n + 1) / (n + 1.0)
            est[s:e + 1, 1] = pelvis[s:e + 1, 1] + (1 - w) * offset[a, 1] + w * offset[b, 1]
    fo = cfg.mode_b.final_one_euro
    return filters.one_euro(est / unit, fps, fo.min_cutoff, fo.beta, cfg.d_cutoff,
                            cfg.zero_phase) * unit


def reach_correction(geom, center_delta, lower_rot, ik_delta, ratio, max_drop, legs=None):
    """各フレームで、脚が ratio × 脚長 以内に届く最大の骨盤高さへの補正量（≤ 0）を求める。

    legs: (T, 2) bool 判定する脚（supporting_legs。None なら両脚）。判定しない脚は補正を求めない。
    """
    hip = geom.hip_positions(center_delta, lower_rot)
    v = hip - (geom.ik + ik_delta)                            # (T, 2, 3)
    limit = ratio * geom.leg_length                            # (2,)
    rhs = limit ** 2 - v[..., 0] ** 2 - v[..., 2] ** 2
    allowed = np.where(rhs >= 0, -v[..., 1] + np.sqrt(np.maximum(rhs, 0)), -v[..., 1])
    need = np.minimum(0.0, allowed)
    if legs is not None:
        need = np.where(np.asarray(legs, bool), need, 0.0)
    return np.maximum(need.min(axis=1), -max_drop)


def stabilize_center(pelvis_raw, pelvis, pelvis_rest, ankles, ik, contact, geom, lower_rot,
                     fps, unit, cfg, depth_reconstructed=True, legs=None):
    """pelvis_raw: 処理前の骨盤位置 / pelvis: ステージ2 後の骨盤位置（どちらも (T, 3)）。

    depth_reconstructed: ステージ6b で骨盤の奥行きを求め直したか（モードA の Z の平滑化の強さを選ぶ）。
    legs: (T, 2) bool 届く高さの判定に使う脚（supporting_legs。None なら両脚）。
    """
    mode = str(cfg.mode).upper()
    if mode == 'A':
        smooth = _axis_one_euro(pelvis, fps, unit, cfg, depth_reconstructed)
    elif mode == 'B':
        smooth = pelvis_from_contacts(pelvis, ankles, ik.target, contact, fps, unit, cfg)
    else:
        raise ValueError(f'center.mode は A か B です: {cfg.mode}')
    smooth_delta = smooth - pelvis_rest
    delta, corr_raw, corr, before, after = apply_reach_clamp(
        geom, smooth_delta, lower_rot, ik.delta, cfg, unit, legs)
    return CenterResult(delta, smooth_delta, pelvis_raw - pelvis_rest, corr_raw, corr, before,
                        after, mode, legs)


def apply_reach_clamp(geom, center_delta, lower_rot, ik_delta, cfg, unit, legs=None):
    """届く高さへのクランプ（両モード共通）。下げる方向の補正だけを掛ける。

    legs: (T, 2) bool 判定する脚（supporting_legs。None なら両脚）。伸び切ったフレームもこの脚だけで数える。
    戻り値: (補正後の差分, フレーム毎の補正量, 適用した補正量, 補正前の超過フレーム数, 補正後の超過フレーム数)
    """
    ratio = float(cfg.reach_ratio)
    corr_raw = reach_correction(geom, center_delta, lower_rot, ik_delta, ratio,
                                float(cfg.reach_max_drop_m) * unit, legs)
    # 普通の平滑化だけだとピークが削れて許容高さを超えるので、先に移動最小値で谷を広げる。
    # ガウシアンのカーネルを移動最小値の窓の半分で打ち切ると、平滑化後の値はどのフレームでも
    # そのフレームで必要な補正量以下（= 十分に下げる側）になる
    window = int(cfg.reach_min_window)
    corr = filters.gaussian_time(filters.moving_min(corr_raw, window),
                                 float(cfg.reach_gaussian_sigma), radius=window // 2)
    corr = np.minimum(corr, 0.0)
    delta = np.array(center_delta, np.float64, copy=True)
    delta[:, 1] += corr
    # 適用後に再判定する（脚長を超えて届かない位置や、下げ幅の上限に当たったフレームが残りうる）
    before = int(geom.overextended(center_delta, lower_rot, ik_delta, ratio, legs=legs).sum())
    after = int(geom.overextended(delta, lower_rot, ik_delta, ratio, legs=legs).sum())
    return delta, corr_raw, corr, before, after
