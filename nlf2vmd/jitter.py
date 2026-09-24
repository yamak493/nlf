"""ステージ2: 姿勢のジッター制御。

回転はクォータニオンのまま扱う（軸角やオイラー角のまま平滑化すると ±180 度の切れ目で跳ねる）。
"""
import numpy as np

from . import filters, quat

JOINT_GROUPS = {
    'torso': [0, 3, 6, 9],
    'head': [12, 15],
    'arm': [13, 14, 16, 17, 18, 19],
    'wrist': [20, 21, 22, 23],
    'leg': [1, 2, 4, 5, 7, 8, 10, 11],
}


def remove_rotation_outliers(q, fps, thresh_deg_per_s):
    """前後との角速度がどちらもしきい値を超える単独フレームを、前後の中間（slerp）で置き換える。

    前後のフレーム同士は近い（= 本当に速い動きではない）ことも条件にする。戻り値は (q, 置換マスク)。
    """
    q = np.array(q, np.float64, copy=True)
    if len(q) < 3:
        return q, np.zeros(q.shape[:-1], bool)
    th = np.deg2rad(thresh_deg_per_s)
    vel = quat.angle_between(q[:-1], q[1:]) * fps                  # (T-1, J)
    across = quat.angle_between(q[:-2], q[2:]) * fps / 2.0         # (T-2, J)
    spike = (vel[:-1] > th) & (vel[1:] > th) & (across < th)
    mid = quat.slerp(q[:-2], q[2:], np.full(spike.shape, 0.5))
    q[1:-1][spike] = mid[spike]
    mask = np.zeros(q.shape[:-1], bool)
    mask[1:-1] = spike
    return quat.make_continuous(q), mask


def group_params(num_joints, groups_cfg):
    """関節ごとの (min_cutoff, beta) を (J, 1) の配列にする。"""
    mc = np.full((num_joints, 1), groups_cfg['torso']['min_cutoff'], np.float64)
    beta = np.full((num_joints, 1), groups_cfg['torso']['beta'], np.float64)
    for name, joints in JOINT_GROUPS.items():
        idx = [j for j in joints if j < num_joints]
        mc[idx] = groups_cfg[name]['min_cutoff']
        beta[idx] = groups_cfg[name]['beta']
    return mc, beta


def stabilize_pose(quats, fps, cfg):
    """(T, J, 4) の関節回転に、符号の連続化 → 外れ値除去 → One Euro → 正規化 を掛ける。"""
    q = quat.make_continuous(quats)
    q, outliers = remove_rotation_outliers(q, fps, float(cfg.outlier_deg_per_s))
    oe = cfg.one_euro
    mc, beta = group_params(q.shape[1], oe.groups)
    q = filters.one_euro(q, fps, mc, beta, oe.d_cutoff, oe.zero_phase, vector_axis=-1)
    q = quat.make_continuous(quat.normalize(q))
    return q, dict(outlier_frames=int(outliers.any(axis=1).sum()),
                   outlier_joint_frames=int(outliers.sum()))


def stabilize_root(root_pos, window):
    """ルート移動量はここではメディアンでスパイクを除くだけ（本格的な安定化はステージ8）。"""
    return filters.median_time(root_pos, int(window))


def angular_acceleration(q, fps):
    """各関節の角加速度の大きさ [deg/s^2] を (T-2, J) で返す（診断用）。"""
    q = quat.make_continuous(q)
    if len(q) < 3:
        return np.zeros((0,) + q.shape[1:-1])
    rel = quat.mul(quat.conj(q[:-1]), q[1:])       # 親から見た 1 フレームの回転
    omega = quat.to_rotvec(rel) * fps              # [rad/s]
    acc = np.linalg.norm(np.diff(omega, axis=0), axis=-1) * fps
    return np.rad2deg(acc)
