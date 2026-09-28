"""ステージ2: 姿勢のジッター制御。

回転はクォータニオンのまま扱う（軸角やオイラー角のまま平滑化すると ±180 度の切れ目で跳ねる）。

One Euro フィルタは関節ごとに掛けるので、腕では鎖骨・肩・ひじのそれぞれの平滑化の誤差が手首の位置で足し合わさる。
特に、手首が胸に対して止まったまま関節だけが動く（ひじを張る・肩をすくめる等、関節の動きが手首では打ち消し合う）
ときは、関節ごとの平滑化で打ち消し合いが崩れ、止まっているはずの手首が揺れる（胸・顔・もう一方の手に当てた手が
浮く・食い込む）。そこで腕だけは、平滑化する前の姿勢から求めた手首の位置（背骨3 の座標系）の軌跡にも One Euro
フィルタを掛け、手首が遅いフレームでは、肩・ひじの回転を少しだけ変えて手首をその軌跡に合わせる
（hold_hand_positions）。手首が速く動いているフレームは関節ごとの平滑化のままにする（位置の軌跡の平滑化は、
速い動きでは関節ごとの平滑化より強く掛かり、手先の動きが小さくなるため）。
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
ARM_CHAINS = ((9, 13, 16, 18, 20), (9, 14, 17, 19, 21))   # 左・右の [背骨3, 鎖骨, 肩, ひじ, 手首]
# 肩 → ひじ → 手首 の曲がりの正弦がこの範囲で、ひじの曲げを変える割合を 0 → 1 にする（まっすぐな腕は曲げる向きが決まらない）
STRAIGHT_SIN = (np.sin(np.deg2rad(3.0)), np.sin(np.deg2rad(10.0)))


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


def stabilize_pose(quats, fps, cfg, rest_joints=None):
    """(T, J, 4) の関節回転に、符号の連続化 → 外れ値除去 → One Euro → 正規化 を掛ける。

    rest_joints（SMPL の初期姿勢の関節 (J, 3) [m]）を渡すと、腕は手首の位置の軌跡にも合わせる（hold_hand_positions）。
    """
    raw = quat.make_continuous(quats)
    raw, outliers = remove_rotation_outliers(raw, fps, float(cfg.outlier_deg_per_s))
    oe = cfg.one_euro
    mc, beta = group_params(raw.shape[1], oe.groups)
    q = filters.one_euro(raw, fps, mc, beta, oe.d_cutoff, oe.zero_phase, vector_axis=-1)
    q = quat.make_continuous(quat.normalize(q))
    info = dict(outlier_frames=int(outliers.any(axis=1).sum()),
                outlier_joint_frames=int(outliers.sum()))
    hp = cfg.hand_position
    if rest_joints is not None and hp.enabled and raw.shape[1] > 21 and len(raw) > 2:
        q, hand_info = hold_hand_positions(raw, q, rest_joints, fps, hp, oe)
        info.update(hand_info)
    return q, info


def hand_positions(q, rest_joints, chain):
    """(T, 3) 手首の位置（背骨3 の座標系で、背骨3 からの相対位置）。q は親に対する回転 (T, J, 4)。"""
    spine, collar, shoulder, elbow, wrist = chain
    J = np.asarray(rest_joints, np.float64)
    R = quat.to_matrix(q[:, collar])
    p = (J[collar] - J[spine]) + R @ (J[shoulder] - J[collar])
    R = R @ quat.to_matrix(q[:, shoulder])
    p = p + R @ (J[elbow] - J[shoulder])
    R = R @ quat.to_matrix(q[:, elbow])
    return p + R @ (J[wrist] - J[elbow])


def _turn_towards(a, b, max_angle):
    """(T, 3, 3) 向き a を b へ回す最小の回転（角は max_angle まで）。"""
    axis = np.cross(a, b)
    sin = np.linalg.norm(axis, axis=-1)
    angle = np.minimum(np.arctan2(sin, np.sum(a * b, -1)), max_angle)
    rotvec = axis / np.maximum(sin, 1e-12)[:, None] * angle[:, None]
    return quat.to_matrix(quat.from_rotvec(rotvec))


def reach_hand_targets(q, targets, rest_joints, max_deg):
    """肩・ひじの回転を少しだけ変えて、手首を targets（背骨3 の座標系 (T, 2 [左, 右], 3)）へ寄せる。

    1. ひじ: 肩 → 手首の距離が目標と同じになるように、上腕と前腕を含む面の中で曲げを変える
       （腕がほぼまっすぐなフレームは曲げる向きが決まらないので変えない）
    2. 肩: 肩 → 手首の向きを目標へ向ける最小の回転
    手首の大域回転（手の向き）は変えない。どちらの補正も max_deg [度] まで。
    戻り値は (回転, (T, 2 [左, 右]) 肩・ひじの補正角の大きいほう [度])。
    """
    q = np.array(q, np.float64, copy=True)
    J = np.asarray(rest_joints, np.float64)
    limit = np.deg2rad(float(max_deg))
    corrections = np.zeros(targets.shape[:2])
    for side, (spine, collar, shoulder, elbow, wrist) in enumerate(ARM_CHAINS):
        Rc, Rs, Re, Rw = (quat.to_matrix(q[:, j]) for j in (collar, shoulder, elbow, wrist))
        start = (J[collar] - J[spine]) + Rc @ (J[shoulder] - J[collar])
        goal = np.einsum('tba,tb->ta', Rc, targets[:, side] - start)      # 鎖骨の座標系での 肩 → 手首
        upper, fore = J[elbow] - J[shoulder], J[wrist] - J[elbow]
        lu, lf = np.linalg.norm(upper), np.linalg.norm(fore)
        f = Re @ fore                                                       # 肩の座標系での前腕
        normal = np.cross(upper, f)
        sin = np.linalg.norm(normal, axis=-1) / (lu * lf)
        bend = np.arctan2(sin, f @ upper / (lu * lf))                       # 0 = 腕がまっすぐ
        dist = np.clip(np.linalg.norm(goal, axis=-1), abs(lu - lf) + 1e-6, lu + lf - 1e-6)
        want = np.arccos(np.clip((dist ** 2 - lu ** 2 - lf ** 2) / (2.0 * lu * lf), -1.0, 1.0))
        lo, hi = STRAIGHT_SIN
        delta = np.clip(want - bend, -limit, limit) * filters.smoothstep((sin - lo) / (hi - lo))
        unit_normal = normal / np.maximum(np.linalg.norm(normal, axis=-1), 1e-12)[:, None]
        Re2 = quat.to_matrix(quat.from_rotvec(unit_normal * delta[:, None])) @ Re
        turn = _turn_towards(np.einsum('tab,tb->ta', Rs, upper + Re2 @ fore), goal, limit)
        Rs2 = turn @ Rs
        Rw2 = np.swapaxes(Rs2 @ Re2, -1, -2) @ Rs @ Re @ Rw
        for j, R in ((shoulder, Rs2), (elbow, Re2), (wrist, Rw2)):
            q[:, j] = quat.from_matrix(R)
        corrections[:, side] = np.rad2deg(np.maximum(
            np.abs(delta), quat.angle_between(quat.from_matrix(turn), quat.IDENTITY)))
    return quat.make_continuous(q), corrections


def hold_hand_positions(raw, smoothed, rest_joints, fps, cfg, oe):
    """手首が遅いフレームで、関節ごとに平滑化した腕（smoothed）の手首を、平滑化する前の姿勢（raw）の手首の位置の
    軌跡を平滑化したものへ寄せる。

    手首の位置の軌跡（背骨3 の座標系 [m]）に One Euro（cfg.min_cutoff / cfg.beta）を掛け、その速さが
    cfg.speed_m_per_s の下限以下なら軌跡に合わせ、上限以上なら関節ごとの平滑化のまま（間はなめらかに混ぜる）。
    戻り値は (回転, 情報 dict)。
    """
    T = len(raw)
    targets = np.empty((T, 2, 3))
    weights = np.empty((T, 2))
    shift = np.empty((T, 2))
    lo, hi = (float(v) for v in cfg.speed_m_per_s)
    for side, chain in enumerate(ARM_CHAINS):
        path = filters.one_euro(hand_positions(raw, rest_joints, chain), fps,
                                float(cfg.min_cutoff), float(cfg.beta), oe.d_cutoff, oe.zero_phase,
                                vector_axis=-1)
        speed = np.linalg.norm(np.gradient(path, axis=0), axis=-1) * fps
        w = filters.gaussian_time(1.0 - filters.smoothstep((speed - lo) / (hi - lo)),
                                  float(cfg.blend_sec) * fps)
        current = hand_positions(smoothed, rest_joints, chain)
        targets[:, side] = current + w[:, None] * (path - current)
        weights[:, side] = w
        shift[:, side] = np.linalg.norm(targets[:, side] - current, axis=-1)
    q, corrections = reach_hand_targets(smoothed, targets, rest_joints, cfg.max_deg)
    return q, dict(hand_hold_ratio=[round(float(v), 3) for v in (weights > 0.5).mean(0)],
                   hand_hold_shift_cm=round(float(shift.max(initial=0.0) * 100.0), 2),
                   hand_hold_max_deg=round(float(corrections.max(initial=0.0)), 2))


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
