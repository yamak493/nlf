"""フィルタ: 震えが減り遅延が許容内であること。クォータニオンの符号反転で跳ねないこと。
腕: 手首が止まっている間は手首の位置の軌跡に合わせ、速く動く腕は関節ごとの平滑化のままにすること。"""
import numpy as np
import pytest

from nlf2vmd import load_config, quat
from nlf2vmd.filters import gaussian_time, moving_min, one_euro
from nlf2vmd.jitter import remove_rotation_outliers, stabilize_pose

from .conftest import rotation_angles

FPS = 30.0


def _jitter(x):
    return float(np.abs(np.diff(x, 2, axis=0)).mean())


def _best_lag(y, ref, max_lag=6):
    lags = np.arange(-max_lag, max_lag + 1)
    core = slice(max_lag, len(ref) - max_lag)
    scores = [np.dot(np.roll(y, -lag)[core], ref[core]) for lag in lags]
    return int(lags[int(np.argmax(scores))])


def test_one_euro_reduces_jitter_without_delay():
    rng = np.random.default_rng(0)
    t = np.arange(300) / FPS
    clean = np.sin(2 * np.pi * 1.0 * t)
    noisy = clean + rng.normal(0, 0.05, len(t))
    out = one_euro(noisy, FPS, min_cutoff=1.5, beta=0.5)
    # 震え = 正解との差の 2 階差分（正弦波そのものの曲率は含めない）
    assert _jitter(out - clean) < 0.5 * _jitter(noisy - clean)
    assert np.sqrt(np.mean((out - clean) ** 2)) < np.sqrt(np.mean((noisy - clean) ** 2))
    assert abs(_best_lag(out, clean)) <= 1          # 遅れは 1 フレーム以内


def test_one_euro_causal_mode_lags_but_zero_phase_does_not():
    t = np.arange(300) / FPS
    clean = np.sin(2 * np.pi * 1.0 * t)
    causal = one_euro(clean, FPS, min_cutoff=1.0, beta=0.0, zero_phase=False)
    assert _best_lag(causal, clean) >= 2
    assert abs(_best_lag(one_euro(clean, FPS, 1.0, 0.0), clean)) <= 1


def test_one_euro_keeps_constant_velocity_at_edges():
    x = np.linspace(0.0, 5.0, 120)[:, None] * np.array([1.0, 0.0, -2.0])
    out = one_euro(x, FPS, min_cutoff=0.3, beta=0.0)
    np.testing.assert_allclose(out, x, atol=1e-9)


def _smooth_rotation(T=200):
    t = np.arange(T) / FPS
    axis = np.stack([np.sin(0.7 * t), np.cos(0.5 * t), 0.4 + 0.0 * t], axis=1)
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    angle = 2.5 + 0.8 * np.sin(2 * np.pi * 0.6 * t)       # ±π をまたぐ大きな回転
    return quat.from_rotvec(axis * angle[:, None])


def test_quaternion_sign_flips_do_not_cause_jumps():
    cfg = load_config().jitter
    rng = np.random.default_rng(1)
    q = np.repeat(_smooth_rotation()[:, None], 24, axis=1)
    clean_max = rotation_angles(q).max()
    flipped = q.copy()
    flip = rng.random(flipped.shape[:2]) < 0.3
    flipped[flip] *= -1
    out, info = stabilize_pose(flipped, FPS, cfg)
    assert info['outlier_frames'] == 0
    assert rotation_angles(out).max() < clean_max * 1.2 + 0.5
    # 符号を揃えた出力は、隣り合うフレームの内積が正
    assert (np.sum(out[1:] * out[:-1], axis=-1) > 0).all()


def test_single_frame_outlier_is_replaced():
    q = _smooth_rotation()[:, None]
    bad = q.copy()
    bad[80, 0] = quat.mul(quat.from_rotvec([0.0, np.deg2rad(60.0), 0.0]), q[80, 0])
    fixed, mask = remove_rotation_outliers(bad, FPS, 720.0)
    assert mask[80, 0] and mask.sum() == 1
    assert np.rad2deg(quat.angle_between(fixed[80, 0], q[80, 0])) < 1.0


def test_fast_real_motion_is_not_treated_as_outlier():
    # 連続して速い（600 deg/s を超える）回転は、単独フレームの外れ値ではない
    t = np.arange(60) / FPS
    q = quat.from_rotvec(np.stack([0 * t, np.deg2rad(900.0) * t, 0 * t], axis=1))[:, None]
    _, mask = remove_rotation_outliers(q, FPS, 720.0)
    assert not mask.any()


def test_moving_min_then_limited_gaussian_never_exceeds_required():
    rng = np.random.default_rng(2)
    required = np.minimum(0.0, rng.normal(0, 1, 300))
    out = gaussian_time(moving_min(required, 9), 2.0, radius=4)
    assert (out <= required + 1e-12).all()


def _hand_on_chest(T=240, noise_deg=2.0, seed=0):
    """両手を胸の前（背骨3 の座標系で固定の点）に置いたまま、鎖骨を上下させ（肩をすくめる）、ひじを肩 → 手首の軸
    まわりに振る（ひじを張る）動き。関節は大きく動くが手首は止まっている。戻り値: (正解の回転, 震えを足した回転)。"""
    from nlf2vmd.jitter import ARM_CHAINS
    from nlf2vmd.synthetic import SMPL_REST_JOINTS as J
    rng = np.random.default_rng(seed)
    t = np.arange(T) / FPS
    q = np.tile(quat.IDENTITY, (T, 24, 1))
    for side, (spine, collar, shoulder, elbow, wrist) in enumerate(ARM_CHAINS):
        sign = 1.0 if side == 0 else -1.0
        target = np.array([0.05 * sign, -0.05, 0.3])
        shrug = sign * 0.25 * np.sin(2 * np.pi * 1.5 * t)
        q[:, collar] = quat.from_rotvec(np.stack([0 * t, 0 * t, shrug], axis=1))
        swivel = np.deg2rad(35.0) * np.sin(2 * np.pi * 1.2 * t + side)
        upper, fore = J[elbow] - J[shoulder], J[wrist] - J[elbow]
        lu, lf = np.linalg.norm(upper), np.linalg.norm(fore)
        hinge = np.cross(upper, [0.0, 0.0, 1.0]) * sign
        hinge /= np.linalg.norm(hinge)
        for i in range(T):
            Rc = quat.to_matrix(q[i, collar])
            goal = Rc.T @ (target - (J[collar] - J[spine]) - Rc @ (J[shoulder] - J[collar]))
            d = np.linalg.norm(goal)
            bend = np.arccos(np.clip((d ** 2 - lu ** 2 - lf ** 2) / (2 * lu * lf), -1.0, 1.0))
            q[i, elbow] = quat.from_rotvec(hinge * bend)
            reach = quat.from_two_vectors(upper + quat.rotate(q[i, elbow], fore), goal)
            q[i, shoulder] = quat.mul(quat.from_rotvec(goal / d * swivel[i]), reach)
    clean = quat.make_continuous(q)
    return clean, _add_arm_noise(clean, noise_deg, rng)


def _add_arm_noise(q, noise_deg, rng):
    """鎖骨・肩・ひじ・手首の回転に、フレームごとの震え（標準偏差 noise_deg 度）を足す。"""
    noisy = q.copy()
    for j in (13, 14, 16, 17, 18, 19, 20, 21):
        shake = quat.from_rotvec(rng.normal(0, np.deg2rad(noise_deg), (len(q), 3)))
        noisy[:, j] = quat.mul(noisy[:, j], shake)
    return noisy


def _swinging_arms(T=240, noise_deg=2.0, seed=0):
    """肩とひじを別の速さで大きく振る（手首が速く動く）動き。"""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / FPS
    q = np.tile(quat.IDENTITY, (T, 24, 1))
    zero = np.zeros(T)
    for side, sign in enumerate((1.0, -1.0)):
        swing = 0.8 * np.sin(2 * np.pi * 1.5 * t + side)
        bend = sign * (1.0 + 0.8 * np.sin(2 * np.pi * 0.7 * t))
        q[:, 16 + side] = quat.from_rotvec(np.stack([swing, zero, zero - sign], axis=1))
        q[:, 18 + side] = quat.from_rotvec(np.stack([zero, bend, zero], axis=1))
    clean = quat.make_continuous(q)
    return clean, _add_arm_noise(clean, noise_deg, rng)


def _hand_errors(clean, q):
    """(左右の手首の、正解からの位置のずれの平均 [cm], 手首の位置の 2 階差分の平均 [cm])。"""
    from nlf2vmd.jitter import ARM_CHAINS, hand_positions
    from nlf2vmd.synthetic import SMPL_REST_JOINTS as J
    err, jit = [], []
    for chain in ARM_CHAINS:
        h = hand_positions(q, J, chain)
        err.append(np.linalg.norm(h - hand_positions(clean, J, chain), axis=-1)[10:-10].mean())
        jit.append(np.linalg.norm(np.diff(h, 2, axis=0), axis=-1).mean())
    return np.mean(err) * 100.0, np.mean(jit) * 100.0


def _arm_smoothing(noisy, enabled):
    from nlf2vmd.synthetic import SMPL_REST_JOINTS as J
    cfg = load_config(overrides=[f'jitter.hand_position.enabled={str(enabled).lower()}']).jitter
    return stabilize_pose(noisy, FPS, cfg, J)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_resting_hand_is_held_while_the_arm_joints_move(seed):
    """胸の前に置いた手（関節は動くが手首は止まっている）は、関節ごとの平滑化より手首が正解に近く、揺れも小さい。"""
    clean, noisy = _hand_on_chest(seed=seed)
    err_off, jit_off = _hand_errors(clean, _arm_smoothing(noisy, False)[0])
    q, info = _arm_smoothing(noisy, True)
    err_on, jit_on = _hand_errors(clean, q)
    assert min(info['hand_hold_ratio']) > 0.9
    assert err_on < 0.92 * err_off
    assert jit_on < 0.85 * jit_off


def test_fast_hands_keep_the_per_joint_smoothing():
    """手首が速く動く腕は、関節ごとの平滑化とほぼ同じ（位置の軌跡の平滑化で動きが小さくならない）。"""
    clean, noisy = _swinging_arms()
    err_off, jit_off = _hand_errors(clean, _arm_smoothing(noisy, False)[0])
    err_on, jit_on = _hand_errors(clean, _arm_smoothing(noisy, True)[0])
    assert err_on < 1.05 * err_off
    assert jit_on < 1.05 * jit_off


def test_holding_the_hand_keeps_its_orientation_and_other_joints():
    """手首の位置に合わせても、手の向き（背骨3 に対する手首の大域回転）と、腕以外の関節は変えない。"""
    clean, noisy = _hand_on_chest(T=90, seed=3)
    off, _ = _arm_smoothing(noisy, False)
    on, _ = _arm_smoothing(noisy, True)
    for collar, shoulder, elbow, wrist in ((13, 16, 18, 20), (14, 17, 19, 21)):
        chain = lambda q: quat.mul(quat.mul(q[:, collar], q[:, shoulder]),  # noqa: E731
                                   quat.mul(q[:, elbow], q[:, wrist]))
        assert np.rad2deg(quat.angle_between(chain(off), chain(on))).max() < 1e-4
    others = [j for j in range(24) if j not in (16, 17, 18, 19, 20, 21)]
    np.testing.assert_allclose(on[:, others], off[:, others], atol=1e-12)
