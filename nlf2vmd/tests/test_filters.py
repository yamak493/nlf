"""フィルタ: 震えが減り遅延が許容内であること。クォータニオンの符号反転で跳ねないこと。"""
import numpy as np

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
