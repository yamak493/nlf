"""センター: 届く高さへのクランプ後、補正が上向きになっているフレームが存在しないこと。"""
import numpy as np

from nlf2vmd import load_config
from nlf2vmd.center import ReachGeometry, apply_reach_clamp
from nlf2vmd.skeleton import Skeleton

UNIT = 12.5   # MMD 単位 / m（ReachGeometry は MMD 単位なので、しきい値 [m] にこれを掛ける）


def _setup(T=200, seed=0):
    rng = np.random.default_rng(seed)
    geom = ReachGeometry.from_skeleton(Skeleton.standard())
    t = np.arange(T) / 30.0
    center = np.zeros((T, 3))
    center[:, 1] = 0.6 * np.sin(2 * np.pi * 0.8 * t) + rng.normal(0, 0.05, T)   # 上下に揺れる骨盤
    center[:, 0] = 0.8 * np.sin(2 * np.pi * 0.3 * t)
    ik = np.zeros((T, 2, 3))
    ik[:, 0, 2] = 1.5 * np.sin(2 * np.pi * 0.5 * t)                            # 足を前後に出す
    ik[:, 1, 2] = -ik[:, 0, 2]
    lower_rot = np.tile(np.eye(3), (T, 1, 1))
    return geom, center, lower_rot, ik


def test_reach_clamp_only_lowers_the_center():
    cfg = load_config().center
    geom, center, lower_rot, ik = _setup()
    delta, corr_raw, corr, before, after = apply_reach_clamp(geom, center, lower_rot, ik, cfg,
                                                             UNIT)
    assert before > 0                          # この入力では脚が伸び切るフレームがある
    assert (corr_raw <= 0).all() and (corr <= 0).all()
    assert (delta[:, 1] <= center[:, 1] + 1e-12).all()
    np.testing.assert_array_equal(delta[:, [0, 2]], center[:, [0, 2]])
    assert after == 0


def test_applied_correction_is_at_least_the_required_drop():
    cfg = load_config().center
    geom, center, lower_rot, ik = _setup(seed=1)
    _, corr_raw, corr, _, _ = apply_reach_clamp(geom, center, lower_rot, ik, cfg, UNIT)
    assert (corr <= corr_raw + 1e-12).all()


def test_center_modes_run_and_never_raise(run_walk):
    for mode in ('A', 'B'):
        _, r = run_walk([f'center.mode={mode}'], name=f'{mode}.vmd', num_frames=180,
                        noise_deg=1.0)
        assert (r.center.correction <= 0).all()
        assert (r.center.delta[:, 1] <= r.center.smoothed[:, 1] + 1e-12).all()
        assert r.center.exceed_after == 0
