"""床面推定: 傾いた床を水平に直し、補正は定数（区間モードでは区間ごとの定数）だけであること。"""
import numpy as np

from nlf2vmd import load_config
from nlf2vmd.body_model import Kinematics

FPS = 30.0


def _kinematics(heights_fn, T=600, hold=10, seed=0):
    """足をあちこちに置いて hold フレームずつ止める。床の高さは heights_fn(x, z, t)。"""
    rng = np.random.default_rng(seed)
    pts = np.zeros((T, 2, 2, 3))
    for s in range(0, T, hold):
        xz = rng.uniform(-1.0, 1.0, (2, 2, 2))
        pts[s:s + hold, ..., 0] = xz[..., 0]
        pts[s:s + hold, ..., 2] = xz[..., 1]
    t = np.arange(T)[:, None, None]
    pts[..., 1] = heights_fn(pts[..., 0], pts[..., 2], t)
    joints = np.zeros((T, 24, 3))
    joints[:, 0] = [0.3, 1.0, -2.0]
    return Kinematics(np.tile(np.eye(3), (T, 24, 1, 1)), joints, pts)


def test_tilted_floor_is_leveled_with_a_constant_offset():
    from nlf2vmd.floor import estimate_floor
    cfg = load_config().floor
    tilt = np.deg2rad(12.0)
    kin = _kinematics(lambda x, z, t: 0.4 + np.tan(tilt) * z + 0.05 * x)
    floor = estimate_floor(kin, FPS, cfg)
    assert floor.tilt_applied and abs(floor.tilt_deg - 12.2) < 0.5
    assert np.ptp(floor.offset, axis=0).max() == 0.0           # フレーム毎に変わる補正は無い
    out = floor.apply(kin)
    assert np.abs(out.contact_points[..., 1]).max() < 1e-6
    np.testing.assert_allclose(out.root_pos[0, [0, 2]], 0.0, atol=1e-9)   # 水平の原点 = 最初の骨盤


def test_collinear_points_do_not_tilt():
    from nlf2vmd.floor import estimate_floor
    cfg = load_config().floor
    kin = _kinematics(lambda x, z, t: 0.0 * x)
    kin.contact_points[..., 0] = 0.05 * np.sign(kin.contact_points[..., 0])   # ほぼ一直線
    floor = estimate_floor(kin, FPS, cfg)
    assert not floor.tilt_applied
    np.testing.assert_array_equal(floor.rotation, np.eye(3))


def test_segment_mode_uses_per_segment_constants_with_cosine_blend():
    from nlf2vmd.floor import estimate_floor
    cfg = load_config(overrides=['floor.segment_mode=true', 'floor.segment_sec=10',
                                 'floor.align_tilt=false']).floor
    kin = _kinematics(lambda x, z, t: np.where(t < 300, 0.0, 0.1) + 0.0 * x)
    floor = estimate_floor(kin, FPS, cfg)
    y = floor.offset[:, 1]
    assert np.allclose(y[:280], 0.0) and np.allclose(y[320:], -0.1)
    ramp = np.flatnonzero((y < -1e-9) & (y > -0.1 + 1e-9))
    assert len(ramp) >= FPS - 2                    # 1 秒以上かけてつなぐ
    assert (np.diff(y) <= 1e-12).all()             # 途中で行き過ぎない
