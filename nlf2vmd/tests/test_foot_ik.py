"""足ＩＫ: 接地区間内の位置が完全に一定であること。境界でのフレーム間移動量が設定値を超えないこと。"""
import numpy as np
import pytest

from nlf2vmd import load_config, quat
from nlf2vmd.contact import ContactResult
from nlf2vmd.foot_ik import boundary_steps, build_foot_ik


def test_locked_positions_are_exactly_constant(run_walk):
    _, r = run_walk(num_frames=240, noise_deg=1.0, seed=5)
    assert all(len(s) > 0 for s in r.contact.segments)
    for foot in range(2):
        for s, e in r.contact.segments[foot]:
            seg = r.foot_ik.delta[s:e + 1, foot]
            assert (seg == seg[0]).all()
            rot = r.foot_ik.rotation[s:e + 1, foot]
            assert (rot == rot[0]).all()
    # VMD に書く値（MMD 座標）でも一定
    for foot, track in enumerate(t for t in r.tracks if t.name.endswith('足ＩＫ')):
        for s, e in r.contact.segments[foot]:
            assert (track.positions[s:e + 1] == track.positions[s]).all()


def test_boundary_steps_stay_within_limit(run_walk):
    _, r = run_walk(num_frames=240, noise_deg=1.0, seed=5)
    limit = r.config.foot_ik.max_boundary_step_m * r.scale
    assert max(boundary_steps(r.foot_ik.target, r.contact)) <= limit


def test_no_negative_height_after_clamp(run_walk):
    _, r = run_walk(num_frames=240, noise_deg=1.0, seed=5)
    assert (r.foot_ik.delta[..., 1] >= 0).all()


def _contact_from_segments(T, segs):
    flags = np.zeros((T, 2), bool)
    for foot in range(2):
        for s, e in segs[foot]:
            flags[s:e + 1, foot] = True
    z = np.zeros((T, 2, 2))
    return ContactResult(flags, segs, z, z)


def test_clamp_runs_after_lock_and_keeps_segment_constant():
    """ロック値が床より下なら、区間全体が一様に 0 へ持ち上がる（区間内に段差が出ない）。"""
    cfg = load_config().foot_ik
    T = 60
    rng = np.random.default_rng(0)
    raw = np.zeros((T, 2, 3))
    raw[..., 1] = -0.01 + rng.normal(0, 0.004, (T, 2))     # 床より少し下で震える
    raw[:, :, 2] = np.linspace(0, 1, T)[:, None]
    raw[20:40, :, 2] = raw[20, 0, 2]
    segs = [[(20, 39)], [(20, 39)]]
    rot = np.tile(quat.IDENTITY, (T, 2, 1))
    ik = build_foot_ik(raw, np.zeros((2, 3)), rot, _contact_from_segments(T, segs), 30.0, 1.0,
                       cfg)
    seg = ik.delta[20:40, 0]
    assert (seg == seg[0]).all() and seg[0, 1] == 0.0
    assert (ik.delta[..., 1] >= 0).all()
    assert ik.clamped_frames > 0


@pytest.mark.parametrize('snap', [True, False])
def test_snap_to_floor_puts_the_sole_on_the_floor(snap):
    """接地中の足が推定のずれで 2cm 浮いていても、ロックの高さは足裏が床に着く高さになる。"""
    cfg = load_config(overrides=[f'foot_ik.snap_to_floor={str(snap).lower()}']).foot_ik
    T = 40
    raw = np.zeros((T, 2, 3))
    raw[..., 1] = 0.10 + 0.02                        # 足首（床に平らなら 0.10）ごと 2cm 浮いている
    segs = [[(10, 29)], [(10, 29)]]
    contact = _contact_from_segments(T, segs)
    contact.heights[:] = 0.02                        # かかと・つま先の高さ
    rot = np.tile(quat.IDENTITY, (T, 2, 1))
    ik = build_foot_ik(raw, np.zeros((2, 3)), rot, contact, 30.0, 1.0, cfg)
    np.testing.assert_allclose(ik.target[10:30, :, 1], 0.10 if snap else 0.12)
