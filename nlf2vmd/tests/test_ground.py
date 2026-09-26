"""接地の拘束: 単眼推定のずれで両足が長く浮いた状態を床に戻し、ジャンプ（短い滞空）はそのまま残すこと。"""
import numpy as np
import pytest

from nlf2vmd import convert, load_config
from nlf2vmd.body_model import Kinematics
from nlf2vmd.diagnostics import hover_mask
from nlf2vmd.filters import runs
from nlf2vmd.ground import flight_window, floating_frames, ground_offset, lowest_foot_height
from nlf2vmd.synthetic import (add_jump, add_joint_noise, add_ray_drift, synthetic_walk,
                               to_camera_coords)

ORIGINAL = ['ground.enabled=false', 'foot_ik.snap_to_floor=false', 'floor.tilt_method=plane']


def _convert(tmp_path, body_model, motion, overrides=(), name='g.vmd'):
    cfg = load_config(overrides=['diagnostics.enabled=false', *overrides])
    return convert(motion, tmp_path / name, body_model=body_model, config=cfg, log=None)


def _floating(r):
    """両足が接地終了の高さより上にいる状態が、ジャンプより長く続くフレームの数。"""
    return floating_frames(lowest_foot_height(r.kin), r.fps,
                           r.config.contact.exit_height_m * r.scale,
                           r.config.ground.max_flight_sec)


def _stance_coverage(motion, r):
    """正解の支持脚のフレームのうち、接地と判定されたフレームの割合。"""
    return float(r.contact.flags[motion['stance']].mean())


def _stance_ik_height_cm(motion, r, margin=2):
    """正解の支持脚区間（端 margin フレームを除く）での足ＩＫの差分 Y の最大値 [cm]（0 = 足裏が床）。"""
    worst = 0.0
    for foot in range(2):
        for s, e in runs(motion['stance'][:, foot]):
            if e - s > 2 * margin + 1:
                seg = r.foot_ik.delta[s + margin:e - margin + 1, foot, 1]
                worst = max(worst, float(seg.max()))
    return worst / r.scale * 100.0


@pytest.mark.parametrize('pitch', [0.0, 10.0])
def test_drift_along_the_camera_ray_does_not_make_the_body_float(tmp_path, body_model, pitch):
    """高さ 1.6m のカメラから、その場で足踏み。距離が視線に沿って 30cm ずれ、体全体が上下に数 cm ずれる。"""
    motion = synthetic_walk(num_frames=450, speed=0.0, sway=0.08, noise_deg=0.5, seed=0)
    cam = add_ray_drift(to_camera_coords(motion, height=1.6, pitch_deg=pitch))
    clean = to_camera_coords(motion, height=1.6, pitch_deg=pitch)
    before = _convert(tmp_path, body_model, cam, ORIGINAL, 'before.vmd')
    after = _convert(tmp_path, body_model, cam, (), 'after.vmd')
    ref = _convert(tmp_path, body_model, clean, (), 'clean.vmd')
    assert _stance_ik_height_cm(motion, before) > 2.0          # 対策なしでは支持脚が床から浮く
    assert _floating(after) == 0
    assert _stance_coverage(motion, after) > _stance_coverage(motion, ref) - 0.05
    assert _stance_ik_height_cm(motion, after) < 1.0          # 支持脚の足裏が床に着いている


def test_jump_is_kept(tmp_path, body_model):
    """0.45 秒・高さ 25cm のジャンプ（両足が床から離れる）は、床に引き戻さない。"""
    motion = add_jump(synthetic_walk(num_frames=300, speed=0.0, seed=0), 5.0, 0.45, 0.25)
    r = _convert(tmp_path, body_model, motion)
    air = motion['airborne']
    peak = lowest_foot_height(r.kin)[air].max() / r.scale
    assert peak > 0.22                                          # ジャンプの高さが残る
    assert np.abs(r.ground.offset[air]).max() / r.scale < 0.01  # 補正はほぼ 0


def test_hovering_longer_than_a_jump_is_grounded(tmp_path, body_model):
    """1.5 秒も両足が 8cm 浮いたまま（重力に反する）なのは推定のずれとみなし、床に戻す。"""
    motion = add_jump(synthetic_walk(num_frames=300, speed=0.0, seed=0), 4.0, 1.5, 0.08)
    before = _convert(tmp_path, body_model, motion, ORIGINAL, 'before.vmd')
    after = _convert(tmp_path, body_model, motion, (), 'after.vmd')
    assert _floating(before) > 20
    assert _floating(after) == 0
    assert _stance_ik_height_cm(motion, after) < 1.0


def test_grounding_only_translates_the_body_vertically(tmp_path, body_model):
    motion = add_ray_drift(to_camera_coords(synthetic_walk(num_frames=200, speed=0.0, seed=0),
                                            height=1.6))
    off = ['depth.reconstruct=false', 'lean.enabled=false']   # 前後の傾きの補正は骨盤の高さを使うので切る
    r = _convert(tmp_path, body_model, motion, off, 'on.vmd')
    r0 = _convert(tmp_path, body_model, motion, off + ['ground.enabled=false'], 'off.vmd')
    shift = r.kin.joints - r0.kin.joints
    expected = np.zeros_like(shift)
    expected[..., 1] = -r.ground.offset[:, None]
    np.testing.assert_allclose(shift, expected, atol=1e-9)
    np.testing.assert_array_equal(r.kin.glob_rot, r0.kin.glob_rot)


@pytest.mark.parametrize('flight_frames', [1, 6, 17, 18])
def test_flight_window_reaches_ground_from_the_middle_of_a_flight(flight_frames):
    """滞空が max_flight_sec 以下なら、滞空の中央のフレームの窓にも接地しているフレームが入る。"""
    fps = 30.0
    window = flight_window(fps, 0.6)
    h = np.zeros(60)
    h[20:20 + flight_frames] = 1.0
    from nlf2vmd.filters import moving_min
    assert (moving_min(h, window) == 0.0).all()


FPS = 30.0
G = 9.8


def _kin(lowest, pelvis_y):
    """最下点の高さと骨盤の高さだけを持つ Kinematics（ground_offset の単体テスト用）。"""
    T = len(lowest)
    joints = np.zeros((T, 24, 3))
    joints[:, 0, 1] = pelvis_y
    pts = np.zeros((T, 2, 2, 3))
    pts[..., 1] = np.asarray(lowest)[:, None, None]
    return Kinematics(np.tile(np.eye(3), (T, 24, 1, 1)), joints, pts)


def _jump(T, start, n):
    """フレーム start から n フレームの滞空（重力加速度の放物線）。戻り値: (骨盤の上昇, 滞空の bool)。"""
    t = (np.arange(T) - (start - 1)) / FPS
    D = (n + 1) / FPS
    rise = np.where((t > 0) & (t < D), 0.5 * G * t * (D - t), 0.0)
    return rise, rise > 0


def test_foot_only_bumps_are_not_kept_as_jumps():
    """足先だけが 5cm 上がって見える区間（足首の向きの推定の揺れ。骨盤は上がらない）はジャンプにしない。
    ジャンプにすると、その区間は前後の値でつながれて全身が浮き、接地判定からも漏れる。"""
    cfg = load_config().ground
    T = 300
    lowest = np.zeros(T)
    for s in range(20, T - 20, 30):
        lowest[s:s + 9] = 0.05 * np.sin(np.pi * np.arange(1, 10) / 10) ** 2
    ground = ground_offset(_kin(lowest, np.full(T, 0.9)), FPS, 1.0, cfg)
    assert not ground.flight.any()
    after = lowest - ground.offset
    assert after.max() < 0.025          # 補正量の平滑化（0.1 秒）で残るのは山の半分未満


def test_jump_next_to_a_foot_bump_is_kept():
    """ジャンプの直前に足先の揺れの山がつながっていても、放物線の部分をジャンプとして残す。"""
    cfg = load_config().ground
    T = 200
    rise, air = _jump(T, 100, 12)
    lowest = rise.copy()
    lowest[94:100] = 0.04                               # 直前の揺れの山（骨盤は上がらない）
    ground = ground_offset(_kin(lowest, 0.9 + rise), FPS, 1.0, cfg)
    assert ground.flight[air].sum() >= 9
    assert not ground.flight[:95].any()
    assert (lowest - ground.offset)[air].max() > 0.9 * rise.max()     # ジャンプの高さが残る


@pytest.mark.parametrize('accel_g', [0.1, 4.0])
def test_non_ballistic_rise_is_not_a_jump(accel_g):
    """両足が 0.4 秒上がっていても、骨盤の加速度が重力に合わない（ゆっくり上下・ガタつく）ならジャンプではない。"""
    cfg = load_config().ground
    T = 200
    rise, air = _jump(T, 100, 12)
    lowest = rise * accel_g                             # 放物線の高さを加速度に比例させる
    ground = ground_offset(_kin(lowest, 0.9 + rise * accel_g), FPS, 1.0, cfg)
    assert not ground.flight.any()


def test_noisy_feet_do_not_create_fake_jumps(tmp_path, body_model):
    """足首・足先の向きの揺れ（8 度）で足先が上下して見えても、ジャンプとして残すのは本物のジャンプだけで、
    それ以外に両足が浮いたままのフレームがほとんど無い。"""
    motion = synthetic_walk(num_frames=600, speed=0.0, sway=0.08, noise_deg=1.0, seed=0,
                            step_sec=0.5)
    motion = add_jump(motion, 10.0, 0.45, 0.22)
    motion = add_joint_noise(motion, {7: 8.0, 8: 8.0, 10: 6.0, 11: 6.0, 4: 3.0, 5: 3.0}, seed=0)
    air = motion['airborne']
    r = _convert(tmp_path, body_model, to_camera_coords(motion, height=1.4, pitch_deg=5.0))
    near_jump = np.convolve(air, np.ones(7), mode='same') > 0          # 本物のジャンプ ±3 フレーム
    assert not r.ground.flight[~near_jump].any()
    assert r.ground.flight[air].sum() >= 0.7 * air.sum()
    assert hover_mask(r, 0.02 * r.scale)[~near_jump].sum() <= 5
