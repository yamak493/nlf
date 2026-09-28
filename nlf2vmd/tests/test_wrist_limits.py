"""手首の可動域（ステージ9c）: 前腕のひねり（回内・回外）と手首の曲げを人の関節の範囲に収めること。"""
import numpy as np
import pytest

from nlf2vmd import convert, load_config, quat
from nlf2vmd.skeleton import Skeleton
from nlf2vmd.synthetic import SMPL_REST_JOINTS, synthetic_body_model, synthetic_walk
from nlf2vmd.wrist_limits import (clamp_wrist, hand_axes, limit_wrists, within_limits,
                                  wrist_angles)

FPS = 30.0
J = SMPL_REST_JOINTS


def _axes(side):
    return hand_axes(J[18 + side], J[20 + side], side)


def _about(axis, deg):
    deg = np.atleast_1d(np.asarray(deg, np.float64))
    return quat.from_rotvec(np.deg2rad(deg)[:, None] * np.asarray(axis, np.float64))


def _wrist(ax, sup=0.0, flex=0.0, rad=0.0):
    """回外 sup・掌屈 flex・橈屈 rad [度] の手首のローカル回転（ひじは単位回転）。"""
    twist = _about(ax.forearm, ax.sign * np.asarray(sup, np.float64))
    bend = quat.from_rotvec(np.deg2rad(np.stack(np.broadcast_arrays(flex, rad), -1).reshape(-1, 2))
                            @ np.stack([ax.flexion, ax.radial]))
    return quat.mul(twist, bend)


@pytest.mark.parametrize('side', [0, 1])
def test_axes_follow_the_anatomy(side):
    """T ポーズで手のひらは下・親指は前。回外で手のひらが前（親指の側）へ、掌屈で手が手のひら側へ、橈屈で親指側へ曲がる。"""
    ax = _axes(side)
    np.testing.assert_allclose(ax.forearm, [1.0 if side == 0 else -1.0, 0, 0], atol=0.06)
    np.testing.assert_allclose(ax.thumb, [0, 0, 1], atol=0.06)
    np.testing.assert_allclose(ax.palm, [0, -1, 0], atol=0.06)
    palm = quat.rotate(_wrist(ax, sup=90.0), ax.palm)[0]
    assert palm @ [0, 0, 1] > 0.99                                    # 回外 90 度: 手のひらが前
    palm = quat.rotate(_wrist(ax, sup=-90.0), ax.palm)[0]
    assert palm @ [0, 0, -1] > 0.99                                   # 回内 90 度: 手のひらが後ろ
    hand = quat.rotate(_wrist(ax, flex=60.0), ax.forearm)[0]
    assert hand @ ax.palm > 0.8                                       # 掌屈: 手のひら（下）側へ
    hand = quat.rotate(_wrist(ax, rad=20.0), ax.forearm)[0]
    assert hand @ ax.thumb > 0.3                                      # 橈屈: 親指（前）側へ


@pytest.mark.parametrize('side', [0, 1])
def test_angles_add_elbow_and_wrist_twist_and_measure_the_bend_in_the_turned_hand(side):
    ax = _axes(side)
    bend_axis = np.cross(ax.forearm, ax.thumb)                                      # 前腕に垂直な軸
    elbow = quat.mul(_about(bend_axis, 70.0), _about(ax.forearm, ax.sign * 50.0))   # 曲げ + 回外 50
    wrist = _wrist(ax, sup=30.0, flex=40.0, rad=-15.0)
    np.testing.assert_allclose(wrist_angles(elbow, wrist, ax)[0], [80.0, 40.0, -15.0], atol=1e-6)


@pytest.mark.parametrize('side', [0, 1])
def test_back_of_hand_under_the_forearm_is_limited(side):
    """手のひらを 170 度返した（前腕の下側に手の甲が来た）手首は、回外 90 度（上限）に収まる。曲げはそのまま。"""
    ax = _axes(side)
    cfg = load_config().wrist_limits
    wrist = _wrist(ax, sup=170.0, flex=20.0)
    elbow = np.tile(quat.IDENTITY, (1, 1))
    out, changed = clamp_wrist(elbow, wrist, ax, cfg, FPS)
    assert changed.all()
    np.testing.assert_allclose(wrist_angles(elbow, out, ax)[0], [90.0, 20.0, 0.0], atol=1e-6)


def test_bend_is_limited_inside_the_ellipse():
    ax = _axes(0)
    cfg = load_config(overrides=['wrist_limits.max_speed_deg_per_s=0']).wrist_limits
    elbow = np.tile(quat.IDENTITY, (3, 1))
    wrist = _wrist(ax, flex=[120.0, -100.0, 60.0], rad=[0.0, 0.0, 40.0])
    out, _ = clamp_wrist(elbow, wrist, ax, cfg, FPS)
    ang = wrist_angles(elbow, out, ax)
    np.testing.assert_allclose(ang[0, 1], 90.0, atol=1e-6)      # 掌屈の上限
    np.testing.assert_allclose(ang[1, 1], -70.0, atol=1e-6)     # 背屈の上限
    assert within_limits(ang, cfg).all()
    # 斜めに曲げたときは、向き（掌屈:橈屈 = 60:40）を保って楕円の上へ
    np.testing.assert_allclose(ang[2, 1] / ang[2, 2], 60.0 / 40.0, rtol=1e-6)
    assert np.hypot(ang[2, 1] / 90.0, ang[2, 2] / 25.0) == pytest.approx(1.0, abs=1e-6)


def test_poses_inside_the_range_are_unchanged():
    ax = _axes(1)
    cfg = load_config().wrist_limits
    rng = np.random.default_rng(0)
    T = 50
    bend_axis = np.cross(ax.forearm, ax.thumb)
    elbow = quat.mul(_about(bend_axis, rng.uniform(0, 120, T)), _about(ax.forearm, rng.uniform(-40, 40, T)))
    wrist = _wrist(ax, sup=rng.uniform(-40, 40, T), flex=rng.uniform(-50, 70, T),
                   rad=rng.uniform(-10, 10, T))
    out, changed = clamp_wrist(elbow, wrist, ax, cfg, FPS)
    assert not changed.any()
    np.testing.assert_array_equal(out, wrist)


def test_turning_through_the_back_does_not_flip_between_the_limits():
    """推定が回外 0 → 180 → 回内側へ通り抜けても、180 度の近くで上限と下限の間を行き来しない。回内の側に大きく
    回ってから（上限より下限に近くなってから）1 度だけ、中立の側を通ってなめらかに返す。"""
    ax = _axes(0)
    cfg = load_config().wrist_limits
    sup = np.concatenate([np.linspace(0, 175, 30), np.linspace(-175, -60, 40), np.full(20, -60.0)])
    sup = sup + np.random.default_rng(1).normal(0.0, 4.0, len(sup))      # 180 度の近くで行き来する
    wrist = _wrist(ax, sup=sup)
    elbow = np.tile(quat.IDENTITY, (len(sup), 1))
    out, _ = clamp_wrist(elbow, wrist, ax, cfg, FPS)
    ang = wrist_angles(elbow, out, ax)
    assert within_limits(ang, cfg).all()
    step = np.abs(np.diff(ang[:, 0]))
    assert step.max() <= cfg.max_speed_deg_per_s / FPS + 1e-6         # 1 フレームに 24 度まで
    # 180 度の近く（推定が ±4 度でぶれて行き来する所）では上限（90）のまま
    assert (ang[25:37, 0] > 80.0).all()
    # 回外 → 回内へ返すのは 1 回だけ
    assert np.count_nonzero(np.diff(np.sign(ang[20:, 0])) != 0) == 1
    np.testing.assert_allclose(ang[-5:, 0], sup[-5:], atol=1e-6)      # 範囲に戻れば推定のまま
    # 手首の回転そのものも 1 フレームで大きく変わらない
    assert np.rad2deg(quat.angle_between(out[1:], out[:-1])).max() < 30.0


def test_disabled_keeps_the_rotations():
    skel = Skeleton.standard()
    ax = hand_axes(skel.internal('左ひじ'), skel.internal('左手首'), 0)
    local = {'左ひじ': np.tile(quat.IDENTITY, (3, 1)), '左手首': _wrist(ax, sup=[0.0, 170.0, 0.0])}
    cfg = load_config(overrides=['wrist_limits.enabled=false']).wrist_limits
    out, res = limit_wrists(skel, local, cfg, FPS)
    np.testing.assert_array_equal(out['左手首'], local['左手首'])
    assert res.info['out_of_range_frames'] == [1, 0] and res.info['changed_frames'] == [0, 0]


@pytest.fixture(scope='module')
def body_model():
    return synthetic_body_model()


def _flipped_walk(T=90):
    """歩きながら、左の前腕が途中で手のひらを 180 度近く返す（推定の誤り）動き。"""
    motion = synthetic_walk(num_frames=T)
    pose = np.asarray(motion['pose'], np.float64).copy()
    q = quat.from_rotvec(pose.reshape(T, -1, 3))
    ax = _axes(0)
    sup = np.concatenate([np.zeros(20), np.linspace(0, 175, 10), np.full(30, 175.0),
                          np.linspace(175, 0, 10), np.zeros(T - 70)])
    q[:, 20] = quat.mul(q[:, 20], _about(ax.forearm, ax.sign * sup))
    return dict(motion, pose=quat.to_rotvec(q).reshape(T, -1)), sup


def test_convert_limits_the_wrists(body_model):
    src, sup = _flipped_walk()
    cfg = load_config(overrides=['diagnostics.enabled=false'])
    r = convert(src, None, body_model=body_model, config=cfg, log=None)
    skel = r.skeleton
    ax = hand_axes(skel.internal('左ひじ'), skel.internal('左手首'), 0)
    ang = wrist_angles(r.local_quats['左ひじ'], r.local_quats['左手首'], ax)
    assert within_limits(ang, cfg.wrist_limits).all()
    # 途中の区間は範囲の外だった（9b の前に範囲に収めるので、9b の後はもう範囲の中）
    before = r.info['wrist_limits']['before_contacts']
    assert before['out_of_range_frames'][0] > 20 and before['max_correction_deg'][0] > 60.0
    assert r.info['wrist_limits']['out_of_range_frames'] == [0, 0]
    steps = np.rad2deg(quat.angle_between(r.local_quats['左手首'][1:], r.local_quats['左手首'][:-1]))
    assert steps.max() < 30.0
    off = convert(src, None, body_model=body_model, log=None,
                  config=load_config(overrides=['diagnostics.enabled=false', 'wrist_limits.enabled=false']))
    ang_off = wrist_angles(off.local_quats['左ひじ'], off.local_quats['左手首'], ax)
    assert not within_limits(ang_off, cfg.wrist_limits).all()
    # 右手・ひじは変えない
    for name in ('左ひじ', '右手首', '右ひじ'):
        np.testing.assert_allclose(r.local_quats[name], off.local_quats[name], atol=1e-9)
