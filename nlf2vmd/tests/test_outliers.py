"""慣性・重力による外れフレームの除外（ステージ1b）: 明らかに間違ったフレームだけを見つけて置き換え、本当の動き
（速い腕の動き・回転・ジャンプ）はそのまま残すこと。"""
import numpy as np
import pytest

from nlf2vmd import convert, load_config, quat
from nlf2vmd.body_model import forward_kinematics
from nlf2vmd.config import Config
from nlf2vmd.filters import runs
from nlf2vmd.motion_io import load_motion
from nlf2vmd.outliers import (GRAVITY, GravityFit, project_gravity_cone, remove_outliers,
                              second_difference)
from nlf2vmd.synthetic import add_arm_reach, add_depth_noise, add_jump, synthetic_walk

FPS = 30.0


def _walk(num_frames=450, **kwargs):
    kwargs.setdefault('noise_deg', 2.0)
    kwargs.setdefault('speed', 1.2)
    return add_depth_noise(synthetic_walk(num_frames=num_frames, **kwargs), sigma=0.03,
                           drift=0.08)


def _glitch(motion, frames, joint=None, rotvec=None, trans=None):
    """motion の frames のフレームだけ、関節 joint の回転に rotvec を掛ける・骨盤を trans だけずらす。"""
    q = quat.from_rotvec(np.asarray(motion['pose'], float))
    tr = np.array(motion['trans'], float, copy=True)
    for f in frames:
        if joint is not None:
            q[f, joint] = quat.mul(q[f, joint], quat.from_rotvec(np.asarray(rotvec, float)))
        if trans is not None:
            tr[f] += trans
    return dict(motion, pose=quat.to_rotvec(q), trans=tr)


def _detect(motion, body_model, overrides=()):
    cfg = load_config(overrides=list(overrides))
    m = load_motion(motion, cfg.input, body_model)
    r = remove_outliers(m.quats, m.root_pos, m.valid, body_model.rest_joints(m.betas),
                        body_model.parents, m.fps, cfg.outliers)
    return m, r


def _flagged(r):
    return {k: runs(v) for k, v in r.flags.items() if v.any()}


@pytest.mark.parametrize('noise_deg', [0.0, 2.0, 5.0])
def test_clean_motion_is_left_alone(body_model, noise_deg):
    m, r = _detect(_walk(num_frames=600, noise_deg=noise_deg), body_model)
    assert _flagged(r) == {}
    assert r.valid.all()
    np.testing.assert_allclose(r.root_pos, m.root_pos)
    assert np.allclose(np.abs(np.sum(r.quats * m.quats, -1)), 1.0)


def test_real_jump_is_kept(body_model):
    """重力だけで動く放物線（0.45 秒・25cm）は、慣性の予測に合わなくても外れにしない。"""
    _, r = _detect(add_jump(_walk(noise_deg=3.0, speed=1.0), 7.0, 0.45, 0.25), body_model)
    assert _flagged(r) == {}


def _arm_motion(target):
    return add_arm_reach(synthetic_walk(num_frames=len(target), noise_deg=2.0, speed=0.0), 0,
                         [0.3, -0.2, 0.5], target)


def test_fast_real_motions_are_kept(body_model):
    t = np.arange(450) / FPS
    # 手を 0.5 秒ごとに 50cm 離れた 2 か所へ、2 フレームで動かして止める（段差の動き）
    strikes = np.where((np.floor(t / 0.5) % 2)[:, None] == 0, [0.5, 0.3, 0.2], [0.2, 0.3, 0.6])
    strikes[1:] = 0.5 * (strikes[1:] + strikes[:-1])
    circle = np.stack([0.4 + 0.25 * np.sin(2 * np.pi * 3 * t), 0.3 + 0.2 * np.cos(2 * np.pi * 3 * t),
                       np.full_like(t, 0.4)], 1)
    for target in (strikes, circle):
        _, r = _detect(_arm_motion(target), body_model)
        assert _flagged(r) == {}
    # 1 秒で 2 回転（最大 1080 度/秒）
    base = synthetic_walk(num_frames=450, noise_deg=2.0, speed=0.0)
    s = np.clip(t - 5.0, 0.0, 1.0)
    yaw = 4 * np.pi * s * s * (3 - 2 * s)
    q = quat.from_rotvec(np.asarray(base['pose'], float))
    q[:, 0] = quat.mul(quat.from_rotvec(np.stack([0 * t, yaw, 0 * t], 1)), q[:, 0])
    _, r = _detect(dict(base, pose=quat.to_rotvec(q)), body_model)
    assert _flagged(r) == {}


@pytest.mark.parametrize('joint, rotvec, frames, part', [
    (18, [0, 0, 1.5], [100], 'left_arm'),                 # ひじが 1 フレームだけ 86 度
    (16, [0, 1.2, 0], range(200, 204), 'left_arm'),       # 肩が 4 フレーム
    (16, [0, 1.2, 0], range(200, 208), 'left_arm'),       # 肩が 8 フレーム
    (0, [0, np.pi, 0], range(300, 306), 'torso'),         # 体の向きが 6 フレームだけ反転
    (4, [0.8, 0, 0], range(400, 403), 'legs'),            # 膝が 3 フレーム
    (20, [2.5, 0, 0], range(540, 543), 'left_arm'),       # 手首が 3 フレームだけひっくり返る
])
def test_pose_glitches_are_found(body_model, joint, rotvec, frames, part):
    motion = _walk(num_frames=600)
    _, r = _detect(_glitch(motion, frames, joint, rotvec), body_model)
    assert _flagged(r) == {part: [(frames[0], frames[-1])]}


@pytest.mark.parametrize('frames', [range(0, 3), range(447, 450)])
def test_glitches_at_the_ends_are_found(body_model, frames):
    _, r = _detect(_glitch(_walk(), frames, 17, [0, 1.2, 0]), body_model)
    assert _flagged(r) == {'right_arm': [(frames[0], frames[-1])]}


def test_replaced_pose_follows_the_motion(body_model):
    motion = _walk()
    clean, _ = _detect(motion, body_model)
    m, r = _detect(_glitch(motion, range(200, 204), 16, [0, 1.2, 0]), body_model)
    before = np.rad2deg(quat.angle_between(m.quats[200:204, 16], clean.quats[200:204, 16]))
    after = np.rad2deg(quat.angle_between(r.quats[200:204, 16], clean.quats[200:204, 16]))
    assert before.min() > 60.0
    assert after.max() < 10.0
    # 腕だけの外れは、床・接地・奥行きの推定にはそのまま使う
    assert r.valid.all()


def test_torso_glitch_is_not_an_observation(body_model):
    motion = _walk()
    clean, _ = _detect(motion, body_model)
    _, r = _detect(_glitch(motion, range(300, 306), 0, [0, np.pi, 0]), body_model)
    assert runs(~r.valid) == [(300, 305)]
    # 体の向き（180 度反転していた）は前後から補間した向きになる（正解にも関節ごとに 2 度の揺れがある）
    err = np.rad2deg(quat.angle_between(r.quats[300:306, 0], clean.quats[300:306, 0]))
    assert err.max() < 15.0
    # 骨盤の位置も置き直す（奥行きは、正解にも推定のぶれ 3cm + ゆっくりしたずれがある）
    err = np.abs(r.root_pos[300:306] - clean.root_pos[300:306])
    assert err[:, 1].max() < 0.05 and err[:, 2].max() < 0.2


@pytest.mark.parametrize('trans, frames', [
    ([0, 0.12, 0], range(450, 455)),     # 骨盤が 5 フレームだけ 12cm 浮く
    ([0, 0.15, 0], range(100, 108)),     # 8 フレームだけ 15cm 浮く
    ([0, 0, 0.5], range(520, 526)),      # 6 フレームだけ奥行きが 50cm ずれる
    ([0, 0, 0.6], range(0, 4)),          # 先頭の 4 フレーム
])
def test_centre_of_mass_glitches_are_found(body_model, trans, frames):
    motion = _walk(num_frames=600)
    clean, _ = _detect(motion, body_model)
    _, r = _detect(_glitch(motion, frames, trans=trans), body_model)
    assert _flagged(r) == {'com': [(frames[0], frames[-1])]}
    assert not r.valid[list(frames)].any()
    # 置き直した骨盤は、正しい位置に近い（上下は、合成の歩行の支持脚が切り替わる瞬間の折れ曲がりに重力の条件では
    # 付いていけない分だけずれる。奥行きは、正解にも推定のぶれ 3cm + ゆっくりしたずれがある）
    err = np.abs(r.root_pos[list(frames)] - clean.root_pos[list(frames)])
    assert err[:, 1].max() < 0.07
    assert err[:, 2].max() < 0.15


def test_long_glitch_is_reported_but_not_replaced(body_model):
    motion = _glitch(_walk(), range(200, 206), 16, [0, 1.2, 0])
    m, r = _detect(motion, body_model, ['outliers.max_run_sec=0.1'])
    assert _flagged(r) == {}
    assert ('left_arm', 200, 205) in r.long_runs
    np.testing.assert_allclose(r.quats, quat.make_continuous(m.quats))


def test_detector_with_too_many_outliers_is_not_used(body_model):
    motion = _glitch(_walk(), range(450 // 2, 450 // 2 + 5), trans=[0, 0.12, 0])
    _, r = _detect(motion, body_model, ['outliers.max_ratio=0.005'])
    assert 'com' in r.skipped
    assert _flagged(r) == {}


def test_disabled(body_model):
    motion = _glitch(_walk(), range(300, 306), 0, [0, np.pi, 0])
    m, r = _detect(motion, body_model, ['outliers.enabled=false'])
    assert not r.enabled and _flagged(r) == {}
    np.testing.assert_allclose(r.root_pos, m.root_pos)


def test_gravity_cone_projection():
    g = GRAVITY
    a = np.array([[0.0, -g, 0.0],            # 自由落下（範囲の端）
                  [0.0, -2 * g, 0.0],        # 重力より速く落ちる → 自由落下
                  [6.0, 0.0, 0.0],           # 摩擦 0.5 × g より速く横へ → 円錐の側面
                  [0.0, 10 * g, 0.0],        # 床から体重の 11 倍の力 → 上限
                  [1.0, 1.0, -1.0]])         # 範囲の中
    p = project_gravity_cone(a, 0.5, 4 * g)
    np.testing.assert_allclose(p[0], a[0])
    np.testing.assert_allclose(p[1], [0.0, -g, 0.0])
    f = p[2] + [0.0, g, 0.0]
    assert abs(np.hypot(f[0], f[2]) - 0.5 * f[1]) < 1e-9 and f[0] > 0 and f[1] > g
    np.testing.assert_allclose(p[3], [0.0, 3 * g, 0.0])
    np.testing.assert_allclose(p[4], a[4])


def test_gravity_fit_keeps_the_acceleration_in_range():
    rng = np.random.default_rng(0)
    T = 300
    t = np.arange(T) / FPS
    y = np.stack([0.3 * np.sin(t), 0.9 + 0.02 * np.sin(4 * t), 1.0 * t], 1)
    y += rng.normal(0.0, 0.03, (T, 3)) * [0.2, 0.2, 1.0]       # 奥行きのぶれが大きい
    cfg = Config(friction=1.0, max_force_g=4.0, iterations=300)
    x = GravityFit(FPS, [0.0, 0.0, 1.0], cfg, sigma=0.025, sigma_depth=0.1).solve(y, np.ones(T))
    acc = second_difference(T) @ x * FPS ** 2
    np.testing.assert_allclose(project_gravity_cone(acc, 1.0, 4 * GRAVITY), acc, atol=0.05)
    assert np.abs(x - y)[:, 1].max() < 0.02


def _run(tmp_path, body_model, motion, overrides=(), name='o.vmd', diag=False):
    cfg = load_config(overrides=[f'diagnostics.enabled={str(diag).lower()}', *overrides])
    return convert(motion, tmp_path / name, body_model=body_model, config=cfg, log=None,
                   diag_dir=tmp_path / 'diag' if diag else None)


def test_convert_replaces_outliers_and_reports_them(tmp_path, body_model):
    pytest.importorskip('matplotlib')
    motion = _glitch(_walk(num_frames=300), range(150, 156), 0, [0, np.pi, 0])
    r = _run(tmp_path, body_model, motion, diag=True)
    assert r.info['outliers']['frames']['torso'] == 6
    assert r.info['outliers']['unobserved_frames'] == 6
    assert r.info['interpolated']['frames'] == 0          # 入力の検出できなかったフレームとは別に数える
    assert runs(~r.motion.valid) == [(150, 155)]
    assert 'outliers' in r.plot_paths
    # 体の向きが反転したフレームは、平滑化の後にも残らない
    yaw = np.rad2deg(quat.angle_between(r.quats[140:166, 0], r.quats[139:165, 0]))
    assert yaw.max() < 20.0


def test_convert_without_outliers_keeps_the_glitch(tmp_path, body_model):
    motion = _glitch(_walk(num_frames=300), range(150, 156), 0, [0, np.pi, 0])
    r = _run(tmp_path, body_model, motion, ['outliers.enabled=false'])
    assert r.info['outliers'] == dict(enabled=False)
    yaw = np.rad2deg(quat.angle_between(r.quats[140:166, 0], r.quats[139:165, 0]))
    assert yaw.max() > 40.0


def test_leg_glitch_during_a_jump_keeps_the_jump(tmp_path, body_model):
    """脚の外れ（観測として使わない）がジャンプの途中にあっても、置き換えた動きは重力で動ける上下の動きなので、
    ジャンプとして残る（検出できなかったフレームと違い、床に着けない）。"""
    base = add_jump(synthetic_walk(num_frames=300, speed=0.0, noise_deg=1.0), 5.0, 0.45, 0.25)
    motion = _glitch(base, range(155, 158), 4, [0.8, 0, 0])
    r = _run(tmp_path, body_model, motion)
    assert r.outliers.flags['legs'][155:158].all()
    air = np.asarray(base['airborne'])
    assert r.ground.flight[air].mean() > 0.8
    kin_clean = _run(tmp_path, body_model, base, name='clean.vmd').kin
    peak = air.nonzero()[0][len(air.nonzero()[0]) // 2]
    assert abs(r.kin.root_pos[peak, 1] - kin_clean.root_pos[peak, 1]) / r.scale < 0.03
