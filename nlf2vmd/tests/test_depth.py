"""奥行きのぶれへの対策: 単眼推定のような奥行きのぶれがあっても、接地中の足が前後に滑らないこと。"""
import numpy as np
import pytest

from nlf2vmd import convert, load_config
from nlf2vmd.depth import depth_jitter, detection_speeds, solve_depth
from nlf2vmd.filters import runs
from nlf2vmd.floor import horizontal_speed
from nlf2vmd.synthetic import add_depth_noise, synthetic_walk, to_camera_coords

ORIGINAL = ['depth.detect_root_cutoff_hz=0', 'depth.detect_foot_cutoff_hz=0',
            'depth.reconstruct=false']


def _convert_noisy(tmp_path, body_model, overrides=(), name='n.vmd', sigma=0.03, drift=0.08,
                   **walk_kwargs):
    kw = dict(num_frames=300, noise_deg=0.5, seed=0)
    kw.update(walk_kwargs)
    motion = synthetic_walk(**kw)
    cam = add_depth_noise(to_camera_coords(motion), sigma=sigma, drift=drift, seed=0)
    cfg = load_config(overrides=['diagnostics.enabled=false', *overrides])
    return motion, convert(cam, tmp_path / name, body_model=body_model, config=cfg, log=None)


def _stance_slide_cm(motion, r, margin=2):
    """正解の支持脚区間（端 margin フレームを除く）での足ＩＫの移動量の平均 [cm/フレーム]（X, Z）。"""
    steps = []
    for foot in range(2):
        for s, e in runs(motion['stance'][:, foot]):
            if e - s > 2 * margin + 1:
                steps.append(np.abs(np.diff(r.foot_ik.target[s + margin:e - margin + 1, foot],
                                            axis=0)))
    d = np.concatenate(steps) / r.scale * 100.0
    return d[:, 0].mean(), d[:, 2].mean()


@pytest.mark.parametrize('walk', [dict(speed=1.0), dict(speed=0.0, sway=0.08)],
                         ids=['walk', 'sway_in_place'])
def test_depth_jitter_does_not_make_planted_feet_slide(tmp_path, body_model, walk):
    """前に歩く / その場で骨盤を前後に揺らしながら足踏みする。奥行きに 3cm のぶれと 8cm のずれがある。"""
    motion, before = _convert_noisy(tmp_path, body_model, ORIGINAL, 'before.vmd', **walk)
    _, after = _convert_noisy(tmp_path, body_model, (), 'after.vmd', **walk)
    _, clean = _convert_noisy(tmp_path, body_model, (), 'clean.vmd', sigma=0.0, drift=0.0,
                              **walk)
    _, z0 = _stance_slide_cm(motion, before)
    x1, z1 = _stance_slide_cm(motion, after)
    assert z0 > 0.15                     # 対策なしでは Z 方向に滑る（X はほとんど滑らない）
    assert z1 < 0.03 and x1 < 0.03       # 対策ありでは X と同程度まで減る
    # 接地区間が途切れない（奥行きのぶれが無いときと同じ数の区間が見つかる）
    assert [len(s) for s in after.contact.segments] == [len(s) for s in clean.contact.segments]


@pytest.mark.parametrize('speed', [1.0, 2.0])
def test_noise_free_walk_is_unchanged_by_depth_handling(run_walk, speed):
    """奥行きのぶれが無ければ、接地区間は対策なしと同じで、歩いた距離もほぼ変わらない
    （接地中の足首の位置を平滑化した姿勢から求めると、1 歩ごとに数 cm ずつ短くなる）。"""
    _, r = run_walk(num_frames=300, speed=speed)
    _, r0 = run_walk(ORIGINAL, name='orig.vmd', num_frames=300, speed=speed)
    assert r.contact.segments == r0.contact.segments
    assert np.abs(r.depth.correction).max() / r.scale < 0.04    # 10 秒（10〜20 m）で 4cm 未満


def test_reconstruction_only_translates_along_the_depth_axis(tmp_path, body_model):
    """奥行きの補正は体全体の平行移動で、姿勢（骨盤からの相対位置）は変えない。"""
    _, r = _convert_noisy(tmp_path, body_model)
    _, r0 = _convert_noisy(tmp_path, body_model, ['depth.reconstruct=false'], 'off.vmd')
    shift = r.kin.joints - r0.kin.joints                          # (T, J, 3)
    expected = r.depth.correction[:, None, None] * r.depth.axis
    np.testing.assert_allclose(shift, np.broadcast_to(expected, shift.shape), atol=1e-9)
    np.testing.assert_array_equal(r.kin.glob_rot, r0.kin.glob_rot)


def test_solve_depth_keeps_the_planted_foot_still():
    """片足ずつ交互に接地。姿勢から求めた足首の相対位置は正確で、骨盤の推定の奥行きだけがぶれる。"""
    rng = np.random.default_rng(0)
    T, fps = 120, 30.0
    t = np.arange(T) / fps
    true_root = 0.8 * t + 0.03 * np.sin(2 * np.pi * 1.5 * t)       # 前進 + 前後の揺れ
    feet = np.floor(t / 0.5)                                          # 0.5 秒ごとに支持脚が交代
    planted = np.stack([0.8 * 0.5 * (feet + 0.5)] * 2, axis=1)       # 支持脚の足首の位置
    rel = planted - true_root[:, None]
    segs = [[], []]
    for foot in range(2):
        segs[foot] = [(s, e) for s, e in runs(feet % 2 == foot)]
    raw = true_root + rng.normal(0, 0.03, T)
    r = solve_depth(raw, rel, segs, fps, contact_sigma=0.005, prior_sigma=0.5,
                    accel_sigma=3.0)
    for foot in range(2):
        for s, e in segs[foot]:
            foot_z = r[s:e + 1] + rel[s:e + 1, foot]
            assert np.ptp(foot_z) < 0.01        # 区間内の足首の奥行きの変化が 1cm 未満
    assert np.abs(np.diff(raw + rel[:, 0])).mean() > 0.02    # 補正前は毎フレーム数 cm 動く


def test_solve_depth_without_contacts_follows_the_estimate_smoothly():
    T = 90
    raw = np.linspace(0.0, 1.0, T)
    r = solve_depth(raw, np.zeros((T, 2)), [[], []], 30.0, 0.005, 0.5, 3.0)
    np.testing.assert_allclose(r, raw, atol=1e-6)      # 等速の移動はそのまま


def test_detection_speeds_without_smoothing_equal_plain_speeds(run_walk):
    _, r = run_walk(num_frames=60)
    cfg = load_config(overrides=['depth.detect_root_cutoff_hz=0',
                                 'depth.detect_foot_cutoff_hz=0']).depth
    got = detection_speeds(r.kin, r.depth.axis, r.fps, r.scale, cfg)
    np.testing.assert_array_equal(got, horizontal_speed(r.kin.contact_points, r.fps))


def test_depth_jitter_estimates_white_noise_level():
    rng = np.random.default_rng(1)
    T = 3000
    pos = np.zeros((T, 3))
    pos[:, 2] = np.linspace(0, 3, T) + rng.normal(0, 0.02, T)
    pos[:, 0] = rng.normal(0, 0.005, T)
    depth, lateral = depth_jitter(pos, np.array([0.0, 0.0, 1.0]))
    assert abs(depth - 0.02) < 0.002 and abs(lateral - 0.005) < 0.0005

