"""ステージ1〜2 の順番（外れフレームの除外 → リサンプル）と、その前処理（prepare）の使い回し。"""
import numpy as np
import pytest

from nlf2vmd import convert, load_config, prepare, quat
from nlf2vmd import hand_detect as hd
from nlf2vmd.filters import frames_for_fps
from nlf2vmd.synthetic import add_joint_noise, synthetic_walk, to_camera_coords

ARM = (16, 17, 18, 19, 20, 21)


def _spiked(motion, frames, seed=0, deg=60.0):
    """frames のフレームだけ、腕の関節の 1 つを deg 度回す（1 フレームだけの推定の破綻）。"""
    rng = np.random.default_rng(seed)
    q = quat.from_rotvec(np.asarray(motion['pose'], float))
    for f in frames:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        j = int(rng.choice(ARM))
        q[f, j] = quat.mul(q[f, j], quat.from_rotvec(np.deg2rad(deg) * axis))
    return dict(motion, pose=quat.to_rotvec(q))


def _convert(motion, body_model, overrides=(), **kwargs):
    cfg = load_config(overrides=['diagnostics.enabled=false', *overrides])
    return convert(motion, None, body_model=body_model, config=cfg, log=None, **kwargs)


@pytest.mark.parametrize('fps', [24.0, 25.0])
def test_single_frame_glitch_is_removed_before_resampling(body_model, fps):
    """24・25fps の入力を 30fps にするとき、1 フレームだけの破綻は補間の前に除く（補間の後では前後 2 フレームに
    薄まって、単独フレームの外れとして見つからない。前の順番では 60 度の破綻が 20 度前後残った）。
    破綻を入れても、入れないときとほぼ同じ動きになる（置き換えたフレームは前後の中間なので、推定の揺れの分だけ違う）。"""
    clean = add_joint_noise(synthetic_walk(num_frames=int(fps * 6), fps=fps),
                            {j: 1.0 for j in range(24)}, smooth_frames=0.3)
    frames = np.arange(15, len(clean['pose']) - 15, 17)
    r_clean = _convert(clean, body_model)
    r_spiked = _convert(_spiked(clean, frames), body_model)
    assert r_spiked.fps == 30.0 and len(r_spiked.quats) == len(r_clean.quats)
    diff = np.rad2deg(quat.angle_between(r_spiked.quats, r_clean.quats))
    assert diff.max() < 3.0


def test_smoothing_at_source_fps_when_downsampling(body_model):
    """入力の fps が出力より高い（60fps）ときは、入力の fps で平滑化してから間引く。1 フレームだけの破綻も除く。"""
    clean = add_joint_noise(synthetic_walk(num_frames=360, fps=60.0), {j: 1.0 for j in range(24)},
                            smooth_frames=0.6)
    r_clean = _convert(clean, body_model)
    r_spiked = _convert(_spiked(clean, np.arange(21, 340, 23)), body_model)
    assert r_clean.fps == 30.0 and len(r_clean.quats) == 180
    assert np.rad2deg(quat.angle_between(r_spiked.quats, r_clean.quats)).max() < 1.0
    # 1b の結果は入力の fps のまま（診断は出力のフレーム番号に直して描く）
    assert len(r_clean.outliers.valid) == 360


def test_frames_for_fps():
    assert frames_for_fps(3, 30.0) == 3
    assert frames_for_fps(3, 24.0) == 3          # 低い fps では減らさない
    assert frames_for_fps(3, 60.0) == 6
    assert frames_for_fps(5, 60.0, odd=True) == 11
    assert frames_for_fps(5, 30.0, odd=True) == 5


def test_prepared_is_reused_with_the_same_result(body_model):
    """prepare の結果を convert に渡すと使い回し、渡さないときと同じ VMD のキーになる。1〜2 に効かない設定
    （体の向きの反転など）が違っても使い回す。"""
    motion = to_camera_coords(add_joint_noise(synthetic_walk(num_frames=150, fps=25.0),
                                              {j: 2.0 for j in range(24)}), height=1.2, pitch_deg=8.0)
    prepared = prepare(motion, body_model)
    logs = []
    cfg = load_config(overrides=['diagnostics.enabled=false', 'floor.flip_facing=true',
                                 'center.mode=B'])
    reused = convert(motion, None, body_model=body_model, config=cfg, log=logs.append,
                     prepared=prepared)
    fresh = convert(motion, None, body_model=body_model, config=cfg, log=None)
    assert any('求めたものを使います' in str(line) for line in logs)
    assert [t.name for t in reused.tracks] == [t.name for t in fresh.tracks]
    for a, b in zip(reused.tracks, fresh.tracks):
        np.testing.assert_allclose(a.positions, b.positions, atol=1e-9)
        np.testing.assert_allclose(a.rotations, b.rotations, atol=1e-9)
    # 手を切り出す位置も同じ前処理から（stabilized_joints は prepare を呼ぶ）
    np.testing.assert_allclose(hd.prepared_joints(prepared), hd.stabilized_joints(motion, body_model))


@pytest.mark.parametrize('change', ['input', 'jitter'])
def test_prepared_is_not_reused_when_it_does_not_match(body_model, change):
    """入力・ステージ1〜2 の設定が違う prepare の結果は使わず、求め直す。"""
    motion = synthetic_walk(num_frames=120, fps=30.0)
    other = add_joint_noise(motion, {j: 3.0 for j in range(24)}) if change == 'input' else motion
    prepared = prepare(other, body_model)
    overrides = ['jitter.one_euro.groups.arm.min_cutoff=1.0'] if change == 'jitter' else []
    logs = []
    cfg = load_config(overrides=['diagnostics.enabled=false', *overrides])
    r = convert(motion, None, body_model=body_model, config=cfg, log=logs.append, prepared=prepared)
    fresh = _convert(motion, body_model, overrides)
    assert not any('求めたものを使います' in str(line) for line in logs)
    np.testing.assert_allclose(r.quats, fresh.quats)


def test_hand_joints_need_camera_coords(body_model):
    prepared = prepare(synthetic_walk(num_frames=60), body_model)   # Y 上向きの入力
    with pytest.raises(ValueError):
        hd.prepared_joints(prepared)
