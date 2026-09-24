"""座標変換: 合成の歩行データで、出力の進行方向と体の向きが一致すること。"""
import numpy as np
import pytest

from nlf2vmd import convert, load_config, quat
from nlf2vmd.synthetic import synthetic_walk, to_camera_coords
from nlf2vmd.vmd import read_vmd

MMD_FRONT = np.array([0.0, 0.0, -1.0])   # MMD のモデルは -Z が正面


def _travel_and_facing(path):
    vmd = read_vmd(path)
    _, center, _ = vmd.track('センター')
    travel = center[-1] - center[0]
    _, _, lower = vmd.track('下半身')
    facing = quat.rotate(lower[len(lower) // 2], MMD_FRONT)
    unit = lambda v: v[[0, 2]] / np.linalg.norm(v[[0, 2]])  # noqa: E731
    return unit(travel), unit(facing)


@pytest.mark.parametrize('heading', [0.0, 90.0, -135.0])
def test_travel_direction_matches_body_facing(tmp_path, body_model, heading):
    motion = synthetic_walk(num_frames=150, heading_deg=heading)
    cfg = load_config(overrides=['diagnostics.enabled=false'])
    convert(motion, tmp_path / 'w.vmd', body_model=body_model, config=cfg, log=None)
    travel, facing = _travel_and_facing(tmp_path / 'w.vmd')
    assert np.dot(travel, facing) > 0.98


def test_walking_plus_z_moves_toward_mmd_front(tmp_path, body_model):
    """+Z（カメラに向かって）歩くと、MMD では正面（-Z）を向いて -Z へ進む。"""
    motion = synthetic_walk(num_frames=150, heading_deg=0.0)
    cfg = load_config(overrides=['diagnostics.enabled=false'])
    convert(motion, tmp_path / 'w.vmd', body_model=body_model, config=cfg, log=None)
    travel, facing = _travel_and_facing(tmp_path / 'w.vmd')
    assert travel[1] < -0.99 and facing[1] < -0.99


def test_flip_facing_turns_the_whole_motion(tmp_path, body_model):
    motion = synthetic_walk(num_frames=150, heading_deg=0.0)
    cfg = load_config(overrides=['diagnostics.enabled=false', 'floor.flip_facing=true'])
    convert(motion, tmp_path / 'f.vmd', body_model=body_model, config=cfg, log=None)
    travel, facing = _travel_and_facing(tmp_path / 'f.vmd')
    assert travel[1] > 0.99 and facing[1] > 0.99


def test_camera_coordinates_give_the_same_motion(tmp_path, body_model):
    """カメラ座標（Y 下向き・Z 奥向き）の入力は、X 軸まわり 180 度の変換で Y 上向きと同じ結果になる。"""
    motion = synthetic_walk(num_frames=120, heading_deg=30.0)
    cfg = load_config(overrides=['diagnostics.enabled=false'])
    a = convert(motion, tmp_path / 'yup.vmd', body_model=body_model, config=cfg, log=None)
    b = convert(to_camera_coords(motion), tmp_path / 'cam.vmd', body_model=body_model,
                config=cfg, log=None)
    for ta, tb in zip(a.tracks, b.tracks):
        assert ta.name == tb.name
        np.testing.assert_allclose(ta.positions, tb.positions, atol=1e-6)
        np.testing.assert_allclose(np.abs(np.sum(ta.rotations * tb.rotations, -1)), 1.0,
                                   atol=1e-6)
