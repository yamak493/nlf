"""出力する VMD の種類（フル / フル [接地優先] / フル [移動なし] / 上半身のみ / 表情のみ）。"""
import numpy as np
import pytest

from nlf2vmd import convert, load_config, quat
from nlf2vmd.__main__ import main as cli_main
from nlf2vmd.legs import LEG_BONES
from nlf2vmd.lipsync import make_lipsync, save_analysis
from nlf2vmd.synthetic import synthetic_walk
from nlf2vmd.variants import (LOWER_BODY_BONES, footprint_shift, footprints, heading_matrices,
                              no_move_motion, upper_body_quats, variant_tracks, write_variant)
from nlf2vmd.vmd import read_vmd

from .test_lipsync import RATE, synthetic


def _convert(body_model, **walk_kwargs):
    motion = synthetic_walk(num_frames=240, noise_deg=2.0, **walk_kwargs)
    cfg = load_config(overrides=['diagnostics.enabled=false'])
    return convert(motion, None, body_model=body_model, config=cfg, log=None)


@pytest.fixture(scope='module')
def walk(body_model):
    return _convert(body_model, sway=0.05)


def _yaw_deg(m):
    return np.rad2deg(np.arctan2(m[:, 0, 2] - m[:, 2, 0], m[:, 0, 0] + m[:, 2, 2]))


def test_convert_without_output_writes_nothing(tmp_path, monkeypatch, body_model):
    monkeypatch.chdir(tmp_path)
    motion = synthetic_walk(num_frames=60)
    r = convert(motion, None, body_model=body_model, log=None)
    assert r.vmd_path == '' and r.num_keys == 0 and r.metrics and r.tracks
    assert list(tmp_path.iterdir()) == []


def _same(actual, expected, **kw):
    np.testing.assert_allclose(actual, np.broadcast_to(expected, np.shape(actual)), **kw)


def _rot(axis, deg):
    deg = np.atleast_1d(np.asarray(deg, np.float64))
    return quat.from_rotvec(np.deg2rad(deg)[:, None] * np.asarray(axis, np.float64))


@pytest.mark.parametrize('axis, deg', [([1, 0, 0], 85.0), ([0, 0, 1], 30.0)])
def test_heading_ignores_bowing_and_tilting(axis, deg):
    # 向きを変えてから、おじぎ（X 軸まわり）か横への傾き（Z 軸まわり）: 向きはそのまま
    y = _rot([0, 1, 0], [70.0, -150.0])
    rot = quat.to_matrix(quat.mul(y, _rot(axis, [deg, -deg])))
    np.testing.assert_allclose(heading_matrices(rot), quat.to_matrix(y), atol=1e-9)
    # 両方に少し傾いたときも、向きはほぼ変わらない
    both = quat.to_matrix(quat.mul(y, quat.mul(_rot([1, 0, 0], 20.0), _rot([0, 0, 1], 10.0))))
    assert (np.abs(_yaw_deg(heading_matrices(both)) - [70.0, -150.0]) < 2.0).all()


def test_full_is_the_converted_motion(walk):
    assert variant_tracks(walk, 'full') is walk.tracks
    assert variant_tracks(walk, 'face') == []
    with pytest.raises(ValueError):
        variant_tracks(walk, 'lower_body')


def test_no_move_keeps_the_body_in_place_without_sliding(walk):
    k = walk.scale
    tracks = {t.name: t for t in variant_tracks(walk, 'no_move', log=None)}
    full = {t.name: t for t in walk.tracks}
    assert tracks.keys() == full.keys()
    # センターは動かない（上下はグルーブ）。回転はフルと同じ
    assert np.abs(full['センター'].positions[:, 2]).max() > 50.0
    np.testing.assert_array_equal(tracks['センター'].positions, 0.0)
    for name, t in full.items():
        # 脚の回転は、この種類のセンター・足ＩＫで解き直す
        if 'ＩＫ' not in name and name not in ('センター', 'グルーブ') + LEG_BONES:
            np.testing.assert_array_equal(tracks[name].rotations, t.rotations)

    center, ik, info = no_move_motion(walk)
    assert info['exceed_after'] == 0
    for foot in range(2):
        for s, e in walk.contact.segments[foot]:
            _same(ik.target[s:e + 1, foot], ik.target[s, foot], atol=1e-9)
    # 足ＩＫの上下と回転はフルと同じで、水平には足跡の周りで足踏みするだけ
    np.testing.assert_allclose(ik.delta[..., 1], walk.foot_ik.delta[..., 1])
    np.testing.assert_array_equal(ik.rotation, walk.foot_ik.rotation)
    assert (np.ptp(ik.target[..., 2], axis=0) / k < 0.5).all()
    step = np.linalg.norm(np.diff(ik.target, axis=0), axis=-1).max() / k
    assert step <= np.linalg.norm(np.diff(walk.foot_ik.target, axis=0), axis=-1).max() / k
    # 上下の動き（グルーブ）はフルとほぼ同じ（届く高さへのクランプを掛け直すだけ）
    assert np.abs(center[:, 1] - walk.center.delta[:, 1]).max() / k < 0.08


def test_split_stance_becomes_one_footprint():
    # 1 つの接地が判定で 2 つに分かれても（ロック位置の差 2cm）、足跡は 1 つで足は動かない
    T = 40
    travel = np.zeros((T, 3))
    travel[:, 2] = np.linspace(0.0, 0.8, T)
    target = np.zeros((T, 3))
    target[20:, 0] = 0.02
    segments = [(0, 18), (22, 39)]
    assert footprints(segments, target, 0.1, 0.03) == [(0, 39)]
    shift = footprint_shift(travel, segments, target, 0.1, 0.03)
    _same(shift, travel.mean(0))


def test_footprints_follow_the_foot_between_steps():
    # 足が 0.6 進む間に体も同じだけ進む: 移動なしでは足は水平に動かず、足跡の上で足踏みする
    T = 60
    target = np.zeros((T, 3))
    swing = np.arange(20, 40)
    target[swing, 2] = 0.6 * (1 - np.cos(np.pi * (swing - 19) / 21)) / 2
    target[40:, 2] = 0.6
    target[swing, 1] = 0.1 * np.sin(np.pi * (swing - 19) / 21)
    travel = target * [0.0, 0.0, 1.0]
    segments = [(0, 19), (40, 59)]
    shift = footprint_shift(travel, segments, target, 0.1, 0.03)
    out = target - shift
    _same(out[:20], out[0])
    _same(out[40:], out[40])
    assert np.abs(out[:, 2] - out[0, 2]).max() < 0.05
    np.testing.assert_allclose(out[:, 1], target[:, 1])


def test_edges_stay_on_the_first_and_last_footprints():
    T = 30
    travel = np.zeros((T, 3))
    travel[:, 2] = np.linspace(0.0, 0.9, T)
    target = np.zeros((T, 3))
    target[:10, 2] = np.linspace(-0.4, 0.0, 10)      # 最初の接地の前は歩いてくる途中
    target[20:, 2] = np.linspace(0.0, 0.4, 10)       # 最後の接地の後は次の歩へ
    shift = footprint_shift(travel, [(10, 19)], target, 0.1, 0.03)
    out = target - shift
    _same(out[:, 2], out[10, 2], atol=1e-9)
    np.testing.assert_allclose(footprint_shift(travel, [], target, 0.1, 0.03), travel)


@pytest.mark.parametrize('heading', [0.0, 90.0])
def test_upper_body_only_moves_the_upper_body(body_model, heading):
    r = _convert(body_model, heading_deg=heading)
    tracks = variant_tracks(r, 'upper_body')
    names = {t.name for t in tracks}
    assert names == set(r.retargeter.bones) - set(LOWER_BODY_BONES)
    assert not any(n in names for n in ('センター', 'グルーブ', '下半身', '左足ＩＫ', '右足ＩＫ'))
    q = upper_body_quats(r)
    # 体全体の向き（heading）は除き、下半身に対するひねりは残す
    rt = r.retargeter
    glob = rt.global_matrices(r.kin.glob_rot)
    assert abs(np.median(_yaw_deg(quat.to_matrix(r.local_quats['上半身']))) - heading) < 5.0
    twist = _yaw_deg(np.swapaxes(glob['下半身'], -1, -2) @ glob['上半身'])
    np.testing.assert_allclose(_yaw_deg(quat.to_matrix(q['上半身'])), twist, atol=3.0)
    for name in q:
        if rt.keyed_parent[name] in q:
            np.testing.assert_array_equal(q[name], r.local_quats[name])


def test_write_variants_with_lipsync(tmp_path, walk):
    data = synthetic([(0.5, 1.5, -10.0)], [(0.5, 'k'), (0.6, 'a'), (1.0, 'o')])
    analysis = tmp_path / 'lipsync_analysis.npz'
    save_analysis(analysis, data['phone_times'], data['phones'], data['level_db'], RATE,
                  data['duration'])
    lip = make_lipsync(analysis, log=None)
    assert lip.tracks and list(tmp_path.glob('*.vmd')) == []

    counts = {}
    for kind in ('full', 'locked', 'no_move', 'upper_body', 'face'):
        morphs = lip.tracks if kind != 'upper_body' else ()
        write_variant(walk, kind, tmp_path / f'{kind}.vmd', morphs=morphs, log=None)
        v = read_vmd(tmp_path / f'{kind}.vmd')
        counts[kind] = (len(v.bone_names()), len(v.morph_names()))
        assert v.model_name == walk.info['model_name']
    n_bones = len(walk.tracks)
    # 上半身のみ: センター・グルーブ・下半身・足ＩＫ（2）・脚の回転（足・ひざ・足首 × 2）を除く
    assert counts == dict(full=(n_bones, 6), locked=(n_bones, 6), no_move=(n_bones, 6),
                          upper_body=(n_bones - 5 - len(LEG_BONES), 0), face=(0, 6))


def test_cli_writes_full_and_locked_by_default(tmp_path):
    out = tmp_path / 'motion_full.vmd'
    assert cli_main(['--demo', '-o', str(out), '--no-plots']) == 0
    assert sorted(p.name for p in tmp_path.glob('*.vmd')) == ['motion_full.vmd',
                                                             'motion_full_locked.vmd']
    assert len(read_vmd(tmp_path / 'motion_full_locked.vmd').bone_names()) == len(
        read_vmd(out).bone_names())


def test_cli_writes_selected_variants(tmp_path):
    out = tmp_path / 'demo.vmd'
    assert cli_main(['--demo', '-o', str(out), '--variants', 'no_move,upper_body',
                     '--no-plots']) == 0
    assert not out.exists()
    assert (tmp_path / 'demo_no_move.vmd').exists() and (tmp_path / 'demo_upper_body.vmd').exists()
    with pytest.raises(SystemExit):
        cli_main(['--demo', '-o', str(out), '--variants', 'face'])
