"""VMD: 書き出したファイルを自前のリーダーで読み戻し、ボーン名・フレーム数・値が一致すること。"""
import struct
import warnings

import numpy as np
import pytest

from nlf2vmd import quat
from nlf2vmd.vmd import (LINEAR_INTERPOLATION, BoneTrack, MorphTrack, bone_interpolation,
                         merge_morphs, read_interpolation, read_vmd, thin_track, thin_weights,
                         write_vmd)


def test_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    names = ['センター', 'グルーブ', '上半身2', '左ひじ', '右足ＩＫ']
    tracks = []
    for i, name in enumerate(names):
        n = 10 + i
        tracks.append(BoneTrack(name, np.arange(n) * 2, rng.normal(size=(n, 3)),
                                quat.normalize(rng.normal(size=(n, 4)))))
    path = tmp_path / 'rt.vmd'
    n_keys = write_vmd(path, tracks, model_name='テストモデル')
    raw = path.read_bytes()
    assert raw[:25] == b'Vocaloid Motion Data 0002' and raw[25:30] == b'\0' * 5
    assert len(raw) == 30 + 20 + 4 + 111 * n_keys + 16

    vmd = read_vmd(path)
    assert vmd.model_name == 'テストモデル'
    assert sorted(vmd.bone_names()) == sorted(names)
    assert vmd.counts == dict(morph=0, camera=0, light=0, self_shadow=0)
    for t in tracks:
        frames, pos, rot = vmd.track(t.name)
        np.testing.assert_array_equal(frames, t.frames)
        np.testing.assert_allclose(pos, t.positions, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(rot, t.rotations, rtol=1e-6, atol=1e-6)
    assert (vmd.keys['interp'] == LINEAR_INTERPOLATION).all()


def test_ik_name_is_full_width():
    raw = '左足ＩＫ'.encode('cp932')
    assert len(raw) == 8 and b'IK' not in raw


def test_interpolation_layout_matches_mmd_tools():
    # MMD Tools（blender_mmd_tools）の書き出しと同じ 64 バイト。線形 = (20, 20), (107, 107)
    a, b = 20, 107
    expected = [
        a, a, 0, 0, a, a, a, a, b, b, b, b, b, b, b, b,
        a, a, a, a, a, a, a, b, b, b, b, b, b, b, b, 0,
        a, a, a, a, a, a, b, b, b, b, b, b, b, b, 0, 0,
        a, a, a, a, a, b, b, b, b, b, b, b, b, 0, 0, 0,
    ]
    assert LINEAR_INTERPOLATION.tolist() == expected
    curves = (((1, 2), (3, 4)), ((5, 6), (7, 8)), ((9, 10), (11, 12)), ((13, 14), (15, 16)))
    assert read_interpolation(bone_interpolation(*curves)) == curves


def test_long_bone_name_warns(tmp_path):
    t = BoneTrack('とても長いボーンの名前です', np.arange(2), np.zeros((2, 3)),
                  np.tile(quat.IDENTITY, (2, 1)))
    with pytest.warns(UserWarning):
        write_vmd(tmp_path / 'long.vmd', [t])


def _center(n=3):
    return BoneTrack('センター', np.arange(n), np.zeros((n, 3)), np.tile(quat.IDENTITY, (n, 1)))


def test_morph_roundtrip(tmp_path):
    morphs = [MorphTrack('まばたき', np.array([0, 5]), np.array([0.0, 1.0])),
              MorphTrack('あ', np.array([0, 1, 2]), np.array([0.25, 0.5, 0.0]))]
    path = tmp_path / 'm.vmd'
    n_keys = write_vmd(path, [_center()], 'モデル', morphs=morphs)
    assert n_keys == 3 + 5
    assert len(path.read_bytes()) == 30 + 20 + 4 + 111 * 3 + 4 + 23 * 5 + 12
    vmd = read_vmd(path)
    assert vmd.morph_names() == sorted(['まばたき', 'あ'])
    assert vmd.counts == dict(morph=5, camera=0, light=0, self_shadow=0)
    for t in morphs:
        frames, weights = vmd.morph_track(t.name)
        np.testing.assert_array_equal(frames, t.frames)
        np.testing.assert_allclose(weights, t.weights, atol=1e-7)


def test_merge_morphs_keeps_bones_other_morphs_and_camera(tmp_path):
    src = tmp_path / 'src.vmd'
    write_vmd(src, [_center()], 'モデル', morphs=[
        MorphTrack('まばたき', np.array([0, 5]), np.array([0.0, 1.0])),
        MorphTrack('あ', np.array([0, 1]), np.array([0.5, 0.5]))])
    # カメラのキーを 1 つ持つファイルにする（モーフより後ろはそのまま写すこと）
    tail = struct.pack('<I', 1) + bytes(range(61)) + struct.pack('<2I', 0, 0)
    src.write_bytes(src.read_bytes()[:-12] + tail)
    out = tmp_path / 'merged.vmd'
    merge_morphs(src, [MorphTrack('あ', np.array([0, 10, 20]), np.array([0.0, 0.8, 0.0]))], out)

    before, after = read_vmd(src), read_vmd(out)
    assert after.model_name == 'モデル'
    np.testing.assert_array_equal(after.keys, before.keys)
    frames, weights = after.morph_track('あ')        # 同じ名前のモーフは置き換わる
    np.testing.assert_array_equal(frames, [0, 10, 20])
    np.testing.assert_allclose(weights, [0.0, 0.8, 0.0], atol=1e-7)
    np.testing.assert_array_equal(after.morph_track('まばたき')[0], [0, 5])   # ほかのモーフは残る
    assert after.counts == dict(morph=5, camera=1, light=0, self_shadow=0)
    assert out.read_bytes().endswith(tail)


def test_thin_weights_accuracy():
    t = np.arange(300)
    w = np.clip(np.sin(t / 7.0) * 1.2, 0.0, 1.0)
    keep = thin_weights(w, 0.01, forced=[100])
    assert keep[[0, 100, len(t) - 1]].all() and keep.sum() < len(t) / 2
    np.testing.assert_allclose(np.interp(t, t[keep], w[keep]), w, atol=0.01 + 1e-9)


def test_thinning_keeps_forced_frames_and_accuracy():
    T = 100
    t = np.arange(T)
    pos = np.stack([np.where(t < 50, t * 0.1, 5.0), 0 * t, 0 * t], axis=1).astype(float)
    rot = np.tile(quat.IDENTITY, (T, 1))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        keep = thin_track(pos, rot, 1e-3, 0.1, forced=[20, 70])
    assert keep[[0, 20, 50, 70, T - 1]].all()
    assert keep.sum() <= 6
