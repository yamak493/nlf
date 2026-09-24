"""PMX の読み込み（頂点・材質を正しく読み飛ばしてボーンを取り出せること）。"""
import struct

import numpy as np
import pytest

from nlf2vmd import convert, load_config
from nlf2vmd.pmx import read_pmx
from nlf2vmd.skeleton import STANDARD_BONES, Skeleton
from nlf2vmd.synthetic import synthetic_walk


def _text(s, enc):
    b = s.encode('utf-16-le' if enc == 0 else 'utf-8')
    return struct.pack('<i', len(b)) + b


def make_pmx(bones, enc=0, bsize=2, n_uv=1):
    """最小限の PMX 2.0。bones: [(名前, 位置, 親の番号, 表示先の番号 or None, IK か)]"""
    out = b'PMX ' + struct.pack('<f', 2.0) + bytes([8, enc, n_uv, 2, 1, 1, bsize, 1, 1])
    out += _text('テストモデル', enc) + _text('test', enc) + _text('', enc) + _text('', enc)
    bi = {1: 'b', 2: 'h', 4: 'i'}[bsize]
    verts = b''
    for kind in (0, 1, 2, 3):   # BDEF1 / BDEF2 / BDEF4 / SDEF
        verts += struct.pack(f'<{8 + 4 * n_uv}f', *[0.0] * (8 + 4 * n_uv)) + bytes([kind])
        verts += {0: struct.pack(f'<{bi}', 0),
                  1: struct.pack(f'<2{bi}f', 0, 1, 0.5),
                  2: struct.pack(f'<4{bi}4f', 0, 1, 0, 1, 0.25, 0.25, 0.25, 0.25),
                  3: struct.pack(f'<2{bi}10f', 0, 1, 0.5, *[0.0] * 9)}[kind]
        verts += struct.pack('<f', 1.0)
    out += struct.pack('<i', 4) + verts
    out += struct.pack('<i', 3) + struct.pack('<3H', 0, 1, 2)
    out += struct.pack('<i', 1) + _text('tex.png', enc)
    out += struct.pack('<i', 1) + _text('材質', enc) + _text('mat', enc)
    out += struct.pack('<11f', *[1.0] * 11) + bytes([0]) + struct.pack('<5f', *[0.0] * 5)
    out += struct.pack('<bb', 0, -1) + bytes([0]) + bytes([1, 0]) + _text('memo', enc)
    out += struct.pack('<i', 3)
    out += struct.pack('<i', len(bones))
    for name, pos, parent, tail, is_ik in bones:
        flags = 0x0002 | 0x0008 | (0x0001 if tail is not None else 0) | (0x0020 if is_ik else 0)
        out += _text(name, enc) + _text('', enc) + struct.pack('<3f', *pos)
        out += struct.pack(f'<{bi}iH', parent, 0, flags)
        out += struct.pack(f'<{bi}', tail) if tail is not None else struct.pack('<3f', 0, 1, 0)
        if is_ik:
            out += struct.pack(f'<{bi}if', 0, 40, 2.0) + struct.pack('<i', 2)
            out += struct.pack(f'<{bi}B', 1, 1) + struct.pack('<6f', *[0.0] * 6)
            out += struct.pack(f'<{bi}B', 2, 0)
    return out


def standard_pmx_bones():
    names = list(STANDARD_BONES)
    bones = []
    for name, (pos, parent, _) in STANDARD_BONES.items():
        bones.append((name, pos, names.index(parent) if parent else -1, None, name.endswith('ＩＫ')))
    return bones


@pytest.mark.parametrize('enc,bsize', [(0, 2), (1, 1), (0, 4)])
def test_read_bones(tmp_path, enc, bsize):
    bones = standard_pmx_bones()
    bones[1] = (bones[1][0], bones[1][1], bones[1][2], 2, False)   # 表示先をボーンで指定
    path = tmp_path / 'm.pmx'
    path.write_bytes(make_pmx(bones, enc=enc, bsize=bsize))
    model = read_pmx(path)
    assert model.name == 'テストモデル'
    assert [b.name for b in model.bones] == [b[0] for b in bones]
    for b, (_, pos, parent, _, _) in zip(model.bones, bones):
        np.testing.assert_allclose(b.position, pos, atol=1e-6)
        assert b.parent == parent
    assert model.bones[1].tail_index == 2
    skel = Skeleton.from_pmx(model)
    assert skel.nearest_ancestor('左ひじ', {'左肩', '上半身2'}) == '左肩'
    assert abs(skel.mean_leg_length() - Skeleton.standard().mean_leg_length()) < 1e-5


def test_convert_with_pmx_without_upper2_and_groove(tmp_path, body_model):
    bones = [b for b in standard_pmx_bones() if b[0] not in ('上半身2', 'グルーブ')]
    names = [b[0] for b in bones]
    remap = {'上半身2': '上半身', 'グルーブ': 'センター'}
    fixed = []
    for name, pos, _, tail, ik in bones:
        parent = STANDARD_BONES[name][1]
        parent = remap.get(parent, parent)
        fixed.append((name, pos, names.index(parent) if parent else -1, tail, ik))
    path = tmp_path / 'small.pmx'
    path.write_bytes(make_pmx(fixed))
    cfg = load_config(overrides=['diagnostics.enabled=false'])
    r = convert(synthetic_walk(num_frames=60), tmp_path / 's.vmd', pmx=path,
                body_model=body_model, config=cfg, log=None)
    track_names = [t.name for t in r.tracks]
    assert '上半身2' not in track_names and 'グルーブ' not in track_names
    center = next(t for t in r.tracks if t.name == 'センター')
    assert np.abs(center.positions[:, 1]).max() > 0     # 上下移動はセンターの Y に入る
    assert any('上半身2' in w for w in r.warnings)
