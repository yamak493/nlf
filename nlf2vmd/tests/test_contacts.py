"""腕・手のひら・指先と、体・相手の腕の接触（ステージ9b）。

体は PMX の剛体（無ければメッシュ・標準の体格の形）、腕は上腕・前腕・手のひら・指の各節のカプセル。重なったら
手首 → ひじ → 肩 の順に動かしやすくして離し、触れているだけの所はそのまま残すこと。
"""
import struct

import numpy as np
import pytest

from nlf2vmd import convert, load_config, quat
from nlf2vmd.contacts import (FINGERS, PARTS, _rigid_rotation, arm_capsules, body_capsules)
from nlf2vmd.hands import build_rig, preset_angles, SHAPES
from nlf2vmd.pipeline import apply_hand_poses
from nlf2vmd.pmx import read_pmx
from nlf2vmd.skeleton import STANDARD_BONES, Skeleton
from nlf2vmd.synthetic import add_arm_cross, add_arm_reach, synthetic_walk
from nlf2vmd.tests.test_hands import _LEFT_HAND
from nlf2vmd.tests.test_pmx import _text, make_pmx, standard_pmx_bones
from nlf2vmd.vmd import BoneTrack, to_mmd_quat

PART = {p: k for k, p in enumerate(PARTS)}


def hand_bones():
    """標準ボーン＋左右の指ボーン（Black.pmx と同じ配置を、標準ボーンの手首に合わせてずらしたもの）。"""
    bones = standard_pmx_bones()
    off = np.array(STANDARD_BONES['左手首'][0]) - np.array(_LEFT_HAND[0][1])
    for s in '左右':
        sign = 1.0 if s == '左' else -1.0
        for name, pos, parent in _LEFT_HAND[1:]:
            names = [b[0] for b in bones]
            p = (np.array(pos) + off) * [sign, 1.0, 1.0]
            bones.append((name.replace('左', s, 1), tuple(p), names.index(parent.replace('左', s, 1)),
                          None, False))
    return bones


def make_rigids(rigids, bsize=2):
    """モーフ 0・表示枠 0・剛体の部分。rigids: [(名前, ボーンの番号, 形, 大きさ, 位置, 回転, モード)]"""
    bi = {1: 'b', 2: 'h', 4: 'i'}[bsize]
    out = struct.pack('<i', 0) + struct.pack('<i', 0) + struct.pack('<i', len(rigids))
    for name, bone, shape, size, pos, rot, mode in rigids:
        out += _text(name, 0) + _text('', 0) + struct.pack(f'<{bi}BHB', bone, 0, 0, shape)
        out += struct.pack('<9f', *size, *pos, *rot) + struct.pack('<5f', 1, 0.5, 0.5, 0, 0.5)
        out += bytes([mode])
    return out + struct.pack('<i', 0)                       # ジョイント 0


def hand_pmx(tmp_path, rigids=None, vertices=None, extra_bones=()):
    bones = hand_bones() + list(extra_bones)
    data = make_pmx(bones, vertices=vertices)
    if rigids is not None:
        data += make_rigids(rigids(lambda n: [b[0] for b in bones].index(n)))
    path = tmp_path / 'hand_body.pmx'
    path.write_bytes(data)
    return path


# 9b だけを見る（ステージ9h（胴に対する手の位置）は、回転のコピーで胴に入り込んだ手を胴の外へ移すので、
# 入り込んだ姿勢を作るこのテストでは切る）
ONLY_9B = ['hand_reach.enabled=false']


def _convert(body_model, motion, pmx, extra=()):
    cfg = load_config(overrides=['diagnostics.enabled=false', *ONLY_9B, *extra])
    return convert(motion, None, pmx=pmx, body_model=body_model, config=cfg, log=None)


def _cm(r, d):
    return d / r.scale * 100.0


# ---- PMX の剛体 ----
def test_read_rigid_bodies_and_body_capsules(tmp_path):
    skirt = ('スカート', (0.0, 10.5, 0.0), 'ignored', None, False)
    hair = ('髪', (0.0, 17.5, 0.5), 'ignored', None, False)

    def rigids(idx):
        return [('胸', idx('上半身2'), 1, (1.3, 1.6, 0.8), (0.0, 14.0, 0.3), (0.0, 0.0, 0.0), 0),
                ('頭', idx('頭'), 0, (1.0, 0.0, 0.0), (0.0, 17.5, 0.2), (0.0, 0.0, 0.0), 0),
                ('腰', idx('下半身'), 2, (1.2, 1.5, 0.0), (0.0, 10.8, 0.2),
                 (0.0, 0.0, np.pi / 2), 0),
                ('左腕', idx('左腕'), 2, (0.4, 2.0, 0.0), (2.4, 14.4, 0.6), (0.0, 0.0, 0.8), 0),
                ('スカート', idx('スカート'), 0, (0.5, 0.0, 0.0), (0.0, 10.5, 0.0), (0, 0, 0), 1),
                ('髪', idx('髪'), 0, (0.5, 0.0, 0.0), (0.0, 17.5, 0.5), (0, 0, 0), 1)]

    bones = hand_bones()
    names = [b[0] for b in bones]
    extra = [(skirt[0], skirt[1], names.index('下半身'), None, False),
             (hair[0], hair[1], names.index('頭'), None, False)]
    path = hand_pmx(tmp_path, rigids, extra_bones=extra)
    model = read_pmx(path)
    assert [rb.name for rb in model.rigid_bodies] == ['胸', '頭', '腰', '左腕', 'スカート', '髪']
    rb = model.rigid_bodies[2]
    assert rb.shape == 2 and rb.mode == 0 and np.allclose(rb.size[:2], (1.2, 1.5))
    skel = Skeleton.from_pmx(model)
    assert skel.rigid_bodies[0]['bone'] == '上半身2'
    cfg = load_config().contacts
    caps, source = body_capsules(skel, cfg, 10.0)
    assert source == 'rigid'
    names = [c.name for c in caps]
    assert '左腕' not in names and '髪' not in names       # 腕の剛体・髪の物理の剛体は使わない
    assert names.count('胸') == 2                           # 薄い箱はカプセル 2 本
    skirt_caps = [c for c in caps if c.name == 'スカート']
    assert len(skirt_caps) == 1 and skirt_caps[0].weight == cfg.soft_ratio   # スカートは弱く離す
    waist = next(c for c in caps if c.name == '腰')
    d = waist.b - waist.a
    assert abs(d[0]) > 1.4 and abs(d[1]) < 1e-6          # Z まわりに 90 度: カプセルの軸が横向き
    assert abs(waist.radius - (1.2 + cfg.cloth_margin_m * 10.0)) < 1e-6
    no_skirt = load_config(overrides=['contacts.skirt=none']).contacts
    assert 'スカート' not in [c.name for c in body_capsules(skel, no_skirt, 10.0)[0]]


def test_rigid_rotation_order_is_z_x_y():
    """MMD の剛体の回転は Z → X → Y の順（D3DX の YawPitchRoll）。内部座標は z を反転。"""
    rx, ry, rz = 0.3, -0.5, 0.7
    R = _rigid_rotation((rx, ry, rz))
    flip = np.diag([1.0, 1.0, -1.0])
    c, s = np.cos, np.sin
    Rx = np.array([[1, 0, 0], [0, c(rx), -s(rx)], [0, s(rx), c(rx)]])
    Ry = np.array([[c(ry), 0, s(ry)], [0, 1, 0], [-s(ry), 0, c(ry)]])
    Rz = np.array([[c(rz), -s(rz), 0], [s(rz), c(rz), 0], [0, 0, 1]])
    v = np.array([0.2, 0.9, -0.4])
    np.testing.assert_allclose(R @ (flip @ v), flip @ (Ry @ (Rx @ (Rz @ v))), atol=1e-12)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12


# ---- 腕・手・指のカプセル ----
def _ring(center, axis, radius, n=10):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    u = np.cross(axis, [0.3, 0.2, 1.0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.asarray(center) + radius * (np.cos(a)[:, None] * u + np.sin(a)[:, None] * v)


def test_arm_capsules_reach_the_fingertips(tmp_path):
    bones = hand_bones()
    names = [b[0] for b in bones]
    pos = {b[0]: np.array(b[1]) for b in bones}
    verts = []
    a, b = pos['左人指２'], pos['左人指３']                  # 人差し指の第 2 節に太さ 0.1 の輪
    for t in np.linspace(0.1, 0.9, 6):
        for p in _ring(a + t * (b - a), b - a, 0.1):
            verts.append((tuple(p), 0, (names.index('左人指２'),), ()))
    skel = Skeleton.from_pmx(read_pmx(hand_pmx(tmp_path, vertices=verts)))
    cfg = load_config()
    caps = arm_capsules(skel, 0, cfg.contacts, cfg.arm_collision, cfg.hands.bones, 10.0)
    parts = [c.part for c in caps]
    assert parts.count(PART['palm']) == 3
    for f in FINGERS:
        assert parts.count(PART[f]) == 3                    # 3 節ずつ
    assert len(caps) == 2 + 3 + 15
    tip = [c for c in caps if c.bone == '左人指３'][0]
    np.testing.assert_allclose(tip.b, skel.internal('左人指３先'))   # 指先は 〇指３先 のボーン
    idx2 = [c for c in caps if c.bone == '左人指２'][0]
    assert abs(idx2.radius - 0.1) < 0.01                   # メッシュから測った太さ
    mid2 = [c for c in caps if c.bone == '左中指２'][0]
    assert abs(mid2.radius - cfg.contacts.finger_radius_m[2] * 10.0) < 1e-9   # 頂点が無ければ設定の値
    # 指の当たり判定を切ると、手は 1 本の棒
    off = load_config(overrides=['contacts.fingers=false'])
    caps = arm_capsules(skel, 0, off.contacts, off.arm_collision, off.hands.bones, 10.0)
    assert [c.part for c in caps] == [0, 1, 2]


# ---- 接触の解決 ----
def _press(T=10, elbow=(0.9, -0.3, 0.3), hand=(0.02, 0.12, 0.16), twist=0.0):
    return add_arm_reach(synthetic_walk(num_frames=T, speed=0.0), 0, elbow, hand, twist_deg=twist)


def test_fingertip_in_the_chest_is_fixed_mostly_at_the_wrist(tmp_path, body_model):
    """親指の先だけが胸に入っている: 手首を主に動かし、ひじ・肩はほとんど動かさない。"""
    pmx = hand_pmx(tmp_path)
    r = _convert(body_model, _press(), pmx)
    c = r.contacts
    before, after = _cm(r, c.depth_before[:, 0]), _cm(r, c.depth_after[:, 0])
    assert before[:, PART['thumb']].min() > 1.0 and before[:, PART['fore']].max() < 0.0
    assert after.max() < 0.1
    corr = c.correction_deg[:, 0]
    assert corr[:, 2].min() > 2.0                          # 手首
    assert (corr[:, 2] > 3.0 * np.maximum(corr[:, 0], corr[:, 1])).all()
    np.testing.assert_allclose(r.local_quats['右手首'],
                               _convert(body_model, _press(), pmx,
                                        ['contacts.enabled=false']).local_quats['右手首'])


def test_forearm_in_the_chest_is_pushed_out(tmp_path, body_model):
    pmx = hand_pmx(tmp_path)
    r = _convert(body_model, _press(elbow=(0.2, -0.8, 0.55), hand=(-0.05, 0.08, 0.08)), pmx)
    c = r.contacts
    assert _cm(r, c.depth_before[:, 0, PART['fore']]).min() > 2.0
    assert _cm(r, c.depth_after).max() < 0.1
    assert c.correction_deg[:, 0, 1].max() > c.correction_deg[:, 0, 2].max()   # 前腕はひじで離す
    b, a = c.overlap_frames(0.005 * r.scale)
    assert b == 10 and a == 0


def test_touching_is_kept(tmp_path, body_model):
    """胸の前で離れている手は動かさない（近づけることもしない）。"""
    pmx = hand_pmx(tmp_path)
    r = _convert(body_model, _press(elbow=(0.9, -0.3, 0.3), hand=(0.02, 0.12, 0.25)), pmx)
    assert _cm(r, r.contacts.depth_before[:, 0, 2:]).max() < 0.0
    assert r.contacts.correction_deg[:, 0, 1:].max() < 1e-6


def test_hand_passing_through_the_chest_stays_in_front(tmp_path, body_model):
    """推定の手が胸を前から後ろへ通り抜けても、手は胸の前に留まり、補正はなめらか。"""
    T = 40
    z = np.linspace(0.25, 0.05, T)
    hand = np.stack([np.full(T, -0.05), np.full(T, 0.08), z], axis=1)
    pmx = hand_pmx(tmp_path)
    r = _convert(body_model, _press(T, elbow=(0.2, -0.8, 0.55), hand=hand), pmx)
    c = r.contacts
    assert _cm(r, c.depth_before[-5:, 0]).max() > 2.0
    assert _cm(r, c.depth_after[:, 0]).max() < 0.3
    for name in ('左腕', '左ひじ', '左手首'):
        steps = np.rad2deg(quat.angle_between(r.local_quats[name][1:], r.local_quats[name][:-1]))
        base = _convert(body_model, _press(T, elbow=(0.2, -0.8, 0.55), hand=hand), pmx,
                        ['contacts.enabled=false']).local_quats[name]
        base_steps = np.rad2deg(quat.angle_between(base[1:], base[:-1]))
        assert steps.max() < base_steps.max() + 4.0, name


def test_walk_with_hanging_arms_is_unchanged(body_model, tmp_path):
    pmx = hand_pmx(tmp_path)
    r = _convert(body_model, synthetic_walk(num_frames=60), pmx)
    assert r.contacts.correction_deg.max() < 1.0
    assert r.contacts.overlap_frames(0.005 * r.scale)[1] == 0


def test_arm_pairs_move_only_the_yielding_shoulder(body_model):
    """腕どうしの組は、自重する腕の肩だけで離す（ステージ9a と同じ。ひじ・手首は変えない）。"""
    motion = add_arm_cross(synthetic_walk(num_frames=30))
    extra = ['arm_collision.mode=left', 'contacts.body_source=none']
    base = _convert(body_model, motion, None, extra + ['contacts.enabled=false'])
    r = _convert(body_model, motion, None, extra)
    for name in ('左ひじ', '左手首', '右腕', '右ひじ', '右手首'):
        np.testing.assert_allclose(r.local_quats[name], base.local_quats[name], atol=1e-9)
    assert r.contacts.overlap_frames(0.005 * r.scale)[1] == 0


def _finger_tracks(skel, shape, T):
    cfg = load_config().hands
    table = preset_angles(cfg)
    tracks = []
    for side in (0, 1):
        rig = build_rig(skel, side, cfg)
        for name, q in rig.local_quats(table[SHAPES.index(shape)][None]).items():
            tracks.append(BoneTrack(name, np.array([0]), np.zeros((1, 3)), to_mmd_quat(q)))
    return tracks


def test_finger_pose_is_used(tmp_path, body_model):
    """指のキー（手の形）を入れると、その形の指で当たり判定をやり直す（指先が胸に入るかは指の形で変わる）。"""
    pmx = hand_pmx(tmp_path)
    motion = _press(elbow=(0.7, -0.2, 0.6), hand=(0.02, 0.12, 0.13), twist=-90.0)
    r = _convert(body_model, motion, pmx)
    fingers = [PART[f] for f in FINGERS[1:]]
    open_depth = _cm(r, r.contacts.depth_before[:, 0, fingers]).max(axis=0)
    assert open_depth.max() > 0.5                           # 伸ばした指（初期姿勢）は胸に入る
    skel = Skeleton.from_pmx(read_pmx(pmx))
    contacts = apply_hand_poses(r, _finger_tracks(skel, 'fist', len(motion['pose'])), log=None)
    fist_depth = _cm(r, contacts.depth_before[:, 0, fingers]).max(axis=0)
    assert np.abs(fist_depth - open_depth).max() > 0.5     # 握った指で測り直している
    assert _cm(r, contacts.depth_after).max() < 0.1
    assert r.info['contacts'] is contacts.info
    # 書き出しのキーも直した回転になる
    track = next(t for t in r.tracks if t.name == '左手首')
    np.testing.assert_allclose(track.rotations, to_mmd_quat(quat.make_continuous(r.local_quats['左手首'])),
                               atol=1e-6)


def test_disabled_keeps_the_rotations(tmp_path, body_model):
    pmx = hand_pmx(tmp_path)
    r = _convert(body_model, _press(), pmx, ['contacts.enabled=false'])
    assert not r.contacts.enabled
    np.testing.assert_allclose(r.local_quats['左手首'], r.local_before_contacts['左手首'])


def test_invalid_body_source(tmp_path, body_model):
    with pytest.raises(ValueError):
        _convert(body_model, _press(), hand_pmx(tmp_path), ['contacts.body_source=cloth'])


def test_body_from_mesh_when_there_are_no_rigid_bodies(tmp_path):
    bones = hand_bones()
    names = [b[0] for b in bones]
    rng = np.random.default_rng(0)
    verts = []
    for bone, center, half in (('上半身2', (0.0, 14.0, 0.3), (1.5, 1.4, 0.9)),
                               ('頭', (0.0, 17.6, 0.2), (0.9, 1.0, 0.9)),
                               ('左ひじ', (4.3, 12.8, 0.6), (0.3, 0.3, 0.3))):   # 腕の頂点は使わない
        for p in rng.uniform(-1, 1, (80, 3)) * half + center:
            verts.append((tuple(p), 0, (names.index(bone),), ()))
    skel = Skeleton.from_pmx(read_pmx(hand_pmx(tmp_path, vertices=verts)))
    caps, source = body_capsules(skel, load_config().contacts, 10.0)
    assert source == 'mesh'
    assert sorted({c.bone for c in caps}) == ['上半身2', '頭']
    chest = [c for c in caps if c.bone == '上半身2']
    assert len(chest) == 2 and all(0.7 < c.radius < 1.0 for c in chest)   # 厚みの 95 パーセンタイル


def test_contacts_plot(tmp_path, body_model):
    pytest.importorskip('matplotlib')
    cfg = load_config(overrides=ONLY_9B)
    r = convert(_press(), None, pmx=hand_pmx(tmp_path), body_model=body_model, config=cfg,
                diag_dir=tmp_path / 'diag', log=None)
    assert (tmp_path / 'diag' / 'contacts.png').exists()
    assert r.metrics['contact_overlap_frames']['before'] > 0
    assert r.metrics['contact_overlap_frames']['after'] == 0


def test_cli_with_hands(tmp_path, body_model):
    """--hands と --pmx: 手首の向きの補正・手の形のキー・その指の形での接触の解き直しを通しで行う。"""
    from nlf2vmd.__main__ import main
    from nlf2vmd.hands import save_analysis
    from nlf2vmd.tests.test_wrist import _analysis, _fk, _true_and_nlf
    from nlf2vmd.vmd import read_vmd
    from nlf2vmd.wrist import WRIST
    T = 30
    motion, true, nlf = _true_and_nlf(T)
    fk, _ = _fk(body_model, T)
    a = _analysis(fk(true)[:, list(WRIST)], np.full((T, 2), 0.95))
    hands_npz = tmp_path / 'hands_analysis.npz'
    save_analysis(hands_npz, a['screen'], a['world'], a['presence'], np.zeros((T, 2)), a['fps'],
                  roi=a['roi'], image_size=a['image_size'])
    src = tmp_path / 'motion.npz'
    np.savez(src, **{k: np.asarray(v) for k, v in dict(motion, pose=quat.to_rotvec(nlf)).items()})
    body_model.save_npz(tmp_path / 'smpl_body_model.npz')
    out = tmp_path / 'out.vmd'
    assert main([str(src), '-o', str(out), '--pmx', str(hand_pmx(tmp_path)), '--hands',
                 str(hands_npz), '--no-plots']) == 0
    vmd = read_vmd(out)
    assert '左人指１' in vmd.bone_names() and '左手首' in vmd.bone_names()
