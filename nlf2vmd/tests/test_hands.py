"""手の形: 指の曲げ → 手の形の分類・時間方向の処理・指ボーンの回転・キーの書き出し。"""
import numpy as np
import pytest

from nlf2vmd import load_config, quat
from nlf2vmd import hand_detect as hd
from nlf2vmd.hands import (EASE_INTERPOLATION, FINGERS, SHAPES, apply_edits, build_hands,
                           build_rig, export_hands, finger_bends, finger_probs, label_runs, main,
                           parse_edits, preset_angles, save_analysis, shape_scores, thumb_away, transition_keys,
                           viterbi)
from nlf2vmd.pmx import read_pmx
from nlf2vmd.skeleton import Skeleton
from nlf2vmd.tests.test_pmx import make_pmx
from nlf2vmd.vmd import BoneTrack, LINEAR_INTERPOLATION, merge_bones, read_vmd, write_vmd

FPS = 30.0

# ---- 合成の手（MediaPipe の 21 点の並び）: 手首が原点・指は +y・手のひらは +z を向く ----
_MCP = {'index': (0.025, 0.090), 'middle': (0.005, 0.095), 'ring': (-0.015, 0.090),
        'pinky': (-0.033, 0.080)}
_LEN = {'index': (0.040, 0.025, 0.020), 'middle': (0.045, 0.028, 0.020),
        'ring': (0.042, 0.026, 0.020), 'pinky': (0.032, 0.020, 0.018)}
FINGER_BENDS = {'straight': (5.0, 5.0, 5.0), 'relaxed': (20.0, 30.0, 20.0),
                'curled': (70.0, 95.0, 55.0)}


def _chain(start, direction, lengths, bends, axis):
    """start から direction へ、関節ごとに axis まわりに bends [度] 曲げながら伸ばした点。"""
    pts, p, d = [], np.asarray(start, np.float64), np.asarray(direction, np.float64)
    for length, b in zip(lengths, bends):
        d = quat.rotate(quat.from_rotvec(np.deg2rad(b) * np.asarray(axis)), d)
        p = p + length * d
        pts.append(p)
    return pts


def synthetic_hand(states, rotation=None, scale=1.0, offset=(0.0, 0.0, 0.0)):
    """states: 5 本の指の状態（親指は 'straight' か 'curled'、ほかは FINGER_BENDS のキー）→ (21, 3)。"""
    lm = np.zeros((21, 3))
    cmc = np.array([0.020, 0.020, 0.005])
    mcp = np.array([0.038, 0.045, 0.012])
    lm[1], lm[2] = cmc, mcp
    if states[0] == 'straight':
        # 伸ばして人差し指から離した親指
        lm[3:5] = _chain(mcp, (0.8, 0.6, 0.0), (0.030, 0.025), (0.0, 5.0), (0.0, 0.0, 1.0))
    else:
        # 曲げて人差し指・中指の上に添えた親指（手のひら側・小指側へ）
        lm[3:5] = _chain(mcp, (-0.2, 0.9, 0.4), (0.030, 0.025), (40.0, 45.0), (0.0, -1.0, 0.0))
    for f, finger in enumerate(FINGERS[1:], start=1):
        base = np.array([*_MCP[finger], 0.0])
        lm[4 * f + 1] = base
        # +x 軸まわりの正の回転で、+y（指先）が +z（手のひら側）へ曲がる
        lm[4 * f + 2:4 * f + 5] = _chain(base, (0.0, 1.0, 0.0), _LEN[finger],
                                         FINGER_BENDS[states[f]], (1.0, 0.0, 0.0))
    if rotation is not None:
        lm = quat.rotate(np.broadcast_to(rotation, (21, 4)), lm)
    return lm * scale + np.asarray(offset)


SHAPE_STATES = {
    'default': ('curled', 'relaxed', 'relaxed', 'relaxed', 'relaxed'),
    'thumb': ('straight', 'curled', 'curled', 'curled', 'curled'),
    'index': ('curled', 'straight', 'curled', 'curled', 'curled'),
    'middle': ('curled', 'curled', 'straight', 'curled', 'curled'),
    'ring': ('curled', 'curled', 'curled', 'straight', 'curled'),
    'pinky': ('curled', 'curled', 'curled', 'curled', 'straight'),
    'fist': ('curled', 'curled', 'curled', 'curled', 'curled'),
    'scissors': ('curled', 'straight', 'straight', 'curled', 'curled'),
    'open': ('straight', 'straight', 'straight', 'straight', 'straight'),
}


def hands_cfg(*overrides):
    return load_config(overrides=list(overrides)).hands


def classify_one(lm, cfg):
    p_ext, p_curl = finger_probs(lm[None], cfg)
    return SHAPES[int(np.argmax(shape_scores(p_ext, p_curl, cfg)[0]))]


def test_finger_features():
    lm = synthetic_hand(SHAPE_STATES['open'])
    bends = finger_bends(lm)
    assert (bends[1:] < 40).all() and bends[0] < 35
    lm = synthetic_hand(SHAPE_STATES['fist'])
    bends = finger_bends(lm)
    assert (bends[1:] > 180).all() and bends[0] > 60
    assert thumb_away(synthetic_hand(SHAPE_STATES['open'])) > 0.8
    assert thumb_away(synthetic_hand(SHAPE_STATES['fist'])) < 0.5


@pytest.mark.parametrize('shape', SHAPES)
def test_classify_every_shape(shape):
    cfg = hands_cfg()
    rng = np.random.default_rng(SHAPES.index(shape))
    # 向き・大きさ・位置に依らない（画像上の点 [px] でも 3D [m] でも同じ）
    for k in range(5):
        rot = quat.normalize(rng.normal(size=4)) if k else None
        lm = synthetic_hand(SHAPE_STATES[shape], rot, scale=(1.0, 3000.0)[k % 2],
                            offset=rng.normal(size=3) * 100)
        assert classify_one(lm, cfg) == shape, (shape, k)


def test_default_score_controls_ambiguous_hands():
    # 中間（軽く曲げた）の指が 1 本だけのパーはパー、2 本ならデフォルト。デフォルトの点を下げればパー
    lm = synthetic_hand(('straight', 'relaxed', 'straight', 'straight', 'straight'))
    assert classify_one(lm, hands_cfg()) == 'open'
    lm = synthetic_hand(('straight', 'relaxed', 'relaxed', 'straight', 'straight'))
    assert classify_one(lm, hands_cfg()) == 'default'
    assert classify_one(lm, hands_cfg('hands.default_score=0.05')) == 'open'


def test_viterbi_switch_cost():
    T = 60
    logp = np.log(np.full((T, 3), 0.3))
    logp[:, 0] = np.log(0.5)
    logp[20:23, 1] = np.log(0.9)     # 短くはっきりした変化（点の差の和 3 × 0.59 = 1.8）
    logp[40:, 2] = np.log(0.95)      # 長く続く変化
    path = viterbi(logp, switch_cost=3.0)
    assert (path[:40] == 0).all() and (path[42:] == 2).all()
    path = viterbi(logp, switch_cost=0.5)
    assert (path[20:23] == 1).all()
    # 見えないフレーム（NaN）では前の形が続く
    logp = np.log(np.full((30, 2), 0.1))
    logp[:10, 1] = 0.0
    logp[10:20] = np.nan
    assert (viterbi(logp, 3.0) == 1).all()


def analysis_from_states(timeline, fps=FPS, presence=None, seed=0):
    """timeline: [(フレーム数, 形の名前 or None = 見えない)] を両手に使った検出結果の dict。"""
    rng = np.random.default_rng(seed)
    frames = []
    for n, shape in timeline:
        for _ in range(n):
            if shape is None:
                frames.append(np.full((21, 3), np.nan))
            else:
                rot = quat.normalize(np.array([0.1, 0.2, 0.0, 1.0]) + rng.normal(size=4) * 0.05)
                frames.append(synthetic_hand(SHAPE_STATES[shape], rot, scale=1000.0))
    lm = np.stack(frames)
    lm = np.stack([lm, lm], axis=1)
    if presence is None:
        presence = np.where(np.isfinite(lm).all(axis=(-1, -2)), 0.9, 0.0)
    return dict(screen=lm, world=lm / 1000.0, presence=presence,
                handedness=np.full(presence.shape, 0.5), fps=np.float64(fps))


def labels_of(result, side=0):
    return [(SHAPES[k], e - s + 1) for s, e, k in label_runs(result.labels[:, side])]


def test_temporal_flicker_and_gaps():
    cfg = hands_cfg()
    # グーの間に 2 フレームだけパー（誤検出）→ グーのまま
    r = build_hands(analysis_from_states([(30, 'fist'), (2, 'open'), (30, 'fist')]), cfg,
                    log=None)
    assert labels_of(r) == [('fist', 62)]
    # 短く見えない区間（0.3 秒）はグーのまま、長く見えない区間（2 秒）はデフォルト
    r = build_hands(analysis_from_states([(30, 'fist'), (9, None), (30, 'fist'), (60, None),
                                          (30, 'open')]), cfg, log=None)
    runs_ = labels_of(r)
    # 見えない区間の端は、ならしたときに前後から数フレーム埋まる
    assert runs_[0][0] == 'fist' and 69 <= runs_[0][1] <= 76
    assert runs_[1][0] == 'default' and runs_[1][1] >= 45
    assert runs_[-1][0] == 'open'
    # hold なら長く見えない区間も直前の形のまま
    r = build_hands(analysis_from_states([(30, 'fist'), (60, None), (30, 'open')]),
                    hands_cfg('hands.missing_shape=hold'), log=None)
    assert [k for k, _ in labels_of(r)] == ['fist', 'open']


def test_changes_follow_the_video():
    timeline = [(40, 'open'), (40, 'fist'), (40, 'scissors'), (40, 'index'), (40, 'thumb')]
    r = build_hands(analysis_from_states(timeline), hands_cfg(), log=None)
    runs_ = labels_of(r)
    assert [k for k, _ in runs_] == ['open', 'fist', 'scissors', 'index', 'thumb']
    # 切り替わるフレームは元の境目から 3 フレーム以内
    bounds = np.cumsum([n for _, n in runs_])[:-1]
    np.testing.assert_allclose(bounds, [40, 80, 120, 160], atol=3)


def test_resample_and_num_frames():
    # 検出は 60fps、キーは 30fps・フレーム数は体の動きに合わせる
    d = analysis_from_states([(60, 'open'), (60, 'fist')], fps=60.0)
    r = build_hands(d, hands_cfg(), fps=FPS, num_frames=70, log=None)
    assert r.labels.shape == (70, 2)
    runs_ = labels_of(r)
    assert runs_[0][0] == 'open' and abs(runs_[0][1] - 30) <= 2 and runs_[1][0] == 'fist'


def test_apply_edits():
    labels = np.zeros((90, 2), np.int64)
    out = apply_edits(labels, [(1.0, 2.0, 'right', 'scissors'), (0.0, 0.5, 'both', 'fist')], FPS)
    assert (out[30:60, 1] == SHAPES.index('scissors')).all() and (out[30:60, 0] == 0).all()
    assert (out[:15] == SHAPES.index('fist')).all() and (out[60:] == 0).all()


def test_parse_edits():
    assert parse_edits('12.0-13.5 右 チョキ; 20〜21 両手 グー\n1-2 left open') == [
        (12.0, 13.5, 'right', 'scissors'), (20.0, 21.0, 'both', 'fist'), (1.0, 2.0, 'left', 'open')]
    assert parse_edits('') == [] and parse_edits(None) == []
    for bad in ('12 右 チョキ', '1-2 上 グー', '1-2 右 きつね', '3-2 右 グー'):
        with pytest.raises(ValueError):
            parse_edits(bad)


# ---- 指ボーン（Black.pmx と同じ配置の左手と、その鏡像の右手）----
_LEFT_HAND = [
    ('左手首', (4.652, 12.41, -0.237), None),
    ('左親指０', (4.772, 12.225, -0.443), '左手首'),
    ('左親指１', (5.008, 11.945, -0.574), '左親指０'),
    ('左親指２', (5.11, 11.703, -0.637), '左親指１'),
    ('左親指２先', (5.296, 11.49, -0.789), '左親指２'),
    ('左人指１', (5.375, 11.884, -0.436), '左手首'),
    ('左人指２', (5.64, 11.687, -0.43), '左人指１'),
    ('左人指３', (5.777, 11.554, -0.421), '左人指２'),
    ('左人指３先', (5.946, 11.456, -0.418), '左人指３'),
    ('左中指１', (5.41, 11.935, -0.261), '左手首'),
    ('左中指２', (5.714, 11.691, -0.264), '左中指１'),
    ('左中指３', (5.87, 11.538, -0.264), '左中指２'),
    ('左中指３先', (6.053, 11.425, -0.263), '左中指３'),
    ('左薬指１', (5.387, 11.944, -0.1), '左手首'),
    ('左薬指２', (5.664, 11.723, -0.105), '左薬指１'),
    ('左薬指３', (5.795, 11.589, -0.111), '左薬指２'),
    ('左薬指３先', (5.973, 11.477, -0.127), '左薬指３'),
    ('左小指１', (5.332, 11.951, 0.038), '左手首'),
    ('左小指２', (5.544, 11.778, 0.03), '左小指１'),
    ('左小指３', (5.659, 11.661, 0.02), '左小指２'),
    ('左小指３先', (5.802, 11.57, 0.007), '左小指３'),
]


def hand_pmx(tmp_path, drop=()):
    rows = []
    for name, pos, parent in _LEFT_HAND:
        rows.append((name, pos, parent))
        rows.append((name.replace('左', '右', 1), (-pos[0], pos[1], pos[2]),
                     parent and parent.replace('左', '右', 1)))
    parents = {n: par for n, _, par in rows}
    rows = [r for r in rows if r[0][1:] not in drop]
    names = [r[0] for r in rows]

    def parent(n):
        par = parents[n]
        while par is not None and par not in names:     # 抜いたボーンの子は、その親につなぐ
            par = parents[par]
        return names.index(par) if par else -1

    bones = [(n, p, parent(n), None, False) for n, p, _ in rows]
    path = tmp_path / 'hand.pmx'
    path.write_bytes(make_pmx(bones))
    return path


def fk_tips(skel, rig, angles):
    """手首を固定して、角度 (5, 4) の指の先の位置 {指: (3,)}（内部座標）。"""
    local = {n: q[0] for n, q in rig.local_quats(angles[None]).items()}
    s = '左右'[rig.side]
    pos, rot = {}, {}

    def solve(n):
        if n in pos:
            return
        if n == s + '手首':
            pos[n], rot[n] = skel.internal(n), quat.IDENTITY
            return
        p = skel.parents[n]
        solve(p)
        pos[n] = pos[p] + quat.rotate(rot[p], skel.internal(n) - skel.internal(p))
        rot[n] = quat.mul(rot[p], local.get(n, quat.IDENTITY))

    tips = {}
    for finger, tip in zip(FINGERS, ('親指２先', '人指３先', '中指３先', '薬指３先', '小指３先')):
        solve(s + tip)
        tips[finger] = pos[s + tip]
    return tips


def test_rig_bends_fingers_toward_the_palm(tmp_path):
    skel = Skeleton.from_pmx(read_pmx(hand_pmx(tmp_path)))
    cfg = hands_cfg()
    table = preset_angles(cfg)
    rigs = [build_rig(skel, side, cfg) for side in (0, 1)]
    for side, rig in enumerate(rigs):
        # A ポーズの MMD モデルの手のひらは下向き（体の内側寄り）
        n = rig.palm_normal
        assert n[1] < -0.5
        assert n[0] * (1 if side == 0 else -1) < 0
        straight = fk_tips(skel, rig, table[SHAPES.index('open')] * 0)
        fist = fk_tips(skel, rig, table[SHAPES.index('fist')])
        w = skel.internal('左右'[side] + '手首')
        for finger in FINGERS[1:]:
            move = fist[finger] - straight[finger]
            assert np.dot(move, n) > 0.2, (side, finger)             # 手のひら側へ
            # 握った指先は手首に近づく
            assert np.linalg.norm(fist[finger] - w) < np.linalg.norm(straight[finger] - w) * 0.75
        # グーの親指は手のひら側・小指側へ
        to_pinky = skel.internal('左右'[side] + '小指１') - skel.internal('左右'[side] + '人指１')
        move = fist['thumb'] - straight['thumb']
        assert np.dot(move, n) > 0 and np.dot(move, to_pinky) > 0
        # パーでは人差し指が親指側へ、小指が外側へ開く
        opened = fk_tips(skel, rig, table[SHAPES.index('open')])
        assert np.dot(opened['index'] - straight['index'], to_pinky) < 0
        assert np.dot(opened['pinky'] - straight['pinky'], to_pinky) > 0
    # 左右は鏡像（x を反転すると一致する）
    left = fk_tips(skel, rigs[0], table[SHAPES.index('scissors')])
    right = fk_tips(skel, rigs[1], table[SHAPES.index('scissors')])
    for finger in FINGERS:
        np.testing.assert_allclose(left[finger] * [-1, 1, 1], right[finger], atol=1e-6)


def test_rig_skips_missing_bones(tmp_path):
    skel = Skeleton.from_pmx(read_pmx(hand_pmx(tmp_path, drop=('親指０',))))
    msgs = []
    rig = build_rig(skel, 0, hands_cfg(), msgs.append)
    assert '左親指０' not in rig.bone_names and '左親指１' in rig.bone_names
    assert any('親指０' in m for m in msgs)
    skel = Skeleton.from_pmx(read_pmx(hand_pmx(tmp_path, drop=('人指１',))))
    assert build_rig(skel, 0, hands_cfg(), msgs.append) is None


def test_transition_keys():
    labels = np.array([0] * 30 + [6] * 30 + [8] * 3 + [6] * 30)
    keys = transition_keys(labels, FPS, 0.2)                 # 6 フレームで切り替える
    frames = [k[0] for k in keys]
    assert frames == sorted(set(frames))
    assert keys[0] == (0, 0, False)
    assert (27, 0, False) in keys and (33, 6, True) in keys
    # 3 フレームしか続かない形の切り替えは、前後の形の半分までに縮める
    assert all(k[0] <= 62 for k in keys if k[1] == 8) and any(k[1] == 8 for k in keys)


def test_build_hands_tracks_and_vmd(tmp_path):
    pmx = hand_pmx(tmp_path)
    d = analysis_from_states([(40, 'open'), (40, 'fist')])
    path = tmp_path / 'hands_analysis.npz'
    save_analysis(path, d['screen'], d['world'], d['presence'], d['handedness'], FPS)
    # 体の動きの VMD（手首のキーと口のモーフ）
    body = tmp_path / 'motion.vmd'
    frames = np.arange(80)
    from nlf2vmd.vmd import MorphTrack
    write_vmd(body, [BoneTrack('左手首', frames, np.zeros((80, 3)), np.tile(quat.IDENTITY, (80, 1))),
                     BoneTrack('左人指１', frames[:1], np.zeros((1, 3)), quat.IDENTITY[None])],
              'テストモデル', morphs=[MorphTrack('あ', frames[:2], np.array([0.0, 1.0]))])
    r = export_hands(path, tmp_path / 'hands.vmd', motion_vmd=body,
                     merged_path=tmp_path / 'merged.vmd', pmx=pmx, log=None)
    assert r.labels.shape == (80, 2)
    names = {t.name for t in r.tracks}
    assert len(names) == 30 and '右小指３' in names and '左親指０' in names
    hands_vmd = read_vmd(tmp_path / 'hands.vmd')
    f, _, rot = hands_vmd.track('左人指１')
    # パー → グー の 2 つのキーだけ（最初のキーと、切り替えの始まり・終わり）
    assert len(f) == 3 and f[0] == 0 and 35 <= f[1] < 40 <= f[2] <= 45
    assert quat.angle_between(rot[0], rot[1]) < 1e-6
    assert np.rad2deg(quat.angle_between(rot[1], rot[2])) > 60
    k = hands_vmd.keys[hands_vmd.keys['name'] == '左人指１'.encode('cp932')]
    k = k[np.argsort(k['frame'])]
    assert (k['interp'][2] == EASE_INTERPOLATION).all()
    assert (k['interp'][1] == LINEAR_INTERPOLATION).all()
    merged = read_vmd(tmp_path / 'merged.vmd')
    assert '左手首' in merged.bone_names() and merged.morph_names() == ['あ']
    # 同じ名前（左人指１）のキーは置き換える
    assert len(merged.track('左人指１')[0]) == 3
    assert len(merged.keys) == 80 + sum(len(t.frames) for t in r.tracks)


def test_merge_bones_keeps_other_keys(tmp_path):
    src = tmp_path / 'a.vmd'
    frames = np.arange(3)
    write_vmd(src, [BoneTrack('頭', frames, np.zeros((3, 3)), np.tile(quat.IDENTITY, (3, 1)))], 'm')
    n = merge_bones(src, [BoneTrack('左親指１', frames[:1], np.zeros((1, 3)), quat.IDENTITY[None])],
                    tmp_path / 'b.vmd')
    out = read_vmd(tmp_path / 'b.vmd')
    assert n == 4 and out.bone_names() == sorted(['頭', '左親指１']) and out.model_name == 'm'


def test_write_variant_with_bones(tmp_path, run_walk):
    from nlf2vmd.variants import write_variant
    _, result = run_walk()
    extra = [BoneTrack('左人指１', np.array([0]), np.zeros((1, 3)), quat.IDENTITY[None])]
    for kind in ('full', 'upper_body'):
        write_variant(result, kind, tmp_path / f'{kind}.vmd', bones=extra, log=None)
        vmd = read_vmd(tmp_path / f'{kind}.vmd')
        assert '左人指１' in vmd.bone_names() and '左手首' in vmd.bone_names()


def test_cli(tmp_path):
    pmx = hand_pmx(tmp_path)
    d = analysis_from_states([(20, 'thumb'), (20, 'scissors')])
    path = tmp_path / 'hands_analysis.npz'
    save_analysis(path, d['screen'], d['world'], d['presence'], d['handedness'], FPS)
    assert main([str(path), '--pmx', str(pmx), '--set', 'hands.sides=left']) == 0
    vmd = read_vmd(tmp_path / 'hands.vmd')
    assert vmd.bone_names() and all(n.startswith('左') for n in vmd.bone_names())
    # 上書きした形（最初の 0.5 秒をグー）のキーが入る
    assert main([str(path), '--pmx', str(pmx), '-o', str(tmp_path / 'e.vmd'),
                 '--edit', '0-0.5 左 グー']) == 0
    f, _, rot = read_vmd(tmp_path / 'e.vmd').track('左人指１')
    assert f[0] == 0 and np.rad2deg(quat.angle_between(rot[0], quat.IDENTITY)) > 60


# ---- 検出（モデルを使わない部分）----
def test_roi_and_crop_roundtrip():
    rng = np.random.default_rng(0)
    img = (rng.random((200, 300, 3)) * 255).astype(np.uint8)
    wrist, hand = np.array([150.0, 120.0]), np.array([170.0, 80.0])
    roi = hd.roi_from_joints(wrist, hand, 64.0)
    # 手首→手先が ROI の上を向く
    pts = np.array([[112.0, 180.0, 0.0], [112.0, 20.0, 0.0]])   # crop の下と上
    img_pts = hd.to_image(pts, roi)
    d_img = img_pts[1, :2] - img_pts[0, :2]
    assert np.dot(d_img, hand - wrist) > 0.99 * np.linalg.norm(d_img) * np.linalg.norm(hand - wrist)
    # 切り出しの中心の色は、画像の ROI の中心の色
    c = hd.crop(img, roi)
    center = roi[:2]
    assert c.shape == (224, 224, 3) and c.dtype == np.float32
    np.testing.assert_allclose(c[112, 112] * 255, img[int(center[1]), int(center[0])], atol=40)
    # preview_image は to_image の逆
    _, back = hd.preview_image(img, roi, img_pts)
    np.testing.assert_allclose(back, pts[:, :2], atol=1e-6)


def test_roi_from_landmarks_matches_hand():
    lm = synthetic_hand(SHAPE_STATES['open'])[:, [0, 1]] * [-2000, -2000] + [300, 400]
    lm = np.c_[lm, np.zeros(21)]
    roi = hd.roi_from_landmarks(lm)
    # 指先が上（ROI の上半分）に、手首が下半分に来る
    _, pts = hd.preview_image(np.zeros((800, 800, 3), np.uint8), roi, lm)
    assert pts[0, 1] > 112 and pts[12, 1] < 112
    assert (pts >= 0).all() and (pts <= 224).all()


def test_limit_roi():
    base = np.array([100.0, 100.0, 40.0, 0.3])
    out = hd.limit_roi([300.0, 100.0, 500.0, 1.0], base)
    np.testing.assert_allclose(out, [120.0, 100.0, 80.0, 1.0])
    out = hd.limit_roi([105.0, 95.0, 1.0, -0.2], base)
    np.testing.assert_allclose(out, [105.0, 95.0, 20.0, -0.2])


def test_detect_with_stub_model():
    # モデルの代わりに、切り出しの中央に手首・上に指先がある手を返す
    class Stub:
        calls = 0

        def __call__(self, crop):
            Stub.calls += 1
            lm = synthetic_hand(SHAPE_STATES['open'])[:, [0, 1, 2]] * [1500, -1500, 1500]
            lm[:, :2] += [112, 200]
            return lm, 0.9 - 0.1 * Stub.calls, 0.3, np.zeros((21, 3))

    img = np.zeros((300, 300, 3), np.uint8)
    roi = hd.roi_from_joints([150, 200], [150, 150], 100.0)
    r = hd.detect(Stub(), img, roi, passes=2)
    assert Stub.calls == 2 and r['presence'] == pytest.approx(0.8)   # 存在スコアが高い 1 回目
    np.testing.assert_allclose(r['roi'], roi)
    out = hd.detect_video([img, img], np.stack([[roi, roi]] * 2), np.full((2, 2, 2), 150.0),
                          Stub(), passes=1, max_wrist_offset=0.5, keep=[1])
    assert out['screen'].shape == (2, 2, 21, 3) and set(out['frames']) == {1}


def test_joint_rois_projection():
    joints = np.zeros((1, 24, 3))
    joints[0, :, 2] = 3000.0                        # 3 m 先
    joints[0, 20] = [200.0, 0.0, 3000.0]            # 左手首（カメラから見て右）
    joints[0, 22] = [260.0, 60.0, 3000.0]
    rois, wrists = hd.joint_rois(joints, (640, 480))
    K = hd.intrinsics((640, 480))
    np.testing.assert_allclose(wrists[0, 0], [320 + K[0, 0] * 200 / 3000, 240], atol=1e-6)
    np.testing.assert_allclose(rois[0, 0, 2], K[0, 0] * 250 / 3000)


def test_stabilized_joints_camera_coords(body_model):
    """手を切り出す位置: カメラ座標のまま・動画の fps のまま、ジッター制御だけを掛けた関節の位置。"""
    from nlf2vmd.body_model import forward_kinematics
    from nlf2vmd.synthetic import add_joint_noise, synthetic_walk, to_camera_coords
    fps = 25.0   # MMD の 30fps にリサンプルしないこと
    motion = to_camera_coords(synthetic_walk(num_frames=100, fps=fps), height=1.2, pitch_deg=10.0)
    T = len(motion['pose'])

    # ジッター制御を無効にすると、入力をそのまま FK した位置（カメラ座標 [mm]）と一致する
    off = ['jitter.outlier_deg_per_s=1e9', 'jitter.root_median_window=1',
           'jitter.hand_position.enabled=false',
           *[f'jitter.one_euro.groups.{g}.min_cutoff=0'
             for g in ('torso', 'head', 'arm', 'wrist', 'leg')]]
    joints = hd.stabilized_joints(motion, body_model, overrides=off)
    rest = body_model.rest_joints(motion['betas'])
    _, ref = forward_kinematics(quat.from_rotvec(motion['pose']), motion['trans'] + rest[0], rest,
                                body_model.parents)
    assert joints.shape == (T, 24, 3)
    np.testing.assert_allclose(joints, ref * 1000.0, atol=1e-6)

    # 既定のジッター制御では、関節の回転の震えによる手首の細かい揺れが小さくなる
    noisy = dict(motion, pose=add_joint_noise(
        motion, {j: 3.0 for j in (13, 14, 16, 17, 18, 19, 20, 21)}, smooth_frames=0.5)['pose'])
    raw = hd.stabilized_joints(noisy, body_model, overrides=off)
    smooth = hd.stabilized_joints(noisy, body_model)
    wrist = list(hd.WRIST_JOINTS)
    shake = lambda j: np.linalg.norm(np.diff(j[:, wrist], 2, axis=0), axis=-1).mean()  # noqa: E731
    assert shake(smooth) < 0.5 * shake(raw)
