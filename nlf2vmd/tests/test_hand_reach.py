"""胴に対する手の位置を保つリターゲット（ステージ9h）。

胴の近くにある手首は、SMPL の胴に対する位置を胴の寸法の比で MMD の胴へ移した所へ 2 ボーン IK で置き、手首の向きは
変えないこと。胴から遠い手・まっすぐ下ろした腕は回転のコピーのままであること。
"""
import time

import numpy as np

from nlf2vmd import convert, load_config, quat
from nlf2vmd.arm_collision import bone_positions
from nlf2vmd.contacts import globals_from_local
from nlf2vmd.hand_reach import TorsoBox, keep_hand_positions, mmd_torso, smpl_torso
from nlf2vmd.pmx import read_pmx
from nlf2vmd.skeleton import Skeleton
from nlf2vmd.synthetic import SMPL_REST_JOINTS, add_arm_reach, synthetic_walk
from nlf2vmd.tests.test_contacts import hand_pmx

# 9h だけを見る（腕どうし・体との当たり判定と手首の可動域は切る）
ONLY_9H = ['arm_collision.mode=none', 'contacts.enabled=false', 'wrist_limits.enabled=false']


def _convert(body_model, motion, pmx=None, extra=()):
    cfg = load_config(overrides=['diagnostics.enabled=false', *ONLY_9H, *extra])
    return convert(motion, None, pmx=pmx, body_model=body_model, config=cfg, log=None)


def _mmd_wrists(r):
    """(T, 2, 3) MMD の手首の位置（胸のボーンの座標系・MMD の初期姿勢の向き）と、(T, 2, 3, 3) 手首の大域回転。"""
    chest = r.hand_reach.info['chest_bone'] if r.hand_reach.enabled else '上半身2'
    glob = globals_from_local(r.retargeter, r.kin.glob_rot, r.local_quats)
    names = ['左手首', '右手首']
    pos = bone_positions(r.skeleton, glob, [chest] + names, len(r.kin.glob_rot))
    u = np.stack([np.einsum('tba,tb->ta', glob[chest], pos[n] - pos[chest]) for n in names], axis=1)
    return u, np.stack([glob[n] for n in names], axis=1)


def _smpl_wrists(r):
    """(T, 2, 3) SMPL の手首の位置（FK の関節から。胸の座標系・MMD の初期姿勢の向き・MMD 単位）。"""
    G = r.retargeter.global_matrix(r.hand_reach.info['chest_bone'], r.kin.glob_rot)
    J = r.kin.joints
    return np.stack([np.einsum('tba,tb->ta', G, J[:, w] - J[:, 9]) for w in (20, 21)], axis=1)


def _cm(r, d):
    return np.asarray(d) / r.scale * 100.0


def _both_hands(T=10, left=(0.02, 0.12, 0.16), right=(-0.02, 0.12, 0.16), elbow=(0.9, -0.3, 0.3)):
    m = add_arm_reach(synthetic_walk(num_frames=T, speed=0.0), 0, elbow, left)
    return add_arm_reach(m, 1, (-elbow[0], elbow[1], elbow[2]), right)


# ---- 胴の寸法と移し方 ----
def test_torso_box_maps_landmarks_to_landmarks():
    src = TorsoBox(0.0, -3.0, 5.0, 4.0, 0.5, 2.0, 'config')
    dst = TorsoBox(0.1, -2.0, 3.0, 2.5, 0.2, 1.5, 'config')
    pts = np.array([[2.0, 2.0, 1.5],      # 左肩の高さ・胸の前の面
                    [-2.0, -3.0, -0.5],   # 右の股関節の高さ・背中の面
                    [0.0, -0.5, 0.5]])    # 胴の中心
    np.testing.assert_allclose(src.map_to(dst, pts), [[1.35, 1.0, 0.95], [-1.15, -2.0, -0.55],
                                                     [0.1, -0.5, 0.2]])
    np.testing.assert_allclose(src.distance(pts), 0.0)
    np.testing.assert_allclose(src.distance([[0.0, 2.0 + 0.3, 1.5 + 0.4]]), [0.5])   # 肩の上・胸の前


def test_smpl_chest_depth_from_the_spine_vertices():
    J = SMPL_REST_JOINTS
    rng = np.random.default_rng(0)
    verts = np.concatenate([rng.uniform(-1, 1, (200, 3)) * [0.15, 0.2, 0.1] + [0.0, 0.0, 0.03],
                            rng.uniform(-1, 1, (50, 3)) * 0.05 + J[20]])      # 手の頂点は使わない
    weights = np.zeros((len(verts), 24))
    weights[:200, 6] = 1.0
    weights[200:, 20] = 1.0
    box = smpl_torso(J, verts, weights, np.eye(3), 10.0)
    assert box.depth_source == 'mesh'
    assert abs(box.depth - 10.0 * 0.2 * 0.9) < 0.1                            # 5〜95 パーセンタイル
    assert abs(box.center_z - 10.0 * (0.03 - J[9, 2])) < 0.05
    np.testing.assert_allclose(box.width, 10.0 * (J[16, 0] - J[17, 0]))
    np.testing.assert_allclose(box.height, 10.0 * (J[[16, 17], 1].mean() - J[[1, 2], 1].mean()))
    few = smpl_torso(J, verts[:10], weights[:10], np.eye(3), 10.0)            # 頂点が少なければ標準の比
    assert few.depth_source == 'config'
    np.testing.assert_allclose(few.depth, 2 * 0.24 * few.width)


def test_mmd_chest_depth_from_the_rigid_bodies(tmp_path):
    def rigids(idx):
        return [('胸', idx('上半身2'), 1, (1.3, 1.6, 0.8), (0.0, 14.0, 0.3), (0.0, 0.0, 0.0), 0),
                ('乳', idx('上半身2'), 0, (0.6, 0.0, 0.0), (0.0, 14.0, -1.2), (0.0, 0.0, 0.0), 1),
                ('頭', idx('頭'), 0, (1.0, 0.0, 0.0), (0.0, 17.5, 0.2), (0.0, 0.0, 0.0), 0)]

    skel = Skeleton.from_pmx(read_pmx(hand_pmx(tmp_path, rigids)))
    cfg = load_config().contacts
    box = mmd_torso(skel, '上半身2', cfg, 10.0)
    assert box.depth_source == 'rigid'
    margin = cfg.cloth_margin_m * 10.0
    np.testing.assert_allclose(box.depth, 2 * (0.8 + margin))                # 物理の剛体（胸）と頭は使わない
    np.testing.assert_allclose(box.center_z, 0.0, atol=1e-9)               # 胸の剛体の中心 = 上半身2 の奥行き
    np.testing.assert_allclose(box.width, 2 * 1.4)
    none = mmd_torso(skel, '上半身2', load_config(overrides=['contacts.body_source=none']).contacts, 10.0)
    assert none.depth_source == 'config'


# ---- 手首を移した位置に置く ----
def test_wrist_near_the_chest_follows_the_torso_proportions(body_model):
    motion = _both_hands()
    r = _convert(body_model, motion)
    h = r.hand_reach
    assert h.enabled and (h.weight == 1.0).all()
    expected = h.smpl.map_to(h.mmd, _smpl_wrists(r))
    after, _ = _mmd_wrists(r)
    np.testing.assert_allclose(_cm(r, after), _cm(r, expected), atol=1e-6)
    copy, _ = _mmd_wrists(_convert(body_model, motion, extra=['hand_reach.enabled=false']))
    assert _cm(r, np.linalg.norm(copy - expected, axis=-1)).min() > 1.0     # 回転のコピーではずれていた
    assert _cm(r, h.residual).max() < 1e-6


def test_hands_that_meet_in_smpl_meet_in_mmd(body_model):
    """SMPL で胸の前で合わせた両手は、肩幅・腕の長さの比が違う MMD でも合う（回転のコピーでは交差する）。"""
    J = SMPL_REST_JOINTS
    elbow_dir = np.array([0.05, -0.9, 0.1])
    elbow = J[16] + np.linalg.norm(J[18] - J[16]) * elbow_dir / np.linalg.norm(elbow_dir)
    fore = np.linalg.norm(J[20] - J[18])
    # 手首が体の中心の面（x = 0）に来るように、前腕の向きの先を置く
    target = np.array([0.0, elbow[1], elbow[2] + np.sqrt(fore ** 2 - elbow[0] ** 2)])
    motion = _both_hands(left=target, right=target * [-1, 1, 1], elbow=tuple(elbow_dir))
    r = _convert(body_model, motion)
    smpl = _smpl_wrists(r)
    assert _cm(r, np.linalg.norm(smpl[:, 0] - smpl[:, 1], axis=-1)).max() < 0.5   # 初期姿勢の左右差の分だけ
    after, _ = _mmd_wrists(r)
    copy, _ = _mmd_wrists(_convert(body_model, motion, extra=['hand_reach.enabled=false']))
    assert (copy[:, 0, 0] < copy[:, 1, 0]).all()                           # 左手首が右へ行きすぎて交差
    assert _cm(r, np.linalg.norm(copy[:, 0] - copy[:, 1], axis=-1)).min() > 3.0
    assert _cm(r, np.linalg.norm(after[:, 0] - after[:, 1], axis=-1)).max() < 0.5


def test_wrist_orientation_is_kept(body_model):
    motion = _both_hands()
    r = _convert(body_model, motion)
    base = _convert(body_model, motion, extra=['hand_reach.enabled=false'])
    _, rot = _mmd_wrists(r)
    _, rot0 = _mmd_wrists(base)
    np.testing.assert_allclose(rot, rot0, atol=1e-9)
    assert np.rad2deg(quat.angle_between(r.local_quats['左腕'], base.local_quats['左腕'])).max() > 5.0


def test_hanging_and_far_arms_are_unchanged(body_model):
    walk = synthetic_walk(num_frames=60)
    r = _convert(body_model, walk)
    base = _convert(body_model, walk, extra=['hand_reach.enabled=false'])
    assert r.hand_reach.weight.max() == 0.0
    for name in base.local_quats:
        np.testing.assert_array_equal(r.local_quats[name], base.local_quats[name])
    # ひじを曲げていても、胴から遠い手（横に広げて前へ曲げた腕）はそのまま
    spread = _both_hands(left=(0.95, 0.25, 0.3), right=(-0.95, 0.25, 0.3), elbow=(1.0, 0.0, 0.0))
    r = _convert(body_model, spread)
    assert r.hand_reach.weight.max() == 0.0


def test_moving_towards_the_chest_blends_smoothly(body_model):
    """ひじを前に出した腕で、前へ伸ばした手（胴から遠い）を胸の前へ引き寄せる: 重みが 0 → 1 に変わっても跳ねない。"""
    T = 40
    t = np.linspace(0.0, 1.0, T)[:, None]
    hand = (1.0 - t) * [0.3, 0.25, 1.0] + t * [0.02, 0.12, 0.16]
    motion = _both_hands(T, left=hand, right=hand * [-1, 1, 1], elbow=(0.2, -0.1, 0.95))
    r = _convert(body_model, motion)
    base = _convert(body_model, motion, extra=['hand_reach.enabled=false'])
    w = r.hand_reach.weight[:, 0]
    assert w[0] == 0.0 and w[-1] == 1.0
    for name in ('左腕', '左ひじ', '左手首'):
        steps = np.rad2deg(quat.angle_between(r.local_quats[name][1:], r.local_quats[name][:-1]))
        base_steps = np.rad2deg(quat.angle_between(base.local_quats[name][1:], base.local_quats[name][:-1]))
        assert steps.max() < base_steps.max() + 3.0, name   # 跳ねない


def test_max_deg_limits_the_correction(body_model):
    r = _convert(body_model, _both_hands(), extra=['hand_reach.max_deg=5'])
    assert r.hand_reach.correction_deg.max() <= 5.0 + 1e-6
    assert _cm(r, r.hand_reach.residual).max() > 0.5                       # 届かない分は残す


def test_with_pmx_and_default_stages(tmp_path, body_model):
    """PMX（指ボーンあり）で、ほかのステージも有効にして通しで変換できる（9h の後に 9a・9b・9c が掛かる）。"""
    cfg = load_config(overrides=['diagnostics.enabled=false'])
    r = convert(_both_hands(), None, pmx=hand_pmx(tmp_path), body_model=body_model, config=cfg, log=None)
    assert r.hand_reach.enabled and r.info['hand_reach']['frames'] == [10, 10]
    assert r.contacts.overlap_frames(0.005 * r.scale)[1] == 0


def test_less_penetration_is_left_for_9b(tmp_path, body_model):
    """回転のコピーで胸に入り込んでいた手（体の比率の違いが原因）は、9h で胴に対する位置を保つと入り込みが減る
    （ステージ9b が離す量が減る）。"""
    from nlf2vmd.tests.test_contacts import _press
    pmx = hand_pmx(tmp_path)
    cases = [dict(), dict(elbow=(0.2, -0.8, 0.55), hand=(-0.05, 0.08, 0.08))]   # 親指の先 / 前腕が胸に入る
    for kw in cases:
        motion = _press(**kw)
        cfg = load_config(overrides=['diagnostics.enabled=false'])
        r = convert(motion, None, pmx=pmx, body_model=body_model, config=cfg, log=None)
        cfg = load_config(overrides=['diagnostics.enabled=false', 'hand_reach.enabled=false'])
        base = convert(motion, None, pmx=pmx, body_model=body_model, config=cfg, log=None)
        before, before_base = (x.contacts.depth_before[:, 0].max() / r.scale * 100 for x in (r, base))
        assert before_base > 1.0 and before < before_base - 1.0, kw
        assert r.contacts.correction_deg.max() < base.contacts.correction_deg.max(), kw


def test_hand_reach_plot(tmp_path, body_model):
    import pytest
    pytest.importorskip('matplotlib')
    cfg = load_config()
    r = convert(_both_hands(), None, body_model=body_model, config=cfg, diag_dir=tmp_path / 'diag', log=None)
    assert (tmp_path / 'diag' / 'hand_reach.png').exists()
    assert 'hand_reach' in r.plot_paths


def test_ten_minutes_takes_well_under_a_second_per_minute(body_model):
    """処理時間: 10 分（30fps・18000 フレーム）でも、全フレームをまとめて解くので数秒もかからない。"""
    r = _convert(body_model, _both_hands(T=30))
    T = 18000
    glob_rot = np.tile(r.kin.glob_rot, (T // 30, 1, 1, 1))
    local = {k: np.tile(v, (T // 30, 1)) for k, v in r.retargeter.local_quats(r.kin.glob_rot).items()}
    start = time.perf_counter()
    out, res = keep_hand_positions(r.skeleton, r.retargeter, glob_rot, local, r.smpl_rest, None, None,
                                   r.config.hand_reach, r.config.contacts, r.scale)
    elapsed = time.perf_counter() - start
    assert res.weight.shape == (T, 2) and (res.weight == 1.0).all()
    assert elapsed < 10.0, elapsed
