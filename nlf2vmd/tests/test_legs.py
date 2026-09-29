"""ステージ9l: 脚（足・ひざ・足首）の回転のキー。股関節から足ＩＫまでをモデルの脚で解き、膝を SMPL の膝の向きの側に置くこと。"""
from types import SimpleNamespace

import numpy as np
import pytest

from nlf2vmd import convert, load_config, quat
from nlf2vmd.center import ReachGeometry
from nlf2vmd.legs import LEG_BONES, MIN_BEND_DEG, knee_poles, solve_legs
from nlf2vmd.skeleton import SIDES, Skeleton
from nlf2vmd.synthetic import synthetic_walk
from nlf2vmd.variants import variant_tracks
from nlf2vmd.vmd import sample_rotations, to_mmd_position, to_mmd_quat

CFG = load_config().leg_keys


def _convert(body_model, motion, overrides=()):
    cfg = load_config(overrides=['diagnostics.enabled=false', *overrides])
    return convert(motion, None, body_model=body_model, config=cfg, log=None)


def _fk_ankles(skel, tracks, T):
    """VMD のキー（センター・グルーブ・下半身・足・ひざ）から FK した足首の位置と、足ＩＫのターゲット（内部座標）。"""
    by = {t.name: t for t in tracks}

    def position(name):
        t = by.get(name)
        out = np.zeros((T, 3))
        if t is not None:
            out[t.frames] = to_mmd_position(t.positions)   # MMD 座標 → 内部座標（z の反転は自分自身の逆）
        return out

    def rotation(name):
        return quat.to_matrix(to_mmd_quat(sample_rotations(by[name], T)))

    center = position('センター') + position('グルーブ')
    geom = ReachGeometry.from_skeleton(skel)
    hips = geom.hip_positions(center, rotation('下半身'))
    ankles, targets = np.empty((T, 2, 3)), np.empty((T, 2, 3))
    for side, s in enumerate(SIDES):
        G1 = rotation('下半身') @ rotation(s + '足')
        G2 = G1 @ rotation(s + 'ひざ')
        v1 = skel.internal(s + 'ひざ') - skel.internal(s + '足')
        v2 = skel.internal(s + '足首') - skel.internal(s + 'ひざ')
        ankles[:, side] = hips[:, side] + G1 @ v1 + G2 @ v2
        targets[:, side] = geom.ik[side] + position(s + '足ＩＫ')
    return ankles, targets


def test_solve_legs_reaches_the_target_with_the_knee_toward_the_pole():
    """届くターゲットなら足首がちょうど届き、膝は「股関節 → ターゲット」と膝の向きが作る面の、膝の向きの側にある。
    ひざ は X 軸まわりだけ（MMD の ひざ の角度制限と同じ）に、後ろへ曲がる。"""
    skel = Skeleton.standard()
    rng = np.random.default_rng(0)
    T = 200
    geom = ReachGeometry.from_skeleton(skel)
    center = rng.normal(0.0, 0.8, (T, 3)) * [1.0, 0.0, 1.0]
    center[:, 1] = -rng.uniform(0.2, 3.0, T)                      # しゃがむほど膝が曲がる
    lower = quat.from_rotvec(rng.normal(0.0, 0.3, (T, 3)))
    hips = geom.hip_positions(center, quat.to_matrix(lower))
    # 股関節の真下から少しずれた、届く所にターゲットを置く
    reach = rng.uniform(0.5, 0.95, (T, 2, 1)) * np.array([skel.leg_length(0), skel.leg_length(1)])[:, None]
    direction = np.stack([rng.normal(0.0, 0.3, (T, 2)), -np.ones((T, 2)), rng.normal(0.0, 0.3, (T, 2))], -1)
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    targets = hips + reach * direction
    poles = rng.normal(0.0, 1.0, (T, 2, 3))
    poles /= np.linalg.norm(poles, axis=-1, keepdims=True)
    ik = SimpleNamespace(delta=targets - geom.ik[None], rotation=np.tile(quat.IDENTITY, (T, 2, 1)))
    legs = solve_legs(skel, center, ik, {'下半身': lower}, poles, CFG)
    for side, s in enumerate(SIDES):
        v1 = skel.internal(s + 'ひざ') - skel.internal(s + '足')
        v2 = skel.internal(s + '足首') - skel.internal(s + 'ひざ')
        G1, G2 = legs.glob[s + '足'], legs.glob[s + 'ひざ']
        ankle = hips[:, side] + G1 @ v1 + G2 @ v2
        reached = ~legs.unreached[:, side]
        assert reached.mean() > 0.9
        np.testing.assert_allclose(ankle[reached], targets[reached, side], atol=1e-6)
        # 膝は面の中の、膝の向きの側
        u = targets[:, side] - hips[:, side]
        u /= np.linalg.norm(u, axis=-1, keepdims=True)
        p = poles[:, side] - np.sum(poles[:, side] * u, -1, keepdims=True) * u
        p /= np.linalg.norm(p, axis=-1, keepdims=True)
        knee = legs.knee[:, side] - hips[:, side]
        np.testing.assert_allclose(np.sum(knee * np.cross(p, u), -1)[reached], 0.0, atol=0.02)
        assert (np.sum(knee * p, -1)[reached] > 0.0).all()
        # ひざ のローカル回転は X 軸まわりだけ（後ろへ曲げる向き）で、ローカル回転は大域回転と合っている
        rv = quat.to_rotvec(legs.local[s + 'ひざ'])
        np.testing.assert_allclose(rv[:, 1:], 0.0, atol=1e-6)
        assert (rv[:, 0] >= np.deg2rad(MIN_BEND_DEG) - 1e-9).all()
        np.testing.assert_allclose(quat.to_matrix(lower) @ quat.to_matrix(legs.local[s + '足']), G1,
                                   atol=1e-9)


def test_leg_keys_reach_the_foot_ik_in_every_variant(body_model):
    """フル・接地優先・移動なし のどれでも、脚のキーを FK した足首が、その種類の足ＩＫのターゲットに届いている
    （MMD の IK は、届いた状態から解き始めるので、膝の向きを変えない）。上半身のみには脚のキーを打たない。"""
    r = _convert(body_model, synthetic_walk(num_frames=240, speed=0.8, noise_deg=1.0, seed=0))
    T = len(r.contact.flags)
    for kind in ('full', 'locked', 'no_move'):
        tracks = variant_tracks(r, kind, log=None)
        assert {t.name for t in tracks} >= set(LEG_BONES)
        ankles, targets = _fk_ankles(r.skeleton, tracks, T)
        err = np.linalg.norm(ankles - targets, axis=-1) / r.scale
        assert np.percentile(err, 99) < 0.002, kind            # 2mm（キーは float32）
    names = {t.name for t in variant_tracks(r, 'upper_body', log=None)}
    assert not names & set(LEG_BONES)


def test_knee_turns_out_with_the_smpl_knee(body_model):
    """SMPL の股関節を外へ 30 度回した（がに股）推定では、MMD の膝も外へ 30 度ほど回る。"""
    base = synthetic_walk(num_frames=180, speed=0.0, seed=0)
    q = quat.from_rotvec(np.asarray(base['pose'], float))
    for joint, sign in ((1, 1.0), (2, -1.0)):                   # 左の股関節は +Y、右は −Y まわりが外向き
        q[:, joint] = quat.mul(q[:, joint], quat.from_rotvec([0.0, sign * np.deg2rad(30.0), 0.0]))
    out = dict(base, pose=quat.to_rotvec(q))
    r0, r1 = _convert(body_model, base), _convert(body_model, out)
    turned = np.median(r1.legs.knee_out_deg, axis=0) - np.median(r0.legs.knee_out_deg, axis=0)
    assert (np.abs(turned - 30.0) < 6.0).all(), turned


def test_straight_leg_points_the_knee_to_the_thigh_front(body_model):
    """脚がまっすぐでも、膝の向きは太ももの回転（曲げの軸）から決まる（太ももの正面の向き）。"""
    r = _convert(body_model, synthetic_walk(num_frames=60, speed=0.0, seed=0))
    kin = r.kin
    rot = quat.to_matrix(quat.from_rotvec([0.0, np.deg2rad(40.0), 0.0]))
    straight = SimpleNamespace(joints=np.array(kin.joints, copy=True),
                               glob_rot=np.array(kin.glob_rot, copy=True))
    for hip, knee, ankle in ((1, 4, 7), (2, 5, 8)):              # 脚を股関節の真下へまっすぐ伸ばす
        H = straight.joints[:, hip]
        straight.joints[:, knee] = H + [0.0, -0.4, 0.0]
        straight.joints[:, ankle] = H + [0.0, -0.8, 0.0]
        straight.glob_rot[:, hip] = rot
    poles = knee_poles(straight)
    np.testing.assert_allclose(poles, np.broadcast_to(rot @ [0.0, 0.0, 1.0], poles.shape), atol=1e-9)


def test_knee_faces_forward_when_the_legs_only_bend_forward(body_model):
    """脚を前後にだけ曲げる歩行では、膝は正面を向く（がに股にならない）。SMPL の初期姿勢の膝は 股関節 → 足首 の線より
    2cm ほど外にあるので、膝の位置のずれの向きを使うと、曲げただけで 10〜16 度外を向いてしまう。"""
    r = _convert(body_model, synthetic_walk(num_frames=240, speed=0.8, seed=0))
    assert np.abs(r.legs.knee_out_deg).max() < 3.0, np.abs(r.legs.knee_out_deg).max(0)


def test_leg_keys_can_be_turned_off(body_model):
    r = _convert(body_model, synthetic_walk(num_frames=60, speed=0.0, seed=0), ['leg_keys.enabled=false'])
    assert r.legs is None and r.info['legs'] == dict(enabled=False)
    assert not {t.name for t in r.tracks} & set(LEG_BONES)


@pytest.mark.parametrize('ankle', [True, False])
def test_ankle_key_follows_the_foot_ik_rotation(body_model, ankle):
    r = _convert(body_model, synthetic_walk(num_frames=60, speed=0.5, seed=0),
                 [f'leg_keys.ankle={str(ankle).lower()}'])
    names = {t.name for t in r.tracks}
    assert ('左足首' in names) == ankle and {'左足', '左ひざ'} <= names
    if ankle:
        G = r.legs.glob['左足首']
        np.testing.assert_allclose(G, quat.to_matrix(r.foot_ik.rotation[:, 0]), atol=1e-9)
        local = quat.to_matrix(r.legs.local['左足首'])
        np.testing.assert_allclose(r.legs.glob['左ひざ'] @ local, G, atol=1e-9)


def test_knee_direction_is_smoothed_in_the_pelvis_frame(body_model):
    """膝の向きは骨盤の座標系でならす: 推定の震え（雑音の無い推定との差の、フレームごとの揺れ）は半分ほどになり、
    体ごと回る動き（骨盤と一緒に回る膝の向き）はならさない。"""
    clean = _convert(body_model, synthetic_walk(num_frames=240, speed=0.8, noise_deg=0.0, seed=1))
    r = _convert(body_model, synthetic_walk(num_frames=240, speed=0.8, noise_deg=3.0, seed=1))

    def in_pelvis(kin, p):
        return np.einsum('tba,tsb->tsa', kin.glob_rot[:, 0], p)

    def shake(p):
        d = in_pelvis(r.kin, p) - in_pelvis(clean.kin, knee_poles(clean.kin))
        return np.percentile(np.linalg.norm(np.diff(d, 2, axis=0), axis=-1), 99)

    smooth = knee_poles(r.kin, r.fps, CFG.pole_one_euro)
    assert shake(smooth) < 0.7 * shake(knee_poles(r.kin))
    # 骨盤ごと鉛直軸まわりに回す（毎フレーム 20 度ずつ）と、ならした膝の向きも同じだけ回る
    spin = quat.to_matrix(quat.from_rotvec(np.outer(np.arange(len(smooth)), [0.0, np.deg2rad(20.0), 0.0])))
    turned = SimpleNamespace(joints=np.einsum('tab,tjb->tja', spin, r.kin.joints),
                             glob_rot=spin[:, None] @ r.kin.glob_rot)
    np.testing.assert_allclose(knee_poles(turned, r.fps, CFG.pole_one_euro),
                               np.einsum('tab,tsb->tsa', spin, smooth), atol=1e-9)
