"""フル [接地優先]（motion_full_locked.vmd）: ジャンプ・片足上げで足が浮いた所も、足を床に着けること。
浮いていない所はフルと同じ動きで、2 つの VMD を切り貼りできること。"""
import numpy as np
import pytest

from nlf2vmd import convert, load_config, quat
from nlf2vmd.contact import ContactResult
from nlf2vmd.filters import runs
from nlf2vmd.foot_ik import build_foot_ik
from nlf2vmd.legs import LEG_BONES
from nlf2vmd.locked import lock_flags, locked_motion, mmd_sole_heights, sole_clearance
from nlf2vmd.synthetic import add_jump, synthetic_walk
from nlf2vmd.variants import variant_tracks

FPS = 30.0


def _max_angle_deg(a, b):
    return float(np.rad2deg(quat.angle_between(a, b)).max(initial=0.0))


def _convert(body_model, motion, overrides=()):
    cfg = load_config(overrides=['diagnostics.enabled=false', *overrides])
    return convert(motion, None, body_model=body_model, config=cfg, log=None)


@pytest.fixture(scope='module')
def jump(body_model):
    """その場で足踏みし、5 秒から 0.45 秒・高さ 25cm のジャンプ。"""
    motion = add_jump(synthetic_walk(num_frames=300, speed=0.0, seed=0), 5.0, 0.45, 0.25)
    return motion, _convert(body_model, motion)


@pytest.fixture(scope='module')
def jump_one_foot(body_model):
    """jump と同じ動きを locked.both_feet: false（片足だけの浮きは残す）で。"""
    motion = add_jump(synthetic_walk(num_frames=300, speed=0.0, seed=0), 5.0, 0.45, 0.25)
    return motion, _convert(body_model, motion, ['locked.both_feet=false'])


def _tracks(tracks):
    return {t.name: t for t in tracks}


def test_jump_is_grounded_with_a_planted_foot(jump):
    motion, r = jump
    k, air = r.scale, motion['airborne']
    # フルはジャンプの高さを残す（両足とも 15cm より上）
    assert sole_clearance(r)[air].min(1).max() / k > 0.15
    lm = locked_motion(r)
    info = lm.info
    assert info['hover_frames']['before'] >= 10 and info['hover_frames']['after'] == 0
    assert info['max_lowest_sole_cm']['after'] < 0.5            # どのフレームも低いほうの足が床に着いている
    # 少なくとも片足は、ジャンプの間ずっと床に固定されて動かない
    planted = [f for f in range(2) if lm.contact.flags[air, f].all()]
    assert planted
    target = lm.foot_ik.target[air, planted[0]]
    np.testing.assert_allclose(target, np.broadcast_to(target[0], target.shape), atol=0.003 * k)
    # 骨盤は跳び上がらない（フルは 15cm 以上上がる）
    rise_full = r.center.delta[air, 1].max() - r.center.delta[~air, 1].max()
    rise = lm.center.delta[air, 1].max() - lm.center.delta[~air, 1].max()
    assert rise_full / k > 0.1 and rise / k < 0.01
    assert info['exceed_after'] == 0


def test_both_feet_stay_on_the_floor(jump):
    """既定（locked.both_feet）では、足踏みで上げる足も含めて、どのフレームでも両足の足裏が床に着いている。
    回転とセンターの水平移動はフルと同じ。"""
    motion, r = jump
    lm = locked_motion(r)
    info = lm.info
    assert info['max_highest_sole_cm']['before'] > 5.0
    assert info['one_foot_hover_frames']['after'] == 0 and info['max_highest_sole_cm']['after'] < 0.5
    assert info['exceed_after'] == 0
    full = _tracks(r.tracks)
    # 脚の回転は、その種類のセンター・足ＩＫで解き直す（足ＩＫと同じく変わる）
    for name, t in _tracks(variant_tracks(r, 'locked', log=None)).items():
        if name not in ('グルーブ', '左足ＩＫ', '右足ＩＫ') + LEG_BONES:
            np.testing.assert_array_equal(t.positions, full[name].positions)
            np.testing.assert_array_equal(t.rotations, full[name].rotations)


def test_one_foot_mode_changes_only_around_the_jump(jump_one_foot):
    motion, r = jump_one_foot
    info = locked_motion(r).info
    assert info['hover_frames']['after'] == 0 and info['one_foot_hover_frames']['after'] > 0
    (s, e), = runs(motion['airborne'])
    assert all(s - 15 <= a and b <= e + 15 for a, b in info['differ_frames'])


def test_locked_can_be_spliced_with_full(jump_one_foot):
    """回転とセンターの水平移動はフルと同じ。ジャンプから離れたフレームは、グルーブ・足ＩＫもフルと同じ
    （locked.both_feet: false。既定では足踏みで上げる足も床に着けるので、どのフレームでも足ＩＫが変わる）。"""
    motion, r = jump_one_foot
    k = r.scale
    full = _tracks(r.tracks)
    locked = _tracks(variant_tracks(r, 'locked', log=None))
    assert locked.keys() == full.keys()
    changed = {name for name, t in full.items()
               if not (np.array_equal(t.positions, locked[name].positions)
                       and np.array_equal(t.rotations, locked[name].rotations))}
    assert changed <= {'グルーブ', '左足ＩＫ', '右足ＩＫ', *LEG_BONES}
    (s, e), = runs(motion['airborne'])
    far = np.ones(len(motion['airborne']), bool)
    far[s - 30:e + 31] = False
    for name in changed:
        np.testing.assert_allclose(locked[name].positions[far], full[name].positions[far],
                                   atol=1e-3 * k)              # 1mm（モデルの足の形で床に合わせ直した分）
        if name in LEG_BONES:
            # 脚の回転は、その 1mm の分だけ解き直した角度が変わる
            assert _max_angle_deg(locked[name].rotations[far], full[name].rotations[far]) < 1.0
        else:
            np.testing.assert_array_equal(locked[name].rotations[far], full[name].rotations[far])


def test_center_mode_b_keeps_the_horizontal_center_of_full(body_model):
    """モードB（接地している足から骨盤を求める）でも、センターの水平移動はフルと同じで、ジャンプは除く。"""
    motion = add_jump(synthetic_walk(num_frames=300, speed=1.0, seed=0), 5.0, 0.45, 0.25)
    cfg = load_config(overrides=['diagnostics.enabled=false', 'center.mode=B'])
    r = convert(motion, None, body_model=body_model, config=cfg, log=None)
    lm = locked_motion(r)
    np.testing.assert_array_equal(lm.center.delta[:, [0, 2]], r.center.delta[:, [0, 2]])
    air = motion['airborne']
    assert (lm.center.delta[air, 1].max() - lm.center.delta[~air, 1].max()) / r.scale < 0.01
    assert lm.info['max_lowest_sole_cm']['after'] < 0.5 and lm.info['exceed_after'] == 0


def test_raised_leg_that_looks_like_a_hop_keeps_the_support_foot_down(body_model):
    """片足を高く上げた（腿上げ）ときに、推定で体全体が 8cm 持ち上がり、フルではジャンプとして残る。
    接地優先では、支持脚（左足）は床に着いたまま動かず、骨盤も上がらない。上げた右足も床に着ける。"""
    motion = add_jump(synthetic_walk(num_frames=300, speed=0.0, lift=0.25, seed=0), 4.95, 0.25, 0.08)
    r = _convert(body_model, motion)
    k, air = r.scale, motion['airborne']
    assert r.ground.flight[air].sum() >= 3                      # フルはジャンプとして残す
    stance = next((s, e) for s, e in runs(motion['stance'][:, 0]) if s <= np.flatnonzero(air)[0] <= e)
    s, e = stance[0] + 2, stance[1] - 2
    lm = locked_motion(r)
    assert lm.contact.flags[s:e + 1, 0].all()
    # フルでは接地が 2 つに分かれていた（ロック位置の差 1mm ほど）。その間はなめらかにつなぐだけ
    np.testing.assert_allclose(lm.foot_ik.target[s:e + 1, 0],
                               np.broadcast_to(lm.foot_ik.target[s, 0], (e - s + 1, 3)),
                               atol=0.003 * k)
    assert lm.info['max_highest_sole_cm']['after'] < 0.5         # 上げた右足も床に着いている
    before = lm.center.delta[s - 10:s, 1].mean()
    assert (r.center.delta[air, 1].max() - before) / k > 0.015     # フルは骨盤が上がる
    assert (lm.center.delta[air, 1].max() - before) / k < 0.005


def _tilt_swing_feet(motion, deg):
    """遊脚の間だけ、足首をつま先が下がる向きに最大 deg 度回す（かかとを上げて、つま先を床に擦って足を寄せる）。"""
    q = quat.from_rotvec(np.asarray(motion['pose'], float))
    t = np.arange(len(q)) / float(motion['fps'])
    envelope = np.sin(np.pi * ((t / 0.6) % 1.0)) ** 2
    for side in range(2):
        angle = np.deg2rad(deg) * envelope * ~motion['stance'][:, side]
        q[:, 7 + side] = quat.mul(q[:, 7 + side], quat.from_rotvec(angle[:, None] * [1.0, 0.0, 0.0]))
    return dict(motion, pose=quat.to_rotvec(q))


def test_tilted_foot_touches_the_floor_with_the_model_foot(body_model):
    """かかとを上げてつま先で足を寄せると、SMPL の足の形ではつま先が床に着いていても、モデルの足（足首の高さ・足の長さの
    比が違う）では浮く。接地優先は、モデルの足の形で、足裏の最も低い点をちょうど床に着ける。"""
    from nlf2vmd.diagnostics import output_sole_heights
    motion = _tilt_swing_feet(synthetic_walk(num_frames=240, speed=0.3, lift=0.0), 35.0)
    r = _convert(body_model, motion)
    k = r.scale
    smpl = np.minimum(output_sole_heights(r), r.foot_ik.delta[..., 1])
    assert np.abs(sole_clearance(r) - smpl).max() / k > 0.03       # 足の形の違いで 3cm 以上ずれる
    lm = locked_motion(r)
    np.testing.assert_allclose(mmd_sole_heights(r.skeleton, lm.foot_ik), 0.0, atol=1e-9)
    assert lm.info['exceed_after'] == 0


def test_motion_without_floating_is_the_same_as_full(body_model):
    r = _convert(body_model, synthetic_walk(num_frames=240, speed=0.0, noise_deg=1.0),
                 ['locked.both_feet=false'])
    lm = locked_motion(r)
    assert lm.info['differ_frames'] == [] and lm.info['added_lock_frames'] == [0, 0]
    full = _tracks(r.tracks)
    for name, t in _tracks(variant_tracks(r, 'locked', log=None)).items():
        # モデルの足の形で、両足とも数 mm 浮いたフレームだけ床に合わせ直す（切り貼りで段差が見えない 5mm 以下）
        np.testing.assert_allclose(t.positions, full[name].positions, atol=5e-3 * r.scale)
        if name in LEG_BONES:
            assert _max_angle_deg(t.rotations, full[name].rotations) < 2.0   # その数 mm の分だけ
        else:
            np.testing.assert_array_equal(t.rotations, full[name].rotations)


def _contact(flags, speeds=None):
    flags = np.asarray(flags, bool)
    T = len(flags)
    speeds = np.zeros((T, 2, 2)) if speeds is None else speeds
    return ContactResult(flags, [runs(flags[:, f]) for f in range(2)], np.zeros((T, 2, 2)), speeds)


def test_extended_contact_keeps_the_locks_of_full():
    """フルの接地区間を延ばしても、その区間のロック位置・回転は変わらず、延ばしたフレームはその値を保つ。
    前後 2 つの接地区間をつないだ所は、2 つのロック位置の間をなめらかに移る。"""
    cfg = load_config().foot_ik
    T = 60
    raw = np.zeros((T, 2, 3))
    raw[:, :, 1] = 0.1
    raw[25:, 0, 2] = 0.02                                   # 2cm 先に着地し直す
    raw[21:25, 0, 1] += 0.1                                 # 間は宙に浮いている
    rot = quat.from_rotvec(np.deg2rad(5.0) * np.array([0.0, 1.0, 0.0])
                           * np.linspace(0, 1, T)[:, None, None] * np.ones((T, 2, 1)))
    full_flags = np.zeros((T, 2), bool)
    full_flags[:21] = full_flags[31:] = True
    ext_flags = full_flags.copy()
    ext_flags[:, 0] = True
    full = build_foot_ik(raw, np.zeros((2, 3)), rot, _contact(full_flags), FPS, 1.0, cfg)
    ext = build_foot_ik(raw, np.zeros((2, 3)), rot, _contact(ext_flags), FPS, 1.0, cfg,
                        swing=full.swing, anchor=full_flags)
    for sl in (slice(0, 21), slice(31, T)):
        np.testing.assert_array_equal(ext.target[sl], full.target[sl])
        np.testing.assert_array_equal(ext.rotation[sl], full.rotation[sl])
    z = ext.target[20:32, 0, 2]
    assert (np.diff(z) >= 0).all() and z[0] == full.target[0, 0, 2] and z[-1] == full.target[31, 0, 2]
    np.testing.assert_allclose(ext.target[21:31, 0, 1], 0.1)   # 浮かずに床の高さのまま
    # 片側だけ延ばしたときは、延ばしたフレームも前の接地区間の値のまま
    one_side = full_flags.copy()
    one_side[21:26, 0] = True
    ext = build_foot_ik(raw, np.zeros((2, 3)), rot, _contact(one_side), FPS, 1.0, cfg,
                        swing=full.swing, anchor=full_flags)
    np.testing.assert_array_equal(ext.target[:26, 0], np.broadcast_to(full.target[0, 0], (26, 3)))


def test_lock_flags_only_hold_feet_that_stay_near_the_lock():
    """足を下ろしたフレームでも、足首がロック位置から still より離れたら固定しない（動いている足を引き留めない）。
    どの接地区間ともつながらず、動かない足は固定する。"""
    cfg = load_config().contact
    T = 40
    full = np.zeros((T, 2), bool)
    full[:10, 0] = True
    ankles = np.zeros((T, 2, 3))
    ankles[10:, 0, 0] = 0.01 * np.arange(1, T - 9)           # 1cm/フレームで離れていく
    ankles[:, 1, 0] = 0.5
    speeds = np.ones((T, 2, 2))
    speeds[18:32, 1] = 0.0                                  # 右足はこの間だけ止まっている
    sole = np.zeros((T, 2))
    lowered = np.ones(T, bool)
    flags = lock_flags(_contact(full, speeds), ankles, sole, lowered, None, cfg, 0.03, 1.0)
    assert runs(flags[:, 0]) == [(0, 11)]                   # 2cm までは延ばし、3cm 離れた所で止める
    assert runs(flags[:, 1]) == [(18, 31)]
    # 足を下ろしていないフレームでは何も足さない
    flags = lock_flags(_contact(full, speeds), ankles, sole, np.zeros(T, bool), None, cfg, 0.03,
                       1.0)
    np.testing.assert_array_equal(flags, full)
