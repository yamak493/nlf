"""足ＩＫ: 接地中につま先だけ・かかとだけが床に着いている所（かかと上げ・ピボット・かかとからの着地）では、床に着いている
点を固定して足を回す。足裏全体が着いている所は、従来どおり足首の位置と回転を固定する。"""
import numpy as np
import pytest

from nlf2vmd import load_config, quat
from nlf2vmd.contact import ContactResult
from nlf2vmd.filters import runs
from nlf2vmd.foot_ik import FLAT, HEEL, TOE, build_foot_ik

FPS = 30.0
ANKLE_H = 0.08
FOOT_L = 0.15
# 足ＩＫから見たかかと（足首の真下）・つま先の床の点。足首が高さ ANKLE_H で足が平らなら、どちらも y = 0
POINTS = np.array([[[0.0, -ANKLE_H, 0.0], [0.0, -ANKLE_H, FOOT_L]]] * 2)
FLOOR = np.zeros(2)
REST = np.array([[0.1, ANKLE_H, 0.0], [-0.1, ANKLE_H, 0.0]])


def _rot(axis, deg):
    deg = np.atleast_1d(np.asarray(deg, np.float64))
    return quat.from_rotvec(np.deg2rad(deg)[:, None] * np.asarray(axis, np.float64))


def _motion(T, stance, rotation, pin_point, pin):
    """左足: stance の間は、点 pin_point（0 かかと / 1 つま先）を pin に固定したまま rotation で回る足首の軌道。
    右足はずっと宙（接地しない）。戻り値 (足首 (T, 2, 3), 回転 (T, 2, 4), 接地, 点の高さ (T, 2, 2))。"""
    q = np.tile(quat.IDENTITY, (T, 2, 1))
    q[:, 0] = rotation
    ankle = np.tile(REST, (T, 1, 1)).astype(float)
    ankle[:, 1, 1] += 0.3
    ankle[:, 0] = pin - quat.rotate(q[:, 0], np.broadcast_to(POINTS[0, pin_point], (T, 3)))
    flags = np.zeros((T, 2), bool)
    flags[stance, 0] = True
    heights = np.stack([ankle[:, 0] + quat.rotate(q[:, 0], np.broadcast_to(POINTS[0, p], (T, 3)))
                        for p in range(2)], axis=1)[..., 1]
    h = np.zeros((T, 2, 2))
    h[:, 0] = heights
    h[:, 1] = 0.3
    return ankle, q, flags, h


def _contact(flags, heights):
    T = len(flags)
    return ContactResult(flags, [runs(flags[:, f]) for f in range(2)], heights, np.zeros((T, 2, 2)))


def _noisy(ankle, q, seed=0, pos_m=0.002, deg=1.0):
    rng = np.random.default_rng(seed)
    a = ankle + rng.normal(0.0, pos_m, ankle.shape)
    q = quat.mul(q, quat.from_rotvec(np.deg2rad(deg) * rng.normal(size=q.shape[:-1] + (3,))))
    return a, q


def _build(ankle, q, flags, heights, pivot=True, **kw):
    cfg = load_config().foot_ik
    extra = dict(sole_points=POINTS, sole_floor=FLOOR) if pivot else {}
    return build_foot_ik(ankle, REST, q, _contact(flags, heights), FPS, 1.0, cfg, **extra, **kw)


def _point(ik, p, foot=0):
    return ik.target[:, foot] + quat.rotate(ik.rotation[:, foot],
                                            np.broadcast_to(POINTS[foot, p], (len(ik.target), 3)))


def _heel_raise(T=60):
    """フレーム 10〜39 が接地。30〜39 でつま先を軸にかかとを 35 度まで上げ（蹴り出し）、40 から宙へ。"""
    theta = np.zeros(T)
    theta[30:40] = np.linspace(3.5, 35.0, 10)
    theta[40:] = 35.0
    rot = _rot([1.0, 0.0, 0.0], theta)                           # X 軸まわりの + でつま先が下がる（かかとが上がる）
    ankle, q, flags, h = _motion(T, slice(10, 40), rot, 1, np.array([0.1, 0.0, FOOT_L]))
    ankle[40:, 0, 1] += np.linspace(0.01, 0.2, T - 40)          # 足が床を離れて上がる
    return ankle, q, flags, h, theta


def test_heel_raise_keeps_the_toe_on_the_floor():
    """蹴り出しでかかとを上げる間、つま先は床の同じ所に着いたままで、足はかかとが上がる向きに回る
    （従来は接地中の足首の位置・回転を 1 つの値に固定したので、かかとが上がらず、区間の終わりで急に回った）。"""
    ankle, q, flags, h, theta = _heel_raise()
    raw, rq = _noisy(ankle, q)
    ik = _build(raw, rq, flags, h)
    phase = ik.phase[10:40, 0]
    assert (phase[:15] == FLAT).all() and (phase[-8:] == TOE).all()
    toe = _point(ik, 1)[10:40]
    assert np.abs(toe - toe[0]).max() < 0.004                    # つま先は動かない
    assert abs(toe[0, 1]) < 0.004                                # 床に着いている
    # 足の向きは推定に付いていく（かかとが 35 度まで上がる）
    pitch = np.rad2deg(quat.angle_between(ik.rotation[39, 0], quat.IDENTITY))
    assert abs(pitch - 35.0) < 4.0
    # 足首は上がる（かかとを上げた分）。足裏全体の所は従来どおり一定
    assert ik.target[39, 0, 1] - ik.target[20, 0, 1] > 0.05
    flat = np.flatnonzero(ik.phase[:, 0] == FLAT)
    assert (ik.target[flat, 0] == ik.target[flat[0], 0]).all()
    # 従来の方法（点を固定しない）では、接地中の足は平均の向きのまま回らず、かかとも上がらない
    old = _build(raw, rq, flags, h, pivot=False)
    assert (old.phase[10:40, 0] == FLAT).all()
    assert np.rad2deg(quat.angle_between(old.rotation[39, 0], quat.IDENTITY)) < 15.0
    assert old.target[39, 0, 1] - old.target[20, 0, 1] < 0.01


@pytest.mark.parametrize('seed', range(4))
def test_no_jump_between_the_parts_of_a_contact(seed):
    """足裏全体 → つま先だけ の境目でも、足首の位置・回転は段差なく移る（前の部分の点からつなぐ。回転は境目での差を
    補正として推定に掛けて減らすので、境目の後に追いつこうとして速く回らない）。"""
    ankle, q, flags, h, _ = _heel_raise()
    raw, rq = _noisy(ankle, q, seed=seed)
    ik = _build(raw, rq, flags, h)
    # 1 フレームの動きは、本当の動き（かかとを 1 フレームに 3.5 度上げる）の 1.5 倍と推定の揺れの分まで
    true_step = np.linalg.norm(np.diff(ankle[10:40, 0], axis=0), axis=-1).max()
    step = np.linalg.norm(np.diff(ik.target[10:40, 0], axis=0), axis=-1)
    assert step.max() < 1.5 * true_step
    turn = np.rad2deg(quat.angle_between(ik.rotation[11:40, 0], ik.rotation[10:39, 0]))
    assert turn.max() < 1.5 * 3.5


def test_pivot_on_the_toe():
    """かかとを少しだけ浮かせて（しきい値より低い）、つま先を軸に 90 度回る。つま先は動かず、足は 90 度回る。"""
    T = 70
    yaw = np.zeros(T)
    yaw[20:35] = np.linspace(6.0, 90.0, 15)
    yaw[35:] = 90.0
    rot = quat.mul(_rot([0.0, 1.0, 0.0], yaw), _rot([1.0, 0.0, 0.0], 2.0))    # かかとが 0.5cm 浮く
    ankle, q, flags, h = _motion(T, slice(10, 50), rot, 1, np.array([0.1, 0.0, FOOT_L]))
    raw, rq = _noisy(ankle, q, seed=1)
    ik = _build(raw, rq, flags, h)
    assert (ik.phase[22:33, 0] == TOE).all()
    toe = _point(ik, 1)[10:50]
    assert np.abs(toe[:, [0, 2]] - toe[0, [0, 2]]).max() < 0.005
    turned = np.rad2deg(quat.angle_between(ik.rotation[49, 0], ik.rotation[10, 0]))
    assert abs(turned - 90.0) < 5.0
    # 従来の方法では、接地中の向きは平均で 1 つに固定され、区間の終わりまで回らない
    old = _build(raw, rq, flags, h, pivot=False)
    assert np.rad2deg(quat.angle_between(old.rotation[49, 0], old.rotation[10, 0])) < 1e-6


def test_heel_strike_keeps_the_heel_on_the_floor():
    """かかとから着地して、つま先を下ろす（区間の最初はかかとだけ）。かかとは床の上を動かない。

    モデルの足の形では、かかとの点は足首の真下にあるので、つま先を上げると足首が初期の高さより少し下がる。足ＩＫの
    最後のクランプ（差分 Y ≥ 0）で足首は初期の高さに留まり、かかとは 0.08 × (1 − cos 30°) ≈ 1cm だけ浮く（実際の
    かかとは足首より後ろにあり、つま先を上げると足首は上がるので、沈めるより近い）。"""
    T = 50
    theta = np.zeros(T)
    theta[:10] = 30.0
    theta[10:18] = np.linspace(30.0, 0.0, 8)
    rot = _rot([1.0, 0.0, 0.0], -theta)                          # つま先が上がっている（30 度 → 床へ）
    ankle, q, flags, h = _motion(T, slice(10, 40), rot, 0, np.array([0.1, 0.0, 0.0]))
    raw, rq = _noisy(ankle, q, seed=2)
    ik = _build(raw, rq, flags, h)
    assert ik.phase[10, 0] == HEEL
    heel = _point(ik, 0)[10:40]
    assert np.abs(heel[:, [0, 2]] - heel[0, [0, 2]]).max() < 0.004        # 水平には動かない
    assert heel[:, 1].min() > -1e-9                                        # 沈まない
    assert heel[:, 1].max() < ANKLE_H * (1.0 - np.cos(np.deg2rad(30.0))) + 0.002
    # つま先は推定どおり床へ下りる（足裏全体になった所で床に着く）
    assert abs(_point(ik, 1)[20, 1]) < 0.004


@pytest.mark.parametrize('seed', range(4))
def test_flat_contact_is_unchanged(seed):
    """足裏全体が着いたままの接地（推定の揺れがあっても）は、従来と同じ値（足首の位置と回転を固定）。"""
    ankle, q, flags, h = _motion(60, slice(10, 45), np.tile(quat.IDENTITY, (60, 1)), 1,
                                 np.array([0.1, 0.0, FOOT_L]))
    raw, rq = _noisy(ankle, q, seed=seed, deg=2.0)
    h = h + np.random.default_rng(seed).normal(0.0, 0.004, h.shape)
    new = _build(raw, rq, flags, h)
    old = _build(raw, rq, flags, h, pivot=False)
    assert (new.phase[flags[:, 0], 0] == FLAT).all()
    np.testing.assert_array_equal(new.target, old.target)
    np.testing.assert_array_equal(new.rotation, old.rotation)


def test_disabled_by_config():
    ankle, q, flags, h, _ = _heel_raise()
    cfg = load_config(overrides=['foot_ik.pivot.enabled=false']).foot_ik
    ik = build_foot_ik(ankle, REST, q, _contact(flags, h), FPS, 1.0, cfg, sole_points=POINTS,
                       sole_floor=FLOOR)
    old = _build(ankle, q, flags, h, pivot=False)
    np.testing.assert_array_equal(ik.target, old.target)


def test_extended_contact_keeps_the_values_of_full_with_pivots():
    """フル [接地優先] のように接地区間を延ばしても（anchor）、フルの接地区間の値（つま先を固定した所も）は変わらない。"""
    ankle, q, flags, h, _ = _heel_raise()
    full = _build(ankle, q, flags, h)
    ext_flags = flags.copy()
    ext_flags[5:10, 0] = True
    ext = _build(ankle, q, ext_flags, h, swing=full.swing, anchor=flags)
    np.testing.assert_array_equal(ext.target[10:40], full.target[10:40])
    np.testing.assert_array_equal(ext.rotation[10:40], full.rotation[10:40])
    np.testing.assert_array_equal(ext.target[5:10, 0], np.broadcast_to(full.target[10, 0], (5, 3)))
