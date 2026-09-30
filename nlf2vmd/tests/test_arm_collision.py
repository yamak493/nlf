"""腕どうしの貫通の防止（ステージ9a）: 自重する腕だけを肩まわりに回して、左右の腕のカプセルを離すこと。

合成の歩行に、胸の前で両腕を交差させた腕の動き（add_arm_cross）を重ねて変換する。回転をコピーしただけでは
左右の前腕・手が同じ奥行きで交わって重なる。
"""
import numpy as np
import pytest

from nlf2vmd import arm_collision as ac
from nlf2vmd import convert, load_config, quat
from nlf2vmd.arm_collision import arm_model, closest_points, resolve_arm_collisions
from nlf2vmd.pmx import read_pmx
from nlf2vmd.skeleton import STANDARD_BONES, Skeleton
from nlf2vmd.synthetic import add_arm_cross, synthetic_walk
from nlf2vmd.tests.test_pmx import make_pmx, standard_pmx_bones

ARM_BONES = [s + b for s in '左右' for b in ('肩', '腕', 'ひじ', '手首')]


def _convert(body_model, motion, mode, pmx=None, extra=()):
    # ステージ9a だけを確かめる（9b の体との接触は test_contacts.py で確かめる）
    cfg = load_config(overrides=['diagnostics.enabled=false', f'arm_collision.mode={mode}',
                                 'contacts.enabled=false', *extra])
    return convert(motion, None, pmx=pmx, body_model=body_model, config=cfg, log=None)


def _max_step_deg(q):
    return float(np.rad2deg(quat.angle_between(q[1:], q[:-1])).max())


def test_closest_points_match_dense_sampling():
    rng = np.random.default_rng(0)
    p0, p1, q0, q1 = rng.normal(size=(4, 200, 3))
    q1[:20] = q0[:20] + 0.5 * (p1[:20] - p0[:20])       # 平行な組も含める
    cp, cq = closest_points(p0, p1, q0, q1)
    got = np.linalg.norm(cp - cq, axis=-1)
    s = np.linspace(0.0, 1.0, 401)
    a = p0[:, None] + s[None, :, None] * (p1 - p0)[:, None]
    b = q0[:, None] + s[None, :, None] * (q1 - q0)[:, None]
    dense = np.linalg.norm(a[:, :, None] - b[:, None], axis=-1).min((1, 2))
    assert np.all(got <= dense + 1e-9)                   # 最近点どうしの距離は、どの点の組よりも近い
    assert np.all(dense - got < 0.02)


def test_crossed_arms_are_separated_by_the_yielding_arm_only(body_model):
    motion = add_arm_cross(synthetic_walk(num_frames=60))
    base = _convert(body_model, motion, 'none', extra=['wrist_limits.enabled=false'])
    tol = 0.005 * base.scale
    before, after = base.arm_collision.overlap_frames(tol)
    assert before == 60 and after == 60                  # 何もしないと全フレームで重なる
    for mode, side in (('left', '左'), ('right', '右')):
        for elbow in (True, False):
            # 9c（手首の可動域）は切る: ひじを曲げると前腕のひねりの分け方が少し変わり、9c が手首を収め直すことがある
            r = _convert(body_model, motion, mode, extra=[f'arm_collision.elbow={str(elbow).lower()}',
                                                          'wrist_limits.enabled=false'])
            a = r.arm_collision
            assert a.elbow == elbow
            assert a.overlap_frames(tol) == (60, 0), mode
            assert a.depth_after.max() < 0.0
            changed = {name for name in base.local_quats
                       if quat.angle_between(r.local_quats[name], base.local_quats[name]).max() > 1e-6}
            if elbow:
                # キーが変わるのは自重する側の腕・ひじボーンだけ（手首の向き・反対の腕はそのまま）
                assert side + '腕' in changed and changed <= {side + '腕', side + 'ひじ'}, (mode, changed)
                assert 5.0 < a.correction_deg.max() < 45.0
            else:
                # 肩だけ: キーが変わるのは自重する側の腕ボーンだけ（ひじの曲げ・手首・反対の腕はそのまま）
                assert changed == {side + '腕'}, (mode, changed)
                assert 5.0 < a.correction_deg.max() < 30.0       # 肩まわりに 20 度前後回せば離れる
                assert a.elbow_deg.max() == 0.0


def _crossing(T, depth_from, depth_to):
    """(自重する左腕 Y, 相手の右腕 O, 離す距離 R)。左右の前腕は胸の前で水平に交わり、推定の左の前腕
    （ひじ・手首・手の先）は奥行き depth_from → depth_to [m] を動く（前から後ろへ右の前腕を通り抜ける）。"""
    dz = np.linspace(depth_from, depth_to, T)
    Y = np.tile([[0.2, 0.4, 0.0], [0.2, 0.1, 0.2], [-0.1, 0.1, 0.2], [-0.2, 0.1, 0.2]], (T, 1, 1))
    Y[:, 1:, 2] += dz[:, None]
    O = np.tile([[-0.2, 0.4, 0.0], [-0.2, 0.1, 0.2], [0.1, 0.1, 0.2], [0.2, 0.1, 0.2]], (T, 1, 1))
    radius = np.array([0.03, 0.03, 0.02])
    return Y, O, radius[ac.PAIR_Y] + radius[ac.PAIR_O] + 0.005


def _params(fps=30.0, depth_cost=1.0, max_deg=45.0, axis=(0.0, 0.0, 1.0)):
    return ac._Params(np.deg2rad(max_deg), ac.MIN_LEVER_M, ac.TOLERANCE_M, ac.DAMPING_M ** 2,
                      np.deg2rad(120.0) / fps, np.asarray(axis, float), depth_cost ** 2)


def _apply(Y, Q):
    return Y[:, :1] + np.einsum('tab,tkb->tka', Q, Y - Y[:, :1])


def test_yielding_arm_stays_on_its_side_when_the_estimate_passes_through():
    """推定の左の前腕が右の前腕の前から後ろへ通り抜けても、自重する左腕は来た側（前）に留まる。"""
    T = 60
    Y, O, R = _crossing(T, 0.15, -0.1)
    Q, _, ignored, _ = ac._resolve_sequential(Y, O, R, np.tile(np.eye(3), (T, 1, 1)), _params())
    assert not ignored.any()
    fixed = _apply(Y, Q)
    cp, cq = ac._pairs(Y, O)
    assert ((R - np.linalg.norm(cp - cq, axis=-1)).max(1) > 0.01).sum() > 10   # 推定では重なる
    cp, cq = ac._pairs(fixed, O)
    assert ((R - np.linalg.norm(cp - cq, axis=-1)).max(1) < 1e-3).all()   # 補正後は重ならない
    fore = 4                                                             # (前腕, 前腕) の組
    assert (cp[:, fore, 2] > cq[:, fore, 2]).all()                       # 左の前腕が前のまま
    angles = np.rad2deg(quat.angle_between(quat.from_matrix(Q[1:]), quat.from_matrix(Q[:-1])))
    assert angles.max() < 3.0                                            # 補正はなめらか


def test_yielding_arm_passes_through_gradually_beyond_the_limit():
    """押しのけるのに max_deg より大きく回す必要があるときは、上限で止めて少しずつ通り抜ける（はね戻らない）。"""
    T = 90
    Y, O, R = _crossing(T, 0.15, -0.6)
    p = _params()
    Q, _, ignored, _ = ac._resolve_sequential(Y, O, R, np.tile(np.eye(3), (T, 1, 1)), p)
    assert ignored.any()
    angle = np.array([ac._angle(q) for q in Q])
    assert angle.max() <= p.max_angle + 1e-6
    assert angle[-1] < np.deg2rad(1.0)                   # 最後は推定の姿勢に戻る
    steps = np.rad2deg(quat.angle_between(quat.from_matrix(Q[1:]), quat.from_matrix(Q[:-1])))
    assert steps.max() < 2.0 * np.rad2deg(p.return_step) + 1.0


# ---- 奥行き優先（depth_cost）----
def _stacked(T, gap):
    """(自重する左腕 Y, 相手の右腕 O, 離す距離 R)。両腕を上げて前腕を胸の高さより上で水平にし、画像上で上下に gap [m]
    離して重ねる（前腕は肩と同じ奥行き = 画像面内。相手の前腕がわずかに手前）。最近点を結ぶ向きは上下（画像面内）。"""
    Y = np.tile([[0.2, 0.0, 0.0], [0.3, 0.3, 0.0], [-0.05, 0.3, 0.0], [-0.15, 0.3, 0.0]], (T, 1, 1))
    O = np.tile([[-0.2, 0.0, 0.0], [-0.3, 0.3, 0.0], [0.05, 0.3, 0.0], [0.15, 0.3, 0.0]], (T, 1, 1))
    O[:, 1:, 1] -= gap
    O[:, 1:, 2] += 0.002
    radius = np.array([0.03, 0.03, 0.02])
    return Y, O, radius[ac.PAIR_Y] + radius[ac.PAIR_O] + 0.005


def _solve_stacked(depth_cost, max_deg=45.0, rotation=np.eye(3)):
    T = 10
    Y, O, R = _stacked(T, 0.05)
    Y, O = Y @ rotation.T, O @ rotation.T
    p = _params(depth_cost=depth_cost, max_deg=max_deg, axis=rotation @ [0.0, 0.0, 1.0])
    Q, _, ignored, _ = ac._resolve_sequential(Y, O, R, np.tile(rotation, (T, 1, 1)), p)
    fixed = _apply(Y, Q)
    cp, cq = ac._pairs(fixed, O)
    overlap = (R - np.linalg.norm(cp - cq, axis=-1)).max(1)
    image, depth = ac._shift(Y, fixed, p.axis)
    return Q, ignored, overlap, image[-1], depth[-1]


def test_depth_cost_moves_the_arm_along_the_camera_axis():
    """画像上で上下に重なった前腕を、depth_cost = 1 は上下（画像面内）に、0.5 は主に前後（奥行き）にずらして離す。"""
    _, ignored1, overlap1, image1, depth1 = _solve_stacked(1.0)
    _, ignored, overlap, image, depth = _solve_stacked(0.5)
    assert not ignored1.any() and not ignored.any()
    assert (overlap1 < 1e-3).all() and (overlap < 1e-3).all()
    assert image1 > 2.0 * depth1                   # 向きを区別しないと、最近点を結ぶ向き（上下）へずらす
    assert depth > 2.0 * image                     # 奥行き優先なら前後へずらす
    assert image < 0.5 * image1


def test_depth_cost_follows_the_given_camera_axis():
    """カメラの奥行きの軸ごと全体を回しても、腕の動き（画像面内・奥行きの移動量）は変わらない。"""
    rotation = quat.to_matrix(quat.from_rotvec(np.array([0.3, -0.8, 0.5])))
    _, _, _, image, depth = _solve_stacked(0.5)
    _, _, overlap, image_r, depth_r = _solve_stacked(0.5, rotation=rotation)
    assert (overlap < 1e-3).all()
    np.testing.assert_allclose([image_r, depth_r], [image, depth], atol=1e-4)


def test_depth_cost_is_relaxed_only_as_much_as_the_limit_needs():
    """奥行きへずらすと max_deg を超えるときは、depth_cost を離れるところまで弱めて、画像面内と奥行きの両方へ
    ずらす（向きを区別しない解へいきなり切り替えない）。"""
    Q1, _, _, image1, depth1 = _solve_stacked(1.0)
    _, _, _, image_d, depth_d = _solve_stacked(0.3)
    limit = np.rad2deg(ac._angle(Q1[-1])) + 0.5        # 最近点を結ぶ向きへずらすなら上限に届かない
    Y, O, R = _stacked(1, 0.05)
    p = _params(depth_cost=0.3, max_deg=limit)
    state = ac._solve_frame(Y[0], O[0], R, (np.eye(3), 0.0), p, (np.eye(3), 0.0))
    assert ac._angle(state[0]) >= p.max_angle - 1e-6          # depth_cost 0.3 のままでは上限に達して
    assert ac._overlapping(Y[0], O[0], R, state, p.tol).any()   # 離れない

    T = 10
    Y, O, R = _stacked(T, 0.05)
    Q, _, ignored, weight = ac._resolve_sequential(Y, O, R, np.tile(np.eye(3), (T, 1, 1)), p)
    fixed = _apply(Y, Q)
    cp, cq = ac._pairs(fixed, O)
    assert not ignored.any()
    assert ((R - np.linalg.norm(cp - cq, axis=-1)).max(1) < 1e-3).all()
    assert all(ac._angle(q) <= p.max_angle + 1e-6 for q in Q)
    assert (0.3 ** 2 < weight).all() and (weight < 1.0).all()   # 弱めたが、向きを区別しない所までは戻さない
    image, depth = (x[-1] for x in ac._shift(Y, fixed, p.axis))
    assert image_d < image < image1                    # 画像面内の動きは必要な分だけ増え
    assert depth1 < depth < depth_d                    # 奥行きの動きも残る


def test_depth_cost_in_the_pipeline(body_model):
    motion = add_arm_cross(synthetic_walk(num_frames=30))
    shoulder = 'arm_collision.elbow=false'             # 肩だけで比べる（ひじも使う場合は下の _solve_arms で）
    iso = _convert(body_model, motion, 'left', extra=[shoulder, 'arm_collision.depth_cost=1.0']).arm_collision
    r = _convert(body_model, motion, 'left', extra=[shoulder])   # 既定は depth_cost 0.5
    a = r.arm_collision
    assert a.depth_cost == 0.5 and iso.depth_cost == 1.0
    assert a.relaxed_frames == 0 and np.allclose(a.depth_cost_used, 0.5)
    assert a.overlap_frames(0.005 * r.scale)[1] == 0
    assert a.shift_image.max() < iso.shift_image.max()
    assert a.shift_depth.max() > iso.shift_depth.max()
    assert r.info['arm_collision']['max_shift_cm']['depth'] > 0.0
    for bad in ('0', '1.5'):
        with pytest.raises(ValueError):
            _convert(body_model, motion, 'left', extra=[f'arm_collision.depth_cost={bad}'])


# ---- ひじも曲げる（elbow）----
ARM_RADIUS = np.array([0.035, 0.03, 0.025])
ARMS = {
    # 上腕が前下がりで、ひじを 90 度ほど曲げた前腕が胸の前で同じ奥行きで交わる
    'cross': ([[0.18, 0.45, 0.0], [0.2, 0.3, 0.15], [-0.06, 0.38, 0.22], [-0.16, 0.39, 0.24]],
              [[-0.18, 0.45, 0.0], [-0.2, 0.3, 0.15], [0.06, 0.38, 0.22], [0.16, 0.39, 0.24]]),
    # 腕組み: 上腕はほぼ真下、前腕は腹の前で水平に重なる（左がわずかに上・同じ奥行き）
    'folded': ([[0.18, 0.45, 0.0], [0.2, 0.18, 0.06], [-0.07, 0.21, 0.16], [-0.15, 0.22, 0.17]],
               [[-0.18, 0.45, 0.0], [-0.2, 0.17, 0.06], [0.07, 0.19, 0.16], [0.15, 0.2, 0.17]]),
}


class _ElbowCfg:
    elbow_max_deg = 30.0


def _solve_arms(name, elbow, depth_cost=0.5, elbow_max_deg=30.0, T=12):
    """ARMS[name] の腕（左が自重する）を解く。戻り値: (補正後の腕の点 (T, 4, 3), 推定の腕の点, ひじの曲げ (T,),
    _Elbow のリスト, 補正後の重なりの深さ (T,))。"""
    Y = np.tile(np.array(ARMS[name][0]), (T, 1, 1))
    O = np.tile(np.array(ARMS[name][1]), (T, 1, 1))
    R = ARM_RADIUS[ac.PAIR_Y] + ARM_RADIUS[ac.PAIR_O] + 0.005
    u, f = Y[0, 1] - Y[0, 0], Y[0, 2] - Y[0, 1]
    b = np.tile(np.cross(u, f) / np.linalg.norm(np.cross(u, f)), (T, 1))   # + に回すと曲げが深くなる
    cfg = _ElbowCfg()
    cfg.elbow_max_deg = elbow_max_deg
    elbows = ac._elbows(Y, b, cfg) if elbow else None
    Q, bend, ignored, _ = ac._resolve_sequential(Y, O, R, np.tile(np.eye(3), (T, 1, 1)),
                                                 _params(depth_cost=depth_cost), elbows)
    assert not ignored.any()
    fixed = ac._pose_all(Y, Q, bend, b if elbow else None)
    return fixed, Y, bend, elbows, ac.overlap_depth(fixed, O, ARM_RADIUS, ARM_RADIUS)


def _moves(fixed, Y):
    """(ひじ・手首・手の先の画像面内の移動 (3,), 奥行きの移動 (3,))（最後のフレーム。奥行きの軸は Z）。"""
    move = fixed[-1, 1:] - Y[-1, 1:]
    return np.linalg.norm(move[:, :2], axis=-1), np.abs(move[:, 2])


@pytest.mark.parametrize('name', ['cross', 'folded'])
def test_elbow_keeps_the_arm_in_the_image(name):
    """ひじも曲げると、肩だけより、ひじ・手首・手の先の画像面内の動きが小さく（その分は前後へ）なる。"""
    fixed0, Y, _, _, after0 = _solve_arms(name, elbow=False)
    fixed, _, bend, _, after = _solve_arms(name, elbow=True)
    assert (after0 < 1e-3).all() and (after < 1e-3).all()
    image0, _ = _moves(fixed0, Y)
    image, depth = _moves(fixed, Y)
    assert np.rad2deg(np.abs(bend[-1])) > 5.0          # ひじを曲げ伸ばしして
    assert image[0] < 0.01                             # ひじは画像上ほぼ動かず
    assert (image[1:] < 0.6 * image0[1:]).all()        # 手首・手の先の画像面内の動きは肩だけの 6 割未満
    assert depth[1:].max() > image[1:].max()           # 主に前後へ動く


def test_elbow_bend_limits():
    """ひじの補正は ±elbow_max_deg、曲げ角（0 = まっすぐ）は 0〜150 度の範囲（推定がその外なら、それ以上外へ動かさない）。"""
    b = np.array([[0.0, 0.0, 1.0]])
    for flex_deg, (lo, hi) in ((10.0, (-10.0, 30.0)), (-5.0, (0.0, 30.0)), (140.0, (-30.0, 10.0)),
                               (160.0, (-30.0, 0.0)), (90.0, (-30.0, 30.0))):
        f = np.deg2rad(flex_deg)
        # 上腕は +x、前腕は z 軸まわりに flex_deg 回した向き
        Y = np.array([[[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3 + 0.25 * np.cos(f), 0.25 * np.sin(f), 0.0],
                       [0.3 + 0.4 * np.cos(f), 0.4 * np.sin(f), 0.0]]])
        np.testing.assert_allclose(np.rad2deg(ac._flexion(Y, b)), [flex_deg], atol=1e-9)
        e = ac._elbows(Y, b, _ElbowCfg())[0]
        np.testing.assert_allclose(np.rad2deg([e.lo, e.hi]), [lo, hi], atol=1e-9, err_msg=str(flex_deg))
    # 解いた曲げは範囲の中（elbow_max_deg を小さくしても、残りは肩で離す）
    fixed, Y, bend, elbows, after = _solve_arms('cross', elbow=True, elbow_max_deg=5.0)
    assert (after < 1e-3).all()
    assert all(e.lo - 1e-9 <= a <= e.hi + 1e-9 for a, e in zip(bend, elbows))
    assert np.rad2deg(np.abs(bend)).max() <= 5.0 + 1e-6
    b3 = np.tile(elbows[0].b, (len(Y), 1))
    flex = np.rad2deg(ac._flexion(fixed, b3))
    assert ((0.0 <= flex) & (flex <= 150.0)).all()


def test_corrected_rotations_give_the_resolved_arms(body_model):
    """補正した腕・ひじのローカル回転（VMD に入る回転）から FK で求めた腕の位置が、ステージ9a の解いた位置と一致して
    重ならない（ステージ9h で直した腕の姿勢も含めて判定する）。"""
    T = 30
    for right_depth in (0.0, 0.05):
        motion = add_arm_cross(synthetic_walk(num_frames=T), right_depth=right_depth)
        r = _convert(body_model, motion, 'left', extra=['wrist_limits.enabled=false'])
        assert r.hand_reach.info['frames'][0] > 0 and r.hand_reach.shift[:, 0].max() > 0.01 * r.scale   # 9h が腕を直している
        model = arm_model(r.skeleton, r.config.arm_collision, r.scale)
        pts = ac.arm_points(r.skeleton, ac.globals_from_local(r.retargeter, r.kin.glob_rot, r.local_quats),
                            model, T)
        depth = ac.overlap_depth(pts[:, 0], pts[:, 1], model.radius[0], model.radius[1])
        np.testing.assert_allclose(depth, r.arm_collision.depth_after, atol=1e-6 * r.scale)
        assert depth.max() < 0.0


def test_no_change_when_the_arms_do_not_touch(body_model):
    motion = synthetic_walk(num_frames=60)               # 腕は体の横に下ろしたまま
    base = _convert(body_model, motion, 'none')
    r = _convert(body_model, motion, 'left')
    assert r.arm_collision.correction_deg.max() < 1e-6
    for name in ARM_BONES:
        np.testing.assert_allclose(r.local_quats[name], base.local_quats[name])


def test_variants_use_the_corrected_arm(body_model):
    from nlf2vmd.variants import variant_tracks
    from nlf2vmd.vmd import to_mmd_quat
    r = _convert(body_model, add_arm_cross(synthetic_walk(num_frames=40)), 'left')
    for bone in ('左腕', '左ひじ'):
        expected = to_mmd_quat(quat.make_continuous(r.local_quats[bone]))
        for kind in ('full', 'no_move', 'upper_body'):
            track = next(t for t in variant_tracks(r, kind, log=None) if t.name == bone)
            np.testing.assert_allclose(track.rotations, expected, atol=1e-6)


def test_invalid_mode(body_model):
    with pytest.raises(ValueError):
        _convert(body_model, synthetic_walk(num_frames=20), 'both')


# ---- 腕の太さ（PMX のメッシュから）----
def _ring(center, axis, radius, n=12):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    u = np.cross(axis, [0.0, 0.0, 1.0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.asarray(center) + radius * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v)


def _arm_mesh_pmx(tmp_path, radii, hand_length):
    """標準ボーン（＋左右の 腕捩）と、上腕・前腕・手のまわりに半径 radii の輪を並べた頂点の PMX。"""
    bones = standard_pmx_bones()
    names = [b[0] for b in bones]
    for s in '左右':   # 腕捩（腕の子）: 上腕の頂点の一部をこのボーンに付けて、根元へたどれることを確かめる
        arm, elbow = STANDARD_BONES[s + '腕'][0], STANDARD_BONES[s + 'ひじ'][0]
        bones.append((s + '腕捩', tuple(0.5 * (np.add(arm, elbow))), names.index(s + '腕'), None, False))
    names = [b[0] for b in bones]
    verts = []
    for s in '左右':
        P = {n: np.array(STANDARD_BONES[s + n][0]) for n in ('腕', 'ひじ', '手首')}
        axis = P['手首'] - P['ひじ']
        tip = P['手首'] + hand_length * axis / np.linalg.norm(axis)
        segs = [(P['腕'], P['ひじ'], s + '腕捩', radii[0]), (P['ひじ'], P['手首'], s + 'ひじ', radii[1]),
                (P['手首'], tip, s + '手首', radii[2])]
        for a, b, bone, r in segs:
            for t in np.linspace(0.1, 1.0 if bone.endswith('手首') else 0.9, 8):
                for pos in _ring(a + t * (b - a), b - a, r):
                    verts.append((tuple(pos), 1, (names.index(bone), 0), (0.9,)))   # BDEF2
    for pos in _ring(STANDARD_BONES['上半身'][0], (0, 1, 0), 1.0):                   # 体の頂点
        verts.append((tuple(pos), 0, (names.index('上半身'),), ()))
    path = tmp_path / 'arms.pmx'
    path.write_bytes(make_pmx(bones, vertices=verts))
    return path, names


def test_arm_radius_from_pmx_mesh(tmp_path):
    radii, hand_length = (0.35, 0.28, 0.2), 1.4
    path, names = _arm_mesh_pmx(tmp_path, radii, hand_length)
    model = read_pmx(path)
    assert model.vertices.shape == (2 * 3 * 8 * 12 + 12, 3)
    assert set(np.unique(model.vertex_bones)) == {names.index(n) for n in (
        '左腕捩', '左ひじ', '左手首', '右腕捩', '右ひじ', '右手首', '上半身')}
    skel = Skeleton.from_pmx(model)
    cfg = load_config().arm_collision
    arms = arm_model(skel, cfg, 1.0)
    assert arms.source == 'mesh'
    np.testing.assert_allclose(arms.radius, np.tile(radii, (2, 1)), rtol=0.02)
    for side, s in enumerate('左右'):
        length = np.linalg.norm(arms.tip[side] - skel.internal(s + '手首'))
        assert abs(length - hand_length) < 0.1 * hand_length

    # 半径を設定で指定すると、メッシュより優先する（メートル × 単位）
    cfg = load_config(overrides=['arm_collision.radius_m=[0.1, 0.2, 0.3]']).arm_collision
    np.testing.assert_allclose(arm_model(skel, cfg, 2.0).radius, [[0.2, 0.4, 0.6]] * 2)
    # メッシュが無い（標準ボーン）ときは fallback_radius_m
    arms = arm_model(Skeleton.standard(), load_config().arm_collision, 10.0)
    assert arms.source == 'config'
    np.testing.assert_allclose(arms.radius[0], np.array(cfg.fallback_radius_m) * 10.0)


def test_thicker_arms_are_pushed_further_apart(body_model):
    motion = add_arm_cross(synthetic_walk(num_frames=30))
    thin = _convert(body_model, motion, 'left')
    thick = _convert(body_model, motion, 'left', extra=['arm_collision.radius_scale=1.5'])
    assert thick.arm_collision.correction_deg.max() > thin.arm_collision.correction_deg.max()
    assert thick.arm_collision.overlap_frames(0.005 * thick.scale)[1] == 0


def test_resolve_with_identity_pose_is_a_no_op():
    """T ポーズ（腕を横に伸ばす）では左右の腕が遠いので、何も変えない。"""
    from nlf2vmd.retarget import Retargeter
    from nlf2vmd.synthetic import SMPL_REST_JOINTS
    skel = Skeleton.standard()
    rt = Retargeter(skel, SMPL_REST_JOINTS)
    G = np.tile(np.eye(3), (5, 24, 1, 1))
    local = rt.local_quats(G)
    fixed, res = resolve_arm_collisions(skel, rt, G, local, load_config().arm_collision, 10.0, 30.0)
    assert res.correction_deg.max() == 0.0
    assert (res.depth_before < 0).all()
    for name in local:
        np.testing.assert_allclose(fixed[name], local[name])
