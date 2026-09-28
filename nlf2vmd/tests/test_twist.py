"""捩りボーン（腕捩・手捩）へのひねりの振り分け: 腕・ひじ・手首のひねりを捩りボーンへ移し、腕の姿勢は変えないこと。"""
import numpy as np
import pytest

from nlf2vmd import convert, load_config, quat
from nlf2vmd.skeleton import STANDARD_BONES, Skeleton
from nlf2vmd.synthetic import add_arm_reach, synthetic_walk
from nlf2vmd.twist import split_twist, twist_angle, twist_chains
from nlf2vmd.vmd import to_mmd_quat

ARM = ('腕', 'ひじ', '手首')


def twist_skeleton(elbow_under_twist=True, axis_tilt_deg=0.0):
    """標準ボーンの 腕 → ひじ、ひじ → 手首 の間に 腕捩・手捩 を入れた骨格（軸制限は骨の向きを axis_tilt_deg 傾けたもの）。"""
    pos = {k: np.array(v[0], float) for k, v in STANDARD_BONES.items()}
    par = {k: v[1] for k, v in STANDARD_BONES.items()}
    axes = {}
    tilt = quat.from_rotvec([0.0, 0.0, np.deg2rad(axis_tilt_deg)])
    for s in '左右':
        for twist, a, b in (('腕捩', '腕', 'ひじ'), ('手捩', 'ひじ', '手首')):
            pos[s + twist] = pos[s + a] + 0.6 * (pos[s + b] - pos[s + a])
            par[s + twist] = s + a
            axes[s + twist] = quat.rotate(tilt, pos[s + b] - pos[s + a])
        if elbow_under_twist:
            par[s + 'ひじ'] = s + '腕捩'
        par[s + '手首'] = s + '手捩'
    return Skeleton(pos, par, {}, axes=axes)


def _config(**twist):
    return load_config(overrides=[f'twist.{k}={v}' for k, v in twist.items()]).twist


def _random_arm(T, seed, max_deg=120.0):
    rng = np.random.default_rng(seed)
    local = {}
    for s in '左右':
        for b in ARM:
            v = rng.normal(size=(T, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            v *= np.deg2rad(rng.uniform(0.0, max_deg, (T, 1)))
            local[s + b] = quat.from_rotvec(v)
    return local


def _chain(local, names):
    q = local[names[0]]
    for n in names[1:]:
        q = quat.mul(q, local[n])
    return q


def _angle_deg(a, b):
    return np.rad2deg(quat.angle_between(a, b))


def _forearm(skel, s='左'):
    """前腕の軸と、それに直交する曲げの軸（内部座標の単位ベクトル）。"""
    axis = skel.internal(s + '手首') - skel.internal(s + 'ひじ')
    axis /= np.linalg.norm(axis)
    side = np.cross(axis, [0.0, 0.0, 1.0])
    return axis, side / np.linalg.norm(side)


def _positions(skel, local, s):
    """肩を動かさないときの ひじ・手首 の位置 (T, 2, 3)（キーの無いボーンは親の回転を受け継ぐ）。"""
    names = [s + n for n in ('腕', '腕捩', 'ひじ', '手捩', '手首')]
    T = len(local[s + '腕'])
    R = np.tile(np.eye(3), (T, 1, 1))
    p = np.tile(skel.internal(names[0]), (T, 1))
    out = {}
    for parent, child in zip(names[:-1], names[1:]):
        if parent in local:
            R = R @ quat.to_matrix(local[parent])
        p = p + R @ (skel.internal(child) - skel.internal(parent))
        out[child] = p
    return np.stack([out[s + 'ひじ'], out[s + '手首']], axis=1)


def test_split_keeps_the_arm_pose():
    skel = twist_skeleton()
    local = _random_arm(400, seed=0)
    r = split_twist(skel, local, _config())
    assert sorted(c.twist for c in r.chains) == ['右手捩', '右腕捩', '左手捩', '左腕捩']
    for s in '左右':
        # 腕 · 腕捩 は元の 腕、腕 から 手首 までの積は元のまま（手の向きは変わらない）
        assert _angle_deg(_chain(r.local, [s + '腕', s + '腕捩']), local[s + '腕']).max() < 1e-4
        assert _angle_deg(_chain(r.local, [s + n for n in ('腕', '腕捩', 'ひじ', '手捩', '手首')]),
                          _chain(local, [s + n for n in ARM])).max() < 1e-4
        # ひじ・手首の位置も変わらない（捩りボーンは 捩り → 子 の線の上でひねる）
        np.testing.assert_allclose(_positions(skel, r.local, s), _positions(skel, local, s),
                                   atol=1e-9)
        for ch in r.chains:
            if ch.twist.startswith(s):
                # 捩りボーンは軸まわりにしか回らない
                rv = quat.to_rotvec(r.local[ch.twist])
                assert np.abs(rv - (rv @ ch.axis)[:, None] * ch.axis).max() < 1e-9
                # ひねりを移した後の 腕・ひじ には、その軸まわりのひねりが残らない（上限・曲げの減衰の外では）
                theta, _ = twist_angle(r.local[ch.parent], ch.axis, 0.0)
                assert np.median(np.abs(np.rad2deg(theta))) < 1e-4


def test_forearm_twist_goes_to_the_wrist_twist_bone():
    """手首を前腕の軸まわりに 70 度ひねると、手捩 が 70 度になり、手首には曲げだけが残る（ひじのひねりも 手捩 へ）。"""
    skel = twist_skeleton()
    T = 10
    axis, side = _forearm(skel)
    bend = quat.from_rotvec(np.tile(np.deg2rad(30.0) * side, (T, 1)))
    local = {'左腕': np.tile(quat.IDENTITY, (T, 1)),
             '左ひじ': quat.mul(bend, quat.from_rotvec(np.tile(np.deg2rad(20.0) * axis, (T, 1)))),
             '左手首': quat.from_rotvec(np.tile(np.deg2rad(50.0) * axis, (T, 1)))}
    r = split_twist(skel, local, _config())
    np.testing.assert_allclose(r.angle_deg['左手捩'], 70.0, atol=1e-6)
    assert _angle_deg(r.local['左手首'], quat.IDENTITY).max() < 1e-4
    assert _angle_deg(r.local['左ひじ'], bend).max() < 1e-4
    np.testing.assert_allclose(r.angle_deg['左腕捩'], 0.0, atol=1e-6)


def test_turning_the_palm_up_does_not_flip():
    """手のひらを上に向ける（前腕の軸まわりに 0 → 200 度 → 0 度）動きで、手捩 が ±180 度で逆向きに跳ねず、
    上限（160 度）を超える分は手首に残る。"""
    skel = twist_skeleton()
    T = 240
    axis, side = _forearm(skel)
    angle = np.deg2rad(200.0) * np.sin(np.pi * np.arange(T) / (T - 1))
    wrist = quat.make_continuous(quat.mul(quat.from_rotvec(np.tile(0.3 * side, (T, 1))),
                                          quat.from_rotvec(angle[:, None] * axis)))
    local = {'左腕': np.tile(quat.IDENTITY, (T, 1)), '左ひじ': np.tile(quat.IDENTITY, (T, 1)),
             '左手首': wrist}
    r = split_twist(skel, local, _config())
    fore = r.angle_deg['左手捩']
    assert np.abs(np.diff(fore)).max() < 3.0
    assert fore.max() == pytest.approx(160.0, abs=1e-6)
    top = np.rad2deg(angle) > 160.0
    theta, _ = twist_angle(r.local['左手首'], axis, 0.0)
    np.testing.assert_allclose(np.rad2deg(theta[top]), np.rad2deg(angle[top]) - 160.0, atol=1e-6)
    assert _angle_deg(_chain(r.local, ['左ひじ', '左手捩', '左手首']), wrist).max() < 1e-4


def test_twist_fades_where_the_arm_points_opposite_its_rest_direction():
    """腕を初期姿勢の向きの反対へ回す（曲げ 180 度: ひねりが決まらない）所では 腕捩 を 0 にし、前後でも跳ねない。"""
    skel = twist_skeleton()
    T = 181
    bone = skel.internal('左ひじ') - skel.internal('左腕')
    bone /= np.linalg.norm(bone)
    side = np.cross(bone, [0.0, 0.0, 1.0])
    side /= np.linalg.norm(side)
    swing = quat.from_rotvec(np.deg2rad(np.linspace(0.0, 180.0, T))[:, None] * side)
    arm = quat.mul(swing, quat.from_rotvec(np.tile(np.deg2rad(40.0) * bone, (T, 1))))
    local = {'左腕': arm, '左ひじ': np.tile(quat.IDENTITY, (T, 1)),
             '左手首': np.tile(quat.IDENTITY, (T, 1))}
    r = split_twist(skel, local, _config())
    upper = r.angle_deg['左腕捩']
    assert np.isfinite(upper).all()
    np.testing.assert_allclose(upper[:100], 40.0, atol=1e-6)     # 曲げ 100 度まではすべて移す
    assert abs(upper[-1]) < 1e-4                                  # 曲げ 180 度では移さない
    assert np.abs(np.diff(upper)).max() < 5.0
    assert _angle_deg(_chain(r.local, ['左腕', '左腕捩']), arm).max() < 1e-4


@pytest.mark.parametrize('skel,reason', [
    (Skeleton.standard(), None),
    (twist_skeleton(elbow_under_twist=False), '左腕捩 が 左腕 → 左ひじ の間にない'),
    (twist_skeleton(axis_tilt_deg=20.0), '左腕捩 の軸・位置が'),
])
def test_unusable_twist_bones_leave_the_arm_as_is(skel, reason):
    """捩りボーンが無い・親 → 捩り → 子 につながっていない・軸が骨の向きと合わないモデルでは、その捩りボーンを使わない。"""
    local = _random_arm(20, seed=1)
    r = split_twist(skel, local, _config())
    if reason is None:
        assert r.skipped == [] and r.chains == []
        assert set(r.local) == set(local)
    else:
        assert any(n.startswith(reason) for n in r.skipped)
    names = {c.twist for c in r.chains}
    for s in '左右':
        if s + '腕捩' not in names:
            np.testing.assert_array_equal(r.local[s + '腕'], local[s + '腕'])


def test_disabled_keeps_the_local_rotations():
    local = _random_arm(5, seed=2)
    r = split_twist(twist_skeleton(), local, _config(enabled='false'))
    assert r.local == local and not r.enabled


def test_convert_writes_twist_bone_keys(tmp_path, body_model):
    """捩りボーンのあるモデルに変換すると 腕捩・手捩 のキーが入り、腕 から 手首 までの回転はキーを分ける前と同じ。"""
    motion = add_arm_reach(synthetic_walk(num_frames=60), 0, [0.3, -0.8, 0.5], [0.05, 0.1, 0.3],
                           twist_deg=90.0)
    skel = twist_skeleton()
    cfg = load_config(overrides=['diagnostics.enabled=false'])
    r = convert(motion, tmp_path / 't.vmd', pmx=skel, body_model=body_model, config=cfg, log=None)
    tracks = {t.name: t for t in r.tracks}
    assert {'左腕捩', '左手捩', '右腕捩', '右手捩'} <= set(tracks)
    assert r.info['twist']['bones'] and r.info['twist']['max_deg']['左手捩'] > 45.0
    T = r.motion.num_frames
    for s in '左右':
        from_keys = _chain({n: to_mmd_quat(tracks[n].rotations) for n in tracks},
                           [s + n for n in ('腕', '腕捩', 'ひじ', '手捩', '手首')])
        assert len(from_keys) == T
        assert _angle_deg(from_keys, _chain(r.local_quats, [s + n for n in ARM])).max() < 1e-4
    cfg_off = load_config(overrides=['diagnostics.enabled=false', 'twist.enabled=false'])
    off = convert(motion, None, pmx=skel, body_model=body_model, config=cfg_off, log=None)
    assert not {'左腕捩', '左手捩'} & {t.name for t in off.tracks}


def test_chains_use_the_fixed_axis():
    skel = twist_skeleton(axis_tilt_deg=3.0)
    chains, skipped = twist_chains(skel)
    assert not skipped
    ch = next(c for c in chains if c.twist == '左手捩')
    expected = skel.axis_internal('左手捩')
    assert abs(np.dot(ch.axis, expected / np.linalg.norm(expected))) > 1.0 - 1e-12
