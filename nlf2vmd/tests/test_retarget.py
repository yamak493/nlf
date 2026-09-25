"""上半身の回転リターゲット: 初期姿勢の違い（T ポーズと A ポーズ）だけを補正すること。

関節を結んだ向きを合わせると、SMPL と PMX で関節の置き方が違う所（背骨・首・頭・肩・足先）に
余計な回転が入る（首が前に倒れて顎が出る、肩がすくむ、足裏が傾く）。
"""
import numpy as np
import pytest

from nlf2vmd import quat
from nlf2vmd.body_model import rest_info
from nlf2vmd.retarget import Retargeter
from nlf2vmd.skeleton import Skeleton
from nlf2vmd.synthetic import SMPL_REST_JOINTS, synthetic_body_model

NEUTRAL = ['下半身', '上半身', '上半身2', '首', '頭', '左肩', '右肩', '左手首', '右手首']


def _retargeter(**options):
    rest = rest_info(synthetic_body_model(), np.zeros(10))
    axes = rest.points[:, 1].mean(1) - rest.points[:, 0].mean(1)
    return Retargeter(Skeleton.standard(), SMPL_REST_JOINTS, axes, options)


def _deg(q):
    return float(np.degrees(np.linalg.norm(quat.to_rotvec(q))))


def _identity_pose():
    return np.tile(np.eye(3), (1, 24, 1, 1))


def test_neutral_smpl_pose_keeps_torso_neck_head_at_rest():
    local = _retargeter().local_quats(_identity_pose())
    for name in NEUTRAL:
        assert _deg(local[name][0]) < 1.0, name


def test_arms_are_raised_from_a_pose_to_t_pose():
    rt = _retargeter()
    S = Skeleton.standard().internal
    J = SMPL_REST_JOINTS
    for s, (shoulder, elbow) in (('左', (16, 18)), ('右', (17, 19))):
        R = rt.global_matrix(s + '腕', _identity_pose())[0]
        d = R @ (S(s + 'ひじ') - S(s + '腕'))
        target = J[elbow] - J[shoulder]
        assert np.dot(d / np.linalg.norm(d), target / np.linalg.norm(target)) > 0.9999


def test_neck_tilt_moves_the_neck_not_the_chin():
    """SMPL が首を 20 度前に倒すと、MMD の首が 20 度倒れ、頭は首に対して回らない。"""
    G = _identity_pose()
    tilt = quat.to_matrix(quat.from_rotvec([np.deg2rad(20.0), 0.0, 0.0]))
    G[0, [12, 15]] = tilt
    local = _retargeter().local_quats(G)
    assert abs(_deg(local['首'][0]) - 20.0) < 0.5
    assert _deg(local['頭'][0]) < 0.5


def _foot_pitch(v):
    return float(np.degrees(np.arcsin(-v[1] / np.linalg.norm(v))))


@pytest.mark.parametrize('foot_mode,flat', [('yaw', True), ('direction', False)])
def test_planted_foot_stays_flat(run_walk, foot_mode, flat):
    """接地中（足裏が床に平行）の足は、MMD でも足首→つま先の傾きが初期姿勢と同じになる。"""
    _, r = run_walk([f'retarget.foot={foot_mode}'], num_frames=180)
    skel = Skeleton.standard()
    errors = []
    for side, s in enumerate('左右'):
        rest_dir = skel.positions[s + 'つま先ＩＫ'] - skel.positions[s + '足ＩＫ']   # MMD 座標
        track = next(t for t in r.tracks if t.name == s + '足ＩＫ')
        for st, _ in r.contact.segments[side]:
            d = quat.rotate(track.rotations[st], rest_dir)
            errors.append(abs(_foot_pitch(d) - _foot_pitch(rest_dir)))
    assert errors
    if flat:
        assert max(errors) < 2.0
    else:
        assert min(errors) > 5.0          # 旧方式では足裏が傾く（このテストで検出できること）
