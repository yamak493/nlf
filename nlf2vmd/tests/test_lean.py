"""前後の傾きの補正: 足を床に着けたまま体全体がカメラの前後に傾いた推定を起こし、本当の前傾（お辞儀）は残すこと。"""
import numpy as np
import pytest

from nlf2vmd import convert, load_config, quat
from nlf2vmd.body_model import compute_kinematics, rest_info
from nlf2vmd.lean import (FIXED, UPPER, LeanResult, _evaluate, body_com, com_along, com_weights,
                          pivot_height, rotation_about)
from nlf2vmd.synthetic import (SMPL_REST_JOINTS, add_depth_noise, synthetic_walk,
                               to_camera_coords)


def _convert(tmp_path, body_model, motion, overrides=(), name='l.vmd', camera=True):
    src = add_depth_noise(to_camera_coords(motion, height=1.6, pitch_deg=10.0), seed=0) \
        if camera else motion
    cfg = load_config(overrides=['diagnostics.enabled=false', *overrides])
    return convert(src, tmp_path / name, body_model=body_model, config=cfg, log=None)


def _torso_lean_deg(r, direction=None, joint=9):
    """出力の背骨（既定は背骨3）の上向きが、direction（既定はカメラの奥行き方向）へ倒れている角度の中央値。"""
    d = r.lean.direction if direction is None else np.asarray(direction, float)
    up = r.kin.glob_rot[:, joint] @ np.array([0.0, 1.0, 0.0])
    return float(np.median(np.rad2deg(np.arctan2(up @ d, up[:, 1]))))


def _correction_deg(r):
    return float(np.median(np.rad2deg(r.lean.angle)))


WALKS = {'walk': dict(speed=1.0), 'step_in_place': dict(speed=0.0, sway=0.08)}


@pytest.mark.parametrize('lean_deg', [10.0, -10.0])
@pytest.mark.parametrize('walk', list(WALKS), ids=list(WALKS))
def test_lean_toward_the_camera_is_straightened(tmp_path, body_model, walk, lean_deg):
    """高さ 1.6m・10 度見下ろしのカメラの方へ（から）体全体が 10 度傾いた推定。足は床に平らに着いている。"""
    kw = dict(num_frames=300, noise_deg=0.5, **WALKS[walk])
    ref = _convert(tmp_path, body_model, synthetic_walk(**kw), ['lean.enabled=false'], 'ref.vmd')
    leaned = synthetic_walk(lean_deg=lean_deg, **kw)
    before = _convert(tmp_path, body_model, leaned, ['lean.enabled=false'], 'before.vmd')
    after = _convert(tmp_path, body_model, leaned, name='after.vmd')
    assert abs(_torso_lean_deg(before) - _torso_lean_deg(ref) - lean_deg) < 1.0   # 補正なしでは傾いたまま
    assert abs(_torso_lean_deg(after) - _torso_lean_deg(ref)) < 3.0   # 傾いていない推定との差が 3 度未満
    assert abs(np.nanmedian(after.lean.after_deg)) < 1.0      # 重心が支持点の上に来る
    # 骨盤も足の上へ戻る（接地中の骨盤と足ＩＫの奥行き方向の差が、傾いていない推定と数 cm 以内）
    def pelvis_ahead_cm(r):
        fl = r.contact.flags
        feet = (r.foot_ik.target @ r.lean.direction * fl).sum(1) / np.maximum(fl.sum(1), 1)
        return float(np.median((r.kin.root_pos @ r.lean.direction - feet)[fl.any(1)]) / r.scale * 100)
    assert abs(pelvis_ahead_cm(before) - pelvis_ahead_cm(ref)) > 10.0
    assert abs(pelvis_ahead_cm(after) - pelvis_ahead_cm(ref)) < 4.0


@pytest.mark.parametrize('walk', list(WALKS), ids=list(WALKS))
def test_upright_motion_is_almost_unchanged(tmp_path, body_model, walk):
    r = _convert(tmp_path, body_model, synthetic_walk(num_frames=300, noise_deg=0.5, **WALKS[walk]))
    assert abs(_correction_deg(r)) < 2.0
    assert np.ptp(np.rad2deg(r.lean.angle)) < 1.0     # 時間窓で求めても、ほぼ一定


def _balanced_bow(body_model, bow_deg, num_frames=300):
    """その場で足踏みしながら背骨1を bow_deg だけ前へ曲げ、重心が直立のときと同じ位置に残るよう骨盤を後ろへ引く。"""
    rest = rest_info(body_model, np.zeros(10))

    def make(shift):
        m = synthetic_walk(num_frames=num_frames, speed=0.0, noise_deg=0.5, pelvis_shift=shift)
        pose = np.array(m['pose'], copy=True)
        pose[:, 3] = [np.deg2rad(bow_deg), 0.0, 0.0]
        return dict(m, pose=pose)

    def com_ahead(m):
        kin = compute_kinematics(quat.from_rotvec(m['pose']), m['trans'] + SMPL_REST_JOINTS[0],
                                 rest)
        return float(np.mean(body_com(kin.joints)[:, 2] - kin.joints[:, [7, 8], 2].mean(1)))

    upright = com_ahead(synthetic_walk(num_frames=num_frames, speed=0.0, noise_deg=0.5))
    shift = 0.0
    for _ in range(4):
        shift -= com_ahead(make(shift)) - upright
    return make(shift), shift


def test_balanced_bow_is_kept(tmp_path, body_model):
    """お辞儀（上半身を 30 度前へ倒し、腰を後ろへ引いて重心を足の上に残す）は、傾きの誤差として起こさない。
    体の縦のラインを鉛直に直す方法では、この前傾が消える。"""
    bowed, shift = _balanced_bow(body_model, 30.0)
    assert shift < -0.05                                    # 腰を 5cm 以上後ろへ引いている
    ref = _convert(tmp_path, body_model, synthetic_walk(num_frames=300, speed=0.0, noise_deg=0.5),
                   name='ref.vmd')
    r = _convert(tmp_path, body_model, bowed, name='bow.vmd')
    assert abs(_correction_deg(r) - _correction_deg(ref)) < 1.5
    assert abs(_torso_lean_deg(r) - _torso_lean_deg(ref) - 30.0) < 2.0


def test_lean_across_the_image_is_not_touched(tmp_path, body_model):
    """カメラの前を横切る向き（+X）に歩き、進行方向へ 10 度傾く。左右の傾きは画像に写るので補正しない。"""
    kw = dict(num_frames=300, noise_deg=0.5, speed=1.0, heading_deg=90.0)
    ref = _convert(tmp_path, body_model, synthetic_walk(**kw), name='ref.vmd')
    r = _convert(tmp_path, body_model, synthetic_walk(lean_deg=10.0, **kw), name='side.vmd')
    across = np.cross([0.0, 1.0, 0.0], r.lean.direction)
    tilt = [_torso_lean_deg(x, across) for x in (r, ref)]
    assert abs(_correction_deg(r) - _correction_deg(ref)) < 1.5
    assert abs(abs(tilt[0] - tilt[1]) - 10.0) < 1.5         # 横切る向きの傾きは残る


def test_window_zero_gives_one_angle(tmp_path, body_model):
    r = _convert(tmp_path, body_model, synthetic_walk(num_frames=240, lean_deg=8.0),
                 ['lean.window_sec=0'])
    assert np.ptp(r.lean.angle) == 0.0
    assert abs(_correction_deg(r) + 8.0) < 3.0


def test_disabled_keeps_the_pose(tmp_path, body_model):
    m = synthetic_walk(num_frames=120, lean_deg=8.0)
    r = _convert(tmp_path, body_model, m, ['lean.enabled=false'], camera=False)
    assert not r.lean.enabled and np.all(r.lean.angle == 0.0)
    assert 'lean_deg' not in r.metrics


def test_apply_moves_only_the_upper_body_and_matches_the_com_model(tmp_path, body_model):
    """apply は足首から下を変えず、重心の奥行き方向の位置は com_along の式と一致する。"""
    r = _convert(tmp_path, body_model, synthetic_walk(num_frames=90), ['lean.enabled=false'],
                 camera=False)
    kin = r.kin
    T = len(kin.joints)
    d = np.array([0.6, 0.0, 0.8])
    axis = np.cross([0.0, 1.0, 0.0], d)
    theta = np.deg2rad(np.linspace(-20.0, 20.0, T))
    height = pivot_height(kin, r.fps)
    shift = (height * np.tan(theta))[:, None] * d
    nan = np.full(T, np.nan)
    lean = LeanResult(theta, shift, d, axis, nan, nan, np.ones(T, bool), True)
    out = lean.apply(kin)
    np.testing.assert_array_equal(out.joints[:, list(FIXED)], kin.joints[:, list(FIXED)])
    np.testing.assert_array_equal(out.glob_rot[:, list(FIXED)], kin.glob_rot[:, list(FIXED)])
    np.testing.assert_array_equal(out.contact_points, kin.contact_points)
    R = rotation_about(axis, theta)
    np.testing.assert_allclose(out.glob_rot[:, list(UPPER)],
                               np.einsum('tab,tjbc->tjac', R, kin.glob_rot[:, list(UPPER)]),
                               atol=1e-12)
    np.testing.assert_allclose(out.joints[:, 0], kin.joints[:, 0] + shift, atol=1e-12)
    model = np.diagonal(_evaluate(com_along(kin, d, height), theta))
    np.testing.assert_allclose(body_com(out.joints) @ d, model, atol=1e-9)
    # 上向きは d の向きへ倒れる
    up = out.glob_rot[:, 9] @ [0.0, 1.0, 0.0]
    ref = kin.glob_rot[:, 9] @ [0.0, 1.0, 0.0]
    assert np.all(np.sign(up @ d - ref @ d)[np.abs(theta) > 1e-3] == np.sign(theta[np.abs(theta) > 1e-3]))


def test_com_weights_are_a_mass_average():
    w = com_weights()
    assert abs(w.sum() - 1.0) < 1e-12 and np.all(w >= 0.0)
    com = body_com(SMPL_REST_JOINTS)
    # 直立した成人の重心は骨盤の少し上（骨盤の関節と背骨2の関節の間）
    assert SMPL_REST_JOINTS[0, 1] < com[1] < SMPL_REST_JOINTS[6, 1]
