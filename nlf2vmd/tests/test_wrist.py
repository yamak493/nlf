"""手首の向きの補正（ステージ2b）: MediaPipe の 3D の点を ROI の回転と位置で戻して、手首の向きを求めること。"""
import numpy as np
import pytest

from nlf2vmd import convert, load_config, quat
from nlf2vmd.body_model import forward_kinematics
from nlf2vmd.motion_io import CAMERA_TO_YUP
from nlf2vmd.synthetic import SMPL_REST_JOINTS, synthetic_walk
from nlf2vmd.wrist import (WRIST, continuous_rotvec, correct_wrists, palm_frame, rate_limit,
                           rest_palm_frames, ray_rotation, roi_rotation, swing_twist, world_to_camera)

IMAGE = (1280, 720)
FPS = 30.0

# 手のひらの座標系（前・横・法線）での 21 点。palm_frame が単位行列になるように置く
_PALM = np.zeros((21, 3))
_PALM[1:5] = [[0.02, 0.03, 0.01], [0.04, 0.05, 0.015], [0.06, 0.065, 0.02], [0.08, 0.075, 0.02]]
for k, (x0, y0) in enumerate([(0.09, 0.025), (0.095, 0.0), (0.09, -0.018), (0.08, -0.035)]):
    base = 5 + 4 * k
    _PALM[base:base + 4] = [[x0 + d, y0, -0.3 * d] for d in (0.0, 0.04, 0.065, 0.085)]


def _rotvec(axis, deg):
    return np.deg2rad(deg) * np.asarray(axis, float) / np.linalg.norm(axis)


def test_palm_frame_is_identity_for_the_canonical_hand():
    np.testing.assert_allclose(palm_frame(_PALM), np.eye(3), atol=1e-12)


def test_world_to_camera_undoes_roi_rotation_and_ray():
    rng = np.random.default_rng(0)
    cam = rng.normal(size=(5, 21, 3)) * 0.05
    roi = np.stack([rng.uniform(0, IMAGE[0], 5), rng.uniform(0, IMAGE[1], 5), np.full(5, 100.0),
                    rng.uniform(-np.pi, np.pi, 5)], axis=1)
    R = ray_rotation(roi[:, :2], IMAGE) @ roi_rotation(roi[:, 3])
    world = np.einsum('nba,nkb->nka', R, cam)                  # 切り出した画像の座標系へ（逆向き）
    np.testing.assert_allclose(world_to_camera(world, roi, IMAGE), cam, atol=1e-12)
    # 画像の中心の ROI では、視線の補正は無い
    np.testing.assert_allclose(ray_rotation([IMAGE[0] / 2, IMAGE[1] / 2], IMAGE), np.eye(3),
                               atol=1e-12)
    # 回転の向きは hand_detect.to_image と同じ（ROI の上（-y）は、角度 θ で画像の (sin θ, -cos θ) へ）
    th = 0.4
    np.testing.assert_allclose(roi_rotation(th) @ [0.0, -1.0, 0.0], [np.sin(th), -np.cos(th), 0.0],
                               atol=1e-12)


def test_swing_twist():
    axis = np.array([1.0, 0.0, 0.0])
    q = quat.mul(quat.from_rotvec(_rotvec([0, 0, 1], 30)), quat.from_rotvec(_rotvec(axis, 70)))
    swing, twist = swing_twist(q, axis)
    assert abs(np.rad2deg(swing) - 30) < 1e-6 and abs(np.rad2deg(twist) - 70) < 1e-6


def _fk(body_model, T):
    rest = body_model.rest_joints(np.zeros(10))

    def fk(q):
        return forward_kinematics(q, np.zeros((T, 3)), rest, body_model.parents)[0]
    return fk, rest


def _analysis(G_true, presence, roi_angle=None, bad_2d=False, seed=0):
    """大域回転（Y 上向き）(T, 2, 3, 3) の手首を持つ手の、MediaPipe の検出結果（切り出した画像の座標系の 3D の点）。"""
    rng = np.random.default_rng(seed)
    T = len(G_true)
    F_rest = rest_palm_frames(SMPL_REST_JOINTS)
    world = np.zeros((T, 2, 21, 3))
    screen = np.zeros((T, 2, 21, 3))
    roi = np.zeros((T, 2, 4))
    for t in range(T):
        for s in range(2):
            center = np.array([400.0 + 500.0 * s, 300.0])
            angle = rng.uniform(-np.pi, np.pi) if roi_angle is None else roi_angle
            roi[t, s] = [*center, 120.0, angle]
            G_cam = CAMERA_TO_YUP @ G_true[t, s]
            cam = (G_cam @ F_rest[s] @ _PALM.T).T
            R = ray_rotation(center, IMAGE) @ roi_rotation(angle)
            world[t, s] = cam @ R                                  # R^T cam
            screen[t, s, :, :2] = center + 1000.0 * cam[:, :2]
            if bad_2d:
                screen[t, s, :, :2] = center + 1000.0 * cam[:, :2] @ np.array([[0, -1], [1, 0]])
    return dict(world=world, screen=screen, presence=np.asarray(presence, float), roi=roi,
                image_size=np.array(IMAGE), fps=FPS)


def _true_and_nlf(T):
    """正しい手首の回転と、NLF が手首をひねり違えた（前腕まわりに 150 度）回転。"""
    motion = synthetic_walk(num_frames=T)
    q = quat.from_rotvec(np.asarray(motion['pose'], float))
    true = q.copy()
    J = SMPL_REST_JOINTS
    for s in range(2):
        axis = J[22 + s] - J[20 + s]
        true[:, WRIST[s]] = quat.mul(quat.from_rotvec(_rotvec([0, 1, 0], 25)),
                                     quat.from_rotvec(_rotvec(axis, 40)))
    nlf = true.copy()
    for s in range(2):
        axis = J[22 + s] - J[20 + s]
        nlf[:, WRIST[s]] = quat.mul(true[:, WRIST[s]], quat.from_rotvec(_rotvec(axis, 150)))
    return motion, true, nlf


def _angles(fk, q, G_ref):
    G = fk(q)
    return np.rad2deg(np.stack([quat.angle_between(quat.from_matrix(G[:, WRIST[s]]),
                                                   quat.from_matrix(G_ref[:, s]))
                                for s in range(2)], axis=1))


def test_correct_wrists_recovers_the_palm_orientation(body_model):
    T = 60
    _, true, nlf = _true_and_nlf(T)
    fk, rest = _fk(body_model, T)
    G_true = fk(true)[:, list(WRIST)]
    presence = np.zeros((T, 2))
    presence[10:50] = 0.95
    cfg = load_config().wrist
    fixed, res = correct_wrists(nlf, fk, SMPL_REST_JOINTS, _analysis(G_true, presence), FPS, cfg)
    err = _angles(fk, fixed, G_true)
    assert (_angles(fk, nlf, G_true)[15:45] > 140).all()      # NLF は 150 度ずれている
    assert (err[20:40] < 2.0).all()                           # 見えている区間は正しい向きに戻る
    assert (np.diff(err[10:20, 0]) < 1e-9).all()              # 見え始めは blend_sec かけて寄せる
    # 見えない区間は、見える区間から離れると NLF の向きのまま（blend_sec の 2 倍の 6 フレームに加えて、150 度の
    # 補正を max_speed_deg_per_s（1 フレームに 12 度）で入れる分、前後に 2 フレームずつ広がる）
    np.testing.assert_allclose(fixed[:2], nlf[:2])
    np.testing.assert_allclose(fixed[-2:], nlf[-2:])
    for k in range(24):                                       # 変えるのは手首の関節だけ
        if k not in WRIST:
            np.testing.assert_allclose(fixed[:, k], nlf[:, k])
    assert res.info['left']['observed_frames'] == 40
    assert res.info['left']['disagreement_deg_median'] > 140
    steps = np.rad2deg(quat.angle_between(fixed[1:, WRIST[0]], fixed[:-1, WRIST[0]]))
    assert steps.max() < 25.0                                 # 見える・見えないの境目でも跳ねない


def test_continuous_rotvec_keeps_the_direction_across_180_degrees():
    """178 度と 182 度（= 逆向きの 178 度）を行き来する回転を、同じ向きの回転ベクトルのままつなぐ。"""
    axis = np.array([1.0, 0.0, 0.0])
    deg = np.array([170.0, 178.0, 182.0, 179.0, 185.0, 176.0])
    r = quat.to_rotvec(quat.from_rotvec(np.deg2rad(deg)[:, None] * axis))   # 180 度を超えると向きが反転する
    assert (r @ axis < 0).any()
    np.testing.assert_allclose(continuous_rotvec(r) @ axis, np.deg2rad(deg), atol=1e-9)
    same = quat.angle_between(quat.from_rotvec(continuous_rotvec(r)), quat.from_rotvec(r))
    assert same.max() < 1e-9                                   # 回転そのものは変えない


def test_rate_limit_is_symmetric_and_bounded():
    x = np.zeros((40, 3))
    x[15:25, 0] = 1.0
    y = rate_limit(x, 0.1)
    assert np.linalg.norm(np.diff(y, axis=0), axis=-1).max() <= 0.1 + 1e-12
    np.testing.assert_allclose(y[:20, 0], y[39:19:-1, 0], atol=1e-12)   # 前後対称（遅れない）


@pytest.mark.parametrize('presence_value', [0.6, 0.95])
def test_half_turn_disagreement_does_not_spin_the_wrist(body_model, presence_value):
    """NLF が手首を前腕まわりに 180 度前後ひねり違えていても、1 フレームで大きく回らない。

    球面線形補間は近いほうへ回るので、差が 180 度をまたぐたびに回す向きが入れ替わる（重みが 1 未満の所で
    手首が 1 フレームに 90 度以上跳ね、捩りボーンに分けると一瞬で 1 回転して見えた）。
    """
    T = 90
    rng = np.random.default_rng(0)
    _, true, _ = _true_and_nlf(T)
    fk, _ = _fk(body_model, T)
    nlf = true.copy()
    for s in range(2):
        axis = (SMPL_REST_JOINTS[22 + s] - SMPL_REST_JOINTS[20 + s])
        off = np.deg2rad(180.0 + rng.normal(0.0, 4.0, T))
        nlf[:, WRIST[s]] = quat.mul(true[:, WRIST[s]],
                                    quat.from_rotvec(off[:, None] * axis / np.linalg.norm(axis)))
    presence = np.full((T, 2), presence_value)
    a = _analysis(fk(true)[:, list(WRIST)], presence)
    fixed, _ = correct_wrists(nlf, fk, SMPL_REST_JOINTS, a, FPS, load_config().wrist)
    for s in range(2):
        q = fixed[:, WRIST[s]]
        assert np.rad2deg(quat.angle_between(q[1:], q[:-1])).max() < 20.0


def test_short_dropouts_keep_the_correction(body_model):
    """手が 0.3 秒見えなくなっても、NLF の（180 度近く違う）向きへ戻って往復しない。"""
    T = 90
    _, true, nlf = _true_and_nlf(T)
    fk, _ = _fk(body_model, T)
    G_true = fk(true)[:, list(WRIST)]
    presence = np.full((T, 2), 0.95)
    presence[40:49] = 0.0
    cfg = load_config().wrist
    fixed, _ = correct_wrists(nlf, fk, SMPL_REST_JOINTS, _analysis(G_true, presence), FPS, cfg)
    assert _angles(fk, fixed, G_true)[40:49].max() < 5.0       # 見えない間も正しい向きのまま
    steps = np.rad2deg(quat.angle_between(fixed[1:, WRIST[0]], fixed[:-1, WRIST[0]]))
    assert steps.max() < cfg.max_speed_deg_per_s / FPS + 5.0
    # つながない設定では NLF の向きへ戻る（補正の速さの上限で、ゆっくり）
    off = load_config(overrides=['wrist.hold_sec=0']).wrist
    fixed, _ = correct_wrists(nlf, fk, SMPL_REST_JOINTS, _analysis(G_true, presence), FPS, off)
    assert _angles(fk, fixed, G_true)[44].min() > 45.0
    steps = np.rad2deg(quat.angle_between(fixed[1:, WRIST[0]], fixed[:-1, WRIST[0]]))
    assert steps.max() < off.max_speed_deg_per_s / FPS + 5.0


@pytest.mark.parametrize('case', ['low_presence', 'inconsistent_2d', 'disagree_gate'])
def test_unreliable_detections_are_not_used(body_model, case):
    T = 40
    _, true, nlf = _true_and_nlf(T)
    fk, _ = _fk(body_model, T)
    G_true = fk(true)[:, list(WRIST)]
    presence = np.full((T, 2), 0.4 if case == 'low_presence' else 0.95)
    over = ['wrist.max_disagree_deg=90'] if case == 'disagree_gate' else []
    cfg = load_config(overrides=over).wrist
    a = _analysis(G_true, presence, bad_2d=case == 'inconsistent_2d')
    fixed, res = correct_wrists(nlf, fk, SMPL_REST_JOINTS, a, FPS, cfg)
    np.testing.assert_allclose(fixed, nlf)
    assert res.correction_deg.max() == 0.0
    key = {'low_presence': 'low_presence', 'inconsistent_2d': 'inconsistent_2d',
           'disagree_gate': 'disagree'}[case]
    assert res.info['rejected'][key] > 0


def test_flipped_palm_targets_are_rejected_by_the_wrist_limits(body_model):
    """MediaPipe が手のひらの表裏を取り違えた（正しい向きから前腕まわりに 180 度）向きは、前腕のひねりが可動域から
    大きく外れるので使わない。手首だけのひねり（max_twist_deg）の確認では通ってしまう（ひじのひねりと合わせて測る）。"""
    T = 30
    _, true, _ = _true_and_nlf(T)
    fk, _ = _fk(body_model, T)
    flipped = true.copy()
    for s in range(2):
        axis = SMPL_REST_JOINTS[22 + s] - SMPL_REST_JOINTS[20 + s]
        flipped[:, WRIST[s]] = quat.mul(true[:, WRIST[s]], quat.from_rotvec(_rotvec(axis, 180)))
    a = _analysis(fk(flipped)[:, list(WRIST)], np.full((T, 2), 0.95))
    cfg = load_config()
    fixed, res = correct_wrists(true, fk, SMPL_REST_JOINTS, a, FPS, cfg.wrist, cfg.wrist_limits)
    np.testing.assert_allclose(fixed, true)
    assert res.info['rejected']['anatomy'] == 2 * T
    _, res = correct_wrists(true, fk, SMPL_REST_JOINTS, a, FPS, cfg.wrist)
    assert res.info['rejected']['anatomy'] == 0


def test_anatomically_impossible_targets_are_not_used(body_model):
    """前腕に対して手首が 150 度曲がる向きは、推定の誤りとみなして使わない。"""
    T = 30
    _, true, nlf = _true_and_nlf(T)
    bent = true.copy()
    for s in range(2):
        bent[:, WRIST[s]] = quat.from_rotvec(_rotvec([0, 0, 1], 150))
    fk, _ = _fk(body_model, T)
    a = _analysis(fk(bent)[:, list(WRIST)], np.full((T, 2), 0.95))
    fixed, res = correct_wrists(nlf, fk, SMPL_REST_JOINTS, a, FPS, load_config().wrist)
    np.testing.assert_allclose(fixed, nlf)
    assert res.info['rejected']['anatomy'] == 2 * T


def test_convert_with_hands_analysis(body_model):
    T = 60
    motion, true, nlf = _true_and_nlf(T)
    fk, _ = _fk(body_model, T)
    a = _analysis(fk(true)[:, list(WRIST)], np.full((T, 2), 0.95))
    cfg = load_config(overrides=['diagnostics.enabled=false'])
    src = dict(motion, pose=quat.to_rotvec(nlf))
    base = convert(src, None, body_model=body_model, config=cfg, log=None)
    r = convert(src, None, body_model=body_model, config=cfg, log=None, hands_analysis=a)
    assert r.info['wrist']['left']['observed_frames'] == T
    moved = np.rad2deg(quat.angle_between(r.local_quats['左手首'], base.local_quats['左手首']))
    assert np.median(moved) > 100                             # 150 度のひねり違いを直す
    for name in ('左ひじ', '右腕', '上半身'):
        # 手の向きで全身の重心がわずかに変わり、前後の傾きの補正（ステージ6c）がごくわずかに変わる
        np.testing.assert_allclose(r.local_quats[name], base.local_quats[name], atol=1e-5)
    # ROI が無い検出結果（古い版）は使わない
    old = {k: v for k, v in a.items() if k != 'roi'}
    r = convert(src, None, body_model=body_model, config=cfg, log=None, hands_analysis=old)
    assert r.wrist is None and any('ROI' in w for w in r.warnings)


def test_wrist_plot(tmp_path, body_model):
    pytest.importorskip('matplotlib')
    T = 30
    motion, true, nlf = _true_and_nlf(T)
    fk, _ = _fk(body_model, T)
    a = _analysis(fk(true)[:, list(WRIST)], np.full((T, 2), 0.95))
    r = convert(dict(motion, pose=quat.to_rotvec(nlf)), None, body_model=body_model,
                diag_dir=tmp_path, log=None, hands_analysis=a)
    assert (tmp_path / 'wrist.png').exists() and 'wrist' in r.plot_paths
