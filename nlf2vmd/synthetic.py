"""テスト・動作確認用の合成データ（SMPL 公式ファイルや GPU が無くても変換を試せる）。"""
import numpy as np

from . import quat
from .body_model import ANKLES, FEET, SMPL_PARENTS, BodyModel

# SMPL（neutral）の初期姿勢の関節位置のおおよその値 [m]
SMPL_REST_JOINTS = np.array([
    [-0.0018, -0.2233, 0.0282], [0.0695, -0.3141, 0.0239], [-0.0678, -0.3143, 0.0219],
    [-0.0043, -0.1144, 0.0015], [0.1022, -0.6890, 0.0169], [-0.1060, -0.6964, 0.0151],
    [0.0012, 0.0209, 0.0026], [0.0885, -1.0874, -0.0266], [-0.0892, -1.0891, -0.0232],
    [0.0019, 0.0735, 0.0283], [0.1195, -1.1439, 0.0924], [-0.1194, -1.1440, 0.0969],
    [-0.0015, 0.2872, -0.0137], [0.0766, 0.1966, -0.0045], [-0.0746, 0.1950, -0.0093],
    [0.0012, 0.3533, 0.0376], [0.1985, 0.2402, -0.0161], [-0.1917, 0.2410, -0.0179],
    [0.4541, 0.2244, -0.0409], [-0.4540, 0.2217, -0.0443], [0.7196, 0.2350, -0.0470],
    [-0.7174, 0.2355, -0.0471], [0.8045, 0.2279, -0.0615], [-0.8081, 0.2226, -0.0631]])


def synthetic_body_model(num_betas=10):
    """関節のまわりに数点ずつ頂点を置き、足裏にかかと・つま先の頂点を置いた簡易体モデル。

    beta[0] は全身の大きさ（+1 で 5% 大きく）を変える。
    """
    J = SMPL_REST_JOINTS
    verts, owner = [], []
    cube = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    for j in range(len(J)):
        r = 0.01 if j in list(ANKLES) + list(FEET) else 0.03
        verts += list(J[j] + r * cube)
        owner += [j] * len(cube)
    for side in range(2):
        a, f = J[ANKLES[side]], J[FEET[side]]
        sole = a[1] - 0.075
        for dx in (-0.02, 0.02):
            for dz in (0.0, -0.01):
                verts.append([a[0] + dx, sole, a[2] - 0.05 + dz])       # かかと
                owner.append(ANKLES[side])
                verts.append([f[0] + dx, sole, f[2] + 0.06 - dz])       # つま先
                owner.append(FEET[side])
        for z in np.linspace(a[2] - 0.02, f[2] + 0.02, 4):              # 土踏まず
            verts.append([0.5 * (a[0] + f[0]), sole + 0.01, z])
            owner.append(FEET[side])
    verts = np.array(verts)
    weights = np.eye(len(J))[owner]
    shapedirs = np.zeros(verts.shape + (num_betas,))
    shapedirs[..., 0] = 0.05 * verts
    J_shapedirs = np.zeros(J.shape + (num_betas,))
    J_shapedirs[..., 0] = 0.05 * J
    return BodyModel(verts, shapedirs, J, J_shapedirs, weights, SMPL_PARENTS)


def _axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    return quat.from_rotvec(np.asarray(angle, float)[..., None] * axis)


ANKLE_HEIGHT = 0.075   # synthetic_body_model の足首の高さ（足裏から）


def _stance_frames(num_frames, fps, step_sec):
    """(T, 2) 各足が支持脚（接地中）のフレーム。偶数番目の歩は左足、奇数番目は右足が支持脚。"""
    k = np.floor(np.arange(num_frames) / fps / step_sec + 1e-9).astype(int)
    return np.stack([(k - side) % 2 == 0 for side in range(2)], axis=1)


def synthetic_walk(num_frames=180, fps=30.0, speed=1.0, heading_deg=0.0, step_sec=0.6,
                   lift=0.08, noise_deg=0.0, seed=0):
    """heading_deg の向きに一定速度で歩く（体もその向きを向く）合成モーション。Y 上向き座標。

    支持脚の足首は床に固定し（倒立振子のように骨盤が上下する）、遊脚は足首の軌道を
    2 リンクの IK で解く。足は常に床と平行。heading_deg=0 で +Z（SMPL の正面）に進む。
    戻り値は load_motion に渡せる dict（'stance' に正解の接地フラグ (T, 2) を入れる）。
    """
    rng = np.random.default_rng(seed)
    J = SMPL_REST_JOINTS
    t = np.arange(num_frames) / fps
    stance = _stance_frames(num_frames, fps, step_sec)
    k = np.floor(t / step_sec + 1e-9)
    s = t / step_sec - k
    P, v = step_sec, speed
    hip_off = J[[1, 2]] - J[0]
    pelvis_z = v * t

    # 各足の足首の目標（体のローカル座標: z = 前, y = 上）
    foot = np.zeros((num_frames, 2, 2))
    for side in range(2):
        st = stance[:, side]
        foot[st, side, 0] = v * (k[st] + 0.5) * P
        z0, z1 = v * (k[~st] - 0.5) * P, v * (k[~st] + 1.5) * P
        ss = s[~st] ** 2 * (3 - 2 * s[~st])
        foot[~st, side, 0] = z0 + (z1 - z0) * ss
        foot[~st, side, 1] = lift * np.sin(np.pi * s[~st])
    foot[..., 1] += ANKLE_HEIGHT

    # 骨盤の高さ: 支持脚が 97% 伸びた長さで床に届く高さ（倒立振子）
    lengths = [(np.linalg.norm(J[4 + i] - J[1 + i]), np.linalg.norm(J[7 + i] - J[4 + i]))
               for i in range(2)]
    sup = np.where(stance[:, 0], 0, 1)
    d = foot[np.arange(num_frames), sup, 0] - (pelvis_z + hip_off[sup, 2])
    L = 0.97 * np.array([sum(lengths[i]) for i in sup])
    pelvis_y = ANKLE_HEIGHT + np.sqrt(L ** 2 - d ** 2) - hip_off[sup, 1]

    q = np.tile(quat.IDENTITY, (num_frames, 24, 1))
    q[:, 0] = _axis_angle([0, 1, 0], np.full(num_frames, np.deg2rad(heading_deg)))
    for side in range(2):
        hip, knee, ankle = 1 + side, 4 + side, 7 + side
        l1, l2 = lengths[side]
        th_t0 = np.arctan2(*(J[knee] - J[hip])[[2, 1]] * [1, -1])    # 前向きを正とする角度
        th_s0 = np.arctan2(*(J[ankle] - J[knee])[[2, 1]] * [1, -1])
        rz = foot[:, side, 0] - (pelvis_z + hip_off[side, 2])
        ry = foot[:, side, 1] - (pelvis_y + hip_off[side, 1])
        D = np.minimum(np.hypot(rz, ry), 0.999 * (l1 + l2))
        alpha = np.arctan2(rz, -ry)
        beta = np.arccos(np.clip((l1 ** 2 + D ** 2 - l2 ** 2) / (2 * l1 * D), -1, 1))
        phi = np.pi - np.arccos(np.clip((l1 ** 2 + l2 ** 2 - D ** 2) / (2 * l1 * l2), -1, 1))
        th_t = alpha + beta
        a_hip = th_t0 - th_t                          # X 軸まわり（正 = 脚が後ろへ）
        a_knee = (th_s0 - th_t0) - (-phi)             # 膝は後ろへ曲がる
        q[:, hip] = _axis_angle([1, 0, 0], a_hip)
        q[:, knee] = _axis_angle([1, 0, 0], a_knee)
        q[:, ankle] = _axis_angle([1, 0, 0], -(a_hip + a_knee))   # 足裏を床と平行に保つ
    arm = np.deg2rad(15.0) * np.sin(np.pi * t / P)
    q[:, 16] = quat.mul(_axis_angle([0, 0, 1], np.full(num_frames, np.deg2rad(-70.0))),
                        _axis_angle([1, 0, 0], arm))
    q[:, 17] = quat.mul(_axis_angle([0, 0, 1], np.full(num_frames, np.deg2rad(70.0))),
                        _axis_angle([1, 0, 0], -arm))
    if noise_deg > 0:
        noise = rng.normal(0, np.deg2rad(noise_deg), (num_frames, 24, 3))
        q = quat.mul(q, quat.from_rotvec(noise))

    h = np.deg2rad(heading_deg)
    R = np.array([[np.cos(h), 0, np.sin(h)], [0, 1, 0], [-np.sin(h), 0, np.cos(h)]])
    pelvis = np.stack([np.zeros(num_frames), pelvis_y, pelvis_z], axis=1) @ R.T
    return dict(pose=quat.to_rotvec(q), betas=np.zeros(10), trans=pelvis - J[0], fps=fps,
                coord_system='yup', stance=stance)


def to_camera_coords(motion):
    """Y 上向きの合成モーションを、NLF と同じカメラ座標（Y 下向き・Z 奥向き）に直す。"""
    R = np.diag([1.0, -1.0, -1.0])
    out = dict(motion)
    q = quat.from_rotvec(np.asarray(motion['pose'], float))
    q[:, 0] = quat.mul(quat.from_matrix(R), q[:, 0])
    out['pose'] = quat.to_rotvec(q)
    pelvis = motion['trans'] + SMPL_REST_JOINTS[0]
    out['trans'] = pelvis @ R.T - SMPL_REST_JOINTS[0] + np.array([0.0, 0.0, 4.0])
    out['coord_system'] = 'camera'
    return out
