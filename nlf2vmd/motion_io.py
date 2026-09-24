"""ステージ1: 読み込み・正規化。

内部共通形式（Motion）:
  quats    (T, 24, 4)  親に対する関節回転（x, y, z, w）。0 番はルートの向き（global orient）
  betas    (10,)       体型。フレーム毎にある場合は中央値で 1 つに固定する
  root_pos (T, 3)      骨盤の位置 [m]（SMPL の trans ではなく、骨盤関節そのものの位置）
  fps
座標系は Y 上向き・右手系（SMPL の初期姿勢と同じく +Z が正面、+X が体の左）。
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import quat

# カメラ座標（x=右 / y=下 / z=奥）→ Y 上向き: X 軸まわりに 180 度
CAMERA_TO_YUP = np.diag([1.0, -1.0, -1.0])


@dataclass
class Motion:
    quats: np.ndarray
    betas: np.ndarray
    root_pos: np.ndarray
    fps: float
    source_fps: float
    valid: np.ndarray
    fk_check_mm: float = float('nan')   # 入力の関節位置と、体モデルで作り直した関節位置の最大差

    @property
    def num_frames(self):
        return len(self.quats)


def _as_dict(source):
    if isinstance(source, (str, Path)):
        with np.load(source, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    return dict(source)


def _select_person(d, person_index):
    # pose が (P, T, 24, 3) のように人物の次元を持つ場合は、指定した 1 人だけを取り出す
    pose = np.asarray(d['pose'])
    has_person_axis = (pose.ndim == 4 and pose.shape[-2:] in [(24, 3), (24, 4)]) or pose.ndim == 5
    if not has_person_axis:
        return d
    n = pose.shape[0]
    if not 0 <= person_index < n:
        raise IndexError(f'person_index={person_index} ですが、人物は {n} 人です')
    out = dict(d)
    for key in ('pose', 'betas', 'trans', 'joints3d', 'valid'):
        if key in d and np.ndim(d[key]) > 0 and np.shape(d[key])[0] == n:
            out[key] = np.asarray(d[key])[person_index]
    return out


def _pose_to_quats(pose):
    pose = np.asarray(pose, np.float64)
    T = pose.shape[0]
    if pose.shape[-2:] == (3, 3):
        return quat.from_matrix(pose.reshape(T, -1, 3, 3))
    if pose.shape[-1] == 4 and pose.ndim == 3:
        return quat.normalize(pose)
    return quat.from_rotvec(pose.reshape(T, -1, 3))


def resample(quats, root_pos, valid, fps, target_fps):
    """回転は球面線形補間、移動量は線形補間で target_fps に揃える。"""
    if target_fps <= 0 or abs(fps - target_fps) < 1e-6 or len(quats) < 2:
        return quats, root_pos, valid
    T = len(quats)
    t_src = np.arange(T) / fps
    t_dst = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / target_fps)
    w = t_dst * fps
    lo = np.minimum(np.floor(w).astype(int), T - 1)
    hi = np.minimum(lo + 1, T - 1)
    frac = w - lo
    q = quat.slerp(quats[lo], quats[hi], np.broadcast_to(frac[:, None], quats[lo].shape[:-1]))
    pos = np.stack([np.interp(t_dst, t_src, root_pos[:, k]) for k in range(3)], axis=1)
    v = valid[np.clip(np.round(w).astype(int), 0, T - 1)]
    return q, pos, v


def load_motion(source, cfg_input, body_model):
    """npz のパス、または同じキーを持つ dict から Motion を作る。

    必要なキー: pose（(T,24,3) 回転ベクトル / (T,24,4) / (T,24,3,3)）, betas, trans, fps
    任意: valid, joints3d（[mm]、体モデルの自己検証に使う）, coord_system（'camera' / 'yup'）
    """
    d = _select_person(_as_dict(source), int(cfg_input.person_index))
    quats = _pose_to_quats(d['pose'])
    T = len(quats)
    fps = float(np.asarray(d.get('fps', 30.0)))
    trans = np.asarray(d['trans'], np.float64).reshape(T, 3)
    valid = np.asarray(d['valid'], bool) if 'valid' in d else np.ones(T, bool)

    # 体型: フレーム毎なら（検出できたフレームの）中央値 1 つに固定する
    betas = np.asarray(d['betas'], np.float64)
    betas_per_frame = betas if betas.ndim == 2 else None
    if betas.ndim == 2:
        use = valid if valid.any() else np.ones(T, bool)
        betas = np.median(betas[use], axis=0)

    # 骨盤の位置 = 初期姿勢の骨盤 + trans（フレーム毎の体型があればそれで求める）
    if betas_per_frame is not None:
        pelvis = body_model.rest_joints(betas_per_frame)[:, 0] + trans
    else:
        pelvis = body_model.rest_joints(betas)[0] + trans

    # 入力の関節位置があれば、体モデルと FK が入力と一致するかを確かめる
    fk_check = float('nan')
    if 'joints3d' in d:
        from .body_model import forward_kinematics
        j_in = np.asarray(d['joints3d'], np.float64).reshape(T, -1, 3)[:, :24] / 1000.0
        rest = body_model.rest_joints(betas)
        _, P = forward_kinematics(quats, pelvis, rest, body_model.parents)
        fk_check = float(np.abs(P - j_in)[valid].max() * 1000.0) if valid.any() else float('nan')

    coords = str(cfg_input.coords)
    if coords == 'auto':
        coords = str(np.asarray(d['coord_system'])) if 'coord_system' in d else 'camera'
    if coords == 'camera':
        # ルートの向きと位置の両方に同じ回転を掛ける
        pelvis = pelvis @ CAMERA_TO_YUP.T
        q_conv = quat.from_matrix(CAMERA_TO_YUP)
        quats[:, 0] = quat.mul(q_conv, quats[:, 0])
    elif coords != 'yup':
        raise ValueError(f'input.coords は auto / camera / yup のいずれかです: {coords}')

    quats = quat.make_continuous(quats)
    quats, pelvis, valid = resample(quats, pelvis, valid, fps, float(cfg_input.target_fps))
    out_fps = float(cfg_input.target_fps) if cfg_input.target_fps > 0 else fps
    return Motion(quat.make_continuous(quats), betas, pelvis, out_fps, fps, valid, fk_check)
