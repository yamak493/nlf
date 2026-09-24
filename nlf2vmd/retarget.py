"""ステージ9: 上半身の回転リターゲット（初期姿勢補正方式）。

SMPL は T ポーズ、MMD モデルは多くが A ポーズなので、回転をそのままコピーせず、
ボーンごとの補正回転 C を使って
    R_mmd_global = R_smpl_global · C
    R_mmd_local  = (R_mmd_global,parent)^-1 · R_mmd_global
とする。C は「PMX の初期姿勢のボーン方向を SMPL の初期姿勢の方向に合わせる回転」。
計算はすべて内部座標（右手系・Y 上向き・+Z 正面）で行い、VMD 書き出し時に MMD 座標へ直す。
"""
import numpy as np

from . import quat
from .skeleton import SIDES

X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])


class Retargeter:
    def __init__(self, skeleton, smpl_rest_joints):
        self.skel = skeleton
        J = np.asarray(smpl_rest_joints, np.float64)
        S = skeleton.internal
        self.has_upper2 = skeleton.has('上半身2')
        self.warnings = []

        # (MMD ボーン, 大域回転を使う SMPL 関節, 補正回転 C)
        specs = []

        def torso(name, joint, up_pmx, lat_pmx, up_smpl, lat_smpl):
            B_pmx = quat.frame_from_up_lateral(up_pmx, lat_pmx)
            B_smpl = quat.frame_from_up_lateral(up_smpl, lat_smpl)
            specs.append((name, joint, B_smpl @ B_pmx.T))

        def limb(name, joint, dir_pmx, dir_smpl):
            specs.append((name, joint, quat.to_matrix(quat.from_two_vectors(dir_pmx, dir_smpl))))

        hips_pmx = 0.5 * (S('左足') + S('右足'))
        lat_hips_pmx = S('左足') - S('右足')
        lat_arms_pmx = S('左腕') - S('右腕')
        lat_hips_smpl, lat_arms_smpl = J[1] - J[2], J[16] - J[17]
        torso('下半身', 0, S('下半身') - hips_pmx, lat_hips_pmx,
              J[0] - 0.5 * (J[1] + J[2]), lat_hips_smpl)
        if self.has_upper2:
            torso('上半身', 6, S('上半身2') - S('上半身'), lat_arms_pmx, J[9] - J[3], lat_arms_smpl)
            torso('上半身2', 9, S('首') - S('上半身2'), lat_arms_pmx, J[12] - J[9], lat_arms_smpl)
        else:
            # 上半身2 が無いモデルは、背骨 3 本分の回転を上半身に合成する
            torso('上半身', 9, S('首') - S('上半身'), lat_arms_pmx, J[12] - J[3], lat_arms_smpl)
        torso('首', 12, S('頭') - S('首'), X_AXIS, J[15] - J[12], X_AXIS)
        specs.append(('頭', 15, np.eye(3)))
        for side, s in enumerate(SIDES):
            collar, shoulder, elbow, wrist, hand = (13, 16, 18, 20, 22) if side == 0 else \
                (14, 17, 19, 21, 23)
            limb(s + '肩', collar, S(s + '腕') - S(s + '肩'), J[shoulder] - J[collar])
            limb(s + '腕', shoulder, S(s + 'ひじ') - S(s + '腕'), J[elbow] - J[shoulder])
            limb(s + 'ひじ', elbow, S(s + '手首') - S(s + 'ひじ'), J[wrist] - J[elbow])
            tail = skeleton.tail_internal(s + '手首')
            if tail is None or np.linalg.norm(tail - S(s + '手首')) < 1e-6:
                tail = S(s + '手首') + (S(s + '手首') - S(s + 'ひじ'))   # 前腕の延長で代用
            limb(s + '手首', wrist, tail - S(s + '手首'), J[hand] - J[wrist])

        self.specs = [sp for sp in specs if skeleton.has(sp[0])]
        self.bones = [sp[0] for sp in self.specs]
        names = set(self.bones)
        self.keyed_parent = {n: skeleton.nearest_ancestor(n, names) for n in self.bones}
        self.correction = {sp[0]: sp[2] for sp in self.specs}

        # 足ＩＫ: 足首の大域回転 × 補正回転（足首→つま先の向きを合わせる）
        self.foot_correction = []
        for side, s in enumerate(SIDES):
            ankle, foot = (7, 10) if side == 0 else (8, 11)
            toe = next((S(s + n) for n in ('つま先', 'つま先ＩＫ') if skeleton.has(s + n)), None)
            base = S(s + '足首') if skeleton.has(s + '足首') else S(s + '足ＩＫ')
            if toe is None or np.linalg.norm(toe - base) < 1e-6:
                self.warnings.append(f'{s}つま先が見つからないため、足先の向きを標準値で代用します')
                toe = base + np.array([0.0, -1.2, 1.6])
            self.foot_correction.append(
                quat.to_matrix(quat.from_two_vectors(toe - base, J[foot] - J[ankle])))
        self.foot_correction = np.stack(self.foot_correction)

    def global_matrices(self, glob_rot):
        """ボーン名 → MMD ボーンの大域回転 (T, 3, 3)。glob_rot は SMPL の大域回転 (T, J, 3, 3)。"""
        return {name: glob_rot[:, j] @ C for name, j, C in self.specs}

    def global_matrix(self, name, glob_rot):
        for n, j, C in self.specs:
            if n == name:
                return glob_rot[:, j] @ C
        raise KeyError(name)

    def local_quats(self, glob_rot):
        """ボーン名 → 親（キーを打つボーンのうち最も近い祖先）に対する回転 (T, 4)。内部座標。"""
        g = self.global_matrices(glob_rot)
        out = {}
        for name in self.bones:
            parent = self.keyed_parent[name]
            local = g[name] if parent is None else np.swapaxes(g[parent], -1, -2) @ g[name]
            out[name] = quat.make_continuous(quat.from_matrix(local))
        return out

    def foot_ik_quats(self, glob_rot):
        """足ＩＫの大域回転 (T, 2, 4)。足ＩＫの親（全ての親）は回転しないので大域 = ローカル。"""
        q = [quat.from_matrix(glob_rot[:, 7 + side] @ self.foot_correction[side])
             for side in range(2)]
        return quat.make_continuous(np.stack(q, axis=1))
