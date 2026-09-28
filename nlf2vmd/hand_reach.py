"""ステージ9h: 胴に対する手の位置を保つリターゲット（体の比率の違いの補正）。

ステージ9 は腕の大域回転をコピーするだけなので、肩幅・腕の長さ・胴の厚みの比が SMPL と MMD モデルで違うと、
胴に対する手の位置が変わる（SMPL で「腰に手を当てる」「胸の前で手を合わせる」動きが、MMD では手が胴に入り込むか、
届かない）。そこで、手首が胴の近くにあるフレームで

1. **手首の目標**: SMPL の手首の位置を胸（背骨3。MMD では 上半身2、無ければ 上半身）の座標系で表し、胴の寸法の比で
   MMD の胴へ移す（軸ごとの一次変換）
   * 左右: 肩幅（左右の肩の関節 / 腕ボーンの間）の比。左右の肩の中点をそろえる
   * 上下: 股関節から肩までの高さの比。股関節の高さをそろえる
   * 前後: 胸の厚みの比。厚みの中心をそろえる。SMPL は体モデルの背骨（関節 3・6・9）の頂点、MMD は胴の当たり判定の
     形（ステージ9b と同じ PMX の剛体 → メッシュ → 標準の形。物理の剛体（胸など）と腕・首から先は除く）から測る。
     頂点・形が無ければ標準の比（厚みの半分 = 0.24 × 肩幅。中心は背骨の線）
2. **2 ボーン IK**: その位置へ手首を置く。MMD の標準のモデルには腕の IK が無いので、ここで解いて 腕・ひじ・手首 の
   回転のキーにする（どのモデルでも同じに再生される）
   * ひじ: 肩 → 手首の距離が目標と同じになるように、上腕と前腕を含む面の中で曲げを変える
   * 肩: 肩 → 手首の向きを目標へ向ける最小の回転（ひじの向き = 腕を含む面の向きは、ほぼ SMPL のまま）
   * 手首: 大域回転（手の向き。ステージ2b で MediaPipe に合わせたもの）は変えない
   * どちらの補正も max_deg まで（届かない分は残す。残った入り込みはステージ9b が離す）
3. **混ぜる**: 手首が胴から遠い（胴の箱からの距離 near_m → far_m で重み 1 → 0）・ひじがほぼまっすぐ（SMPL のひじの
   曲げ bend_deg で重み 0 → 1）フレームは、今までの回転のコピーのまま。その間はなめらかに混ぜる。下ろした腕・広げた
   腕は胴との位置ではなく腕の向きを保つ（まっすぐ下ろした腕を、腰の高さに手首を合わせようとして曲げない）

胴の座標系は胸の 1 つだけを使う（体を前に倒しても胸と一緒に動く。腰に当てた手は、胸と腰の向きの差の分だけずれる）。
計算は全フレームをまとめて行い、反復もしないので、処理時間はほとんど増えない。
"""
from dataclasses import dataclass

import numpy as np

from . import filters, quat
from .arm_collision import bone_positions
from .contacts import _arm_bone, _fallback_capsules, body_capsules, globals_from_local
from .jitter import _turn_towards
from .skeleton import SIDES

ARM_JOINTS = ((13, 16, 18, 20), (14, 17, 19, 21))   # SMPL の 左・右の [鎖骨, 肩, ひじ, 手首]
CHEST_JOINT = 9                                     # 背骨3
HIP_JOINTS = (1, 2)
SPINE_JOINTS = (3, 6, 9)                            # 胸の厚みを測る頂点の持ち主（背骨1〜3）
SPINE_LINE = (3, 12)                                # 頂点が無いときの厚みの中心（背骨1 → 首 の線）
MIN_VERTICES = 30
DEPTH_PERCENTILE = 95.0                             # 胸の厚み: 頂点の前後の位置のこのパーセンタイル（と 100 − これ）
STANDARD_HALF_DEPTH = 0.24                          # 標準の形の胸の厚みの半分 ÷ 肩幅（contacts._fallback_capsules と同じ）
_EPS = 1e-9


@dataclass
class TorsoBox:
    """胴の寸法（胸の座標系。MMD の初期姿勢の向き・MMD 単位。原点は胸のボーン / 関節）。"""
    center_x: float     # 左右の肩の中点
    hip_y: float        # 股関節の高さ
    height: float       # 股関節 → 肩
    width: float        # 左右の肩の間
    center_z: float     # 胸の厚みの中心
    depth: float        # 胸の厚み
    depth_source: str   # 厚みを測ったもの: mesh / rigid / config（標準の比）

    def map_to(self, other, u):
        """(..., 3) この胴に対する位置を、胴 other の同じ所へ移す（軸ごとの一次変換）。"""
        u = np.asarray(u, np.float64)
        out = np.empty_like(u)
        out[..., 0] = other.center_x + (u[..., 0] - self.center_x) * (other.width / self.width)
        out[..., 1] = other.hip_y + (u[..., 1] - self.hip_y) * (other.height / self.height)
        out[..., 2] = other.center_z + (u[..., 2] - self.center_z) * (other.depth / self.depth)
        return out

    def distance(self, u):
        """(...,) 胴の箱（左右は肩の間・上下は股関節から肩・前後は胸の厚み）の外への距離（中なら 0）。"""
        lo = np.array([self.center_x - 0.5 * self.width, self.hip_y, self.center_z - 0.5 * self.depth])
        hi = np.array([self.center_x + 0.5 * self.width, self.hip_y + self.height,
                       self.center_z + 0.5 * self.depth])
        u = np.asarray(u, np.float64)
        return np.linalg.norm(np.maximum(lo - u, 0.0) + np.maximum(u - hi, 0.0), axis=-1)

    def as_dict(self, unit):
        cm = 100.0 / unit
        return dict(width_cm=round(self.width * cm, 2), height_cm=round(self.height * cm, 2),
                    depth_cm=round(self.depth * cm, 2), depth_source=self.depth_source)


def _box(shoulders, hips, depth_range, depth_source):
    width = max(abs(float(shoulders[0, 0] - shoulders[1, 0])), _EPS)
    hip_y = float(hips[:, 1].mean())
    height = max(float(shoulders[:, 1].mean()) - hip_y, _EPS)
    back, front = depth_range
    return TorsoBox(float(shoulders[:, 0].mean()), hip_y, height, width, 0.5 * (back + front),
                    max(front - back, _EPS), depth_source)


def smpl_torso(rest_joints, rest_vertices, vertex_weights, correction, unit):
    """SMPL の胴の寸法。correction: MMD の胸のボーンの補正回転 C（MMD の初期姿勢の向きへ直すのに使う）。"""
    J = np.asarray(rest_joints, np.float64)

    def to_u(p):   # C^T (p − 背骨3)。行ベクトルなので右から C を掛ける
        return (np.asarray(p, np.float64) - J[CHEST_JOINT]) @ correction * unit

    shoulders = to_u(J[[ARM_JOINTS[0][1], ARM_JOINTS[1][1]]])
    hips = to_u(J[list(HIP_JOINTS)])
    source = 'config'
    depth_range = None
    if rest_vertices is not None and vertex_weights is not None:
        owner = np.argmax(np.asarray(vertex_weights), axis=1)
        pts = np.asarray(rest_vertices, np.float64)[np.isin(owner, SPINE_JOINTS)]
        if len(pts) >= MIN_VERTICES:
            z = to_u(pts)[:, 2]
            depth_range = (float(np.percentile(z, 100.0 - DEPTH_PERCENTILE)),
                           float(np.percentile(z, DEPTH_PERCENTILE)))
            source = 'mesh'
    if depth_range is None:
        width = abs(float(shoulders[0, 0] - shoulders[1, 0]))
        cz = float(to_u(J[list(SPINE_LINE)])[:, 2].mean())
        depth_range = (cz - STANDARD_HALF_DEPTH * width, cz + STANDARD_HALF_DEPTH * width)
    return _box(shoulders, hips, depth_range, source)


def _torso_bone(skel, name):
    return (skel.is_descendant(name, '上半身') and not _arm_bone(skel, name)
            and not skel.is_descendant(name, '首'))


def mmd_torso(skel, chest, contacts_cfg, unit):
    """MMD モデルの胴の寸法。胸の厚みは、ステージ9b の体の当たり判定の形のうち胴（上半身から先。腕・首から先と
    物理の剛体を除く）のカプセルの前後の広がり。"""
    base = skel.internal(chest)

    def S(name):
        return skel.internal(name) - base

    shoulders = np.stack([S('左腕'), S('右腕')])
    hips = np.stack([S('左足'), S('右足')])
    caps, source = body_capsules(skel, contacts_cfg, unit)
    torso = [c for c in caps if c.weight >= 1.0 and _torso_bone(skel, c.bone)]
    if not torso:
        caps, source = _fallback_capsules(skel, unit, contacts_cfg), 'config'
        torso = [c for c in caps if _torso_bone(skel, c.bone)]
    front = max(max(c.a[2], c.b[2]) + c.radius for c in torso) - base[2]
    back = min(min(c.a[2], c.b[2]) - c.radius for c in torso) - base[2]
    return _box(shoulders, hips, (float(back), float(front)), source)


@dataclass
class HandReachResult:
    enabled: bool
    weight: np.ndarray          # (T, 2 [左, 右]) 手首を目標へ寄せた割合（0 = 回転のコピーのまま）
    shift: np.ndarray           # (T, 2) 手首の目標と、回転のコピーの手首の距離 [MMD 単位]
    residual: np.ndarray        # (T, 2) 補正後の手首と目標の距離 [MMD 単位]（max_deg で届かなかった分）
    correction_deg: np.ndarray  # (T, 2, 2 [肩, ひじ]) 補正の回転角 [度]
    smpl: TorsoBox = None
    mmd: TorsoBox = None
    info: dict = None


def _angle_between(a, b):
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    return np.arctan2(cross, np.sum(a * b, -1))


def reach_wrists(pos, glob, targets, max_deg):
    """2 ボーン IK で、手首を targets へ寄せる 腕・ひじ の大域回転（手首の大域回転は変えない）。

    pos: {'S', 'E', 'W'} 肩（腕）・ひじ・手首の位置 (T, 3) / glob: {'arm', 'elbow'} 腕・ひじの大域回転 (T, 3, 3) /
    targets: (T, 3) 手首の目標。戻り値は (腕の大域回転, ひじの大域回転, 補正後の手首, (T, 2) 肩・ひじの補正角 [rad])。
    """
    S, E, W = pos['S'], pos['E'], pos['W']
    upper, fore = E - S, W - E
    a = np.linalg.norm(upper, axis=-1)
    b = np.linalg.norm(fore, axis=-1)
    limit = np.deg2rad(float(max_deg))
    # ひじ: 上腕と前腕を含む面の中で、肩 → 手首の距離が目標と同じになる曲げへ（0 = 腕がまっすぐ）
    normal = np.cross(upper, fore)
    n_len = np.linalg.norm(normal, axis=-1)
    unit_normal = normal / np.maximum(n_len, 1e-12)[:, None]
    bend = np.arctan2(n_len, np.sum(upper * fore, -1))
    goal = targets - S
    dist = np.clip(np.linalg.norm(goal, axis=-1), np.abs(a - b) + 1e-6 * (a + b), (a + b) * (1.0 - 1e-6))
    want = np.arccos(np.clip((dist ** 2 - a ** 2 - b ** 2) / np.maximum(2.0 * a * b, _EPS), -1.0, 1.0))
    delta = np.clip(want - bend, -limit, limit)
    delta = np.where(n_len > 1e-9 * np.maximum(a * b, _EPS), delta, 0.0)   # まっすぐなら面が決まらない
    Rb = quat.to_matrix(quat.from_rotvec(unit_normal * delta[:, None]))
    fore2 = np.einsum('tab,tb->ta', Rb, fore)
    # 肩: 肩 → 手首の向きを目標へ向ける最小の回転
    turn = _turn_towards(upper + fore2, goal, limit)
    G_arm = turn @ glob['arm']
    G_elb = turn @ Rb @ glob['elbow']
    W2 = S + np.einsum('tab,tb->ta', turn, upper + fore2)
    swing = quat.angle_between(quat.from_matrix(turn), quat.IDENTITY)
    return G_arm, G_elb, W2, np.stack([swing, np.abs(delta)], axis=-1)


def keep_hand_positions(skel, rt, glob_rot, local, smpl_rest, rest_vertices, vertex_weights, cfg,
                        contacts_cfg, unit):
    """ステージ9h。local（ステージ9 のローカル回転）の 腕・ひじ・手首 を直した dict と HandReachResult を返す。

    glob_rot: SMPL の大域回転 (T, J, 3, 3) / smpl_rest: SMPL の初期姿勢の関節 [m] /
    rest_vertices・vertex_weights: SMPL の初期姿勢の頂点 [m] とスキニングのウェイト（胸の厚みを測る。None でもよい）/
    cfg: 設定 hand_reach / contacts_cfg: 設定 contacts（MMD の胴の形）/ unit: スケール係数（MMD 単位 / m）
    """
    T = len(glob_rot)
    zeros = np.zeros((T, 2))
    chest = next((name for name, j, _ in rt.specs if j == CHEST_JOINT), None)
    if not cfg.enabled or chest is None or T == 0:
        return local, HandReachResult(False, zeros, zeros, zeros, np.zeros((T, 2, 2)),
                                      info=dict(enabled=False))
    J = np.asarray(smpl_rest, np.float64)
    C = rt.correction[chest]
    src = smpl_torso(J, rest_vertices, vertex_weights, C, unit)
    dst = mmd_torso(skel, chest, contacts_cfg, unit)

    glob = globals_from_local(rt, glob_rot, local)
    names = {s: [s + b for b in ('腕', 'ひじ', '手首')] for s in SIDES}
    pos = bone_positions(skel, glob, [chest] + names['左'] + names['右'], T)
    G_chest_smpl = glob_rot[:, CHEST_JOINT] @ C              # = MMD の胸のボーンの大域回転
    near, far = float(cfg.near_m) * unit, float(cfg.far_m) * unit
    lo, hi = np.deg2rad(np.asarray(cfg.bend_deg, np.float64))

    out = dict(local)
    weight, shift, residual = np.zeros((T, 2)), np.zeros((T, 2)), np.zeros((T, 2))
    correction = np.zeros((T, 2, 2))
    for side, s in enumerate(SIDES):
        collar, shoulder, elbow, wrist = ARM_JOINTS[side]
        # SMPL の手首: 胸（背骨3）からの位置を、MMD の胸の初期姿勢の向き・MMD 単位で
        upper_s = glob_rot[:, shoulder] @ (J[elbow] - J[shoulder])
        fore_s = glob_rot[:, elbow] @ (J[wrist] - J[elbow])
        rel = (glob_rot[:, CHEST_JOINT] @ (J[collar] - J[CHEST_JOINT])
               + glob_rot[:, collar] @ (J[shoulder] - J[collar]) + upper_s + fore_s)
        u_smpl = np.einsum('tba,tb->ta', G_chest_smpl, rel) * unit
        u_mmd = src.map_to(dst, u_smpl)
        mapped = pos[chest] + np.einsum('tab,tb->ta', glob[chest], u_mmd)
        # 重み: 胴に近く、ひじが曲がっているほど 1
        w_near = 1.0 - filters.smoothstep((src.distance(u_smpl) - near) / max(far - near, _EPS))
        w_bend = filters.smoothstep((_angle_between(upper_s, fore_s) - lo) / max(hi - lo, _EPS))
        w = w_near * w_bend
        arm, elb, wr = names[s]
        W = pos[wr]
        target = W + w[:, None] * (mapped - W)
        G_arm, G_elb, W2, corr = reach_wrists(dict(S=pos[arm], E=pos[elb], W=W),
                                              dict(arm=glob[arm], elbow=glob[elb]), target,
                                              cfg.max_deg)
        # 大域回転 → ローカル回転（親はキーを打つボーンのうち最も近い祖先。手首の大域回転はそのまま）
        new = {arm: G_arm, elb: G_elb, wr: glob[wr]}

        def parent_global(name):
            p = rt.keyed_parent[name]
            return None if p is None else new.get(p, glob[p])

        active = w > 0.0
        for name in (arm, elb, wr) if active.any() else ():
            Gp = parent_global(name)
            L = new[name] if Gp is None else np.swapaxes(Gp, -1, -2) @ new[name]
            q = np.array(out[name], np.float64, copy=True)   # 重みが 0 のフレームは元の値のまま
            q[active] = quat.from_matrix(L[active])
            out[name] = quat.make_continuous(q)
        weight[:, side] = w
        shift[:, side] = np.linalg.norm(target - W, axis=-1)
        residual[:, side] = np.where(active, np.linalg.norm(target - W2, axis=-1), 0.0)
        correction[:, side] = np.where(active[:, None], np.rad2deg(corr), 0.0)

    cm = 100.0 / unit
    info = dict(enabled=True, chest_bone=chest,
                smpl=src.as_dict(unit), mmd=dst.as_dict(unit),
                ratio=dict(width=round(dst.width / src.width, 3),
                           height=round(dst.height / src.height, 3),
                           depth=round(dst.depth / src.depth, 3)),
                frames=[int(v) for v in (weight > 0.5).sum(0)],
                max_shift_cm=[round(float(v), 2) for v in shift.max(0, initial=0.0) * cm],
                max_residual_cm=[round(float(v), 2) for v in residual.max(0, initial=0.0) * cm],
                max_correction_deg=dict(zip(('shoulder', 'elbow'), (
                    round(float(v), 2) for v in correction.max((0, 1), initial=0.0)))))
    return out, HandReachResult(True, weight, shift, residual, correction, src, dst, info)
