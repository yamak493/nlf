"""ステージ9l: 脚（足・ひざ・足首）の回転のキー（膝の向き）。

MMD の脚は足ＩＫで決まる。足先の位置と骨盤の位置は伝わるが、膝がどちらを向くか（内股・がに股・膝を寄せる振り付け）は
股関節から足首までの軸まわりに自由で、IK ソルバの解き方しだいになる（SMPL の推定が持っている膝の向きが使われない）。
MMD / MMM の IK は、リンクボーン（足・ひざ）のその時の回転（キーの値）から解き始めるので、キーの姿勢がすでに足ＩＫに
届いていれば、IK はほとんど動かさず、キーの膝の向きがそのまま残る。

そこで、MMD のモデルの脚（足 → ひざ → 足首 の初期位置の長さ・曲がり）で、股関節から足ＩＫのターゲットまでを 2 ボーンの
IK として解き、膝を SMPL の膝の向きの側に置いた 足・ひざ・足首 の回転をキーにする。

* **膝の向き**: SMPL の膝の曲げの軸（太ももの X 軸）を 股関節 → 足首 の線に垂直にして、その軸で曲げたときに膝が出る向き
  （MMD の ひざ の曲げの軸を SMPL にそろえる）。骨盤の座標系でならす（pole_one_euro）
* **ひざ**: MMD の ひざ の IK の角度制限（ローカルの X 軸まわりだけ・後ろへ曲げる向きだけ）に合わせ、X 軸まわりだけに回す。
  曲げ角は、股関節から足ＩＫのターゲットまでの距離になる角（届かないときは伸ばせる所まで。近すぎるときは曲げられる所まで）
* **足**: 太もも → 足首 の向きを股関節からターゲットへの向きに合わせ、ひざ の曲げの軸（X）を膝の向きで決まる面の法線に
  合わせる回転
* **足首**: 足ＩＫの回転と同じ向き（つま先ＩＫの IK が、すでに届いた状態から始まる。足先の左右の傾きも SMPL に合わせる）

センター・足ＩＫは種類（フル・接地優先・移動なし）ごとに違うので、キーを作るとき（pipeline.build_tracks）に種類ごとに
解く。膝の向きは種類によらない（kin から一度だけ求める）。ステージ9b は、フルの脚の回転で脚の当たり判定を作る
（MMD で表示される脚と同じ位置）。
"""
from dataclasses import dataclass, field

import numpy as np

from . import filters, quat
from .body_model import LEG_JOINTS
from .center import ReachGeometry
from .skeleton import SIDES

LEG_BONES = tuple(s + n for s in SIDES for n in ('足', 'ひざ', '足首'))   # キーを打つ脚のボーン
X_AXIS = np.array([1.0, 0.0, 0.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])
MIN_BEND_DEG = 0.5    # MMD の ひざ の角度制限の既定（初期姿勢から後ろへ 0.5 度以上）
_EPS = 1e-9


@dataclass
class LegResult:
    local: dict                  # ボーン名 → ローカル回転 (T, 4)（内部座標。キーを打つ親に対する回転）
    glob: dict                   # ボーン名 → 大域回転 (T, 3, 3)（初期姿勢からの回転）
    knee: np.ndarray             # (T, 2, 3) ひざ の位置
    bend_deg: np.ndarray         # (T, 2) ひざ の曲げ（初期姿勢から）[度]
    knee_out_deg: np.ndarray     # (T, 2) 膝の向きの、骨盤の正面からの外向きの角（+ = 外 / がに股）[度]
    unreached: np.ndarray        # (T, 2) bool 足ＩＫのターゲットに届かないフレーム
    info: dict = field(default_factory=dict)


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), _EPS)


def _perpendicular(v, axis):
    """v の、単位ベクトル axis に垂直な成分。"""
    return v - np.sum(v * axis, -1, keepdims=True) * axis


def _frames(up, lateral):
    """(N, 3) の「上向き」と「左右方向」から直交座標系 (N, 3, 3)（列が x, y, z 軸）。quat.frame_from_up_lateral の配列版。"""
    ey = _unit(up)
    ex = _unit(_perpendicular(np.broadcast_to(lateral, ey.shape), ey))
    return np.stack([ex, ey, np.cross(ex, ey)], axis=-1)


def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.zeros(np.shape(theta) + (3, 3))
    R[..., 0, 0] = 1.0
    R[..., 1, 1], R[..., 1, 2] = c, -s
    R[..., 2, 1], R[..., 2, 2] = s, c
    return R


def knee_poles(kin, fps=None, one_euro=None):
    """(T, 2 [左, 右], 3) SMPL の膝の向き（股関節 → 足首 の線に垂直な単位ベクトル）。

    SMPL の膝の曲げの軸（股関節の大域回転で回した X 軸。SMPL の膝は太ももの X 軸まわりに曲がる）を、股関節 → 足首 の線に
    垂直にして、その軸で後ろへ曲げたときに膝が出る向き。MMD の ひざ も X 軸まわりに曲がるので、曲げの軸をそろえることになる。
    膝の位置の、股関節 → 足首 の線からのずれの向きは使わない（SMPL の初期姿勢の膝はその線より 2cm ほど外にあるので、
    前後に曲げただけでも外を向いて見える。曲げ 20 度で 16 度、45〜90 度で 8 度ほどのがに股になる）。軸はまっすぐな脚でも決まる。
    one_euro（設定 leg_keys.pole_one_euro。fps も渡す）があれば、骨盤の座標系で見た膝の向きをゼロ位相の One Euro で
    ならす（太もものひねりは単眼推定でいちばん決まりにくく、そのまま使うと脚全体のひねりが震える。骨盤の座標系で
    ならすので、体ごと速く回る動きはならさない）。
    """
    J = np.asarray(kin.joints, np.float64)
    R = np.asarray(kin.glob_rot, np.float64)
    out = np.empty((len(J), 2, 3))
    for side, (hip, _, ankle, _) in enumerate(LEG_JOINTS):
        u = _unit(J[:, ankle] - J[:, hip])
        axis = _unit(_perpendicular(R[:, hip] @ X_AXIS, u))
        out[:, side] = np.cross(u, axis)
    if fps is None or one_euro is None or float(one_euro.min_cutoff) <= 0:
        return out
    pelvis = R[:, 0]
    local = np.einsum('tba,tsb->tsa', pelvis, out)                  # 骨盤の座標系で見た向き
    local = filters.one_euro(local, fps, float(one_euro.min_cutoff), float(one_euro.beta),
                             zero_phase=True, vector_axis=-1)
    return _unit(np.einsum('tab,tsb->tsa', pelvis, local))


def _global_of(skel, local, name, cache):
    """キーを打つボーン name の大域回転 (T, 3, 3)（local のローカル回転を、キーを打つ祖先から順に掛けたもの）。"""
    if name not in cache:
        parent = skel.nearest_ancestor(name, set(local))
        R = quat.to_matrix(local[name])
        cache[name] = R if parent is None else _global_of(skel, local, parent, cache) @ R
    return cache[name]


def solve_legs(skel, center_delta, ik, local, poles, cfg):
    """センター・足ＩＫ・キーを打つボーンのローカル回転 local（下半身を含む）から、足・ひざ（・足首）の回転（LegResult）。

    center_delta: (T, 3) センター＋グルーブの差分 / ik: FootIKResult（delta・rotation）/ poles: knee_poles の結果。
    cfg: 設定 leg_keys。
    """
    center_delta = np.asarray(center_delta, np.float64)
    T = len(center_delta)
    cache = {}
    lower = _global_of(skel, local, '下半身', cache)
    geom = ReachGeometry.from_skeleton(skel)
    hips = geom.hip_positions(center_delta, lower)                  # (T, 2, 3)
    targets = geom.ik[None] + np.asarray(ik.delta, np.float64)      # (T, 2, 3)
    forward = lower @ Z_AXIS
    min_bend = np.deg2rad(MIN_BEND_DEG)
    out_local, out_glob = {}, {}
    knee_pos = np.empty((T, 2, 3))
    bend_deg = np.empty((T, 2))
    knee_out = np.empty((T, 2))
    unreached = np.zeros((T, 2), bool)
    for side, s in enumerate(SIDES):
        names = [s + n for n in ('足', 'ひざ', '足首')]
        v1 = skel.internal(names[1]) - skel.internal(names[0])       # 太もも（初期姿勢）
        v2 = skel.internal(names[2]) - skel.internal(names[1])       # すね
        # ひざ を X 軸まわりに θ 回したときの 足 → 足首 の距離: |v1|² + |v2|² + 2 v1x v2x + 2ρ cos(θ − ψ)
        a = v1[1] * v2[1] + v1[2] * v2[2]
        b = v1[2] * v2[1] - v1[1] * v2[2]
        rho, psi = np.hypot(a, b), np.arctan2(b, a)
        d = targets[:, side] - hips[:, side]
        dist = np.linalg.norm(d, axis=-1)
        c = (dist ** 2 - v1 @ v1 - v2 @ v2 - 2.0 * v1[0] * v2[0]) / max(2.0 * rho, _EPS)
        theta = psi + np.arccos(np.clip(c, -1.0, 1.0))               # 後ろへ曲げる側の解
        unreached[:, side] = (np.abs(c) > 1.0) | (theta < min_bend)
        theta = np.maximum(theta, min_bend)
        u = _unit(d)
        # 膝の向き: SMPL の膝の向きを、股関節 → ターゲット の線に垂直にしたもの（決まらなければ骨盤の正面）
        p = _perpendicular(poles[:, side], u)
        weak = np.linalg.norm(p, axis=-1) < 1e-3
        p[weak] = _perpendicular(forward[weak], u[weak])
        p = _unit(p)
        n = np.cross(p, u)                                           # ひざ の曲げの軸（初期姿勢の X に当たる）
        w = v1[None] + np.einsum('tab,b->ta', _rot_x(theta), v2)     # 太ももの座標系での 足 → 足首
        G1 = _frames(u, n) @ np.swapaxes(_frames(w, X_AXIS), -1, -2)
        G2 = G1 @ _rot_x(theta)
        out_glob[names[0]], out_glob[names[1]] = G1, G2
        knee_pos[:, side] = hips[:, side] + G1 @ v1
        bend_deg[:, side] = np.rad2deg(theta)
        # 膝の外向きの角: 膝の向きと骨盤の正面を水平面に写した角（左足は +X、右足は −X へ回るのが外向き）
        outward = np.arctan2(np.cross(forward, p)[:, 1], np.sum(forward * p, -1))
        knee_out[:, side] = np.rad2deg(outward if side == 0 else -outward)
        if cfg.ankle:
            out_glob[names[2]] = quat.to_matrix(np.asarray(ik.rotation, np.float64)[:, side])
    keyed = set(local) | set(out_glob)
    for name, G in out_glob.items():
        parent = skel.nearest_ancestor(name, keyed)
        if parent is None:
            Gp = np.eye(3)
        elif parent in out_glob:
            Gp = out_glob[parent]
        else:
            Gp = _global_of(skel, local, parent, cache)
        out_local[name] = quat.make_continuous(quat.from_matrix(np.swapaxes(Gp, -1, -2) @ G))
    info = dict(bones=list(out_local), unreached_frames=unreached.sum(0).tolist(),
                knee_out_deg={q: np.percentile(knee_out, p, axis=0).round(1).tolist()
                              for q, p in (('p5', 5), ('median', 50), ('p95', 95))},
                max_bend_deg=bend_deg.max(0).round(1).tolist())
    return LegResult(out_local, out_glob, knee_pos, bend_deg, knee_out, unreached, info)
