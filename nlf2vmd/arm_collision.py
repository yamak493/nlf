"""ステージ9a: 腕どうしの貫通の防止。

回転をコピーしただけでは、組んだ腕・交差した腕が MMD で互いを貫通する（MMD モデルは SMPL と腕の長さ・肩幅・
太さが違い、単眼推定では重なって写った腕どうしの前後の距離もぶれるため）。そこで MMD モデルの左右の腕を、
それぞれ上腕（腕→ひじ）・前腕（ひじ→手首）・手（手首→指先）の 3 本のカプセル（線分＋半径）で表し、左右の
カプセルが重なったフレームで、自重する側の腕（設定 arm_collision.mode）を肩の関節（腕ボーンの位置）まわりに
回して離す。ひじ・手首のローカル回転は変えないので、腕の形（ひじの曲げ・手首の向き）はそのままで、腕全体の
向きだけが変わる。キーが変わるのは自重する側の腕ボーンだけ。

* 半径と手の長さは、PMX のメッシュ（ウェイトが最も大きいボーンが腕・ひじ・手首とその子の頂点）から求める。
  半径は骨からの距離の中央値、手の向きは手首から手の頂点の重心への向き、手の長さはその向きに測った頂点の
  95 パーセンタイル。メッシュが無い（PMX を指定しない）ときは設定の値を使う（手の向きは手首→中指１など）
* 1 フレームずつ順に、前のフレームの補正（肩の座標系で持ち越す）から始めて、補正をなるべく 0（推定の姿勢）に
  戻しながら、どの組も重ならないように、肩まわりの小さな回転を繰り返し足す（線形化した「重ならない」条件の
  もとで、推定の姿勢に最も近づく回転）。相手の腕が来れば押しのけられ、離れれば推定の姿勢に戻る。前のフレームの
  配置から続けて動かすので、推定の腕が相手の腕を通り抜けても、自重する腕は来た側に留まる（押しのけきれずに
  補正が max_deg に達したときだけ、通り抜けて推定の側に移る）
* 離す向きは、重なっている 2 本のカプセルの最近点を結ぶ向き。肩のすぐ近くの重なり（相手の手が肩に触れている
  等）は、肩まわりに回しても離せないので扱わない
* 補正の回転を肩の親（肩ボーン）の座標系で時間方向にならしてから、ならして浅くなった重なりをもう一度離す
"""
from dataclasses import dataclass

import numpy as np

from . import filters, quat
from .skeleton import SIDES

MODES = ('left', 'right', 'none')
MODE_LABELS = dict(left='左腕の動きを自重する', right='右腕の動きを自重する', none='何もしない')
SEGMENT_BONES = ('腕', 'ひじ', '手首')      # カプセルの根元のボーン（上腕・前腕・手）
# 組（自重する腕の線分, 相手の腕の線分）の 9 通り
PAIR_Y = np.repeat(np.arange(3), 3)
PAIR_O = np.tile(np.arange(3), 3)
ITERATIONS = 30               # 1 フレームで回す回数の上限（推定の姿勢へ戻しながら離す）
FINAL_ITERATIONS = 10         # そのあと、離すだけで回す回数の上限（戻しすぎた重なりを残さない）
DAMPING_M = 0.1               # 1 回の回転の減衰（長さ。肩から最近点までの腕の長さより十分小さく）
MAX_STEP_DEG = 20.0           # 1 回に回す角度の上限
STOP_RAD = 1e-4               # 1 回に回す角度がこれより小さくなったら止める
MIN_LEVER_M = 0.03            # 肩からこれより近い重なりは、肩まわりに回しても離せないので扱わない
TOLERANCE_M = 1e-4
HAND_TIP_PERCENTILE = 95.0
MIN_VERTICES = 20             # 半径を測る頂点がこれより少ない部位は設定の値を使う
DEEP_START = 0.5              # 持ち越す補正が無いのに、半径の和のこの割合より深く重なっていれば別の向きからも解く
OVERLAP_TOL_M = 0.005         # 評価指標: カプセルがこれより深く重なったフレームを「重なり」と数える
_EPS = 1e-9


@dataclass
class ArmModel:
    """左右の腕のカプセル（内部座標・MMD 単位）。"""
    tip: np.ndarray        # (2, 3) 初期姿勢の手の先（手首に固定した点）
    radius: np.ndarray     # (2, 3) 上腕・前腕・手の半径
    source: str            # 'mesh'（PMX のメッシュから）| 'config'（設定の値）


@dataclass
class ArmCollisionResult:
    mode: str
    side: int                    # 自重する腕（0 = 左 / 1 = 右 / -1 = なし）
    radius: np.ndarray           # (2, 3) [MMD 単位]
    radius_source: str
    correction_deg: np.ndarray   # (T,) 自重する腕に掛けた補正の回転角 [度]
    depth_before: np.ndarray     # (T,) 左右の腕のカプセルの最も深い重なり [MMD 単位]（負 = 離れている）
    depth_after: np.ndarray

    def overlap_frames(self, tol):
        """(処理前, 処理後) カプセルが tol [MMD 単位] より深く重なっているフレーム数。"""
        return int((self.depth_before > tol).sum()), int((self.depth_after > tol).sum())


def _ancestor_or_self(skel, name, candidates):
    return name if name in candidates else skel.nearest_ancestor(name, candidates)


def _distance_to_segment(points, a, b):
    ab = b - a
    t = np.clip((points - a) @ ab / max(float(ab @ ab), _EPS), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=-1)


def _hand_axis(skel, s):
    """初期姿勢の手の向き（単位ベクトル。内部座標）: 手首 → 中指１、無ければ手首の表示先、前腕の延長。"""
    wrist = skel.internal(s + '手首')
    for target in (skel.internal(s + '中指１') if skel.has(s + '中指１') else None,
                   skel.tail_internal(s + '手首'),
                   wrist + (wrist - skel.internal(s + 'ひじ'))):
        if target is not None and np.linalg.norm(target - wrist) > 1e-6:
            return (target - wrist) / np.linalg.norm(target - wrist)
    return np.array([1.0 if s == '左' else -1.0, 0.0, 0.0])


def arm_model(skel, cfg, unit):
    """腕のカプセルの寸法。半径は設定 radius_m（auto なら PMX のメッシュから）× radius_scale。"""
    fallback = np.asarray(cfg.fallback_radius_m, np.float64) * unit
    radius = np.tile(fallback, (2, 1))
    tips = []
    measured = False
    mesh = skel.mesh_points is not None and len(skel.mesh_points) > 0
    if mesh:
        points = skel.mesh_points * [1.0, 1.0, -1.0]   # 内部座標へ
        names, inverse = np.unique(skel.mesh_bones, return_inverse=True)
    for side, s in enumerate(SIDES):
        wrist = skel.internal(s + '手首')
        axis = _hand_axis(skel, s)
        hand_length = float(cfg.hand_length_m) * unit
        if mesh:
            # 頂点の部位 = ウェイトが最も大きいボーンから根元へたどって最初に着く 腕 / ひじ / 手首
            roots = {s + b for b in SEGMENT_BONES}
            owner = np.array([(_ancestor_or_self(skel, n, roots) or '') if skel.has(n) else ''
                              for n in names])[inverse.reshape(-1)]
            for k, bone in enumerate(SEGMENT_BONES):
                pts = points[owner == s + bone]
                if len(pts) < MIN_VERTICES:
                    continue
                if k == 2:   # 手: 向き（手首 → 手の頂点の重心）と長さも頂点から測る
                    centroid = pts.mean(0) - wrist
                    if np.linalg.norm(centroid) > 1e-6:
                        axis = centroid / np.linalg.norm(centroid)
                    hand_length = max(float(np.percentile((pts - wrist) @ axis,
                                                          HAND_TIP_PERCENTILE)), 0.05 * unit)
                    end = wrist + axis * hand_length
                else:
                    end = skel.internal(s + SEGMENT_BONES[k + 1])
                d = _distance_to_segment(pts, skel.internal(s + bone), end)
                radius[side, k] = float(np.percentile(d, float(cfg.radius_percentile)))
                measured = True
        tips.append(wrist + axis * hand_length)
    if cfg.radius_m != 'auto':
        radius = np.tile(np.asarray(cfg.radius_m, np.float64).reshape(3) * unit, (2, 1))
        measured = False
    radius = np.maximum(radius * float(cfg.radius_scale), 1e-6)
    return ArmModel(np.stack(tips), radius, 'mesh' if measured else 'config')


def _chain(skel, name):
    """name から根元までのボーン（根元が先頭）。"""
    chain, seen = [], set()
    while name is not None and skel.has(name) and name not in seen:
        chain.append(name)
        seen.add(name)
        name = skel.parents.get(name)
    return chain[::-1]


def bone_positions(skel, glob, names, num_frames):
    """ボーン名 → 位置 (T, 3)（内部座標）。glob: キーを打つボーンの大域回転 {名前: (T, 3, 3)}。

    キーの無いボーン（捩り・肩C・腰など）は親の回転をそのまま受け継ぐ。センター・グルーブの移動は含めない
    （左右の腕の相対位置には効かないため）。
    """
    out = {}
    for name in names:
        chain = _chain(skel, name)
        p = np.tile(skel.internal(chain[0]), (num_frames, 1))
        R = glob.get(chain[0])
        for parent, child in zip(chain[:-1], chain[1:]):
            off = skel.internal(child) - skel.internal(parent)
            p = p + (off if R is None else R @ off)
            R = glob.get(child, R)
        out[name] = p
    return out


def arm_points(skel, glob, model, num_frames):
    """(T, 2 [左, 右], 4 [腕, ひじ, 手首, 手の先], 3) 腕のカプセルの端点（内部座標）。"""
    names = [s + b for s in SIDES for b in SEGMENT_BONES]
    pos = bone_positions(skel, glob, names, num_frames)
    out = np.empty((num_frames, 2, 4, 3))
    for side, s in enumerate(SIDES):
        for k, b in enumerate(SEGMENT_BONES):
            out[:, side, k] = pos[s + b]
        wrist = s + '手首'
        out[:, side, 3] = pos[wrist] + glob[wrist] @ (model.tip[side] - skel.internal(wrist))
    return out


def closest_points(p0, p1, q0, q1):
    """線分 p0–p1 と q0–q1 の最近点の組 (..., 3)（Ericson『Real-Time Collision Detection』5.1.9）。"""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a = np.maximum(np.sum(d1 * d1, -1), _EPS)
    e = np.maximum(np.sum(d2 * d2, -1), _EPS)
    b, c, f = np.sum(d1 * d2, -1), np.sum(d1 * r, -1), np.sum(d2 * r, -1)
    denom = a * e - b * b
    general = denom > 1e-12 * a * e                     # 平行でない
    s = np.where(general, np.clip((b * f - c * e) / np.where(general, denom, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0.0, np.clip(-c / a, 0.0, 1.0),
                 np.where(t > 1.0, np.clip((b - c) / a, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    return p0 + d1 * s[..., None], q0 + d2 * t[..., None]


def _pairs(Y, O):
    """9 組の最近点 (..., 9, 3) の組。Y, O: (..., 4, 3) 自重する腕 / 相手の腕の端点。"""
    return closest_points(Y[..., PAIR_Y, :], Y[..., PAIR_Y + 1, :],
                          O[..., PAIR_O, :], O[..., PAIR_O + 1, :])


def overlap_depth(Y, O, radius_y, radius_o):
    """(T,) 左右の腕のカプセルの最も深い重なり（負なら離れている距離）。"""
    cp, cq = _pairs(Y, O)
    dist = np.linalg.norm(cp - cq, axis=-1)
    return (radius_y[PAIR_Y] + radius_o[PAIR_O] - dist).max(-1)


def _rotation(rotvec):
    """回転ベクトル (3,) → 回転行列（ロドリゲスの公式）。"""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rotvec / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _angle(R):
    return float(np.arccos(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)))


def _rotvec(R):
    """回転行列 → 回転ベクトル（180 度未満の回転）。"""
    theta = _angle(R)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(theta)
    return w * (0.5 if s < 1e-9 else 0.5 * theta / s)


def _cross(a, b):
    """np.cross と同じ（(..., 3) どうし。小さな配列では np.cross より速い）。"""
    return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1)


def _perpendicular(Y, O):
    """(9, 3) 各組の 2 本の線分に垂直な向き（最近点が重なって向きが決まらないときの代わり）。"""
    n = _cross(Y[PAIR_Y + 1] - Y[PAIR_Y], O[PAIR_O + 1] - O[PAIR_O])
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return np.where(norm > _EPS, n / np.maximum(norm, _EPS), [0.0, 0.0, 1.0])


def _constrained_step(J, b, damping):
    """J δ ≥ b をなるべく満たす小さな δ（減衰付きのアクティブセット法）。

    満たされていない組（b > 0）を等式として、減衰付きの最小ノルム解 δ = Jᵀ (J Jᵀ + damping·I)⁻¹ b を求め、
    ラグランジュ乗数が負の組を外し、満たされていない組を加えることを繰り返す。減衰で、同時には満たせない組
    （腕を前へも後ろへも動かす必要がある等）があっても回転が暴れず、少しずつ動かすうちに腕が相手の腕の回りを
    滑って、満たせる配置に移る。
    """
    active = [int(c) for c in np.flatnonzero(b > 0.0)]
    delta = np.zeros(3)
    tol = 1e-9 * max(1.0, float(np.abs(b).max()))
    for _ in range(2 * len(b) + 2):
        if active:
            Ja = J[active]
            lam = np.linalg.solve(Ja @ Ja.T + damping * np.eye(len(active)), b[active])
            if (lam < -tol).any():
                active.pop(int(np.argmin(lam)))
                continue
            delta = Ja.T @ lam
        short = b - J @ delta
        short[active] = -np.inf
        c = int(np.argmax(short))
        if short[c] <= tol:
            break
        active.append(c)
    return delta


@dataclass
class _Params:
    max_angle: float
    min_lever: float
    tol: float
    damping: float
    return_step: float    # 1 フレームに推定の姿勢へ戻す角度の上限 [rad]


def _solve_frame(Y, O, R, Q, p, goal=None, ignore=None):
    """1 フレーム分。自重する腕 Y (4, 3) を肩 Y[0] まわりに回す補正 Q (3, 3) を、初期値 Q から求める。

    R: (9,) 組ごとの離す距離（半径の和＋余裕）。goal (3, 3) を渡すと、重ならない範囲で Q を goal に近づける
    （渡さなければ離すだけ）。ignore: (9,) 離さない組。
    """
    A = Y[0]
    rel = Y - A
    skip = np.zeros(len(R), bool) if ignore is None else ignore
    pulling = goal is not None
    for it in range(ITERATIONS + FINAL_ITERATIONS):
        # 推定の姿勢へ戻しながら離し、止まったら（または ITERATIONS 回で）離すだけにする。減衰のため、戻す力と
        # 釣り合った所ではわずかに重なりが残るので、それを離すだけで消す
        pulling = pulling and it < ITERATIONS
        cur = A + rel @ Q.T
        cp, cq = _pairs(cur, O)
        d = cp - cq
        dist = np.linalg.norm(d, axis=-1)
        n = d / np.maximum(dist, _EPS)[:, None]
        if (dist <= _EPS).any():   # 線分が交わっていて向きが決まらない
            n = np.where((dist > _EPS)[:, None], n, _perpendicular(cur, O))
        pen = R - dist
        # 肩まわりの小さな回転 ω で、最近点は ω × v 動き、組は n·(ω × v) = (v × n)·ω だけ離れる。
        # (v × n) が小さい組（肩のすぐ近く・肩から見た向きと n が平行）は肩まわりに回しても離せない
        J = _cross(cp - A, n)
        use = (np.linalg.norm(J, axis=-1) > p.min_lever) & ~skip
        target = _rotvec(goal @ Q.T) if pulling else np.zeros(3)   # goal へ近づける回転
        if not (use & (pen > p.tol)).any() and np.linalg.norm(target) < 1e-9:
            break
        omega = target
        if use.any():
            Ju = J[use]
            omega = target + _constrained_step(Ju, pen[use] + p.tol - Ju @ target, p.damping)
        step = float(np.linalg.norm(omega))
        if step < STOP_RAD:
            if pulling:
                pulling = False
                continue
            break
        Q = _rotation(omega * min(1.0, np.deg2rad(MAX_STEP_DEG) / step)) @ Q
        if _angle(Q) > p.max_angle:
            rv = _rotvec(Q)
            Q = _rotation(rv * (p.max_angle / np.linalg.norm(rv)))
    return Q


def _overlapping(Y, O, R, Q, tol):
    """(9,) 補正 Q で重なっている組。"""
    A = Y[0]
    cp, cq = _pairs(A + (Y - A) @ Q.T, O)
    return R - np.linalg.norm(cp - cq, axis=-1) > tol


def _toward_identity(local, step):
    """補正 local を単位回転（推定の姿勢）の向きへ最大 step [rad] 戻した回転。"""
    rv = _rotvec(local)
    a = float(np.linalg.norm(rv))
    return np.eye(3) if a <= step else _rotation(rv * (1.0 - step / a))


def _resolve_sequential(Y, O, R, Gp, p):
    """1 回目: フレーム順に、前のフレームの補正から始めて解く（押しのけ・戻り）。

    Gp: (T, 3, 3) 自重する腕の親（肩）の大域回転。補正はこの座標系で次のフレームへ持ち越し、推定の姿勢へは
    1 フレームに p.return_step までしか戻さない（相手の腕が離れても、はね戻らない）。押しのけきれずに補正が
    上限に達しても重なっている組は、離れるまで扱わない（その間は腕どうしが少しずつ通り抜ける）。
    戻り値: 補正 (T, 3, 3) と、扱わなかった組 (T, 9)。
    """
    T = len(Y)
    cp, cq = _pairs(Y, O)
    raw_pen = R - np.linalg.norm(cp - cq, axis=-1)   # 推定の姿勢のままでの重なり
    Q = np.tile(np.eye(3), (T, 1, 1))
    ignored = np.zeros((T, len(R)), bool)
    local = np.eye(3)                   # 前のフレームの補正（肩の座標系）
    skip = np.zeros(len(R), bool)       # 扱わない組
    for t in range(T):
        # _angle は arccos なので 1e-8 rad 程度より小さい角を区別できない（丸め誤差で単位回転に戻りきらない）
        carried = _angle(local) > 1e-6
        if not carried and (raw_pen[t] <= 0.0).all():
            skip[:] = False
            continue
        if not carried and raw_pen[t].max() > DEEP_START * R.min():
            Q[t] = _solve_deep_start(Y[t], O[t], R, Gp[t], p, skip)
        else:
            goal = Gp[t] @ _toward_identity(local, p.return_step) @ Gp[t].T
            Q[t] = _solve_frame(Y[t], O[t], R, Gp[t] @ local @ Gp[t].T, p, goal, skip)
        over = _overlapping(Y[t], O[t], R, Q[t], p.tol)
        if _angle(Q[t]) >= p.max_angle - 1e-6:
            skip = skip | over
        skip = skip & over
        ignored[t] = skip
        local = Gp[t].T @ Q[t] @ Gp[t]
    return Q, ignored


def _solve_deep_start(Y, O, R, G, p, ignore):
    """持ち越す補正が無いのに深く重なっているフレーム（動画の最初から腕を組んでいる・急に重なった等）。

    推定の姿勢から離すと、離す向きによっては腕が相手の腕に絡んで大きく回す解に落ちるので、肩の座標系 G の
    6 方向へ MAX_STEP_DEG 回した所からも解き、重なりが残らない解のうち補正の回転角が最も小さいものを選ぶ。
    """
    best, best_score = None, np.inf
    starts = [np.eye(3)] + [_rotation(G @ (sign * np.deg2rad(MAX_STEP_DEG) * np.eye(3)[axis]))
                            for axis in range(3) for sign in (1.0, -1.0)]
    A = Y[0]
    for Q0 in starts:
        Q = _solve_frame(Y, O, R, Q0, p, np.eye(3), ignore)
        cp, cq = _pairs(A + (Y - A) @ Q.T, O)
        pen = np.where(ignore, -np.inf, R - np.linalg.norm(cp - cq, axis=-1))
        residual = max(0.0, float(pen.max()) - p.tol)
        score = _angle(Q) + 100.0 * residual / p.min_lever   # 重なりが残る解は選ばない
        if score < best_score:
            best, best_score = Q, score
    return best


def _resolve_residual(Y, O, R, Q, ignored, p):
    """2 回目: ならした補正 Q (T, 3, 3) で重なるフレームだけ、重ならないところまで回す（戻さない）。
    ignored: (T, 9) 1 回目で扱わなかった組。"""
    A = Y[:, :1]
    cp, cq = _pairs(A + np.einsum('tab,tkb->tka', Q, Y - A), O)
    touching = (np.linalg.norm(cp - cq, axis=-1) < R) & ~ignored
    Q = Q.copy()
    for t in np.flatnonzero(touching.any(1)):
        Q[t] = _solve_frame(Y[t], O[t], R, Q[t], p, ignore=ignored[t])
    return Q


def resolve_arm_collisions(skel, rt, glob_rot, local, cfg, unit, fps):
    """ステージ9a。local（Retargeter.local_quats の結果）の自重する側の腕ボーンの回転を直した dict と、
    ArmCollisionResult を返す。glob_rot: SMPL の大域回転 (T, J, 3, 3)。"""
    mode = str(cfg.mode)
    if mode not in MODES:
        raise ValueError(f'arm_collision.mode は {" / ".join(MODES)} のいずれかです: {mode}')
    T = len(glob_rot)
    model = arm_model(skel, cfg, unit)
    glob = rt.global_matrices(glob_rot)
    pts = arm_points(skel, glob, model, T)
    before = overlap_depth(pts[:, 0], pts[:, 1], model.radius[0], model.radius[1])
    if mode == 'none' or T == 0:
        return local, ArmCollisionResult(mode, -1, model.radius, model.source, np.zeros(T),
                                         before, before.copy())

    y = SIDES.index('左') if mode == 'left' else SIDES.index('右')
    o = 1 - y
    arm = SIDES[y] + '腕'
    Y, O = pts[:, y], pts[:, o]
    R = model.radius[y][PAIR_Y] + model.radius[o][PAIR_O] + float(cfg.margin_m) * unit
    p = _Params(np.deg2rad(float(cfg.max_deg)), MIN_LEVER_M * unit, TOLERANCE_M * unit,
                (DAMPING_M * unit) ** 2, np.deg2rad(float(cfg.return_deg_per_s)) / fps)
    # 補正は肩の親（キーを打つ祖先 = 肩）の座標系で持ち越し・ならす（体が回っても向きが変わらないように）
    parent = rt.keyed_parent[arm]
    Gp = glob[parent] if parent is not None else np.tile(np.eye(3), (T, 1, 1))
    Gp_t = np.swapaxes(Gp, -1, -2)
    Q, ignored = _resolve_sequential(Y, O, R, Gp, p)
    rv = quat.to_rotvec(quat.from_matrix(Gp_t @ Q @ Gp))
    rv = filters.gaussian_time(rv, float(cfg.smooth_sec) * fps)
    Q = _resolve_residual(Y, O, R, Gp @ quat.to_matrix(quat.from_rotvec(rv)) @ Gp_t, ignored, p)

    q_local = quat.from_matrix(Gp_t @ Q @ Gp)
    angle = np.rad2deg(np.linalg.norm(quat.to_rotvec(q_local), axis=-1))
    if angle.max(initial=0.0) > 1e-9:   # 補正が無ければ回転はそのまま（丸め誤差も入れない）
        local = dict(local)
        local[arm] = quat.make_continuous(quat.mul(q_local, local[arm]))
    Y_after = Y[:, :1] + np.einsum('tab,tkb->tka', Q, Y - Y[:, :1])
    after = overlap_depth(Y_after, O, model.radius[y], model.radius[o])
    return local, ArmCollisionResult(mode, y, model.radius, model.source, angle, before, after)
