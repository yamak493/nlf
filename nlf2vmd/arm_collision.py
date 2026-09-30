"""ステージ9a: 腕どうしの貫通の防止。

回転をコピーしただけでは、組んだ腕・交差した腕が MMD で互いを貫通する（MMD モデルは SMPL と腕の長さ・肩幅・
太さが違い、単眼推定では重なって写った腕どうしの前後の距離もぶれるため）。そこで MMD モデルの左右の腕を、
それぞれ上腕（腕→ひじ）・前腕（ひじ→手首）・手（手首→指先）の 3 本のカプセル（線分＋半径）で表し、左右の
カプセルが重なったフレームで、自重する側の腕（設定 arm_collision.mode）を、肩の関節（腕ボーンの位置）まわりに
回し、ひじを曲げ伸ばしして離す（elbow: false なら肩だけ）。手首のローカル回転は変えないので、手首の向きは前腕に
対してそのまま。キーが変わるのは自重する側の腕・ひじボーンだけ。判定は、ステージ9h（胴に対する手の位置）で直した
後の腕の姿勢で行う。

* 半径と手の長さは、PMX のメッシュ（ウェイトが最も大きいボーンが腕・ひじ・手首とその子の頂点）から求める。
  半径は骨からの距離の中央値、手の向きは手首から手の頂点の重心への向き、手の長さはその向きに測った頂点の
  95 パーセンタイル。メッシュが無い（PMX を指定しない）ときは設定の値を使う（手の向きは手首→中指１など）
* 1 フレームずつ順に、前のフレームの補正（肩の回転は肩の座標系で、ひじの曲げはそのまま持ち越す）から始めて、補正を
  なるべく 0（推定の姿勢）に戻しながら、どの組も重ならないように、肩まわりの小さな回転とひじの小さな曲げを繰り返し
  足す（線形化した「重ならない」条件のもとで、推定の姿勢に最も近づく補正）。相手の腕が来れば押しのけられ、離れれば
  推定の姿勢に戻る。前のフレームの配置から続けて動かすので、推定の腕が相手の腕を通り抜けても、自重する腕は来た側に
  留まる（押しのけきれずに肩の補正が max_deg に達したときだけ、通り抜けて推定の側に移る）
* 離す向きは、重なっている 2 本のカプセルの最近点を結ぶ向き。肩のすぐ近くの重なり（相手の手が肩に触れている
  等）は、肩まわりに回しても離せないので扱わない
* **ひじ**（elbow）: ひじはひじの軸（ステージ9b と同じ。前腕の初期の向き × 正面）まわりの曲げ 1 自由度で、補正は
  ±elbow_max_deg、曲げ角（0 = まっすぐ）が 0〜150 度の範囲（まっすぐより先へ伸ばさない・曲げすぎない。推定の姿勢が
  すでに範囲の外ならそれ以上外へは動かさない）。肩とひじを合わせたコストは、腕の点（ひじ・手首・手の先）の動きを
  画像面内と奥行きに分けて測ったもの（_point_metric）なので、肩とひじのどちらをどれだけ使うかは、腕の点の動きが
  最も小さくなるように決まる（肩だけだと、肩より前にある前腕を前後へ動かすと上下左右にも動くが、ひじも使うと
  ひじ・手首・手の先を画像上の位置に近いまま前後へずらせる）
* **奥行き優先**（depth_cost < 1）: 単眼推定では、画像面内（上下左右）の腕の位置は動画に写ったとおりで確かだが、
  カメラの奥行き方向の位置はぶれる。そこで、腕（ひじ・手首・手の先）をカメラの奥行き方向へ動かす補正を安く見積もり
  （腕の点の動きのうち、奥行き方向の分のコストを depth_cost² 倍にする）、画像面内の動きを自重して前後へずらす。
  重なっている組の離す向きも、最近点を結ぶ向きと奥行きの軸の間から、そのコストで最も安く離せる向きを選ぶ（横に
  並んで重なった腕も、前後へずらして離せる）。前後の順（どちらの腕が手前か）は推定のまま。画像面内と奥行きの動きは
  1 つのコストで同時に決めるので、腕は奥行きを主に、画像面内にも少し動く（depth_cost の比で配分される）。奥行きを
  優先して回転角が max_deg に達しても離れないフレームは、そのフレームだけ depth_cost を 1（向きを区別しない）へ向けて
  離れるところまで弱める（画像面内の動きを必要な分だけ増やす）
* 補正（肩の回転は肩の親（肩ボーン）の座標系で、ひじの曲げはそのまま）を時間方向にならしてから、ならして浅くなった
  重なりをもう一度離す
"""
from dataclasses import dataclass, replace

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
DEPTH_SAMPLES = 13            # 奥行き優先: 離す向きの候補の数（最近点を結ぶ向きから奥行きの軸までの 90 度を等分）
SAME_DEPTH = 0.02             # 奥行き優先: 離す向きの奥行きの成分がこれより小さい（最近点がほぼ同じ奥行き）なら前後どちらへも離せる
RELAX_STEPS = 4               # 奥行き優先で上限に達したとき、depth_cost を 1 へ向けて弱める二分法の回数
ELBOW_FLEX_DEG = (0.0, 150.0) # ひじの曲げ角（0 = まっすぐ）をこの範囲に収める（まっすぐより先へ伸ばさない・曲げすぎない）
ELBOW_REG = 0.02              # ひじも曲げるときのコストに足す、肩・ひじの回転角の 2 乗の重み（どの点も動かない向き（まっすぐな
                              # 腕の軸まわりのひねり）を残さないため。大きくすると腕の向き・曲げを変えにくくなり、ひじの効きが減る）
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
    # (T,) 補正で自重する腕（ひじ・手首・手の先）が動いた距離の最大 [MMD 単位]。画像面内（カメラの奥行きの軸に垂直）と奥行き
    shift_image: np.ndarray = None
    shift_depth: np.ndarray = None
    depth_cost: float = 1.0
    depth_cost_used: np.ndarray = None   # (T,) フレームごとに使った depth_cost（上限に達して弱めたフレームは大きい）
    elbow_deg: np.ndarray = None         # (T,) 自重する腕のひじに掛けた曲げの補正 [度]
    elbow: bool = False                  # ひじも曲げて離したか（arm_collision.elbow）

    @property
    def relaxed_frames(self):
        """奥行きの優先を弱めたフレーム数。"""
        return 0 if self.depth_cost_used is None else int((self.depth_cost_used > self.depth_cost + 1e-9).sum())

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


def globals_from_local(rt, glob_rot, local):
    """キーを打つボーンの大域回転 {名前: (T, 3, 3)}（ローカル回転 local を親から順に掛けたもの）。"""
    base = rt.global_matrices(glob_rot)
    out = {}
    for name in rt.bones:          # 親が子より先に並んでいる
        parent = rt.keyed_parent[name]
        R = quat.to_matrix(local[name])
        out[name] = R if parent is None else out[parent] @ R
    for name, G in base.items():
        out.setdefault(name, G)
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
    delta = np.zeros(J.shape[1])   # 変数の数は J の列の数（肩だけなら 3、ひじも曲げるなら 4）
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
    axis: np.ndarray = None       # (3,) カメラの奥行きの軸（内部座標の単位ベクトル）
    depth_weight: float = 1.0     # 腕を奥行き方向へ動かすコストの倍率（depth_cost²。1 なら向きを区別しない）

    @property
    def prefers_depth(self):
        return self.axis is not None and self.depth_weight < 1.0


@dataclass
class _Elbow:
    """1 フレーム分の自重する腕のひじ（ひじも曲げて離すとき）。"""
    b: np.ndarray     # (3,) ひじの曲げの軸（大域。推定の姿勢。+ に回すと曲げが深くなる）
    lo: float         # 曲げの補正の下限・上限 [rad]（まっすぐより先へ伸ばさない・曲げすぎない・elbow_max_deg）
    hi: float


def _pose(Y, state, elbow=None):
    """補正 state = (Q, a) を当てはめた腕の点 (4, 3)。Q: 肩 Y[0] まわりの回転（大域）/ a: ひじの曲げの補正 [rad]
    （肩で回したあとのひじの軸 Q b まわりに、前腕と手を回す）。"""
    Q, a = state
    A = Y[0]
    cur = A + (Y - A) @ Q.T
    if elbow is not None and a != 0.0:
        E = cur[1]
        cur[2:] = E + (cur[2:] - E) @ _rotation(a * (Q @ elbow.b)).T
    return cur


def _skew(r):
    """(3, 3) r × ・ の行列。"""
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


def _metric(Y, p):
    """(3, 3) 肩まわりの回転 ω のコスト ωᵀ M ω の M（ひじを動かさないときの奥行き優先）。

    |ω|² から、腕の点 k（ひじ・手首・手の先。肩からの向き r̂_k）が奥行き方向へ動く速さ a·(ω × r̂_k) = ω·(r̂_k × a)
    の 2 乗の平均の (1 − depth_weight) 倍を引く。奥行き方向へ腕を振る回転ほど安く、画像面内で振る回転・腕の軸まわりの
    ひねりは |ω|² のまま。M ≥ depth_weight·I なので正定値。
    """
    r = Y[1:] - Y[0]
    u = _cross(r, p.axis) / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), _EPS)
    return np.eye(3) - (1.0 - p.depth_weight) * (u.T @ u) / len(u)


def _point_metric(cur, bq, p):
    """(4, 4) 肩まわりの回転 ω とひじの曲げ a を合わせた x = (ω, a) のコスト xᵀ M x の M（ひじも曲げるとき）。

    腕の点 k（ひじ・手首・手の先）の動き Δ_k = G_k x を、画像面内は 1、奥行き方向は depth_weight の重みで測った
    2 乗の平均を、推定の姿勢で腕全体を肩まわりに回したときと同じ尺度（点の肩からの距離の 2 乗の平均）で割ったもの。
    肩とひじをどちらも使って、画像面内の動きが小さく（奥行きの動きは安く）なる組み合わせを選ぶ。どの点も動かない
    向き（まっすぐな腕の軸まわりのひねり）が残らないよう、ELBOW_REG·I を足す。
    cur: (4, 3) 補正後の腕の点 / bq: (3,) 補正後のひじの軸。
    """
    A, E = cur[0], cur[1]
    W = np.eye(3) - (1.0 - p.depth_weight) * np.outer(p.axis, p.axis)
    M = np.zeros((4, 4))
    scale = 0.0
    for k in (1, 2, 3):
        r = cur[k] - A
        G = np.zeros((3, 4))
        G[:, :3] = -_skew(r)                       # ω × r
        if k >= 2:
            G[:, 3] = _cross(bq, cur[k] - E)       # a (b × (x − E))
        M += G.T @ W @ G
        scale += float(r @ r)
    return M / max(scale, _EPS) + ELBOW_REG * np.eye(4)


def _inv_sqrt(M):
    """対称正定値行列 M の M^{-1/2}。"""
    lam, V = np.linalg.eigh(M)
    return (V / np.sqrt(lam)) @ V.T


def _jacobian(cp, n, A, cur, bq):
    """(9, 3 か 4) 組ごとに、x = (ω, a) で n·(最近点の動き) が増える速さ。bq が None なら肩だけ（3 列）。"""
    J = _cross(cp - A, n)
    if bq is None:
        return J
    ve = (cp - cur[1]) * (PAIR_Y >= 1)[:, None]   # 上腕の組はひじを曲げても動かない
    return np.concatenate([J, (_cross(ve, n) @ bq)[:, None]], axis=1)


def _depth_normals(cp, d, n, pen, R, M_inv, p, A, cur, bq):
    """奥行き優先: 重なっている組の離す向き n と重なり pen を選び直す。

    cp: (9, 3) 自重する腕の最近点 / d: (9, 3) 最近点どうしの差 / M_inv: コストの行列の逆行列。
    離す向き n' の候補は、n と奥行きの軸 a の間（n の画像面内の成分の向き ê と ±a を結ぶ 4 分の 1 の円）。n'·d ≥ R
    なら |d| ≥ R なので、どの n' で離しても重ならない。n'·d を必要なだけ増やすコストは (R − n'·d)² / (jᵀ M⁻¹ j)
    （j: n' に対する _jacobian の行）なので、それが最も小さい n' を選ぶ（n' = n は最近点を結ぶ向きで、必要な移動が
    最も短い。奥行き寄りの n' ほど必要な移動は長いが、奥行き方向の移動は安い）。前後の順は推定のまま（最近点が
    ほぼ同じ奥行きの組だけ、前後どちらへも離せる）。
    """
    a = p.axis
    over = pen > p.tol
    along = n @ a
    e = n - along[:, None] * a
    e_norm = np.linalg.norm(e, axis=-1)
    over &= e_norm > 1e-6          # すでに奥行きの軸に沿って離している組はそのまま
    if not over.any():
        return n, pen
    idx = np.flatnonzero(over)
    e_hat = e[idx] / e_norm[idx, None]
    lo = np.where(along[idx] > SAME_DEPTH, 0.0, -0.5 * np.pi)
    hi = np.where(along[idx] < -SAME_DEPTH, 0.0, 0.5 * np.pi)
    phi = lo[:, None] + (hi - lo)[:, None] * np.linspace(0.0, 1.0, DEPTH_SAMPLES)
    phi = np.concatenate([np.arctan2(along[idx], e_norm[idx])[:, None], phi], axis=1)   # 先頭は n
    cand = np.cos(phi)[..., None] * e_hat[:, None] + np.sin(phi)[..., None] * a       # (m, K, 3)
    need = R[idx, None] - np.einsum('mkc,mc->mk', cand, d[idx]) + p.tol
    j = _cross(cp[idx, None] - A, cand)
    if bq is not None:
        ve = (cp[idx] - cur[1]) * (PAIR_Y[idx] >= 1)[:, None]
        j = np.concatenate([j, (_cross(ve[:, None], cand) @ bq)[..., None]], axis=-1)
    ease = np.einsum('mkc,cd,mkd->mk', j, M_inv, j)
    # 回しても離せない向き（_solve_frame が使わない組になる）は選ばない
    lever = np.linalg.norm(j, axis=-1) > p.min_lever
    cost = np.where(lever, np.maximum(need, 0.0) ** 2 / np.maximum(ease, _EPS), np.inf)
    best = np.argmin(cost, axis=1)
    ok = np.isfinite(cost[np.arange(len(idx)), best])
    n, pen = n.copy(), pen.copy()
    sel = idx[ok]
    n[sel] = cand[ok, best[ok]]
    pen[sel] = need[ok, best[ok]] - p.tol
    return n, pen


def _solve_frame(Y, O, R, state, p, goal=None, ignore=None, elbow=None):
    """1 フレーム分。自重する腕 Y (4, 3) の補正 state = (Q, a)（肩 Y[0] まわりの回転 Q (3, 3) と、ひじの曲げ a [rad]）
    を、初期値 state から求める。elbow（_Elbow）が None ならひじは曲げない（a = 0 のまま）。

    R: (9,) 組ごとの離す距離（半径の和＋余裕）。goal (Q, a) を渡すと、重ならない範囲で state を goal に近づける
    （渡さなければ離すだけ）。ignore: (9,) 離さない組。
    """
    Q, a = state
    A = Y[0]
    rel = Y - A
    skip = np.zeros(len(R), bool) if ignore is None else ignore
    pulling = goal is not None
    for it in range(ITERATIONS + FINAL_ITERATIONS):
        # 推定の姿勢へ戻しながら離し、止まったら（または ITERATIONS 回で）離すだけにする。減衰のため、戻す力と
        # 釣り合った所ではわずかに重なりが残るので、それを離すだけで消す
        pulling = pulling and it < ITERATIONS
        cur = A + rel @ Q.T if elbow is None else _pose(Y, (Q, a), elbow)
        bq = None if elbow is None else Q @ elbow.b
        cp, cq = _pairs(cur, O)
        d = cp - cq
        dist = np.linalg.norm(d, axis=-1)
        n = d / np.maximum(dist, _EPS)[:, None]
        if (dist <= _EPS).any():   # 線分が交わっていて向きが決まらない
            n = np.where((dist > _EPS)[:, None], n, _perpendicular(cur, O))
        pen = R - dist
        C = M = None
        if elbow is not None or p.prefers_depth:
            # コスト xᵀ M x を |u|² にする変数 u = M^{1/2} x で解く（x = C u、C = M^{-1/2}）
            M = _metric(cur, p) if elbow is None else _point_metric(cur, bq, p)
            C = _inv_sqrt(M)
            if p.prefers_depth:
                n, pen = _depth_normals(cp, d, n, pen, R, C @ C, p, A, cur, bq)
        # 肩まわりの小さな回転 ω で、最近点は ω × v 動き、組は n·(ω × v) = (v × n)·ω だけ離れる（ひじの曲げ a では
        # 前腕・手の最近点が a (b × (c − E)) 動く）。回しても動かない組（肩のすぐ近く・肩から見た向きと n が平行）は離せない
        J = _jacobian(cp, n, A, cur, bq)
        use = (np.linalg.norm(J, axis=-1) > p.min_lever) & ~skip
        if elbow is None:
            target = _rotvec(goal[0] @ Q.T) if pulling else np.zeros(3)   # goal へ近づける回転
        else:
            target = (np.append(_rotvec(goal[0] @ Q.T), goal[1] - a) if pulling else np.zeros(4))
        if not (use & (pen > p.tol)).any() and np.linalg.norm(target) < 1e-9:
            break
        x = target
        if use.any():
            Ju = J[use]
            b = pen[use] + p.tol - Ju @ target
            x = target + (_constrained_step(Ju, b, p.damping) if C is None
                          else C @ _constrained_step(Ju @ C, b, p.damping))
            if elbow is not None and ((a <= elbow.lo + 1e-9 and x[3] < 0.0) or (a >= elbow.hi - 1e-9 and x[3] > 0.0)):
                # ひじが範囲の端にあって、さらに外へ曲げようとした: ひじを止めて、肩だけで離す一歩を求め直す
                b = pen[use] + p.tol - Ju[:, :3] @ target[:3]
                C3 = _inv_sqrt(M[:3, :3])
                x = np.append(target[:3] + C3 @ _constrained_step(Ju[:, :3] @ C3, b, p.damping), 0.0)
        step = float(np.linalg.norm(x)) if elbow is None else max(float(np.linalg.norm(x[:3])), abs(float(x[3])))
        if step < STOP_RAD:
            if pulling:
                pulling = False
                continue
            break
        x = x * min(1.0, np.deg2rad(MAX_STEP_DEG) / step)
        Q = _rotation(x[:3]) @ Q
        if _angle(Q) > p.max_angle:
            rv = _rotvec(Q)
            Q = _rotation(rv * (p.max_angle / np.linalg.norm(rv)))
        if elbow is not None:
            a = float(np.clip(a + x[3], elbow.lo, elbow.hi))
    return Q, a


def _overlapping(Y, O, R, state, tol, elbow=None):
    """(9,) 補正 state で重なっている組。"""
    if elbow is None:
        A = Y[0]
        cp, cq = _pairs(A + (Y - A) @ state[0].T, O)
    else:
        cp, cq = _pairs(_pose(Y, state, elbow), O)
    return R - np.linalg.norm(cp - cq, axis=-1) > tol


def _saturated(state, p):
    """肩の回転角が上限に達したか（ひじの曲げが範囲の端にあっても、肩で押しのけられるうちは上限とみなさない）。"""
    return _angle(state[0]) >= p.max_angle - 1e-6


def _toward_identity(local, step):
    """補正 local を単位回転（推定の姿勢）の向きへ最大 step [rad] 戻した回転。"""
    rv = _rotvec(local)
    a = float(np.linalg.norm(rv))
    return np.eye(3) if a <= step else _rotation(rv * (1.0 - step / a))


def _resolve_sequential(Y, O, R, Gp, p, elbows=None):
    """1 回目: フレーム順に、前のフレームの補正から始めて解く（押しのけ・戻り）。

    Gp: (T, 3, 3) 自重する腕の親（肩）の大域回転。肩の補正はこの座標系で、ひじの曲げはそのまま次のフレームへ持ち越し、
    推定の姿勢へは 1 フレームに p.return_step までしか戻さない（相手の腕が離れても、はね戻らない）。押しのけきれずに
    補正が上限に達しても重なっている組は、離れるまで扱わない（その間は腕どうしが少しずつ通り抜ける）。奥行きを優先して
    上限に達したときは、先に奥行きの優先を弱めて解き直す（_relax_depth）。elbows: フレームごとの _Elbow（None ならひじは
    曲げない）。
    戻り値: 肩の補正 (T, 3, 3)、ひじの曲げの補正 (T,) [rad]、扱わなかった組 (T, 9)、フレームごとに使った depth_weight (T,)。
    """
    T = len(Y)
    cp, cq = _pairs(Y, O)
    raw_pen = R - np.linalg.norm(cp - cq, axis=-1)   # 推定の姿勢のままでの重なり
    Q = np.tile(np.eye(3), (T, 1, 1))
    bend = np.zeros(T)
    ignored = np.zeros((T, len(R)), bool)
    weight = np.full(T, p.depth_weight)
    local = np.eye(3)                   # 前のフレームの肩の補正（肩の座標系）
    a_prev = 0.0                        # 前のフレームのひじの曲げの補正
    skip = np.zeros(len(R), bool)       # 扱わない組
    for t in range(T):
        elbow = None if elbows is None else elbows[t]
        # _angle は arccos なので 1e-8 rad 程度より小さい角を区別できない（丸め誤差で単位回転に戻りきらない）
        carried = _angle(local) > 1e-6 or abs(a_prev) > 1e-6
        if not carried and (raw_pen[t] <= 0.0).all():
            skip[:] = False
            continue
        deep = not carried and raw_pen[t].max() > DEEP_START * R.min()
        if elbow is not None:   # 前のフレームの曲げを、このフレームのひじの範囲に収めて持ち越す
            a_prev = float(np.clip(a_prev, min(elbow.lo, 0.0), max(elbow.hi, 0.0)))

        def solve(q):
            if deep:
                return _solve_deep_start(Y[t], O[t], R, Gp[t], q, skip, elbow)
            goal = (Gp[t] @ _toward_identity(local, q.return_step) @ Gp[t].T,
                    float(np.sign(a_prev) * max(abs(a_prev) - q.return_step, 0.0)))
            return _solve_frame(Y[t], O[t], R, (Gp[t] @ local @ Gp[t].T, a_prev), q, goal, skip, elbow)

        state, over, saturated = _attempt(solve, Y[t], O[t], R, p, elbow)
        if saturated and over.any() and p.prefers_depth:
            state, over, saturated, weight[t] = _relax_depth(solve, Y[t], O[t], R, p, (state, over, saturated),
                                                             elbow)
        Q[t], bend[t] = state
        if saturated:
            skip = skip | over
        skip = skip & over
        ignored[t] = skip
        local = Gp[t].T @ Q[t] @ Gp[t]
        a_prev = bend[t]
    return Q, bend, ignored, weight


def _attempt(solve, Y, O, R, q, elbow=None):
    """solve(q) で解いた補正と、重なっている組 (9,) と、補正が上限に達したか。"""
    state = solve(q)
    return state, _overlapping(Y, O, R, state, q.tol, elbow), _saturated(state, q)


def _relax_depth(solve, Y, O, R, p, first, elbow=None):
    """奥行きを優先して補正が上限に達しても離れないフレーム: depth_cost を 1（向きを区別しない）へ向けて、離れる
    ところまで弱める（二分法。奥行きの優先をなるべく残し、画像面内の動きを必要な分だけ増やす）。

    solve(q): _Params q で解いた補正 / first: p で解いた (補正, 重なっている組, 上限に達したか)。
    戻り値: (補正, 重なっている組, 上限に達したか, 使った depth_weight)。1 でも離れなければ、重なっている組の少ない方。
    """
    def failed(res):
        return res[2] and res[1].any()

    lo, hi = float(np.sqrt(p.depth_weight)), 1.0
    iso = _attempt(solve, Y, O, R, replace(p, depth_weight=1.0), elbow)
    if failed(iso):
        if iso[1].sum() < first[1].sum():
            return (*iso, 1.0)
        return (*first, p.depth_weight)
    best = (*iso, 1.0)
    for _ in range(RELAX_STEPS):
        mid = 0.5 * (lo + hi)
        res = _attempt(solve, Y, O, R, replace(p, depth_weight=mid * mid), elbow)
        if failed(res):
            lo = mid
        else:
            hi, best = mid, (*res, mid * mid)
    return best


def _solve_deep_start(Y, O, R, G, p, ignore, elbow=None):
    """持ち越す補正が無いのに深く重なっているフレーム（動画の最初から腕を組んでいる・急に重なった等）。

    推定の姿勢から離すと、離す向きによっては腕が相手の腕に絡んで大きく回す解に落ちるので、肩の座標系 G の
    6 方向へ MAX_STEP_DEG 回した所からも解き、重なりが残らない解のうち補正の大きさが最も小さいものを選ぶ
    （奥行き優先・ひじも曲げるときは、推定の姿勢でのコストの行列で重み付けした大きさ）。
    """
    best, best_score = None, np.inf
    starts = [np.eye(3)] + [_rotation(G @ (sign * np.deg2rad(MAX_STEP_DEG) * np.eye(3)[axis]))
                            for axis in range(3) for sign in (1.0, -1.0)]
    if elbow is not None:
        M = _point_metric(Y, elbow.b, p)
    else:
        M = _metric(Y, p) if p.prefers_depth else None
    for Q0 in starts:
        state = _solve_frame(Y, O, R, (Q0, 0.0), p, (np.eye(3), 0.0), ignore, elbow)
        cp, cq = _pairs(Y[0] + (Y - Y[0]) @ state[0].T if elbow is None else _pose(Y, state, elbow), O)
        pen = np.where(ignore, -np.inf, R - np.linalg.norm(cp - cq, axis=-1))
        residual = max(0.0, float(pen.max()) - p.tol)
        if M is None:
            size = _angle(state[0])
        else:
            x = _rotvec(state[0]) if elbow is None else np.append(_rotvec(state[0]), state[1])
            size = float(np.sqrt(x @ M @ x))
        score = size + 100.0 * residual / p.min_lever   # 重なりが残る解は選ばない
        if score < best_score:
            best, best_score = state, score
    return best


def _pose_all(Y, Q, bend, b=None):
    """(T, 4, 3) 全フレームに補正（肩の回転 Q (T, 3, 3)・ひじの曲げ bend (T,)・ひじの軸 b (T, 3)）を当てはめた腕の点。"""
    A = Y[:, :1]
    cur = A + np.einsum('tab,tkb->tka', Q, Y - A)
    if b is not None:
        E = cur[:, 1:2]
        Re = quat.to_matrix(quat.from_rotvec(bend[:, None] * np.einsum('tab,tb->ta', Q, b)))
        cur[:, 2:] = E + np.einsum('tab,tkb->tka', Re, cur[:, 2:] - E)
    return cur


def _resolve_residual(Y, O, R, Q, bend, ignored, p, weight, elbows=None):
    """2 回目: ならした補正（肩 Q (T, 3, 3)・ひじ bend (T,)）で重なるフレームだけ、重ならないところまで回す（戻さない）。
    ignored: (T, 9) 1 回目で扱わなかった組 / weight: (T,) 1 回目で使った depth_weight。"""
    b = None if elbows is None else np.array([e.b for e in elbows])
    cp, cq = _pairs(_pose_all(Y, Q, bend, b), O)
    touching = (np.linalg.norm(cp - cq, axis=-1) < R) & ~ignored
    Q, bend = Q.copy(), bend.copy()
    for t in np.flatnonzero(touching.any(1)):
        q = p if weight[t] == p.depth_weight else replace(p, depth_weight=float(weight[t]))
        elbow = None if elbows is None else elbows[t]
        a = 0.0 if elbow is None else float(np.clip(bend[t], min(elbow.lo, 0.0), max(elbow.hi, 0.0)))
        Q[t], bend[t] = _solve_frame(Y[t], O[t], R, (Q[t], a), q, ignore=ignored[t], elbow=elbow)
    return Q, bend


def _shift(Y, Y_after, axis):
    """(T,) × 2 補正で腕の点（ひじ・手首・手の先）が動いた距離の最大: 画像面内（axis に垂直）と奥行き（axis 沿い）。"""
    move = Y_after[:, 1:] - Y[:, 1:]
    depth = move @ axis
    image = np.linalg.norm(move - depth[..., None] * axis, axis=-1)
    return image.max(-1), np.abs(depth).max(-1)


def elbow_axis_local(skel, side):
    """ひじの曲げの軸（腕ボーンの初期姿勢の座標系。前腕の初期の向き × 正面。+ に回すと前腕が正面へ曲がる）。
    ステージ9b（contacts.py）と同じ。"""
    s = SIDES[side]
    d = skel.internal(s + '手首') - skel.internal(s + 'ひじ')
    b = np.cross(d, [0.0, 0.0, 1.0])
    return b / max(float(np.linalg.norm(b)), _EPS)


def _flexion(Y, b):
    """(T,) ひじの曲げ角 [rad]（上腕の向きから前腕の向きへの、ひじの軸 b まわりの角。0 = まっすぐ、+ = 曲げ）。"""
    u = Y[:, 1] - Y[:, 0]
    f = Y[:, 2] - Y[:, 1]
    return np.arctan2(np.einsum('tc,tc->t', b, _cross(u, f)), np.einsum('tc,tc->t', u, f))


def _elbows(Y, b, cfg):
    """フレームごとの _Elbow。補正の範囲は ±elbow_max_deg のうち、曲げ角が ELBOW_FLEX_DEG の範囲に収まる所
    （推定の姿勢がすでに範囲の外なら、それ以上外へは動かさない）。"""
    limit = np.deg2rad(float(cfg.elbow_max_deg))
    flex = _flexion(Y, b)
    lo = np.minimum(0.0, np.maximum(-limit, np.deg2rad(ELBOW_FLEX_DEG[0]) - flex))
    hi = np.maximum(0.0, np.minimum(limit, np.deg2rad(ELBOW_FLEX_DEG[1]) - flex))
    return [_Elbow(b[t], float(lo[t]), float(hi[t])) for t in range(len(Y))]


def resolve_arm_collisions(skel, rt, glob_rot, local, cfg, unit, fps, depth_axis=None):
    """ステージ9a。local（Retargeter.local_quats の結果）の自重する側の腕（腕・ひじ）ボーンの回転を直した dict と、
    ArmCollisionResult を返す。glob_rot: SMPL の大域回転 (T, J, 3, 3)。depth_axis: カメラの奥行きの軸
    （内部座標。depth.depth_axis。None なら Z 軸）。"""
    mode = str(cfg.mode)
    if mode not in MODES:
        raise ValueError(f'arm_collision.mode は {" / ".join(MODES)} のいずれかです: {mode}')
    depth_cost = float(cfg.depth_cost)
    if not 0.0 < depth_cost <= 1.0:
        raise ValueError(f'arm_collision.depth_cost は 0 より大きく 1 以下で指定してください: {cfg.depth_cost}')
    axis = np.array([0.0, 0.0, 1.0]) if depth_axis is None else np.asarray(depth_axis, np.float64)
    axis = axis / np.linalg.norm(axis)
    T = len(glob_rot)
    model = arm_model(skel, cfg, unit)
    # local から求める（ステージ9h で直した腕・ひじの回転も含めて判定する）
    glob = globals_from_local(rt, glob_rot, local)
    pts = arm_points(skel, glob, model, T)
    before = overlap_depth(pts[:, 0], pts[:, 1], model.radius[0], model.radius[1])
    if mode == 'none' or T == 0:
        return local, ArmCollisionResult(mode, -1, model.radius, model.source, np.zeros(T),
                                         before, before.copy(), np.zeros(T), np.zeros(T), depth_cost,
                                         np.full(T, depth_cost), np.zeros(T), bool(cfg.elbow))

    y = SIDES.index('左') if mode == 'left' else SIDES.index('右')
    o = 1 - y
    arm, elbow_bone = SIDES[y] + '腕', SIDES[y] + 'ひじ'
    Y, O = pts[:, y], pts[:, o]
    R = model.radius[y][PAIR_Y] + model.radius[o][PAIR_O] + float(cfg.margin_m) * unit
    p = _Params(np.deg2rad(float(cfg.max_deg)), MIN_LEVER_M * unit, TOLERANCE_M * unit,
                (DAMPING_M * unit) ** 2, np.deg2rad(float(cfg.return_deg_per_s)) / fps,
                axis, depth_cost ** 2)
    use_elbow = bool(cfg.elbow) and elbow_bone in local and elbow_bone in glob
    b = None
    elbows = None
    if use_elbow:
        b = glob[arm] @ elbow_axis_local(skel, y)          # (T, 3) ひじの軸（大域。推定の姿勢）
        elbows = _elbows(Y, b, cfg)
    # 補正は肩の親（キーを打つ祖先 = 肩）の座標系で持ち越し・ならす（体が回っても向きが変わらないように）
    parent = rt.keyed_parent[arm]
    Gp = glob[parent] if parent is not None else np.tile(np.eye(3), (T, 1, 1))
    Gp_t = np.swapaxes(Gp, -1, -2)
    Q, bend, ignored, weight = _resolve_sequential(Y, O, R, Gp, p, elbows)
    sigma = float(cfg.smooth_sec) * fps
    rv = quat.to_rotvec(quat.from_matrix(Gp_t @ Q @ Gp))
    rv = filters.gaussian_time(rv, sigma)
    if use_elbow:
        bend = filters.gaussian_time(bend, sigma)
    Q, bend = _resolve_residual(Y, O, R, Gp @ quat.to_matrix(quat.from_rotvec(rv)) @ Gp_t, bend, ignored, p,
                                weight, elbows)

    q_local = quat.from_matrix(Gp_t @ Q @ Gp)
    angle = np.rad2deg(np.linalg.norm(quat.to_rotvec(q_local), axis=-1))
    elbow_deg = np.rad2deg(np.abs(bend))
    if angle.max(initial=0.0) > 1e-9 or elbow_deg.max(initial=0.0) > 1e-9:
        # 補正が無ければ回転はそのまま（丸め誤差も入れない）
        local = dict(local)
        local[arm] = quat.make_continuous(quat.mul(q_local, local[arm]))
        if use_elbow and elbow_deg.max(initial=0.0) > 1e-9:
            # ひじの軸 Q b まわりの曲げを、ひじの親（キーを打つ祖先。補正後は Q·P）の座標系で表す: (Q P)ᵀ Q b = Pᵀ b
            ep = rt.keyed_parent[elbow_bone]
            P = glob[ep] if ep is not None else np.tile(np.eye(3), (T, 1, 1))
            b_parent = np.einsum('tba,tb->ta', P, b)
            q_elbow = quat.from_rotvec(bend[:, None] * b_parent)
            local[elbow_bone] = quat.make_continuous(quat.mul(q_elbow, local[elbow_bone]))
    Y_after = _pose_all(Y, Q, bend, b)
    after = overlap_depth(Y_after, O, model.radius[y], model.radius[o])
    return local, ArmCollisionResult(mode, y, model.radius, model.source, angle, before, after,
                                     *_shift(Y, Y_after, axis), depth_cost, np.sqrt(weight), elbow_deg,
                                     use_elbow)
