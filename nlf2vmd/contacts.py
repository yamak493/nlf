"""ステージ9b: 腕・手のひら・指先と、体・相手の腕の接触の解決（指先までの当たり判定）。

ステージ9a（腕どうしの貫通の防止）は、左右の腕を上腕・前腕・手の 3 本のカプセルで表し、自重する腕を肩まわりに
回すだけだった。手は 1 本の棒で、指の形も体（胴・頭・脚・スカート）との重なりも見ていない。ここでは

* **腕の当たり判定**: 上腕・前腕・手のひら（手首 → 人指１、手首 → 小指１、人指１ → 小指１ の 3 本）・5 本の指の
  各節（付け根 → 第 2 → 第 3 → 指先）のカプセル（片腕で最大 20 本）。指の形は指ボーンのローカル回転
  （hands.py のキーを毎フレームに展開したもの。無ければ初期姿勢）で FK して求める。太さは PMX のメッシュ
  （ウェイトが最も大きいボーンが、その指ボーン・手首の頂点）から測り、指先は 〇指３先 のボーン → 表示先 →
  頂点の順に探す。指ボーンが無いモデルは、ステージ9a と同じ 1 本の棒の手にする
* **体の当たり判定**: PMX の剛体（モデルの作者が置いた当たり判定。ボーン追従の剛体と、スカート・胸などの物理の剛体）。
  剛体が無ければメッシュの頂点をボーンごとに箱で包んだもの、それも無ければ標準の体格の形。どれもカプセルにする
  （箱は、薄い向きの厚みを半径にしたカプセル 2 本）。腕・肩の剛体と、髪・脚の物理の剛体は使わない。
  脚の位置は MMD では足ＩＫで決まる（キーを打つ回転が無い）ので、SMPL の股関節・膝の向きから求める
* **組**: 腕の各カプセル × 体の各カプセル（両腕とも）、自重する腕の各カプセル × 相手の腕の各カプセル
  （arm_collision.mode。none なら腕どうしは見ない）。同じ腕の中（指と手のひら等）は見ない。
  初期姿勢（A ポーズ）ですでに重なっている組（脇の下の上腕と胸など）は扱わず、上腕の肩側 upper_arm_skip の
  割合は体との判定から外す
* **解き方**: ステージ9a と同じく、フレームの順に、前のフレームの補正から始めて、推定の姿勢へ戻しながら
  （1 フレームに return_deg_per_s まで）どの組も重ならない最小の補正を、線形化した条件と減衰付きの
  アクティブセット法で求める。補正する関節は、肩（腕ボーン。3 自由度）・ひじ（ひじボーン。曲げの軸まわりの
  1 自由度）・手首（手首ボーン。3 自由度）。関節ごとの回転のコストを「係数 × (関節から先の長さ × 角度)²」にして、
  先の関節ほど動かしやすくする（指先が胸に触れただけで腕全体が回らない）。腕どうしの組はステージ9a と同じく
  自重する腕の肩だけで離す（ひじの曲げ・手首の向きは変えない）。補正を時間方向にならしてから、ならして浅くなった
  重なりをもう一度離す
* **触れるのは正しい**: 表面どうしを margin_m まで離すだけで、それ以上は離さない（胸に手を当てる・手を合わせる動きは
  触れたまま残る）。スカート・胸の物理の剛体は、重なりの soft_ratio の割合だけ離す（残りは MMD の物理に任せる）
"""
from dataclasses import dataclass, field

import numpy as np

from . import filters, quat
from .arm_collision import (OVERLAP_TOL_M, _angle, _cross, _rotation, _rotvec, arm_model,
                            bone_positions, closest_points)
from .skeleton import SIDES

FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
PARTS = ('upper', 'fore', 'palm') + FINGERS
PART_LABELS = dict(upper='上腕', fore='前腕', palm='手のひら', thumb='親指', index='人差し指',
                   middle='中指', ring='薬指', pinky='小指')
BODY_BONES = ('下半身', '上半身', '上半身2', '首', '頭', '左足', '右足', '左ひざ', '右ひざ')
ITERATIONS = 30
FINAL_ITERATIONS = 10
MAX_STEP_DEG = 20.0
STOP_RAD = 1e-4
MIN_LEVER_M = 0.01
TOLERANCE_M = 1e-4
DAMPING = 0.05
MIN_VERTICES = 20
_EPS = 1e-9
# 剛体もメッシュも無いときの体の形。モデルの体格に合わせて、肩幅（左右の腕ボーンの間）と腰幅（左右の足ボーンの間）
# に対する割合で決める（成人の体の寸法の比から）:
#   胴: 上半身 → 首 のカプセルを左右に ±0.15 × 肩幅 ずらした 2 本、半径 0.24 × 肩幅（幅 0.78 × 肩幅・厚み 0.48 × 肩幅。
#       下ろした腕の内側は「肩幅の半分 − 腕の太さ」の所にあるので、胴の幅はそれより少し狭くする）
#   腰: 左右の足ボーンを結ぶカプセル、半径 0.9 × 腰幅の半分
#   頭: 頭ボーンの 0.25 × 肩幅 上の球、半径 0.33 × 肩幅
#   太もも: 足 → ひざ、半径 0.8 × 腰幅の半分
@dataclass
class Capsule:
    a: np.ndarray          # (3,) 始点（初期姿勢・内部座標）
    b: np.ndarray          # (3,) 終点
    radius: float
    bone: str              # この形が付いているボーン（FK で一緒に動く）
    part: int = -1         # 腕: PARTS の番号 / 体: -1
    level: int = -1        # 腕: 0 = 上腕（肩の補正で動く）/ 1 = 前腕（＋ひじ）/ 2 = 手（＋手首）。体: -1
    weight: float = 1.0    # 体: 離す割合（物理の剛体は soft_ratio）
    name: str = ''


# ---- 腕の当たり判定 ----
def _owners(skel, candidates):
    """(頂点ごとの持ち主のボーン名（candidates のうち、ウェイトが最も大きいボーン自身か最も近い祖先。無ければ ''）,
    頂点の位置（内部座標）)。メッシュが無ければ (None, None)。"""
    if skel.mesh_points is None or len(skel.mesh_points) == 0:
        return None, None
    names, inverse = np.unique(skel.mesh_bones, return_inverse=True)
    own = np.array([(n if n in candidates else skel.nearest_ancestor(n, candidates)) or ''
                    if skel.has(n) else '' for n in names])
    return own[inverse.reshape(-1)], skel.mesh_points * [1.0, 1.0, -1.0]


def _segment_distance(points, a, b):
    ab = b - a
    t = np.clip((points - a) @ ab / max(float(ab @ ab), _EPS), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=-1)


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _finger_tip(skel, chain, base, pts):
    """指の最後のボーンの先: 〇〇先 のボーン → 表示先 → 頂点（骨の向きに測った 95 パーセンタイル）→ 最後の節の 0.8 倍。"""
    last = chain[-1]
    if skel.has(last + '先'):
        return skel.internal(last + '先')
    tail = skel.tail_internal(last)
    if tail is not None and np.linalg.norm(tail - skel.internal(last)) > 1e-6:
        return tail
    prev = skel.internal(chain[-2]) if len(chain) > 1 else base
    d = skel.internal(last) - prev
    length = 0.8 * float(np.linalg.norm(d))
    if pts is not None and len(pts) >= 5:
        length = max(float(np.percentile((pts - skel.internal(last)) @ _unit(d), 95.0)), 0.3 * length)
    return skel.internal(last) + _unit(d) * length


def arm_capsules(skel, side, cfg, arm_cfg, finger_bones, unit):
    """片腕のカプセルのリスト。cfg: 設定 contacts / arm_cfg: 設定 arm_collision（上腕・前腕・手の太さ）/
    finger_bones: 設定 hands.bones（指ごとのボーン名。左右は付けない）。"""
    s = SIDES[side]
    S = skel.internal
    model = arm_model(skel, arm_cfg, unit)
    r = model.radius[side]
    caps = [Capsule(S(s + '腕'), S(s + 'ひじ'), r[0], s + '腕', 0, 0, name=s + '上腕'),
            Capsule(S(s + 'ひじ'), S(s + '手首'), r[1], s + 'ひじ', 1, 1, name=s + '前腕')]
    wrist = S(s + '手首')
    chains = {f: [s + n for n in finger_bones[f] if skel.has(s + n)] for f in FINGERS}
    idx1, pky1 = s + finger_bones['index'][0], s + finger_bones['pinky'][0]
    if not (cfg.fingers and skel.has(idx1) and skel.has(pky1)):
        caps.append(Capsule(wrist, model.tip[side], r[2], s + '手首', 2, 2, name=s + '手'))
    else:
        finger_names = {n for c in chains.values() for n in c}
        owner, points = _owners(skel, finger_names | {s + '手首'})
        palm = [(wrist, S(idx1)), (wrist, S(pky1)), (S(idx1), S(pky1))]
        radius = float(cfg.palm_radius_m) * unit
        if owner is not None:
            pts = points[owner == s + '手首']
            if len(pts) >= MIN_VERTICES:
                d = np.min([_segment_distance(pts, a, b) for a, b in palm], axis=0)
                radius = float(np.percentile(d, float(cfg.radius_percentile)))
        for a, b in palm:
            caps.append(Capsule(a, b, radius, s + '手首', 2, 2, name=s + '手のひら'))
        for f, finger in enumerate(FINGERS):
            chain = chains[finger]
            if not chain:
                continue
            last_pts = None if owner is None else points[owner == chain[-1]]
            ends = [S(n) for n in chain] + [_finger_tip(skel, chain, wrist, last_pts)]
            for i, name in enumerate(chain):
                radius = float(cfg.finger_radius_m[f]) * unit
                if owner is not None:
                    pts = points[owner == name]
                    if len(pts) >= MIN_VERTICES // 2:
                        radius = float(np.percentile(_segment_distance(pts, ends[i], ends[i + 1]),
                                                     float(cfg.radius_percentile)))
                caps.append(Capsule(ends[i], ends[i + 1], radius, name, 3 + f, 2,
                                    name=s + PART_LABELS[finger]))
    for c in caps:
        c.radius = max(c.radius * float(cfg.radius_scale), 1e-6)
    return caps


# ---- 体の当たり判定 ----
def _rigid_rotation(rot):
    """PMX の剛体の回転（ラジアン、x・y・z。Z → X → Y の順に回す）→ 内部座標の回転行列。"""
    rx, ry, rz = (float(v) for v in rot)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    flip = np.diag([1.0, 1.0, -1.0])               # MMD 座標（左手系）→ 内部座標（右手系）
    return flip @ (Ry @ Rx @ Rz) @ flip


def _box_capsules(center, axes, half, bone, weight, name):
    """中心・軸（列）・半分の長さの箱を、カプセル 1〜2 本で包む。いちばん長い向きを線分にし、残りの 2 方向の
    うち薄いほうを半径にする（2 方向の差が大きければ、厚いほうへずらした 2 本）。"""
    order = np.argsort(half)[::-1]
    L, M, Sh = half[order]
    long_ax, mid_ax = axes[:, order[0]], axes[:, order[1]]
    if M <= 1.3 * Sh:
        r = 0.5 * (M + Sh)
        h = max(L - r, 0.0)
        return [Capsule(center - h * long_ax, center + h * long_ax, r, bone, weight=weight, name=name)]
    h = max(L - Sh, 0.0)
    out = []
    for sign in (1.0, -1.0):
        c = center + sign * (M - Sh) * mid_ax
        out.append(Capsule(c - h * long_ax, c + h * long_ax, Sh, bone, weight=weight, name=name))
    return out


def _arm_bone(skel, name):
    return any(skel.is_descendant(name, s + '肩') or skel.is_descendant(name, s + '腕') for s in SIDES)


def _rigid_capsules(skel, cfg, unit):
    caps = []
    margin = float(cfg.cloth_margin_m) * unit
    soft = float(cfg.soft_ratio)
    for rb in skel.rigid_bodies or []:
        bone = rb['bone']
        if bone is None or not skel.has(bone) or _arm_bone(skel, bone):
            continue
        if not (skel.is_descendant(bone, '上半身') or skel.is_descendant(bone, '下半身')):
            continue
        weight = 1.0
        if rb['mode'] != 0:                        # 物理演算の剛体
            if skel.is_descendant(bone, '首') or any(skel.is_descendant(bone, s + '足') for s in SIDES):
                continue                           # 髪・リボン・脚の飾りは動くので使わない
            if skel.is_descendant(bone, '下半身'):   # スカート
                if cfg.skirt == 'none':
                    continue
                weight = soft if cfg.skirt == 'soft' else 1.0
            else:                                  # 胸など
                weight = soft
        center = rb['position'] * [1.0, 1.0, -1.0]
        R = _rigid_rotation(rb['rotation'])
        size = np.asarray(rb['size'], np.float64)
        if rb['shape'] == 0:
            caps.append(Capsule(center, center, size[0] + margin, bone, weight=weight, name=rb['name']))
        elif rb['shape'] == 2:
            h = 0.5 * size[1]
            caps.append(Capsule(center - h * R[:, 1], center + h * R[:, 1], size[0] + margin, bone,
                                weight=weight, name=rb['name']))
        else:
            caps += _box_capsules(center, R, size + margin, bone, weight, rb['name'])
    return caps


def _mesh_capsules(skel, cfg, unit):
    """メッシュの頂点を、体のボーンごとに主軸の向きの箱（95 パーセンタイル）で包む。腕・足首から先の頂点は使わない。"""
    body = {b for b in BODY_BONES if skel.has(b)}
    owner, points = _owners(skel, body)
    if owner is None:
        return []
    names, inverse = np.unique(skel.mesh_bones, return_inverse=True)
    excluded = np.array([skel.has(n) and (_arm_bone(skel, n) or any(
        skel.has(s + '足首') and skel.is_descendant(n, s + '足首') for s in SIDES)) for n in names])
    owner = np.where(excluded[inverse.reshape(-1)], '', owner)
    caps = []
    for bone in sorted(body):
        pts = points[owner == bone]
        if len(pts) < 30:
            continue
        c = pts.mean(0)
        _, vecs = np.linalg.eigh(np.cov((pts - c).T))
        half = np.percentile(np.abs((pts - c) @ vecs), 95.0, axis=0)
        caps += _box_capsules(c, vecs, np.maximum(half, 1e-6), bone, 1.0, bone)
    return caps


def _fallback_capsules(skel, unit, cfg):
    margin = float(cfg.cloth_margin_m) * unit
    S = skel.internal
    shoulder = float(np.linalg.norm(S('左腕') - S('右腕')))
    hip = 0.5 * float(np.linalg.norm(S('左足') - S('右足')))
    chest = '上半身2' if skel.has('上半身2') else '上半身'
    lateral = np.array([0.15 * shoulder, 0.0, 0.0])
    caps = [Capsule(S('上半身') + d, S('首') + d, 0.24 * shoulder + margin, chest, name='胴')
            for d in (lateral, -lateral)]
    caps.append(Capsule(S('左足'), S('右足'), 0.9 * hip + margin, '下半身', name='腰'))
    head = S('頭') + [0.0, 0.25 * shoulder, 0.0]
    caps.append(Capsule(head, head, 0.33 * shoulder + margin, '頭', name='頭'))
    for s in SIDES:
        if skel.has(s + 'ひざ'):
            caps.append(Capsule(S(s + '足'), S(s + 'ひざ'), 0.8 * hip + margin, s + '足', name=s + '太もも'))
    return caps


def body_capsules(skel, cfg, unit):
    """(体のカプセルのリスト, 'rigid' | 'mesh' | 'config' | 'none')。"""
    src = str(cfg.body_source)
    if src not in ('auto', 'rigid', 'mesh', 'config', 'none'):
        raise ValueError(f'contacts.body_source は auto / rigid / mesh / config / none のいずれかです: {src}')
    if src == 'none':
        return [], 'none'
    if src in ('auto', 'rigid'):
        caps = _rigid_capsules(skel, cfg, unit)
        if caps or src == 'rigid':
            return caps, 'rigid' if caps else 'none'
    if src in ('auto', 'mesh'):
        caps = _mesh_capsules(skel, cfg, unit)
        if caps or src == 'mesh':
            return caps, 'mesh' if caps else 'none'
    return _fallback_capsules(skel, unit, cfg), 'config'


# ---- FK ----
def leg_globals(skel, glob_rot, smpl_rest):
    """MMD の脚（足・ひざ）の大域回転 {名前: (T, 3, 3)}。MMD の脚は足ＩＫで決まり回転のキーが無いので、SMPL の
    股関節・膝の大域回転に、初期姿勢の骨の向きを合わせる最小回転を掛けたもので近似する。"""
    J = np.asarray(smpl_rest, np.float64)
    out = {}
    for side, s in enumerate(SIDES):
        for bone, child, j, jc in ((s + '足', s + 'ひざ', 1 + side, 4 + side),
                                   (s + 'ひざ', s + '足首', 4 + side, 7 + side)):
            if skel.has(bone) and skel.has(child):
                C = quat.to_matrix(quat.from_two_vectors(skel.internal(child) - skel.internal(bone),
                                                         J[jc] - J[j]))
                out[bone] = glob_rot[:, j] @ C
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


def _effective(skel, glob, name, T):
    n, seen = name, set()
    while n is not None and n not in seen:
        if n in glob:
            return glob[n]
        seen.add(n)
        n = skel.parents.get(n)
    return np.tile(np.eye(3), (T, 1, 1))


def _attach(skel, glob, caps, T):
    """(T, n, 2, 3) カプセルの端点の位置（センターの移動は含めない）。"""
    if not caps:
        return np.zeros((T, 0, 2, 3))
    bones = sorted({c.bone for c in caps})
    pos = bone_positions(skel, glob, bones, T)
    rot = {b: _effective(skel, glob, b, T) for b in bones}
    out = np.empty((T, len(caps), 2, 3))
    for i, c in enumerate(caps):
        base = skel.internal(c.bone)
        for k, p in enumerate((c.a, c.b)):
            out[:, i, k] = pos[c.bone] + rot[c.bone] @ (p - base)
    return out


# ---- 解く ----
@dataclass
class _Arm:
    """1 フレーム分の片腕の姿勢（補正前）と、補正の状態を当てはめる関数。"""
    S: np.ndarray
    E: np.ndarray
    W: np.ndarray
    Gp: np.ndarray
    G_arm: np.ndarray
    G_elb: np.ndarray
    ends: np.ndarray       # (n, 2, 3)
    level: np.ndarray      # (n,)
    b_local: np.ndarray    # ひじの曲げの軸（腕ボーンの初期姿勢の座標系）

    def apply(self, state):
        """state = (As, alpha, Aw) → (補正後の端点 (n, 2, 3), E', W', ひじの軸（大域）, G_elb')。"""
        As, alpha, Aw = state
        Qs = self.Gp @ As @ self.Gp.T
        E2 = self.S + Qs @ (self.E - self.S)
        b_g = Qs @ self.G_arm @ self.b_local
        Re = _rotation(alpha * b_g)
        ReQs = Re @ Qs
        W2 = E2 + ReQs @ (self.W - self.E)
        G_elb2 = ReQs @ self.G_elb
        Qw = G_elb2 @ Aw @ G_elb2.T
        out = np.empty_like(self.ends)
        l0, l1, l2 = self.level == 0, self.level == 1, self.level == 2
        out[l0] = self.S + (self.ends[l0] - self.S) @ Qs.T
        out[l1] = E2 + (self.ends[l1] - self.E) @ ReQs.T
        out[l2] = W2 + (self.ends[l2] - self.W) @ (Qw @ ReQs).T
        return out, E2, W2, b_g, G_elb2


@dataclass
class _Pairs:
    kind: np.ndarray       # 0 = 体 / 1 = 相手の腕
    side: np.ndarray       # 動かす腕
    i: np.ndarray          # 動かす腕のカプセル
    j: np.ndarray          # 体のカプセル / 相手の腕のカプセル
    R: np.ndarray          # 離す距離（半径の和 + margin）
    w: np.ndarray          # 離す割合
    level: np.ndarray
    trim: np.ndarray       # 動かす腕のカプセルの始点側を外す割合（体との組の上腕: upper_arm_skip）


@dataclass
class _Params:
    max_angle: np.ndarray    # (3,) 肩・ひじ・手首 [rad]
    cost: np.ndarray         # (2, 3) 腕ごと・関節ごとのコスト（係数 × 先の長さ²）
    min_lever: float
    tol: float
    return_step: float


def _identity_state():
    return (np.eye(3), 0.0, np.eye(3))


def _pair_geometry(arms, states, body, pairs):
    """最近点の組と、補正後の腕の情報。"""
    applied = [a.apply(s) for a, s in zip(arms, states)]
    P0 = np.empty((len(pairs.i), 2, 3))
    Q0 = np.empty_like(P0)
    for s in range(2):
        m = pairs.side == s
        ends = applied[s][0]
        P0[m] = ends[pairs.i[m]]
        mb = m & (pairs.kind == 0)
        ma = m & (pairs.kind == 1)
        Q0[mb] = body[pairs.j[mb]]
        Q0[ma] = applied[1 - s][0][pairs.j[ma]]
    P0[:, 0] += pairs.trim[:, None] * (P0[:, 1] - P0[:, 0])
    cp, cq = closest_points(P0[:, 0], P0[:, 1], Q0[:, 0], Q0[:, 1])
    return cp, cq, applied, P0, Q0


def _penetration(arms, states, body, pairs):
    cp, cq, _, _, _ = _pair_geometry(arms, states, body, pairs)
    return (pairs.R - np.linalg.norm(cp - cq, axis=-1)) * pairs.w


def _solve_frame(arms, body, pairs, states, p, goal=None, skip=None):
    """1 フレーム分。両腕の補正 states = [(As, alpha, Aw)] * 2 を初期値から求める。"""
    states = [tuple(np.copy(x) if isinstance(x, np.ndarray) else x for x in st) for st in states]
    skip = np.zeros(len(pairs.i), bool) if skip is None else skip
    pulling = goal is not None
    scale = 1.0 / np.sqrt(np.repeat(p.cost, [3, 1, 3], axis=1).reshape(-1))   # (14,)
    for it in range(ITERATIONS + FINAL_ITERATIONS):
        pulling = pulling and it < ITERATIONS
        cp, cq, applied, P0, Q0 = _pair_geometry(arms, states, body, pairs)
        d = cp - cq
        dist = np.linalg.norm(d, axis=-1)
        n = d / np.maximum(dist, _EPS)[:, None]
        flat = dist <= _EPS
        if flat.any():
            perp = _cross(P0[flat, 1] - P0[flat, 0], Q0[flat, 1] - Q0[flat, 0])
            norm = np.linalg.norm(perp, axis=-1, keepdims=True)
            n[flat] = np.where(norm > _EPS, perp / np.maximum(norm, _EPS), [0.0, 0.0, 1.0])
        pen = (pairs.R - dist) * pairs.w
        J = np.zeros((len(pen), 14))
        for s in range(2):
            m = pairs.side == s
            if not m.any():
                continue
            _, E2, W2, b_g, _ = applied[s]
            c, nn = cp[m], n[m]
            o = 7 * s
            J[m, o:o + 3] = _cross(c - arms[s].S, nn)
            body_rows = pairs.kind[m] == 0
            lv = pairs.level[m]
            je = (_cross(c - E2, nn) @ b_g) * ((lv >= 1) & body_rows)
            J[m, o + 3] = je
            jw = _cross(c - W2, nn) * ((lv == 2) & body_rows)[:, None]
            J[m, o + 4:o + 7] = jw
        use = (np.linalg.norm(J, axis=1) > p.min_lever) & ~skip
        target = np.zeros(14)
        if pulling:
            for s in range(2):
                As, alpha, Aw = states[s]
                gs, ga, gw = goal[s]
                G_elb2 = applied[s][4]
                target[7 * s:7 * s + 3] = _rotvec(arms[s].Gp @ (gs @ As.T) @ arms[s].Gp.T)
                target[7 * s + 3] = ga - alpha
                target[7 * s + 4:7 * s + 7] = _rotvec(G_elb2 @ (gw @ Aw.T) @ G_elb2.T)
        if not (use & (pen > p.tol)).any() and np.abs(target).max() < 1e-9:
            break
        x = target
        if use.any():
            Ju = J[use]
            u = _constrained_step(Ju * scale, pen[use] + p.tol - Ju @ target, DAMPING)
            x = target + scale * u
        joint = np.array([np.linalg.norm(x[0:3]), abs(x[3]), np.linalg.norm(x[4:7]),
                          np.linalg.norm(x[7:10]), abs(x[10]), np.linalg.norm(x[11:14])])
        step = float(joint.max())
        if step < STOP_RAD:
            if pulling:
                pulling = False
                continue
            break
        x = x * min(1.0, np.deg2rad(MAX_STEP_DEG) / step)
        new = []
        for s in range(2):
            As, alpha, Aw = states[s]
            xs, xa, xw = x[7 * s:7 * s + 3], x[7 * s + 3], x[7 * s + 4:7 * s + 7]
            G_elb2 = applied[s][4]
            Gp = arms[s].Gp
            As = Gp.T @ _rotation(xs) @ Gp @ As
            Aw = G_elb2.T @ _rotation(xw) @ G_elb2 @ Aw
            alpha = float(np.clip(alpha + xa, -p.max_angle[1], p.max_angle[1]))
            As, Aw = _clamp(As, p.max_angle[0]), _clamp(Aw, p.max_angle[2])
            new.append((As, alpha, Aw))
        states = new
    return states


def _constrained_step(J, b, damping):
    """J δ ≥ b をなるべく満たす小さな δ（arm_collision._constrained_step と同じ。変数の数は J の列の数）。"""
    active = [int(c) for c in np.flatnonzero(b > 0.0)]
    delta = np.zeros(J.shape[1])
    tol = 1e-9 * max(1.0, float(np.abs(b).max(initial=0.0)))
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


def _clamp(A, limit):
    if _angle(A) <= limit:
        return A
    rv = _rotvec(A)
    return _rotation(rv * (limit / np.linalg.norm(rv)))


def _toward_identity(state, step):
    As, alpha, Aw = state

    def back(A):
        rv = _rotvec(A)
        a = float(np.linalg.norm(rv))
        return np.eye(3) if a <= step else _rotation(rv * (1.0 - step / a))

    return (back(As), float(np.sign(alpha) * max(abs(alpha) - step, 0.0)), back(Aw))


def _saturated(state, p):
    As, alpha, Aw = state
    return (_angle(As) >= p.max_angle[0] - 1e-6 or abs(alpha) >= p.max_angle[1] - 1e-6
            or _angle(Aw) >= p.max_angle[2] - 1e-6)


@dataclass
class ContactResult:
    enabled: bool
    body_source: str
    num_body: int
    num_arm: tuple                # 腕ごとのカプセルの数
    fingers: bool                 # 指の節まで当たり判定に入れたか
    depth_before: np.ndarray      # (T, 2 腕, 8 部位) 体・相手の腕との最も深い重なり [MMD 単位]（負 = 離れている）
    depth_after: np.ndarray
    correction_deg: np.ndarray    # (T, 2 腕, 3 [肩, ひじ, 手首]) 掛けた補正の角度 [度]
    radius: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def overlap_frames(self, tol):
        """(処理前, 処理後) どこかの部位が tol [MMD 単位] より深く重なっているフレーム数。"""
        return (int((self.depth_before > tol).any(axis=(1, 2)).sum()),
                int((self.depth_after > tol).any(axis=(1, 2)).sum()))


def _part_depth(pen, pairs, parts):
    """組ごとの重なり（離す割合を掛ける前）(..., npairs) → 腕ごと・部位ごとの最も深い重なり (..., 2, 8)。"""
    out = np.full(pen.shape[:-1] + (2, len(PARTS)), -np.inf)
    part = np.array([parts[s][i] for s, i in zip(pairs.side, pairs.i)], int)
    for s in range(2):
        for k in range(len(PARTS)):
            m = (pairs.side == s) & (part == k)
            if m.any():
                out[..., s, k] = pen[..., m].max(-1)
    return out


def _raw_penetration(arm_ends, body_ends, pairs):
    """補正の無いときの、全フレーム・全組の重なり (T, npairs)（離す割合を掛ける前）。"""
    T, n = len(body_ends), len(pairs.i)
    P0 = np.empty((T, n, 2, 3))
    Q0 = np.empty_like(P0)
    for s in range(2):
        m = pairs.side == s
        P0[:, m] = arm_ends[s][:, pairs.i[m]]
        mb, ma = m & (pairs.kind == 0), m & (pairs.kind == 1)
        Q0[:, mb] = body_ends[:, pairs.j[mb]]
        Q0[:, ma] = arm_ends[1 - s][:, pairs.j[ma]]
    P0[:, :, 0] += pairs.trim[:, None] * (P0[:, :, 1] - P0[:, :, 0])
    cp, cq = closest_points(P0[..., 0, :], P0[..., 1, :], Q0[..., 0, :], Q0[..., 1, :])
    return pairs.R - np.linalg.norm(cp - cq, axis=-1)


def _is_identity(states):
    return all(_angle(st[0]) < 1e-9 and abs(st[1]) < 1e-9 and _angle(st[2]) < 1e-9 for st in states)


def resolve_contacts(skel, rt, glob_rot, local, cfg, arm_cfg, finger_bones, yield_mode, unit, fps,
                     smpl_rest, finger_local=None, log=None):
    """ステージ9b。local（ステージ9a の後のローカル回転）の 腕・ひじ・手首 を直した dict と ContactResult を返す。

    glob_rot: SMPL の大域回転 (T, J, 3, 3) / finger_local: 指ボーンのローカル回転 {名前: (T, 4)}（内部座標。
    None なら初期姿勢の指）/ yield_mode: arm_collision.mode（腕どうしで自重する腕）/ smpl_rest: SMPL の初期姿勢の関節
    """
    T = len(glob_rot)
    caps = [arm_capsules(skel, s, cfg, arm_cfg, finger_bones, unit) for s in range(2)]
    fingers = any(c.part >= 3 for c in caps[0] + caps[1])
    bodies, source = body_capsules(skel, cfg, unit)
    parts = [np.array([c.part for c in cs]) for cs in caps]
    empty = ContactResult(False, source, len(bodies), (len(caps[0]), len(caps[1])), fingers,
                          np.full((T, 2, len(PARTS)), -np.inf), np.full((T, 2, len(PARTS)), -np.inf),
                          np.zeros((T, 2, 3)))
    if not cfg.enabled or T == 0:
        return local, empty

    # ---- 組（初期姿勢で重なっている組は扱わない） ----
    margin = float(cfg.margin_m) * unit
    skip_len = float(cfg.upper_arm_skip)
    kind, side, ii, jj, R, w = [], [], [], [], [], []
    for s in range(2):
        for i, c in enumerate(caps[s]):
            a = c.a + (skip_len * (c.b - c.a) if c.part == 0 else 0.0)
            for j, b in enumerate(bodies):
                cp, cq = closest_points(a, c.b, b.a, b.b)
                if np.linalg.norm(cp - cq) < c.radius + b.radius + margin:
                    continue
                kind.append(0), side.append(s), ii.append(i), jj.append(j)
                R.append(c.radius + b.radius + margin), w.append(b.weight)
    yield_side = {'left': 0, 'right': 1}.get(str(yield_mode), -1)
    if yield_side >= 0:
        o = 1 - yield_side
        for i, c in enumerate(caps[yield_side]):
            for j, d in enumerate(caps[o]):
                kind.append(1), side.append(yield_side), ii.append(i), jj.append(j)
                R.append(c.radius + d.radius + margin), w.append(1.0)
    pairs = _Pairs(np.array(kind, int), np.array(side, int), np.array(ii, int), np.array(jj, int),
                   np.array(R, float), np.array(w, float),
                   np.array([caps[s][i].level for s, i in zip(side, ii)], int),
                   np.array([skip_len if k_ == 0 and caps[s][i].part == 0 else 0.0
                             for k_, s, i in zip(kind, side, ii)], float))
    if len(pairs.i) == 0:
        return local, empty

    # ---- FK（指・脚も含めた大域回転） ----
    glob = globals_from_local(rt, glob_rot, local)
    glob.update({k: v for k, v in leg_globals(skel, glob_rot, smpl_rest).items() if k not in glob})
    def depth(name):
        n, d = name, 0
        while skel.parents.get(n) is not None and d < 100:
            n, d = skel.parents[n], d + 1
        return d

    # 指は根元から順に（親の指の大域回転を先に作る）
    for name in sorted((n for n in (finger_local or {}) if skel.has(n)), key=depth):
        glob[name] = _effective(skel, glob, skel.parents.get(name), T) @ quat.to_matrix(finger_local[name])
    arm_ends = [_attach(skel, glob, caps[s], T) for s in range(2)]
    body_ends = _attach(skel, glob, bodies, T)
    names = [[SIDES[s] + b for b in ('腕', 'ひじ', '手首')] for s in range(2)]
    pos = bone_positions(skel, glob, names[0] + names[1], T)
    parents = [rt.keyed_parent[SIDES[s] + '腕'] for s in range(2)]

    def arm_at(s, t):
        n = names[s]
        Gp = glob[parents[s]][t] if parents[s] is not None else np.eye(3)
        d = skel.internal(n[2]) - skel.internal(n[1])
        b_local = _unit(np.cross(d, [0.0, 0.0, 1.0]))
        return _Arm(pos[n[0]][t], pos[n[1]][t], pos[n[2]][t], Gp, glob[n[0]][t], glob[n[1]][t],
                    arm_ends[s][t], np.array([c.level for c in caps[s]]), b_local)

    # ---- 関節ごとのコスト（係数 × 関節から先の長さ²）----
    k = cfg.joint_weights
    cost = np.zeros((2, 3))
    for s in range(2):
        n = names[s]
        upper = np.linalg.norm(skel.internal(n[1]) - skel.internal(n[0]))
        fore = np.linalg.norm(skel.internal(n[2]) - skel.internal(n[1]))
        hand = max(float(np.max([np.linalg.norm(c.b - skel.internal(n[2])) for c in caps[s]
                                 if c.level == 2])), 1e-3)
        cost[s] = [float(k.shoulder) * (upper + fore + hand) ** 2, float(k.elbow) * (fore + hand) ** 2,
                   float(k.wrist) * hand ** 2]
    mx = cfg.max_deg
    p = _Params(np.deg2rad([float(mx.shoulder), float(mx.elbow), float(mx.wrist)]), cost,
                MIN_LEVER_M * unit, TOLERANCE_M * unit, np.deg2rad(float(cfg.return_deg_per_s)) / fps)

    # ---- 補正の無いときの重なり（全フレームをまとめて）----
    raw_all = _raw_penetration(arm_ends, body_ends, pairs)            # (T, npairs)
    before = _part_depth(raw_all, pairs, parts)
    raw_w = raw_all * pairs.w

    # ---- 1 回目: フレームの順に（前のフレームの補正を持ち越す） ----
    identity = [_identity_state(), _identity_state()]
    states = [identity] * T
    prev = identity
    skip = np.zeros(len(pairs.i), bool)
    ignored = np.zeros((T, len(pairs.i)), bool)
    for t in range(T):
        if _is_identity(prev) and (raw_w[t] <= p.tol).all():
            skip[:] = False
            continue
        arms = [arm_at(0, t), arm_at(1, t)]
        body = body_ends[t]
        goal = [_toward_identity(st, p.return_step) for st in prev]
        cur = _solve_frame(arms, body, pairs, prev, p, goal, skip)
        over = _penetration(arms, cur, body, pairs) > p.tol
        for s in range(2):
            if _saturated(cur[s], p):
                skip |= over & (pairs.side == s)
        skip &= over
        ignored[t] = skip
        states[t] = cur
        prev = cur

    # ---- ならす（ローカルの補正を時間方向に）→ 2 回目: ならして浅くなった重なりを離す ----
    sigma = float(cfg.smooth_sec) * fps
    As = quat.to_rotvec(quat.from_matrix(np.array([[st[s][0] for s in range(2)] for st in states])))
    al = np.array([[st[s][1] for s in range(2)] for st in states])
    Aw = quat.to_rotvec(quat.from_matrix(np.array([[st[s][2] for s in range(2)] for st in states])))
    As, al, Aw = (filters.gaussian_time(x, sigma) for x in (As, al, Aw))
    after = before.copy()
    final = []
    for t in range(T):
        st = [(_rotation(As[t, s]), float(al[t, s]), _rotation(Aw[t, s])) for s in range(2)]
        if _is_identity(st) and (raw_w[t] <= p.tol).all():
            final.append(identity)
            continue
        arms = [arm_at(0, t), arm_at(1, t)]
        body = body_ends[t]
        pen = _penetration(arms, st, body, pairs)
        if ((pen > p.tol) & ~ignored[t]).any():
            st = _solve_frame(arms, body, pairs, st, p, skip=ignored[t])
            pen = _penetration(arms, st, body, pairs)
        after[t] = _part_depth(pen / np.maximum(pairs.w, 1e-9), pairs, parts)
        final.append(st)

    # ---- ローカル回転へ ----
    out = dict(local)
    corr = np.zeros((T, 2, 3))
    for s in range(2):
        n = names[s]
        d = skel.internal(n[2]) - skel.internal(n[1])
        b_local = _unit(np.cross(d, [0.0, 0.0, 1.0]))
        qs = quat.from_matrix(np.stack([f[s][0] for f in final]))
        qa = quat.from_rotvec(np.array([f[s][1] for f in final])[:, None] * b_local)
        qw = quat.from_matrix(np.stack([f[s][2] for f in final]))
        corr[:, s] = np.rad2deg(np.stack([quat.angle_between(q, quat.IDENTITY)
                                          for q in (qs, qa, qw)], axis=1))
        for name, q in zip(n, (qs, qa, qw)):
            if corr[:, s].max(initial=0.0) > 1e-9 and name in out:
                out[name] = quat.make_continuous(quat.mul(q, out[name]))
    labels = dict(PART_LABELS) if fingers else dict(PART_LABELS, palm='手')
    radius = {SIDES[s]: {labels[PARTS[pt]]: float(np.mean([c.radius for c in caps[s] if c.part == pt]))
                         / unit * 100.0 for pt in sorted({c.part for c in caps[s]})} for s in range(2)}
    res = ContactResult(True, source, len(bodies), (len(caps[0]), len(caps[1])), fingers, before,
                        after, corr, radius)
    tol = OVERLAP_TOL_M * unit
    b_frames, a_frames = res.overlap_frames(tol)
    res.info = dict(body_source=source, body_capsules=len(bodies), fingers=fingers,
                    pairs=dict(body=int((pairs.kind == 0).sum()), arm=int((pairs.kind == 1).sum())),
                    overlap_frames=dict(before=b_frames, after=a_frames),
                    overlap_frames_by_part={
                        labels[pt]: [int((before[:, :, k_] > tol).any(1).sum()),
                                     int((after[:, :, k_] > tol).any(1).sum())]
                        for k_, pt in enumerate(PARTS) if pt in ('upper', 'fore', 'palm') or fingers},
                    max_correction_deg={j: float(corr[..., m].max(initial=0.0))
                                        for m, j in enumerate(('shoulder', 'elbow', 'wrist'))},
                    radius_cm=radius)
    return out, res
