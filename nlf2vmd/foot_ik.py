"""ステージ7: 足ＩＫの生成（ロック → 遊脚平滑化 → 境界ブレンド → 0 クランプ の順）。

足ＩＫのターゲットは足首関節の位置。VMD に書く値は「初期位置からの差分」なので、
差分 = ターゲット − 初期姿勢で直立したときの足首位置 とし、差分の Y=0 が
「足裏が床に着いた状態」になる。

接地区間の中は、床に着いている点を固定する（foot_ik.pivot）。接地はかかと「または」つま先が低く遅いことで決めるので、
1 つの接地区間には、足裏全体が着いている所のほかに、かかとを上げてつま先だけが着いている所（蹴り出し・つま先立ち）や、
つま先・かかとを軸に足を回す所（ピボット）も入る。

* **足裏全体**（かかと・つま先とも床に着いて止まっている）: 足首の位置と回転を、その部分区間の中央値・平均で固定する
  （足裏全体の接地区間は、従来どおり区間全体で 1 つの値）
* **つま先だけ・かかとだけ**（もう一方の点が床から上がっている、または足を回していてもう一方の点が軸のまわりを動いている。
  contact_phases）: 床に着いている点（MMD の
  モデルの足の形の点。locked.py の接地優先と同じ）の位置を固定し、足の回転は推定のまま（前後の固定した回転とは
  blend_frames でなめらかにつなぐ）。足首の位置は「固定した点 − 回転 × 足首からその点へのベクトル」で求める。
  かかとを上げる・つま先を軸に回ると、足首は固定した点のまわりに動き、固定した点は床を滑らない
* 部分区間どうしは、前の部分区間の最後のフレームの足の位置から次の固定の値を求めてつなぐので、段差が出ない
"""
from dataclasses import dataclass

import numpy as np

from . import filters, quat
from .skeleton import SIDES

# 接地の状態（FootIKResult.phase）
SWING, FLAT, TOE, HEEL = 0, 1, 2, 3
PHASE_POINT = {TOE: 1, HEEL: 0}   # 固定する点（sole_points の [かかと, つま先] の番号）


@dataclass
class FootIKResult:
    target: np.ndarray        # (T, 2, 3) 最終的なターゲット位置（SMPL をスケールした空間、床 y=0）
    delta: np.ndarray         # (T, 2, 3) VMD に書く差分（内部座標）
    rotation: np.ndarray      # (T, 2, 4) 足ＩＫの回転（内部座標）
    raw: np.ndarray           # (T, 2, 3) 処理前のターゲット
    swing: np.ndarray         # (T, 2, 3) 遊脚の平滑化結果（ロック前の全フレーム）
    locks: list               # 足ごとの [(開始, 終了, ロック位置), ...]（つま先・かかとだけの部分区間は開始の足首の位置）
    clamped_frames: int       # 0 クランプで持ち上げたフレーム数（足ごとの合計）
    phase: np.ndarray = None  # (T, 2) 接地の状態: SWING 遊脚 / FLAT 足裏全体 / TOE つま先だけ / HEEL かかとだけ
    sole_points: np.ndarray = None  # (2, 2, 3) 足ＩＫから見たモデルのかかと・つま先（mmd_sole_points。固定する点）

    def pinned_points(self):
        """(T, 2, 3) 各フレームで固定している点の位置（ターゲットの空間）。足裏全体・遊脚は足首（ターゲット）、
        つま先だけ・かかとだけはその点。同じ状態が続く間は、固定している点は動かない。"""
        out = np.array(self.target, np.float64, copy=True)
        if self.phase is None or self.sole_points is None:
            return out
        for foot in range(2):
            for label, p in PHASE_POINT.items():
                m = self.phase[:, foot] == label
                if m.any():
                    out[m, foot] += quat.rotate(self.rotation[m, foot],
                                                np.broadcast_to(self.sole_points[foot, p], (m.sum(), 3)))
        return out

    def planted_segments(self, segments):
        """足ごとの、同じ点を固定している区間 [(開始, 終了)]（segments の接地区間を、状態の変わり目で分けたもの）。"""
        if self.phase is None:
            return segments
        out = []
        for foot in range(2):
            out.append([(s + u, s + v) for s, e in segments[foot]
                        for u, v, _ in _label_runs(self.phase[s:e + 1, foot])])
        return out


def mmd_sole_points(skel):
    """(2, 2, 3) 左右の足の、足ＩＫの初期位置から見た かかと・つま先 の床の点（内部座標）。

    かかとは足首（足ＩＫ）の真下、つま先はつま先ＩＫ（無ければつま先）の真下の床（y = 0）。
    """
    out = np.zeros((2, 2, 3))
    for side, s in enumerate(SIDES):
        ik = skel.internal(s + '足ＩＫ')
        toe = next((skel.internal(s + n) for n in ('つま先ＩＫ', 'つま先') if skel.has(s + n)),
                   ik + np.array([0.0, 0.0, 0.13 * skel.leg_length(side)]))
        out[side, 0] = [0.0, -ik[1], 0.0]
        out[side, 1] = [toe[0] - ik[0], -ik[1], toe[2] - ik[2]]
    return out


def sole_floor_levels(skel, ankle_rest):
    """(2,) ターゲット + 回転 × sole_points の点が床（y = 0）に着くときの、その点の高さ（ターゲットの空間）。

    VMD の足ＩＫの位置は「初期位置 + ターゲット − ankle_rest」なので、点の高さは ターゲットの空間では
    ankle_rest − 足ＩＫの初期位置 だけずれる。
    """
    ik_y = np.array([skel.internal(s + '足ＩＫ')[1] for s in SIDES])
    return np.asarray(ankle_rest, np.float64)[:, 1] - ik_y


def _blend_weights(n):
    # 接地区間の外側 1..n フレーム目の重み（1 → 0 へスムーズステップで減衰）
    i = np.arange(1, n + 1)
    return 1.0 - filters.smoothstep(i / (n + 1.0))


def _lock_pieces(anchor, s, e):
    """接地区間 s〜e の中で、ロックの値を求める部分区間 [(開始, 終了)]（anchor の連続区間。無ければ区間全体）。"""
    if anchor is None or not anchor[s:e + 1].any():
        return [(s, e)]
    return [(s + a, s + b) for a, b in filters.runs(anchor[s:e + 1])]


def _label_runs(labels):
    """[(開始, 終了, ラベル)]（同じラベルが続く区間。終了を含む）。"""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return []
    cut = np.flatnonzero(np.diff(labels) != 0) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut - 1, [len(labels) - 1]])
    return [(int(s), int(e), int(labels[s])) for s, e in zip(starts, ends)]


def _hold_runs(core, keep):
    """keep の連続区間のうち、core のフレームを含むもの（前後どちらへも keep の間だけ延ばすヒステリシス）。"""
    out = np.zeros(len(keep), bool)
    for s, e in filters.runs(keep):
        if core[s:e + 1].any():
            out[s:e + 1] = True
    return out


def contact_phases(flags, heights, rotation, swing, sole_points, fps, unit, cfg):
    """(T,) 1 つの足の接地の状態（SWING / FLAT / TOE / HEEL）。cfg: 設定 foot_ik.pivot。

    flags: (T,) 接地 / heights: (T, 2 [かかと, つま先]) 高さ / rotation: (T, 4) 足ＩＫの回転（推定）/
    swing: (T, 3) 平滑化した足首の位置 / sole_points: (2, 3) 足ＩＫから見たかかと・つま先。
    次の 2 つで床に着いていない点を決める（どちらでもなければ足裏全体 FLAT）。

    * **上がっている点**: かかと（つま先）が、その足の最も低い点より raise_m 以上高い。[始める値, 続ける値] の
      ヒステリシスで、始める値を超えたフレームから前後へ、続ける値を超えている間だけ延ばす。高さはその足の最も低い点
      からの差なので、体全体の上下のぶれ・ずれには影響されない。かかとが上がっていれば TOE、つま先なら HEEL
    * **足を回している（ピボット）**: 足の向き（かかと → つま先の、鉛直軸まわりの角）が turn_deg_per_s 以上の速さで
      回り（同じヒステリシス）、その間に合計 min_turn_deg 以上回った区間。軸は、かかと・つま先のうち、区間の中の水平の
      移動の道のりが、もう一方の pivot_ratio 倍より短い点（どちらとも言えなければ FLAT のまま）。点の速さは体全体の
      前後左右のぶれがかかと・つま先の両方に乗り、片方だけがしきい値を超えることがあるので、判定には使わない
      （足の向きの回る速さは、体全体のぶれでは変わらない）
    """
    flags = np.asarray(flags, bool)
    phase = np.where(flags, FLAT, SWING).astype(np.int8)
    if not cfg.enabled:
        return phase
    heights = np.asarray(heights, np.float64)
    rise = heights - heights.min(-1, keepdims=True)
    r_on, r_keep = (float(v) * unit for v in cfg.raise_m)
    # 始める値・続ける値とも接地しているフレームだけで見る（遊脚の間の足の向きの揺れから、接地の中へ延ばさない）
    lifted = [_hold_runs((rise[:, p] > r_on) & flags, (rise[:, p] > r_keep) & flags) for p in range(2)]
    phase[flags & lifted[0] & ~lifted[1]] = TOE
    phase[flags & ~lifted[0] & lifted[1]] = HEEL

    # ピボット（上がっている点の無い、足裏全体のフレームだけ）
    T = len(flags)
    pts = np.asarray(sole_points, np.float64)
    q = np.asarray(rotation, np.float64)
    along = quat.rotate(q, np.broadcast_to(pts[1] - pts[0], (T, 3)))
    yaw = np.unwrap(np.arctan2(along[:, 0], along[:, 2]))
    t = np.arange(T)
    lo, hi = np.clip(t - 2, 0, T - 1), np.clip(t + 2, 0, T - 1)    # 前後 2 フレームの差（ぶれを抑える）
    rate = np.abs(yaw[hi] - yaw[lo]) / np.maximum(hi - lo, 1) * fps
    w_on, w_keep = (np.deg2rad(float(v)) for v in cfg.turn_deg_per_s)
    turning = _hold_runs((rate > w_on) & flags, (rate > w_keep) & flags)
    points = [np.asarray(swing, np.float64) + quat.rotate(q, np.broadcast_to(pts[p], (T, 3)))
              for p in range(2)]
    ratio = float(cfg.pivot_ratio)
    for s, e in filters.runs(turning & (phase == FLAT)):
        if e <= s or abs(yaw[e] - yaw[s]) < np.deg2rad(float(cfg.min_turn_deg)):
            continue
        path = [float(np.linalg.norm(np.diff(p[s:e + 1][:, [0, 2]], axis=0), axis=-1).sum())
                for p in points]
        if path[1] < ratio * path[0]:
            phase[s:e + 1] = TOE                                    # つま先を軸に回る（かかとが動く）
        elif path[0] < ratio * path[1]:
            phase[s:e + 1] = HEEL                                   # かかとを軸に回る（つま先が動く）
    return phase


def _clean_labels(labels, min_frames):
    """つま先だけ・かかとだけの部分が min_frames より短ければ足裏全体にする（推定の揺れで一瞬だけ回転を離さない）。"""
    labels = np.array(labels, copy=True)
    for s, e, lab in _label_runs(labels):
        if lab in (TOE, HEEL) and e - s + 1 < min_frames:
            labels[s:e + 1] = FLAT
    return labels


def build_foot_ik(ankle_pos, ankle_rest, rot_quats, contact, fps, unit, cfg, swing=None,
                  anchor=None, sole_points=None, sole_floor=None):
    """ankle_pos: (T, 2, 3) 足首の位置 / ankle_rest: (2, 3) 直立時の足首位置（どちらも同じ空間）。
    rot_quats: (T, 2, 4) 足ＩＫの回転（足首の大域回転 × 補正回転）。unit: スケール係数。
    swing: (T, 2, 3) 遊脚の軌道。与えると平滑化をせずにこれを使う（フル [接地優先] で、フルの軌道を使うため）
    anchor: (T, 2) bool ロックの値を求めるフレーム。接地区間の中に anchor のフレームがあれば、その連続区間ごとに
    ロックの値（位置・回転）を求め、区間の残りのフレームは、前・後の値を保つか、前後 2 つの値の間をなめらかに
    つなぐ（フル [接地優先] で、フルの接地区間を広げても、フルの接地区間の値は変えないため）
    sole_points: (2, 2, 3) 足ＩＫから見たかかと・つま先（mmd_sole_points）/ sole_floor: (2,) その点が床に着く高さ
    （sole_floor_levels）。両方を渡し、cfg.pivot.enabled・cfg.lock_rotation のときだけ、つま先だけ・かかとだけが
    床に着いている所でその点を固定する（渡さなければ、接地区間は足裏全体として足首の位置・回転を固定する）
    """
    raw = np.asarray(ankle_pos, np.float64)
    T = len(raw)
    oe = cfg.swing_one_euro
    target = np.empty_like(raw)
    swing_in = swing
    swing = np.empty_like(raw)
    rot = np.array(rot_quats, np.float64, copy=True)
    weights = _blend_weights(int(cfg.blend_frames))
    pivot = cfg.get('pivot')
    use_pivot = (sole_points is not None and sole_floor is not None and bool(cfg.lock_rotation)
                 and pivot is not None and bool(pivot.enabled))
    phase_out =np.where(np.asarray(contact.flags, bool), FLAT, SWING).astype(np.int8)
    n_blend = int(cfg.blend_frames)
    locks = []
    for foot in range(2):
        segs = contact.segments[foot]
        flags = contact.flags[:, foot]
        # 遊脚の平滑化（速い動きは残し、遅い震えを消す）。フィルタは m 単位で掛ける
        if swing_in is None:
            sw = filters.one_euro(raw[:, foot] / unit, fps, oe.min_cutoff, oe.beta, oe.d_cutoff,
                                  oe.zero_phase) * unit
        else:
            sw = np.array(swing_in[:, foot], np.float64, copy=True)
        swing[:, foot] = sw
        out = sw.copy()
        q = quat.make_continuous(rot[:, foot])
        q_out = q.copy()
        # 接地の状態（床に着いている点）。遊脚の平滑化をした足首の位置でピボットの軸を決める
        phase = contact_phases(flags, np.asarray(contact.heights)[:, foot], q, sw, sole_points[foot],
                               fps, unit, pivot) if use_pivot else None
        # つま先だけ・かかとだけの所で使う回転: 推定を軽くならしたもの（足裏全体は区間の平均で固定するので揺れが消えるが、
        # ここは推定に付いていくので、フレームごとの揺れが足の揺れになる）。全フレームでならしてから使うので、部分の端で歪まない
        sigma = float(pivot.smooth_sec) * fps if use_pivot else 0.0
        q_piv = quat.normalize(filters.gaussian_time(q, sigma)) if sigma > 0 else q
        foot_locks = []
        # 接地区間のロック: 位置は区間内の中央値、回転は区間内の平均。
        # snap_to_floor なら、上下は「足首の高さ − その足の最下点の高さ」（足裏を床に着けたときの足首の高さ）
        # の中央値にする（推定のずれで接地中の足が床から浮いていても、足裏が床に着く）
        sole = np.asarray(contact.heights)[:, foot].min(-1)
        pinned = None if anchor is None else np.asarray(anchor, bool)[:, foot]

        def lock_flat(a, b):
            """足裏全体の部分区間 a〜b の足首の位置（中央値）と回転（平均。lock_rotation でなければ None）。"""
            lock = np.median(raw[a:b + 1, foot], axis=0)
            if cfg.snap_to_floor:
                lock[1] = np.median(raw[a:b + 1, foot, 1] - sole[a:b + 1])
            return lock, (quat.average(q[a:b + 1]) if cfg.lock_rotation else None)

        def plant(a, b):
            """ロックの値を求める部分区間 a〜b の足ＩＫの位置・回転を out・q_out に書く。"""
            labels = _clean_labels(phase[a:b + 1], int(pivot.min_frames)) if use_pivot \
                else np.full(b - a + 1, FLAT)
            phase_out[a:b + 1, foot] = labels
            if (labels == FLAT).all():
                # 足裏全体（従来と同じ）
                lock, q_lock = lock_flat(a, b)
                out[a:b + 1] = lock
                if q_lock is not None:
                    q_out[a:b + 1] = q_lock
                foot_locks.append((a, b, lock))
                return
            parts = [(a + u, a + v, lab) for u, v, lab in _label_runs(labels)]
            flat_q = {i: quat.average(q[u:v + 1]) for i, (u, v, lab) in enumerate(parts) if lab == FLAT}
            pin = None                                       # 前の部分で固定した点の位置と、その点
            for i, (u, v, lab) in enumerate(parts):
                if lab == FLAT:
                    q_lock = flat_q[i]
                    if pin is None:
                        lock, _ = lock_flat(u, v)
                    else:
                        # 前の部分で固定していた点を動かさずに、足裏全体を床に着ける
                        lock = pin[0] - quat.rotate(q_lock, pin[1])
                    out[u:v + 1] = lock
                    q_out[u:v + 1] = q_lock
                    foot_locks.append((u, v, lock))
                    pin = None
                    continue
                o = np.asarray(sole_points[foot, PHASE_POINT[lab]], np.float64)
                if i == 0:
                    # 区間の最初からつま先だけ・かかとだけ: その点の位置の中央値（高さは床）
                    point = np.median(raw[u:v + 1, foot] + quat.rotate(q_piv[u:v + 1], o), axis=0)
                    if cfg.snap_to_floor:
                        point[1] = float(sole_floor[foot])
                else:
                    # 前のフレームの足のその点（前の部分から段差なくつなぐ）
                    point = out[u - 1] + quat.rotate(q_out[u - 1], o)
                # 回転は推定のまま。前後の部分との境目では、境目での差を補正の回転として推定に掛け、blend_frames で
                # 0 へ減らす（遊脚側の境界のブレンドと同じ。推定の向きへ寄せる slerp だと、境目の後に追いつこうと速く回る）
                qs = q_piv[u:v + 1].copy()
                n = v - u + 1
                ident = np.broadcast_to(quat.IDENTITY, (n, 4))
                if i > 0:
                    k = min(n_blend, n)
                    d = quat.mul(q_out[u - 1], quat.conj(q_piv[u - 1]))   # 前のフレームでの 出力 ← 推定 の差
                    w = 1.0 - filters.smoothstep(np.arange(1, k + 1) / (k + 1.0))
                    qs[:k] = quat.mul(quat.slerp(ident[:k], np.broadcast_to(d, (k, 4)), w), qs[:k])
                if i + 1 < len(parts) and parts[i + 1][2] == FLAT:
                    k = min(n_blend, n)
                    d = quat.mul(flat_q[i + 1], quat.conj(q_piv[v + 1]))  # 次のフレームでの 出力 ← 推定 の差
                    w = filters.smoothstep(np.arange(1, k + 1) / (k + 1.0))
                    qs[n - k:] = quat.mul(quat.slerp(ident[:k], np.broadcast_to(d, (k, 4)), w), qs[n - k:])
                q_out[u:v + 1] = qs
                out[u:v + 1] = point - quat.rotate(qs, np.broadcast_to(o, (n, 3)))
                foot_locks.append((u, v, out[u].copy()))
                pin = (point, o)

        for s, e in segs:
            pieces = _lock_pieces(pinned, s, e)
            for a, b in pieces:
                plant(a, b)
            # 部分区間の外（anchor の無いフレーム）は、区間の端では最寄りの値を保ち、部分区間の間はなめらかにつなぐ
            (a0, _), (_, b1) = pieces[0], pieces[-1]
            out[s:a0], out[b1 + 1:e + 1] = out[a0], out[b1]
            for (_, b), (a, _) in zip(pieces[:-1], pieces[1:]):
                w = filters.smoothstep(np.arange(1, a - b) / (a - b))
                out[b + 1:a] = out[b] + (out[a] - out[b]) * w[:, None]
                if cfg.lock_rotation:
                    q_out[b + 1:a] = quat.slerp(q_out[b], q_out[a], w)
            if cfg.lock_rotation:
                q_out[s:a0], q_out[b1 + 1:e + 1] = q_out[a0], q_out[b1]
        # 境界のブレンド: 「ロック値 − 遊脚値」を遊脚側だけで減衰させて足す
        offset = np.zeros((T, 3))
        rot_delta = []
        for s, e in segs:
            for edge, step in ((e, 1), (s, -1)):
                d = out[edge] - sw[edge]
                dq = quat.mul(q_out[edge], quat.conj(q[edge]))
                for i, w in enumerate(weights, start=1):
                    f = edge + step * i
                    if f < 0 or f >= T or flags[f]:
                        break
                    offset[f] += d * w
                    rot_delta.append((f, dq, w))
        out[~flags] += offset[~flags]
        for f, dq, w in rot_delta:
            q_out[f] = quat.mul(quat.slerp(quat.IDENTITY, dq, w), q_out[f])
        target[:, foot] = out
        rot[:, foot] = quat.make_continuous(q_out)
        locks.append(foot_locks)

    delta = target - np.asarray(ankle_rest)[None]
    clamped = 0
    if cfg.clamp_floor:
        # 残差の保険。ロックより後で行う（先にやるとロック値に歪んだ値が混ざる）
        below = delta[..., 1] < 0
        clamped = int(below.sum())
        delta[..., 1] = np.maximum(delta[..., 1], 0.0)
        target = delta + np.asarray(ankle_rest)[None]
    return FootIKResult(target, delta, rot, raw, swing, locks, clamped, phase_out,
                        None if sole_points is None else np.asarray(sole_points, np.float64))


def boundary_steps(target, contact):
    """接地区間の境界（区間の端から外側へのフレーム間）での移動量の最大値（足ごと）。"""
    out = []
    T = len(target)
    for foot in range(2):
        steps = [0.0]
        for s, e in contact.segments[foot]:
            for a, b in ((e, e + 1), (s - 1, s)):
                if 0 <= a and b < T:
                    steps.append(float(np.linalg.norm(target[b, foot] - target[a, foot])))
        out.append(max(steps))
    return out
