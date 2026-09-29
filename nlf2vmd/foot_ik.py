"""ステージ7: 足ＩＫの生成（ロック → 遊脚平滑化 → 境界ブレンド → 0 クランプ の順）。

足ＩＫのターゲットは足首関節の位置。VMD に書く値は「初期位置からの差分」なので、
差分 = ターゲット − 初期姿勢で直立したときの足首位置 とし、差分の Y=0 が
「足裏が床に着いた状態」になる。
"""
from dataclasses import dataclass

import numpy as np

from . import filters, quat


@dataclass
class FootIKResult:
    target: np.ndarray        # (T, 2, 3) 最終的なターゲット位置（SMPL をスケールした空間、床 y=0）
    delta: np.ndarray         # (T, 2, 3) VMD に書く差分（内部座標）
    rotation: np.ndarray      # (T, 2, 4) 足ＩＫの回転（内部座標）
    raw: np.ndarray           # (T, 2, 3) 処理前のターゲット
    swing: np.ndarray         # (T, 2, 3) 遊脚の平滑化結果（ロック前の全フレーム）
    locks: list               # 足ごとの [(開始, 終了, ロック位置), ...]
    clamped_frames: int       # 0 クランプで持ち上げたフレーム数（足ごとの合計）


def _blend_weights(n):
    # 接地区間の外側 1..n フレーム目の重み（1 → 0 へスムーズステップで減衰）
    i = np.arange(1, n + 1)
    return 1.0 - filters.smoothstep(i / (n + 1.0))


def _lock_pieces(anchor, s, e):
    """接地区間 s〜e の中で、ロックの値を求める部分区間 [(開始, 終了)]（anchor の連続区間。無ければ区間全体）。"""
    if anchor is None or not anchor[s:e + 1].any():
        return [(s, e)]
    return [(s + a, s + b) for a, b in filters.runs(anchor[s:e + 1])]


def build_foot_ik(ankle_pos, ankle_rest, rot_quats, contact, fps, unit, cfg, swing=None,
                  anchor=None):
    """ankle_pos: (T, 2, 3) 足首の位置 / ankle_rest: (2, 3) 直立時の足首位置（どちらも同じ空間）。
    rot_quats: (T, 2, 4) 足ＩＫの回転（足首の大域回転 × 補正回転）。unit: スケール係数。
    swing: (T, 2, 3) 遊脚の軌道。与えると平滑化をせずにこれを使う（フル [接地優先] で、フルの軌道を使うため）
    anchor: (T, 2) bool ロックの値を求めるフレーム。接地区間の中に anchor のフレームがあれば、その連続区間ごとに
    ロックの値（位置・回転）を求め、区間の残りのフレームは、前・後の値を保つか、前後 2 つの値の間をなめらかに
    つなぐ（フル [接地優先] で、フルの接地区間を広げても、フルの接地区間の値は変えないため）
    """
    raw = np.asarray(ankle_pos, np.float64)
    T = len(raw)
    oe = cfg.swing_one_euro
    target = np.empty_like(raw)
    swing_in = swing
    swing = np.empty_like(raw)
    rot = np.array(rot_quats, np.float64, copy=True)
    weights = _blend_weights(int(cfg.blend_frames))
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
        foot_locks = []
        # 接地区間のロック: 位置は区間内の中央値、回転は区間内の平均。
        # snap_to_floor なら、上下は「足首の高さ − その足の最下点の高さ」（足裏を床に着けたときの足首の高さ）
        # の中央値にする（推定のずれで接地中の足が床から浮いていても、足裏が床に着く）
        sole = np.asarray(contact.heights)[:, foot].min(-1)
        pinned = None if anchor is None else np.asarray(anchor, bool)[:, foot]
        for s, e in segs:
            pieces = []
            for a, b in _lock_pieces(pinned, s, e):
                lock = np.median(raw[a:b + 1, foot], axis=0)
                if cfg.snap_to_floor:
                    lock[1] = np.median(raw[a:b + 1, foot, 1] - sole[a:b + 1])
                q_lock = quat.average(q[a:b + 1]) if cfg.lock_rotation else None
                pieces.append((a, b, lock, q_lock))
                foot_locks.append((a, b, lock))
            # 部分区間の外（anchor の無いフレーム）は、区間の端では最寄りの値を保ち、部分区間の間はなめらかにつなぐ
            (a0, _, lock0, q0), (_, b1, lock1, q1) = pieces[0], pieces[-1]
            out[s:a0], out[b1 + 1:e + 1] = lock0, lock1
            for a, b, lock, _ in pieces:
                out[a:b + 1] = lock
            for (_, b, lock, qa), (a, _, nxt, qb) in zip(pieces[:-1], pieces[1:]):
                w = filters.smoothstep(np.arange(1, a - b) / (a - b))
                out[b + 1:a] = lock + (nxt - lock) * w[:, None]
                if cfg.lock_rotation:
                    q_out[b + 1:a] = quat.slerp(qa, qb, w)
            if cfg.lock_rotation:
                q_out[s:a0], q_out[b1 + 1:e + 1] = q0, q1
                for a, b, _, q_lock in pieces:
                    q_out[a:b + 1] = q_lock
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
    return FootIKResult(target, delta, rot, raw, swing, locks, clamped)


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
