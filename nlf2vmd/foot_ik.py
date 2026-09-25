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


def build_foot_ik(ankle_pos, ankle_rest, rot_quats, contact, fps, unit, cfg):
    """ankle_pos: (T, 2, 3) 足首の位置 / ankle_rest: (2, 3) 直立時の足首位置（どちらも同じ空間）。
    rot_quats: (T, 2, 4) 足ＩＫの回転（足首の大域回転 × 補正回転）。unit: スケール係数。
    """
    raw = np.asarray(ankle_pos, np.float64)
    T = len(raw)
    oe = cfg.swing_one_euro
    target = np.empty_like(raw)
    swing = np.empty_like(raw)
    rot = np.array(rot_quats, np.float64, copy=True)
    weights = _blend_weights(int(cfg.blend_frames))
    locks = []
    for foot in range(2):
        segs = contact.segments[foot]
        flags = contact.flags[:, foot]
        # 遊脚の平滑化（速い動きは残し、遅い震えを消す）。フィルタは m 単位で掛ける
        sw = filters.one_euro(raw[:, foot] / unit, fps, oe.min_cutoff, oe.beta, oe.d_cutoff,
                              oe.zero_phase) * unit
        swing[:, foot] = sw
        out = sw.copy()
        q = quat.make_continuous(rot[:, foot])
        q_out = q.copy()
        foot_locks = []
        # 接地区間のロック: 位置は区間内の中央値、回転は区間内の平均。
        # snap_to_floor なら、上下は「足首の高さ − その足の最下点の高さ」（足裏を床に着けたときの足首の高さ）
        # の中央値にする（推定のずれで接地中の足が床から浮いていても、足裏が床に着く）
        sole = np.asarray(contact.heights)[:, foot].min(-1)
        for s, e in segs:
            lock = np.median(raw[s:e + 1, foot], axis=0)
            if cfg.snap_to_floor:
                lock[1] = np.median(raw[s:e + 1, foot, 1] - sole[s:e + 1])
            out[s:e + 1] = lock
            foot_locks.append((s, e, lock))
            if cfg.lock_rotation:
                q_out[s:e + 1] = quat.average(q[s:e + 1])
        # 境界のブレンド: 「ロック値 − 遊脚値」を遊脚側だけで減衰させて足す
        offset = np.zeros((T, 3))
        rot_delta = []
        for s, e, lock in foot_locks:
            for edge, step in ((e, 1), (s, -1)):
                d = lock - sw[edge]
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
