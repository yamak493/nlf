"""ステージ1b: 慣性と重力で、明らかに間違ったフレーム（外れ値）を見つけて置き換える。

モーションキャプチャ（単眼推定）の結果には、腕や脚が数フレームだけ別の位置へ飛ぶ・体の向きが一瞬反転する・
骨盤が数フレームだけ前後や上下へずれる、といった誤りが混ざる。ステージ2の外れ値の除去（角速度が 720 度/秒を
超える単独のフレーム）では 1 フレームの誤りしか扱えず、数フレーム続く誤りは平滑化で前後へ広がって残る。
そこで平滑化の前に、次の 2 つで外れたフレームを見つける。

* 姿勢（慣性）: 体は急に止まったり向きを変えたりできないので、各フレームの位置は前後のフレームから予測できる。
  部位ごと（胴・脚・頭・左腕・右腕）に、関節の位置と向き（関節の軸の先に置いた点）を部位の付け根の座標系で表し、
  各フレームを、前後の外れていないフレーム（各側 3 つ）に当てはめた 3 次式で予測する。予測との差が大きいフレームを
  外れとし、外れを除いて予測し直すことを繰り返す。予測には、近いフレームを 1〜4 つ飛ばしたものも使う（数フレーム
  続く外れの内側は、外れの外側から予測しないと外れどうしで合ってしまう）。
  外れのそばの正しいフレームは、予測に外れが混ざって合わなく見えることがあるので、前だけ・後ろだけから一定の速度で
  動くとして予測し、どちらかによく合うフレームは外れにしない。本当に速く動いて止まった（段差の）動きは、段差の
  前後で予測との差の向きが反対になるので外れにしない。しきい値は点ごとの予測の差の中央値（その動画の推定のぶれの
  大きさ）の threshold 倍で、min_m より小さい差は外れにしない。
* 重心（重力）: 体に掛かる力は重力と、床から受ける力（押すだけで引けない・摩擦の範囲）だけなので、重心の加速度 a は
      a_y ≥ -g（重力より速くは落ちない）
      |a_水平| ≤ μ (a_y + g)（床を押す力の摩擦の範囲でしか横へ加速できない。宙に浮いていれば a = (0, -g, 0)）
      a_y + g ≤ max_force_g × g（床から受ける力は体重の数倍まで）
  を満たす。この条件を満たす重心の軌道のうち、推定の重心に最も近いものを ADMM で求め、それから大きく離れる
  フレームを外れとする（外れを除いて求め直すことを繰り返す）。奥行き（カメラの光軸）は推定のぶれが大きいので、
  奥行きと、それ以外（上下・左右）でしきい値を分ける。数フレームだけ浮く・沈む・前後へ飛ぶ骨盤を除き、
  本当のジャンプ（重力だけで動く放物線）は残る。

置き換え:
* 姿勢の外れは、その部位の関節の回転だけを前後の外れていないフレームから球面線形補間する（胴が外れたら全関節）。
* 骨盤の位置（胴か重心が外れたフレーム）は、重力の条件を満たす重心の軌道に、前後のフレームでの推定との差を
  直線でつないで足したものに、置き換えた姿勢の重心が来るように置く（重心を使わないときと、先頭・末尾の外れは、
  加速度が最小になるようにつなぐ）。
* 胴・脚・重心が外れたフレームは、以降のステージで観測として使わない（valid=False。床・接地・奥行き・前後の
  傾きの推定から除く）。頭・腕だけが外れたフレームは、回転を置き換えるだけにする。

max_run_sec より長く続く外れは、前後どちらが正しいのか決められないので置き換えない（警告する）。外れが
max_ratio より多い検出（カメラが動いている・人に持ち上げられる等で、重力の条件が合わない動画）は使わない。
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from . import quat
from .body_model import forward_kinematics
from .filters import frames_for_fps, median_time, runs
from .lean import com_weights

GRAVITY = 9.8
MAX_SKIP = 5        # 何フレーム離れた外側まで予測に使うか（これより遠くからの予測は、本当の動きでも外れやすい）

# 部位: (名前, 付け根の関節（None = 骨盤の位置・世界の向き）, 位置を使う関節, 向きを使う関節, 置き換える関節,
#        観測として使わなくするか)
PARTS = (
    ('torso', None, (3, 6, 9), (0, 9), tuple(range(24)), True),
    ('legs', 0, (4, 5, 7, 8, 10, 11), (7, 8), (1, 2, 4, 5, 7, 8, 10, 11), True),
    ('head', 9, (12, 15), (15,), (12, 15), False),
    ('left_arm', 9, (16, 18, 20, 22), (20,), (13, 16, 18, 20, 22), False),
    ('right_arm', 9, (17, 19, 21, 23), (21,), (14, 17, 19, 21, 23), False),
)
PART_LABELS = dict(torso='胴', legs='脚', head='頭', left_arm='左腕', right_arm='右腕',
                   com='重心')


@dataclass
class OutlierResult:
    quats: np.ndarray            # (T, J, 4) 置き換えた後の回転
    root_pos: np.ndarray         # (T, 3) 置き換えた後の骨盤の位置
    valid: np.ndarray            # (T,) 観測として使うフレーム（入力の valid と、胴・脚・重心が外れていないこと）
    flags: dict                  # 部位の名前・'com' → (T,) bool 外れとして置き換えたフレーム
    scores: dict                 # 同じキー → (T,) 予測との差 / しきい値（1 を超えると外れ）
    com: np.ndarray = None       # (T, 3) 重力の向きにそろえた推定の重心 [m]（置き換える前）
    com_fit: np.ndarray = None   # (T, 3) 重力の条件を満たす重心の軌道
    depth_axis: np.ndarray = None
    skipped: dict = field(default_factory=dict)    # 使わなかった検出 → 理由
    long_runs: list = field(default_factory=list)  # (名前, 開始, 終了) 長すぎて置き換えなかった外れ
    enabled: bool = True

    @property
    def replaced(self):
        """(T,) 何か置き換えたフレーム。"""
        out = np.zeros(len(self.valid), bool)
        for f in self.flags.values():
            out |= f
        return out

    @property
    def info(self):
        T = len(self.valid)
        return dict(
            enabled=self.enabled,
            frames={k: int(v.sum()) for k, v in self.flags.items()},
            replaced_frames=int(self.replaced.sum()),
            unobserved_frames=int(self.unobserved.sum()),
            segments={k: [[int(s), int(e)] for s, e in runs(v)] for k, v in self.flags.items()
                      if v.any()},
            long_runs=[[n, int(s), int(e)] for n, s, e in self.long_runs],
            skipped=dict(self.skipped),
            ratio=round(float(self.replaced.sum()) / max(T, 1), 4))

    @property
    def unobserved(self):
        """(T,) 観測として使わなくしたフレーム（胴・脚・重心の外れ）。"""
        out = np.zeros(len(self.valid), bool)
        for name, *_, unobserve in PARTS:
            if unobserve and name in self.flags:
                out |= self.flags[name]
        if 'com' in self.flags:
            out |= self.flags['com']
        return out


# ---------------------------------------------------------------- 姿勢（慣性）

def part_points(G, P, base, joints, axes_of, axis_length):
    """(T, C, 3) 部位の点: 関節の位置と、関節の x・z 軸の先（長さ axis_length）の点を、付け根の座標系で表したもの。

    base が None なら骨盤からの相対位置を世界の向きのまま使う（体全体の向きの反転も外れとして見つかる）。
    """
    if base is None:
        origin, R = P[:, 0], None
    else:
        origin, R = P[:, base], G[:, base]
    pts = [P[:, list(joints)] - origin[:, None]]
    for j in axes_of:
        pts.append(np.stack([G[:, j, :, 0], G[:, j, :, 2]], axis=1) * axis_length)
    pts = np.concatenate(pts, axis=1)
    if R is not None:
        pts = np.einsum('tab,tca->tcb', R, pts)      # R^T p
    return pts


def _nearest_slots(idx, T, skip):
    """各フレーム t について、t より前の good なフレームのうち近いほうから skip-1 個を飛ばした残りの末尾（idx の位置）と、
    t より後ろの同じく先頭。飛ばすのは good なフレームの数なので、外れとして除いたフレームは数えない。"""
    t = np.arange(T)
    left_end = np.searchsorted(idx, t, side='left') - (skip - 1)
    right_start = np.searchsorted(idx, t, side='right') + (skip - 1)
    return np.maximum(left_end, 0), np.minimum(right_start, len(idx))


@dataclass
class Prediction:
    value: np.ndarray        # (T, C, 3) 予測
    ok: np.ndarray           # (T,) bool 予測できた（両側から: 両側に 2 つ以上 / 片側から: 2 つ以上）
    distance: np.ndarray     # (T,) 予測に使った最も近いフレームまでの間隔（両側から: 左右の間隔の半分）
    fit_rms: np.ndarray = None   # (T, C) 予測に使ったフレームへの 3 次式の当てはまりの悪さ（両側から）


def neighbour_prediction(y, good, per_side=3, skip=1):
    """(T, C, 3) y の各フレーム t を、前後の good なフレーム（t に近いほうから skip-1 個ずつ飛ばして、各側 per_side 個）に
    当てはめた 3 次式で予測する（両側から）。skip を大きくすると、t を含む数フレームの外れ（t の近くのフレームも
    外れている）も、その外側から予測できる。good なフレームが 4 つ未満なら None。
    """
    y = np.asarray(y, np.float64)
    T = len(y)
    idx = np.flatnonzero(good)
    n = len(idx)
    if n < 4:
        return None
    k, h = int(per_side), max(int(skip), 1)
    t = np.arange(T)
    left_end, right_start = _nearest_slots(idx, T, h)
    nl = np.minimum(left_end, k)
    nr = np.minimum(n - right_start, k)
    slots = np.arange(2 * k)
    col = np.where(slots[None] < nl[:, None],
                   left_end[:, None] - nl[:, None] + slots[None],
                   right_start[:, None] + slots[None] - nl[:, None])
    used = slots[None] < (nl + nr)[:, None]
    frames = idx[np.clip(col, 0, n - 1)]                                  # (T, 2k)
    ok = (nl >= 2) & (nr >= 2)
    gap = np.where(ok, idx[np.clip(right_start, 0, n - 1)] - idx[np.clip(left_end - 1, 0, n - 1)], 0)
    dt = (frames - t[:, None]) / np.maximum(gap, 1)[:, None] * used
    V = np.stack([np.ones_like(dt), dt, dt ** 2, dt ** 3], axis=-1) * used[..., None]
    A = np.einsum('tsi,tsj->tij', V, V)
    A[~ok] = np.eye(4)
    A += np.eye(4) * 1e-9
    H = np.linalg.solve(A, np.swapaxes(V, 1, 2))                          # (T, 4, 2k)
    yn = y[frames]                                                        # (T, 2k, C, 3)
    beta = np.einsum('tis,ts...->ti...', H, yn)
    resid = (yn - np.einsum('tsi,ti...->ts...', V, beta)) * used[:, :, None, None]
    dof = np.maximum(used.sum(1) - 4, 1)
    fit_rms = np.sqrt(np.einsum('tscd->tc', resid ** 2) / dof[:, None])
    return Prediction(beta[:, 0], ok, (gap + 1) // 2, fit_rms)


def side_prediction(y, good, side, skip=1, per_side=3):
    """(T, C, 3) y の各フレーム t を、片側（side=-1: 前、+1: 後ろ）の good なフレーム（t に近いほうから skip-1 個を
    飛ばして per_side 個）に直線を当てはめて（一定の速度で動くとして。慣性）予測する（片側から）。"""
    y = np.asarray(y, np.float64)
    T = len(y)
    idx = np.flatnonzero(good)
    n = len(idx)
    t = np.arange(T)
    k = int(per_side)
    left_end, right_start = _nearest_slots(idx, T, max(int(skip), 1))
    if side < 0:
        slots = left_end[:, None] - k + np.arange(k)[None]
    else:
        slots = right_start[:, None] + np.arange(k)[None]
    used = (slots >= 0) & (slots < n)
    frames = idx[np.clip(slots, 0, max(n - 1, 0))] if n else np.zeros_like(slots)
    dt = np.where(used, (frames - t[:, None]).astype(np.float64), 0.0)
    m = used.sum(1)
    ok = m >= 2
    near = np.where(used, np.abs(dt), np.inf).min(1)
    mm = np.maximum(m, 1)
    mean = dt.sum(1) / mm
    var = np.where(used, (dt - mean[:, None]) ** 2, 0.0).sum(1)
    w = np.where(used, 1.0 / mm[:, None]
                 - (dt - mean[:, None]) * mean[:, None] / np.maximum(var, 1e-9)[:, None], 0.0)
    w = np.where(ok[:, None], w, 0.0)
    pred = np.einsum('ts,ts...->t...', w, y[frames]) if n else np.zeros_like(y)
    return Prediction(pred, ok, np.where(ok, near, 0).astype(int))


def coherent(res):
    """外れの区間の、予測との差 res (n, C, 3) の向きがそろっているか（区間の前後で元の動きに戻る「外れ」か）。

    本当に速く動いて止まった（段差の）動きでは、段差の前は差が一方へ、後は反対へ向くのでそろわない。
    差の合計が最も大きい点で、差のベクトルの和の長さ / 長さの和 が 0.5 以上ならそろっているとする。
    """
    if len(res) < 2:
        return True
    norms = np.linalg.norm(res, axis=-1)                                  # (n, C)
    c = int(np.argmax(norms.sum(0)))
    return float(np.linalg.norm(res[:, c].sum(0))) >= 0.5 * float(norms[:, c].sum())


class InertiaScores:
    """前後のフレームからの予測との差 / しきい値。

    * 両側から（neighbour_prediction）: 前後のフレームに当てはめた 3 次式。精度が高いので外れを見つけるのに使う。
      ただし予測に使うフレームに外れが混ざると、外れのそばの正しいフレームも合わなく見える
    * 片側から（side_prediction）: 前だけ・後ろだけから一定の速度で。精度は低いが、外れのそばの正しいフレームは、
      外れと反対の側からの予測にはよく合う。どちらかの側によく合う（veto 以下の）フレームは外れにしない
    しきい値は予測の種類・予測に使ったフレームまでの間隔・点ごとに、最初（全フレームを使った）予測の差の中央値
    （その動画の推定のぶれと、間隔が空くほど大きくなる本当の動きの予測の誤差）の threshold 倍（min_m 以上）。
    両側からの予測は、予測に使ったフレームそのものに 3 次式が当てはまらない（速く回っている・段差がある・外れが
    混ざっている）ときは当てにならないので、しきい値を当てはめの残差の reliability 倍まで上げる。turn（部位の向きの
    1 フレームあたりの回転 [rad]）が大きいフレームでは離して予測しない（予測に使う範囲で向きが 90 度以上回ると、
    向きの点は円を描くので、離した予測は本当の動きでも大きく外れる）。
    """

    reliability = 5.0
    veto = 0.5
    one_sided_margin = 1.5   # 片側からしか予測できない（先頭・末尾）フレームは、しきい値をこの倍にする

    def __init__(self, y, observed, threshold, min_m, max_skip, per_side=3, turn=None):
        self.y = np.asarray(y, np.float64)
        self.per_side = int(per_side)
        self.max_skip = max(int(max_skip), 1)
        T = len(self.y)
        if turn is None:
            self.skip_limit = np.full(T, self.max_skip)
        else:
            span = (np.pi / 2) / np.maximum(np.asarray(turn, np.float64), 1e-9)
            self.skip_limit = np.clip(np.floor(span).astype(int) - self.per_side + 1, 1,
                                      self.max_skip)
        mins = np.broadcast_to(np.asarray(min_m, np.float64), self.y.shape[1:2])
        self.taus = {}                        # (種類 0: 両側 / -1・+1: 片側, 間隔) → (C,)
        for kind in (0, -1, 1):
            for h in range(1, self.max_skip + 1):
                p = self._predict(observed, kind, h)
                use = observed & p.ok & (self.skip_limit >= h) if p is not None else None
                if use is None or use.sum() < 2 * self.per_side:
                    break
                r = np.linalg.norm(self.y - p.value, axis=-1)
                self.taus[kind, h] = np.maximum(
                    float(threshold) * np.median(r[use], axis=0), mins)

    def _predict(self, good, kind, h):
        if kind == 0:
            return neighbour_prediction(self.y, good, self.per_side, h)
        return side_prediction(self.y, good, kind, h, self.per_side)

    def _kind(self, good, kind):
        """(T,) その種類の予測との差 / しきい値 の最大（点・離し方について）と、予測できたか。"""
        T = len(self.y)
        hs = [h for (kd, h) in self.taus if kd == kind]
        score, avail = np.zeros(T), np.zeros(T, bool)
        if not hs:
            return score, avail
        table = np.stack([self.taus[kind, h] for h in hs])
        for h in hs:
            p = self._predict(good, kind, h)
            if p is None:
                break
            ok = p.ok & ((self.skip_limit >= h) | (h == 1))
            tau = table[np.clip(p.distance, 1, len(hs)) - 1]
            if kind == 0:
                tau = np.maximum(tau, self.reliability * p.fit_rms)
            r = (np.linalg.norm(self.y - p.value, axis=-1) / tau).max(axis=1)
            score = np.where(ok, np.maximum(score, r), score)
            avail |= ok
        return score, avail

    def explained(self, observed):
        """(T,) 片側からの予測によく合う（veto 以下の）フレーム。外れのそばの正しいフレームは、外れと反対の側からの
        予測に合う。外れを除く前（observed 全部）で求める（除いた後は、片側の予測が遠くのフレームからになり、
        外れの内側のフレームも合って見えることがある）。そろわない側（full_side。先頭・末尾の数フレームの、端の側）は
        使わない。"""
        out = np.zeros(len(self.y), bool)
        for side in (-1, 1):
            s, _ = self._kind(observed, side)
            full = self.full_side(observed, side)
            out |= full & (s <= self.veto)
        return out

    def full_side(self, observed, side):
        """(T,) その側に、最も離した予測に使うフレームまでそろっている（per_side + max_skip - 1 個以上ある）。
        そろわない側は、先頭・末尾の外れの中だけで予測が合ってしまうことがあるので当てにしない。"""
        idx = np.flatnonzero(observed)
        need = self.per_side + self.max_skip - 1
        left_end, right_start = _nearest_slots(idx, len(self.y), 1)
        return (left_end >= need) if side < 0 else (len(idx) - right_start >= need)

    def _one_side(self, good, side):
        """(T,) 片側から（近いほうから）の予測との差 / しきい値（予測に使ったフレームまでの間隔のしきい値で）。"""
        p = self._predict(good, side, 1)
        tau = np.stack([self.taus[side, h] for h in range(1, self.max_skip + 1)
                        if (side, h) in self.taus])
        tau = tau[np.clip(p.distance, 1, len(tau)) - 1]
        r = (np.linalg.norm(self.y - p.value, axis=-1) / tau).max(axis=1)
        return np.where(p.ok, r, 0.0)

    def grow(self, observed, flags, ends, max_len):
        """(T,) 外れの区間の隣で、区間の外側（区間と反対の側）からの予測に合わないフレーム。外側に予測に使う
        フレームがそろわない（先頭・末尾に近い）ときは、区間を飛ばした反対側からの予測で確かめる。"""
        T = len(self.y)
        good = observed & ~flags
        left, right = np.zeros(T, bool), np.zeros(T, bool)
        for s, e in runs(flags):
            if e - s + 1 < max_len:
                left[max(s - 1, 0)] = True
                right[min(e + 1, T - 1)] = True
        left &= good
        right &= good
        past, future = self._one_side(good, -1), self._one_side(good, 1)
        full_past, full_future = ends
        grow = (left & full_past & (past > 1.0)) | (right & full_future & (future > 1.0))
        # 先頭・末尾に近い（外側に予測に使うフレームがそろわない）隣のフレームは、区間の端の外れの値と、区間の
        # 反対側から延ばした予測の、近いほうに合わせる（片側から延ばした予測は離れるほど当てにならないので、
        # 予測に合わないかどうかだけでは決められない）
        tau = self.taus[0, 1]
        for side, mask, full in ((1, left & ~full_past, full_future),
                                 (-1, right & ~full_future, full_past)):
            if not mask.any():
                continue
            p = self._predict(good, side, 1)
            for t in np.flatnonzero(mask & full & p.ok):
                edge = t + 1 if side > 0 else t - 1                   # 区間の端（外れ）
                d_out = (np.linalg.norm(self.y[t] - self.y[edge], axis=-1) / tau).max()
                d_in = (np.linalg.norm(self.y[t] - p.value[t], axis=-1) / tau).max()
                grow[t] |= d_out < d_in and d_in > 1.0
        return grow

    def score(self, good, explained, ends):
        """(T,) 外れの度合い（1 を超えると外れ）。

        good: 予測に使うフレーム / explained: 片側からの予測によく合う（外れにしない）フレーム /
        ends: (前の側がそろう, 後ろの側がそろう)。片側がそろわない先頭・末尾のフレームは、そろう側からの予測
        （1〜max_skip フレーム離したもの）で、しきい値を one_sided_margin 倍にして確かめる。それ以外は両側からの予測で。
        """
        full_past, full_future = ends
        s2, a2 = self._kind(good, 0)
        out = np.where(a2 & full_past & full_future & ~explained, s2, 0.0)
        for side, full, other in ((1, full_future, full_past), (-1, full_past, full_future)):
            end = full & ~other
            if end.any():
                one, ok = self._kind(good, side)
                out = np.where(end & ok, one / self.one_sided_margin, out)
        return out


def detect_part(y, observed, threshold, min_m, max_skip, max_len, per_side=3, turn=None,
                long_runs=None, max_iter=20):
    """部位の点 y (T, C, 3) の外れ。戻り値: (外れ (T,) bool, 外れの度合い (T,)。1 を超えると外れ)。

    外れの度合い（InertiaScores.score）が 1 を超えるフレームを外れとし、外れを除いて（予測に使わずに）求め直すことを、
    変わらなくなるまで繰り返す（予測に使うフレームは、除いたフレームを数えずに近いほうから選ぶので、数フレーム続く
    外れも内側から順に見つかる）。外れの区間の隣のフレームは、区間の外側からの予測に合わなければ区間に加える
    （InertiaScores.grow。両側からの予測では、区間の両端の外れがお互いの予測に混ざって見つからないことがある）。
    最後に、向きのそろわない区間（本当に速く動いて止まった動き）は外れにしない（coherent）。
    turn: (T,) 部位の向きの 1 フレームあたりの回転 [rad]（離して予測してよい間隔を決める）
    long_runs: リストを渡すと、max_len より長く続いて外れにしなかった区間 (開始, 終了) を加える
    """
    T = len(y)
    flags = np.zeros(T, bool)
    if observed.sum() < 4 * per_side:
        return flags, np.zeros(T)
    sc = InertiaScores(y, observed, threshold, min_m, max_skip, per_side, turn)
    if not sc.taus:
        return flags, np.zeros(T)
    explained = sc.explained(observed)
    ends = (sc.full_side(observed, -1), sc.full_side(observed, 1))
    score = sc.score(observed, explained, ends)
    dropped = []
    for _ in range(max_iter):
        new = observed & (sc.score(observed & ~flags, explained, ends) > 1.0) if flags.any() \
            else observed & (score > 1.0)
        dropped = []
        new = drop_long_runs(new, max_len, '', dropped)
        if np.array_equal(new, flags):
            break
        flags = new
    if long_runs is not None:
        long_runs.extend((s, e) for _, s, e in dropped)
    for _ in range(max_len):                                    # 区間の端を広げる
        grow = sc.grow(observed, flags, ends, max_len)
        if not grow.any():
            break
        flags |= grow
    if flags.any():
        p = neighbour_prediction(y, observed & ~flags, per_side, 1)
        res = y - p.value
        for s, e in runs(flags):
            if p.ok[s:e + 1].all() and not coherent(res[s:e + 1]):
                flags[s:e + 1] = False
    return flags, np.maximum(score, sc.score(observed & ~flags, explained, ends))


# ---------------------------------------------------------------- 重心（重力）

def second_difference(T):
    """(T-2, T) 2 階差分（中心のフレーム 1..T-2）。"""
    return sparse.diags([np.ones(T - 2), -2.0 * np.ones(T - 2), np.ones(T - 2)], [0, 1, 2],
                        shape=(T - 2, T), format='csr')


def project_gravity_cone(a, mu, max_force):
    """(N, 3) 加速度 a（Y 上向き）を、a + g·ŷ が「上向きの成分 ≤ max_force・水平の成分 ≤ μ × 上向きの成分」の
    範囲（床から受ける力の範囲）に入るよう、最も近い点へ射影する。"""
    f = np.array(a, np.float64, copy=True)
    f[:, 1] += GRAVITY
    t = f[:, 1]
    v = f[:, [0, 2]]
    r = np.linalg.norm(v, axis=1)
    inside = r <= mu * t
    polar = mu * r <= -t                                 # 最も近い点は原点（自由落下）
    tp = (t + mu * r) / (1.0 + mu * mu)                  # 円錐の側面へ
    t_new = np.where(inside, t, np.where(polar, 0.0, tp))
    r_new = np.where(inside, r, np.where(polar, 0.0, mu * tp))
    cap = t_new > max_force                              # 上限の面（円板）へ
    t_new = np.where(cap, max_force, t_new)
    r_new = np.where(cap, np.minimum(r, mu * max_force), r_new)
    v_new = v * (r_new / np.maximum(r, 1e-12))[:, None]
    return np.stack([v_new[:, 0], t_new - GRAVITY, v_new[:, 1]], axis=1)


def _basis(axis):
    """奥行きの軸を 1 行目とする正規直交基底 (3, 3)。"""
    a = np.asarray(axis, np.float64)
    a = a / np.linalg.norm(a)
    helper = np.array([0.0, 1.0, 0.0]) if abs(a[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    b = np.cross(a, helper)
    b /= np.linalg.norm(b)
    return np.stack([a, b, np.cross(a, b)])


class GravityFit:
    """重力の条件を満たす重心の軌道のうち、推定の重心に（重み付きで）最も近いものを ADMM で求める。

        最小化  Σ_t w_t |Σ^{-1/2} (x_t − y_t)|²   条件  (D² x)_t · fps² ∈ 床から受ける力の範囲 − g·ŷ

    Σ は奥行きの軸の向きだけ大きい（sigma_depth）ぶれの大きさ。重みを変えて何度も解くので、双対変数を持ち越す。
    """

    def __init__(self, fps, depth_axis, cfg, sigma=0.02, sigma_depth=0.1, rho=0.3):
        self.fps = float(fps)
        self.Q = _basis(depth_axis)
        self.sig2 = np.array([sigma_depth, sigma, sigma]) ** 2
        self.mu = float(cfg.friction)
        self.max_force = float(cfg.max_force_g) * GRAVITY
        self.iterations = int(cfg.iterations)
        # ρ（2 階差分の単位）: 観測の重み 1/σ² と釣り合う大きさ
        self.rho = float(rho) / (sigma * sigma)
        self.z = self.u = None

    def solve(self, y, w):
        T = len(y)
        D = second_difference(T)
        DtD = (D.T @ D).tocsc()
        yq = y @ self.Q.T                                        # 奥行き・その他 2 軸の成分
        lus = [splu((sparse.diags(w / s2 + 1e-9) + self.rho * DtD).tocsc())
               for s2 in self.sig2]
        if self.z is None or len(self.z) != T - 2:
            self.z = project_gravity_cone(D @ y * self.fps ** 2, self.mu, self.max_force)
            self.u = np.zeros_like(self.z)
        z, u = self.z, self.u
        s = self.fps ** 2
        for _ in range(self.iterations):
            target = ((z - u) / s) @ self.Q.T                    # D x の目標（基底の成分）
            xq = np.stack([lus[i].solve(w * yq[:, i] / self.sig2[i]
                                        + self.rho * (D.T @ target[:, i]))
                           for i in range(3)], axis=1)
            x = xq @ self.Q
            ax = D @ x * s
            z = project_gravity_cone(ax + u, self.mu, self.max_force)
            u = u + ax - z
        self.z, self.u = z, u
        return x


def com_channels(com, axis):
    """(T, 2, 3) 重心の奥行きの軸の成分と、それ以外（上下・左右）の成分。"""
    d = (com @ axis)[:, None] * axis
    return np.stack([d, com - d], axis=1)


def turn_rate(G, joints, span=3, window=11):
    """(T,) 関節 joints の大域回転の、1 フレームあたりの回転角 [rad] の最大（関節について）。

    次の 2 つのうち小さいほう（窓幅 window のメディアンでならしたもの）。本当に速く回っているときはどちらも大きい。
    * 隣のフレームとの回転角: 推定のぶれが大きいと、ぶれの分だけ大きくなる
    * t - span と t + span のフレームの間の回転角を 2·span で割ったもの: ぶれは足し合わさらないが、数フレームの
      外れで向きが跳ねると、その前後 span フレームが大きくなる
    """
    T = len(G)
    if T < 2:
        return np.zeros(T)
    t = np.arange(T)

    def rate(d):
        a, b = np.clip(t - d, 0, T - 1), np.clip(t + d, 0, T - 1)
        out = np.zeros(T)
        for j in joints:
            R = G[:, j]
            c = (np.einsum('tab,tab->t', R[b], R[a]) - 1.0) / 2.0
            out = np.maximum(out, np.arccos(np.clip(c, -1.0, 1.0)) / np.maximum(b - a, 1))
        return median_time(out, window)

    return np.minimum(rate(1), rate(span))


def detect_com(com, observed, fps, depth_axis, cfg, max_len, long_runs=None, max_iter=8,
               huber=0.3, max_skip=MAX_SKIP, per_side=3):
    """重心 (T, 3)（Y 上向き・重力の向きにそろえた座標 [m]）の外れ。

    1. 慣性: 姿勢と同じく、前後のフレームからの予測との差で外れの候補を見つける（detect_part）
    2. 重力: 候補を除いて（重み 0 で）、重力の条件を満たす軌道を求める。差がしきい値を超えるフレームを外れとし、
       外れに強くするため、差がしきい値の huber 倍を超えるフレームの重みは差に反比例して下げて（Huber の重み）
       解き直すことを繰り返す（数フレーム続く外れに軌道が引き寄せられて、外れの前後が外れに見えるのを防ぐ）
    3. 2 の外れを含む候補の区間は、区間全体を外れにする。2 の外れを含まない候補（本当のジャンプなど、慣性の予測には
       合わないが重力だけで動ける動き）は外れにしない
    戻り値: (外れ (T,) bool, 差 / しきい値 (T,), 重力の条件を満たす軌道 (T, 3))。
    """
    T = len(com)
    flags = np.zeros(T, bool)
    if observed.sum() < 8 or T < 8:
        return flags, np.zeros(T), np.array(com, np.float64, copy=True)
    axis = np.asarray(depth_axis, np.float64) / np.linalg.norm(depth_axis)
    k = float(cfg.threshold)
    mins = (float(cfg.min_depth_m), float(cfg.min_m))
    seeds, _ = detect_part(com_channels(com, axis), observed, k, mins, max_skip, max_len,
                           per_side, long_runs=long_runs)
    fit = GravityFit(fps, axis, cfg, sigma=mins[1] / 2.0, sigma_depth=mins[0] / 2.0)
    w = (observed & ~seeds).astype(np.float64)
    flags = seeds
    tau = x = score = None
    for _ in range(max_iter):
        x = fit.solve(com, w)
        r = com - x
        rd = r @ axis
        ro = np.linalg.norm(r - rd[:, None] * axis, axis=1)
        if tau is None:
            ok = observed & ~seeds
            tau = (max(k * float(np.median(np.abs(rd[ok]))), mins[0]),
                   max(k * float(np.median(ro[ok])), mins[1]))
        score = np.maximum(np.abs(rd) / tau[0], ro / tau[1])
        new = observed & (score > 1.0)
        w_new = np.where(observed & ~new, np.minimum(1.0, huber / np.maximum(score, 1e-9)), 0.0)
        if np.array_equal(new, flags) and np.allclose(w_new, w, atol=0.02):
            break
        flags, w = new, w_new
    # 重力の条件で外れと確かめられた外れの候補（慣性）は、候補の区間全体を外れにする（重力の条件だけでは、外れの
    # 一部を重力で動ける範囲へならして、外れの端のフレームを残すことがある）
    for s, e in runs(seeds):
        if flags[s:e + 1].any():
            flags[s:e + 1] = True
    return flags, score, x


# ---------------------------------------------------------------- 置き換え

def interpolate_rotations(q, mask):
    """(T, J, 4) の回転の mask (T, J) のフレームを、関節ごとに前後の mask でないフレームから球面線形補間する
    （端は最も近いフレームの値のまま）。"""
    q = np.array(q, np.float64, copy=True)
    T = len(q)
    for j in range(q.shape[1]):
        for s, e in runs(mask[:, j]):
            a, b = s - 1, e + 1
            if a < 0 and b >= T:
                continue
            if a < 0:
                q[s:e + 1, j] = q[b, j]
            elif b >= T:
                q[s:e + 1, j] = q[a, j]
            else:
                frac = (np.arange(s, e + 1) - a) / float(b - a)
                q[s:e + 1, j] = quat.slerp(np.broadcast_to(q[a, j], (len(frac), 4)),
                                           np.broadcast_to(q[b, j], (len(frac), 4)), frac)
    return quat.make_continuous(q)


def fill_min_accel(y, mask):
    """(T, 3) y の mask のフレームを、加速度（2 階差分）の二乗和が最小になるように、前後の値からつなぐ。"""
    y = np.array(y, np.float64, copy=True)
    T = len(y)
    known = ~np.asarray(mask, bool)
    if not mask.any() or known.sum() < 2 or T < 3:
        return y
    D = second_difference(T).tocsc()
    Dn, Dk = D[:, np.flatnonzero(~known)], D[:, np.flatnonzero(known)]
    A = (Dn.T @ Dn + sparse.identity(Dn.shape[1]) * 1e-12).tocsc()
    y[~known] = splu(A).solve(-(Dn.T @ (Dk @ y[known])))
    return y


def drop_long_runs(flags, max_len, name, long_runs):
    """max_len より長く続く外れは置き換えない（前後どちらが正しいのか決められない）。"""
    flags = flags.copy()
    for s, e in runs(flags):
        if e - s + 1 > max_len:
            flags[s:e + 1] = False
            long_runs.append((name, s, e))
    return flags


def remove_outliers(quats, root_pos, valid, rest_joints, parents, fps, cfg, gravity_rotation=None,
                    depth_axis=None):
    """ステージ1b。quats (T, J, 4)・root_pos (T, 3) [m] はステージ1の出力（Y 上向き）。

    gravity_rotation: (3, 3) 床の傾き補正（床の法線を +Y へ回す。重力の向きを決める）。None なら +Y を上とする。
    depth_axis: 補正後の座標での奥行きの軸（カメラの光軸）。None なら Z 軸。
    """
    quats = quat.make_continuous(np.asarray(quats, np.float64))
    root_pos = np.asarray(root_pos, np.float64)
    observed = np.asarray(valid, bool)
    T = len(quats)
    R = np.eye(3) if gravity_rotation is None else np.asarray(gravity_rotation, np.float64)
    axis = np.array([0.0, 0.0, 1.0]) if depth_axis is None else np.asarray(depth_axis, np.float64)
    names = [p[0] for p in PARTS] + ['com']
    empty = {n: np.zeros(T, bool) for n in names}
    if not cfg.enabled or T < 8:
        return OutlierResult(quats, root_pos, observed.copy(), empty,
                             {n: np.zeros(T) for n in names}, enabled=False)

    max_len = int(np.floor(float(cfg.max_run_sec) * fps))
    max_ratio = float(cfg.max_ratio)
    # フレーム数で決めた値（予測に使う前後のフレーム数・離して予測する間隔・向きの回転の速さの窓）は 30fps で決めたもの。
    # 高い fps では同じ時間になるように増やす（フレーム数のままだと、予測に使う範囲が短くなり、数フレーム続く外れの
    # 外側から予測できなくなる）
    neighbours = frames_for_fps(cfg.pose.neighbours, fps)
    max_skip = frames_for_fps(MAX_SKIP, fps)
    turn_span, turn_window = frames_for_fps(3, fps), frames_for_fps(11, fps, odd=True)
    n_obs = max(int(observed.sum()), 1)
    flags, scores, skipped, long_runs = dict(empty), {n: np.zeros(T) for n in names}, {}, []

    def accept(name, f, dropped=()):
        long_runs.extend((name, s, e) for s, e in dropped)
        f = drop_long_runs(f, max_len, name, long_runs)
        if f.sum() > max_ratio * n_obs:
            skipped[name] = (f'外れが {f.sum() / n_obs * 100:.0f}% と多すぎるので使いません'
                             f'（outliers.max_ratio = {max_ratio}）')
            return np.zeros(T, bool)
        return f

    # ---- 姿勢（慣性） ----
    G, P = forward_kinematics(quats, root_pos, rest_joints, parents)
    joint_mask = np.zeros(quats.shape[:2], bool)
    if cfg.pose.enabled:
        for name, base, joints, axes_of, replace_joints, _ in PARTS:
            y = part_points(G, P, base, joints, axes_of, float(cfg.pose.axis_length_m))
            dropped = []
            f, scores[name] = detect_part(y, observed, float(cfg.pose.threshold),
                                          float(cfg.pose.min_m), max_skip, max_len, neighbours,
                                          turn_rate(G, axes_of, turn_span, turn_window), dropped)
            flags[name] = accept(name, f, dropped)
            joint_mask[:, list(replace_joints)] |= flags[name][:, None]
        quats = interpolate_rotations(quats, joint_mask)
        G, P = forward_kinematics(quats, root_pos, rest_joints, parents)

    # ---- 重心（重力） ----
    w_com = com_weights(P.shape[1])
    com_cam = np.einsum('tjc,j->tc', P, w_com)
    offset = com_cam - root_pos                    # 骨盤 → 重心（置き換えた姿勢）
    com = com_cam @ R.T                            # 重力の向きにそろえる
    fit = None
    reposition = flags['torso'].copy()             # 骨盤の位置を置き直すフレーム
    if cfg.gravity.enabled:
        use = observed & ~flags['torso']
        dropped = []
        f, scores['com'], fit = detect_com(com, use, fps, axis, cfg.gravity, max_len, dropped,
                                           max_skip=max_skip, per_side=frames_for_fps(3, fps))
        flags['com'] = accept('com', f, dropped)
        reposition |= flags['com']
    if reposition.any() and not reposition.all():
        good = ~reposition
        t = np.arange(T)
        target = fill_min_accel(com, reposition)
        if fit is not None and 'com' not in skipped:
            # 重力の条件を満たす軌道 + 前後のフレームでの推定との差（直線でつなぐ）。先頭・末尾の外れ（片側に
            # 外れていないフレームが無い）は、重力の条件だけでは軌道が決まらないので、加速度が最小になるように延ばす
            resid = com - fit
            inner = (t > t[good][0]) & (t < t[good][-1])
            bridged = fit + np.stack([np.interp(t, t[good], resid[good, i]) for i in range(3)],
                                     axis=1)
            target = np.where(inner[:, None], bridged, target)
        new_root = target @ R - offset
        root_pos = np.where(reposition[:, None], new_root, root_pos)

    result = OutlierResult(quats, root_pos, observed.copy(), flags, scores, com=com,
                           com_fit=fit, depth_axis=axis, skipped=skipped, long_runs=long_runs)
    result.valid = observed & ~result.unobserved
    return result
