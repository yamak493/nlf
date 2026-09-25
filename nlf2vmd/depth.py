"""奥行きのぶれへの対策（ステージ6の速度と、ステージ6b: 奥行きの再構成）。

単眼動画からの推定では、カメラの奥行き方向の位置が毎フレーム数 cm ぶれ、ゆっくりずれていく
（左右・上下は画像上の位置から決まるので安定している）。骨盤の奥行きがぶれると体全体が前後に揺れ、
床に着いている足も一緒に動く。すると水平速度が上がって接地判定が途切れ、ロックされない区間や
ロック位置の食い違いが Z 方向の足滑りになる。そこで次の 2 つを行う。

* 接地判定の速度（ステージ6）: 奥行き方向のぶれを 2 通りの方法で除き、速度の小さいほうを使う
    - 骨盤の奥行きを平滑化する（体全体の前後の揺れを除く）。着地・離地の瞬間の速度の変化は鋭いまま残るが、
      体が本当に前後に揺れる動き（ダンスの重心移動）も消えるので、床に着いた足が動いて見えることがある
    - 足の接地点の奥行きを直接平滑化する。床に着いた足の奥行きは一定なので、体が前後に揺れていても
      速度は 0 になる。ただし遊脚の速度の変化がなまるので、これだけだと接地の始まりと終わりを取りこぼす
* 奥行きの再構成（ステージ6b）: 骨盤の奥行きを、次の 3 つの重み付き最小二乗で求め直す
    - 接地中の足首の奥行きが区間内で一定（足は床の上で前後に動かない）
    - 推定された奥行きから大きく離れない（全体の移動の目安。弱い重み）
    - 奥行きの加速度が小さい（体の前後の動きはなめらか）
  接地中の骨盤の前後の動きは「姿勢から求めた足首→骨盤の相対位置」で決まり、推定の奥行きのぶれは入らない。
  補正は体全体（全関節・接地点）を奥行きの軸に沿って平行移動するだけで、姿勢は変えない

奥行きの軸はカメラの光軸（床の傾き補正の後の内部座標で表したもの）。傾き補正が無ければ Z 軸になる。
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from . import filters
from .body_model import ANKLES
from .floor import horizontal_speed


@dataclass
class DepthResult:
    axis: np.ndarray         # (3,) 奥行きの軸（内部座標の単位ベクトル）
    raw: np.ndarray          # (T,) 補正前の骨盤の奥行き
    depth: np.ndarray        # (T,) 求め直した骨盤の奥行き
    enabled: bool

    @property
    def correction(self):
        return self.depth - self.raw

    def apply(self, kin):
        """体全体を奥行きの軸に沿って平行移動する。"""
        if not self.enabled:
            return kin
        return kin.transformed(None, self.correction[:, None] * self.axis)


def depth_axis(floor_rotation):
    """カメラの奥行き方向を、床の補正（回転）の後の内部座標で表した単位ベクトル。

    ステージ1でカメラ座標は X 軸まわりに 180 度回すだけなので、床の補正前は Z 軸が光軸になる。
    """
    u = np.asarray(floor_rotation, np.float64)[:, 2]
    return u / np.linalg.norm(u)


def depth_jitter(root_pos, axis):
    """骨盤の位置 (T, 3) の細かいぶれの大きさ [奥行き, 横]（診断用。平滑化する前の入力に使う）。

    白色ノイズなら 2 階差分の標準偏差は元の √6 倍になることから推定する（なめらかな動きの 2 階差分は小さい）。
    横は、奥行きの軸と鉛直の両方に直交する水平の軸。
    """
    lateral = np.cross([0.0, 1.0, 0.0], axis)
    lateral /= np.linalg.norm(lateral)
    root_pos = np.asarray(root_pos, np.float64)
    if len(root_pos) < 3:
        return np.zeros(2)
    acc = np.diff(root_pos, 2, axis=0)
    return np.array([np.std(acc @ axis), np.std(acc @ lateral)]) / np.sqrt(6.0)


def _lowpass(x, fps, unit, cutoff_hz):
    # 遅れの無いローパス（One Euro の beta=0）。フィルタは m 単位で掛ける
    return filters.one_euro(x / unit, fps, float(cutoff_hz), 0.0) * unit


def detection_speeds(kin, axis, fps, unit, cfg):
    """接地判定に使う接地点の水平速度 (T, 2 足, 2 [かかと, つま先])。

    骨盤の奥行きを平滑化した場合と、接地点の奥行きを平滑化した場合の小さいほう。
    カットオフが 0 の方法は平滑化しない（両方 0 なら元の速度）。
    """
    pts = kin.contact_points
    by_root = pts
    if cfg.detect_root_cutoff_hz > 0:
        d = kin.root_pos @ axis
        shift = _lowpass(d, fps, unit, cfg.detect_root_cutoff_hz) - d
        by_root = pts + shift[:, None, None, None] * axis
    by_point = pts
    if cfg.detect_foot_cutoff_hz > 0:
        d = pts @ axis
        by_point = pts + (_lowpass(d, fps, unit, cfg.detect_foot_cutoff_hz) - d)[..., None] * axis
    return np.minimum(horizontal_speed(by_root, fps), horizontal_speed(by_point, fps))


def solve_depth(raw, rel, segments, fps, contact_sigma, prior_sigma, accel_sigma):
    """骨盤の奥行き r (T,) を重み付き最小二乗で求める（単位は m）。

    raw: (T,) 推定された骨盤の奥行き / rel: (T, 2) 足首の奥行き − 骨盤の奥行き（姿勢から求めたもの）
    segments: 足ごとの接地区間 [(開始, 終了), ...]
    未知数は r と、接地区間ごとの足首の奥行き L。残差は
      接地:   (r_t + rel_t − L) / contact_sigma      （区間内の各フレーム）
      事前:   (r_t − raw_t) / prior_sigma
      加速度: (r_{t−1} − 2 r_t + r_{t+1}) / (accel_sigma / fps²)
    """
    raw = np.asarray(raw, np.float64)
    T = len(raw)
    rows, cols, vals, rhs = [], [], [], []

    def add(terms, b):
        # terms: [(列の番号 (k,), 係数), ...] の k 行を追加する
        start = sum(len(r) for r in rhs)
        for c, v in terms:
            rows.append(np.arange(start, start + len(b)))
            cols.append(c)
            vals.append(np.broadcast_to(v, (len(b),)))
        rhs.append(b)

    t = np.arange(T)
    wp = 1.0 / prior_sigma
    add([(t, wp)], wp * raw)
    if T >= 3:
        wa = fps ** 2 / accel_sigma
        m = t[1:-1]
        add([(m - 1, wa), (m, -2.0 * wa), (m + 1, wa)], np.zeros(T - 2))
    wc = 1.0 / contact_sigma
    num_seg = 0
    for foot in range(2):
        for s, e in segments[foot]:
            f = np.arange(s, e + 1)
            add([(f, wc), (np.full(len(f), T + num_seg), -wc)], -wc * rel[f, foot])
            num_seg += 1
    b = np.concatenate(rhs)
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(b), T + num_seg))
    x = spsolve((A.T @ A).tocsc(), A.T @ b)
    return np.asarray(x[:T])


def reconstruct_depth(kin, kin_pose, contact, axis, fps, unit, cfg):
    """ステージ6b。kin: ステージ5の後（MMD 単位・床 y=0）/ contact: ステージ6の接地判定。

    kin_pose: 足首→骨盤の相対位置を求める姿勢。ステージ2で平滑化する前の姿勢を渡す（平滑化した姿勢では、
    遊脚の速い動きが接地の前後に混ざり、床に着いた足首が 1 歩あたり数 cm 前へずれる。それを止めようとして
    骨盤が後ろへ引き戻され、歩いた距離が短くなる）。単独フレームの外れ値はメディアン（窓幅 5）で除く。
    メディアンは単調な変化（床に着いた足首に対して骨盤が進む動き）をそのまま残す。線形の平滑化は
    幅 1 フレームのガウシアンでも接地の前後に遊脚の動きが混ざるので掛けない（姿勢のぶれは加速度の項でならす）。
    """
    raw = kin.root_pos @ axis
    if not cfg.reconstruct:
        return DepthResult(axis, raw, raw.copy(), False)
    rel = filters.median_time((kin_pose.joints[:, ANKLES] - kin_pose.root_pos[:, None]) @ axis, 5)
    depth = solve_depth(raw / unit, rel / unit, contact.segments, fps,
                        float(cfg.contact_sigma_m), float(cfg.prior_sigma_m),
                        float(cfg.accel_sigma_m_per_s2)) * unit
    return DepthResult(axis, raw, depth, True)
