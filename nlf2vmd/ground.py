"""ステージ6a: 接地の拘束（全身の上下のずれを除き、少なくとも片足が床に着くようにする）。

単眼動画からの推定では、カメラの視線に沿った距離（奥行き）がぶれ、ゆっくりずれていく。カメラが骨盤より
高い位置にあったり見下ろしていたりすると、視線は斜め下を向いているので、このずれが上下の位置のずれにも
なる（例: 高さ 1.6m のカメラから 4m 先の人物で、距離が 30cm ずれると体全体が約 5cm 上下する）。
床の高さの補正（ステージ4）はシーケンス全体の定数なので、このずれは残り、両足が床から浮いた状態
（または埋まった状態）が数秒続く。浮いている間は接地判定からも漏れるので、足もロックされない。

人は重力に逆らって長く宙に浮いていられない（ジャンプの滞空は長くても 0.6 秒程度）。そこで、両足の
最下点の高さ h(t)（かかと・つま先の低いほう）から上下のずれを求め、体全体を上下に平行移動して打ち消す。

1. h のメディアン（窓幅 5）で、1〜2 フレームだけ足が床の下へ飛ぶ推定の破綻を除く
2. 移動最小値（窓幅は最長の滞空時間より長い）で「下側の包絡線」を作る。ジャンプの区間も、窓の中に
   接地しているフレームがあるので包絡線は持ち上がらない
3. 包絡線より flight_height_m 以上高いフレームを「滞空」の候補とし、物理的にジャンプとしてありうるものだけを
   滞空とする。それ以外は「支持あり」（推定のずれ・揺れで足が浮いて見えるだけ）に戻す
   - max_flight_sec より長い候補: 人は長く宙に浮いていられない
   - min_flight_sec より短い候補: 骨盤の動きからは確かめられない。滞空しても高さは 3cm 未満なので床に着けてよい
   - 骨盤の上下の加速度（候補の区間に 2 次式を当てはめたもの）が -flight_accel_g の範囲に無い候補: 宙に浮いている
     体は重力だけで動くので、骨盤は加速度 -g の放物線を描く。足の向きや膝の推定の揺れで足先だけが上がった区間
     （骨盤は上がらない）や、上下にガタつく推定の揺れ（加速度が大きすぎる）を除く
   候補の区間に揺れの山がつながっていることがあるので、候補の中の足が最も高いフレームを含む部分区間を調べ、
   条件を満たす最も長いものを滞空とする（残りは支持あり）
   （足の高さの揺れの山まで滞空とすると、その区間は前後の谷の値でつながれて全身が数 cm 浮き、接地判定からも漏れる）
4. 支持ありのフレームでは h そのものを上下のずれとし（足がちょうど床に着く）、滞空の区間はその前後の値を
   直線でつなぐ（ジャンプの高さはそのまま残る）。最後にガウシアンで軽くならす
   （包絡線そのものをずれとすると、ゆっくり浮き上がる区間で移動最小値の窓の半分だけ遅れ、数 cm 浮いたまま残る）

補正は体全体（全関節・接地点）の上下の平行移動だけで、姿勢は変えない。
"""
from dataclasses import dataclass

import numpy as np

from . import filters


@dataclass
class GroundResult:
    lowest: np.ndarray       # (T,) 補正前の両足の最下点の高さ
    offset: np.ndarray       # (T,) 上下のずれ。体全体からこの値を引く
    flight: np.ndarray       # (T,) bool ジャンプとして残した滞空のフレーム
    window: int              # 移動最小値の窓幅 [フレーム]
    enabled: bool

    def apply(self, kin):
        if not self.enabled:
            return kin
        shift = np.zeros((len(self.offset), 3))
        shift[:, 1] = -self.offset
        return kin.transformed(None, shift)


def lowest_foot_height(kin):
    """(T,) 両足のかかと・つま先のうち、最も低い点の高さ。"""
    h = np.asarray(kin.contact_points)[..., 1]
    return h.reshape(len(h), -1).min(axis=1)


def flight_window(fps, max_flight_sec):
    """滞空が max_flight_sec 以下の区間の中央のフレームからも、窓の中に接地しているフレームが入る窓幅（奇数）。"""
    n = int(np.ceil(float(max_flight_sec) * fps))
    return 2 * ((n + 2) // 2) + 1


def vertical_accel(y, s, e, fps):
    """フレーム s〜e（前後 1 フレームずつ含めて当てはめる）の y に 2 次式を当てはめた加速度 [単位/s²]。"""
    lo, hi = max(0, s - 1), min(len(y), e + 2)
    t = (np.arange(lo, hi) - 0.5 * (lo + hi - 1)) / fps
    return 2.0 * np.polyfit(t, y[lo:hi], 2)[0]


def is_ballistic(pelvis_y, s, e, fps, unit, cfg):
    """フレーム s〜e の滞空がジャンプとしてありうるか（長さと、骨盤の上下の加速度が重力に合うか）。"""
    n = e - s + 1
    if n > float(cfg.max_flight_sec) * fps or n < float(cfg.min_flight_sec) * fps:
        return False
    lo_g, hi_g = cfg.flight_accel_g
    a = -vertical_accel(pelvis_y, s, e, fps) / (9.8 * unit)    # 下向きを正、g 単位
    return float(lo_g) <= a <= float(hi_g)


def find_flight(h, pelvis_y, s, e, fps, unit, cfg):
    """滞空の候補 s〜e のうち、足が最も高いフレームを含み、ジャンプとしてありうる最も長い部分区間（無ければ None）。"""
    peak = s + int(np.argmax(h[s:e + 1]))
    longest = min(e - s + 1, int(np.floor(float(cfg.max_flight_sec) * fps)))
    shortest = int(np.ceil(float(cfg.min_flight_sec) * fps))
    for n in range(longest, shortest - 1, -1):
        for a in range(max(s, peak - n + 1), min(peak, e - n + 1) + 1):
            if is_ballistic(pelvis_y, a, a + n - 1, fps, unit, cfg):
                return a, a + n - 1
    return None


def ground_offset(kin, fps, unit, cfg):
    """ステージ6a。kin: 床 y=0 の座標（ステージ5 の後、MMD 単位）。unit: スケール係数。"""
    lowest = lowest_foot_height(kin)
    T = len(lowest)
    window = flight_window(fps, cfg.max_flight_sec)
    if not cfg.enabled or T == 0:
        return GroundResult(lowest, np.zeros(T), np.zeros(T, bool), window, False)
    h = filters.median_time(lowest, 5)
    envelope = filters.moving_min(h, window)
    flight = h - envelope > float(cfg.flight_height_m) * unit
    pelvis_y = filters.median_time(kin.root_pos[:, 1], 3)
    for s, e in filters.runs(flight):
        flight[s:e + 1] = False
        found = find_flight(h, pelvis_y, s, e, fps, unit, cfg)
        if found is not None:
            flight[found[0]:found[1] + 1] = True
    t = np.arange(T)
    offset = np.interp(t, t[~flight], h[~flight]) if (~flight).any() else envelope
    offset = filters.gaussian_time(offset, float(cfg.smooth_sec) * fps)
    return GroundResult(lowest, offset, flight, window, True)


def floating_frames(lowest, fps, height, max_flight_sec):
    """両足の最下点が height より高い状態が max_flight_sec より長く続くフレームの数（ジャンプより長い浮き）。"""
    limit = float(max_flight_sec) * fps
    return int(sum(e - s + 1 for s, e in filters.runs(np.asarray(lowest) > height)
                   if e - s + 1 > limit))
