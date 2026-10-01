"""ステージ6: 接地判定（ヒステリシス付き）。"""
from dataclasses import dataclass

import numpy as np

from .filters import runs
from .floor import horizontal_speed


@dataclass
class ContactResult:
    flags: np.ndarray       # (T, 2 足) bool
    segments: list          # 足ごとの [(開始, 終了), ...]（終了フレームを含む）
    heights: np.ndarray     # (T, 2 足, 2 [かかと, つま先]) [MMD 単位]
    speeds: np.ndarray      # (T, 2, 2) 水平速度 [MMD 単位/s]


def hysteresis(enter, stay):
    state = False
    out = np.zeros(len(enter), bool)
    for t in range(len(enter)):
        state = bool(stay[t]) if state else bool(enter[t])
        out[t] = state
    return out


def clean_flags(flags, fill_gap, min_len):
    """短い非接地の隙間を埋めてから、短い接地区間を捨てる。"""
    flags = np.array(flags, bool, copy=True)
    for s, e in runs(~flags):
        if s > 0 and e < len(flags) - 1 and e - s + 1 <= fill_gap:
            flags[s:e + 1] = True
    for s, e in runs(flags):
        if e - s + 1 < min_len:
            flags[s:e + 1] = False
    return flags


def without_frames(contact, mask, cfg):
    """mask のフレームで、どちらの足も接地していないことにした ContactResult。

    ステージ6a の滞空（ground.flight）を渡す。宙にある体はどの足も床に着いていないが、接地判定のヒステリシス
    （終了の高さ 5cm）で、踏み切り・着地の数フレームは接地が続いていることがある。除いた後に短い隙間を埋め、
    短い接地区間を捨て直す（clean_flags）。
    """
    mask = np.asarray(mask, bool)
    if not mask.any():
        return contact
    flags = np.array(contact.flags, bool, copy=True)
    flags[mask] = False
    for foot in range(flags.shape[1]):
        flags[:, foot] = clean_flags(flags[:, foot], int(cfg.fill_gap_frames),
                                     int(cfg.min_contact_frames)) & ~mask
    return ContactResult(flags, [runs(flags[:, f]) for f in range(flags.shape[1])], contact.heights,
                         contact.speeds)


def detect_contacts(points, fps, cfg, unit=1.0, speeds=None, valid=None):
    """points: (T, 2 足, P 点, 3)。床が y=0 の座標。しきい値 [m] には unit を掛けて使う。

    かかとまたはつま先のどちらかが「高さ < しきい値」かつ「水平速度 < しきい値」なら接地。
    接地の開始より終了の条件を緩くして、境目でのバタつきを防ぐ。
    speeds: (T, 2, P) の水平速度（奥行きのぶれを除いたもの）。None なら points から求める。
    valid: (T,) bool 人物を検出できたフレーム。False のフレーム（前後から補間しただけ）では接地を始めない
    （接地していた足は、終了の条件を満たす間は接地のまま続ける）。
    """
    points = np.asarray(points, np.float64)
    heights = points[..., 1]
    speeds = horizontal_speed(points, fps) if speeds is None else np.asarray(speeds, np.float64)
    enter = ((heights < cfg.enter_height_m * unit)
             & (speeds < cfg.enter_speed_m_per_s * unit)).any(-1)
    if valid is not None:
        enter &= np.asarray(valid, bool)[:, None]
    stay = ((heights < cfg.exit_height_m * unit)
            & (speeds < cfg.exit_speed_m_per_s * unit)).any(-1)
    flags = np.zeros(points.shape[:2], bool)
    segments = []
    for foot in range(points.shape[1]):
        f = hysteresis(enter[:, foot], stay[:, foot])
        f = clean_flags(f, int(cfg.fill_gap_frames), int(cfg.min_contact_frames))
        flags[:, foot] = f
        segments.append(runs(f))
    return ContactResult(flags, segments, heights, speeds)
