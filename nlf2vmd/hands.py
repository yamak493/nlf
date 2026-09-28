"""手の形（指）: MediaPipe Hands の手のランドマーク（21 点）から手の形を分けて、指ボーンのキーを作る。

ノートブック（mp4_to_mannequin_ja.ipynb のセル 12）では、hand_detect.py で体の手首の位置から手を切り出して
ランドマークを求め、hands_analysis.npz に保存する。このモジュールはそれを読んで（GPU・MediaPipe 不要）:

  1. 指ごとの曲げ角（人差し指〜小指は付け根・第 2・第 3 関節の曲げの和、親指は MCP・IP の和）から、
     指ごとに「伸び」「曲げ」の確率を出す（間は中間 = 軽く曲げた指）。親指は、指先が人差し指・中指の
     付け根側から離れていることも「伸び」の条件にする（握りこぶしの上に添えた親指を伸びとみなさない）
  2. 手の形（SHAPES の 9 種）ごとに、指の伸び・曲げのパターンに合う度合い（確率の幾何平均）を点にする。
     デフォルトは一定の点で、どの形にもはっきり当てはまらない手（中間の指がある手）はデフォルトになる
  3. 点を時間方向にならし（手が見えないフレームは使わない）、形を切り替えるたびに switch_cost を引いて、
     全体でいちばん点の高い形の並びを選ぶ（Viterbi）。1 フレームごとのランドマークは小さく写った手では
     大きくぶれるが、はっきりした変化はすぐに、あいまいな変化は長く続いたときだけ切り替わる。
     手が長く見えない区間はデフォルトにし、短く続く形は前後の形にまとめる
  4. 形ごとに決めた指の角度（設定の presets と angles）を、形が変わる所だけキーにする。
     キーは切り替えの始まりと終わりの 2 つで、その間はイーズイン・アウトで補間する

指の曲げの軸は PMX のボーン位置（手首・人指１・小指１から求めた手のひらの向きと、各指ボーンの向き）から
求めるので、モデルのボーンのローカル軸の設定に依らない。

保存した検出結果から作り直すには: python -m nlf2vmd.hands hands_analysis.npz --pmx モデル.pmx --merge motion_full.vmd
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit

from . import quat
from .filters import runs
from .vmd import BoneTrack, bone_interpolation, merge_bones, read_vmd, to_mmd_quat, write_vmd

SHAPES = ('default', 'thumb', 'index', 'middle', 'ring', 'pinky', 'fist', 'scissors', 'open')
LABELS = dict(default='デフォルト', thumb='親指', index='人差し指', middle='中指', ring='薬指',
              pinky='小指', fist='グー', scissors='チョキ', open='パー')
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
SIDES = ('左', '右')                 # 0 = 左手、1 = 右手（体の左右。MMD の 左・右 と同じ）
SIDE_NAMES = ('left', 'right')
STATES = ('open', 'straight', 'relaxed', 'curled')

# 手の形ごとの指（親指・人差し指・中指・薬指・小指）のパターン: 1 = 伸び / 0 = 曲げ / None = どちらでもよい。
# 親指の判定がいちばん不安定なので、親指が形を決める 親指・グー 以外では見ない
PATTERNS = {
    'thumb': (1, 0, 0, 0, 0),
    'index': (None, 1, 0, 0, 0),
    'middle': (None, 0, 1, 0, 0),
    'ring': (None, 0, 0, 1, 0),
    'pinky': (None, 0, 0, 0, 1),
    'fist': (0, 0, 0, 0, 0),
    'scissors': (None, 1, 1, 0, 0),
    'open': (None, 1, 1, 1, 1),
}

# MediaPipe Hands の 21 点: 0 = 手首、親指 1〜4（CMC・MCP・IP・先）、人差し指 5〜8（MCP・PIP・DIP・先）、
# 中指 9〜12、薬指 13〜16、小指 17〜20
FINGER_POINTS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
HAND_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
              (17, 18), (18, 19), (19, 20))

# 形を切り替えるキーの補間（イーズイン・アウト。MMD の補間曲線の制御点 0〜127）
EASE_INTERPOLATION = bone_interpolation(((40, 0), (88, 127)))


# ---- 検出結果の保存・読み込み ----
def save_analysis(path, screen, world, presence, handedness, fps, **extra):
    """手のランドマークの検出結果を npz に保存する（キーを作り直すときはこれだけあればよい）。

    screen: (T, 2, 21, 3) 画像上の位置 [px] と奥行き（手首からの相対値、x と同じ尺度）
    world: (T, 2, 21, 3) 3D [m] / presence: (T, 2) 手の存在スコア / handedness: (T, 2) 右手らしさ
    手の軸は 0 = 左手、1 = 右手（体の関節で決めたもの）。
    """
    np.savez_compressed(
        path, screen=np.asarray(screen, np.float32), world=np.asarray(world, np.float32),
        presence=np.asarray(presence, np.float32), handedness=np.asarray(handedness, np.float32),
        fps=np.float64(fps), **{k: np.asarray(v) for k, v in extra.items()})


def load_analysis(source):
    if isinstance(source, (str, Path)):
        with np.load(source, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    return dict(source)


# ---- 指の曲げ → 指ごとの確率 ----
def _angle_deg(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def finger_bends(landmarks):
    """指ごとの曲げ角の和 [度]（(..., 21, 3) → (..., 5)）。

    人差し指〜小指は 手首→MCP の向きに対する MCP・PIP・DIP の 3 関節の曲げ、親指は MCP・IP の 2 関節
    （CMC は、手首→CMC→MCP の角度が伸ばしていても大きいので含めない）。
    """
    lm = np.asarray(landmarks, np.float64)
    out = []
    for f, idx in enumerate(FINGER_POINTS):
        pts = np.concatenate([lm[..., :1, :], lm[..., list(idx), :]], axis=-2)
        seg = np.diff(pts, axis=-2)                       # (..., 4, 3)
        bend = _angle_deg(seg[..., :-1, :], seg[..., 1:, :])  # (..., 3)
        out.append(bend[..., 1:].sum(-1) if f == 0 else bend.sum(-1))
    return np.stack(out, axis=-1)


def thumb_away(landmarks):
    """親指の先と、人差し指・中指の MCP・PIP の最短距離 ÷ 手のひらの幅（人差し指 MCP〜小指 MCP）。"""
    lm = np.asarray(landmarks, np.float64)
    tip = lm[..., 4, :]
    d = np.min(np.stack([np.linalg.norm(tip - lm[..., j, :], axis=-1) for j in (5, 6, 9, 10)]),
               axis=0)
    width = np.linalg.norm(lm[..., 5, :] - lm[..., 17, :], axis=-1)
    return d / np.maximum(width, 1e-9)


def finger_probs(landmarks, cfg):
    """指ごとの (伸びの確率, 曲げの確率)。どちらも (..., 5)。

    人差し指〜小指は、曲げ角の和が extended_below_deg より小さいほど伸び、curled_above_deg より大きいほど
    曲げ（間は両方とも小さい = 中間）。親指は 伸び = 曲げが小さい かつ 指先が離れている、曲げ = 1 − 伸び。
    """
    bends = finger_bends(landmarks)
    s = max(float(cfg.softness_deg), 1e-6) / 4.0
    p_ext = expit((float(cfg.extended_below_deg) - bends) / s)
    p_curl = expit((bends - float(cfg.curled_above_deg)) / s)
    st = max(float(cfg.thumb_softness_deg), 1e-6) / 4.0
    sa = max(float(cfg.thumb_away_softness), 1e-6) / 4.0
    thumb = (expit((float(cfg.thumb_extended_below_deg) - bends[..., 0]) / st)
             * expit((thumb_away(landmarks) - float(cfg.thumb_away)) / sa))
    p_ext[..., 0] = thumb
    p_curl[..., 0] = 1.0 - thumb
    return p_ext, p_curl


def shape_scores(p_ext, p_curl, cfg):
    """手の形ごとの点 (..., 9)（SHAPES の順）。パターンの指ごとの確率の幾何平均。デフォルトは一定。"""
    out = np.empty(p_ext.shape[:-1] + (len(SHAPES),))
    out[..., 0] = float(cfg.default_score)
    for k, shape in enumerate(SHAPES[1:], start=1):
        logs = [np.log(np.clip(p_ext[..., f] if want else p_curl[..., f], 1e-6, 1.0))
                for f, want in enumerate(PATTERNS[shape]) if want is not None]
        out[..., k] = np.exp(np.mean(logs, axis=0))
    return out


# ---- 時間方向 ----
def smooth_scores(scores, weights, sigma_frames):
    """見えているフレームだけで（重み weights で）ならした点 (T, K)。近くに見えているフレームが無い所は NaN。"""
    scores = np.asarray(scores, np.float64)
    w = np.asarray(weights, np.float64)
    if sigma_frames <= 0:
        out = scores.copy()
        out[w <= 0] = np.nan
        return out
    num = gaussian_filter1d(scores * w[:, None], sigma_frames, axis=0, mode='nearest')
    den = gaussian_filter1d(w, sigma_frames, mode='nearest')
    out = num / np.maximum(den, 1e-9)[:, None]
    out[den < 0.05] = np.nan
    return out


def viterbi(log_scores, switch_cost):
    """形を切り替えるたびに switch_cost を引いたときに、点（対数）の和がいちばん大きい形の並び (T,)。

    log_scores: (T, K)。NaN のフレーム（手が見えない）はどの形も 0 点（前の形が続く）。
    """
    ls = np.nan_to_num(np.asarray(log_scores, np.float64), nan=0.0)
    T, K = ls.shape
    if T == 0:
        return np.zeros(0, np.int64)
    back = np.zeros((T, K), np.int64)
    total = ls[0].copy()
    stay = np.arange(K)
    for t in range(1, T):
        best = int(np.argmax(total))
        switch = total[best] - switch_cost
        moved = switch > total
        back[t] = np.where(moved, best, stay)
        total = np.where(moved, switch, total) + ls[t]
    path = np.empty(T, np.int64)
    path[-1] = int(np.argmax(total))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def fill_missing(labels, missing_mask, max_gap, missing):
    """手が max_gap フレームより長く見えない区間を missing の形にする（'default'、または 'hold' =
    直前の形のまま）。それ以下の区間は前後の形のまま（viterbi がつないだもの）。"""
    labels = np.array(labels, np.int64, copy=True)
    default = SHAPES.index('default')
    for s, e in runs(missing_mask):
        if e - s + 1 <= max_gap:
            continue
        if missing == 'hold' and s > 0:
            labels[s:e + 1] = labels[s - 1]
        else:
            labels[s:e + 1] = default
    return labels


def label_runs(labels):
    """[(開始, 終了, 形の番号), ...]（終了を含む）。"""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return []
    cut = np.flatnonzero(np.diff(labels) != 0) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut - 1, [len(labels) - 1]])
    return [(int(s), int(e), int(labels[s])) for s, e in zip(starts, ends)]


def merge_short(labels, scores, min_len):
    """min_len フレームより短く続く形を、前後の形のうちその区間の点が高いほうにまとめる。"""
    labels = np.array(labels, np.int64, copy=True)
    if min_len <= 1:
        return labels
    while True:
        rs = label_runs(labels)
        if len(rs) < 2:
            return labels
        lengths = [e - s + 1 for s, e, _ in rs]
        i = int(np.argmin(lengths))
        if lengths[i] >= min_len:
            return labels
        s, e, _ = rs[i]
        cands = [rs[j][2] for j in (i - 1, i + 1) if 0 <= j < len(rs)]
        seg = scores[s:e + 1]

        def score(k):
            v = seg[:, k]
            v = v[np.isfinite(v)]
            return float(v.mean()) if len(v) else -np.inf

        labels[s:e + 1] = max(cands, key=score)


def classify(scores, weights, fps, cfg):
    """1 つの手の形の番号 (T,) と、ならした点 (T, K)。weights: フレームごとの重み（0 = 手が見えない）。"""
    if cfg.missing_shape not in ('default', 'hold'):
        raise ValueError(f'hands.missing_shape は default / hold のいずれかです: {cfg.missing_shape}')
    smoothed = smooth_scores(scores, weights, float(cfg.smooth_sec) * fps)
    labels = viterbi(np.log(np.clip(smoothed, 1e-6, None)), float(cfg.switch_cost))
    missing = ~np.isfinite(smoothed).all(axis=1)
    labels = fill_missing(labels, missing, int(round(float(cfg.max_gap_sec) * fps)),
                          cfg.missing_shape)
    labels = merge_short(labels, smoothed, int(round(float(cfg.min_hold_sec) * fps)))
    return labels, smoothed


def resample_labels(labels, fps_in, fps_out, num_frames):
    """形の番号を、出力のフレーム（時刻 k / fps_out）にいちばん近い入力フレームの値にする。"""
    labels = np.asarray(labels)
    idx = np.clip(np.round(np.arange(num_frames) * fps_in / fps_out).astype(np.int64), 0,
                  len(labels) - 1)
    return labels[idx]


def apply_edits(labels, edits, fps):
    """手で直した形を上書きする。edits: (開始 [秒], 終了 [秒], 手, 形) の並び。

    手は 'left' / 'right' / 'both'（または 0 / 1）、形は SHAPES の名前か番号。終了の時刻のフレームは含まない。
    """
    labels = np.array(labels, np.int64, copy=True)
    for start, end, side, shape in edits or ():
        k = SHAPES.index(shape) if isinstance(shape, str) else int(shape)
        if side in ('both', None):
            cols = [0, 1]
        else:
            cols = [SIDE_NAMES.index(side) if isinstance(side, str) else int(side)]
        a = max(0, int(round(float(start) * fps)))
        b = min(len(labels), int(round(float(end) * fps)))
        labels[a:b, cols] = k
    return labels


_SIDE_WORDS = {'left': 'left', '左': 'left', '左手': 'left', 'right': 'right', '右': 'right',
               '右手': 'right', 'both': 'both', '両': 'both', '両手': 'both'}


def parse_edits(text):
    """「開始-終了 手 形」を ; か改行で区切って並べた文字列を、apply_edits の edits にする。

    例: '12.0-13.5 右 チョキ; 20-21 both fist'。手は 左 / 右 / 両手（left / right / both）、
    形は SHAPES の名前か LABELS の日本語（グー・チョキ・パー・デフォルト・親指・人差し指・中指・薬指・小指）。
    """
    names = {**{k: k for k in SHAPES}, **{v: k for k, v in LABELS.items()}}
    edits = []
    for item in str(text or '').replace('；', ';').replace('\n', ';').split(';'):
        item = item.strip()
        if not item:
            continue
        parts = item.replace('〜', '-').replace('～', '-').split()
        try:
            span, side, shape = parts
            start, end = (float(v) for v in span.split('-'))
            side, shape = _SIDE_WORDS[side.lower()], names[shape.lower()]
        except (ValueError, KeyError):
            raise ValueError(f'手の形の上書きは「開始-終了 手 形」の形で書いてください（例: 12.0-13.5 右 チョキ）: '
                             f'{item}') from None
        if end <= start:
            raise ValueError(f'終了は開始より後にしてください: {item}')
        edits.append((start, end, side, shape))
    return edits


# ---- 指ボーンの回転 ----
def preset_angles(cfg):
    """形ごと・指ごとの [付け根, 第 2, 第 3, 開き] の角度 [度] (9, 5, 4)。"""
    table = np.zeros((len(SHAPES), len(FINGERS), 4))
    for k, shape in enumerate(SHAPES):
        states = list(cfg.presets[shape])
        if len(states) != len(FINGERS):
            raise ValueError(f'hands.presets.{shape} は 5 本の指の状態を書いてください: {states}')
        for f, (finger, state) in enumerate(zip(FINGERS, states)):
            if state not in STATES:
                raise ValueError(f'hands.presets.{shape} の「{state}」は {" / ".join(STATES)} のいずれかです')
            table[k, f] = np.asarray(cfg.angles[finger][state], np.float64)
    return table


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _perp(v, d):
    """v の d に垂直な成分（単位ベクトル）。"""
    return _unit(v - np.dot(v, d) * d)


@dataclass
class HandRig:
    """片手の指ボーンと、その曲げ・開きの軸（内部座標、初期姿勢）。"""
    side: int
    bones: list             # 指ごとのボーン名のリスト（モデルにあるものだけ）
    joints: list            # 指ごとの、各ボーンの関節番号（0 = 付け根）
    bend_axes: list         # 指ごとの (N, 3)
    spread_axes: list       # 指ごとの (3,) 付け根のボーンの開きの軸（親指は None）
    palm_normal: np.ndarray

    def local_quats(self, angles):
        """angles (T, 5, 4) [度] → {ボーン名: (T, 4) ローカル回転（内部座標）}。

        付け根のボーンは、曲げてから手のひらの面内で開く（開きの軸は手のひらに固定）。
        """
        angles = np.deg2rad(np.asarray(angles, np.float64))
        out = {}
        for f in range(len(FINGERS)):
            for name, j, axis in zip(self.bones[f], self.joints[f], self.bend_axes[f]):
                q = quat.from_rotvec(angles[:, f, j, None] * axis)
                if j == 0 and self.spread_axes[f] is not None:
                    q = quat.mul(quat.from_rotvec(angles[:, f, 3, None] * self.spread_axes[f]), q)
                out[name] = quat.normalize(q)
        return out

    @property
    def bone_names(self):
        return [b for bs in self.bones for b in bs]


def _bone_direction(skel, name, next_name, prev_dir):
    p = skel.internal(name)
    if next_name is not None and skel.has(next_name):
        d = skel.internal(next_name) - p
    else:
        tail = skel.tail_internal(name)
        d = tail - p if tail is not None else None
    if d is None or np.linalg.norm(d) < 1e-6:
        return prev_dir
    return _unit(d)


def build_rig(skel, side, cfg, warn=None):
    """PMX の骨格から片手の HandRig を作る（手首・人指１・小指１が無ければ None）。"""
    warn = warn or (lambda msg: None)
    s = SIDES[side]
    names = {f: [s + n for n in cfg.bones[f]] for f in FINGERS}
    wrist = s + '手首'
    idx1, pky1 = names['index'][0], names['pinky'][0]
    missing = [n for n in (wrist, idx1, pky1) if not skel.has(n)]
    if missing:
        warn(f'モデルに {"・".join(missing)} が無いので、{s}手の指のキーは打ちません')
        return None
    w = skel.internal(wrist)
    a, b = skel.internal(idx1) - w, skel.internal(pky1) - w
    # 内部座標（右手系）で、手のひら側の法線。左手は (小指 − 手首) × (人指 − 手首)、右手はその鏡像
    normal = _unit(np.cross(b, a) if side == 0 else np.cross(a, b))
    to_pinky = _perp(skel.internal(pky1) - skel.internal(idx1), normal)   # 手のひらの面内で小指側
    across = list(cfg.thumb_across)
    bones, joints, bend_axes, spread_axes = [], [], [], []
    for f, finger in enumerate(FINGERS):
        chain = names[finger]
        fb, fj, fa = [], [], []
        prev_dir = _unit(skel.internal(chain[0]) - w) if skel.has(chain[0]) else None
        for j, name in enumerate(chain):
            if not skel.has(name):
                continue
            nxt = chain[j + 1] if j + 1 < len(chain) else None
            d = _bone_direction(skel, name, nxt, prev_dir)
            if d is None:
                continue
            prev_dir = d
            target = normal + (float(across[j]) * to_pinky if f == 0 else 0.0)
            fb.append(name)
            fj.append(j)
            fa.append(_unit(np.cross(d, _perp(target, d))))
        lost = [n for n in chain if n not in fb]
        if lost:
            warn(f'モデルに {"・".join(lost)} が無いので、そのボーンのキーは打ちません')
        bones.append(fb)
        joints.append(fj)
        bend_axes.append(np.array(fa).reshape(-1, 3))
        spread = None
        if f > 0 and fb and fj[0] == 0:
            d0 = _bone_direction(skel, fb[0], fb[1] if len(fb) > 1 else None, None)
            spread = _unit(np.cross(d0, _perp(-to_pinky, d0)))   # + で親指側へ
        spread_axes.append(spread)
    return HandRig(side, bones, joints, bend_axes, spread_axes, normal)


# ---- キー ----
def transition_keys(labels, fps, transition_sec):
    """形の切り替えのキー [(フレーム, 形の番号, イーズの有無), ...]。

    切り替えは形が変わるフレームを中心に transition_sec かける（前後の形が続く長さの半分まで）。
    切り替えの始まりに前の形、終わりに次の形のキーを打ち、終わりのキーの補間をイーズにする。
    """
    rs = label_runs(labels)
    if not rs:
        return []
    n = max(1, int(round(float(transition_sec) * fps)))
    before, after = n // 2, n - n // 2
    keys = [(0, rs[0][2], False)]
    for (s0, e0, a), (s1, e1, b) in zip(rs[:-1], rs[1:]):
        a_key = max(s1 - before, (s0 + e0 + 1) // 2)
        b_key = min(s1 + after, (s1 + e1 + 1) // 2)
        b_key = max(b_key, a_key + 1)
        if a_key > keys[-1][0]:
            keys.append((a_key, a, False))
        keys.append((b_key, b, True))
    return keys


def hand_tracks(labels, rig, table, fps, cfg):
    """片手の形の番号 (T,) から指ボーンのキー列（BoneTrack のリスト）。"""
    keys = transition_keys(labels, fps, cfg.transition_sec)
    frames = np.array([k[0] for k in keys], np.int64)
    angles = table[[k[1] for k in keys]]                      # (N, 5, 4)
    interp = np.stack([EASE_INTERPOLATION if k[2] else bone_interpolation() for k in keys])
    zeros = np.zeros((len(keys), 3))
    return [BoneTrack(name, frames, zeros, to_mmd_quat(quat.make_continuous(q)), interp)
            for name, q in rig.local_quats(angles).items()]


# ---- まとめ ----
@dataclass
class HandsResult:
    fps: float                # キーのフレームレート
    labels: np.ndarray        # (T, 2) キーのフレームごとの形の番号（0 = 左手、1 = 右手）
    tracks: list              # BoneTrack のリスト
    analysis_fps: float
    presence: np.ndarray      # (N, 2) 検出のフレームごと
    valid: np.ndarray         # (N, 2)
    p_ext: np.ndarray         # (N, 2, 5)
    p_curl: np.ndarray        # (N, 2, 5)
    scores: np.ndarray        # (N, 2, 9) ならした点
    config: object
    info: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def segments(self, side):
        """[(開始 [秒], 終了 [秒], 形の名前), ...]（あとで手で直すときの一覧）。"""
        return [(s / self.fps, (e + 1) / self.fps, SHAPES[k])
                for s, e, k in label_runs(self.labels[:, side])]


def build_hands(analysis, cfg, skeleton=None, fps=30.0, num_frames=None, edits=None, log=print):
    """手のランドマークの検出結果（save_analysis の npz のパスか dict）から、指ボーンのキーを作る。

    cfg: 設定の hands の部分 / skeleton: 対象モデルの骨格（Skeleton。None なら形の判定だけ）
    num_frames: キーを打つフレーム数（None なら検出の長さ × fps）/ edits: apply_edits の形の上書き
    """
    log = log or (lambda *a: None)
    warns = []

    def warn(msg):
        warns.append(msg)
        log('⚠️ ' + msg)

    d = load_analysis(analysis)
    fps_in = float(d['fps'])
    lm = np.asarray(d['world'] if cfg.landmarks == 'world' else d['screen'], np.float64)
    presence = np.asarray(d['presence'], np.float64)
    valid = (presence >= float(cfg.min_presence)) & np.isfinite(lm).all(axis=(-1, -2))
    weights = np.where(valid, presence, 0.0)
    lm = np.nan_to_num(lm)
    p_ext, p_curl = finger_probs(lm, cfg)
    raw = shape_scores(p_ext, p_curl, cfg)
    if num_frames is None:
        num_frames = max(1, int(round(len(presence) * fps / fps_in)))
    labels = np.zeros((num_frames, 2), np.int64)
    smoothed = np.zeros_like(raw)
    for side in range(2):
        lab, smoothed[:, side] = classify(raw[:, side], weights[:, side], fps_in, cfg)
        labels[:, side] = resample_labels(lab, fps_in, fps, num_frames)
    if edits:
        labels = apply_edits(labels, edits, fps)

    sides = {'both': (0, 1), 'left': (0,), 'right': (1,)}.get(cfg.sides)
    if sides is None:
        raise ValueError(f'hands.sides は both / left / right のいずれかです: {cfg.sides}')
    tracks = []
    if skeleton is None:
        warn('PMX が無いので、指のキーは作りません（形の判定だけ行います）')
    else:
        table = preset_angles(cfg)
        for side in sides:
            rig = build_rig(skeleton, side, cfg, warn)
            if rig is not None:
                tracks += hand_tracks(labels[:, side], rig, table, fps, cfg)

    info = dict(frames=num_frames, fps=fps, visible={
        SIDE_NAMES[s]: round(float(valid[:, s].mean()), 3) if len(valid) else 0.0
        for s in range(2)})
    for s in range(2):
        rs = label_runs(labels[:, s])
        info[SIDE_NAMES[s]] = {LABELS[k]: round(sum(e - b + 1 for b, e, x in rs if x == i) / fps, 2)
                               for i, k in enumerate(SHAPES)
                               if any(x == i for _, _, x in rs)}
        info[SIDE_NAMES[s] + '_changes'] = max(0, len(rs) - 1)
    info['keys'] = sum(len(t.frames) for t in tracks)
    return HandsResult(fps, labels, tracks, fps_in, presence, valid, p_ext, p_curl, smoothed, cfg,
                       info, warns)


def make_hands(analysis, pmx=None, config=None, overrides=None, plot_path=None, num_frames=None,
               edits=None, log=print):
    """指ボーンのキーを作る（VMD は書き出さない。result.tracks を variants.write_variant などに渡す）。

    pmx: モデルの .pmx（指ボーンの位置から曲げの軸を求める）/ config・overrides: load_config と同じ
    num_frames: キーを打つフレーム数（体の動きの VMD とそろえる）/ edits: apply_edits の形の上書き
    """
    from .config import load_config
    from .pmx import read_pmx
    from .skeleton import Skeleton

    log = log or (lambda *a: None)
    cfg = load_config(config, overrides)
    model_name, skeleton = '', None
    if pmx:
        model = read_pmx(pmx)
        model_name, skeleton = model.name, Skeleton.from_pmx(model)
    result = build_hands(analysis, cfg.hands, skeleton, fps=float(cfg.input.target_fps),
                         num_frames=num_frames, edits=edits, log=log)
    result.info['model_name'] = model_name
    v = result.info['visible']
    log(f'[手の形] 手が見えたフレーム: 左手 {v["left"] * 100:.0f}% / 右手 {v["right"] * 100:.0f}%')
    for s, side in enumerate(SIDE_NAMES):
        parts = ' / '.join(f'{k} {sec:.1f} 秒' for k, sec in result.info[side].items())
        log(f'    {SIDES[s]}手: {parts}（切り替え {result.info[side + "_changes"]} 回）')
    log(f'    キー: {len(result.tracks)} ボーン / {result.info["keys"]} 個')
    if plot_path:
        try:
            result.info['plot'] = str(save_plot(result, plot_path))
        except ImportError:
            log('⚠️ matplotlib が無いのでグラフは出力しません')
    return result


def export_hands(analysis, out_path, motion_vmd=None, merged_path=None, pmx=None, config=None,
                 overrides=None, plot_path=None, edits=None, log=print):
    """指ボーンのキーを作り、out_path に指だけの VMD を、merged_path に motion_vmd（体の動き）と
    指を合わせた VMD を書き出す。戻り値は HandsResult（書き出したパスは info に入れる）。"""
    log = log or (lambda *a: None)
    num_frames = None
    if motion_vmd and Path(motion_vmd).exists():
        src = read_vmd(motion_vmd)
        if len(src.keys):
            num_frames = int(src.keys['frame'].max()) + 1
    result = make_hands(analysis, pmx=pmx, config=config, overrides=overrides,
                        plot_path=plot_path, num_frames=num_frames, edits=edits, log=log)
    model_name = result.info['model_name']
    if not model_name and motion_vmd and Path(motion_vmd).exists():
        model_name = read_vmd(motion_vmd).model_name
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = write_vmd(out_path, result.tracks, model_name or 'nlf2vmd')
    result.info['vmd'] = str(out_path)
    log(f'    指だけの VMD: {out_path}（キー {n}）')
    if motion_vmd and merged_path:
        if Path(motion_vmd).exists():
            n = merge_bones(motion_vmd, result.tracks, merged_path)
            result.info['merged_vmd'] = str(merged_path)
            log(f'    体の動き＋指の VMD: {merged_path}（キー {n}）')
        else:
            log(f'⚠️ {motion_vmd} が無いので、体の動きとの合成はしません')
    return result


# ---- グラフ ----
_COLORS = dict(default='0.75', thumb='tab:orange', index='tab:red', middle='tab:purple',
               ring='tab:pink', pinky='tab:brown', fist='tab:blue', scissors='tab:green',
               open='gold')
_EN = dict(default='default', thumb='thumb', index='index', middle='middle', ring='ring',
           pinky='pinky', fist='fist', scissors='scissors (V)', open='open')


def save_plot(result, path):
    """左右それぞれ: 形のタイムライン（上の帯）、指ごとの伸び（実線）・曲げ（点線）の確率（形の点と同じ
    幅でならしたもの）、手の見えないフレーム（灰色）。"""
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    r = result
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t_in = np.arange(len(r.presence)) / r.analysis_fps
    duration = max(len(r.labels) / r.fps, t_in[-1] if len(t_in) else 0.0, 1.0)
    fig = Figure(figsize=(min(40.0, max(12.0, duration / 2.5)), 7.5))
    axes = fig.subplots(2, 1, sharex=True)
    finger_colors = ('tab:orange', 'tab:red', 'tab:purple', 'tab:pink', 'tab:brown')
    sigma = float(r.config.smooth_sec) * r.analysis_fps
    for side, ax in enumerate(axes):
        for s, e, k in label_runs(r.labels[:, side]):
            ax.axvspan(s / r.fps, (e + 1) / r.fps, ymin=0.9, ymax=1.0, color=_COLORS[SHAPES[k]],
                       lw=0)
        for s, e in runs(~r.valid[:, side]):
            ax.axvspan(t_in[s], t_in[e] + 1.0 / r.analysis_fps, ymin=0.0, ymax=0.9, color='0.9',
                       lw=0)
        weights = np.where(r.valid[:, side], r.presence[:, side], 0.0)
        ext = smooth_scores(r.p_ext[:, side], weights, sigma)
        curl = smooth_scores(r.p_curl[:, side], weights, sigma)
        for f, finger in enumerate(FINGERS):
            ax.plot(t_in, ext[:, f] * 0.85, color=finger_colors[f], lw=1.0,
                    label=f'{finger} extended')
            if f > 0:
                ax.plot(t_in, curl[:, f] * 0.85, color=finger_colors[f], lw=0.7, ls=':')
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.425, 0.85])
        ax.set_yticklabels(['0', '0.5', '1'])
        ax.set_ylabel(f'{SIDE_NAMES[side]} hand')
        ax.set_title(f'{SIDE_NAMES[side]} hand: top band = hand shape; lines = finger extended '
                     '(solid) / curled (dotted) probability; gray = hand not visible',
                     fontsize=10)
    handles = [Patch(color=_COLORS[s], label=_EN[s]) for s in SHAPES]
    axes[0].legend(handles=handles, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8)
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8)
    axes[1].set_xlabel(f'time [s] (frame = time x {r.fps:g})')
    axes[1].set_xlim(0, duration)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


# ---- コマンドライン ----
def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        prog='python -m nlf2vmd.hands',
        description='ノートブックのセル 12 が保存した手のランドマークの検出結果（hands_analysis.npz）から、'
                    '指ボーンのキー（手の形）を VMD に書き出します。')
    ap.add_argument('analysis', help='hands_analysis.npz')
    ap.add_argument('--pmx', required=True, help='モデルの .pmx（指ボーンの位置から曲げの軸を求める）')
    ap.add_argument('-o', '--output', help='指だけの VMD（既定: 入力と同じフォルダの hands.vmd）')
    ap.add_argument('--merge', metavar='VMD', help='体の動きの VMD（motion_full.vmd など）。指のキーを足して '
                                                  '--merged-output に書き出す')
    ap.add_argument('--merged-output', help='--merge の書き出し先（既定: <VMD 名>_hands.vmd）')
    ap.add_argument('--config', help='設定ファイル（YAML / JSON）。書いた項目だけ既定値を上書き')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='設定を 1 項目上書き（例: --set hands.min_hold_sec=0.4）。複数指定可')
    ap.add_argument('--plot', help='グラフ（PNG）の保存先')
    ap.add_argument('--edit', action='append', default=[], metavar='"開始-終了 手 形"',
                    help='判定した形を上書きする（例: --edit "12.0-13.5 右 チョキ"）。複数指定可')
    args = ap.parse_args(argv)

    out = args.output or str(Path(args.analysis).with_name('hands.vmd'))
    merged = None
    if args.merge:
        merged = args.merged_output or str(Path(args.merge).with_name(
            Path(args.merge).stem + '_hands.vmd'))
    export_hands(args.analysis, out, motion_vmd=args.merge, merged_path=merged, pmx=args.pmx,
                 config=args.config, overrides=args.set, plot_path=args.plot,
                 edits=parse_edits(';'.join(args.edit)))
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
