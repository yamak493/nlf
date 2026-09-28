"""口パク（リップシンク）: 音素認識の結果から、MMD の口のモーフ（あ・い・う・え・お・ん）のキーを作る。

ノートブック（mp4_to_mannequin_ja.ipynb のセル 12）では、Demucs で動画の音声からボーカルだけを取り出し、
Allosaurus でボーカルの音素（IPA）とその時刻を認識して、このモジュールでモーフのキーにする。
Allosaurus の時刻は音素が鳴り始めるあたりの 1 点（CTC のスパイク。30ms 刻み）で、音素の長さは分からない。
長さ（声が続いている間）と口の開き具合は、ボーカルの音量から決める。

処理:
  1. 音素を口の形に分ける: a / i / u / e / o（IPA の母音を近い日本語の母音へ）、
     n（口を閉じる: 両唇音 m・b・p と撥音 ɴ）、それ以外の子音は形を持たない
  2. ボーカルの音量から声が出ている区間を決める（2 つのしきい値のヒステリシス・短い隙間を埋める・
     短い区間を捨てる）。声が出ていない間はどのモーフも 0（口を閉じる）。無音の所で認識された音素
     （伴奏の残りや息を音素と取り違えたもの）は使わない
  3. 声が出ている区間ごとに、各時刻の口の形を「直前に始まった音素の形」にする。子音は、すぐ後の母音の形を
     子音の時刻から始める（日本語の子音は次の母音の口の形のまま発音される）。区間の頭は最初の音素の形、
     母音が 1 つも認識されなかった区間は fallback_vowel
  4. 母音のモーフの値 = 口の開き（音量から min_open〜1）× strength。ん は closed_weight
  5. ガウシアンでなめらかにしてから、フレームごとに平均してキーにし、線形補間で済む所は間引く

保存した認識結果から作り直すには: python -m nlf2vmd.lipsync lipsync_analysis.npz --merge motion.vmd
"""
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .filters import runs
from .vmd import MorphTrack, merge_morphs, read_vmd, thin_weights, write_vmd

SHAPES = ('a', 'i', 'u', 'e', 'o', 'n')
KANA = dict(zip(SHAPES, 'あいうえおん'))
CLOSED = SHAPES.index('n')

# IPA の母音 → 近い日本語の母音（口の形）。唇を丸める母音は う・お に寄せる
_VOWELS = {
    'a': 'aɑɐæʌɶəɜɝɚ',
    'i': 'iɪɨ',
    'u': 'uɯʊʉyʏ',
    'e': 'eɛɘ',
    'o': 'oɔɒɤɵøœɞ',
}
VOWEL_SHAPE = {ch: s for s, chars in _VOWELS.items() for ch in chars}
CLOSED_PHONES = set('mbpɓʙɴ')                              # 唇を閉じる音（両唇音）と撥音
GLIDE_SHAPE = {'j': 'i', 'w': 'u', 'ɥ': 'u', 'ʍ': 'u'}   # 半母音は、続く母音の前に い・う の形が一瞬入る
_MODIFIERS = set('ːˑʰʲʷˠˤʼ˞')


def _base(phone):
    """ダイアクリティカルマーク・長音記号などを除いた音素記号（ɯ̃ → ɯ、aː → a）。"""
    s = unicodedata.normalize('NFD', str(phone))
    return ''.join(ch for ch in s if not unicodedata.combining(ch) and ch not in _MODIFIERS)


def phone_shape(phone):
    """音素（IPA）の口の形: 'a' / 'i' / 'u' / 'e' / 'o' / 'n'（口を閉じる）/ None（子音）。

    二重母音（aɪ など）は最初の母音にする。音節主音的な子音（ɻ̩ など）は子音として扱う。
    """
    base = _base(phone)
    for ch in base:
        if ch in VOWEL_SHAPE:
            return VOWEL_SHAPE[ch]
    if base[:1] in CLOSED_PHONES:
        return 'n'
    return GLIDE_SHAPE.get(base[:1])


def parse_allosaurus(text):
    """Allosaurus の recognize(..., timestamp=True) の出力（1 行 = '時刻 [秒] 長さ [秒] 音素'）を
    (時刻 (N,), 音素 (N,)) にする。長さは一定値（音素の長さではない）なので使わない。"""
    times, phones = [], []
    for line in str(text).splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            t = float(parts[0])
        except ValueError:
            continue
        times.append(t)
        phones.append(parts[2])
    return np.asarray(times, np.float64), np.asarray(phones, dtype=str)


def vocal_level_db(samples, sample_rate, rate=100.0, window_sec=0.03):
    """音量 [dB]（フルスケール = 0 dB）を rate [Hz] ごとに求める。samples: (N,) か (N, チャンネル)。"""
    x = np.asarray(samples)
    if np.issubdtype(x.dtype, np.integer):
        x = x / float(np.iinfo(x.dtype).max)
    x = np.asarray(x, np.float64).reshape(len(x), -1).mean(axis=1)
    n = int(np.ceil(len(x) / sample_rate * rate))
    centers = np.arange(n) / rate * sample_rate
    half = 0.5 * window_sec * sample_rate
    lo = np.clip(np.round(centers - half).astype(np.int64), 0, len(x))
    hi = np.clip(np.round(centers + half).astype(np.int64), 0, len(x))
    cs = np.concatenate([[0.0], np.cumsum(x * x)])
    ms = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    return 10.0 * np.log10(np.maximum(ms, 1e-12))


def vowel_text(times, phones, gap_sec=0.3):
    """認識した母音と ん をかなで並べる（確認用）。gap_sec 以上あいた所で区切り、[(開始時刻, かな)] を返す。"""
    lines = []
    last = -np.inf
    for t, p in zip(times, phones):
        shape = phone_shape(p)
        if shape is None:
            continue
        if t - last >= gap_sec or not lines:
            lines.append([float(t), ''])
        lines[-1][1] += KANA[shape]
        last = t
    return [tuple(x) for x in lines]


# ---- 認識結果の保存・読み込み ----
def save_analysis(path, phone_times, phones, level_db, level_rate, duration, **extra):
    """音素認識の結果とボーカルの音量を npz に保存する（キーを作り直すときはこれだけあればよい）。"""
    np.savez_compressed(
        path, phone_times=np.asarray(phone_times, np.float64),
        phones=np.asarray(phones, dtype=str), level_db=np.asarray(level_db, np.float32),
        level_rate=np.float64(level_rate), duration=np.float64(duration),
        **{k: np.asarray(v) for k, v in extra.items()})


def load_analysis(source):
    if isinstance(source, (str, Path)):
        with np.load(source, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    return dict(source)


# ---- 口の形とモーフの値 ----
def voice_levels(level_db, cfg):
    """音量の基準（大きな声）・雑音（伴奏の残り）と、声の区間のしきい値 [dB] を求める。"""
    level = np.asarray(level_db, np.float64)
    if len(level) == 0:
        return dict(reference=float('-inf'), noise=float('-inf'), on=float('inf'),
                    off=float('inf'))
    # 基準 = 大きな声の音量。伴奏の残りなどの小さな音（最大から voice_off_db より下）は除いて求める
    ref = float(np.percentile(level[level >= level.max() + cfg.voice_off_db],
                              cfg.reference_percentile))
    noise = float(np.percentile(level, cfg.noise_percentile))
    # しきい値は基準からの差と雑音からの差の大きいほう（Demucs の分離で伴奏が残っていても声と取り違えない）。
    # ただし基準+max_on_db を超えない（間の無い歌で雑音の見積もりが声になったとき、声まで捨てないため）
    on = min(max(ref + cfg.voice_on_db, noise + cfg.on_above_noise_db), ref + cfg.max_on_db)
    off = min(max(ref + cfg.voice_off_db, noise + cfg.off_above_noise_db), on)
    return dict(reference=ref, noise=noise, on=on, off=off)


def voice_activity(level_db, rate, cfg):
    """声が出ている時刻（bool (M,)）と、voice_levels の値を返す。基準が小さすぎれば（声が無い）すべて False。"""
    level = np.asarray(level_db, np.float64)
    levels = voice_levels(level, cfg)
    voiced = np.zeros(len(level), bool)
    if len(level) == 0 or levels['reference'] < cfg.min_reference_dbfs:
        return voiced, levels
    # 声の区間 = off より大きい連続区間のうち、on を超える時刻を含むもの（ヒステリシス）
    loud = level >= levels['on']
    for s, e in runs(level >= levels['off']):
        if loud[s:e + 1].any():
            voiced[s:e + 1] = True
    gap = int(round(cfg.fill_gap_sec * rate))
    segs = runs(voiced)
    for (_, e0), (s1, _) in zip(segs[:-1], segs[1:]):
        if s1 - e0 - 1 <= gap:
            voiced[e0 + 1:s1] = True
    min_len = int(round(cfg.min_voice_sec * rate))
    for s, e in runs(voiced):
        if e - s + 1 < min_len:
            voiced[s:e + 1] = False
    return voiced, levels


def _run_events(events, t_end, fallback, cfg):
    """声の区間の音素 [(時刻, 形)] から、口の形が切り替わる [(時刻, 形の番号)] を作る。"""
    out = []
    for k, (t, shape) in enumerate(events):
        if shape is None:
            # 子音: すぐ後の母音（か口を閉じる音）の形を、子音の時刻から始める
            nxt = next(((t2, s2) for t2, s2 in events[k + 1:] if s2 is not None), None)
            if nxt is None or nxt[0] - t > cfg.consonant_sec:
                continue
            shape = nxt[1]
        out.append((t, SHAPES.index(shape)))
    # 声が出ている間に口を閉じ続けるのは max_closed_sec まで（ɴ の誤認識で口が閉じたままになるのを防ぐ）
    fixed = []
    for k, (t, idx) in enumerate(out):
        fixed.append((t, idx))
        t_next = out[k + 1][0] if k + 1 < len(out) else t_end
        if idx != CLOSED or t_next - t <= cfg.max_closed_sec:
            continue
        back = next((i for _, i in reversed(fixed) if i != CLOSED), None)
        ahead = next((i for _, i in out[k + 1:] if i != CLOSED), None)
        vowel = back if back is not None else (ahead if ahead is not None else fallback)
        if vowel >= 0:
            fixed.append((t + cfg.max_closed_sec, vowel))
    return fixed


def shape_timeline(times, shapes, voiced, rate, cfg):
    """各時刻（rate [Hz] ごと）の口の形（SHAPES の番号。-1 = 口を閉じる・無音）。"""
    label = np.full(len(voiced), -1, np.int64)
    if cfg.fallback_vowel and cfg.fallback_vowel not in SHAPES:
        raise ValueError(f'lipsync.fallback_vowel は a / i / u / e / o / n か null です: '
                         f'{cfg.fallback_vowel}')
    fallback = SHAPES.index(cfg.fallback_vowel) if cfg.fallback_vowel else -1
    order = np.argsort(times, kind='stable')
    times = np.asarray(times, np.float64)[order]
    shapes = [shapes[i] for i in order]
    margin, lead = float(cfg.event_margin_sec), float(cfg.lead_sec)
    for s, e in runs(voiced):
        t0, t1 = s / rate, (e + 1) / rate
        sel = np.flatnonzero((times >= t0 - margin) & (times <= t1 + margin))
        changes = _run_events([(times[i], shapes[i]) for i in sel], t1, fallback, cfg)
        if not changes:
            label[s:e + 1] = fallback
            continue
        starts = (np.array([t for t, _ in changes]) - lead) * rate
        k = np.searchsorted(starts, np.arange(s, e + 1), side='right') - 1
        label[s:e + 1] = np.array([i for _, i in changes])[np.maximum(k, 0)]
    return label


def shape_weights(label, level_db, reference_db, rate, cfg):
    """口の形ごとのモーフの値 (len(SHAPES), M)。母音は音量で開き具合を変え、ん は一定。"""
    amp = np.clip(1.0 + (np.asarray(level_db, np.float64) - reference_db) / cfg.open_range_db,
                  0.0, 1.0)
    vowel = cfg.strength * (cfg.min_open + (1.0 - cfg.min_open) * amp)
    w = np.zeros((len(SHAPES), len(label)))
    for i in range(len(SHAPES)):
        on = label == i
        w[i, on] = cfg.closed_weight if i == CLOSED else vowel[on]
    sigma = float(cfg.smooth_sec) * rate
    if sigma > 0 and w.shape[1] > 1:
        w = gaussian_filter1d(w, sigma, axis=1, mode='nearest')
    return w


def to_frames(values, rate, fps, num_frames):
    """rate [Hz] ごとの値 (..., M) を、各フレームの前後 0.5 フレームの平均 (..., num_frames) にする。
    音声より後ろのフレームは 0。"""
    values = np.asarray(values, np.float64)
    M = values.shape[-1]
    idx = np.floor(np.arange(M) / rate * fps + 0.5).astype(np.int64)
    ok = idx < num_frames
    counts = np.bincount(idx[ok], minlength=num_frames).astype(np.float64)
    flat = values.reshape(int(np.prod(values.shape[:-1])), M)
    out = np.stack([np.bincount(idx[ok], weights=v[ok], minlength=num_frames) for v in flat])
    return (out / np.maximum(counts, 1.0)).reshape(values.shape[:-1] + (num_frames,))


@dataclass
class LipsyncResult:
    fps: float
    weights: np.ndarray      # (len(SHAPES), T) フレームごとのモーフの値（間引く前）
    tracks: list             # MorphTrack（名前が空の形・モデルに無いモーフは除く）
    voiced: np.ndarray       # (T,) 声が出ているフレーム
    level_db: np.ndarray     # (M,) ボーカルの音量（level_rate [Hz] ごと）
    level_rate: float
    voiced_level: np.ndarray  # (M,) 声が出ている時刻
    levels: dict             # 音量の基準・雑音・声の区間のしきい値 [dB]（voice_levels）
    phone_times: np.ndarray
    phones: np.ndarray
    config: object
    info: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def build_lipsync(analysis, cfg, fps=30.0, num_frames=None, available_morphs=None, log=print):
    """音素認識の結果（save_analysis の npz のパスか dict）から、口のモーフのキーを作る。

    cfg: 設定の lipsync の部分（load_config().lipsync）
    num_frames: キーを打つフレーム数（None なら音声の長さ × fps）
    available_morphs: モデルにあるモーフ名のリスト（None なら確かめない）
    """
    log = log or (lambda *a: None)
    warns = []

    def warn(msg):
        warns.append(msg)
        log('⚠️ ' + msg)

    d = load_analysis(analysis)
    times = np.asarray(d['phone_times'], np.float64).reshape(-1)
    phones = np.asarray(d['phones'], dtype=str).reshape(-1)
    level = np.asarray(d['level_db'], np.float64).reshape(-1)
    rate = float(d['level_rate'])
    if num_frames is None:
        num_frames = max(1, int(round(float(d['duration']) * fps)))
    shapes = [phone_shape(p) for p in phones]

    voiced, levels = voice_activity(level, rate, cfg)
    if not voiced.any():
        warn(f'声が見つかりませんでした（ボーカルの音量の基準 {levels["reference"]:.1f} dB が '
             f'lipsync.min_reference_dbfs = {cfg.min_reference_dbfs} dB 未満か、音声が空です）。'
             '口のモーフはすべて 0 にします')
    label = shape_timeline(times, shapes, voiced, rate, cfg)
    weights = to_frames(shape_weights(label, level, levels['reference'], rate, cfg), rate, fps,
                        num_frames)
    voiced_frames = to_frames(voiced.astype(np.float64), rate, fps, num_frames) >= 0.5

    names = dict(cfg.morphs)
    available = None if available_morphs is None else set(available_morphs)
    frames = np.arange(num_frames)
    tracks = []
    for i, shape in enumerate(SHAPES):
        name = names.get(shape) or ''
        if not name:
            continue
        if available is not None and name not in available:
            warn(f'モデルにモーフ「{name}」がありません（{KANA[shape]} の口の形のキーは打ちません。'
                 f'モーフ名は lipsync.morphs.{shape} で変えられます）')
            continue
        w = weights[i]
        keep = (thin_weights(w, cfg.thin_tolerance) if cfg.thin_tolerance > 0
                else np.ones(num_frames, bool))
        tracks.append(MorphTrack(name, frames[keep], w[keep]))

    used = np.zeros(len(times), bool)
    for s, e in runs(voiced):
        used |= (times >= s / rate - cfg.event_margin_sec) & (times <= (e + 1) / rate
                                                              + cfg.event_margin_sec)
    info = dict(
        phones=len(phones), phones_in_voice=int(used.sum()),
        vowels={KANA[s]: sum(1 for x in shapes if x == s) for s in SHAPES},
        consonants=sum(1 for x in shapes if x is None),
        voiced_sec=float(voiced.sum() / rate), voiced_runs=len(runs(voiced)),
        levels_db={k: round(v, 1) for k, v in levels.items()}, fps=fps, frames=num_frames,
        keys={t.name: len(t.frames) for t in tracks})
    return LipsyncResult(fps, weights, tracks, voiced_frames, level, rate, voiced, levels, times,
                         phones, cfg, info, warns)


def make_lipsync(analysis, pmx=None, config=None, overrides=None, plot_path=None, log=print):
    """口のモーフのキーを作る（VMD は書き出さない。result.tracks を variants.write_variant などに渡す）。

    pmx: モデルの .pmx（モーフ名があるか確かめる）/ config・overrides: load_config と同じ
    plot_path: グラフ（PNG）の保存先。モデル名は info['model_name']（PMX が無ければ空）
    """
    from .config import load_config
    from .pmx import read_pmx

    log = log or (lambda *a: None)
    cfg = load_config(config, overrides)
    model_name, available = '', None
    if pmx:
        model = read_pmx(pmx)
        model_name = model.name
        if model.morphs is None:
            log('⚠️ PMX のモーフを読めなかったので、モーフ名があるかは確かめません')
        else:
            available = model.morph_names()
    result = build_lipsync(analysis, cfg.lipsync, fps=float(cfg.input.target_fps),
                           available_morphs=available, log=log)
    result.info['model_name'] = model_name
    log(f'[口パク] 声の区間 {result.info["voiced_runs"]}（計 {result.info["voiced_sec"]:.1f} 秒）/ '
        f'使った音素 {result.info["phones_in_voice"]} / {result.info["phones"]}')
    log('    キー: ' + ' / '.join(f'{k} {v}' for k, v in result.info['keys'].items()))
    if plot_path:
        try:
            result.info['plot'] = str(save_plot(result, plot_path))
        except ImportError:
            log('⚠️ matplotlib が無いのでグラフは出力しません')
    return result


def export_lipsync(analysis, out_path, motion_vmd=None, merged_path=None, pmx=None, config=None,
                   overrides=None, plot_path=None, log=print):
    """口のモーフのキーを作り、out_path に口パクだけの VMD を、merged_path に motion_vmd（体の動き）と
    口パクを合わせた VMD を書き出す。戻り値は LipsyncResult（書き出したパスは info に入れる）。

    pmx: モデルの .pmx（モーフ名があるか確かめる）/ config・overrides: load_config と同じ
    """
    log = log or (lambda *a: None)
    result = make_lipsync(analysis, pmx=pmx, config=config, overrides=overrides,
                          plot_path=plot_path, log=log)
    model_name = result.info['model_name']
    if not model_name and motion_vmd and Path(motion_vmd).exists():
        model_name = read_vmd(motion_vmd).model_name
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = write_vmd(out_path, [], model_name or 'nlf2vmd', morphs=result.tracks)
    result.info['vmd'] = str(out_path)
    log(f'    口パクだけの VMD: {out_path}（キー {n}）')
    if motion_vmd and merged_path:
        if Path(motion_vmd).exists():
            n = merge_morphs(motion_vmd, result.tracks, merged_path)
            result.info['merged_vmd'] = str(merged_path)
            log(f'    体の動き＋口パクの VMD: {merged_path}（キー {n}）')
        else:
            log(f'⚠️ {motion_vmd} が無いので、体の動きとの合成はしません')
    return result


# ---- グラフ ----
_COLORS = dict(a='tab:red', i='tab:green', u='tab:blue', e='tab:orange', o='tab:purple',
               n='black')
_LABELS = dict(a='a', i='i', u='u', e='e', o='o', n='N (closed)')


def save_plot(result, path):
    """ボーカルの音量と声の区間・認識した音素（上）と、口のモーフの値（下）のグラフ。"""
    # pyplot を使わない（ノートブックの描画バックエンドを変えないため）
    from matplotlib.figure import Figure

    r, cfg = result, result.config
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t_level = np.arange(len(r.level_db)) / r.level_rate
    t_frame = np.arange(r.weights.shape[1]) / r.fps
    duration = max(t_level[-1] if len(t_level) else 0.0, t_frame[-1] if len(t_frame) else 0.0,
                   1.0)
    fig = Figure(figsize=(min(40.0, max(12.0, duration / 2.5)), 6.5))
    ax1, ax2 = fig.subplots(2, 1, sharex=True)

    for s, e in runs(r.voiced_level):
        ax1.axvspan(s / r.level_rate, (e + 1) / r.level_rate, color='tab:green', alpha=0.15, lw=0)
    ax1.plot(t_level, r.level_db, color='0.35', lw=0.8, label='vocal level [dB]')
    lv = r.levels
    if np.isfinite(lv['reference']):
        for key, style, color in (('reference', '-', 'tab:green'), ('on', ':', 'tab:green'),
                                  ('off', '--', 'tab:green'), ('noise', '-', 'tab:brown')):
            ax1.axhline(lv[key], color=color, ls=style, lw=0.8, label=key)
        lo = max(min(lv['noise'], lv['off']) - 10.0, lv['reference'] - 60.0)
        ax1.set_ylim(lo, max(lv['reference'] + 6.0, float(np.max(r.level_db)) + 2.0))
    top = ax1.get_ylim()[1]
    annotate = len(r.phone_times) <= 400
    for t, p in zip(r.phone_times, r.phones):
        shape = phone_shape(p)
        color = _COLORS.get(shape, '0.7')
        ax1.axvline(t, color=color, lw=0.6 if shape else 0.4, alpha=0.8 if shape else 0.5)
        if annotate and shape:
            ax1.text(t, top, _LABELS[shape][0], color=color, fontsize=7, ha='center',
                     va='bottom')
    ax1.set_ylabel('level [dB]')
    ax1.set_title('vocals: green = voice detected; lines = recognized phones (colored = vowel / '
                  'closed, gray = consonant)', fontsize=10, pad=14)
    ax1.legend(loc='lower right', fontsize=8)

    for i, shape in enumerate(SHAPES):
        ax2.plot(t_frame, r.weights[i], color=_COLORS[shape], lw=1.1, label=_LABELS[shape])
    ax2.set_ylabel('morph weight')
    ax2.set_xlabel(f'time [s] (frame = time x {r.fps:g})')
    ax2.set_ylim(-0.02, max(1.0, cfg.strength, cfg.closed_weight) * 1.05)
    ax2.set_title('mouth morphs: ' + ', '.join(f'{_LABELS[s][0]} {int((r.weights[i] > 0.05).sum())}'
                                               for i, s in enumerate(SHAPES))
                  + ' frames above 0.05', fontsize=10)
    ax2.legend(loc='upper right', fontsize=8, ncol=6)
    ax2.set_xlim(0, duration)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


# ---- コマンドライン ----
def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        prog='python -m nlf2vmd.lipsync',
        description='ノートブックのセル 12 が保存した音素認識の結果（lipsync_analysis.npz）から、'
                    '口のモーフ（あ・い・う・え・お・ん）のキーを VMD に書き出します。')
    ap.add_argument('analysis', help='lipsync_analysis.npz')
    ap.add_argument('-o', '--output', help='口パクだけの VMD（既定: 入力と同じフォルダの lipsync.vmd）')
    ap.add_argument('--merge', metavar='VMD', help='体の動きの VMD（motion.vmd）。口パクのキーを足して '
                                                  '--merged-output に書き出す')
    ap.add_argument('--merged-output', help='--merge の書き出し先（既定: <VMD 名>_lipsync.vmd）')
    ap.add_argument('--pmx', help='モデルの .pmx（モーフ名があるか確かめる）')
    ap.add_argument('--config', help='設定ファイル（YAML / JSON）。書いた項目だけ既定値を上書き')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='設定を 1 項目上書き（例: --set lipsync.strength=0.8）。複数指定可')
    ap.add_argument('--plot', help='グラフ（PNG）の保存先')
    args = ap.parse_args(argv)

    out = args.output or str(Path(args.analysis).with_name('lipsync.vmd'))
    merged = None
    if args.merge:
        merged = args.merged_output or str(Path(args.merge).with_name(
            Path(args.merge).stem + '_lipsync.vmd'))
    export_lipsync(args.analysis, out, motion_vmd=args.merge, merged_path=merged, pmx=args.pmx,
                   config=args.config, overrides=args.set, plot_path=args.plot)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
