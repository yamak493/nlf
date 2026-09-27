"""口パク: 音素の分類・声の区間・口の形の並び・モーフのキーの書き出し。"""
import numpy as np

from nlf2vmd import load_config, quat
from nlf2vmd.lipsync import (SHAPES, build_lipsync, export_lipsync, parse_allosaurus,
                             phone_shape, save_analysis, vocal_level_db, vowel_text)
from nlf2vmd.vmd import BoneTrack, read_vmd, write_vmd

RATE = 100.0
FPS = 30.0


def synthetic(voices, phones, duration=3.0, noise_db=-45.0, bumps=()):
    """認識結果の代わり。voices: [(開始, 終了, 音量 dB)] 声の区間、phones: [(時刻, 音素)]、
    bumps: [(開始, 終了, 音量 dB)] 声の無い所に残った伴奏。"""
    t = np.arange(int(duration * RATE)) / RATE
    level = noise_db + 1.0 * np.sin(t * 37.0)
    for s, e, db in list(bumps) + list(voices):
        level[(t >= s) & (t < e)] = db
    return dict(phone_times=np.array([p[0] for p in phones], np.float64),
                phones=np.array([p[1] for p in phones], dtype=str), level_db=level,
                level_rate=RATE, duration=duration)


def lipsync_cfg(*overrides):
    return load_config(overrides=['lipsync.thin_tolerance=0', *overrides]).lipsync


def at(result, shape, sec):
    return result.weights[SHAPES.index(shape), int(round(sec * FPS))]


def test_phone_shapes():
    # Allosaurus の日本語（jpn）の音素
    jpn = {'a': 'a', 'aː': 'a', 'i': 'i', 'iː': 'i', 'ɯ': 'u', 'ɯː': 'u', 'ɯ̃': 'u', 'u': 'u',
           'e': 'e', 'ɛ': 'e', 'ɛː': 'e', 'o': 'o', 'ɔ': 'o', 'ɔː': 'o',
           'm': 'n', 'b': 'n', 'p': 'n', 'pː': 'n', 'ɴ': 'n', 'j': 'i', 'w': 'u',
           'k': None, 'kː': None, 's̪': None, 'n': None, 'ɾ': None, 'ʔ': None, 'ŋ': None}
    # 全言語（ipa）の音素の例
    ipa = {'æ': 'a', 'ʌ': 'a', 'ə': 'a', 'ɪ': 'i', 'ʊ': 'u', 'y': 'u', 'ø': 'o', 'uə': 'u',
           'aɪ': 'a', 'b̞': 'n', 'tʰ': None, 'tɕʰ': None, 'ɻ̩': None, 'ð': None}
    for phone, shape in {**jpn, **ipa}.items():
        assert phone_shape(phone) == shape, phone


def test_parse_allosaurus_and_vowel_text():
    times, phones = parse_allosaurus('0.180 0.045 k\n0.270 0.045 o\n\n注意\n0.450 0.045 ɴ\n'
                                     '1.200 0.045 m\n1.260 0.045 iː')
    np.testing.assert_allclose(times, [0.18, 0.27, 0.45, 1.2, 1.26])
    assert phones.tolist() == ['k', 'o', 'ɴ', 'm', 'iː']
    assert vowel_text(times, phones) == [(0.27, 'おん'), (1.2, 'んい')]
    times, phones = parse_allosaurus('')
    assert len(times) == 0 and len(phones) == 0


def test_vocal_level_db():
    sr = 16000
    t = np.arange(sr) / sr
    x = np.where(t < 0.5, 0.5 * np.sin(2 * np.pi * 220 * t), 0.0)
    level = vocal_level_db((x * 32767).astype(np.int16), sr, rate=RATE)
    assert len(level) == 100
    np.testing.assert_allclose(level[5:45], 20 * np.log10(0.5 / np.sqrt(2)), atol=0.2)
    assert level[55:].max() < -100
    stereo = np.stack([x, x], axis=1)
    np.testing.assert_allclose(vocal_level_db(stereo, sr, rate=RATE), level, atol=1e-3)


def test_mouth_follows_vowels_only_while_voiced():
    data = synthetic([(0.5, 1.0, -10.0), (1.5, 2.3, -12.0)],
                     [(0.2, 'k'),                  # 無音の所で認識された音素は使わない
                      (0.5, 'k'), (0.58, 'a'),     # か: 子音の時刻から あ
                      (1.5, 'm'), (1.7, 'o'),      # も: 口を閉じてから お
                      (2.0, 'iː')])
    r = build_lipsync(data, lipsync_cfg(), fps=FPS, log=None)
    assert r.weights.shape == (len(SHAPES), 90)
    assert r.info['voiced_runs'] == 2 and r.info['phones_in_voice'] == 5
    assert at(r, 'a', 0.52) > 0.5 and at(r, 'a', 0.8) > 0.9
    assert at(r, 'n', 1.58) > 0.8 and at(r, 'o', 1.85) > 0.8 and at(r, 'i', 2.15) > 0.8
    for sec in (0.8, 1.85, 2.15):
        frame = r.weights[:, int(round(sec * FPS))]
        assert np.sort(frame)[-2] < 0.05         # その時刻の形だけが開いている
    assert np.abs(r.weights[:, :int(0.35 * FPS)]).max() < 1e-3       # 声が出る前
    assert np.abs(r.weights[:, int(1.15 * FPS):int(1.35 * FPS)]).max() < 1e-3   # 声の間
    assert np.abs(r.weights[:, int(2.5 * FPS):]).max() < 1e-3        # 声が止んだあと
    assert r.voiced[int(0.8 * FPS)] and not r.voiced[int(1.25 * FPS)]


def test_louder_voice_opens_the_mouth_wider():
    data = synthetic([(0.5, 1.0, -10.0), (1.5, 2.0, -28.0)], [(0.55, 'a'), (1.55, 'a')])
    r = build_lipsync(data, lipsync_cfg(), fps=FPS, log=None)
    loud, quiet = at(r, 'a', 0.8), at(r, 'a', 1.8)
    assert loud > 0.95 and 0.25 < quiet < 0.5


def test_accompaniment_left_by_demucs_is_not_voice():
    # 声（-10 dB）の無い所に、分離で残った伴奏（-40 dB 前後、ときどき -34 dB）がある
    data = synthetic([(0.5, 1.0, -10.0), (2.0, 2.5, -12.0)],
                     [(0.55, 'a'), (1.45, 'o'), (2.05, 'e')], noise_db=-40.0,
                     bumps=[(1.3, 1.6, -34.0)])
    r = build_lipsync(data, lipsync_cfg(), fps=FPS, log=None)
    assert r.info['voiced_runs'] == 2
    assert at(r, 'o', 1.45) < 1e-3
    assert at(r, 'a', 0.8) > 0.9 and at(r, 'e', 2.3) > 0.8


def test_closed_mouth_does_not_last_while_singing():
    # ɴ のあと母音が認識されないまま声が続く: 口を閉じるのは max_closed_sec まで
    data = synthetic([(0.5, 1.6, -10.0)], [(0.55, 'a'), (0.7, 'ɴ')])
    r = build_lipsync(data, lipsync_cfg(), fps=FPS, log=None)
    assert at(r, 'n', 0.8) > 0.8
    assert at(r, 'a', 1.3) > 0.9 and at(r, 'n', 1.3) < 1e-3


def test_voice_without_vowels_and_missing_morphs():
    data = synthetic([(0.5, 1.0, -10.0)], [(0.6, 'k')])
    r = build_lipsync(data, lipsync_cfg(), fps=FPS, log=None,
                      available_morphs=['あ', 'い', 'う', 'え', 'お', 'まばたき'])
    assert at(r, 'a', 0.8) > 0.9                  # 母音が無い声の区間は fallback_vowel
    assert [t.name for t in r.tracks] == ['あ', 'い', 'う', 'え', 'お']
    assert any('ん' in w for w in r.warnings)
    r = build_lipsync(data, lipsync_cfg('lipsync.fallback_vowel=null'), fps=FPS, log=None)
    assert np.abs(r.weights).max() == 0.0


def test_no_voice_gives_zero_keys():
    data = synthetic([], [(0.5, 'a')], noise_db=-80.0)
    r = build_lipsync(data, lipsync_cfg(), fps=FPS, log=None)
    assert np.abs(r.weights).max() == 0.0 and r.warnings
    empty = dict(phone_times=np.zeros(0), phones=np.zeros(0, str), level_db=np.zeros(0),
                 level_rate=RATE, duration=0.0)
    r = build_lipsync(empty, lipsync_cfg(), fps=FPS, log=None)
    assert r.weights.shape == (len(SHAPES), 1) and r.warnings


def test_export_writes_lipsync_and_merges_into_motion(tmp_path):
    motion = tmp_path / 'motion.vmd'
    center = BoneTrack('センター', np.arange(90), np.zeros((90, 3)), np.tile(quat.IDENTITY, (90, 1)))
    write_vmd(motion, [center], 'モデル')
    data = synthetic([(0.5, 1.0, -10.0), (1.5, 2.3, -12.0)],
                     [(0.5, 'k'), (0.58, 'a'), (1.5, 'm'), (1.7, 'ɯ'), (2.0, 'e')])
    analysis = tmp_path / 'lipsync_analysis.npz'
    save_analysis(analysis, data['phone_times'], data['phones'], data['level_db'], RATE,
                  data['duration'], lang='jpn')
    r = export_lipsync(analysis, tmp_path / 'lipsync.vmd', motion_vmd=motion,
                       merged_path=tmp_path / 'motion_lipsync.vmd',
                       overrides=['lipsync.morphs.u=ω'], log=None)
    names = ['あ', 'い', 'ω', 'え', 'お', 'ん']
    lip, merged = read_vmd(tmp_path / 'lipsync.vmd'), read_vmd(tmp_path / 'motion_lipsync.vmd')
    assert lip.model_name == 'モデル' and len(lip.keys) == 0
    assert lip.morph_names() == sorted(names) == merged.morph_names()
    np.testing.assert_array_equal(merged.keys, read_vmd(motion).keys)
    T = r.weights.shape[1]
    assert T == 90
    for i, name in enumerate(names):
        frames, weights = merged.morph_track(name)
        assert frames[0] == 0 and frames[-1] == T - 1 and len(frames) < T
        # 間引いたキーを MMD と同じく線形補間すると、間引く前の値との差は thin_tolerance 以内
        np.testing.assert_allclose(np.interp(np.arange(T), frames, weights), r.weights[i],
                                   atol=r.config.thin_tolerance + 1e-6)
    assert at(r, 'u', 1.85) > 0.8
