"""接地判定: 接地区間が既知の合成歩行データで、推定区間が正解と ±1 フレーム以内で一致すること。"""
import numpy as np

from nlf2vmd.contact import clean_flags, hysteresis
from nlf2vmd.filters import runs


def test_detected_segments_match_known_stance(run_walk):
    motion, result = run_walk(num_frames=216, step_sec=0.6)
    for foot in range(2):
        truth = runs(motion['stance'][:, foot])
        found = result.contact.segments[foot]
        assert len(found) == len(truth), (foot, truth, found)
        for (s0, e0), (s1, e1) in zip(truth, found):
            assert abs(s0 - s1) <= 1 and abs(e0 - e1) <= 1, (foot, truth, found)


def test_detection_is_robust_to_pose_noise(run_walk):
    """関節ごとに 0.5 度のノイズを加えても、区間の数は変わらず、ずれは数フレームに収まる。

    速度のしきい値の近くでノイズが揺れるため、接地の開始が少し遅れることがある（仕様の ±1 フレームは
    ノイズの無いデータで確認する）。
    """
    motion, result = run_walk(num_frames=216, step_sec=0.6, noise_deg=0.5, seed=3)
    for foot in range(2):
        truth = runs(motion['stance'][:, foot])
        found = result.contact.segments[foot]
        assert len(found) == len(truth), (foot, truth, found)
        for (s0, e0), (s1, e1) in zip(truth, found):
            assert abs(s0 - s1) <= 3 and abs(e0 - e1) <= 3, (foot, truth, found)


def test_hysteresis_uses_looser_exit_condition():
    enter = np.array([0, 1, 0, 0, 0, 0, 1, 0], bool)
    stay = np.array([0, 1, 1, 1, 0, 1, 1, 1], bool)
    np.testing.assert_array_equal(hysteresis(enter, stay), [0, 1, 1, 1, 0, 0, 1, 1])


def test_clean_flags_fills_short_gaps_then_drops_short_contacts():
    f = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0], bool)
    out = clean_flags(f, fill_gap=2, min_len=4)
    # 2 フレームの隙間は埋まって 0〜7 が 1 区間、3 フレーム離れた長さ 2 の区間は捨てられる
    np.testing.assert_array_equal(out, [1] * 8 + [0] * 8)
