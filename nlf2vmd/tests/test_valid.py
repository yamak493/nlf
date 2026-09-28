"""検出できなかったフレーム（valid=False。前後から補間しただけ）を、床・接地・奥行き・傾きの推定に使わないこと。"""
import numpy as np

from nlf2vmd import convert, load_config
from nlf2vmd.body_model import Kinematics
from nlf2vmd.contact import detect_contacts
from nlf2vmd.depth import solve_depth
from nlf2vmd.filters import runs
from nlf2vmd.floor import estimate_floor
from nlf2vmd.ground import ground_offset
from nlf2vmd.synthetic import synthetic_walk

FPS = 30.0


def _standing_kin(heights):
    """足（かかと・つま先の 4 点）が動かず、高さ heights (T,) にある体。"""
    T = len(heights)
    pts = np.zeros((T, 2, 2, 3))
    pts[:, :, 0, 2], pts[:, :, 1, 2] = -0.1, 0.1           # かかと・つま先
    pts[:, 0, :, 0], pts[:, 1, :, 0] = 0.1, -0.1           # 左右の足
    pts[..., 1] = np.asarray(heights)[:, None, None]
    joints = np.zeros((T, 24, 3))
    joints[:, 0, 1] = 0.9 + np.asarray(heights)
    return Kinematics(np.tile(np.eye(3), (T, 24, 1, 1)), joints, pts)


def test_floor_ignores_interpolated_frames():
    T = 100
    valid = np.ones(T, bool)
    valid[20:80] = False                                   # 6 割が補間しただけのフレーム
    heights = np.where(valid, 0.0, 0.3)                    # 補間の区間では足が 30cm 浮いている
    kin = _standing_kin(heights)
    cfg = load_config().floor
    rest = np.zeros((2, 2, 3))
    floor = estimate_floor(kin, FPS, cfg, rest, valid)
    assert abs(floor.offset[0, 1]) < 1e-6                  # 床は検出できたフレームの足の高さ（0）
    assert abs(estimate_floor(kin, FPS, cfg, rest).offset[0, 1] + 0.3) < 1e-6   # 使うと 30cm ずれる


def test_interpolated_frames_are_never_jumps():
    """ジャンプとしてありうる浮き（0.3 秒・骨盤が放物線）でも、補間しただけのフレームなら床に着ける。"""
    T = 90
    t = np.arange(T) / FPS
    rise = np.where((t >= 1.0) & (t < 1.3), 0.5 * 9.8 * (t - 1.0) * (1.3 - t), 0.0)
    kin = _standing_kin(rise)
    cfg = load_config().ground
    assert ground_offset(kin, FPS, 1.0, cfg).flight.any()  # 検出できたフレームならジャンプとして残す
    valid = rise == 0.0
    g = ground_offset(kin, FPS, 1.0, cfg, valid)
    assert not g.flight.any()
    np.testing.assert_allclose(g.lowest, rise)


def test_contacts_do_not_start_in_interpolated_frames():
    T = 60
    cfg = load_config().contact
    pts = np.zeros((T, 2, 2, 3))
    pts[..., 1] = 0.3                                      # 足は浮いている
    pts[20:40, 0, :, 1] = 0.0                              # 左足が床に着くのは補間の区間だけ
    pts[:40, 1, :, 1] = 0.0                                # 右足は検出できた区間から着いたまま補間の区間へ
    valid = np.ones(T, bool)
    valid[20:40] = False
    c = detect_contacts(pts, FPS, cfg, speeds=np.zeros((T, 2, 2)), valid=valid)
    assert not c.flags[:, 0].any()                         # 補間の区間の中では接地を始めない
    assert c.segments[1] == [(0, 39)]                      # 接地していた足はそのまま続く
    c = detect_contacts(pts, FPS, cfg, speeds=np.zeros((T, 2, 2)))
    assert c.segments[0] == [(20, 39)]


def test_depth_prior_ignores_interpolated_frames():
    T = 90
    raw = np.linspace(0.0, 0.3, T)
    valid = np.ones(T, bool)
    valid[30:60] = False
    bad = raw.copy()
    bad[30:60] += 0.5                                      # 補間の区間の奥行きは当てにならない
    rel = np.zeros((T, 2))
    args = ([[], []], FPS, 0.005, 0.05, 3.0)
    fixed = solve_depth(bad, rel, *args, prior_weights=valid.astype(float))
    assert np.abs(fixed - raw).max() < 0.01                # 前後から加速度の小さい線でつなぐ
    assert np.abs(solve_depth(bad, rel, *args) - raw).max() > 0.2
    np.testing.assert_allclose(solve_depth(bad, rel, *args, prior_weights=np.zeros(T)),
                               solve_depth(bad, rel, *args))   # 重みがすべて 0 なら重みを使わない


def test_convert_reports_and_skips_interpolated_frames(body_model):
    motion = synthetic_walk(num_frames=150)
    T = len(motion['pose'])
    valid = np.ones(T, bool)
    valid[60:90] = False                                   # 1 秒
    trans = np.array(motion['trans'], copy=True)
    trans[60:90, 1] += 0.3                                 # 補間の区間で体全体が 30cm 浮いた推定
    cfg = load_config(overrides=['diagnostics.enabled=false'])
    r = convert(dict(motion, trans=trans, valid=valid), None, body_model=body_model, config=cfg,
                log=None)
    clean = convert(motion, None, body_model=body_model, config=cfg, log=None)
    assert r.info['interpolated'] == dict(frames=30, segments=1, longest_sec=1.0)
    assert any('1.0 秒' in w for w in r.warnings)
    assert abs(r.floor.offset[0, 1] - clean.floor.offset[0, 1]) < 0.005 * r.scale
    for foot in range(2):                                  # 補間の区間の中で始まる接地区間は無い
        assert all(not 60 <= s < 90 for s, _ in r.contact.segments[foot])
    # 補間の区間の浮きはジャンプとみなさず、床に戻す（区間の端は、この合成データの 30cm の段差を
    # 地面への拘束のガウシアンがならす分だけ残る）
    assert not r.ground.flight[60:90].any()
    lowest = r.kin.contact_points[66:84, ..., 1].min(axis=(1, 2))
    assert lowest.max() < 0.02 * r.scale
    assert len(runs(~valid)) == 1
