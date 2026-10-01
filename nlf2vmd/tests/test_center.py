"""センター: 届く高さへのクランプ後、補正が上向きになっているフレームが存在しないこと。"""
import numpy as np

from nlf2vmd import load_config
from nlf2vmd.center import ReachGeometry, apply_reach_clamp
from nlf2vmd.skeleton import Skeleton

UNIT = 12.5   # MMD 単位 / m（ReachGeometry は MMD 単位なので、しきい値 [m] にこれを掛ける）


def _setup(T=200, seed=0):
    rng = np.random.default_rng(seed)
    geom = ReachGeometry.from_skeleton(Skeleton.standard())
    t = np.arange(T) / 30.0
    center = np.zeros((T, 3))
    center[:, 1] = 0.6 * np.sin(2 * np.pi * 0.8 * t) + rng.normal(0, 0.05, T)   # 上下に揺れる骨盤
    center[:, 0] = 0.8 * np.sin(2 * np.pi * 0.3 * t)
    ik = np.zeros((T, 2, 3))
    ik[:, 0, 2] = 1.5 * np.sin(2 * np.pi * 0.5 * t)                            # 足を前後に出す
    ik[:, 1, 2] = -ik[:, 0, 2]
    lower_rot = np.tile(np.eye(3), (T, 1, 1))
    return geom, center, lower_rot, ik


def test_reach_clamp_only_lowers_the_center():
    cfg = load_config().center
    geom, center, lower_rot, ik = _setup()
    delta, corr_raw, corr, before, after = apply_reach_clamp(geom, center, lower_rot, ik, cfg,
                                                             UNIT)
    assert before > 0                          # この入力では脚が伸び切るフレームがある
    assert (corr_raw <= 0).all() and (corr <= 0).all()
    assert (delta[:, 1] <= center[:, 1] + 1e-12).all()
    np.testing.assert_array_equal(delta[:, [0, 2]], center[:, [0, 2]])
    assert after == 0


def test_applied_correction_is_at_least_the_required_drop():
    cfg = load_config().center
    geom, center, lower_rot, ik = _setup(seed=1)
    _, corr_raw, corr, _, _ = apply_reach_clamp(geom, center, lower_rot, ik, cfg, UNIT)
    assert (corr <= corr_raw + 1e-12).all()


def test_center_modes_run_and_never_raise(run_walk):
    for mode in ('A', 'B'):
        _, r = run_walk([f'center.mode={mode}'], name=f'{mode}.vmd', num_frames=180,
                        noise_deg=1.0)
        assert (r.center.correction <= 0).all()
        assert (r.center.delta[:, 1] <= r.center.smoothed[:, 1] + 1e-12).all()
        assert r.center.exceed_after == 0


def _sway_amplitude(x, fps, hz):
    """x (T,) の hz の成分の振幅（一定速度の移動は除く）。"""
    t = np.arange(len(x)) / fps
    A = np.stack([np.sin(2 * np.pi * hz * t), np.cos(2 * np.pi * hz * t), np.ones_like(t), t], 1)
    c = np.linalg.lstsq(A, x, rcond=None)[0]
    return np.hypot(c[0], c[1])


def test_center_z_keeps_the_reconstructed_depth_sway(run_walk):
    """足を床に着けたままの前後の体重移動は、ステージ6b で求め直した奥行きに残り、ステージ8（モードA）が
    さらに削らない（Z を 0.5Hz で平滑化していた頃は 1Hz の揺れが 3 割ほどしか残らなかった）。"""
    _, r = run_walk(num_frames=300, speed=0.0, sway=0.06, sway_hz=1.0)
    before = _sway_amplitude(r.kin.root_pos[:, 2], r.fps, 1.0)       # 6b の後
    after = _sway_amplitude(r.center.delta[:, 2], r.fps, 1.0)
    assert before > 0.04 * r.scale
    assert after > 0.65 * before


def test_center_z_is_smoothed_harder_without_depth_reconstruction():
    """6b で奥行きを求め直していない（depth.reconstruct: false）ときは、Z に z_raw の強い平滑化を掛ける。"""
    from nlf2vmd.center import _axis_one_euro
    cfg = load_config().center
    rng = np.random.default_rng(0)
    p = rng.normal(0, 0.03 * UNIT, (300, 3))
    rec = _axis_one_euro(p, 30.0, UNIT, cfg, True)
    raw = _axis_one_euro(p, 30.0, UNIT, cfg, False)
    np.testing.assert_array_equal(rec[:, :2], raw[:, :2])
    assert np.abs(np.diff(raw[:, 2], 2)).mean() < 0.5 * np.abs(np.diff(rec[:, 2], 2)).mean()


def test_reach_clamp_ignores_legs_that_are_not_supporting():
    """届く高さは、判定する脚（legs）だけで決める。伸び切った脚が判定しない脚なら、センターを下げない。"""
    cfg = load_config().center
    geom, center, lower_rot, ik = _setup()
    legs = np.zeros((len(center), 2), bool)
    delta, corr_raw, corr, before, after = apply_reach_clamp(geom, center, lower_rot, ik, cfg, UNIT,
                                                             legs)
    assert (corr_raw == 0).all() and (corr == 0).all() and before == after == 0
    np.testing.assert_array_equal(delta, center)
    # 片脚だけを判定すると、その脚が伸び切るフレームだけ下げる（両脚のときより下げ幅は小さいか同じ）
    legs[:, 0] = True
    _, left_raw, _, _, after = apply_reach_clamp(geom, center, lower_rot, ik, cfg, UNIT, legs)
    _, both_raw, _, _, _ = apply_reach_clamp(geom, center, lower_rot, ik, cfg, UNIT)
    assert (left_raw >= both_raw - 1e-12).all() and (left_raw < 0).any()
    assert after == 0


def test_supporting_legs():
    """接地している脚（前後 margin フレームも）と、足裏が床の近くにある脚を、体を支えている脚とする。"""
    from nlf2vmd.center import supporting_legs
    flags = np.zeros((20, 2), bool)
    flags[5:8, 0] = True
    sole = np.full((20, 2), 0.3)
    sole[15, 1] = 0.01
    legs = supporting_legs(flags, sole, 0.05, 2)
    assert legs[3:10, 0].all() and not legs[:3, 0].any() and not legs[10:, 0].any()
    assert legs[15, 1] and legs[:, 1].sum() == 1
    # ジャンプの滞空（ステージ6a）のフレームは、接地の前後へ広げた分も、足裏が床の近くの脚も使わない
    flight = np.zeros(20, bool)
    flight[8:16] = True
    legs = supporting_legs(flags, sole, 0.05, 2, flight)
    assert legs[3:8, 0].all() and not legs[8:, 0].any() and not legs[:, 1].any()


def test_split_flight_keeps_the_ballistic_height():
    """滞空の区間は、前後を結ぶ直線（地上の動き）と放物線（跳んだ高さ）に分ける。地上の動きを平滑化しても、
    跳んだ高さは削れない。"""
    from nlf2vmd.center import split_flight
    T = 60
    y = np.linspace(0.0, 0.02, T)                      # ゆっくり上がる地上の動き
    flight = np.zeros(T, bool)
    flight[20:32] = True                               # フレーム 19〜32 の間で放物線（両端で 0）
    tau = np.arange(14.0)
    y[19:33] += 0.0015 * tau * (13 - tau)              # 高さ約 6cm
    ground, jump = split_flight(y, flight)
    np.testing.assert_allclose(ground + jump, y, atol=1e-12)
    np.testing.assert_allclose(ground, np.linspace(0.0, 0.02, T), atol=1e-12)
    assert (jump[:19] == 0).all() and (jump[33:] == 0).all()
    # 滞空が無ければそのまま。先頭・末尾にかかる滞空は分けない
    g0, j0 = split_flight(y, None)
    np.testing.assert_array_equal(g0, y)
    assert not j0.any()
    edge = np.zeros(T, bool)
    edge[:5] = True
    g1, j1 = split_flight(y, edge)
    np.testing.assert_array_equal(g1, y)
    assert not j1.any()


def _kick(peak_deg, fps=30.0, T=120):
    """右足で立ったまま、左脚をまっすぐ前へ peak_deg まで蹴り上げる（1.0〜2.0 秒）合成モーション。"""
    from nlf2vmd import quat
    from nlf2vmd.body_model import ANKLES
    from nlf2vmd.synthetic import ANKLE_HEIGHT, SMPL_REST_JOINTS as J
    t = np.arange(T) / fps
    ang = np.deg2rad(peak_deg) * np.clip(np.sin(np.pi * (t - 1.0)), 0, None) * ((t > 1.0) & (t < 2.0))
    q = np.tile(quat.IDENTITY, (T, 24, 1))
    q[:, 1] = quat.from_rotvec(-ang[:, None] * [1.0, 0.0, 0.0])
    stand = J[0, 1] - J[ANKLES, 1].mean() + ANKLE_HEIGHT
    pelvis = np.tile([0.0, stand, 0.0], (T, 1))
    return dict(pose=quat.to_rotvec(q), betas=np.zeros(10), trans=pelvis - J[0], fps=fps,
                coord_system='yup'), (t > 1.0) & (t < 2.0)


def test_kicking_leg_does_not_lower_the_body():
    """まっすぐな脚を前へ蹴り上げても、遊脚の足ＩＫに届かせるために体全体（センター）を下げない
    （以前は遊脚も判定したので、股関節 75 度の蹴りで 20cm 沈み、軸足の膝が曲がった）。"""
    from nlf2vmd.pipeline import convert
    from nlf2vmd.synthetic import synthetic_body_model
    from nlf2vmd.variants import variant_tracks
    bm = synthetic_body_model()
    for peak in (45.0, 75.0):
        m, kicking = _kick(peak)
        r = convert(m, None, body_model=bm, log=None,
                    overrides=['diagnostics.enabled=false', 'lean.enabled=false'])
        drop = -r.center.correction / r.scale
        standing = drop[~kicking].max()                      # 立っているだけの所（両脚がまっすぐ）の下げ幅
        assert drop[kicking].max() < standing + 0.005        # 蹴っている間も、ほぼ同じ
        assert not r.center.legs[kicking.nonzero()[0][len(kicking.nonzero()[0]) // 2], 0]   # 蹴り脚は判定しない
        assert r.center.legs[:, 1].all()                      # 軸足はずっと判定する
        assert r.center.exceed_after == 0
        # 接地優先（両足を床に着ける）も、移動なしも同じ判定で作れる
        variant_tracks(r, 'locked', log=None)
        variant_tracks(r, 'no_move', log=None)


def _jump(duration, height, seed=0):
    from nlf2vmd.synthetic import add_joint_noise, add_jump, synthetic_walk
    m = add_jump(synthetic_walk(180, 30.0, speed=0.0, lift=0.0), 2.0, duration, height)
    return add_joint_noise(m, {j: 2.0 for j in range(24)}, seed=seed), m['airborne']


def test_jump_height_is_kept():
    """両足で跳んだ高さを、センターの平滑化と届く高さのクランプで削らない（どちらのモードも）。
    以前は、滞空をまたいだ平滑化で踏み切りの前・着地の後へ広がった上昇を、床にある足ＩＫに届かないとしてクランプが
    下げ、0.3 秒・8cm の跳びは 23%、0.4 秒・15cm は 53% しか残らなかった（モードB は、接地していない区間に重力の
    放物線を描き直すので 132〜151% に跳び上がることもあった）。"""
    from nlf2vmd.pipeline import convert
    from nlf2vmd.synthetic import synthetic_body_model
    from nlf2vmd.variants import no_move_motion
    bm = synthetic_body_model()
    for duration, height in ((0.3, 0.08), (0.4, 0.15)):
        m, airborne = _jump(duration, height)
        for mode in ('A', 'B'):
            r = convert(m, None, body_model=bm, log=None,
                        overrides=['diagnostics.enabled=false', f'center.mode={mode}'])
            y = r.center.delta[:, 1] / r.scale
            kept = (y.max() - np.median(y[:40])) / height
            assert 0.85 < kept < 1.1, (duration, mode, kept)
            # 滞空（ステージ6a）のフレームは、接地にも、体を支える脚にもしない
            flight = np.asarray(r.ground.flight, bool)
            assert flight.sum() >= 0.8 * airborne.sum()
            assert not r.contact.flags[flight].any() and not r.center.legs[flight].any()
            assert r.center.exceed_after == 0
            # 移動なしも同じ高さを残す
            nm, _, _ = no_move_motion(r)
            assert abs((nm[:, 1].max() - np.median(nm[:40, 1])) / r.scale / height - kept) < 0.05
