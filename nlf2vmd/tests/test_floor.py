"""床面推定: 傾いた床を水平に直し、補正は定数（区間モードでは区間ごとの定数）だけであること。"""
import numpy as np
import pytest

from nlf2vmd import load_config
from nlf2vmd.body_model import Kinematics
from nlf2vmd.synthetic import add_ray_drift, synthetic_walk, to_camera_coords

FPS = 30.0


def _kinematics(heights_fn, T=600, hold=10, seed=0, rotation=np.eye(3)):
    """足をあちこちに置いて hold フレームずつ止める。床の高さは heights_fn(x, z, t)。
    rotation: 全関節の大域回転（足裏が床と平行になる向き）。"""
    rng = np.random.default_rng(seed)
    pts = np.zeros((T, 2, 2, 3))
    for s in range(0, T, hold):
        xz = rng.uniform(-1.0, 1.0, (2, 2, 2))
        pts[s:s + hold, ..., 0] = xz[..., 0]
        pts[s:s + hold, ..., 2] = xz[..., 1]
    t = np.arange(T)[:, None, None]
    pts[..., 1] = heights_fn(pts[..., 0], pts[..., 2], t)
    joints = np.zeros((T, 24, 3))
    joints[:, 0] = [0.3, 1.0, -2.0]
    return Kinematics(np.tile(rotation, (T, 24, 1, 1)), joints, pts)


def test_tilted_floor_is_leveled_with_a_constant_offset():
    from nlf2vmd.floor import estimate_floor
    cfg = load_config().floor
    from nlf2vmd import quat
    tilt = np.deg2rad(12.0)
    normal = [-0.05, 1.0, -np.tan(tilt)]
    kin = _kinematics(lambda x, z, t: 0.4 + np.tan(tilt) * z + 0.05 * x,
                      rotation=quat.to_matrix(quat.from_two_vectors([0.0, 1.0, 0.0], normal)))
    floor = estimate_floor(kin, FPS, cfg)
    assert floor.tilt_applied and abs(floor.tilt_deg - 12.2) < 0.5
    assert np.ptp(floor.offset, axis=0).max() == 0.0           # フレーム毎に変わる補正は無い
    out = floor.apply(kin)
    assert np.abs(out.contact_points[..., 1]).max() < 1e-6
    np.testing.assert_allclose(out.root_pos[0, [0, 2]], 0.0, atol=1e-9)   # 水平の原点 = 最初の骨盤


@pytest.mark.parametrize('method', ['vectors', 'plane'])
def test_collinear_points_do_not_tilt(method):
    from nlf2vmd.floor import estimate_floor
    cfg = load_config(overrides=[f'floor.tilt_method={method}']).floor
    kin = _kinematics(lambda x, z, t: 0.0 * x)
    kin.contact_points[..., 0] = 0.05 * np.sign(kin.contact_points[..., 0])   # ほぼ一直線
    floor = estimate_floor(kin, FPS, cfg)
    if method == 'plane':
        assert not floor.tilt_applied        # 線まわりの傾きが決まらないので補正しない
    np.testing.assert_allclose(floor.rotation, np.eye(3), atol=1e-9)


def _camera_floor(tmp_path, body_model, method, pitch_deg, **walk):
    from nlf2vmd import convert
    kw = dict(num_frames=450, noise_deg=0.5, seed=0)
    kw.update(walk)
    cam = add_ray_drift(to_camera_coords(synthetic_walk(**kw), height=1.6, pitch_deg=pitch_deg))
    cfg = load_config(overrides=['diagnostics.enabled=false', f'floor.tilt_method={method}'])
    return convert(cam, tmp_path / f'{method}.vmd', body_model=body_model, config=cfg,
                   log=None).floor


@pytest.mark.parametrize('walk', [dict(speed=0.0, sway=0.08), dict(speed=0.5, heading_deg=180.0),
                                  dict(speed=0.5, heading_deg=90.0)],
                         ids=['in_place', 'away_from_camera', 'sideways'])
@pytest.mark.parametrize('pitch', [0.0, 10.0])
def test_vector_tilt_ignores_drift_along_the_camera_ray(tmp_path, body_model, walk, pitch):
    """高さ 1.6m のカメラ。単眼推定の距離のずれ（視線に沿って 30cm）があっても、見下ろし角だけを床の傾きとして
    求める（全フレームの点に 1 枚の平面を当てはめると、ずれで伸びた点群の向き＝視線の傾きを床の傾きと取り違える）。
    横向きのまま横へ歩くと、かかと→つま先はすべて同じ向きになるので、前後の傾きは足裏の向きで補う。"""
    floor = _camera_floor(tmp_path, body_model, 'vectors', pitch, **walk)
    assert floor.tilt_mode == 'full' and abs(floor.tilt_deg - pitch) < 1.0


def test_plane_tilt_is_fooled_by_drift_along_the_camera_ray(tmp_path, body_model):
    """（比較用）従来の平面の当てはめは、カメラが水平でも視線の傾きを床の傾きとして補正してしまう。"""
    floor = _camera_floor(tmp_path, body_model, 'plane', 0.0, speed=0.5, heading_deg=90.0)
    assert floor.tilt_applied and floor.tilt_deg > 3.0


def test_vector_tilt_skips_tiptoe_vectors():
    """かかとを上げて止まっている（つま先立ち）ベクトルが混ざっても、床と平行なベクトルから傾きを求める。"""
    from nlf2vmd.floor import fit_tilt_vectors
    rng = np.random.default_rng(0)
    tilt = np.deg2rad(8.0)
    yaw = rng.uniform(0, 2 * np.pi, 400)
    v = np.stack([0.2 * np.cos(yaw), np.zeros(400), 0.2 * np.sin(yaw)], axis=1)
    v[:, 1] = np.tan(tilt) * v[:, 2] + rng.normal(0, 0.002, 400)
    v[:100, 1] -= 0.08                                      # 1/4 はつま先立ち（つま先が 8cm 下）
    n, inl, mode, _, _ = fit_tilt_vectors(v, np.zeros(400), 0.08)
    assert mode == 'full' and not inl[:100].any()
    assert abs(np.rad2deg(np.arccos(n[1])) - 8.0) < 0.3


def test_segment_mode_uses_per_segment_constants_with_cosine_blend():
    from nlf2vmd.floor import estimate_floor
    cfg = load_config(overrides=['floor.segment_mode=true', 'floor.segment_sec=10',
                                 'floor.align_tilt=false']).floor
    kin = _kinematics(lambda x, z, t: np.where(t < 300, 0.0, 0.1) + 0.0 * x)
    floor = estimate_floor(kin, FPS, cfg)
    y = floor.offset[:, 1]
    assert np.allclose(y[:280], 0.0) and np.allclose(y[320:], -0.1)
    ramp = np.flatnonzero((y < -1e-9) & (y > -0.1 + 1e-9))
    assert len(ramp) >= FPS - 2                    # 1 秒以上かけてつなぐ
    assert (np.diff(y) <= 1e-12).all()             # 途中で行き過ぎない
