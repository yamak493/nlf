"""NLF → VMD 変換パイプライン（vmd.md のステージ 1〜10 をこの順番で実行する）。

順番を入れ替えないこと。特に「床オフセット → 接地判定 → ロック → クランプ」の順が崩れると、
接地判定の基準がずれたり、ロック値に歪んだ値が混ざる。
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import diagnostics, quat
from .body_model import ANKLES, BodyModel, compute_kinematics, rest_info
from .center import ReachGeometry, stabilize_center
from .config import Config, load_config
from .contact import detect_contacts
from .floor import estimate_floor
from .foot_ik import boundary_steps, build_foot_ik
from .jitter import stabilize_pose, stabilize_root
from .motion_io import load_motion
from .pmx import PmxModel, read_pmx
from .retarget import Retargeter
from .skeleton import REQUIRED_BONES, SIDES, Skeleton
from .vmd import BoneTrack, thin_track, to_mmd_position, to_mmd_quat, write_vmd

BODY_MODEL_FILENAME = 'smpl_body_model.npz'


@dataclass
class ConversionResult:
    vmd_path: str
    num_keys: int
    config: Config
    fps: float
    scale: float
    motion: object
    quats: np.ndarray
    kin: object
    kin_raw: object
    floor: object
    contact: object
    foot_ik: object
    center: object
    ankle_rest: np.ndarray
    reach_geometry: object
    lower_rot_raw: np.ndarray
    local_quats: dict
    tracks: list
    info: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    diagnostics_path: str = ''
    plot_paths: dict = field(default_factory=dict)


def resolve_skeleton(pmx):
    if pmx is None or pmx == '':
        return Skeleton.standard()
    if isinstance(pmx, Skeleton):
        return pmx
    if isinstance(pmx, PmxModel):
        return Skeleton.from_pmx(pmx)
    return Skeleton.from_pmx(read_pmx(pmx))


def resolve_body_model(body_model, source):
    """体モデル: 引数 → 入力 npz に埋め込まれた bm_* → 入力と同じフォルダの smpl_body_model.npz
    → smplfitter（SMPL 公式ファイル）の順に探す。"""
    if isinstance(body_model, BodyModel):
        return body_model
    if body_model:
        return BodyModel.from_npz(body_model)
    if isinstance(source, (str, Path)):
        with np.load(source) as d:
            if 'bm_v_template' in d.files:
                return BodyModel.from_npz({k: d[k] for k in d.files}, prefix='bm_')
        sibling = Path(source).with_name(BODY_MODEL_FILENAME)
        if sibling.exists():
            return BodyModel.from_npz(sibling)
    elif 'bm_v_template' in source:
        return BodyModel.from_npz(source, prefix='bm_')
    try:
        return BodyModel.from_smplfitter()
    except Exception as e:
        raise RuntimeError(
            'SMPL の体モデルが見つかりません。ノートブックのセル 14 で書き出した '
            f'{BODY_MODEL_FILENAME} を --body-model で指定してください。') from e


def apply_scale(kin, k, depth_scale):
    """ステージ5: 全位置にスケール係数を掛け、奥行き方向の移動量だけを depth_scale 倍する。"""
    kin = kin.scaled(k)
    if depth_scale != 1.0:
        rz = kin.root_pos[:, 2]
        off = np.zeros((len(rz), 3))
        off[:, 2] = (depth_scale - 1.0) * (rz - rz[0])
        kin = kin.transformed(None, off)
    return kin


def build_tracks(skel, center_delta, ik, local, contact, cfg_vmd, unit):
    """VMD のキー列を作る（MMD 座標へ変換し、必要なら間引く）。"""
    T = len(center_delta)
    frames = np.arange(T)
    identity = np.tile(quat.IDENTITY, (T, 1))
    zeros = np.zeros((T, 3))
    items = []   # (名前, 位置 (内部座標), 回転 (内部座標), 必ず残すフレーム)
    if skel.has('グルーブ'):
        items.append(('センター', center_delta * [1.0, 0.0, 1.0], identity, ()))
        items.append(('グルーブ', center_delta * [0.0, 1.0, 0.0], identity, ()))
    else:
        items.append(('センター', center_delta, identity, ()))
    for name, q in local.items():
        items.append((name, zeros, q, ()))
    for side, s in enumerate(SIDES):
        forced = sorted({f for seg in contact.segments[side] for f in seg})
        items.append((s + '足ＩＫ', ik.delta[:, side], ik.rotation[:, side], forced))

    tracks = []
    for name, pos, rot, forced in items:
        p, q = to_mmd_position(pos), to_mmd_quat(quat.make_continuous(rot))
        keep = np.ones(T, bool)
        if cfg_vmd.thin_keys:
            keep = thin_track(p, q, cfg_vmd.thin_pos_tol_m * unit, cfg_vmd.thin_rot_tol_deg,
                              forced)
        tracks.append(BoneTrack(name, frames[keep], p[keep], q[keep]))
    return tracks


def convert(source, out_path, pmx=None, body_model=None, config=None, overrides=None,
            diag_dir=None, log=print):
    """NLF のモーション（npz のパス、または pose / betas / trans / fps を持つ dict）を VMD に変換する。

    pmx: 対象モデルの .pmx（None なら標準ボーンの寸法）/ body_model: SMPL 体モデルの npz
    config: 設定ファイルのパス・dict・Config / overrides: ['center.mode=B', ...]
    diag_dir: 診断出力（JSON・PNG）の保存先。None なら <VMD 名>_diag/
    """
    cfg = config if isinstance(config, Config) and not overrides else load_config(config,
                                                                                  overrides)
    log = log or (lambda *a: None)
    warns = []

    def warn(msg):
        warns.append(msg)
        log('⚠️ ' + msg)

    skel = resolve_skeleton(pmx)
    missing = skel.missing(REQUIRED_BONES)
    if missing:
        raise ValueError('PMX に必要なボーンがありません: ' + ', '.join(missing))
    if skel.source == 'standard':
        warn('PMX が指定されていないので、標準的な体格のボーン寸法で変換します')
    for name, note in (('上半身2', '上半身に合成します'), ('グルーブ', '上下移動はセンターの Y に書きます')):
        if not skel.has(name):
            warn(f'{name} が無いモデルです（{note}）')
    for s in SIDES:
        # 足ＩＫの回転は、子の つま先ＩＫ を動かすことで足首の向きになる
        if skel.source == 'pmx' and not skel.has(s + 'つま先ＩＫ'):
            warn(f'{s}つま先ＩＫ が無いモデルです（足ＩＫの回転が足首に伝わらず、足先の向きが変わりません）')
    bm = resolve_body_model(body_model, source)

    # ---- 1. 読み込み・正規化 ----
    motion = load_motion(source, cfg.input, bm)
    fps = motion.fps
    log(f'[1] 読み込み: {motion.num_frames} フレーム @ {fps:g} fps（元 {motion.source_fps:g} fps）')
    if abs(fps - 30.0) > 1e-6:
        warn(f'MMD は 30fps で再生します（出力は {fps:g}fps）。input.target_fps を 30 にしてください')
    if np.isfinite(motion.fk_check_mm):
        log(f'    体モデルの自己検証: 入力の関節との最大差 {motion.fk_check_mm:.2f} mm')
        if motion.fk_check_mm > 20.0:
            warn('体モデルと入力の関節位置が一致しません。体モデルのファイルを確認してください')

    # ---- 2. 姿勢のジッター制御 ----
    quats, jitter_info = stabilize_pose(motion.quats, fps, cfg.jitter)
    root = stabilize_root(motion.root_pos, cfg.jitter.root_median_window)
    log(f'[2] ジッター制御: 外れ値として置き換えたフレーム {jitter_info["outlier_frames"]}')

    # ---- 3. FK で関節位置・かかと・つま先を算出 ----
    rest = rest_info(bm, motion.betas, int(cfg.body.heel_toe_vertices),
                     float(cfg.body.sole_band_m))
    kin = compute_kinematics(quats, root, rest)
    kin_raw = compute_kinematics(motion.quats, motion.root_pos, rest)   # 診断の「処理前」用

    # ---- 4. 床面推定と定数オフセット ----
    floor = estimate_floor(kin, fps, cfg.floor)
    kin, kin_raw = floor.apply(kin), floor.apply(kin_raw)
    log(f'[4] 床: 傾き {floor.tilt_deg:.1f} 度（'
        + ('補正済み' if floor.tilt_applied else '補正なし') + f'）/ 候補点 {floor.num_points}'
        f'（インライア {floor.num_inliers}・広がり {floor.spread:.2f} m）'
        + ('（候補点が少ないため高さだけ推定）' if floor.fallback else ''))

    # ---- 5. MMD スケールへ ----
    smpl_leg = rest.leg_length()
    k = skel.mean_leg_length() / smpl_leg
    kin = apply_scale(kin, k, float(cfg.scale.depth_scale))
    kin_raw = apply_scale(kin_raw, k, float(cfg.scale.depth_scale))
    ankle_rest = k * rest.standing(rest.joints[ANKLES])
    pelvis_rest = k * rest.standing(rest.joints[0])
    log(f'[5] スケール係数 {k:.3f}（MMD 脚長 {skel.mean_leg_length():.2f} / SMPL 脚長 '
        f'{smpl_leg:.3f} m）')

    # ---- 6. 接地判定 ----
    contact = detect_contacts(kin.contact_points, fps, cfg.contact, unit=k)
    log(f'[6] 接地区間: 左 {len(contact.segments[0])} / 右 {len(contact.segments[1])}')

    # ---- 7. 足ＩＫ ----
    foot_axes = rest.points[:, 1].mean(1) - rest.points[:, 0].mean(1)   # かかと → つま先
    rt = Retargeter(skel, rest.joints, foot_axes, cfg.retarget)
    for w in rt.warnings:
        warn(w)
    ik = build_foot_ik(kin.joints[:, ANKLES], ankle_rest, rt.foot_ik_quats(kin.glob_rot),
                       contact, fps, k, cfg.foot_ik)
    steps = boundary_steps(ik.target, contact)
    max_step = float(cfg.foot_ik.max_boundary_step_m) * k
    if max(steps) > max_step:
        warn(f'接地区間の境界で足ＩＫが 1 フレームに {max(steps) / k * 100:.1f} cm 動いています')

    # ---- 8. センターの安定化 ----
    geom = ReachGeometry.from_skeleton(skel)
    center = stabilize_center(kin_raw.root_pos, kin.root_pos, pelvis_rest,
                              kin.joints[:, ANKLES], ik, contact, geom,
                              rt.global_matrix('下半身', kin.glob_rot), fps, k, cfg.center)
    log(f'[8] センター（モード {center.mode}）: 脚の伸び切り {center.exceed_before} → '
        f'{center.exceed_after} フレーム')
    if center.exceed_after:
        warn(f'届く高さへのクランプ後も {center.exceed_after} フレームで脚が伸び切っています')

    # ---- 9. 上半身の回転リターゲット ----
    local = rt.local_quats(kin.glob_rot)

    # ---- 10. VMD 書き出しと診断出力 ----
    tracks = build_tracks(skel, center.delta, ik, local, contact, cfg.vmd, k)
    model_name = cfg.vmd.model_name or skel.model_name or 'nlf2vmd'
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_keys = write_vmd(out_path, tracks, model_name)
    log(f'[10] VMD を書き出しました: {out_path}（ボーン {len(tracks)} 本 / キー {n_keys}）')

    result = ConversionResult(
        str(out_path), n_keys, cfg, fps, k, motion, quats, kin, kin_raw, floor, contact, ik,
        center, ankle_rest, geom, rt.global_matrix('下半身', kin_raw.glob_rot), local, tracks,
        warnings=warns)
    result.info = dict(
        frames=motion.num_frames, fps=fps, source_fps=motion.source_fps, scale=k,
        smpl_leg_length_m=smpl_leg, mmd_leg_length=skel.mean_leg_length(),
        skeleton=skel.source, model_name=model_name, bones=[t.name for t in tracks],
        fk_check_mm=motion.fk_check_mm, jitter=jitter_info,
        floor=dict(tilt_deg=floor.tilt_deg, tilt_applied=floor.tilt_applied,
                   normal=floor.normal.tolist(), points=floor.num_points,
                   inliers=floor.num_inliers, spread_m=floor.spread, fallback=floor.fallback,
                   segment_heights=floor.segment_heights),
        foot_ik=dict(clamped_frames=ik.clamped_frames,
                     max_boundary_step_cm=[s / k * 100.0 for s in steps],
                     max_boundary_step_limit_cm=float(cfg.foot_ik.max_boundary_step_m) * 100),
        center=dict(mode=center.mode, exceed_before=center.exceed_before,
                    exceed_after=center.exceed_after,
                    max_drop_cm=float(-center.correction.min() / k * 100.0)),
        contact_segments=dict(left=contact.segments[0], right=contact.segments[1]),
        warnings=warns)

    if cfg.diagnostics.enabled:
        diag = Path(diag_dir) if diag_dir else out_path.with_name(out_path.stem + '_diag')
        diag.mkdir(parents=True, exist_ok=True)
        result.metrics = diagnostics.compute_metrics(result)
        result.diagnostics_path = str(diagnostics.save_json(
            diag / 'diagnostics.json', dict(metrics=result.metrics, info=result.info,
                                            config=cfg.to_dict())))
        if cfg.diagnostics.plots:
            try:
                result.plot_paths = diagnostics.save_plots(result, diag)
            except ImportError:
                warn('matplotlib が無いのでグラフは出力しません')
        log(f'    診断出力: {diag}')
    return result
