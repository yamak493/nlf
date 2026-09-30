"""NLF → VMD 変換パイプライン（vmd.md のステージ 1〜10 をこの順番で実行する）。

順番を入れ替えないこと。特に「床オフセット → 接地判定 → ロック → クランプ」の順が崩れると、
接地判定の基準がずれたり、ロック値に歪んだ値が混ざる。
"""
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import diagnostics, quat
from .arm_collision import MODE_LABELS, OVERLAP_TOL_M, resolve_arm_collisions
from .body_model import ANKLES, BodyModel, compute_kinematics, forward_kinematics, rest_info
from .center import ReachGeometry, stabilize_center
from .config import Config, load_config
from .contact import detect_contacts
from .contacts import BODY_PART_LABELS, resolve_contacts
from .depth import depth_axis, depth_jitter, detection_speeds, reconstruct_depth
from .floor import estimate_floor
from .filters import runs
from .foot_ik import boundary_steps, build_foot_ik
from .ground import floating_frames, ground_offset
from .hand_reach import keep_hand_positions
from .jitter import stabilize_pose, stabilize_root
from .lean import estimate_lean
from .legs import knee_poles, solve_legs
from .motion_io import load_motion
from .outliers import PART_LABELS, remove_outliers
from .pmx import PmxModel, read_pmx
from .retarget import Retargeter
from .skeleton import REQUIRED_BONES, SIDES, Skeleton
from .twist import split_twist
from .vmd import BoneTrack, sample_rotations, thin_track, to_mmd_position, to_mmd_quat, write_vmd
from .wrist import correct_wrists
from .wrist_limits import limit_wrists

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
    local_quats: dict            # ボーン名 → ローカル回転（捩りボーンに分ける前。キーは build_tracks で分けて作る）
    tracks: list
    info: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    diagnostics_path: str = ''
    plot_paths: dict = field(default_factory=dict)
    depth: object = None
    ground: object = None
    lean: object = None
    hand_reach: object = None    # ステージ9h（胴に対する手の位置）の結果
    arm_collision: object = None   # ステージ9a（腕どうしの貫通の防止）の結果
    wrist: object = None         # ステージ2b（手首の向きの補正）の結果
    contacts: object = None      # ステージ9b（腕・手のひら・指先と体・相手の腕の接触）の結果
    local_before_contacts: dict = None   # ステージ9b の前のローカル回転（指の形を入れて 9b をやり直すのに使う）
    twist: object = None         # 捩りボーンへのひねりの振り分け（twist.TwistResult）
    wrist_limits: object = None  # ステージ9c（手首の可動域）の結果（9b の後に掛けたもの）
    outliers: object = None      # ステージ1b（慣性・重力による外れフレームの除外）の結果
    smpl_rest: np.ndarray = None  # SMPL の初期姿勢の関節（体型を固定したもの）
    skeleton: object = None      # 対象モデルの骨格（variants.py で種類別のキーを作り直すのに使う）
    retargeter: object = None
    pelvis_rest: np.ndarray = None   # (3,) 直立したときの骨盤の位置（センターの差分の基準。variants.py でセンターを求め直すのに使う）
    knee_poles: np.ndarray = None    # (T, 2, 3) SMPL の膝の向き（ステージ9l。種類ごとに脚の回転を解き直すのに使う）
    legs: object = None              # ステージ9l（脚の回転）の結果（フル）。脚のキーを打たないときは None


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
            'SMPL の体モデルが見つかりません。ノートブックのセル 9 で書き出した '
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


def build_tracks(skel, center_delta, ik, local, contact, cfg, unit, poles=None):
    """VMD のキー列を作る（捩りボーンへひねりを分け、MMD 座標へ変換し、必要なら間引く）。

    local: ボーン名 → ローカル回転（捩りボーンに分ける前。ここで cfg.twist に従って分ける）
    center_delta / ik が None なら、センター・グルーブ / 足ＩＫのキーは打たない。
    poles: SMPL の膝の向き（legs.knee_poles）。渡すと、このセンター・足ＩＫで脚（足・ひざ・足首）の回転を解いて
    キーを打つ（ステージ9l。cfg.leg_keys.enabled のとき）。
    """
    cfg_vmd = cfg.vmd
    legs = leg_rotations(skel, center_delta, ik, local, poles, cfg)
    if legs is not None:
        local = dict(local, **legs.local)
    local = split_twist(skel, local, cfg.twist).local
    T = len(contact.flags)
    frames = np.arange(T)
    identity = np.tile(quat.IDENTITY, (T, 1))
    zeros = np.zeros((T, 3))
    items = []   # (名前, 位置 (内部座標), 回転 (内部座標), 必ず残すフレーム)
    if center_delta is None:
        pass
    elif skel.has('グルーブ'):
        items.append(('センター', center_delta * [1.0, 0.0, 1.0], identity, ()))
        items.append(('グルーブ', center_delta * [0.0, 1.0, 0.0], identity, ()))
    else:
        items.append(('センター', center_delta, identity, ()))
    for name, q in local.items():
        items.append((name, zeros, q, ()))
    for side, s in enumerate(SIDES if ik is not None else ()):
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


def leg_rotations(skel, center_delta, ik, local, poles, cfg):
    """ステージ9l の LegResult（脚のキーを打たないときは None）。"""
    if poles is None or center_delta is None or ik is None or not cfg.leg_keys.enabled:
        return None
    return solve_legs(skel, center_delta, ik, local, poles, cfg.leg_keys)


def log_legs(legs, log):
    if log is None:
        return
    if legs is None:
        log('[9l] 脚の回転（膝の向き）: キーを打ちません（足ＩＫだけ）')
        return
    info = legs.info
    med = info['knee_out_deg']['median']
    log(f'[9l] 脚の回転（膝の向き）: {"・".join(info["bones"])} にキー / 膝の向き（骨盤の正面から外向き +）'
        f'中央値 左 {med[0]:+.0f}・右 {med[1]:+.0f} 度 / 足ＩＫに届かないフレーム '
        f'左 {info["unreached_frames"][0]}・右 {info["unreached_frames"][1]}')


def log_outliers(outliers, fps, log, warn):
    if not outliers.enabled:
        log('[1b] 慣性・重力による外れフレーム: 扱いません')
        return
    info = outliers.info
    parts = '・'.join(f'{PART_LABELS[k]} {n}' for k, n in info['frames'].items())
    log(f'[1b] 慣性・重力による外れフレーム: {parts} フレームを置き換えました（計 {info["replaced_frames"]} '
        f'フレーム・{info["ratio"] * 100:.1f}%。うち床・接地・奥行き・傾きの推定に使わないもの '
        f'{info["unobserved_frames"]}）')
    for name, reason in outliers.skipped.items():
        warn(f'{PART_LABELS[name]}の外れ: {reason}。outliers.png を確認してください')
    if outliers.long_runs:
        runs_ = ', '.join(f'{PART_LABELS[n]} {s / fps:.1f}〜{(e + 1) / fps:.1f} 秒'
                          for n, s, e in outliers.long_runs[:5])
        more = f' ほか {len(outliers.long_runs) - 5} 区間' if len(outliers.long_runs) > 5 else ''
        warn(f'外れらしい動きが長く続く区間があります（{runs_}{more}）。前後どちらが正しいのか'
             '決められないので置き換えていません')


def _contact_targets(info):
    """ステージ9b で当たり判定をする相手（腕・体の部位）と、判定しない体の部位の文字列。"""
    parts = info.get('body_parts', {})
    on = [BODY_PART_LABELS[p] for p, v in parts.items() if v['enabled']]
    off = [BODY_PART_LABELS[p] for p, v in parts.items() if not v['enabled']]
    arm = info.get('arm_mode', 'none')
    if arm in ('left', 'right'):
        on.insert(0, f'腕（{MODE_LABELS[arm]}）')
    return '・'.join(on) or 'なし', '・'.join(off)


def log_contacts(contacts, unit, log, label='[9b]'):
    if log is None:
        return
    on, off = _contact_targets(contacts.info)
    if not contacts.enabled:
        log(f'{label} 腕・指と体の接触: 扱いません'
            + ('（判定する部位がありません）' if contacts.info and on == 'なし' else ''))
        return
    skipped = f'（判定しない部位: {off}）' if off else ''
    info = contacts.info
    source = dict(rigid='PMX の剛体', mesh='PMX のメッシュ', config='標準の体格', none='なし')[
        info['body_source']]
    parts = ' / '.join(f'{k} {b}→{a}' for k, (b, a) in info['overlap_frames_by_part'].items()
                       if b or a)
    corr = info['max_correction_deg']
    log(f'{label} 腕・指と体・相手の腕の接触（判定する相手: {on}{skipped} / 体の形: {source} '
        f'{info["body_capsules"]} 個・{"指の各節まで" if info["fingers"] else "手は 1 本の棒"}）: 重なり '
        f'{info["overlap_frames"]["before"]} → {info["overlap_frames"]["after"]} フレーム'
        + (f'（{parts}）' if parts else '')
        + f' / 補正 最大 肩 {corr["shoulder"]:.1f}・ひじ {corr["elbow"]:.1f}・手首 {corr["wrist"]:.1f} 度')


def log_hand_reach(reach, log):
    if log is None:
        return
    if not reach.enabled:
        log('[9h] 胴に対する手の位置: 扱いません')
        return
    info = reach.info
    ratio = info['ratio']
    lr = '左 {}・右 {}'.format
    log(f'[9h] 胴に対する手の位置: 胴の比（MMD / SMPL）肩幅 {ratio["width"]:.2f}・高さ {ratio["height"]:.2f}・'
        f'厚み {ratio["depth"]:.2f}（厚み: SMPL {info["smpl"]["depth_source"]} / MMD {info["mmd"]["depth_source"]}）'
        f' / 位置を保ったフレーム ' + lr(*info['frames'])
        + ' / 手首の移動 最大 ' + lr(*(f'{v:.1f}' for v in info['max_shift_cm'])) + ' cm'
        + f' / 補正 最大 肩 {info["max_correction_deg"]["shoulder"]:.1f}・'
        f'ひじ {info["max_correction_deg"]["elbow"]:.1f} 度')


def log_twist(twist, log):
    if log is None or not twist.enabled:
        return
    if twist.chains:
        log('[10] 捩りボーン: ' + ' / '.join(f'{name} 最大 {deg:.0f} 度'
                                        for name, deg in twist.info['max_deg'].items()))
    for note in twist.skipped:
        log(f'[10] {note}')


def log_wrist_limits(before, after, log, label='[9c]'):
    """before / after: 9b の前・後に掛けた limit_wrists の WristLimitResult（before は None でもよい）。"""
    if log is None:
        return
    if not after.enabled:
        log(f'{label} 手首の可動域: 扱いません')
        return
    lr = '左 {}・右 {}'.format
    parts = []
    if before is not None:
        parts.append('範囲の外 ' + lr(*before.info['out_of_range_frames']) + ' フレーム（9b の前）')
    parts.append(lr(*after.info['out_of_range_frames']) + ' フレーム（9b の後）')
    corr = [max(a, b) for a, b in zip(after.info['max_correction_deg'],
                                      before.info['max_correction_deg'] if before else (0.0, 0.0))]
    log(f'{label} 手首の可動域: ' + ' / '.join(parts)
        + f' / 補正 最大 左 {corr[0]:.1f}・右 {corr[1]:.1f} 度')


def apply_hand_poses(result, hand_tracks, log=print):
    """指のキー（hands.make_hands の tracks）を入れた指の形で、ステージ9b（接触の解決）をやり直す。

    result（convert の結果）の local_quats・tracks・contacts を置き換える（ノートブックのセル 13 で、手の形のキーを作ったあとに
    呼ぶ）。指のキーそのものは変えない。戻り値は ContactResult。
    """
    T = len(result.contact.flags)
    cfg = result.config
    # VMD の回転（MMD 座標）→ 内部座標（どちらの向きも同じ符号の反転）
    finger_local = {t.name: quat.normalize(to_mmd_quat(sample_rotations(t, T)))
                    for t in hand_tracks if len(t.frames)}
    local, contacts = resolve_contacts(
        result.skeleton, result.retargeter, result.kin.glob_rot, result.local_before_contacts,
        cfg.contacts, cfg.arm_collision, cfg.hands.bones, cfg.arm_collision.mode, result.scale,
        result.fps, result.smpl_rest, finger_local=finger_local,
        leg_glob=None if result.legs is None else result.legs.glob)
    local, limits = limit_wrists(result.skeleton, local, cfg.wrist_limits, result.fps)
    result.local_quats = local
    result.wrist_limits = limits
    result.info['wrist_limits'] = limits.info
    result.contacts = contacts
    result.twist = split_twist(result.skeleton, local, cfg.twist)
    result.tracks = build_tracks(result.skeleton, result.center.delta, result.foot_ik, local,
                                 result.contact, cfg, result.scale, poles=result.knee_poles)
    result.info['contacts'] = contacts.info
    result.info['twist'] = result.twist.info
    log_contacts(contacts, result.scale, log, label='[9b 指の形を入れて]')
    log_wrist_limits(None, limits, log, label='[9c 指の形を入れて]')
    return contacts


def convert(source, out_path, pmx=None, body_model=None, config=None, overrides=None,
            diag_dir=None, log=print, hands_analysis=None):
    """NLF のモーション（npz のパス、または pose / betas / trans / fps を持つ dict）を VMD に変換する。

    out_path: 書き出す .vmd（フル: 体の動きすべて）。None なら書き出さない（variants.write_variant で
    種類を選んで書き出す）
    pmx: 対象モデルの .pmx（None なら標準ボーンの寸法）/ body_model: SMPL 体モデルの npz
    config: 設定ファイルのパス・dict・Config / overrides: ['center.mode=B', ...]
    diag_dir: 診断出力（JSON・PNG）の保存先。None なら <VMD 名>_diag/（out_path も None なら保存しない）
    hands_analysis: 手のランドマークの検出結果（hands.save_analysis の npz のパスか dict）。渡すと、
    ステージ2b で手首の向きを MediaPipe の手のひらの向きへ寄せる（フレーム 0 は source のフレーム 0 とそろえる）
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
    valid = np.asarray(motion.valid, bool)
    gaps = runs(~valid)
    longest_gap = max((e - s + 1 for s, e in gaps), default=0) / fps
    if gaps:
        log(f'    検出できなかった（前後から補間した）フレーム: {int((~valid).sum())} '
            f'（{len(gaps)} 区間・最長 {longest_gap:.2f} 秒）。床・接地・奥行き・傾きの推定には使いません')
        if longest_gap > float(cfg.input.max_gap_warn_sec):
            warn(f'人物を検出できなかった区間が {longest_gap:.1f} 秒続いています（補間しただけの動きです）')
    if np.isfinite(motion.fk_check_mm):
        log(f'    体モデルの自己検証: 入力の関節との最大差 {motion.fk_check_mm:.2f} mm')
        if motion.fk_check_mm > 20.0:
            warn('体モデルと入力の関節位置が一致しません。体モデルのファイルを確認してください')

    # ---- 1b. 慣性・重力による外れフレームの除外 ----
    # 重力の向き（床の法線）は、外れを除く前の体で仮に求める（ステージ4で外れを除いた体から求め直す）
    rest = rest_info(bm, motion.betas, int(cfg.body.heel_toe_vertices),
                     float(cfg.body.sole_band_m))
    rest_joints = bm.rest_joints(motion.betas)
    valid_input = valid
    outliers = None
    if cfg.outliers.enabled:
        pre_floor = estimate_floor(compute_kinematics(motion.quats, motion.root_pos, rest), fps,
                                   cfg.floor, rest.points.mean(2), valid)
        outliers = remove_outliers(motion.quats, motion.root_pos, valid, rest_joints, bm.parents,
                                   fps, cfg.outliers, pre_floor.rotation,
                                   depth_axis(pre_floor.rotation))
        motion = replace(motion, quats=outliers.quats, root_pos=outliers.root_pos,
                         valid=outliers.valid)
        valid = outliers.valid
        log_outliers(outliers, fps, log, warn)

    # ---- 2. 姿勢のジッター制御 ----
    quats, jitter_info = stabilize_pose(motion.quats, fps, cfg.jitter, rest_joints)
    root = stabilize_root(motion.root_pos, cfg.jitter.root_median_window)
    hold = ''
    if 'hand_hold_ratio' in jitter_info:
        left, right = (v * 100.0 for v in jitter_info['hand_hold_ratio'])
        hold = (f' / 手首の位置の軌跡に合わせたフレーム 左 {left:.0f}%・右 {right:.0f}%'
                f'（肩・ひじの補正 最大 {jitter_info["hand_hold_max_deg"]:.1f} 度）')
    log(f'[2] ジッター制御: 外れ値として置き換えたフレーム {jitter_info["outlier_frames"]}' + hold)

    # ---- 2b. 手首の向きの補正（MediaPipe Hands） ----
    wrist = None
    if hands_analysis is not None and cfg.wrist.enabled:
        from .hands import load_analysis
        analysis = load_analysis(hands_analysis)
        if 'roi' not in analysis or 'image_size' not in analysis:
            warn('手の検出結果に ROI・画像の大きさが無いので、手首の向きは補正しません'
                 '（ノートブックのセル 10 で検出し直してください）')
        else:
            def fk(q):
                return forward_kinematics(q, root, rest_joints, bm.parents)[0]

            quats, wrist = correct_wrists(quats, fk, rest_joints, analysis, fps, cfg.wrist,
                                          cfg.wrist_limits)
            parts = []
            for s, name in zip(SIDES, ('left', 'right')):
                d = wrist.info[name]
                med = d['disagreement_deg_median']
                parts.append(f'{s}手 使えたフレーム {d["observed_frames"]}'
                             + ('' if med is None else
                                f'・NLF との差の中央値 {med:.0f} 度'
                                f'（45 度超 {d["disagreement_over_45_ratio"] * 100:.0f}%）'))
            log('[2b] 手首の向きの補正（MediaPipe）: ' + ' / '.join(parts))

    # ---- 3. FK で関節位置・かかと・つま先を算出 ----
    kin = compute_kinematics(quats, root, rest)
    kin_raw = compute_kinematics(motion.quats, motion.root_pos, rest)   # 診断の「処理前」用

    # ---- 4. 床面推定と定数オフセット ----
    floor = estimate_floor(kin, fps, cfg.floor, rest.points.mean(2), valid)
    kin, kin_raw = floor.apply(kin), floor.apply(kin_raw)
    log(f'[4] 床: 傾き {floor.tilt_deg:.1f} 度（'
        + {'full': '補正済み', 'line': '1 方向だけ補正', 'none': '補正なし'}[floor.tilt_mode]
        + f'）/ 候補点 {floor.num_points}'
        f'（{"ベクトル" if cfg.floor.tilt_method == "vectors" else "インライア"} '
        f'{floor.num_inliers}・広がり {floor.spread:.2f} m）'
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

    # ---- 6a. 接地の拘束（両足が長く浮いた・埋まった状態を、体全体の上下の平行移動で床に戻す） ----
    # ---- 6. 接地判定（速度は奥行きのぶれを除いて求める） ----
    # ---- 6c. 前後の傾きの補正（重心が接地している足の上に来るように、骨盤から上を起こす） ----
    # ---- 6b. 奥行きの再構成（接地している足から骨盤の奥行きを求め直す） ----
    # 2 回目以降は、奥行きを補正した体でもう一度接地を判定する（両足が同時に前後へ動いて見えて
    # 判定から漏れたフレームを拾い、その区間も含めて奥行きを求め直す）。奥行きの軸は床の傾きを補正すると
    # 上下の成分を持つので、接地の拘束は奥行きを補正した体に毎回掛け直す。
    # 接地の拘束は、最初（接地判定の前）は両足の最下点から、接地判定の後は接地している足の高さだけから求める
    # （遊脚の足先が床より下に見えても体全体を上下させない）。
    # 前後の傾きは 1 回目の接地判定のあとに 1 度だけ求める。足首から下は動かさないので接地判定は変わらず、
    # 奥行きの再構成は傾きを補正した足首→骨盤の相対位置を使う（平滑化する前の姿勢にも同じ補正を掛ける）
    axis = depth_axis(floor.rotation)
    # 接地の拘束は、検出できなかった（補間しただけの）フレームだけをジャンプにしない。外れとして置き換えた
    # フレームは、重力で動ける上下の動き（ステージ1b）なので、ジャンプの途中でもそのまま使う
    ground = ground_offset(kin, fps, k, cfg.ground, valid_input)
    judged = ground.apply(kin)
    kin_pose, lean = kin_raw, None
    for _ in range(max(1, int(cfg.depth.passes)) if cfg.depth.reconstruct else 1):
        contact = detect_contacts(judged.contact_points, fps, cfg.contact, unit=k,
                                  speeds=detection_speeds(judged, axis, fps, k, cfg.depth),
                                  valid=valid)
        if lean is None:
            lean = estimate_lean(judged, contact, axis, rest, fps, k, cfg.lean, ground.flight, valid)
            kin, kin_pose = lean.apply(kin), lean.apply(kin_raw)
        depth = reconstruct_depth(kin, kin_pose, contact, axis, fps, k, cfg.depth, valid)
        moved = depth.apply(kin)
        ground = ground_offset(moved, fps, k, cfg.ground, valid_input, contact)
        judged = ground.apply(moved)
    kin = judged
    # 接地判定の高さを最後の体の高さにする（足ＩＫの床への吸着は、同じ体の「足首 − 足裏」の高さを使う。
    # 判定した体のままだと、最後の接地の拘束・奥行きの補正で動いた分だけ、接地中の足が浮く・埋まる）
    contact = replace(contact, heights=np.asarray(kin.contact_points)[..., 1])
    if ground.enabled:
        log(f'[6a] 接地の拘束: 上下の補正 {-ground.offset.max() / k * 100:.1f} 〜 '
            f'{-ground.offset.min() / k * 100:.1f} cm / ジャンプとして残した区間 '
            f'{len(runs(ground.flight))}')
    log(f'[6] 接地区間: 左 {len(contact.segments[0])} / 右 {len(contact.segments[1])}')
    if lean.enabled:
        deg = np.rad2deg(lean.angle)
        log(f'[6c] 前後の傾きの補正: {deg.min():+.1f} 〜 {deg.max():+.1f} 度（重心の傾き '
            f'{np.nanmedian(lean.before_deg):+.1f} → {np.nanmedian(lean.after_deg):+.1f} 度）')
        if lean.clamped:
            warn(f'前後の傾きの補正角が上限（lean.max_deg = {cfg.lean.max_deg} 度）に当たりました。'
                 '接地判定（contact.png）と lean.png を確認してください')
    elif cfg.lean.enabled:
        log('[6c] 前後の傾きの補正: 接地しているフレームが少ないので補正しません')
    if depth.enabled:
        log(f'[6b] 奥行きの補正: 最大 {np.abs(depth.correction).max() / k * 100:.1f} cm')

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
                              rt.global_matrix('下半身', kin.glob_rot), fps, k, cfg.center,
                              depth.enabled)
    log(f'[8] センター（モード {center.mode}）: 脚の伸び切り {center.exceed_before} → '
        f'{center.exceed_after} フレーム')
    if center.exceed_after:
        warn(f'届く高さへのクランプ後も {center.exceed_after} フレームで脚が伸び切っています')

    # ---- 9. 上半身の回転リターゲット ----
    local = rt.local_quats(kin.glob_rot)

    # ---- 9l. 脚の回転（膝の向き）: 股関節から足ＩＫまでを、SMPL の膝の向きの側に膝を置いて解く ----
    # キーは種類ごと（フル・接地優先・移動なし）に build_tracks で解き直す。ここではフルの脚を 9b の当たり判定に使う
    poles = knee_poles(kin, fps, cfg.leg_keys.pole_one_euro)
    legs = leg_rotations(skel, center.delta, ik, local, poles, cfg)
    log_legs(legs, log)

    # ---- 9h. 胴に対する手の位置（胴の近くの手首を、胴の寸法の比で移した位置へ 2 ボーン IK で置く） ----
    local, reach = keep_hand_positions(skel, rt, kin.glob_rot, local, rest.joints,
                                       bm.rest_vertices(motion.betas), bm.weights, cfg.hand_reach,
                                       cfg.contacts, k)
    log_hand_reach(reach, log)

    # ---- 9a. 腕どうしの貫通の防止 ----
    local, arms = resolve_arm_collisions(skel, rt, kin.glob_rot, local, cfg.arm_collision, k, fps, axis)
    n_before, n_after = arms.overlap_frames(OVERLAP_TOL_M * k)
    radius_cm = ' / '.join(f'{n} {r:.1f}' for n, r in zip(('上腕', '前腕', '手'),
                                                         arms.radius.mean(0) / k * 100.0))
    log(f'[9a] 腕どうしの貫通の防止（{MODE_LABELS[arms.mode]}）: 腕の半径 {radius_cm} cm'
        f'（{"PMX のメッシュから" if arms.radius_source == "mesh" else "設定の値"}）/ '
        f'重なり {n_before} → {n_after} フレーム'
        + (f' / 補正 最大 {arms.correction_deg.max(initial=0.0):.1f} 度（腕の移動 最大 画像面内 '
           f'{arms.shift_image.max(initial=0.0) / k * 100:.1f} cm・奥行き '
           f'{arms.shift_depth.max(initial=0.0) / k * 100:.1f} cm。depth_cost {arms.depth_cost:g}'
           + (f'。上限に達して奥行きの優先を弱めたフレーム {arms.relaxed_frames}' if arms.relaxed_frames else '')
           + '）' if arms.side >= 0 else ''))

    # ---- 9b. 腕・手のひら・指先と、体・相手の腕の接触 ----
    # 手首は先に可動域に収めてから解く（9c。人の関節では届かない向きの手で当たり判定をしない）
    local, limits_pre = limit_wrists(skel, local, cfg.wrist_limits, fps)
    local_before_contacts = local
    local, contacts = resolve_contacts(skel, rt, kin.glob_rot, local, cfg.contacts, cfg.arm_collision,
                                       cfg.hands.bones, cfg.arm_collision.mode, k, fps, rest.joints,
                                       leg_glob=None if legs is None else legs.glob)
    log_contacts(contacts, k, log)

    # ---- 9c. 手首の可動域（9b で手首を回した分も含めて、人の関節の範囲に収める） ----
    local, limits = limit_wrists(skel, local, cfg.wrist_limits, fps)
    log_wrist_limits(limits_pre, limits, log)

    # ---- 10. 捩りボーンへのひねりの振り分け（キーは build_tracks が同じように分けて作る）・VMD 書き出しと診断出力 ----
    twist = split_twist(skel, local, cfg.twist)
    log_twist(twist, log)
    tracks = build_tracks(skel, center.delta, ik, local, contact, cfg, k, poles=poles)
    model_name = cfg.vmd.model_name or skel.model_name or 'nlf2vmd'
    n_keys = 0
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n_keys = write_vmd(out_path, tracks, model_name)
        log(f'[10] VMD を書き出しました: {out_path}（ボーン {len(tracks)} 本 / キー {n_keys}）')

    result = ConversionResult(
        str(out_path or ''), n_keys, cfg, fps, k, motion, quats, kin, kin_raw, floor, contact,
        ik, center, ankle_rest, geom, rt.global_matrix('下半身', kin_raw.glob_rot), local, tracks,
        warnings=warns, depth=depth, ground=ground, lean=lean, hand_reach=reach, arm_collision=arms,
        skeleton=skel, retargeter=rt, wrist=wrist, contacts=contacts, local_before_contacts=local_before_contacts,
        twist=twist, smpl_rest=rest.joints, wrist_limits=limits, outliers=outliers,
        pelvis_rest=pelvis_rest, knee_poles=poles, legs=legs)
    result.info = dict(
        frames=motion.num_frames, fps=fps, source_fps=motion.source_fps, scale=k,
        smpl_leg_length_m=smpl_leg, mmd_leg_length=skel.mean_leg_length(),
        skeleton=skel.source, model_name=model_name, bones=[t.name for t in tracks],
        fk_check_mm=motion.fk_check_mm, jitter=jitter_info,
        wrist=None if wrist is None else wrist.info,
        hand_reach=reach.info,
        legs=dict(enabled=False) if legs is None else dict(enabled=True, **legs.info),
        contacts=contacts.info,
        wrist_limits=dict(limits.info, before_contacts={
            k: limits_pre.info[k] for k in ('out_of_range_frames', 'changed_frames', 'max_correction_deg')}),
        twist=twist.info,
        interpolated=dict(frames=int((~valid_input).sum()), segments=len(gaps),
                          longest_sec=round(float(longest_gap), 3)),
        outliers=dict(enabled=False) if outliers is None else outliers.info,
        floor=dict(tilt_deg=floor.tilt_deg, tilt_applied=floor.tilt_applied,
                   tilt_mode=floor.tilt_mode,
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
        depth=dict(reconstructed=depth.enabled, axis=axis.tolist(),
                   root_jitter_cm=dict(zip(('depth', 'lateral'),
                                           (depth_jitter(kin_raw.root_pos, axis) / k * 100.0)
                                           .tolist())),
                   max_correction_cm=float(np.abs(depth.correction).max() / k * 100.0)),
        lean=dict(enabled=lean.enabled, direction=lean.direction.tolist(),
                  correction_deg=dict(zip(('min', 'median', 'max'), np.percentile(
                      np.rad2deg(lean.angle), [0, 50, 100]).tolist())),
                  com_lean_deg=dict(before=float(np.nanmedian(lean.before_deg))
                                    if lean.enabled else None,
                                    after=float(np.nanmedian(lean.after_deg))
                                    if lean.enabled else None),
                  supported_frames=int(lean.supported.sum()), clamped=lean.clamped),
        ground=dict(enabled=ground.enabled, window_frames=ground.window,
                    correction_cm=dict(min=float(-ground.offset.max() / k * 100.0),
                                       max=float(-ground.offset.min() / k * 100.0)),
                    floating_frames_before=floating_frames(
                        ground.lowest, fps, float(cfg.contact.exit_height_m) * k,
                        cfg.ground.max_flight_sec),
                    jumps_sec=[[round(s0 / fps, 2), round((e0 + 1) / fps, 2)]
                               for s0, e0 in runs(ground.flight)]),
        arm_collision=dict(mode=arms.mode, radius_source=arms.radius_source,
                           radius_cm=dict(zip(SIDES, (arms.radius / k * 100.0).tolist())),
                           overlap_frames=dict(before=n_before, after=n_after),
                           max_depth_cm={key: float(d.max(initial=0.0) / k * 100.0) for key, d in
                                         (('before', arms.depth_before),
                                          ('after', arms.depth_after))},
                           max_correction_deg=float(arms.correction_deg.max(initial=0.0)),
                           depth_cost=arms.depth_cost, relaxed_frames=arms.relaxed_frames,
                           max_shift_cm={key: float(d.max(initial=0.0) / k * 100.0) for key, d in
                                         (('image', arms.shift_image), ('depth', arms.shift_depth))}),
        warnings=warns)

    if cfg.diagnostics.enabled:
        result.metrics = diagnostics.compute_metrics(result)
        if diag_dir:
            diag = Path(diag_dir)
        elif out_path is not None:
            diag = out_path.with_name(out_path.stem + '_diag')
        else:
            return result
        diag.mkdir(parents=True, exist_ok=True)
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
