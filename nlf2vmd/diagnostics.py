"""検証・診断出力。処理前後の指標を JSON に、判定の妥当性を確かめるグラフを PNG に保存する。"""
import json
from pathlib import Path

import numpy as np

from .body_model import ANKLES
from .ground import floating_frames, lowest_foot_height
from .jitter import JOINT_GROUPS, angular_acceleration

METRIC_LABELS = {
    'foot_slide_cm_per_frame': ('足滑り', '接地区間内の足ＩＫの水平移動量の平均', '0'),
    'penetration_frames': ('埋まり', '足ＩＫの差分Yが0未満のフレーム数（足ごと）', '0'),
    'floating_frames': ('浮き', '両足の最下点が接地終了の高さより上にいる状態が、ジャンプの最長滞空時間'
                        '（ground.max_flight_sec）より長く続くフレーム数', '0'),
    'hover_frames': ('短い浮き', '両足とも接地区間の外で、ジャンプとして残した区間でもないのに、出力の両足の足裏が'
                     '床から 2cm より上にあるフレーム数', '0 に近い'),
    'center_jitter_cm_per_frame2': ('センターの震え', 'センター・グルーブ位置の加速度の絶対値の平均（X, Y, Z）',
                                    '処理前より大幅に減少'),
    'pose_jitter_deg_per_s2': ('姿勢の震え', '関節の角加速度の平均（グループ別）', '処理前より減少'),
    'overextended_frames': ('脚の伸び切り', '股関節から足ＩＫまでの距離が脚長の98%を超えるフレーム数', '0'),
    'contact_segments': ('接地切り替え回数', '足ごとの接地区間の数', '目視との整合を確認'),
    'foot_motion_near_floor_cm_per_frame': (
        '床付近の足の動き', '足が床付近（かかとかつま先が接地終了の高さ未満）にあるときの足ＩＫの移動量の平均（X, Z）',
        'Z（奥行き）が X と同程度まで減少'),
}


def _round(x, nd=4):
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_round(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), nd)
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x


def foot_slide(pos, segments, unit):
    """接地区間内の水平移動量の平均 [cm/フレーム]。pos: (T, 2, 3)。"""
    steps = []
    for foot in range(2):
        for s, e in segments[foot]:
            if e > s:
                d = np.diff(pos[s:e + 1, foot][:, [0, 2]], axis=0)
                steps.append(np.linalg.norm(d, axis=-1))
    if not steps:
        return 0.0
    return float(np.concatenate(steps).mean() / unit * 100.0)


def foot_motion_near_floor(pos, heights, limit, unit):
    """足が床付近にある（前後のフレームとも高さ < limit）ときの移動量の平均 [cm/フレーム]（X, Z 別）。

    接地区間の外（判定から漏れたフレーム）の滑りも含めて、軸ごとに比べるための指標。
    pos: (T, 2, 3) / heights: (T, 2, 2 [かかと, つま先])。
    """
    near = np.asarray(heights).min(-1) < limit
    both = near[:-1] & near[1:]
    if not both.any():
        return [0.0, 0.0]
    d = np.abs(np.diff(pos, axis=0))[both]
    return (d[:, [0, 2]].mean(0) / unit * 100.0).tolist()


def output_sole_heights(r):
    """(T, 2) 出力の足ＩＫでの足裏の高さ（足首の高さ − その姿勢での足首から足裏の最下点までの高さ）。

    接地区間は、足ＩＫの回転を区間内の平均で固定しているので、足首から足裏までの高さも区間内の中央値を使う。
    """
    ankle_above_sole = r.kin.joints[:, ANKLES, 1] - r.kin.contact_points[..., 1].min(-1)
    sole = r.foot_ik.target[..., 1] - ankle_above_sole
    for foot in range(2):
        for s, e in r.contact.segments[foot]:
            sole[s:e + 1, foot] = r.foot_ik.target[s, foot, 1] - np.median(
                ankle_above_sole[s:e + 1, foot])
    return sole


def hover_mask(r, height):
    """(T,) 両足とも接地区間の外で、ジャンプとして残した区間でもないのに、出力の両足の足裏が height より上のフレーム。"""
    flight = r.ground.flight if r.ground is not None else np.zeros(len(r.kin.joints), bool)
    return (~r.contact.flags.any(1) & ~flight
            & (output_sole_heights(r).min(1) > height))


def accel_per_axis(p, unit):
    """位置 (T, 3) の加速度の絶対値の平均 [cm/フレーム^2]（軸別）。"""
    if len(p) < 3:
        return [0.0, 0.0, 0.0]
    acc = np.abs(np.diff(p, 2, axis=0)).mean(0) / unit * 100.0
    return acc.tolist()


def pose_jitter(quats, fps):
    acc = angular_acceleration(quats, fps)
    if len(acc) == 0:
        return {g: 0.0 for g in JOINT_GROUPS}
    return {g: float(acc[:, idx].mean()) for g, idx in JOINT_GROUPS.items()}


def compute_metrics(r):
    """r: pipeline.ConversionResult。処理前・処理後を並べた辞書を返す。

    処理前の値は、ジッター制御も奥行きの再構成もしていない入力（kin_raw）から求める。
    """
    k, seg = r.scale, r.contact.segments
    float_h = r.config.contact.exit_height_m * k
    raw_ankles = r.kin_raw.joints[:, ANKLES]
    raw_ik_delta = raw_ankles - r.ankle_rest[None]
    m = {
        'foot_slide_cm_per_frame': dict(before=foot_slide(raw_ankles, seg, k),
                                        after=foot_slide(r.foot_ik.target, seg, k)),
        'penetration_frames': dict(before=(raw_ik_delta[..., 1] < 0).sum(0).tolist(),
                                   after=(r.foot_ik.delta[..., 1] < 0).sum(0).tolist()),
        'floating_frames': dict(before=floating_frames(lowest_foot_height(r.kin_raw), r.fps, float_h,
                                                       r.config.ground.max_flight_sec),
                                after=floating_frames(lowest_foot_height(r.kin), r.fps, float_h,
                                                      r.config.ground.max_flight_sec)),
        'hover_frames': dict(after=int(hover_mask(r, 0.02 * k).sum())),
        'center_jitter_cm_per_frame2': dict(before=accel_per_axis(r.center.raw, k),
                                            after=accel_per_axis(r.center.delta, k)),
        'pose_jitter_deg_per_s2': dict(before=pose_jitter(r.motion.quats, r.fps),
                                       after=pose_jitter(r.quats, r.fps)),
        'overextended_frames': dict(
            before=int(r.reach_geometry.overextended(r.center.raw, r.lower_rot_raw,
                                                     raw_ik_delta, r.config.center.reach_ratio)
                       .sum()),
            before_clamp=r.center.exceed_before,
            after=r.center.exceed_after),
        'contact_segments': dict(after=[len(s) for s in seg]),
        'foot_motion_near_floor_cm_per_frame': dict(
            before=foot_motion_near_floor(raw_ankles, r.contact.heights,
                                          r.config.contact.exit_height_m * k, k),
            after=foot_motion_near_floor(r.foot_ik.target, r.contact.heights,
                                         r.config.contact.exit_height_m * k, k)),
    }
    for key, (label, definition, goal) in METRIC_LABELS.items():
        m[key].update(label=label, definition=definition, goal=goal)
    return m


def save_json(path, data):
    Path(path).write_text(json.dumps(_round(data), ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def format_metrics(metrics):
    """評価指標を「名前: 処理前 → 処理後」の行にする（ノートブックでも使う）。"""
    def fmt(v):
        v = _round(v, 3)
        if isinstance(v, dict):
            return ', '.join(f'{k} {x}' for k, x in v.items())
        return str(v)

    units = {'foot_slide_cm_per_frame': ' cm/フレーム', 'center_jitter_cm_per_frame2':
             ' cm/フレーム²（X, Y, Z）', 'pose_jitter_deg_per_s2': ' deg/s²',
             'foot_motion_near_floor_cm_per_frame': ' cm/フレーム（X, Z）',
             'floating_frames': ' フレーム', 'hover_frames': ' フレーム'}
    lines = []
    for key, m in metrics.items():
        before = fmt(m['before']) if 'before' in m else '-'
        lines.append(f"{m['label']}: {before} → {fmt(m['after'])}{units.get(key, '')}")
    return lines


# ---- グラフ ----
def _bands(ax, flags, color, alpha=0.18):
    from .filters import runs
    for s, e in runs(flags):
        ax.axvspan(s - 0.5, e + 0.5, color=color, alpha=alpha, lw=0)


def save_plots(r, out_dir):
    # pyplot を使わない（ノートブックの描画バックエンドを変えないため）
    from matplotlib.figure import Figure

    out_dir = Path(out_dir)
    k, cfg = r.scale, r.config
    frames = np.arange(len(r.center.delta))
    cm = 100.0 / k
    paths = {}
    feet = ('left foot', 'right foot')

    # 1. 足の高さ・水平速度・接地フラグ
    fig = Figure(figsize=(12, 6))
    axes = fig.subplots(2, 1, sharex=True)
    for foot, ax in enumerate(axes):
        h = r.contact.heights[:, foot] * cm
        v = r.contact.speeds[:, foot] / k
        _bands(ax, r.contact.flags[:, foot], 'tab:green')
        ax.plot(frames, h[:, 0], color='tab:blue', lw=1, label='heel height [cm]')
        ax.plot(frames, h[:, 1], color='tab:cyan', lw=1, label='toe height [cm]')
        ax.axhline(cfg.contact.enter_height_m * 100, color='tab:blue', ls=':', lw=0.8)
        ax.axhline(cfg.contact.exit_height_m * 100, color='tab:blue', ls='--', lw=0.8)
        ax.set_ylabel('height [cm]')
        ax2 = ax.twinx()
        ax2.plot(frames, v.min(-1), color='tab:orange', lw=1, label='min horizontal speed [m/s]')
        ax2.axhline(cfg.contact.enter_speed_m_per_s, color='tab:orange', ls=':', lw=0.8)
        ax2.axhline(cfg.contact.exit_speed_m_per_s, color='tab:orange', ls='--', lw=0.8)
        ax2.set_ylabel('speed [m/s]')
        ax2.set_ylim(0, max(1.5, cfg.contact.exit_speed_m_per_s * 3))
        ax.set_title(f'{feet[foot]}: contact bands (green), dotted = enter / dashed = exit '
                     'threshold')
        lines = ax.get_lines()[:2] + ax2.get_lines()[:1]
        ax.legend(lines, [ln.get_label() for ln in lines], loc='upper right', fontsize=8)
    axes[-1].set_xlabel('frame')
    fig.tight_layout()
    paths['contact'] = out_dir / 'contact.png'
    fig.savefig(paths['contact'], dpi=110)

    # 2. センターの処理前後
    fig = Figure(figsize=(12, 7))
    axes = fig.subplots(3, 1, sharex=True)
    for i, (ax, name) in enumerate(zip(axes, 'XYZ')):
        ax.plot(frames, r.center.raw[:, i] * cm, color='0.6', lw=1, label='before')
        ax.plot(frames, r.center.delta[:, i] * cm, color='tab:red', lw=1.2, label='after')
        ax.set_ylabel(f'{name} [cm]')
        ax.legend(loc='upper right', fontsize=8)
    axes[0].set_title(f'center (pelvis) displacement, mode {r.center.mode} '
                      '(internal coords: +Z = facing the camera)')
    axes[-1].set_xlabel('frame')
    fig.tight_layout()
    paths['center'] = out_dir / 'center.png'
    fig.savefig(paths['center'], dpi=110)

    # 2b. 骨盤の奥行きの再構成（接地している足から求め直した奥行き）
    if r.depth is not None:
        fig = Figure(figsize=(12, 4))
        ax = fig.subplots()
        _bands(ax, r.contact.flags.any(1), 'tab:green', alpha=0.12)
        ax.plot(frames, (r.depth.raw - r.depth.raw[0]) * cm, color='0.6', lw=1,
                label='before (estimated depth)')
        if r.depth.enabled:
            ax.plot(frames, (r.depth.depth - r.depth.raw[0]) * cm, color='tab:red', lw=1.2,
                    label='after (from planted feet)')
        jit = r.info.get('depth', {}).get('root_jitter_cm', {})
        ax.set_title('pelvis depth along the camera axis; green = a foot is in contact '
                     f"(jitter: depth {jit.get('depth', 0):.1f} cm / lateral "
                     f"{jit.get('lateral', 0):.1f} cm)")
        ax.set_ylabel('depth [cm]')
        ax.set_xlabel('frame')
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()
        paths['depth'] = out_dir / 'depth.png'
        fig.savefig(paths['depth'], dpi=110)

    # 2c. 接地の拘束（両足の最下点と、体全体の上下の補正量）
    if r.ground is not None:
        fig = Figure(figsize=(12, 4))
        ax = fig.subplots()
        _bands(ax, r.contact.flags.any(1), 'tab:green', alpha=0.12)
        _bands(ax, r.ground.flight, 'tab:orange', alpha=0.35)
        _bands(ax, hover_mask(r, 0.02 * k), 'tab:red', alpha=0.35)
        ax.plot(frames, r.ground.lowest * cm, color='0.6', lw=1,
                label='lowest foot point: before (smoothed pose)')
        ax.plot(frames, lowest_foot_height(r.kin) * cm, color='tab:red', lw=1.2,
                label='lowest foot point: after')
        if r.ground.enabled:
            ax.plot(frames, -r.ground.offset * cm, color='tab:purple', lw=1, ls='--',
                    label='vertical correction')
        ax.axhline(cfg.contact.exit_height_m * 100, color='tab:blue', ls='--', lw=0.8)
        ax.set_title('grounding: green = a foot is in contact, orange = kept as a jump, '
                     'red = both feet above 2 cm without contact (not a jump)', fontsize=10)
        ax.set_ylabel('height [cm]')
        ax.set_xlabel('frame')
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()
        paths['ground'] = out_dir / 'ground.png'
        fig.savefig(paths['ground'], dpi=110)

    # 3. 届く高さへのクランプの補正量
    fig = Figure(figsize=(12, 3.5))
    ax = fig.subplots()
    ax.plot(frames, r.center.correction_raw * cm, color='0.6', lw=1, label='per-frame required')
    ax.plot(frames, r.center.correction * cm, color='tab:purple', lw=1.3,
            label='applied (moving min -> gaussian)')
    ax.set_ylabel('correction [cm]')
    ax.set_xlabel('frame')
    ax.set_title(f'reach clamp: over-extended frames {r.center.exceed_before} -> '
                 f'{r.center.exceed_after}')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    paths['reach_clamp'] = out_dir / 'reach_clamp.png'
    fig.savefig(paths['reach_clamp'], dpi=110)

    # 4. 足ＩＫの水平軌跡（上面図）
    fig = Figure(figsize=(12, 6))
    axes = fig.subplots(1, 2)
    for foot, ax in enumerate(axes):
        raw = r.kin_raw.joints[:, ANKLES[foot]] * cm
        tgt = r.foot_ik.target[:, foot] * cm
        ax.plot(raw[:, 0], raw[:, 2], color='0.75', lw=0.8, label='before')
        ax.plot(tgt[:, 0], tgt[:, 2], color='tab:blue', lw=1, label='after')
        locks = np.array([lock for _, _, lock in r.foot_ik.locks[foot]]).reshape(-1, 3) * cm
        ax.scatter(locks[:, 0], locks[:, 2], s=30, color='tab:red', zorder=3,
                   label='locked contacts')
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('X [cm]')
        ax.set_ylabel('Z [cm]')
        ax.set_title(f'{feet[foot]} IK top view ({len(locks)} contacts)')
        ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    paths['foot_ik_topview'] = out_dir / 'foot_ik_topview.png'
    fig.savefig(paths['foot_ik_topview'], dpi=110)
    return {k: str(v) for k, v in paths.items()}
