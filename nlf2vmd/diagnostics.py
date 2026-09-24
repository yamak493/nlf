"""検証・診断出力。処理前後の指標を JSON に、判定の妥当性を確かめるグラフを PNG に保存する。"""
import json
from pathlib import Path

import numpy as np

from .jitter import JOINT_GROUPS, angular_acceleration

METRIC_LABELS = {
    'foot_slide_cm_per_frame': ('足滑り', '接地区間内の足ＩＫの水平移動量の平均', '0'),
    'penetration_frames': ('埋まり', '足ＩＫの差分Yが0未満のフレーム数（足ごと）', '0'),
    'center_jitter_cm_per_frame2': ('センターの震え', 'センター・グルーブ位置の加速度の絶対値の平均（X, Y, Z）',
                                    '処理前より大幅に減少'),
    'pose_jitter_deg_per_s2': ('姿勢の震え', '関節の角加速度の平均（グループ別）', '処理前より減少'),
    'overextended_frames': ('脚の伸び切り', '股関節から足ＩＫまでの距離が脚長の98%を超えるフレーム数', '0'),
    'contact_segments': ('接地切り替え回数', '足ごとの接地区間の数', '目視との整合を確認'),
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
    """r: pipeline.ConversionResult。処理前・処理後を並べた辞書を返す。"""
    k, seg = r.scale, r.contact.segments
    raw_ik_delta = r.foot_ik.raw - r.ankle_rest[None]
    m = {
        'foot_slide_cm_per_frame': dict(before=foot_slide(r.foot_ik.raw, seg, k),
                                        after=foot_slide(r.foot_ik.target, seg, k)),
        'penetration_frames': dict(before=(raw_ik_delta[..., 1] < 0).sum(0).tolist(),
                                   after=(r.foot_ik.delta[..., 1] < 0).sum(0).tolist()),
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
             ' cm/フレーム²（X, Y, Z）', 'pose_jitter_deg_per_s2': ' deg/s²'}
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
        raw = r.foot_ik.raw[:, foot] * cm
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
