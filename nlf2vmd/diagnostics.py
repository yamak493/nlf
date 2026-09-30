"""検証・診断出力。処理前後の指標を JSON に、判定の妥当性を確かめるグラフを PNG に保存する。"""
import json
from pathlib import Path

import numpy as np

from .arm_collision import OVERLAP_TOL_M
from .body_model import ANKLES
from .ground import floating_frames, lowest_foot_height
from .jitter import JOINT_GROUPS, angular_acceleration

METRIC_LABELS = {
    'foot_slide_cm_per_frame': ('足滑り', '接地区間内で床に固定している点（足裏全体が着いていれば足ＩＫ、つま先だけ・かかとだけ'
                                'が着いていればその点。処理前は足首）の水平移動量の平均', '0'),
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
    'lean_deg': ('前後の傾き', '接地している足の支持点の中心から全身の重心への線の、カメラの奥行き方向の傾きの中央値'
                 '（度。+ はカメラへ近づく向き。処理前はステージ6cの補正前の姿勢）', '0 に近い'),
    'contact_overlap_frames': ('腕・指と体の重なり', '腕・手のひら・指のカプセルが、体・相手の腕と 5mm より深く重なっている'
                               'フレーム数（処理前はステージ9bの前）', '0（contacts.enabled: false では処理前と同じ）'),
    'arm_overlap_frames': ('腕の重なり', '左右の腕のカプセル（上腕・前腕・手）が 5mm より深く重なっているフレーム数'
                           '（処理前はステージ9aの前）', '0（arm_collision.mode: none では処理前と同じ）'),
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


def output_sole_heights(r, foot_ik=None, contact=None):
    """(T, 2) 出力の足ＩＫでの足裏の高さ（足首の高さ − その姿勢での足首から足裏の最下点までの高さ）。

    接地区間のうち足裏全体が着いている所は、足ＩＫの回転をその部分の平均で固定しているので、足首から足裏までの高さも
    その部分の中央値を使う（つま先だけ・かかとだけが着いている所は、回転が推定のままなのでフレームごとの値）。
    foot_ik / contact を渡すと、r の代わりにその足ＩＫ・接地区間で求める（フル [接地優先] の出力を調べる）。
    """
    from .foot_ik import FLAT
    foot_ik = r.foot_ik if foot_ik is None else foot_ik
    contact = r.contact if contact is None else contact
    ankle_above_sole = r.kin.joints[:, ANKLES, 1] - r.kin.contact_points[..., 1].min(-1)
    sole = foot_ik.target[..., 1] - ankle_above_sole
    phase = getattr(foot_ik, 'phase', None)
    for foot, segs in enumerate(foot_ik.planted_segments(contact.segments) if phase is not None
                                else contact.segments):
        for s, e in segs:
            if phase is None or phase[s, foot] == FLAT:
                sole[s:e + 1, foot] = foot_ik.target[s:e + 1, foot, 1] - np.median(
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
                                        after=foot_slide(r.foot_ik.pinned_points(),
                                                         r.foot_ik.planted_segments(seg), k)),
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
    if r.lean is not None and r.lean.enabled:
        m['lean_deg'] = dict(before=float(np.nanmedian(r.lean.before_deg)),
                             after=float(np.nanmedian(r.lean.after_deg)))
    if getattr(r, 'contacts', None) is not None and r.contacts.enabled:
        before, after = r.contacts.overlap_frames(OVERLAP_TOL_M * k)
        m['contact_overlap_frames'] = dict(before=before, after=after)
    if r.arm_collision is not None:
        before, after = r.arm_collision.overlap_frames(OVERLAP_TOL_M * k)
        m['arm_overlap_frames'] = dict(before=before, after=after)
    for key, (label, definition, goal) in METRIC_LABELS.items():
        if key in m:
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
             'floating_frames': ' フレーム', 'hover_frames': ' フレーム', 'lean_deg': ' 度',
             'arm_overlap_frames': ' フレーム', 'contact_overlap_frames': ' フレーム'}
    lines = []
    for key, m in metrics.items():
        before = fmt(m['before']) if 'before' in m else '-'
        lines.append(f"{m['label']}: {before} → {fmt(m['after'])}{units.get(key, '')}")
    return lines


# ---- グラフ ----
def _bands(ax, flags, color, alpha=0.18, scale=1.0):
    """flags の True の区間を帯で描く。scale: フレーム番号に掛ける倍率（入力の fps の列を出力のフレーム番号で描くとき）。"""
    from .filters import runs
    for s, e in runs(flags):
        ax.axvspan((s - 0.5) * scale, (e + 0.5) * scale, color=color, alpha=alpha, lw=0)


def save_plots(r, out_dir):
    # pyplot を使わない（ノートブックの描画バックエンドを変えないため）
    from matplotlib.figure import Figure
    from scipy.ndimage import gaussian_filter1d

    out_dir = Path(out_dir)
    k, cfg = r.scale, r.config
    frames = np.arange(len(r.center.delta))
    cm = 100.0 / k
    paths = {}
    feet = ('left foot', 'right foot')
    # ステージ1b・2b は入力の fps のまま行うので、その結果は出力のフレーム番号に直して描く
    src_scale = float(r.fps) / float(r.info.get('source_fps') or r.fps)

    def src_frames(n):
        return np.arange(n) * src_scale

    # 0. 慣性・重力による外れフレーム（部位ごとの外れの度合いと、重心の上下・奥行きの推定と重力の条件を満たす軌道）
    out = getattr(r, 'outliers', None)
    if out is not None and out.enabled:
        o_frames = src_frames(len(out.valid))
        fig = Figure(figsize=(12, 9))
        axes = fig.subplots(3, 1, sharex=True)
        ax = axes[0]
        colors = dict(torso='tab:red', legs='tab:blue', head='tab:purple', left_arm='tab:green',
                      right_arm='tab:olive', com='black')
        for name, score in out.scores.items():
            ax.plot(o_frames, np.minimum(score, 5.0), lw=0.8, color=colors.get(name),
                    label=f'{name} ({int(out.flags[name].sum())} replaced)')
            _bands(ax, out.flags[name], colors.get(name), alpha=0.15, scale=src_scale)
        ax.axhline(1.0, color='0.3', ls='--', lw=0.8)
        ax.set_ylabel('score (1 = threshold)')
        ax.set_title('outlier score per part (inertia) and centre of mass (gravity); '
                     'bands = replaced frames')
        ax.legend(loc='upper right', fontsize=7, ncol=3)
        com_frames = out.flags['com'] | out.flags['torso']
        if out.com is not None:
            depth_axis = out.depth_axis / np.linalg.norm(out.depth_axis)
            for ax, label, f in ((axes[1], 'height', lambda c: c[:, 1]),
                                 (axes[2], 'depth (camera axis)', lambda c: c @ depth_axis)):
                _bands(ax, com_frames, 'tab:red', alpha=0.15, scale=src_scale)
                ax.plot(o_frames, f(out.com) * 100.0, color='0.5', lw=0.8, label='estimated')
                if out.com_fit is not None:
                    ax.plot(o_frames, f(out.com_fit) * 100.0, color='tab:red', lw=1.0,
                            label='gravity-consistent')
                ax.set_ylabel(f'CoM {label} [cm]')
                ax.legend(loc='upper right', fontsize=8)
        axes[-1].set_xlabel('frame')
        fig.tight_layout()
        paths['outliers'] = out_dir / 'outliers.png'
        fig.savefig(paths['outliers'], dpi=110)

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
        if r.ground.height is not None:
            ax.plot(frames, r.ground.height * cm, color='tab:green', lw=0.8,
                    label='height used: planted feet (lowest point if none)')
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

    # 2d. 前後の傾きの補正（支持点の中心から重心への線の傾きと、補正角）
    if r.lean is not None and r.lean.enabled:
        fig = Figure(figsize=(12, 4))
        ax = fig.subplots()
        _bands(ax, r.contact.flags.any(1), 'tab:green', alpha=0.12)
        sigma = max(float(cfg.lean.window_sec), 0.5) * r.fps
        for vals, color, name in ((r.lean.before_deg, '0.6', 'before'),
                                  (r.lean.after_deg, 'tab:red', 'after')):
            ok = np.isfinite(vals)
            ax.scatter(frames, vals, s=3, color=color, alpha=0.5)
            mean = (gaussian_filter1d(np.where(ok, vals, 0.0), sigma, mode='constant')
                    / np.maximum(gaussian_filter1d(ok * 1.0, sigma, mode='constant'), 1e-9))
            ax.plot(frames, mean, color=color, lw=1.5, label=f'COM lean: {name} (dots) / '
                    'windowed mean (line)')
        ax.plot(frames, np.rad2deg(r.lean.angle), color='tab:purple', lw=1.5,
                label='applied correction (upper body rotation)')
        ax.axhline(0.0, color='k', lw=0.6)
        ax.set_title('lean toward (+) / away from (-) the camera: angle of the line from the support '
                     f'center to the whole-body COM, median {np.nanmedian(r.lean.before_deg):+.1f} -> '
                     f'{np.nanmedian(r.lean.after_deg):+.1f} deg; green = a foot is in contact',
                     fontsize=10)
        ax.set_ylabel('angle [deg]')
        ax.set_xlabel('frame')
        lim = np.nanpercentile(np.abs(np.concatenate([r.lean.before_deg, r.lean.after_deg])), 98)
        ax.set_ylim(-max(10.0, 1.2 * lim), max(10.0, 1.2 * lim))
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()
        paths['lean'] = out_dir / 'lean.png'
        fig.savefig(paths['lean'], dpi=110)

    # 2h. 胴に対する手の位置（位置を保った割合・手首の移動・肩とひじの補正角）
    if getattr(r, 'hand_reach', None) is not None and r.hand_reach.enabled:
        h = r.hand_reach
        ratio = h.info['ratio']
        fig = Figure(figsize=(12, 5))
        axes = fig.subplots(2, 1, sharex=True)
        for side, ax in enumerate(axes):
            name = ('left', 'right')[side]
            ax.plot(frames, h.shift[:, side] * cm, color='tab:blue', lw=1.2,
                    label='wrist shift to the torso-relative target [cm]')
            ax.plot(frames, h.residual[:, side] * cm, color='tab:red', lw=1, ls='--',
                    label='left over by max_deg [cm]')
            ax.plot(frames, h.correction_deg[:, side, 0], color='tab:purple', lw=1,
                    label='shoulder correction [deg]')
            ax.plot(frames, h.correction_deg[:, side, 1], color='tab:orange', lw=1,
                    label='elbow correction [deg]')
            ax.set_ylabel('[cm] / [deg]')
            ax.set_ylim(bottom=0.0)
            ax2 = ax.twinx()
            ax2.fill_between(frames, 0.0, h.weight[:, side], color='tab:green', alpha=0.15, lw=0,
                             label='weight (1 = keep the position relative to the torso)')
            ax2.set_ylim(0.0, 1.05)
            ax2.set_ylabel('weight')
            ax.set_title(f'{name} hand relative to the torso: {h.info["frames"][side]} frames; torso ratio '
                         f'MMD / SMPL width {ratio["width"]:.2f}, height {ratio["height"]:.2f}, '
                         f'depth {ratio["depth"]:.2f}', fontsize=10)
            lines = ax.get_lines()[:4] + ax2.collections[:1]
            ax.legend(lines, [ln.get_label() for ln in lines], loc='upper right', fontsize=7)
        axes[-1].set_xlabel('frame')
        fig.tight_layout()
        paths['hand_reach'] = out_dir / 'hand_reach.png'
        fig.savefig(paths['hand_reach'], dpi=110)

    # 2e. 腕どうしの貫通の防止（左右の腕のカプセルの重なりと、自重する腕に掛けた補正角・腕の移動量（画像面内 / 奥行き））
    if r.arm_collision is not None:
        a = r.arm_collision
        fig = Figure(figsize=(12, 6.5))
        ax, ax3 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
        tol = OVERLAP_TOL_M * 100.0
        top = max(tol, 1.1 * float(a.depth_before.max(initial=0.0)) * cm)
        ax.axhspan(tol, top, color='tab:red', alpha=0.08, lw=0)
        ax.plot(frames, a.depth_before * cm, color='0.6', lw=1, label='overlap depth: before')
        ax.plot(frames, a.depth_after * cm, color='tab:red', lw=1.2, label='overlap depth: after')
        ax.axhline(0.0, color='k', lw=0.6)
        ax.set_ylabel('overlap depth [cm] (< 0: apart)')
        ax.set_ylim(bottom=max(-20.0, float(min(a.depth_before.min(initial=0.0),
                                                a.depth_after.min(initial=0.0))) * cm))
        ax2 = ax.twinx()
        ax2.plot(frames, a.correction_deg, color='tab:purple', lw=1.2, label='correction of the '
                 'yielding arm (shoulder) [deg]')
        if a.elbow:
            ax2.plot(frames, a.elbow_deg, color='tab:green', lw=1.2, label='bend of the elbow [deg]')
        ax2.set_ylabel('correction [deg]')
        ax2.set_ylim(0.0, max(10.0, 1.2 * float(max(a.correction_deg.max(initial=0.0),
                                                     a.elbow_deg.max(initial=0.0)))))
        before, after = a.overlap_frames(OVERLAP_TOL_M * k)
        radius = a.radius.mean(0) * cm
        yields = {0: 'left arm yields', 1: 'right arm yields'}.get(a.side, 'no correction')
        ax.set_title(f'arm collision ({yields}): overlapping frames {before} -> {after}; '
                     f'radius upper / fore / hand '
                     f'{radius[0]:.1f} / {radius[1]:.1f} / {radius[2]:.1f} cm ({a.radius_source})',
                     fontsize=10)
        lines = ax.get_lines()[:2] + ax2.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], loc='upper right', fontsize=8)
        ax3.plot(frames, a.shift_image * cm, color='tab:blue', lw=1.2, label='image plane')
        ax3.plot(frames, a.shift_depth * cm, color='tab:orange', lw=1.2, label='depth (camera axis)')
        ax3.set_ylabel('shift of the yielding arm [cm]')
        ax3.set_ylim(bottom=0.0)
        ax3.set_title(f'largest shift of the elbow / wrist / hand tip by the correction '
                      f'(depth_cost {a.depth_cost:g}: moving in depth costs {a.depth_cost:g} x the image plane)',
                      fontsize=9)
        ax3.legend(loc='upper right', fontsize=8)
        ax3.set_xlabel('frame')
        fig.tight_layout()
        paths['arm_collision'] = out_dir / 'arm_collision.png'
        fig.savefig(paths['arm_collision'], dpi=110)

    # 2g. 腕・手のひら・指先と体・相手の腕の接触（部位ごとの重なりの深さの処理前後と、関節ごとの補正角）
    if getattr(r, 'contacts', None) is not None and r.contacts.enabled:
        from .contacts import PARTS
        c = r.contacts
        fig = Figure(figsize=(12, 7))
        axes = fig.subplots(3, 1, sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 0.8]))
        tol = OVERLAP_TOL_M * 100.0
        for side, ax in enumerate(axes[:2]):
            before = np.clip(np.nan_to_num(c.depth_before[:, side] * cm, neginf=-1.0), 0.0, None)
            after = np.clip(np.nan_to_num(c.depth_after[:, side] * cm, neginf=-1.0), 0.0, None)
            n = len(PARTS)
            img = np.concatenate([before.T, np.full((1, len(frames)), np.nan), after.T])
            im = ax.imshow(img, aspect='auto', cmap='Reds', vmin=0.0, vmax=max(2.0, tol),
                           interpolation='nearest', extent=(-0.5, len(frames) - 0.5, 2 * n + 0.5, -0.5))
            ax.set_yticks(list(range(n)) + list(range(n + 1, 2 * n + 1)))
            ax.set_yticklabels([f'before {p}' for p in PARTS] + [f'after {p}' for p in PARTS],
                               fontsize=6)
            b, a = (int((d[:, side] > OVERLAP_TOL_M * k).any(1).sum())
                    for d in (c.depth_before, c.depth_after))
            ax.set_title(f'{("left", "right")[side]} arm: overlapping frames {b} -> {a} '
                         f'(body: {c.body_source}, {c.num_body} capsules)', fontsize=9)
            fig.colorbar(im, ax=ax, pad=0.01, label='overlap [cm]')
        ax = axes[2]
        for side, ls in enumerate(('-', '--')):
            for j, (name, color) in enumerate((('shoulder', 'tab:blue'), ('elbow', 'tab:orange'),
                                               ('wrist', 'tab:green'))):
                ax.plot(frames, c.correction_deg[:, side, j], color=color, ls=ls, lw=1,
                        label=f'{("left", "right")[side]} {name}')
        ax.set_ylabel('correction [deg]')
        ax.set_xlabel('frame')
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        fig.tight_layout()
        paths['contacts'] = out_dir / 'contacts.png'
        fig.savefig(paths['contacts'], dpi=110)

    # 2f. 手首の向きの補正（MediaPipe の手のひらの向きと NLF の手首の向きの差・掛けた補正・重み）
    if getattr(r, 'wrist', None) is not None:
        w = r.wrist
        # 2b は平滑化と同じ fps（入力の fps が出力より高いときは入力の fps）で掛けている
        w_frames = frames if len(w.weight) == len(frames) else src_frames(len(w.weight))
        fig = Figure(figsize=(12, 5))
        axes = fig.subplots(2, 1, sharex=True)
        for side, ax in enumerate(axes):
            name = ('left', 'right')[side]
            d = w.disagreement_deg[:, side]
            ax.scatter(w_frames, d, s=4, color='0.55', label='NLF vs MediaPipe palm [deg]')
            ax.plot(w_frames, w.correction_deg[:, side], color='tab:purple', lw=1.2,
                    label='applied correction [deg]')
            ax.set_ylim(0.0, 180.0)
            ax.set_ylabel('[deg]')
            ax2 = ax.twinx()
            ax2.fill_between(w_frames, 0.0, w.weight[:, side], color='tab:green', alpha=0.15,
                             lw=0, label='weight of MediaPipe')
            ax2.set_ylim(0.0, 1.05)
            ax2.set_ylabel('weight')
            info = w.info[name]
            med = info['disagreement_deg_median']
            ax.set_title(f'{name} wrist: {info["observed_frames"]} usable frames'
                         + ('' if med is None else
                            f', median disagreement {med:.0f} deg, '
                            f'> 45 deg in {info["disagreement_over_45_ratio"] * 100:.0f}%'),
                         fontsize=10)
            lines = ax.collections[:1] + ax.get_lines()[:1] + ax2.collections[:1]
            ax.legend(lines, [ln.get_label() for ln in lines], loc='upper right', fontsize=8)
        axes[-1].set_xlabel('frame')
        fig.tight_layout()
        paths['wrist'] = out_dir / 'wrist.png'
        fig.savefig(paths['wrist'], dpi=110)

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

    # 3b. 脚の回転（膝の向き・曲げ）
    legs = getattr(r, 'legs', None)
    if legs is not None:
        fig = Figure(figsize=(12, 5))
        axes = fig.subplots(2, 1, sharex=True)
        for side, ax in enumerate(axes):
            _bands(ax, r.contact.flags[:, side], 'tab:green', alpha=0.12)
            _bands(ax, legs.unreached[:, side], 'tab:red', alpha=0.3)
            ax.plot(frames, legs.knee_out_deg[:, side], color='tab:blue', lw=1.2,
                    label='knee direction from the pelvis front [deg] (+ = out)')
            ax.plot(frames, legs.bend_deg[:, side], color='tab:orange', lw=1,
                    label='knee bend [deg]')
            ax.axhline(0.0, color='k', lw=0.6)
            ax.set_ylabel('[deg]')
            ax.set_title(f'{feet[side].split()[0]} leg keys: green = contact, red = foot IK out of reach',
                         fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
        axes[-1].set_xlabel('frame')
        fig.tight_layout()
        paths['legs'] = out_dir / 'legs.png'
        fig.savefig(paths['legs'], dpi=110)

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
