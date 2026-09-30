"""フル [接地優先]（motion_full_locked.vmd）: ジャンプ・片足上げなどで足が床から離れる所も、足を床に着ける。

フル（convert の結果）は、ジャンプ（ステージ6a で、骨盤が重力の放物線を描く短い滞空）の高さをそのまま残す。
単眼推定では、片足を上げたときに体全体が持ち上がって見え、それがジャンプと判定されたり、支持脚が接地判定から
漏れて床の上を漂ったりする。フル [接地優先] は、フルの動きから次のように作る（ジャンプはしなかったものとし、
既定（locked.both_feet: true）では、どのフレームでも両足を床に着ける。片足を上げる動きも、足を床に着けたまま
床の上を動かす動きになる。false なら、片足だけの浮きは残し、両足とも浮いたときだけ低いほうの足を床に着ける）。

1. ジャンプとして残した区間（ステージ6a の滞空）で、体（姿勢）が床から浮いている高さ（接地している足があれば
   その足、無ければ両足の、足裏の最も低い点の高さ。ステージ6a と同じ）を求め、軽くならす（locked.sigma_frames）
2. フルの出力の足ＩＫでの足裏の高さ（床からの浮き）を足ごとに求める。高さは MMD のモデルの足の形で求める
   （mmd_sole_heights。足ＩＫの位置に、足ＩＫの回転で回したかかと（足首の真下の床の点）とつま先（つま先ＩＫの真下の
   床の点）を置いた低いほう）。SMPL の足の形で求めると、足首の高さ・足の長さの比がモデルと違うので、かかとを
   上げて足を傾けたとき（足を引き寄せる・つま先で立つ）に、SMPL ではつま先が床に着いていても、モデルでは浮く
3. 足ＩＫを下ろす。locked.both_feet なら、足ごとにその足の浮きだけ下ろす（両足とも足裏がちょうど床に着く）。
   false なら、下ろす量は両足の浮きの小さいほう（両足とも浮いているフレームで、低いほうの足が床に着くまで）と
   1. の大きいほうで、足ごとに、その足の浮きより下へは下ろさない（低いほうの足はちょうど床に着き、高いほうの足は
   体と一緒に下りる。床より下へは下ろさない）
4. 足を大きく下ろしたフレーム（接地判定の開始の高さ contact.enter_height_m より下ろした所と、その前後
   foot_ik.blend_frames。フルでは足の高さのせいで接地にならなかった所）で、床の近くにある足を床に固定する
   （足ＩＫのロック）。フルの接地区間は、足首がそのロック位置から水平に locked.still_m 以内にある間だけ前後へ延ばし、
   延ばしたフレームはフルの接地区間のロック位置・回転をそのまま保つ（前後 2 つの接地区間とつながったときは、2 つの
   ロック位置・回転の間をなめらかにつなぐ）。どの接地区間ともつながらない所は、フルと同じしきい値で接地を判定し、
   足首が動かない区間だけ固定する（lock_flags）。水平に動いている足は固定せず、床の高さへ下ろすだけ
   足ＩＫを作り直したあと、もう一度モデルの足の形で高さを求め、足裏の最も低い点がちょうど床に来るように上下を
   合わせる（both_feet なら両足。false なら両足とも浮いたときの低いほうの足）
5. センターは、骨盤・足首を 1. だけ下ろした体と 4. の足ＩＫ・接地で、ステージ8（平滑化と届く高さへのクランプ）を
   やり直して求める（体を下ろさない所は、足ＩＫが変わった分だけクランプが変わる）。水平（X・Z）はフルと同じにする

回転（上半身・腕・下半身など）とセンターの水平移動はフルと同じ。浮いていたフレームから離れた所では、センター・
グルーブ・足ＩＫもフルと同じ値になるので、2 つの VMD の好きな所を切り貼りできる（フルと違うフレームは
info['differ_frames'] とログに出す）。
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import maximum_filter1d

from . import filters, quat
from .body_model import ANKLES
from .center import apply_reach_clamp, stabilize_center
from .contact import ContactResult, clean_flags, hysteresis
from .foot_ik import build_foot_ik, mmd_sole_points, sole_floor_levels
from .ground import support_height
from .skeleton import SIDES

HOVER_M = 0.01      # 足裏がこれより上 [m] なら「浮いている」（ログ・info の数え方）
SAME_POS_M = 0.005  # フルとの差がこれ以下 [m]・SAME_ROT_DEG 度以下なら「フルと同じ」（切り貼りしても段差が見えない）
SAME_ROT_DEG = 0.5


@dataclass
class LockedMotion:
    center: object             # CenterResult（delta がセンター＋グルーブの差分）
    foot_ik: object            # FootIKResult
    contact: object            # ContactResult（フルの接地区間に、床に固定したフレームを足したもの）
    jump: np.ndarray           # (T,) 体（姿勢）を下ろした量（1.）
    foot_drop: np.ndarray      # (T, 2) 足ＩＫを下ろした量（3.）
    info: dict


def jump_height(result):
    """(T,) フルでジャンプとして残した区間で、体（姿勢）が床から浮いている高さ（1.）。ならす前。"""
    T = len(result.contact.flags)
    flight = getattr(result.ground, 'flight', None)
    if flight is None or not np.any(flight):
        return np.zeros(T)
    height = support_height(result.kin, result.contact)
    return np.where(np.asarray(flight, bool), np.maximum(height, 0.0), 0.0)


def mmd_sole_heights(skel, ik):
    """(T, 2) MMD のモデルの足裏（かかと・つま先の低いほう）の床からの高さ（足ＩＫの位置と回転から）。"""
    points = mmd_sole_points(skel)
    rest_y = np.array([skel.internal(s + '足ＩＫ')[1] for s in SIDES])
    heights = np.empty(np.shape(ik.delta)[:2] + (2,))
    for side in range(2):
        for i in range(2):
            heights[:, side, i] = quat.rotate(ik.rotation[:, side],
                                              np.broadcast_to(points[side, i], (len(ik.delta), 3)))[:, 1]
    return rest_y + ik.delta[..., 1] + heights.min(-1)


def ground_feet(skel, ik, both_feet):
    """足ＩＫを上下に動かして、モデルの足裏を床に合わせた FootIKResult（4. の最後）。

    both_feet なら両足とも足裏の最も低い点をちょうど床に（浮いていれば下ろし、埋まっていれば上げる）。false なら、
    両足とも浮いているフレームだけ、低いほうの足が床に着くまで両足を下ろす（床より下へは下ろさない）。
    """
    sole = mmd_sole_heights(skel, ik)
    if both_feet:
        shift = sole
    else:
        shift = np.minimum(np.maximum(sole.min(1), 0.0)[:, None], np.maximum(sole, 0.0))
    delta = np.array(ik.delta, copy=True)
    delta[..., 1] -= shift
    return replace(ik, delta=delta, target=np.asarray(ik.target) + (delta - ik.delta))


def sole_clearance(result):
    """(T, 2) フルの出力の足ＩＫでの、モデルの足裏の床からの高さ（2.）。"""
    return mmd_sole_heights(result.skeleton, result.foot_ik)


def foot_drops(clearance, jump, both_feet=True):
    """(T, 2) 足ＩＫを下ろす量（3.）。"""
    clearance = np.asarray(clearance, np.float64)
    if both_feet:
        return np.maximum(clearance, 0.0)
    drop = np.maximum(np.maximum(clearance.min(1), 0.0), jump)
    return np.minimum(drop[:, None], np.maximum(clearance, 0.0))


def lock_flags(contact, ankles, sole, lowered, valid, cfg_contact, still, unit):
    """(T, 2) bool 足を床に固定するフレーム（4.）。フルの接地に、足を下ろしたフレーム lowered で足した接地を加える。

    ankles: (T, 2, 3) 足首の位置 / sole: (T, 2) 下ろした後の足裏の高さ / still: ロック位置からの水平の距離の上限。
    足裏が接地終了の高さ（contact.exit_height_m）より低いフレームで、
      * フルの接地区間を、足首がそのロック位置（区間の足首の水平位置の中央値）から still 以内にある間だけ前後へ延ばす
        （延ばしたフレームは、フルの接地区間のロックの値を保つ）
      * どの接地区間ともつながらない所は、フルと同じしきい値（contact.*）で接地を判定し、足首が区間の中央値から
        still 以内に留まっている区間（contact.min_contact_frames 以上）だけ足す
    最後に、contact.fill_gap_frames 以下の隙間を埋める。
    valid: (T,) bool 人物を検出できたフレーム（False のフレームでは接地を始めない。フルと同じ）。
    """
    full = np.asarray(contact.flags, bool)
    T = len(full)
    speeds = np.asarray(contact.speeds, np.float64)
    near = np.asarray(lowered, bool)[:, None] & (sole < float(cfg_contact.exit_height_m) * unit)
    enter = (near & (sole < float(cfg_contact.enter_height_m) * unit)
             & (speeds < float(cfg_contact.enter_speed_m_per_s) * unit).any(-1))
    if valid is not None:
        enter &= np.asarray(valid, bool)[:, None]
    stay = near & (speeds < float(cfg_contact.exit_speed_m_per_s) * unit).any(-1)
    min_len = int(cfg_contact.min_contact_frames)
    flags = full.copy()
    for foot in range(2):
        xz = np.asarray(ankles, np.float64)[:, foot][:, [0, 2]]
        f = flags[:, foot]
        for s, e in filters.runs(full[:, foot]):
            lock = np.median(xz[s:e + 1], axis=0)
            for t, step in ((e + 1, 1), (s - 1, -1)):
                while (0 <= t < T and near[t, foot] and not full[t, foot]
                       and np.linalg.norm(xz[t] - lock) < still):
                    f[t] = True
                    t += step
        for s, e in filters.runs(hysteresis(enter[:, foot], stay[:, foot]) & ~f):
            touching = (s > 0 and f[s - 1]) or (e < T - 1 and f[e + 1])
            still_run = (np.linalg.norm(xz[s:e + 1] - np.median(xz[s:e + 1], axis=0), axis=1)
                         < still).all()
            if e - s + 1 >= min_len and not touching and still_run:
                f[s:e + 1] = True
        flags[:, foot] = clean_flags(f, int(cfg_contact.fill_gap_frames), 1)
    return flags


def locked_motion(result):
    """フル [接地優先] の動き（LockedMotion）。result は convert の結果（フル）。"""
    cfg, k, fps = result.config, result.scale, result.fps
    kin, contact, ik_full = result.kin, result.contact, result.foot_ik
    T = len(contact.flags)

    # 1〜3. 体（姿勢）と足ＩＫを下ろす量
    jump = filters.gaussian_time(jump_height(result), float(cfg.locked.sigma_frames))
    clearance = sole_clearance(result)
    foot_drop = foot_drops(clearance, jump, bool(cfg.locked.both_feet))
    down = np.zeros((T, 2, 3))
    down[..., 1] = foot_drop

    # 4. 下ろした足を床に固定し直す（フルの接地区間のロックの値はそのまま使う）
    # 接地判定の開始の高さより下ろしたフレーム（フルでは高さのせいで接地にならなかった所）とその前後だけ。
    # それより浮きが小さい所は、フルでも高さの条件は満たしていた（足が動いていたので接地にならなかった）
    lowered = maximum_filter1d(foot_drop.max(1) > float(cfg.contact.enter_height_m) * k,
                               2 * int(cfg.foot_ik.blend_frames) + 1, mode='constant')
    flags = lock_flags(contact, ik_full.raw, clearance - foot_drop, lowered,
                       getattr(result.motion, 'valid', None), cfg.contact,
                       float(cfg.locked.still_m) * k, k)
    # 接地点の高さは足首と一緒に下ろす（床への吸着は「足首 − 足裏」の高さなので、下ろしても変わらない）
    locked_contact = ContactResult(flags, [filters.runs(flags[:, f]) for f in range(2)],
                                   np.asarray(contact.heights) - foot_drop[..., None],
                                   contact.speeds)
    rt = result.retargeter
    # フルと同じく、つま先だけ・かかとだけが床に着いている所はその点を固定する（フルの接地区間では同じ値になる）
    ik = build_foot_ik(ik_full.raw - down, result.ankle_rest, rt.foot_ik_quats(kin.glob_rot),
                       locked_contact, fps, k, cfg.foot_ik, swing=ik_full.swing - down,
                       anchor=contact.flags, sole_points=mmd_sole_points(result.skeleton),
                       sole_floor=sole_floor_levels(result.skeleton, result.ankle_rest))
    ik = ground_feet(result.skeleton, ik, bool(cfg.locked.both_feet))

    # 5. 体を下ろしてセンターを求め直す（ジャンプの高さはステージ8の平滑化の前に除く）。水平はフルと同じにして
    # （モードB は接地している足から骨盤を求めるので、固定した足を足すと水平も少し変わる）、クランプを掛け直す
    body = np.zeros((T, 3))
    body[:, 1] = jump
    lower_rot = rt.global_matrix('下半身', kin.glob_rot)
    center = stabilize_center(result.kin_raw.root_pos, kin.root_pos - body, result.pelvis_rest,
                              kin.joints[:, ANKLES] - body[:, None], ik, locked_contact,
                              result.reach_geometry, lower_rot, fps, k, cfg.center)
    smoothed = np.array(center.smoothed, copy=True)
    smoothed[:, [0, 2]] = result.center.smoothed[:, [0, 2]]
    delta, corr_raw, corr, exceed_before, exceed_after = apply_reach_clamp(
        result.reach_geometry, smoothed, lower_rot, ik.delta, cfg.center, k)
    center = replace(center, delta=delta, smoothed=smoothed, correction_raw=corr_raw,
                     correction=corr, exceed_before=exceed_before, exceed_after=exceed_after)

    sole_after = mmd_sole_heights(result.skeleton, ik)
    flight = getattr(result.ground, 'flight', None)
    rot_diff = np.rad2deg(quat.angle_between(ik.rotation, ik_full.rotation)).max(1)
    differ = ((np.abs(center.delta - result.center.delta).max(1) > SAME_POS_M * k)
              | (np.abs(ik.delta - ik_full.delta).max((1, 2)) > SAME_POS_M * k)
              | (rot_diff > SAME_ROT_DEG))
    info = dict(
        jump_frames=int(np.asarray(flight, bool).sum()) if flight is not None else 0,
        max_jump_cm=float(jump.max(initial=0.0) / k * 100.0),
        max_foot_drop_cm=float(foot_drop.max(initial=0.0) / k * 100.0),
        added_lock_frames=[int((flags[:, f] & ~contact.flags[:, f]).sum()) for f in range(2)],
        hover_frames=dict(before=int((clearance.min(1) > HOVER_M * k).sum()),
                          after=int((sole_after.min(1) > HOVER_M * k).sum())),
        one_foot_hover_frames=dict(before=int((clearance.max(1) > HOVER_M * k).sum()),
                                   after=int((sole_after.max(1) > HOVER_M * k).sum())),
        max_highest_sole_cm=dict(before=float(clearance.max(1).max(initial=0.0) / k * 100.0),
                                 after=float(sole_after.max(1).max(initial=0.0) / k * 100.0)),
        max_lowest_sole_cm=dict(before=float(clearance.min(1).max(initial=0.0) / k * 100.0),
                                after=float(sole_after.min(1).max(initial=0.0) / k * 100.0)),
        exceed_before=center.exceed_before, exceed_after=center.exceed_after,
        differ_frames=filters.runs(differ))
    return LockedMotion(center, ik, locked_contact, jump, foot_drop, info)


def format_ranges(ranges, fps, limit=8):
    """[(開始, 終了)] を「96〜114（3.2〜3.8 秒）, 150（5.0 秒）, ...」の文字列にする（limit 区間まで）。"""
    parts = [f'{s}（{s / fps:.1f} 秒）' if s == e else
             f'{s}〜{e}（{s / fps:.1f}〜{(e + 1) / fps:.1f} 秒）' for s, e in ranges[:limit]]
    more = f' ほか {len(ranges) - limit} 区間' if len(ranges) > limit else ''
    return ', '.join(parts) + more
