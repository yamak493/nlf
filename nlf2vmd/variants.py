"""出力する VMD の種類: フル / フル [接地優先] / フル [移動なし] / 上半身のみ / 表情のみ。

convert（pipeline.py）の結果から、種類ごとにボーンのキー列を作って書き出す。口パク（lipsync.py の
MorphTrack）と手の形（hands.py の指ボーンの BoneTrack）はどの種類にも足せる（ノートブックでは、口パクは
フル・フル [接地優先]・フル [移動なし]・表情のみ に、手の形は フル・フル [接地優先]・フル [移動なし]・上半身のみ
に入れる）。

  full        センター・グルーブ・足ＩＫ・全身の回転（convert の結果そのまま）。脚（足・ひざ・足首）の回転は、どの種類でも
              その種類のセンター・足ＩＫに届くように解き直す（ステージ9l。pipeline.build_tracks）
  locked      フルから、ジャンプ・片足上げなどで両足が床から離れた所を除く（locked.py）。ジャンプとして残した区間は
              体を床へ下ろしてセンターを求め直し、どのフレームでも両足（locked.both_feet: false なら低いほうの
              足）を床に着け、床に下ろした足はフルで着いていた位置に固定する。回転とセンターの水平移動はフルと同じで、浮いていた所から
              離れたフレームはフルと同じ値になる（フルと切り貼りして使う）
  no_move     フルから体の水平移動（センターの X・Z）を除く。足ＩＫは、接地している間はその場に固定し
              （足が滑らない）、足跡を「その足が着いている間のセンターの平均位置」の分だけずらす。
              足が浮いている間は、足が次の足跡へ進んだ割合だけずらし量も進める（その場で足踏みする）。
              上下（グルーブ）は、ひざの曲げ・しゃがみ・ジャンプでの骨盤の高さなので残し、脚が届く高さへの
              クランプだけ掛け直す（上下も除くと、脚が床に届かず足が浮くため）
  upper_body  上半身から先（上半身・上半身2・首・頭・肩・腕・ひじ・手首）の回転だけ。センター・下半身・
              脚（足・ひざ・足首）・足ＩＫにはキーを打たない。体全体の向き（下半身の鉛直軸まわりの回転）を除くので、
              振り向いても上半身だけが回ることはなく、下半身に対するひねり・おじぎ・体の傾きは残る
  face        ボーンのキーは無し（口パクのモーフのキーだけ）
"""
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import filters, quat
from .center import apply_reach_clamp
from .locked import HOVER_M, SAME_POS_M, SAME_ROT_DEG, format_ranges, locked_motion
from .pipeline import build_tracks
from .vmd import write_vmd

VARIANTS = ('full', 'locked', 'no_move', 'upper_body', 'face')
LABELS = dict(full='フル', locked='フル [接地優先]', no_move='フル [移動なし]', upper_body='上半身のみ',
              face='表情のみ')
LOWER_BODY_BONES = ('下半身',)
STRIDE_M = 0.1   # 移動なし: 足跡がこれ以上 [m] 動いた歩は、足が進んだ割合だけで足跡をずらす
STILL_M = 0.03   # 移動なし: 足ＩＫがロック位置からこれ以内 [m] なら、足が着いたままとみなす


def heading_matrices(rot):
    """回転 (T, 3, 3) に最も近い鉛直軸（Y）まわりの回転 (T, 3, 3)。内部座標（Y 上向き・+Z 正面）。

    trace(Ry(θ)^T R) を最大にする θ。前後だけ・左右だけの傾きは向きを変えない（おじぎで正面が
    ほぼ真下を向いても向きが決まる）。両方に傾いたときは、傾きが小さいほど元の向きに近い。
    """
    rot = np.asarray(rot, np.float64)
    theta = np.arctan2(rot[:, 0, 2] - rot[:, 2, 0], rot[:, 0, 0] + rot[:, 2, 2])
    c, s = np.cos(theta), np.sin(theta)
    out = np.zeros(rot.shape)
    out[:, 0, 0], out[:, 0, 2], out[:, 1, 1] = c, s, 1.0
    out[:, 2, 0], out[:, 2, 2] = -s, c
    return out


def upper_body_quats(result):
    """上半身のみ: ボーン名 → ローカル回転 (T, 4)（内部座標）。下半身の鉛直軸まわりの向きを除く。"""
    rt = result.retargeter
    glob = rt.global_matrices(result.kin.glob_rot)
    heading_inv = np.swapaxes(heading_matrices(glob[LOWER_BODY_BONES[0]]), -1, -2)
    bones = [b for b in rt.bones if b not in LOWER_BODY_BONES]
    out = {}
    for name in bones:
        if rt.keyed_parent[name] in bones:
            out[name] = result.local_quats[name]
        else:
            # 親にキーを打たない（初期姿勢のまま）ので、ローカル回転 = 向きを除いた大域回転
            out[name] = quat.make_continuous(quat.from_matrix(heading_inv @ glob[name]))
    return out


def footprints(segments, target, stride, still):
    """足跡の区間 [(開始, 終了)]。接地区間を、足が止まっている範囲まで広げてまとめたもの。

    前後の接地区間でロック位置が水平に stride 未満しか離れていなければ 1 つの足跡にまとめ（接地判定が
    途切れて 2 つに分かれた区間や、足を上げて同じ所に戻した動き）、区間の外でも足ＩＫがロック位置から
    still 以内にあるフレームは足跡に含める（接地判定が遅れた・早く切れたフレーム）。どちらも、足跡の間の
    センターの平均位置を、足が本当に着いていた間の値にするため。
    """
    horizontal = np.array([1.0, 0.0, 1.0])
    groups = []
    for s, e in segments:
        if groups and np.linalg.norm((target[s] - target[groups[-1][1]]) * horizontal) < stride:
            groups[-1][1] = e
        else:
            groups.append([s, e])
    out = []
    for i, (s, e) in enumerate(groups):
        lo = out[-1][1] + 1 if out else 0
        hi = groups[i + 1][0] - 1 if i + 1 < len(groups) else len(target) - 1
        start, end = target[s], target[e]
        while s > lo and np.linalg.norm(target[s - 1] - start) < still:
            s -= 1
        while e < hi and np.linalg.norm(target[e + 1] - end) < still:
            e += 1
        out.append((s, e))
    return out


def footprint_shift(travel, segments, target, stride, still):
    """フル [移動なし] で 1 つの足ＩＫから引く水平の量 (T, 3)。

    足跡（footprints）の間は一定（その間のセンターの平均位置）なので、着いている足は動かない。足跡の間は、
    足が前の足跡から次の足跡へ進んだ割合だけ、ずらし量も次の足跡の値へ進める。足跡がほとんど動かない
    （その場で足を上げ下げした）ときは、進んだ割合の代わりに時間のスムーズステップを使う。
    最初の足跡より前・最後の足跡より後は、フレーム 0・最後のフレームを仮の足跡とし、そのずらし量は
    足の水平の動きをちょうど打ち消す値にする（足は最初・最後の足跡の上で上下するだけになる）。
    接地が無い足は、センターの動きをそのまま引く。
    travel: (T, 3) センターの水平移動 / target: (T, 3) その足の足ＩＫのターゲット（フルの値）/
    stride: この距離 [MMD 単位] 以上動いた足跡は、進んだ割合だけでつなぐ / still: footprints の still
    """
    travel = np.asarray(travel, np.float64)
    target = np.asarray(target, np.float64)
    T = len(travel)
    if not segments:
        return travel.copy()
    shift = np.empty_like(travel)
    horizontal = np.array([1.0, 0.0, 1.0])
    anchors = [(s, e, travel[s:e + 1].mean(0))
               for s, e in footprints(segments, target, stride, still)]
    first, last = anchors[0][0], anchors[-1][1]
    if first > 0:
        anchors.insert(0, (0, 0, anchors[0][2] - (target[first] - target[0]) * horizontal))
    if last < T - 1:
        end = anchors[-1][2] + (target[T - 1] - target[last]) * horizontal
        anchors.append((T - 1, T - 1, end))
    for s, e, ref in anchors:
        shift[s:e + 1] = ref
    for (_, e0, r0), (s1, _, r1) in zip(anchors[:-1], anchors[1:]):
        n = s1 - e0 - 1
        if n <= 0:
            continue
        w_time = filters.smoothstep(np.arange(1, n + 1) / (n + 1.0))
        step = (target[s1] - target[e0]) * horizontal
        length = float(np.linalg.norm(step))
        w = w_time
        if length > 0:
            progress = np.clip((target[e0 + 1:s1] - target[e0]) @ step / length ** 2, 0.0, 1.0)
            alpha = min(1.0, length / stride)
            w = alpha * progress + (1.0 - alpha) * w_time
        shift[e0 + 1:s1] = r0 + (r1 - r0) * w[:, None]
    return shift


def no_move_motion(result):
    """フル [移動なし] の (センターの差分 (T, 3), 足ＩＫ, 情報 dict)。回転はフルと同じ。"""
    center, ik, contact, k = result.center, result.foot_ik, result.contact, result.scale
    T = len(center.delta)
    travel = np.zeros((T, 3))
    travel[:, [0, 2]] = center.delta[:, [0, 2]]
    shift = np.stack([footprint_shift(travel, contact.segments[foot], ik.target[:, foot],
                                      STRIDE_M * k, STILL_M * k) for foot in range(2)], axis=1)
    ik = replace(ik, target=ik.target - shift, delta=ik.delta - shift)
    # 上下はクランプ前の値から、足の位置が変わった脚が届く高さへ下げ直す
    center_delta = np.zeros((T, 3))
    center_delta[:, 1] = center.smoothed[:, 1]
    lower_rot = result.retargeter.global_matrix('下半身', result.kin.glob_rot)
    # 届く高さを判定する脚はフルと同じ（足ＩＫは水平にずらしただけなので、接地も足裏の高さも変わらない）
    center_delta, _, corr, before, after = apply_reach_clamp(
        result.reach_geometry, center_delta, lower_rot, ik.delta, result.config.center, k,
        getattr(center, 'legs', None))
    info = dict(removed_travel_cm=float(np.linalg.norm(travel, axis=1).max() / k * 100.0),
                max_drop_cm=float(max(0.0, -corr.min()) / k * 100.0),
                exceed_before=before, exceed_after=after)
    return center_delta, ik, info


def log_locked(info, fps, log):
    hover, one = info['hover_frames'], info['one_foot_hover_frames']
    added = info['added_lock_frames']
    jump = (f'ジャンプとして残した {info["jump_frames"]} フレームの体を床へ下ろした量 最大 '
            f'{info["max_jump_cm"]:.1f} cm' if info['jump_frames'] else 'ジャンプとして残した区間なし')
    log(f'[接地優先] {jump} / 足ＩＫを下ろした量 最大 {info["max_foot_drop_cm"]:.1f} cm / '
        f'床に固定した足のフレームを追加 左 {added[0]}・右 {added[1]} / {HOVER_M * 100:.0f}cm より浮いたフレーム '
        f'両足とも {hover["before"]} → {hover["after"]}・片足でも {one["before"]} → {one["after"]}')
    if info['differ_frames']:
        log('[接地優先] フルと違うフレーム: ' + format_ranges(info['differ_frames'], fps))
    else:
        log(f'[接地優先] フルと同じ動きです（どのフレームもフルとの差が {SAME_POS_M * 1000:.0f}mm・'
            f'{SAME_ROT_DEG:g} 度以下）')
    if info['exceed_after']:
        log(f'⚠️ [接地優先] {info["exceed_after"]} フレームで脚が伸び切っています（床へ下ろした足に脚が届きません。'
            'center.reach_max_drop_m を大きくすると下げられます）')


def variant_tracks(result, kind, log=print):
    """種類 kind（VARIANTS）のボーンのキー列（BoneTrack のリスト）。"""
    log = log or (lambda *a: None)
    if kind not in VARIANTS:
        raise ValueError(f'VMD の種類は {" / ".join(VARIANTS)} のいずれかです: {kind}')
    if kind == 'full':
        return result.tracks
    if kind == 'face':
        return []
    if result.retargeter is None or result.skeleton is None:
        raise ValueError('convert の結果に骨格の情報がありません（この版の convert で変換し直してください）')
    args = (result.contact, result.config, result.scale)
    poles = getattr(result, 'knee_poles', None)
    if kind == 'upper_body':
        return build_tracks(result.skeleton, None, None, upper_body_quats(result), *args)
    if kind == 'locked':
        lm = locked_motion(result)
        log_locked(lm.info, result.fps, log)
        return build_tracks(result.skeleton, lm.center.delta, lm.foot_ik, result.local_quats,
                            lm.contact, result.config, result.scale, poles=poles)
    center_delta, ik, info = no_move_motion(result)
    log(f'[移動なし] 取り除いた水平移動 最大 {info["removed_travel_cm"]:.1f} cm / '
        f'届く高さへ下げた量 最大 {info["max_drop_cm"]:.1f} cm')
    if info['exceed_after']:
        log(f'⚠️ [移動なし] {info["exceed_after"]} フレームで脚が伸び切っています'
            '（足跡の位置が骨盤から遠い所。center.reach_max_drop_m を大きくすると下げられます）')
    return build_tracks(result.skeleton, center_delta, ik, result.local_quats, *args, poles=poles)


def write_variant(result, kind, path, morphs=(), bones=(), log=print):
    """種類 kind の VMD を path に書き出す。戻り値はキーの総数。

    morphs: 足すモーフのキー（MorphTrack のリスト。口パク）
    bones: 足すボーンのキー（BoneTrack のリスト。hands.py の指など）。同じ名前のボーンは置き換える
    """
    tracks = variant_tracks(result, kind, log)
    if bones:
        names = {t.name for t in bones}
        tracks = [t for t in tracks if t.name not in names] + list(bones)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_vmd(path, tracks, result.info.get('model_name') or 'nlf2vmd', morphs=morphs)
