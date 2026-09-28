"""ステージ2b: 手首の向きの補正（MediaPipe Hands の 3D の手のランドマークから）。

NLF の手首（SMPL の関節 20・21）の回転は、全身を 384×384 に収めた画像から手の頂点に SMPL を当てはめた結果で、
手は十数〜数十 px しか写っていない。MediaPipe Hands のランドマークモデルは手だけを 224×224 に切り出して
21 点を推定するので、手のひらの向き（手首・人差し指と小指の付け根で決まる面）はこちらのほうが直接的に決まる。

hand_detect.py はランドマークモデルに入れる前に、画像から ROI（中心 x, y・一辺・回転角）を「手首 → 指の付け根が
上を向く」ように回して切り出している。モデルの 3D の点（world、メートル）はその切り出した画像の座標系のままなので、

  1. ROI の回転角だけ画像の面内（カメラの z 軸まわり）に回し戻す（hand_detect.to_image の x, y と同じ回し方）
  2. 切り出した所がカメラの光軸からずれている分、光軸 (0, 0, 1) を ROI の中心を通る視線へ重ねる回転を掛ける
     （切り出した画像は「その視線の方向を正面に見た小さなカメラ」とみなす。K は NLF と同じ長辺の画角 55 度）

でカメラ座標（x = 右・y = 下・z = 奥。MediaPipe の 3D の点と同じ向き）に直す。そこから手のひらの座標系
（前 = 手首 → 中指の付け根、横 = 小指の付け根 → 人差し指の付け根、法線 = 前 × 横）を作り、SMPL の初期姿勢の
手のひらの座標系（前 = 手首 → 手の関節、横 = +Z（T ポーズの手のひらは下向きで、親指が前））を重ねる回転を、
手首の大域回転の目標にする。左右どちらの手も同じ作り方なので、左手・右手の鏡像の違いは自然に扱える。

使うのは次の条件を満たすフレームだけ:
  * 手の存在スコアが min_presence 以上（hand_detect で体の手首から離れた所に見つかった手は、すでに 0）
  * 回し戻した 3D の点の「手首 → 中指の付け根」の画面内の向きが、画像上の 21 点の同じ向きと check_2d_deg 以内
    （ROI の戻し方が合っていること・3D の点と画像上の点が同じ手を表していることの確認）
  * 前腕に対する手首の曲げ・ひねりが、関節として無理のない範囲（max_swing_deg・max_twist_deg）
  * NLF の手首の向きとの差が max_disagree_deg 以内（既定の 180 度では使わない）
存在スコアを重み（min_presence で 0、full_presence 以上で 1）にして、NLF の手首のローカル回転（前腕に対する回転）と球面線形補間する。見えない区間は、直前・直後に
見えたときの手首のローカル回転を続け、重みを blend_sec のガウシアン（2σ で打ち切る）でならして NLF の向きへ戻す。最後に One Euro
フィルタでならす（1 フレームごとの MediaPipe の推定はぶれるため）。
"""
from dataclasses import dataclass, field

import numpy as np

from . import filters, quat
from .hand_detect import intrinsics
from .motion_io import CAMERA_TO_YUP

WRIST = (20, 21)
ELBOW = (18, 19)
HAND = (22, 23)
# MediaPipe の 21 点: 0 = 手首、5 = 人差し指の付け根、9 = 中指の付け根、17 = 小指の付け根
LM_WRIST, LM_INDEX, LM_MIDDLE, LM_PINKY = 0, 5, 9, 17


def roi_rotation(angle):
    """(..., 3, 3) 切り出した画像の座標 → 画像の座標（ROI の回転を戻す。z 軸まわり）。"""
    a = np.asarray(angle, np.float64)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1] = c, -s, s, c
    R[..., 2, 2] = 1.0
    return R


def ray_rotation(center, image_size, fov_deg=55.0):
    """(..., 3, 3) 光軸 (0, 0, 1) を、画像上の点 center (..., 2) [px] を通る視線へ重ねる最小回転。"""
    K = intrinsics(image_size, fov_deg)
    c = np.asarray(center, np.float64)
    ray = np.stack([(c[..., 0] - K[0, 2]) / K[0, 0], (c[..., 1] - K[1, 2]) / K[1, 1],
                    np.ones(c.shape[:-1])], axis=-1)
    flat = ray.reshape(-1, 3)
    R = np.stack([quat.to_matrix(quat.from_two_vectors([0.0, 0.0, 1.0], r)) for r in flat])
    return R.reshape(c.shape[:-1] + (3, 3))


def world_to_camera(world, roi, image_size, fov_deg=55.0):
    """MediaPipe の 3D の点 (..., 21, 3)（切り出した画像の座標系）→ カメラ座標の向き（原点はそのまま）。

    roi: (..., 4) hand_detect の ROI（中心 x, y [px]・一辺 [px]・回転角 [rad]）。
    """
    roi = np.asarray(roi, np.float64)
    R = ray_rotation(roi[..., :2], image_size, fov_deg) @ roi_rotation(roi[..., 3])
    return np.einsum('...ab,...kb->...ka', R, np.asarray(world, np.float64))


def palm_frame(landmarks):
    """(..., 3, 3) 手のひらの座標系（列が 前・横・法線）。前 = 手首 → 中指の付け根、横 = 小指 → 人差し指の付け根。"""
    lm = np.asarray(landmarks, np.float64)
    return frame_from(lm[..., LM_MIDDLE, :] - lm[..., LM_WRIST, :],
                      lm[..., LM_INDEX, :] - lm[..., LM_PINKY, :])


def frame_from(forward, lateral):
    """前と横（前に直交化する）から、列が 前・横・前×横 の回転行列。"""
    f = np.asarray(forward, np.float64)
    f = f / np.maximum(np.linalg.norm(f, axis=-1, keepdims=True), 1e-12)
    a = np.asarray(lateral, np.float64)
    a = a - np.sum(a * f, axis=-1, keepdims=True) * f
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    return np.stack([f, a, np.cross(f, a)], axis=-1)


def rest_palm_frames(rest_joints):
    """(2, 3, 3) SMPL の初期姿勢（T ポーズ）の左右の手のひらの座標系。前 = 手首 → 手の関節、横 = +Z。"""
    J = np.asarray(rest_joints, np.float64)
    return np.stack([frame_from(J[HAND[s]] - J[WRIST[s]], [0.0, 0.0, 1.0]) for s in range(2)])


def swing_twist(q, axis):
    """q (..., 4) を axis まわりのひねりと、それ以外（曲げ）に分けたときの (曲げの角, ひねりの角) [rad]。"""
    q = quat.normalize(q)
    axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    proj = np.sum(q[..., :3] * axis, axis=-1)
    twist = 2.0 * np.arctan2(proj, q[..., 3])
    twist = (twist + np.pi) % (2.0 * np.pi) - np.pi
    tq = quat.normalize(np.concatenate([proj[..., None] * axis, q[..., 3:]], axis=-1))
    swing = quat.angle_between(quat.mul(q, quat.conj(tq)), quat.IDENTITY)
    return swing, twist


@dataclass
class WristResult:
    weight: np.ndarray          # (T, 2) MediaPipe の向きを使った重み（0〜1。ならした後）
    observed: np.ndarray        # (T, 2) bool 条件を満たして使ったフレーム（キーのフレームに直したもの）
    disagreement_deg: np.ndarray  # (T, 2) 使えるフレームでの NLF の手首の向きとの差 [度]（見えないフレームは nan）
    correction_deg: np.ndarray  # (T, 2) 手首のローカル回転に掛けた補正の角度 [度]
    info: dict = field(default_factory=dict)


def _nearest_fill(values, mask):
    """mask が False のフレームの値を、最も近い True のフレームの値で埋める（mask が全部 False なら元のまま）。"""
    if not mask.any():
        return values
    idx = np.flatnonzero(mask)
    t = np.arange(len(mask))
    near = idx[np.clip(np.searchsorted(idx, t), 0, len(idx) - 1)]
    prev = idx[np.clip(np.searchsorted(idx, t) - 1, 0, len(idx) - 1)]
    near = np.where(np.abs(prev - t) <= np.abs(near - t), prev, near)
    return values[near]


def observed_targets(analysis, rest_joints, cfg):
    """検出結果（hands.load_analysis の dict）から、フレームごとの手首の大域回転の目標（Y 上向き、(N, 2, 3, 3)）と、
    使えるか (N, 2)・重み (N, 2)・却下の理由ごとの数 dict。"""
    world = np.asarray(analysis['world'], np.float64)
    screen = np.asarray(analysis['screen'], np.float64)
    presence = np.asarray(analysis['presence'], np.float64)
    roi = np.asarray(analysis['roi'], np.float64)
    image_size = tuple(np.asarray(analysis['image_size']).tolist())
    finite = np.isfinite(world).all(axis=(-1, -2)) & np.isfinite(screen).all(axis=(-1, -2)) \
        & np.isfinite(roi).all(-1)
    world, screen, roi = np.nan_to_num(world), np.nan_to_num(screen), np.nan_to_num(roi)
    ok_presence = finite & (presence >= float(cfg.min_presence))

    # 画面内の向きの確認: 回し戻した 3D の点と、画像上の点の「手首 → 中指の付け根」
    unrot = np.einsum('...ab,...kb->...ka', roi_rotation(roi[..., 3]), world)
    fwd3 = unrot[..., LM_MIDDLE, :] - unrot[..., LM_WRIST, :]
    fwd2 = screen[..., LM_MIDDLE, :2] - screen[..., LM_WRIST, :2]
    n3 = np.linalg.norm(fwd3[..., :2], axis=-1)
    n2 = np.linalg.norm(fwd2, axis=-1)
    measurable = (n3 > 0.3 * np.linalg.norm(fwd3, axis=-1)) & (n2 > 1.0)
    cos = np.sum(fwd3[..., :2] * fwd2, axis=-1) / np.maximum(n3 * n2, 1e-12)
    ok_2d = ~measurable | (cos >= np.cos(np.deg2rad(float(cfg.check_2d_deg))))

    cam = world_to_camera(world, roi, image_size, float(cfg.fov_deg))
    F = palm_frame(cam)                                                    # (N, 2, 3, 3)
    target = np.einsum('ab,nsbc,scd->nsad', CAMERA_TO_YUP, F,
                       np.swapaxes(rest_palm_frames(rest_joints), -1, -2))
    usable = ok_presence & ok_2d
    lo, hi = float(cfg.min_presence), float(cfg.full_presence)
    weight = np.where(usable, np.clip((presence - lo) / max(hi - lo, 1e-6), 0.0, 1.0), 0.0)
    rejected = dict(low_presence=int((finite & ~ok_presence).sum()),
                    inconsistent_2d=int((ok_presence & ~ok_2d).sum()))
    return target, usable, weight, rejected


def correct_wrists(quats, fk, rest_joints, analysis, fps, cfg):
    """手首のローカル回転（quats の関節 20・21）を MediaPipe の向きへ寄せる。

    quats: (T, 24, 4) 親に対する回転（Y 上向き。ステージ2の後）/ fk: quats → 大域回転 (T, 24, 3, 3) の関数
    analysis: hands.load_analysis の dict（world・screen・presence・roi・image_size・fps）
    戻り値: (直した quats, WristResult)
    """
    T = len(quats)
    fps_a = float(analysis.get('fps', fps))
    target_a, usable_a, weight_a, rejected = observed_targets(analysis, rest_joints, cfg)
    N = len(target_a)
    idx = np.clip(np.round(np.arange(T) * fps_a / fps).astype(int), 0, max(N - 1, 0))
    target, usable, weight = target_a[idx], usable_a[idx], weight_a[idx]
    if N == 0:
        usable = np.zeros((T, 2), bool)
        weight = np.zeros((T, 2))

    G = fk(quats)
    out = np.array(quats, np.float64, copy=True)
    disagreement = np.full((T, 2), np.nan)
    correction = np.zeros((T, 2))
    final_w = np.zeros((T, 2))
    counts = dict(rejected, anatomy=0, disagree=0)
    J = np.asarray(rest_joints, np.float64)
    for s in range(2):
        g_wrist, g_elbow = G[:, WRIST[s]], G[:, ELBOW[s]]
        cur = quat.make_continuous(quats[:, WRIST[s]])
        local_t = quat.from_matrix(np.swapaxes(g_elbow, -1, -2) @ target[:, s])
        diff = np.rad2deg(quat.angle_between(quat.from_matrix(g_wrist), quat.from_matrix(target[:, s])))
        swing, twist = swing_twist(local_t, J[HAND[s]] - J[WRIST[s]])
        ok_anat = (np.rad2deg(swing) <= float(cfg.max_swing_deg)) \
            & (np.abs(np.rad2deg(twist)) <= float(cfg.max_twist_deg))
        ok_diff = diff <= float(cfg.max_disagree_deg)
        use = usable[:, s] & ok_anat & ok_diff
        counts['anatomy'] += int((usable[:, s] & ~ok_anat).sum())
        counts['disagree'] += int((usable[:, s] & ok_anat & ~ok_diff).sum())
        disagreement[usable[:, s], s] = diff[usable[:, s]]
        if not use.any():
            continue
        w = np.where(use, weight[:, s], 0.0)
        # カーネルは 2σ で打ち切る（見える区間から blend_sec の 2 倍より離れたフレームは NLF の向きのまま）
        sigma = float(cfg.blend_sec) * fps
        w = np.clip(filters.gaussian_time(w, sigma, radius=2.0 * sigma), 0.0, 1.0) \
            * float(cfg.strength)
        local_t = _nearest_fill(local_t, use)
        q = quat.make_continuous(quat.slerp(cur, local_t, w))
        sm = cfg.smooth
        q = quat.normalize(filters.one_euro(q, fps, float(sm.min_cutoff), float(sm.beta),
                                            float(sm.d_cutoff), True, vector_axis=-1))
        # 補正の届かないフレーム（重みが 0 の所）は、NLF の回転をそのまま残す
        q = np.where((w > 1e-4)[:, None], q, cur)
        out[:, WRIST[s]] = quat.make_continuous(q)
        final_w[:, s] = w
        correction[:, s] = np.rad2deg(quat.angle_between(out[:, WRIST[s]], cur))

    observed = np.zeros((T, 2), bool)
    for s in range(2):
        observed[:, s] = usable[:, s] & np.isfinite(disagreement[:, s])
    info = dict(used_ratio=[round(float((final_w[:, s] > 0.5).mean()), 3) for s in range(2)],
                rejected=counts)
    for s, name in enumerate(('left', 'right')):
        d = disagreement[:, s][np.isfinite(disagreement[:, s])]
        info[name] = dict(
            observed_frames=int(len(d)),
            disagreement_deg_median=float(np.median(d)) if len(d) else None,
            disagreement_over_45_ratio=float((d > 45.0).mean()) if len(d) else None,
            correction_deg_max=float(correction[:, s].max(initial=0.0)))
    return out, WristResult(final_w, observed, disagreement, correction, info)
