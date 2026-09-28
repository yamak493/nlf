"""捩りボーン（腕捩・手捩）へのひねりの振り分け（ステージ10 で VMD のキーを作る直前）。

ステージ9 のリターゲットは、腕の回転を 腕 / ひじ / 手首 の 3 本に入れる。MMD のモデルの多くは、上腕・前腕の途中に
腕捩・手捩 があり、その子の 腕捩1〜3・手捩1〜3（回転付与）が、捩りボーンのひねりを上腕・前腕に沿って少しずつ
（例: 25・50・75%）分けて、袖・前腕のメッシュをねじる。腕・ひじ・手首 にひねりを入れると、関節の所でメッシュが
まとめてねじれてつぶれ、体や服を貫通しているように見える。

そこで、ボーンの軸まわりの回転（ひねり）を swing-twist 分解で取り出し、捩りボーンへ移す。
  * 腕 のローカル回転 = 曲げ · ひねり（上腕の軸まわり）→ 腕 に曲げ、腕捩 にひねり
  * 前腕のひねり = ひじ のひねり + 手首 のひねり（どちらも前腕の軸まわり。手首の関節そのものはひねれないので、
    手の前腕に対するひねりは前腕の回内・回外）→ 手捩 に入れ、ひじ・手首 には曲げだけを残す
大域回転は変えない（ひじ・手首から先の向きと位置はそのまま）。軸は捩りボーンの軸制限（無ければ 捩りボーン → 子 の
向き）。捩りボーンの回転はその位置を中心に掛かるので、軸が 捩りボーン → 子 の線からずれていると子の位置が動く。
軸がこの線、またはこの線が 親 → 子 の線と MAX_AXIS_DEG より大きくずれるモデルでは、その捩りボーンを使わない。

ひねりの角は時間方向に連続につなぐ（手のひらを上に向ける動きは前腕の軸まわりに約 180 度なので、±180 度で
折り返すと捩りボーンが逆向きにねじれる）。捩りボーンに入れるのは ±max_deg まで（回転付与は 180 度を超える回転を
逆向きに分けるため）で、残りは元のボーンに残す（手捩 の分は 手首 に残す）。曲げが 180 度に近いと
（腕を初期姿勢の向きの反対へ向けた等）ひねりの向きが決まらないので、fade_swing_deg の範囲で移す量を 0 へ減らす。

捩りボーンが 親 → 捩り → 子 の順につながっていないモデル（子が捩りボーンの下に無い）は、その捩りボーンを使わない。
"""
from dataclasses import dataclass, field

import numpy as np

from . import filters, quat
from .skeleton import SIDES

# (捩りボーン, ひねりを取り出すボーン, 捩りボーンの子側のボーン, 子側のひねりも集めるか)
TWIST_BONES = (('腕捩', '腕', 'ひじ', False), ('手捩', 'ひじ', '手首', True))
MAX_AXIS_DEG = 5.0   # 軸・捩りボーンの位置のずれの許容量 [度]（5 度なら 90 度ひねっても子の位置のずれは骨の長さの 6% 未満）


@dataclass
class TwistChain:
    twist: str             # 捩りボーン（例: 左腕捩）
    parent: str            # ひねりを取り出すボーン（左腕 / 左ひじ）
    child: str             # 捩りボーンの子側のボーン（左ひじ / 左手首）
    axis: np.ndarray       # ひねりの軸（内部座標の単位ベクトル）
    collect_child: bool    # 子側のボーンのひねりも捩りボーンに集める（手捩）


@dataclass
class TwistResult:
    local: dict            # 捩りボーンを足したローカル回転（内部座標）
    chains: list           # 使った TwistChain
    skipped: list          # 使わなかった捩りボーンと理由
    angle_deg: dict = field(default_factory=dict)   # 捩りボーン → (T,) 入れたひねりの角 [度]
    enabled: bool = True

    @property
    def info(self):
        return dict(enabled=self.enabled, bones=[c.twist for c in self.chains],
                    skipped=self.skipped,
                    max_deg={k: round(float(np.abs(v).max(initial=0.0)), 2)
                             for k, v in self.angle_deg.items()})


def twist_chains(skel):
    """使える捩りボーン [TwistChain] と、使えない捩りボーンの理由 [str]。"""
    chains, skipped = [], []
    for s in SIDES:
        for twist, parent, child, collect in TWIST_BONES:
            t, p, c = s + twist, s + parent, s + child
            if not skel.has(t):
                continue
            if not (skel.has(p) and skel.has(c) and t != p and skel.is_descendant(t, p)
                    and skel.is_descendant(c, t)):
                skipped.append(f'{t} が {p} → {c} の間にないので使いません')
                continue
            bone = skel.internal(c) - skel.internal(t)     # この向きの軸なら、ひねっても子の位置が変わらない
            axis = skel.axis_internal(t)
            if axis is None or np.linalg.norm(axis) < 1e-9:
                axis = bone
            limb = skel.internal(c) - skel.internal(p)
            if max(_angle_deg(axis, bone), _angle_deg(bone, limb)) > MAX_AXIS_DEG:
                skipped.append(f'{t} の軸・位置が {p} → {c} の線と合わないので使いません')
                continue
            chains.append(TwistChain(t, p, c, axis / np.linalg.norm(axis), collect))
    return chains, skipped


def _angle_deg(a, b):
    """2 本の軸のなす角 [度]（向きの正負は区別しない）。"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 180.0
    return float(np.rad2deg(np.arccos(np.clip(abs(np.dot(a, b)) / (na * nb), 0.0, 1.0))))


def twist_angle(q, axis, min_cos):
    """q (T, 4) の axis まわりのひねりの角 (T,) [rad] と、曲げの半分の角の余弦 (T,)。

    ひねりの角は時間方向に連続につなぐ（±180 度で折り返さない）。曲げが 180 度に近いフレーム（余弦が min_cos 未満）は
    ひねりの向きが決まらないので、そこで区切り、区間ごとに、中央値が ±180 度に入るように 360 度の倍数をずらす。
    """
    q = quat.normalize(q)
    s, w = q[:, :3] @ axis, q[:, 3]
    cos_half = np.hypot(s, w)
    theta = 2.0 * np.arctan2(s, w)
    theta = (theta + np.pi) % (2.0 * np.pi) - np.pi
    for a, b in filters.runs(cos_half >= min_cos):
        seg = np.unwrap(theta[a:b + 1])
        theta[a:b + 1] = seg - 2.0 * np.pi * np.round(np.median(seg) / (2.0 * np.pi))
    return theta, cos_half


def _about(axis, angle):
    return quat.from_rotvec(np.asarray(angle)[:, None] * axis)


def split_twist(skel, local, cfg):
    """local（ボーン名 → ローカル回転 (T, 4)。捩りボーンを含まない）のひねりを捩りボーンへ移した TwistResult。"""
    if not cfg.enabled:
        return TwistResult(dict(local), [], [], enabled=False)
    chains, skipped = twist_chains(skel)
    out = dict(local)
    full, zero = np.deg2rad(np.asarray(cfg.fade_swing_deg, np.float64))
    cos_full, cos_zero = np.cos(0.5 * full), np.cos(0.5 * zero)
    limit = np.deg2rad(float(cfg.max_deg))

    def removable(q, axis):
        # 取り出すひねりの角（曲げが 180 度に近いほど 0 へ減らす）
        theta, cos_half = twist_angle(q, axis, cos_zero)
        return theta * filters.smoothstep((cos_half - cos_zero) / (cos_full - cos_zero))

    used = []
    for ch in chains:
        if ch.parent not in out or ch.child not in out:
            skipped.append(f'{ch.parent} か {ch.child} にキーが無いので、{ch.twist} は使いません')
            continue
        own = removable(out[ch.parent], ch.axis)
        if ch.collect_child:
            amount = np.clip(own + removable(out[ch.child], ch.axis), -limit, limit)
        else:
            own = np.clip(own, -limit, limit)
            amount = own
        # 親 · 捩り · 子 の積を変えない: 親から own を除き、捩りに amount を入れ、残り（own − amount）は子へ
        out[ch.parent] = quat.mul(out[ch.parent], quat.conj(_about(ch.axis, own)))
        out[ch.twist] = _about(ch.axis, amount)
        out[ch.child] = quat.mul(_about(ch.axis, own - amount), out[ch.child])
        used.append(ch)
    return TwistResult(out, used, skipped,
                       {ch.twist: np.rad2deg(2.0 * np.arctan2(out[ch.twist][:, :3] @ ch.axis,
                                                              out[ch.twist][:, 3]))
                        for ch in used})
