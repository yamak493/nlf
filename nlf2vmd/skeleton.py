"""MMD 側の骨格（PMX から、または PMX が無いときの標準ボーン）。

位置は PMX と同じ MMD 座標（左手系・Y 上向き・正面 -Z・体の左が +X）で持つ。
変換の計算は内部座標（右手系）で行うので、internal() で z を反転した値を使う。
"""
import numpy as np

# 標準的な体格（身長 20 単位前後・A ポーズ）のボーン位置。PMX を指定しないときに使う。
# 名前: (位置, 親, 表示先の相対位置)
_STANDARD_LEFT = {
    '左肩': ((0.3, 15.5, 0.5), '上半身2', None),
    '左腕': ((1.4, 15.3, 0.6), '左肩', None),
    '左ひじ': ((3.45, 13.58, 0.7), '左腕', None),
    '左手首': ((5.29, 12.04, 0.6), '左ひじ', (0.69, -0.58, -0.1)),
    '左足': ((0.95, 10.9, 0.2), '下半身', None),
    '左ひざ': ((0.95, 6.1, -0.1), '左足', None),
    '左足首': ((0.95, 1.25, 0.35), '左ひざ', None),
    '左つま先': ((0.95, 0.0, -1.3), '左足首', None),
    '左足ＩＫ': ((0.95, 1.25, 0.35), '全ての親', None),
    '左つま先ＩＫ': ((0.95, 0.0, -1.3), '左足ＩＫ', None),
}
STANDARD_BONES = {
    '全ての親': ((0.0, 0.0, 0.0), None, None),
    'センター': ((0.0, 8.0, 0.0), '全ての親', None),
    'グルーブ': ((0.0, 8.2, 0.0), 'センター', None),
    '下半身': ((0.0, 11.6, 0.2), 'グルーブ', None),
    '上半身': ((0.0, 11.8, 0.2), 'グルーブ', None),
    '上半身2': ((0.0, 13.2, 0.3), '上半身', None),
    '首': ((0.0, 16.0, 0.4), '上半身2', None),
    '頭': ((0.0, 16.9, 0.2), '首', (0.0, 1.6, 0.0)),
}
for _name, (_pos, _parent, _tail) in _STANDARD_LEFT.items():
    STANDARD_BONES[_name] = (_pos, _parent, _tail)
    _mirror = lambda s: s.replace('左', '右', 1) if s and s.startswith('左') else s  # noqa: E731
    STANDARD_BONES[_mirror(_name)] = (
        (-_pos[0], _pos[1], _pos[2]), _mirror(_parent),
        None if _tail is None else (-_tail[0], _tail[1], _tail[2]))

MMD_TO_INTERNAL = np.array([1.0, 1.0, -1.0])
SIDES = ('左', '右')


class Skeleton:
    def __init__(self, positions, parents, tails, model_name='', source='standard'):
        self.positions = {k: np.asarray(v, np.float64) for k, v in positions.items()}
        self.parents = dict(parents)
        self.tails = {k: np.asarray(v, np.float64) for k, v in tails.items() if v is not None}
        self.model_name = model_name
        self.source = source

    @classmethod
    def standard(cls):
        pos = {k: v[0] for k, v in STANDARD_BONES.items()}
        par = {k: v[1] for k, v in STANDARD_BONES.items()}
        tails = {k: np.add(v[0], v[2]) for k, v in STANDARD_BONES.items() if v[2] is not None}
        return cls(pos, par, tails, '', 'standard')

    @classmethod
    def from_pmx(cls, model):
        pos, par, tails = {}, {}, {}
        for b in model.bones:
            if b.name in pos:
                continue   # 同名ボーンは最初のものを使う（MMD と同じ）
            pos[b.name] = b.position
            par[b.name] = model.bones[b.parent].name if 0 <= b.parent < len(model.bones) else None
            if b.flags & 0x0001:
                if 0 <= b.tail_index < len(model.bones):
                    tails[b.name] = model.bones[b.tail_index].position
            elif np.linalg.norm(b.tail_offset) > 1e-6:
                tails[b.name] = b.position + b.tail_offset
        return cls(pos, par, tails, model.name, 'pmx')

    def has(self, name):
        return name in self.positions

    def internal(self, name):
        """内部座標（右手系）での初期位置。"""
        return self.positions[name] * MMD_TO_INTERNAL

    def tail_internal(self, name):
        return self.tails[name] * MMD_TO_INTERNAL if name in self.tails else None

    def missing(self, names):
        return [n for n in names if not self.has(n)]

    def leg_length(self, side):
        """足→ひざ→足首の長さ。"""
        s = SIDES[side]
        p = [self.positions[s + n] for n in ('足', 'ひざ', '足首')]
        return float(np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[1]))

    def mean_leg_length(self):
        return 0.5 * (self.leg_length(0) + self.leg_length(1))

    def nearest_ancestor(self, name, candidates):
        """name の祖先のうち、candidates に含まれる最も近いボーン（無ければ None）。"""
        p = self.parents.get(name)
        seen = set()
        while p is not None and p not in seen:
            if p in candidates:
                return p
            seen.add(p)
            p = self.parents.get(p)
        return None


REQUIRED_BONES = [
    'センター', '下半身', '上半身', '首', '頭',
    *[s + n for s in SIDES for n in ('肩', '腕', 'ひじ', '手首', '足', 'ひざ', '足首', '足ＩＫ')]]
