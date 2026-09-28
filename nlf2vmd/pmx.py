"""PMX（2.0 / 2.1）の読み込み。変換に必要なボーン情報と、モーフの名前・剛体だけを取り出す。

頂点は位置と、ウェイトが最も大きいボーンだけを読む（腕・指の太さや体の当たり判定を測るのに使う）。
面・テクスチャ・材質は読み飛ばす（ボーンはそれらの後ろに並んでいるため）。
モーフ（口パクのキーを打つ先があるかの確認に使う）はボーンの後ろ、剛体（体の当たり判定に使う）は
モーフ・表示枠の後ろにある。
"""
import struct
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PmxBone:
    name: str
    name_en: str
    position: np.ndarray      # (3,) MMD 座標（左手系・Y 上向き・正面 -Z）
    parent: int               # 親ボーンの番号（-1 = なし）
    flags: int
    tail_index: int           # 表示先がボーン指定のとき、その番号（-1 = なし）
    tail_offset: np.ndarray   # 表示先が相対位置指定のとき、その値


@dataclass
class PmxMorph:
    name: str
    name_en: str
    panel: int                # 1 = 眉 / 2 = 目 / 3 = 口 / 4 = その他
    kind: int                 # 0 = グループ / 1 = 頂点 / 2 = ボーン / 3〜7 = UV / 8 = 材質 / 9 = フリップ / 10 = インパルス


@dataclass
class PmxRigid:
    name: str
    bone: int                 # 関連ボーンの番号（-1 = なし）
    group: int                # 衝突グループ（0〜15）
    mask: int                 # 非衝突グループのビット
    shape: int                # 0 = 球 / 1 = 箱 / 2 = カプセル
    size: np.ndarray          # 球: (半径, -, -) / 箱: (x, y, z の半分の長さ) / カプセル: (半径, 高さ, -)
    position: np.ndarray      # (3,) MMD 座標（初期姿勢での位置）
    rotation: np.ndarray      # (3,) ラジアン（x, y, z。回転の順は Z → X → Y）
    mode: int                 # 0 = ボーン追従 / 1 = 物理演算 / 2 = 物理演算（ボーン位置合わせ）


@dataclass
class PmxModel:
    name: str
    name_en: str
    bones: list
    morphs: list = field(default_factory=list)   # 読めなかったときは None
    vertices: np.ndarray = None       # (N, 3) 頂点の位置（MMD 座標）
    vertex_bones: np.ndarray = None   # (N,) 頂点のウェイトが最も大きいボーンの番号
    rigid_bodies: list = None         # PmxRigid のリスト（読めなかったときは None）

    def bone_index(self, name):
        for i, b in enumerate(self.bones):
            if b.name == name:
                return i
        return -1

    def morph_names(self):
        return [m.name for m in self.morphs or []]


class _Reader:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, n):
        if self.pos + n > len(self.data):
            raise ValueError('PMX ファイルが途中で終わっています')
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def skip(self, n):
        self.take(n)

    def unpack(self, fmt):
        return struct.unpack('<' + fmt, self.take(struct.calcsize('<' + fmt)))

    def u8(self):
        return self.unpack('B')[0]

    def i32(self):
        return self.unpack('i')[0]

    def f32(self):
        return self.unpack('f')[0]

    def vec(self, n):
        return np.array(self.unpack(f'{n}f'), np.float64)

    def index(self, size):
        # ボーン・材質などの番号は符号付き（-1 = なし）
        return self.unpack({1: 'b', 2: 'h', 4: 'i'}[size])[0]


def read_pmx(path):
    with open(path, 'rb') as f:
        r = _Reader(f.read())
    if r.take(4) != b'PMX ':
        raise ValueError(f'PMX ファイルではありません: {path}')
    r.f32()   # バージョン
    g = list(r.take(r.u8()))
    encoding = 'utf-16-le' if g[0] == 0 else 'utf-8'
    n_uv, vsize, tsize, msize, bsize = g[1], g[2], g[3], g[4], g[5]
    morph_size, rigid_size = (g[6], g[7]) if len(g) >= 8 else (4, 4)

    def text():
        return r.take(r.i32()).decode(encoding, errors='replace')

    name, name_en = text(), text()
    text(), text()   # コメント

    # ---- 頂点（位置と、ウェイトが最も大きいボーン） ----
    bi = {1: 'b', 2: 'h', 4: 'i'}[bsize]
    positions, owners = [], []
    for _ in range(r.i32()):
        positions.append(r.unpack('3f'))
        r.skip(4 * (3 + 2 + 4 * n_uv))
        kind = r.u8()
        if kind == 0:        # BDEF1
            owners.append(r.index(bsize))
        elif kind in (1, 3):  # BDEF2 / SDEF（ウェイトは 1 本目のボーンの分）
            b0, b1, w = r.unpack(f'2{bi}f')
            owners.append(b0 if w >= 0.5 else b1)
            if kind == 3:
                r.skip(36)
        elif kind in (2, 4):  # BDEF4 / QDEF
            v = r.unpack(f'4{bi}4f')
            w = v[4:]
            owners.append(v[w.index(max(w))])
        else:
            raise ValueError(f'未知のウェイト変形方式です: {kind}')
        r.skip(4)            # エッジ倍率
    vertices = np.array(positions, np.float64).reshape(-1, 3)
    vertex_bones = np.array(owners, np.int64)
    # ---- 面・テクスチャ ----
    r.skip(r.i32() * vsize)
    for _ in range(r.i32()):
        text()
    # ---- 材質 ----
    for _ in range(r.i32()):
        text(), text()
        r.skip(4 * (4 + 3 + 1 + 3) + 1 + 4 * (4 + 1))   # 色・描画フラグ・エッジ
        r.skip(2 * tsize + 1)                          # テクスチャ・スフィア・スフィアモード
        r.skip(tsize if r.u8() == 0 else 1)            # トゥーン
        text()                                         # メモ
        r.skip(4)                                      # 面数
    # ---- ボーン ----
    bones = []
    for _ in range(r.i32()):
        bname, bname_en = text(), text()
        pos = r.vec(3)
        parent = r.index(bsize)
        r.i32()                  # 変形階層
        flags = r.unpack('H')[0]
        tail_index, tail_offset = -1, np.zeros(3)
        if flags & 0x0001:
            tail_index = r.index(bsize)
        else:
            tail_offset = r.vec(3)
        if flags & (0x0100 | 0x0200):   # 回転付与・移動付与
            r.index(bsize)
            r.f32()
        if flags & 0x0400:              # 軸固定
            r.vec(3)
        if flags & 0x0800:              # ローカル軸
            r.vec(6)
        if flags & 0x2000:              # 外部親変形
            r.i32()
        if flags & 0x0020:              # IK
            r.index(bsize)
            r.i32()
            r.f32()
            for _ in range(r.i32()):
                r.index(bsize)
                if r.u8():
                    r.vec(6)
        bones.append(PmxBone(bname, bname_en, pos, parent, flags, tail_index, tail_offset))

    # ---- モーフ（読めなくてもボーンは使えるので、失敗したら None にする） ----
    morphs = []
    if r.pos < len(r.data):
        try:
            # オフセット 1 つの大きさ（種類ごと）
            offset_size = {0: morph_size + 4, 1: vsize + 12, 2: bsize + 28, 8: msize + 113,
                           9: morph_size + 4, 10: rigid_size + 25}
            offset_size.update({k: vsize + 16 for k in range(3, 8)})
            for _ in range(r.i32()):
                mname, mname_en = text(), text()
                panel, kind = r.u8(), r.u8()
                if kind not in offset_size:
                    raise ValueError(f'未知のモーフの種類です: {kind}')
                r.skip(r.i32() * offset_size[kind])
                morphs.append(PmxMorph(mname, mname_en, panel, kind))
        except (ValueError, struct.error):
            morphs = None

    # ---- 表示枠（読み飛ばす）・剛体 ----
    rigids = [] if morphs is not None else None
    if morphs is not None and r.pos < len(r.data):
        try:
            for _ in range(r.i32()):
                text(), text()
                r.u8()                            # 特殊枠
                for _ in range(r.i32()):
                    r.index(bsize if r.u8() == 0 else morph_size)
            for _ in range(r.i32()):
                rname = text()
                text()
                bone = r.index(bsize)
                group, mask = r.u8(), r.unpack('H')[0]
                shape = r.u8()
                size, pos, rot = r.vec(3), r.vec(3), r.vec(3)
                r.skip(20)                        # 質量・移動減衰・回転減衰・反発力・摩擦力
                mode = r.u8()
                rigids.append(PmxRigid(rname, bone, group, mask, shape, size, pos, rot, mode))
        except (ValueError, struct.error):
            rigids = None
    return PmxModel(name, name_en, bones, morphs, vertices, vertex_bones, rigids)
