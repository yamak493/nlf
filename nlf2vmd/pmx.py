"""PMX（2.0 / 2.1）の読み込み。変換に必要なボーン情報だけを取り出す。

頂点・面・テクスチャ・材質は読み飛ばす（ボーンはそれらの後ろに並んでいるため）。
"""
import struct
from dataclasses import dataclass

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
class PmxModel:
    name: str
    name_en: str
    bones: list

    def bone_index(self, name):
        for i, b in enumerate(self.bones):
            if b.name == name:
                return i
        return -1


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
    n_uv, vsize, tsize, _msize, bsize = g[1], g[2], g[3], g[4], g[5]

    def text():
        return r.take(r.i32()).decode(encoding, errors='replace')

    name, name_en = text(), text()
    text(), text()   # コメント

    # ---- 頂点 ----
    for _ in range(r.i32()):
        r.skip(4 * (3 + 3 + 2 + 4 * n_uv))
        kind = r.u8()
        if kind == 0:        # BDEF1
            r.skip(bsize)
        elif kind == 1:      # BDEF2
            r.skip(2 * bsize + 4)
        elif kind in (2, 4):  # BDEF4 / QDEF
            r.skip(4 * bsize + 16)
        elif kind == 3:      # SDEF
            r.skip(2 * bsize + 4 + 36)
        else:
            raise ValueError(f'未知のウェイト変形方式です: {kind}')
        r.skip(4)            # エッジ倍率
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
    return PmxModel(name, name_en, bones)
