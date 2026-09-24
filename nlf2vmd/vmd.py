"""VMD（Vocaloid Motion Data 0002）の書き出しと読み込み。

ファイル構成:
  ヘッダ 30 バイト（"Vocaloid Motion Data 0002" をヌル埋め）＋ モデル名 20 バイト（cp932）
  ボーンキー数 uint32 ＋ 1 キー 111 バイト
    ボーン名 15 バイト（cp932） / フレーム番号 uint32 / 位置 float32×3 / 回転 float32×4 (x,y,z,w)
    / 補間パラメータ 64 バイト
  モーフ・カメラ・照明・セルフ影のキー数（このツールではどれも 0）

補間パラメータ 64 バイトの配置は MMD Tools（blender_mmd_tools）の実装で確認したもの:
  書き出し core/vmd/exporter.py の __getVMDBoneInterpolation
  読み込み core/vmd/importer.py（X = [0,4,8,12] / Y = [16,20,24,28] / Z = [32,..] / 回転 = [48,..]）
各 16 バイトの行は、並び
  [X_x1, Y_x1, Z_x1, R_x1, X_y1, Y_y1, Z_y1, R_y1, X_x2, Y_x2, Z_x2, R_x2, X_y2, Y_y2, Z_y2, R_y2]
を 1 行ごとに 1 つずつ左へずらしたもので、はみ出た末尾は 0。1 行目の 3・4 バイト目
（Z_x1, R_x1 の位置）は MMD Tools と同じく 0 にする（MMD が別の用途に使う領域のため）。
"""
import struct
import warnings
from dataclasses import dataclass

import numpy as np

SIGNATURE = b'Vocaloid Motion Data 0002'
BONE_KEY_DTYPE = np.dtype([
    ('name', 'S15'), ('frame', '<u4'), ('position', '<f4', 3), ('rotation', '<f4', 4),
    ('interp', 'u1', 64)])
assert BONE_KEY_DTYPE.itemsize == 111

# MMD 座標へ: 位置 (x, y, z) → (x, y, -z)、クォータニオン (x, y, z, w) → (-x, -y, z, w)
POSITION_TO_MMD = np.array([1.0, 1.0, -1.0])
QUAT_TO_MMD = np.array([-1.0, -1.0, 1.0, 1.0])


def bone_interpolation(x=((20, 20), (107, 107)), y=None, z=None, r=None):
    """補間パラメータ 64 バイト。各軸は ((x1, y1), (x2, y2))。既定は線形。"""
    y, z, r = y or x, z or x, r or x
    (xx1, xy1), (xx2, xy2) = x
    (yx1, yy1), (yx2, yy2) = y
    (zx1, zy1), (zx2, zy2) = z
    (rx1, ry1), (rx2, ry2) = r
    base = [xx1, yx1, zx1, rx1, xy1, yy1, zy1, ry1, xx2, yx2, zx2, rx2, xy2, yy2, zy2, ry2]
    rows = [base[k:] + [0] * k for k in range(4)]
    rows[0][2] = rows[0][3] = 0
    return np.array(sum(rows, []), np.uint8)


LINEAR_INTERPOLATION = bone_interpolation()


def read_interpolation(interp):
    """64 バイトから (X, Y, Z, 回転) それぞれの ((x1, y1), (x2, y2)) を読む（MMD Tools と同じ位置）。"""
    b = np.asarray(interp, np.uint8)
    out = []
    for start in (0, 16, 32, 48):
        x1, y1, x2, y2 = (int(v) for v in b[start:start + 16:4])
        out.append(((x1, y1), (x2, y2)))
    return tuple(out)


def encode_name(name, nbytes, what='ボーン名'):
    raw = name.encode('cp932')
    if len(raw) > nbytes:
        warnings.warn(f'{what}「{name}」は cp932 で {len(raw)} バイトあり、{nbytes} バイトを超えるため'
                      '切り詰めます（MMD で別のボーン扱いになる可能性があります）', stacklevel=2)
        raw = raw[:nbytes]
    return raw


def decode_name(raw):
    return bytes(raw).split(b'\0', 1)[0].decode('cp932', errors='replace')


@dataclass
class BoneTrack:
    """1 ボーン分のキー列（MMD 座標の値）。"""
    name: str
    frames: np.ndarray       # (N,) int
    positions: np.ndarray    # (N, 3)
    rotations: np.ndarray    # (N, 4) x, y, z, w


def to_mmd_position(p):
    return np.asarray(p, np.float64) * POSITION_TO_MMD


def to_mmd_quat(q):
    return np.asarray(q, np.float64) * QUAT_TO_MMD


def write_vmd(path, tracks, model_name=''):
    """tracks（BoneTrack のリスト）を VMD に書き出す。戻り値はキーの総数。"""
    n = sum(len(t.frames) for t in tracks)
    keys = np.zeros(n, BONE_KEY_DTYPE)
    i = 0
    for t in tracks:
        m = len(t.frames)
        keys['name'][i:i + m] = encode_name(t.name, 15)
        keys['frame'][i:i + m] = np.asarray(t.frames, np.int64)
        keys['position'][i:i + m] = t.positions
        keys['rotation'][i:i + m] = t.rotations
        keys['interp'][i:i + m] = LINEAR_INTERPOLATION
        i += m
    # フレーム順に並べる（MMD はどちらでも読めるが、他のツールとの互換のため）
    keys = keys[np.argsort(keys['frame'], kind='stable')]
    with open(path, 'wb') as f:
        f.write(struct.pack('<30s', SIGNATURE))
        f.write(struct.pack('<20s', encode_name(model_name, 20, 'モデル名')))
        f.write(struct.pack('<I', n))
        f.write(keys.tobytes())
        f.write(struct.pack('<4I', 0, 0, 0, 0))   # モーフ・カメラ・照明・セルフ影
    return n


@dataclass
class VmdMotion:
    model_name: str
    keys: np.ndarray          # BONE_KEY_DTYPE の配列
    counts: dict              # モーフ・カメラ・照明・セルフ影のキー数

    def bone_names(self):
        return sorted({decode_name(n) for n in self.keys['name']})

    def track(self, name):
        """ボーン名のキーをフレーム順に (frames, positions, rotations) で返す。"""
        raw = encode_name(name, 15)
        k = self.keys[self.keys['name'] == raw]
        k = k[np.argsort(k['frame'], kind='stable')]
        return (k['frame'].astype(np.int64), k['position'].astype(np.float64),
                k['rotation'].astype(np.float64))


def read_vmd(path):
    with open(path, 'rb') as f:
        data = f.read()
    if not data[:30].startswith(SIGNATURE):
        raise ValueError(f'VMD ファイルではありません: {path}')
    model_name = decode_name(data[30:50])
    (n,) = struct.unpack_from('<I', data, 50)
    pos = 54
    keys = np.frombuffer(data, BONE_KEY_DTYPE, count=n, offset=pos).copy()
    pos += n * BONE_KEY_DTYPE.itemsize
    counts = {}
    for name, size in (('morph', 23), ('camera', 61), ('light', 28), ('self_shadow', 9)):
        if pos + 4 > len(data):
            counts[name] = 0
            continue
        (m,) = struct.unpack_from('<I', data, pos)
        counts[name] = m
        pos += 4 + m * size
    return VmdMotion(model_name, keys, counts)


# ---- キーの間引き（任意） ----
def _rdp_keep(err_fn, n, forced):
    """両端と forced を必ず残し、間を線形補間した誤差が許容値を超える所だけキーを足す。"""
    keep = np.zeros(n, bool)
    keep[[0, n - 1]] = True
    keep[list(forced)] = True
    anchors = np.flatnonzero(keep)
    stack = list(zip(anchors[:-1], anchors[1:]))
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        err = err_fn(a, b)            # (b - a - 1,) 内側フレームの誤差（許容値で割った値）
        i = int(np.argmax(err))
        if err[i] > 1.0:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return keep


def thin_track(positions, rotations, pos_tol, rot_tol_deg, forced=()):
    """位置は線形・回転は球面線形で補間したときの誤差が許容内なら、キーを省く。残すフレームの bool を返す。"""
    from . import quat
    n = len(positions)
    if n <= 2:
        return np.ones(n, bool)
    rot_tol = np.deg2rad(rot_tol_deg)

    def err(a, b):
        t = (np.arange(a + 1, b) - a) / float(b - a)
        p = positions[a] + (positions[b] - positions[a]) * t[:, None]
        e_pos = np.linalg.norm(positions[a + 1:b] - p, axis=-1) / max(pos_tol, 1e-12)
        q = quat.slerp(np.broadcast_to(rotations[a], (len(t), 4)),
                       np.broadcast_to(rotations[b], (len(t), 4)), t)
        e_rot = quat.angle_between(q, rotations[a + 1:b]) / max(rot_tol, 1e-12)
        return np.maximum(e_pos, e_rot)

    return _rdp_keep(err, n, [f for f in forced if 0 <= f < n])
