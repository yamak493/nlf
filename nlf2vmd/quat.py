"""クォータニオンのユーティリティ。

成分の順番は scipy と同じ (x, y, z, w)。どの関数も先頭の次元は自由で、最後の次元が 4。
"""
import numpy as np
from scipy.spatial.transform import Rotation

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def normalize(q):
    q = np.asarray(q, np.float64)
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def mul(a, b):
    """ハミルトン積 a * b（b を先に、a を後に適用する回転）。"""
    ax, ay, az, aw = np.moveaxis(np.asarray(a, np.float64), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b, np.float64), -1, 0)
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def conj(q):
    return np.asarray(q, np.float64) * np.array([-1.0, -1.0, -1.0, 1.0])


def rotate(q, v):
    """ベクトル v を q で回転する。"""
    q = normalize(q)
    u, w = q[..., :3], q[..., 3:]
    v = np.asarray(v, np.float64)
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def to_matrix(q):
    q = np.asarray(q, np.float64)
    m = Rotation.from_quat(q.reshape(-1, 4)).as_matrix()
    return m.reshape(q.shape[:-1] + (3, 3))


def from_matrix(m):
    m = np.asarray(m, np.float64)
    q = Rotation.from_matrix(m.reshape(-1, 3, 3)).as_quat()
    return q.reshape(m.shape[:-2] + (4,))


def from_rotvec(r):
    r = np.asarray(r, np.float64)
    return Rotation.from_rotvec(r.reshape(-1, 3)).as_quat().reshape(r.shape[:-1] + (4,))


def to_rotvec(q):
    q = np.asarray(q, np.float64)
    return Rotation.from_quat(q.reshape(-1, 4)).as_rotvec().reshape(q.shape[:-1] + (3,))


def make_continuous(q):
    """時間方向 (axis 0) に、前フレームとの内積が負なら符号を反転して連続にする。

    q と -q は同じ回転なので、符号を揃えないまま成分を平滑化・補間すると途中で回転が跳ねる。
    """
    q = np.array(q, np.float64, copy=True)
    if len(q) < 2:
        return q
    dots = np.sum(q[1:] * q[:-1], axis=-1)
    signs = np.cumprod(np.where(dots < 0, -1.0, 1.0), axis=0)
    q[1:] *= signs[..., None]
    return q


def angle_between(a, b):
    """2 つの回転のなす角 [rad]（符号の違いは無視）。"""
    d = np.abs(np.sum(normalize(a) * normalize(b), axis=-1))
    return 2.0 * np.arccos(np.clip(d, 0.0, 1.0))


def slerp(a, b, t):
    """球面線形補間。t は a / b の先頭の形にブロードキャストできる配列かスカラー。"""
    a, b = normalize(a), normalize(b)
    d = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(d < 0, -b, b)
    d = np.abs(d)
    t = np.asarray(t, np.float64)
    t = t.reshape(t.shape + (1,))
    theta = np.arccos(np.clip(d, 0.0, 1.0))
    s = np.sin(theta)
    small = s < 1e-6
    s_safe = np.where(small, 1.0, s)
    w0 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / s_safe)
    w1 = np.where(small, t, np.sin(t * theta) / s_safe)
    return normalize(w0 * a + w1 * b)


def average(q, weights=None):
    """回転の平均（Markley の方法: sum q q^T の最大固有ベクトル）。q は (N, 4)。"""
    q = normalize(np.asarray(q, np.float64).reshape(-1, 4))
    w = np.ones(len(q)) if weights is None else np.asarray(weights, np.float64).reshape(-1)
    m = (q * w[:, None]).T @ q
    _, vecs = np.linalg.eigh(m)
    mean = vecs[:, -1]
    if np.dot(mean, q[0]) < 0:
        mean = -mean
    return mean


def from_two_vectors(a, b):
    """ベクトル a の向きを b の向きに合わせる最小回転。"""
    a = np.asarray(a, np.float64) / np.linalg.norm(a)
    b = np.asarray(b, np.float64) / np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d < -1.0 + 1e-9:
        # 真逆: a に垂直な任意の軸で 180 度
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        return np.append(axis / np.linalg.norm(axis), 0.0)
    return normalize(np.append(np.cross(a, b), 1.0 + d))


def frame_from_up_lateral(up, lateral):
    """「上向き」と「左右方向」の 2 ベクトルから直交座標系（列が x, y, z 軸）を作る。"""
    ey = np.asarray(up, np.float64) / np.linalg.norm(up)
    lateral = np.asarray(lateral, np.float64)
    ex = lateral - np.dot(lateral, ey) * ey
    ex = ex / np.linalg.norm(ex)
    ez = np.cross(ex, ey)
    return np.stack([ex, ey, ez], axis=1)
