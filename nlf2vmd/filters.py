"""時系列フィルタ。入力はどれも (T, ...) で、時間方向は axis 0。"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, minimum_filter1d


def _alpha(cutoff, fps):
    tau = 1.0 / (2.0 * np.pi * np.asarray(cutoff, np.float64))
    return 1.0 / (1.0 + tau * fps)


def _speed(dx_hat, vector_axis):
    if vector_axis is None:
        return np.abs(dx_hat)
    return np.linalg.norm(dx_hat, axis=vector_axis, keepdims=True)


def _one_euro_pass(x, fps, min_cutoff, beta, d_cutoff, vector_axis):
    out = np.empty_like(x)
    # 先頭数フレームの傾きで一定速度に動いているとみなし、その定常状態（一定の遅れ）から始める。
    # 0 速度から始めると、動いている信号の先頭で遅れが立ち上がるまでの過渡応答が出る
    m = min(5, len(x) - 1)
    v0 = (x[m] - x[0]) / m
    lag = np.zeros_like(x[0])
    for _ in range(5):
        a = _alpha(min_cutoff + beta * _speed((v0 + lag) * fps, vector_axis), fps)
        lag = v0 * (1.0 - a) / a
    out[0] = x[0] - lag
    dx_hat = (v0 + lag) * fps
    a_d = _alpha(d_cutoff, fps)
    for t in range(1, len(x)):
        dx = (x[t] - out[t - 1]) * fps
        dx_hat = a_d * dx + (1.0 - a_d) * dx_hat
        a = _alpha(min_cutoff + beta * _speed(dx_hat, vector_axis), fps)
        out[t] = a * x[t] + (1.0 - a) * out[t - 1]
    return out


def one_euro(x, fps, min_cutoff, beta, d_cutoff=1.0, zero_phase=True, vector_axis=None):
    """One Euro フィルタ（Casiez et al. 2012）。

    遅い動きほど強く（カットオフ min_cutoff [Hz]）、速い動きほど弱く（カットオフが
    beta × 速さ だけ上がる）平滑化する。min_cutoff / beta は x[0] の形にブロードキャスト
    できる配列でもよい（関節グループ別・軸別の強さ）。

    zero_phase=True のときは、順方向と逆方向に 1 回ずつ掛けた結果を平均して遅れを打ち消す
    （オフライン処理なので未来のフレームも使える）。
    vector_axis を指定すると、その軸のノルムを速さとして使う（クォータニオンの 4 成分で
    カットオフを共有したいときなど）。
    """
    x = np.asarray(x, np.float64)
    T = len(x)
    if T < 2 or np.all(np.asarray(min_cutoff) <= 0):
        return x.copy()
    # 両端を点対称に延長してから掛ける（scipy の filtfilt と同じ考え方）。一定速度で動いている
    # 区間の端でも、フィルタの立ち上がりの遅れが延長部分で済み、端に加速度が出ない
    pad = min(T - 1, int(np.ceil(2.0 * fps)))
    xp = np.concatenate([2 * x[:1] - x[pad:0:-1], x, 2 * x[-1:] - x[-2:-pad - 2:-1]])
    fwd = _one_euro_pass(xp, fps, min_cutoff, beta, d_cutoff, vector_axis)
    if zero_phase:
        bwd = _one_euro_pass(xp[::-1], fps, min_cutoff, beta, d_cutoff, vector_axis)[::-1]
        fwd = 0.5 * (fwd + bwd)
    return fwd[pad:pad + T]


def median_time(x, window):
    """時間方向のメディアンフィルタ（端は端の値で延長）。"""
    x = np.asarray(x, np.float64)
    if window <= 1 or len(x) < 2:
        return x.copy()
    size = (int(window),) + (1,) * (x.ndim - 1)
    return median_filter(x, size=size, mode='nearest')


def moving_min(x, window):
    x = np.asarray(x, np.float64)
    if window <= 1:
        return x.copy()
    return minimum_filter1d(x, int(window), axis=0, mode='nearest')


def gaussian_time(x, sigma, radius=None):
    """時間方向のガウシアン。radius を指定すると、カーネルをその幅（フレーム）で打ち切る。"""
    x = np.asarray(x, np.float64)
    if sigma <= 0:
        return x.copy()
    truncate = 4.0 if radius is None else max(float(radius), 0.5) / float(sigma)
    return gaussian_filter1d(x, float(sigma), axis=0, mode='nearest', truncate=truncate)


def smoothstep(t):
    t = np.clip(np.asarray(t, np.float64), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def runs(mask):
    """bool 列の True の連続区間を [(開始, 終了), ...]（終了を含む）で返す。"""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))
