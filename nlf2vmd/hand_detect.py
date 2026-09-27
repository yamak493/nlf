"""手のランドマークの検出: MediaPipe Hands のランドマークモデルを、体の関節から作った ROI に直接使う。

pip の mediapipe の HandLandmarker は、まず手のひら検出で手を探してからランドマークを求める。ダンス動画では
手が小さく（数十 px）ぶれているので、手のひら検出で見つからないフレームが多い（試した動画では 1〜2 割しか
見つからなかった）。そこで手のひら検出は使わずに:

  1. 体の関節（NLF の手首と手先）から、MediaPipe と同じ形の ROI（手首→指の付け根が上を向く正方形）を作り、
     hand_landmarker.task に入っているランドマークモデル（hand_landmarks_detector.tflite）に直接入れる
  2. 1 回目の結果から、MediaPipe のトラッキングと同じ方法（手首と指の付け根・第 2 関節を囲む矩形を 2 倍に
     広げて少し指先側へずらす）で ROI を作り直して、もう一度推定する。存在スコアが高いほうを使う

ランドマークモデルは手の存在スコアも出すので、袖・体に隠れた手はそれで除く（hands.py の min_presence）。
手首がずれた所に見つかった手（隣のもう一方の手など）も除く。

必要なもの: ai-edge-litert（無ければ tensorflow の tf.lite）と hand_landmarker.task（MODEL_URL）。
"""
import os
import zipfile

import numpy as np
from scipy.ndimage import map_coordinates

MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/'
             'float16/latest/hand_landmarker.task')
INPUT_SIZE = 224
# NLF の関節（SMPL 24 関節）: 手首と手先（指の付け根あたり）。0 = 左手、1 = 右手
WRIST_JOINTS = (20, 21)
HAND_JOINTS = (22, 23)
# MediaPipe の HandLandmarksToRect で使う点（手首・親指の CMC〜IP・各指の MCP と PIP）
_ROI_POINTS = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]


def _interpreter(model_content, num_threads):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter
        except ImportError:
            raise ImportError('ai-edge-litert をインストールしてください（pip install ai-edge-litert）') \
                from None
    return Interpreter(model_content=model_content, num_threads=num_threads)


class HandLandmarkModel:
    """hand_landmarker.task の中のランドマークモデル。入力は 224×224 の RGB（0〜1）。"""

    def __init__(self, task_path, num_threads=None):
        with zipfile.ZipFile(task_path) as z:
            name = next(n for n in z.namelist() if n.endswith('hand_landmarks_detector.tflite'))
            content = z.read(name)
        self.interpreter = _interpreter(content, num_threads or os.cpu_count() or 1)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        # 出力: 画像上の 21 点 (63)・手の存在 (1)・右手らしさ (1)・3D の 21 点 (63)。
        # 名前（Identity, Identity_1, ...）の順がこの並び
        outs = sorted(self.interpreter.get_output_details(), key=lambda d: d['name'])
        self.output_index = [d['index'] for d in outs]

    def __call__(self, crop):
        """crop: (224, 224, 3) float32 0〜1 → (画像上の点 (21, 3) [px], 存在, 右手らしさ, 3D の点 (21, 3) [m])。

        画像上の点の x・y は crop の中の位置 [px]、z は手首からの奥行き（x と同じ尺度）。
        """
        it = self.interpreter
        it.set_tensor(self.input_index, np.asarray(crop, np.float32)[None])
        it.invoke()
        screen, presence, handed, world = (it.get_tensor(i)[0] for i in self.output_index)
        return (screen.reshape(21, 3).astype(np.float64), float(presence.reshape(-1)[0]),
                float(handed.reshape(-1)[0]), world.reshape(21, 3).astype(np.float64))


def load_model(task_path, num_threads=None):
    return HandLandmarkModel(task_path, num_threads)


# ---- ROI（中心 x, y [px]・一辺 [px]・回転 [rad]）----
def roi_from_joints(wrist, hand, side, along=0.8):
    """体の手首と手先（画像上 [px]）から ROI。手首→手先が ROI の上を向くように回す。"""
    wrist, hand = np.asarray(wrist, np.float64), np.asarray(hand, np.float64)
    d = hand - wrist
    angle = np.arctan2(d[0], -d[1])
    center = wrist + along * d
    return np.array([center[0], center[1], float(side), angle])


def roi_from_landmarks(points):
    """画像上の 21 点 [px] から、MediaPipe のトラッキングと同じ ROI を作る。

    手首→（人差し指と薬指の付け根の中点と、中指の付け根の中点）が上を向くように回し、手首・指の付け根・
    第 2 関節を囲む矩形を、上へ高さの 0.1 だけずらして、長いほうの辺の 2 倍の正方形にする。
    """
    p = np.asarray(points, np.float64)[:, :2]
    tip = ((p[5] + p[13]) / 2 + p[9]) / 2
    d = tip - p[0]
    angle = np.arctan2(d[0], -d[1])
    c, s = np.cos(angle), np.sin(angle)
    to_roi = np.array([[c, s], [-s, c]])        # 画像 → ROI の向き（ROI の上 = 手首→指）
    q = p[_ROI_POINTS] @ to_roi.T
    lo, hi = q.min(axis=0), q.max(axis=0)
    w, h = hi - lo
    center = (lo + hi) / 2 + np.array([0.0, -0.1 * h])
    center = center @ to_roi                    # ROI の向き → 画像
    return np.array([center[0], center[1], 2.0 * max(w, h), angle])


def _roi_grid(roi, size):
    cx, cy, side, angle = roi
    u = (np.arange(size) + 0.5) / size - 0.5
    x, y = np.meshgrid(u * side, u * side)
    c, s = np.cos(angle), np.sin(angle)
    return cx + c * x - s * y, cy + s * x + c * y


def crop(image, roi, size=INPUT_SIZE):
    """ROI を size×size に切り出す（双線形補間。はみ出した所は端の色）。float32 0〜1。"""
    X, Y = _roi_grid(roi, size)
    img = np.asarray(image)
    out = np.empty((size, size, 3), np.float32)
    for k in range(3):
        out[..., k] = map_coordinates(img[..., k].astype(np.float32), [Y - 0.5, X - 0.5], order=1,
                                      mode='nearest')
    return out / 255.0


def to_image(points, roi, size=INPUT_SIZE):
    """crop の中の点 (N, 3) → 画像上の点 (N, 3) [px]（z も同じ倍率で [px] にする）。"""
    cx, cy, side, angle = roi
    p = np.asarray(points, np.float64)
    k = side / size
    x, y = (p[:, 0] - size / 2) * k, (p[:, 1] - size / 2) * k
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([cx + c * x - s * y, cy + s * x + c * y, p[:, 2] * k], axis=-1)


def limit_roi(roi, base, max_shift=0.5, scale=(0.5, 2.0)):
    """作り直した ROI を、最初の ROI base から大きく外れないようにする（中心のずれは base の一辺の
    max_shift 倍まで、一辺は base の scale 倍の範囲）。手が写っていないときの推定で ROI が飛ばないため。"""
    roi = np.array(roi, np.float64)
    shift = roi[:2] - base[:2]
    limit = max_shift * base[2]
    norm = float(np.linalg.norm(shift))
    if norm > limit:
        roi[:2] = base[:2] + shift * (limit / norm)
    roi[2] = np.clip(roi[2], scale[0] * base[2], scale[1] * base[2])
    return roi


def detect(model, image, roi, passes=2):
    """1 つの手を推定する。戻り値 dict: screen (21, 3) [px]・world (21, 3) [m]・presence・handedness・roi。"""
    base = np.array(roi, np.float64)
    best = None
    for _ in range(max(1, int(passes))):
        screen, presence, handed, world = model(crop(image, roi))
        pts = to_image(screen, roi)
        if best is None or presence > best['presence']:
            best = dict(screen=pts, world=world, presence=presence, handedness=handed,
                        roi=np.array(roi))
        roi = limit_roi(roi_from_landmarks(pts), base)
    return best


# ---- NLF の関節から ROI ----
def intrinsics(image_size, fov_deg=55.0):
    """NLF が仮定しているカメラ（長辺の画角 fov_deg）。image_size = (幅, 高さ)。"""
    w, h = image_size
    f = max(w, h) / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])


def joint_rois(joints3d, image_size, roi_m=0.25, fov_deg=55.0):
    """NLF の関節（カメラ座標 [mm]、(T, 24, 3)）から、左右の手の最初の ROI (T, 2, 4)。

    一辺は、手首の奥行きで roi_m [m] の長さが画像に写る大きさ。
    """
    j = np.asarray(joints3d, np.float64)
    K = intrinsics(image_size, fov_deg)
    z = np.maximum(j[..., 2:], 1.0)
    uv = j[..., :2] / z * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    rois = np.zeros((len(j), 2, 4))
    for side in range(2):
        wrist, hand = uv[:, WRIST_JOINTS[side]], uv[:, HAND_JOINTS[side]]
        side_px = K[0, 0] * roi_m * 1000.0 / z[:, WRIST_JOINTS[side], 0]
        for t in range(len(j)):
            rois[t, side] = roi_from_joints(wrist[t], hand[t], side_px[t])
    return rois, uv[:, list(WRIST_JOINTS)]


def detect_video(frames, rois, wrists, model, passes=2, max_wrist_offset=0.5, progress=None,
                 keep=()):
    """動画の各フレームの左右の手を推定する。

    frames: 画像 (H, W, 3) を順に返すもの / rois: joint_rois の ROI (T, 2, 4) / wrists: 手首の位置 (T, 2, 2)
    max_wrist_offset: 見つかった手の手首が、体の手首から最初の ROI の一辺のこの割合より離れていたら
    別の物（もう一方の手など）とみなして存在スコアを 0 にする
    keep: 確認用に画像を残すフレーム番号
    戻り値 dict: screen (T, 2, 21, 3)・world (T, 2, 21, 3)・presence (T, 2)・handedness (T, 2)・
    roi (T, 2, 4)・frames（keep の画像の dict）
    """
    T = len(rois)
    out = dict(screen=np.full((T, 2, 21, 3), np.nan, np.float32),
               world=np.full((T, 2, 21, 3), np.nan, np.float32),
               presence=np.zeros((T, 2), np.float32), handedness=np.full((T, 2), np.nan, np.float32),
               roi=np.zeros((T, 2, 4), np.float32), frames={})
    keep = set(keep)
    n = 0
    for t, image in enumerate(frames):
        if t >= T:
            break
        image = np.asarray(image)[..., :3]
        if t in keep:
            out['frames'][t] = image.copy()
        for side in range(2):
            r = detect(model, image, rois[t, side], passes)
            offset = np.linalg.norm(r['screen'][0, :2] - wrists[t, side]) / max(rois[t, side, 2], 1.0)
            out['screen'][t, side] = r['screen']
            out['world'][t, side] = r['world']
            out['presence'][t, side] = r['presence'] if offset <= max_wrist_offset else 0.0
            out['handedness'][t, side] = r['handedness']
            out['roi'][t, side] = r['roi']
        n += 1
        if progress is not None:
            progress(1)
    if n < T:
        for k in ('screen', 'world', 'presence', 'handedness', 'roi'):
            out[k] = out[k][:n]
    return out


def download_model(path, url=MODEL_URL):
    """hand_landmarker.task を path にダウンロードする（既にあれば何もしない）。"""
    import shutil
    import urllib.request

    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    tmp = path + '.part'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(tmp, 'wb') as f:
        shutil.copyfileobj(resp, f)
    zipfile.ZipFile(tmp).testzip()
    os.replace(tmp, path)
    return path


def preview_image(image, roi, screen=None, size=INPUT_SIZE):
    """確認用: ROI の切り出し (size, size, 3) uint8 と、その中のランドマーク (21, 2)（無ければ None）。"""
    img = (crop(image, roi, size) * 255.0).clip(0, 255).astype(np.uint8)
    if screen is None or not np.isfinite(screen).all():
        return img, None
    cx, cy, side, angle = roi
    k = size / side
    c, s = np.cos(angle), np.sin(angle)
    dx, dy = screen[:, 0] - cx, screen[:, 1] - cy
    pts = np.stack([(c * dx + s * dy) * k + size / 2, (-s * dx + c * dy) * k + size / 2], axis=-1)
    return img, pts


__all__ = ['MODEL_URL', 'HandLandmarkModel', 'crop', 'detect', 'detect_video', 'download_model',
           'intrinsics', 'joint_rois', 'limit_roi', 'load_model', 'preview_image',
           'roi_from_joints',
           'roi_from_landmarks', 'to_image']
