import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Any

MODEL_PATH = "ai/models/pose_landmarker_full.task"

# ==============================
# ✅モデルはグローバルで1回だけロード
# ==============================
base = python.BaseOptions(model_asset_path=MODEL_PATH)
detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(base_options=base)
)

# ==============================
# ✅骨格抽出（軽量版）
# framesは絶対保存しない
# ==============================
def extract_pose_landmarks(video_path: str):
    """
    動画から骨格ランドマークを抽出

    Returns:
    {
        "norm":  [F,33,3] (0〜1正規化座標)
        "pixel": [F,33,2] (ピクセル座標)
    }

    ※ framesは返さない（Renderで落ちるため）
    """

    cap = cv2.VideoCapture(video_path)

    all_norm: List[Any] = []
    all_pixel: List[Any] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # MediaPipe入力
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

        result = detector.detect(mp_image)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]

            # norm座標（0〜1）
            norm_points = [[p.x, p.y, p.z] for p in lm]

            # pixel座標（絶対ズレない）
            pixel_points = [[int(p.x * w), int(p.y * h)] for p in lm]

            all_norm.append(norm_points)
            all_pixel.append(pixel_points)

    cap.release()

    # 空なら返す
    if len(all_norm) == 0:
        return {
            "norm": np.zeros((0, 0, 0)),
            "pixel": np.zeros((0, 0, 0)),
        }

    return {
        "norm": np.array(all_norm),
        "pixel": np.array(all_pixel),
    }
