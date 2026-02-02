import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Any, Dict

MODEL_PATH = "ai/models/pose_landmarker_full.task"

# ==============================
# Mediapipeモデルはグローバルで1回だけ生成
# ==============================
base = python.BaseOptions(model_asset_path=MODEL_PATH)
detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(base_options=base)
)


# ==============================
# ✅完全版：norm + pixel + frames を返す
# ==============================
def extract_pose_landmarks(video_path: str) -> Dict[str, Any]:
    """
    動画から骨格ランドマークを抽出する（完全版）

    Returns:
    {
        "norm":   [F,33,3] 正規化座標 (0-1)
        "pixel":  [F,33,2] ピクセル座標
        "frames": [F]      元画像フレーム
    }
    """

    cap = cv2.VideoCapture(video_path)

    norm_list: List[Any] = []
    pixel_list: List[Any] = []
    frame_list: List[Any] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # 保存（描画用）
        frame_list.append(frame.copy())

        # Mediapipe入力
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

        # 推論
        result = detector.detect(mp_image)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]

            # norm座標（0-1）
            norm_list.append([[p.x, p.y, p.z] for p in lm])

            # pixel座標（絶対ズレない）
            pixel_list.append([[int(p.x * w), int(p.y * h)] for p in lm])

        else:
            # 骨格取れなかったフレームはスキップ
            continue

    cap.release()

    # 何も取れなかった場合
    if len(norm_list) == 0:
        return {
            "norm": np.zeros((0, 0, 0)),
            "pixel": np.zeros((0, 0, 0)),
            "frames": [],
        }

    return {
        "norm": np.array(norm_list),
        "pixel": np.array(pixel_list),
        "frames": frame_list,
    }
