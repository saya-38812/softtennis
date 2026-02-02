import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Dict, Any

MODEL_PATH = "ai/models/pose_landmarker_full.task"

# ============================
# MediaPipe初期化（1回だけ）
# ============================
base = python.BaseOptions(model_asset_path=MODEL_PATH)

detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(base_options=base)
)

# ============================
# 骨格抽出（norm + pixel + frame）
# ============================
def extract_pose_landmarks(video_path: str) -> Dict[str, Any]:

    cap = cv2.VideoCapture(video_path)

    norm_list = []
    pixel_list = []
    frame_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

        result = detector.detect(mp_image)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]

            # norm座標（0〜1）
            norm = np.array([[p.x, p.y, p.z] for p in lm])

            # pixel座標（絶対一致）
            pixel = np.array([[int(p.x * w), int(p.y * h)] for p in lm])

            norm_list.append(norm)
            pixel_list.append(pixel)
            frame_list.append(frame)

    cap.release()

    if len(norm_list) == 0:
        return {
            "norm": np.zeros((0, 0, 0)),
            "pixel": np.zeros((0, 0, 0)),
            "frames": []
        }

    return {
        "norm": np.array(norm_list),      # [F,33,3]
        "pixel": np.array(pixel_list),    # [F,33,2]
        "frames": frame_list              # frameそのもの
    }
