import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Any

MODEL_PATH = "ai/models/pose_landmarker_full.task"

# ==============================
# PoseLandmarker 正しい初期化
# ==============================

base = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.PoseLandmarkerOptions(
    base_options=base,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
)

detector = vision.PoseLandmarker.create_from_options(options)

# ==============================
# 骨格抽出
# ==============================

def extract_pose_landmarks(video_path: str):
    """
    Returns:
      norm: [F,33,3] 正規化座標 (0-1)
      pixel: [F,33,2] ピクセル座標
    """

    cap = cv2.VideoCapture(video_path)

    all_norm = []
    all_pixel = []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = Image(
            image_format=ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(mp_image)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]

            norm_points = []
            pixel_points = []

            for p in lm:
                # 正規化
                norm_points.append([p.x, p.y, p.z])

                # ピクセル変換
                px = int(p.x * width)
                py = int(p.y * height)
                pixel_points.append([px, py])

            all_norm.append(norm_points)
            all_pixel.append(pixel_points)

    cap.release()

    if len(all_norm) == 0:
        return {
            "norm": np.zeros((0, 33, 3)),
            "pixel": np.zeros((0, 33, 2)),
        }

    return {
        "norm": np.array(all_norm),
        "pixel": np.array(all_pixel),
    }
