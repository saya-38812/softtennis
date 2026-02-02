import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Any

MODEL_PATH = "ai/models/pose_landmarker_full.task"

# MediaPipeモデルは1回だけ生成
base = python.BaseOptions(model_asset_path=MODEL_PATH)
detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(base_options=base)
)


def extract_pose_landmarks(video_path: str):

    cap = cv2.VideoCapture(video_path)

    all_norm = []
    all_pixel = []
    all_frames = []

    MAX_FRAMES = 60
    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count > MAX_FRAMES:
            break

        # 解像度縮小（最重要）
        frame = cv2.resize(frame, (640, 360))

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

        result = detector.detect(mp_image)

        if result.pose_landmarks:

            lm = result.pose_landmarks[0]

            norm = []
            pixel = []

            for p in lm:
                norm.append([p.x, p.y, p.z])
                pixel.append([int(p.x * w), int(p.y * h)])

            all_norm.append(norm)
            all_pixel.append(pixel)
            all_frames.append(frame.copy())

    cap.release()

    if not all_norm:
        return {
            "norm": np.zeros((0, 33, 3)),
            "pixel": np.zeros((0, 33, 2)),
            "frames": [],
        }

    return {
        "norm": np.array(all_norm),
        "pixel": np.array(all_pixel),
        "frames": all_frames,
    }
