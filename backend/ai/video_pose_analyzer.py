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
    """
    動画から骨格ランドマークを抽出する（完全版）

    Returns:
    {
        "norm":   [F,33,3]   正規化座標（0-1）
        "pixel":  [F,33,2]   ピクセル座標（描画用）
        "frames": [F,H,W,3]  元フレーム画像
    }
    """

    cap = cv2.VideoCapture(video_path)

    all_norm = []
    all_pixel = []
    all_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # フレーム保存（描画用）
        all_frames.append(frame.copy())

        # MediaPipe入力
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

        result = detector.detect(mp_image)

        if result.pose_landmarks:

            lm = result.pose_landmarks[0]

            # norm座標（0-1）
            norm_points = np.array([[p.x, p.y, p.z] for p in lm])

            # pixel座標（描画用）
            pixel_points = np.array([[int(p.x * w), int(p.y * h)] for p in lm])

            all_norm.append(norm_points)
            all_pixel.append(pixel_points)

        else:
            # 検出失敗フレームはスキップせず補完（超重要）
            all_norm.append(np.zeros((33, 3)))
            all_pixel.append(np.zeros((33, 2)))

    cap.release()

    if len(all_norm) == 0:
        return {
            "norm": np.zeros((0, 0, 0)),
            "pixel": np.zeros((0, 0, 0)),
            "frames": [],
        }

    return {
        "norm": np.array(all_norm),
        "pixel": np.array(all_pixel),
        "frames": all_frames,
    }
