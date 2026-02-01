import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Any, List, Dict

MODEL_PATH = "ai/models/pose_landmarker_full.task"

# ==============================
# MediaPipe PoseLandmarker 初期化（1回だけ）
# ==============================
base = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.PoseLandmarkerOptions(
    base_options=base,
    running_mode=vision.RunningMode.IMAGE,
    output_segmentation_masks=False,
)

detector = vision.PoseLandmarker.create_from_options(options)


# ==============================
# 骨格抽出（解析用＋描画用）
# ==============================

def extract_pose_landmarks(video_path: str) -> Dict[str, np.ndarray]:
    """
    動画から骨格ランドマークを抽出する（完全版）

    Returns:
        {
          "norm":  [frame数, 33, 3] → 正規化xyz（解析用）
          "pixel": [frame数, 33, 2] → ピクセル座標（描画用）
        }

    ✅これを使えば丸が絶対ズレません
    """

    cap = cv2.VideoCapture(video_path)

    all_norm: List[Any] = []
    all_pixel: List[Any] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = Image(
            image_format=ImageFormat.SRGB,
            data=rgb
        )

        # 骨格推定
        result = detector.detect(mp_image)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]

            # --------------------------
            # ① 正規化座標（解析用）
            # --------------------------
            norm_frame = []
            for p in lm:
                norm_frame.append([p.x, p.y, p.z])

            # --------------------------
            # ② ピクセル座標（描画用）
            # --------------------------
            pixel_frame = []
            for p in lm:
                px = int(p.x * w)
                py = int(p.y * h)
                pixel_frame.append([px, py])

            all_norm.append(norm_frame)
            all_pixel.append(pixel_frame)

    cap.release()

    # 解析失敗時
    if len(all_norm) == 0:
        return {
            "norm": np.zeros((0, 0, 0)),
            "pixel": np.zeros((0, 0, 0)),
        }

    return {
        "norm": np.array(all_norm),     # [F,33,3]
        "pixel": np.array(all_pixel),   # [F,33,2]
    }
