import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Any

MODEL_PATH = "ai/models/pose_landmarker_full.task"

# Mediapipeモデル作成処理はグローバルで一度だけ
base = python.BaseOptions(model_asset_path=MODEL_PATH)
detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(base_options=base)
)

def extract_pose_landmarks(video_path: str) -> np.ndarray:
    """
    指定した動画ファイルからMediapipeで骨格ランドマーク配列を取得
    Returns: [frame数, landmark数, 3(xyz)] のnumpy配列
    missing frame等には空配列返却
    """
    cap = cv2.VideoCapture(video_path)
    all_landmarks: List[Any] = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            all_landmarks.append([[p.x, p.y, p.z] for p in lm])
    cap.release()
    if not all_landmarks:
        return np.zeros((0,0,0))
    return np.array(all_landmarks)
