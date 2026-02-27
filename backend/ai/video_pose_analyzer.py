import cv2
import numpy as np
import logging
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
# ✅骨格抽出（軽量版・インパクト周辺のみ）
# framesは絶対保存しない
# ==============================
def extract_pose_landmarks(video_path: str, impact_index: int = None):
    """
    動画から骨格ランドマークを抽出（インパクト周辺5フレームのみ）

    Args:
        video_path: 動画ファイルのパス
        impact_index: インパクトフレームのインデックス（Noneの場合は全フレーム処理）

    Returns:
    {
        "norm":  [F,33,3] (0〜1正規化座標)
        "pixel": [F,33,2] (ピクセル座標)
    }

    ※ framesは返さない（Renderで落ちるため）
    """

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {
            "norm": np.zeros((0, 0, 0)),
            "pixel": np.zeros((0, 0, 0)),
        }
    
    # 総フレーム数を取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 処理するフレーム範囲を決定
    if impact_index is not None:
        # インパクト周辺 ±2フレーム（最大5フレーム）
        start_frame = max(0, impact_index - 2)
        end_frame = min(total_frames - 1, impact_index + 2)
        frame_indices = list(range(start_frame, end_frame + 1))
        logging.info(f"処理フレーム数: {len(frame_indices)} (インパクト: {impact_index})")
    else:
        # 後方互換: 全フレーム処理
        frame_indices = None
        logging.info(f"全フレーム処理モード（後方互換）")

    all_norm: List[Any] = []
    all_pixel: List[Any] = []
    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # インパクト周辺のみ処理
        if frame_indices is not None and current_frame not in frame_indices:
            current_frame += 1
            continue

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
        else:
            # 骨格検出失敗時も空データを追加（フレーム数維持）
            all_norm.append([[0.0, 0.0, 0.0] for _ in range(33)])
            all_pixel.append([[0, 0] for _ in range(33)])

        current_frame += 1

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
