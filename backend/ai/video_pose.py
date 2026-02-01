import os
import logging
import numpy as np
import cv2

from ultralytics import YOLO

from .video_pose_analyzer import extract_pose_landmarks
from .normalize_pose import normalize_pose
from .coach_ai_utils import safe_mean
from .angle_utils import (
    calculate_elbow_angle,
)

logging.basicConfig(level=logging.INFO)

# ================================
# YOLO（最軽量）ロード
# ================================
yolo_model = YOLO("yolov8n.pt")  # Renderでも動く最軽量


# ================================
# focus辞書（トップ1だけ）
# ================================

FOCUS_LABELS = {
    "elbow_angle": "肘の角度",
    "impact_height": "打点の高さ",
    "shoulder_angle": "肩の開き",
    "waist_rotation": "腰の回転",
    "body_sway": "体軸のブレ",
    "impact_forward": "打点の前後位置",
    "toss_sync": "トスのタイミング",
    "weight_left_right": "左右の体重バランス",
}

FOCUS_MESSAGES = {
    "elbow_angle": "肘が曲がりすぎています。インパクトで伸ばしましょう。",
    "impact_height": "打点が低いです。もっと高い位置で当てましょう。",
    "shoulder_angle": "体が開きすぎています。横向きを意識しましょう。",
    "waist_rotation": "腰の回転が足りません。下半身から回しましょう。",
    "body_sway": "体の軸がブレています。頭を安定させましょう。",
    "impact_forward": "打点が後ろです。少し前で捉えましょう。",
    "toss_sync": "トスとスイングのタイミングがずれています。",
    "weight_left_right": "体重バランスが崩れています。軸足を意識しましょう。",
}

FOCUS_PRIORITY = list(FOCUS_LABELS.keys())

FOCUS_MARK_LANDMARK = {
    "elbow_angle": 14,
    "impact_height": 16,
    "impact_forward": 16,
    "impact_left_right": 16,
    "toss_sync": 16,
    "shoulder_angle": 12,
    "waist_rotation": 24,
    "body_sway": 24,
    "weight_left_right": 26,
}


# ================================
# Utility
# ================================

def pick_focus(weakness_dict):
    for key in FOCUS_PRIORITY:
        if weakness_dict.get(key) != "ok":
            return key
    return "elbow_angle"


def to_pixel(point, width, height):
    return int(point[0] * width), int(point[1] * height)


def save_frame(video_path, frame_index, output_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    cv2.imwrite(output_path, frame)
    return output_path


# ================================
# 骨格版インパクト推定（候補）
# ================================

def detect_impact_frame_pose(landmarks_3d):
    if len(landmarks_3d) < 10:
        return int(len(landmarks_3d) * 0.7)

    WRIST_ID = 16

    speeds = []
    for i in range(1, len(landmarks_3d)):
        prev = landmarks_3d[i - 1][WRIST_ID]
        curr = landmarks_3d[i][WRIST_ID]
        speeds.append(np.linalg.norm(curr - prev))

    peak_idx = int(np.argmax(speeds))

    # 前後で最も高い打点
    window = 5
    start = max(0, peak_idx - window)
    end = min(len(landmarks_3d), peak_idx + window)

    best_idx = peak_idx
    best_y = 9999

    for i in range(start, end):
        y = landmarks_3d[i][WRIST_ID][1]
        if y < best_y:
            best_y = y
            best_idx = i

    return best_idx


# ================================
# ラケット検出（YOLO）
# ================================

def detect_racket_center(frame):
    """
    YOLOでラケット検出（sports racket=43）
    """
    results = yolo_model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 43:
                x1, y1, x2, y2 = box.xyxy[0]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                return cx, cy

    return None


# ================================
# Render最適：数フレームだけラケット検出
# ================================

def refine_impact_with_racket(video_path, impact_idx):
    """
    骨格推定impact_idxの前後±3フレームだけYOLOを使う
    """
    cap = cv2.VideoCapture(video_path)

    best_idx = impact_idx
    best_speed = 0
    prev_pos = None

    for offset in range(-3, 4):
        idx = impact_idx + offset
        if idx < 0:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        pos = detect_racket_center(frame)

        if pos and prev_pos:
            speed = np.linalg.norm(np.array(pos) - np.array(prev_pos))
            if speed > best_speed:
                best_speed = speed
                best_idx = idx

        prev_pos = pos

    cap.release()
    return best_idx


# ================================
# メイン解析
# ================================

def analyze_video(file_path):
    BASE_DIR = os.path.dirname(__file__)
    success_path = os.path.join(BASE_DIR, "success.mp4")

    # landmark取得
    success_landmarks_3d = extract_pose_landmarks(success_path)
    target_landmarks_3d = extract_pose_landmarks(file_path)

    success_seq = normalize_pose(success_landmarks_3d)
    target_seq = normalize_pose(target_landmarks_3d)

    if len(target_seq) == 0 or len(success_seq) == 0:
        return {
            "diagnosis": None,
            "menu": ["基本フォーム練習"],
            "ai_text": "動画を解析できませんでした。",
        }

    # ================================
    # スコア計算
    # ================================
    frame_scores = []
    frame_diffs = []

    for t in target_seq:
        dists = np.linalg.norm(success_seq - t, axis=(1, 2))
        min_idx = np.argmin(dists)
        frame_scores.append(dists[min_idx])
        frame_diffs.append(success_seq[min_idx] - t)

    frame_diffs = np.array(frame_diffs)

    mean_dist = np.mean(frame_scores)
    score = int(max(0, min(100, 100 - mean_dist * 28)))

    # ================================
    # 指標（残す）
    # ================================
    impact_height = safe_mean(frame_diffs, 0, 1)

    success_elbow = calculate_elbow_angle(success_seq, True)
    target_elbow = calculate_elbow_angle(target_seq, True)
    elbow_angle_diff = np.mean(target_elbow) - np.mean(success_elbow)

    weakness = {
        "elbow_angle": "too_bent" if elbow_angle_diff < -20 else "ok",
        "impact_height": "low" if impact_height < -0.15 else "ok",
    }

    diagnosis = {
        "player": {
            "age": 13,
            "hand": "right",
            "serve_score": score,
        },
        "weakness": weakness,
        "raw_values": {
            "elbow_angle_diff": round(float(elbow_angle_diff), 2),
            "impact_height_diff": round(float(impact_height), 3),
        }
    }

    focus = pick_focus(weakness)

    # ================================
    # インパクト推定（骨格→ラケット補正）
    # ================================
    user_idx = detect_impact_frame_pose(target_landmarks_3d)
    user_idx = refine_impact_with_racket(file_path, user_idx)

    ideal_idx = detect_impact_frame_pose(success_landmarks_3d)

    # ================================
    # 図解生成
    # ================================
    output_dir = os.path.join(BASE_DIR, "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    ideal_img_path = os.path.join(output_dir, "ideal.png")
    user_img_path = os.path.join(output_dir, "user.png")

    landmark_id = FOCUS_MARK_LANDMARK.get(focus, 16)

    user_point = target_landmarks_3d[user_idx][landmark_id]
    ideal_point = success_landmarks_3d[ideal_idx][landmark_id]

    cap = cv2.VideoCapture(file_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    ux, uy = to_pixel(user_point, width, height)
    ix, iy = to_pixel(ideal_point, width, height)

    # 理想
    save_frame(success_path, ideal_idx, ideal_img_path)
    ideal_frame = cv2.imread(ideal_img_path)
    cv2.circle(ideal_frame, (ix, iy), 18, (0, 255, 0), -1)
    cv2.imwrite(ideal_img_path, ideal_frame)

    # あなた
    save_frame(file_path, user_idx, user_img_path)
    user_frame = cv2.imread(user_img_path)

    cv2.circle(user_frame, (ux, uy), 18, (0, 0, 255), -1)
    cv2.circle(user_frame, (ix, iy), 18, (0, 255, 0), -1)
    cv2.arrowedLine(user_frame, (ux, uy), (ix, iy), (255, 255, 255), 4)

    cv2.imwrite(user_img_path, user_frame)

    # ================================
    # MVP文章
    # ================================
    ai_text = f"改善ポイントは「{FOCUS_LABELS[focus]}」です。まずは1つだけ意識しましょう！"

    return {
        "diagnosis": diagnosis,
        "menu": ["肘を高く固定する素振り練習"],
        "ai_text": ai_text,

        "ideal_image": "/outputs/ideal.png",
        "user_image": "/outputs/user.png",

        "focus_label": FOCUS_LABELS.get(focus, focus),
        "message": FOCUS_MESSAGES.get(focus),
    }
