import os
import logging
import numpy as np
import cv2

from .video_pose_analyzer import extract_pose_landmarks
from .normalize_pose import normalize_pose
from .coach_ai_utils import safe_mean
from .coach_generator import generate_ai_menu

from .angle_utils import calculate_elbow_angle

logging.basicConfig(level=logging.INFO)

# ================================
# MVP用：focus辞書（トップ1だけ）
# ================================

FOCUS_LABELS = {
    "elbow_angle": "肘の角度",
    "impact_height": "打点の高さ",
}

FOCUS_MESSAGES = {
    "elbow_angle": "肘が曲がりすぎています。もう少し高く固定しましょう！",
    "impact_height": "打点が低いです。もう少し高い位置で当てましょう！",
}

FOCUS_PRIORITY = [
    "elbow_angle",
    "impact_height",
]

# 赤丸をつけるMediaPipe landmark（右利き固定）
FOCUS_MARK_LANDMARK = {
    "elbow_angle": 14,     # 右肘
    "impact_height": 16,   # 右手首
}


# ================================
# Utility
# ================================

def pick_focus(weakness_dict):
    """weaknessトップ1を選ぶ"""
    for key in FOCUS_PRIORITY:
        if weakness_dict.get(key) != "ok":
            return key
    return "elbow_angle"


def to_pixel(point, width, height):
    """0-1座標を画像ピクセルに変換"""
    x = int(point[0] * width)
    y = int(point[1] * height)
    return x, y


def save_frame(video_path, frame_index, output_path):
    """動画から指定フレームを保存"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    cv2.imwrite(output_path, frame)
    return output_path


def detect_impact_frame_by_wrist_speed(landmarks_3d):
    """
    インパクトを「右手首速度が最大の瞬間」で推定する
    """
    if len(landmarks_3d) < 5:
        return int(len(landmarks_3d) * 0.7)

    WRIST_ID = 16  # 右手首固定

    speeds = []
    for i in range(1, len(landmarks_3d)):
        prev = landmarks_3d[i - 1][WRIST_ID]
        curr = landmarks_3d[i][WRIST_ID]
        speeds.append(np.linalg.norm(curr - prev))

    return int(np.argmax(speeds))


# ================================
# メイン解析
# ================================

def analyze_video(file_path):
    BASE_DIR = os.path.dirname(__file__)
    success_path = os.path.join(BASE_DIR, "success.mp4")

    # --------------------------
    # landmark取得
    # --------------------------
    success_landmarks_3d = extract_pose_landmarks(success_path)
    target_landmarks_3d = extract_pose_landmarks(file_path)

    success_seq = normalize_pose(success_landmarks_3d)
    target_seq = normalize_pose(target_landmarks_3d)

    # --------------------------
    # 解析失敗時
    # --------------------------
    if len(target_seq) == 0 or len(success_seq) == 0:
        return {
            "diagnosis": None,
            "menu": ["基本フォーム練習"],
            "ai_text": "動画をうまく解析できませんでした。",
        }

    # --------------------------
    # スコア計算（距離ベース）
    # --------------------------
    frame_scores = []

    for t in target_seq:
        dists = np.linalg.norm(success_seq - t, axis=(1, 2))
        frame_scores.append(np.min(dists))

    mean_dist = np.mean(frame_scores)
    score = int(max(0, min(100, 100 - mean_dist * 28)))

    # --------------------------
    # MVP弱点：肘＋打点だけ
    # --------------------------
    is_right_handed = True

    success_elbow = calculate_elbow_angle(success_seq, is_right_handed)
    target_elbow = calculate_elbow_angle(target_seq, is_right_handed)

    elbow_angle_diff = np.mean(target_elbow) - np.mean(success_elbow)

    # 打点高さ（簡易）
    impact_height = safe_mean(target_seq, 0, 1)

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

    # --------------------------
    # focusトップ1決定
    # --------------------------
    focus = pick_focus(weakness)

    # --------------------------
    # 練習メニュー（1つだけ）
    # --------------------------
    menu = [f"{FOCUS_LABELS[focus]}を意識した素振り（10回×3）"]

    # --------------------------
    # 図解画像生成（丸＋矢印）
    # --------------------------
    output_dir = os.path.join(BASE_DIR, "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    ideal_img_path = os.path.join(output_dir, "ideal.png")
    user_img_path = os.path.join(output_dir, "user.png")

    # インパクト推定
    user_idx = detect_impact_frame_by_wrist_speed(target_landmarks_3d)
    ideal_idx = detect_impact_frame_by_wrist_speed(success_landmarks_3d)

    landmark_id = FOCUS_MARK_LANDMARK.get(focus, 14)

    user_point = target_landmarks_3d[user_idx][landmark_id]
    ideal_point = success_landmarks_3d[ideal_idx][landmark_id]

    # 動画サイズ取得
    cap = cv2.VideoCapture(file_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    ux, uy = to_pixel(user_point, width, height)
    ix, iy = to_pixel(ideal_point, width, height)

    # 理想フォーム画像（緑丸）
    save_frame(success_path, ideal_idx, ideal_img_path)
    ideal_frame = cv2.imread(ideal_img_path)
    cv2.circle(ideal_frame, (ix, iy), 18, (0, 255, 0), -1)
    cv2.imwrite(ideal_img_path, ideal_frame)

    # あなたフォーム画像（赤丸＋矢印）
    save_frame(file_path, user_idx, user_img_path)
    user_frame = cv2.imread(user_img_path)

    cv2.circle(user_frame, (ux, uy), 18, (0, 0, 255), -1)
    cv2.circle(user_frame, (ix, iy), 18, (0, 255, 0), -1)

    cv2.arrowedLine(
        user_frame,
        (ux, uy),
        (ix, iy),
        (255, 255, 255),
        4,
        tipLength=0.3
    )

    cv2.imwrite(user_img_path, user_frame)

    # --------------------------
    # AI文章生成（短く）
    # --------------------------
    ai_text = generate_ai_menu(diagnosis)

    # --------------------------
    # 最終レスポンス
    # --------------------------
    return {
        "diagnosis": diagnosis,
        "menu": menu,
        "ai_text": ai_text,

        "ideal_image": "/outputs/ideal.png",
        "user_image": "/outputs/user.png",

        "focus_label": FOCUS_LABELS.get(focus, focus),
        "message": FOCUS_MESSAGES.get(focus, "フォームを改善しましょう！"),
    }
