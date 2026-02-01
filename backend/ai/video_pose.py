import os
import logging
import numpy as np
import cv2
import uuid

from .video_pose_analyzer import extract_pose_landmarks
from .normalize_pose import normalize_pose
from .coach_ai_utils import safe_mean
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

# 赤丸をつけるMediaPipe landmark（右利き）
FOCUS_MARK_LANDMARK = {
    "elbow_angle": 14,      # 右肘
    "impact_height": 16,    # 右手首
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
        logging.warning("フレーム取得に失敗しました")
        return None

    cv2.imwrite(output_path, frame)
    return output_path


# ================================
# インパクト推定（完成版）
# ================================

def detect_impact_frame(landmarks_3d, is_right_handed=True):
    """
    インパクト推定：
    ① 手首速度最大フレーム
    ② 前後±5で最も高い打点フレームを採用
    """

    if len(landmarks_3d) < 10:
        return int(len(landmarks_3d) * 0.7)

    WRIST_ID = 16 if is_right_handed else 15

    speeds = []
    for i in range(1, len(landmarks_3d)):
        prev = landmarks_3d[i - 1][WRIST_ID]
        curr = landmarks_3d[i][WRIST_ID]
        speeds.append(np.linalg.norm(curr - prev))

    peak_idx = int(np.argmax(speeds))

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
    # インパクト推定
    # --------------------------
    user_idx = detect_impact_frame(target_landmarks_3d, True)
    ideal_idx = detect_impact_frame(success_landmarks_3d, True)

    # --------------------------
    # 打点高さ（手首Y座標で直接評価）
    # --------------------------
    WRIST_ID = 16
    user_y = target_landmarks_3d[user_idx][WRIST_ID][1]
    ideal_y = success_landmarks_3d[ideal_idx][WRIST_ID][1]

    impact_height_diff = ideal_y - user_y  # プラスなら低い

    # --------------------------
    # 肘角度差
    # --------------------------
    success_elbow = calculate_elbow_angle(success_seq, True)
    target_elbow = calculate_elbow_angle(target_seq, True)

    elbow_angle_diff = np.mean(target_elbow) - np.mean(success_elbow)

    # --------------------------
    # weakness判定（トップ2だけ）
    # --------------------------
    weakness = {
        "elbow_angle": "too_bent" if elbow_angle_diff < -20 else "ok",
        "impact_height": "low" if impact_height_diff > 0.05 else "ok",
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
            "impact_height_diff": round(float(impact_height_diff), 3),
        }
    }

    # --------------------------
    # focusトップ1決定
    # --------------------------
    focus = pick_focus(weakness)

    # --------------------------
    # メニュー生成（短く1個だけ）
    # --------------------------
    menu = ["肘を高く固定する素振り練習"]

    # --------------------------
    # 図解画像生成（キャッシュ対策付き）
    # --------------------------
    output_dir = os.path.join(BASE_DIR, "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    unique_id = str(uuid.uuid4())[:8]

    ideal_img_path = os.path.join(output_dir, f"ideal_{unique_id}.png")
    user_img_path = os.path.join(output_dir, f"user_{unique_id}.png")

    landmark_id = FOCUS_MARK_LANDMARK.get(focus, 16)

    user_point = target_landmarks_3d[user_idx][landmark_id]
    ideal_point = success_landmarks_3d[ideal_idx][landmark_id]

    cap = cv2.VideoCapture(file_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    ux, uy = to_pixel(user_point, width, height)
    ix, iy = to_pixel(ideal_point, width, height)

    # 理想画像（緑丸）
    save_frame(success_path, ideal_idx, ideal_img_path)
    ideal_frame = cv2.imread(ideal_img_path)

    if ideal_frame is not None:
        cv2.circle(ideal_frame, (ix, iy), 18, (0, 255, 0), -1)
        cv2.imwrite(ideal_img_path, ideal_frame)

    # あなた画像（赤＋緑＋矢印）
    save_frame(file_path, user_idx, user_img_path)
    user_frame = cv2.imread(user_img_path)

    if user_frame is not None:
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
    # AI文章（短く1文）
    # --------------------------
    ai_text = f"改善ポイントは「{FOCUS_LABELS[focus]}」です。まずは1つだけ意識しましょう！"

    # --------------------------
    # 最終レスポンス
    # --------------------------
    return {
        "diagnosis": diagnosis,
        "menu": menu,
        "ai_text": ai_text,

        "ideal_image": f"/outputs/ideal_{unique_id}.png",
        "user_image": f"/outputs/user_{unique_id}.png",

        "focus_label": FOCUS_LABELS.get(focus, focus),
        "message": FOCUS_MESSAGES.get(focus, "フォームを改善しましょう！"),
    }
