import os
import logging
import numpy as np
import cv2

from .video_pose_analyzer import extract_pose_landmarks
from .normalize_pose import normalize_pose
from .coach_ai_utils import generate_menu, safe_mean
from .coach_generator import generate_ai_menu

from .angle_utils import (
    calculate_shoulder_angle,
    calculate_elbow_angle,
    calculate_wrist_angle,
    calculate_shoulder_tilt,
    calculate_waist_rotation,
    calculate_waist_rotation_speed,
    calculate_body_sway,
    calculate_impact_height,
    calculate_impact_forward,
    calculate_toss_sync,
    calculate_impact_left_right,
    calculate_weight_left_right
)

logging.basicConfig(level=logging.INFO)

# ================================
# MVP用：focus辞書（トップ1だけ）
# ================================

FOCUS_LABELS = {
    "elbow_angle": "肘の角度",
    "impact_height": "打点の高さ",
    "shoulder_angle": "肩の開き",
    "waist_rotation": "腰の回転",
    "body_sway": "体の軸ブレ",
    "impact_forward": "打点の前後位置",
    "toss_sync": "トスのタイミング",
    "impact_left_right": "打点の左右位置",
    "weight_left_right": "左右の体重バランス",
}

FOCUS_MESSAGES = {
    "elbow_angle": "肘が曲がりすぎています。もう少し高く固定しましょう！",
    "impact_height": "打点が低いです。もう少し高い位置で当てましょう！",
    "shoulder_angle": "体が開きすぎています。横向きを保つ意識が大切です！",
    "waist_rotation": "腰の回転が足りません。下半身から回す意識を持ちましょう！",
    "body_sway": "体の軸がブレています。頭の位置を安定させましょう！",
    "impact_forward": "打点が後ろすぎます。少し前で捉えましょう！",
    "toss_sync": "トスとスイングのタイミングがずれています！",
    "impact_left_right": "打点が左右にずれています。体の正面で当てましょう！",
    "weight_left_right": "左右の体重バランスが崩れています。軸足を意識しましょう！",
}

FOCUS_PRIORITY = [
    "elbow_angle",
    "impact_height",
    "shoulder_angle",
    "waist_rotation",
    "body_sway",
    "impact_forward",
    "toss_sync",
    "impact_left_right",
    "weight_left_right",
]

# 赤丸をつけるMediaPipe landmark（右利き）
FOCUS_MARK_LANDMARK = {
    "elbow_angle": 14,
    "impact_height": 16,
    "impact_forward": 16,
    "impact_left_right": 16,
    "shoulder_angle": 12,
    "waist_rotation": 24,
    "body_sway": 24,
    "weight_left_right": 26,
    "toss_sync": 16,
}


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

    # --------------------------
    # 解析失敗時の返却
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
    frame_diffs = []

    for t in target_seq:
        dists = np.linalg.norm(success_seq - t, axis=(1, 2))
        min_idx = np.argmin(dists)
        frame_scores.append(dists[min_idx])
        frame_diffs.append(success_seq[min_idx] - t)

    frame_diffs = np.array(frame_diffs)

    mean_dist = np.mean(frame_scores)
    score = int(max(0, min(100, 100 - mean_dist * 28)))

    # --------------------------
    # 基本弱点
    # --------------------------
    impact_height = safe_mean(frame_diffs, 0, 1)
    weight_transfer = safe_mean(frame_diffs, 1, 0)
    body_open = safe_mean(frame_diffs, 2, 0)

    # --------------------------
    # 角度系指標
    # --------------------------
    is_right_handed = True
    fps = 30.0

    success_elbow = calculate_elbow_angle(success_seq, is_right_handed)
    target_elbow = calculate_elbow_angle(target_seq, is_right_handed)

    elbow_angle_diff = np.mean(target_elbow) - np.mean(success_elbow)

    # --------------------------
    # weakness判定（MVP）
    # --------------------------
    weakness = {
        "elbow_angle": "too_bent" if elbow_angle_diff < -20 else "ok",
        "impact_height": "low" if impact_height < -0.15 else "ok",
        "shoulder_angle": "ok",
        "waist_rotation": "ok",
        "body_sway": "ok",
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
    # メニュー生成
    # --------------------------
    menu = generate_menu(np.array([impact_height, weight_transfer, body_open]))
    if not isinstance(menu, list) or len(menu) == 0:
        menu = ["基本フォーム練習"]

    # --------------------------
    # 図解画像生成（ideal/user）
    # --------------------------
    output_dir = os.path.join(BASE_DIR, "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    ideal_img_path = os.path.join(output_dir, "ideal.png")
    user_img_path = os.path.join(output_dir, "user.png")

    user_idx = int(len(target_landmarks_3d) * 0.7)
    ideal_idx = int(len(success_landmarks_3d) * 0.7)

    landmark_id = FOCUS_MARK_LANDMARK.get(focus, 14)

    user_point = target_landmarks_3d[user_idx][landmark_id]
    ideal_point = success_landmarks_3d[ideal_idx][landmark_id]

    cap = cv2.VideoCapture(file_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    ux, uy = to_pixel(user_point, width, height)
    ix, iy = to_pixel(ideal_point, width, height)

    # 理想画像（緑丸だけ）
    save_frame(success_path, ideal_idx, ideal_img_path)
    ideal_frame = cv2.imread(ideal_img_path)
    cv2.circle(ideal_frame, (ix, iy), 18, (0, 255, 0), -1)
    cv2.imwrite(ideal_img_path, ideal_frame)

    # あなた画像（赤＋緑＋矢印）
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
        tipLength=0.3,
    )

    cv2.imwrite(user_img_path, user_frame)

    # --------------------------
    # AI文章生成
    # --------------------------
    ai_text = generate_ai_menu(diagnosis)


    # ================================
    # 図解画像生成（トップ1弱点）
    # ================================

    focus = pick_focus(diagnosis["weakness"])

    BASE_DIR = os.path.dirname(__file__)
    output_dir = os.path.join(BASE_DIR, "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    ideal_img_path = os.path.join(output_dir, "ideal.png")
    user_img_path  = os.path.join(output_dir, "user.png")

    # インパクト位置（割合で揃える）
    user_idx  = int(len(target_landmarks_3d) * 0.7)
    ideal_idx = int(len(success_landmarks_3d) * 0.7)

    landmark_id = FOCUS_MARK_LANDMARK.get(focus, 14)

    user_point  = target_landmarks_3d[user_idx][landmark_id]
    ideal_point = success_landmarks_3d[ideal_idx][landmark_id]

    # 動画サイズ取得
    cap = cv2.VideoCapture(file_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    ux, uy = to_pixel(user_point, width, height)
    ix, iy = to_pixel(ideal_point, width, height)

    # 理想画像（緑丸だけ）
    save_frame(success_path, ideal_idx, ideal_img_path)
    ideal_frame = cv2.imread(ideal_img_path)
    cv2.circle(ideal_frame, (ix, iy), 18, (0,255,0), -1)
    cv2.imwrite(ideal_img_path, ideal_frame)

    # あなた画像（赤＋緑＋矢印）
    save_frame(file_path, user_idx, user_img_path)
    user_frame = cv2.imread(user_img_path)

    cv2.circle(user_frame, (ux, uy), 18, (0,0,255), -1)
    cv2.circle(user_frame, (ix, iy), 18, (0,255,0), -1)

    cv2.arrowedLine(
        user_frame,
        (ux, uy),
        (ix, iy),
        (255,255,255),
        4,
        tipLength=0.3
    )

    cv2.imwrite(user_img_path, user_frame)


    # --------------------------
    # 最終レスポンス
    # --------------------------
    return {
    "diagnosis": diagnosis,
    "menu": menu,
    "ai_text": ai_text,

    # ★追加
    "ideal_image": "/outputs/ideal.png",
    "user_image": "/outputs/user.png",
    "focus_label": FOCUS_LABELS.get(focus, focus),
    "message": FOCUS_MESSAGES.get(focus, "フォームを改善しましょう！"),
    }

