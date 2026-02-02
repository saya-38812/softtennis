import os
import logging
import numpy as np
import cv2

from .video_pose_analyzer import extract_pose_landmarks
from .normalize_pose import normalize_pose

from .angle_utils import (
    calculate_shoulder_angle,
    calculate_elbow_angle,
    calculate_wrist_angle,
    calculate_waist_rotation,
    calculate_body_sway,
    calculate_impact_height,
    calculate_impact_forward,
    calculate_toss_sync,
    calculate_impact_left_right,
    calculate_weight_left_right,
)

logging.basicConfig(level=logging.INFO)

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
    "impact_left_right": "打点の左右位置",
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
    "impact_left_right": "打点が左右にずれています。体の正面で当てましょう。",
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

def pick_focus(weakness):
    for k in FOCUS_PRIORITY:
        if weakness.get(k) != "ok":
            return k
    return "impact_height"


def smooth(x, w=5):
    return np.convolve(x, np.ones(w) / w, mode="same")


def save_frame(video_path, idx, out_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    cv2.imwrite(out_path, frame)
    return frame


# ================================
# 🎯最強インパクト推定
# 右手首が最高点→下降開始フレーム
# ================================

def detect_best_contact_frame(norm_landmarks):
    """
    norm_landmarks : [F,33,3]
    """

    n = len(norm_landmarks)
    if n < 15:
        return int(n * 0.7)

    WRIST = 16

    wrist_y = np.array([norm_landmarks[i][WRIST][1] for i in range(n)])
    wrist_y = smooth(wrist_y, 5)

    # ①最高点（y最小）
    peak = int(np.argmin(wrist_y))

    # ②下降開始
    search_end = min(n - 1, peak + 10)

    best = peak
    for i in range(peak + 1, search_end):
        if wrist_y[i] > wrist_y[i - 1]:
            best = i
            break

    # ③接触は1フレ後が多い
    return min(n - 1, best + 1)


# ================================
# メイン解析
# ================================

def analyze_video(file_path):
    BASE_DIR = os.path.dirname(__file__)
    success_path = os.path.join(BASE_DIR, "success.mp4")

    # ============================
    # 骨格抽出（norm + pixel）
    # ============================

    success_data = extract_pose_landmarks(success_path)
    target_data  = extract_pose_landmarks(file_path)

    success_norm = success_data["norm"]
    target_norm  = target_data["norm"]

    success_pixel = success_data["pixel"]
    target_pixel  = target_data["pixel"]

    # 正規化（解析用）
    success_seq = normalize_pose(success_norm)
    target_seq  = normalize_pose(target_norm)

    if len(success_seq) == 0 or len(target_seq) == 0:
        return {
            "menu": ["基本フォーム練習"],
            "ai_text": "解析できませんでした",
        }

    # ============================
    # スコア計算
    # ============================

    dists = []
    for t in target_seq:
        d = np.linalg.norm(success_seq - t, axis=(1, 2))
        dists.append(np.min(d))

    score = int(max(0, min(100, 100 - np.mean(dists) * 28)))

    # ============================
    # 全指標計算（全部残す）
    # ============================

    is_right = True

    shoulder_diff = np.mean(calculate_shoulder_angle(target_seq, is_right)) - np.mean(
        calculate_shoulder_angle(success_seq, is_right)
    )

    elbow_diff = np.mean(calculate_elbow_angle(target_seq, is_right)) - np.mean(
        calculate_elbow_angle(success_seq, is_right)
    )

    wrist_diff = np.mean(calculate_wrist_angle(target_seq, is_right)) - np.mean(
        calculate_wrist_angle(success_seq, is_right)
    )

    waist_rot_diff = np.mean(calculate_waist_rotation(target_seq, is_right)) - np.mean(
        calculate_waist_rotation(success_seq, is_right)
    )

    sway_diff = np.mean(calculate_body_sway(target_seq)) - np.mean(
        calculate_body_sway(success_seq)
    )

    impact_h_diff = np.mean(calculate_impact_height(target_seq, is_right)) - np.mean(
        calculate_impact_height(success_seq, is_right)
    )

    impact_f_diff = np.mean(calculate_impact_forward(target_seq, is_right)) - np.mean(
        calculate_impact_forward(success_seq, is_right)
    )

    toss_diff = np.mean(calculate_toss_sync(target_seq, is_right)) - np.mean(
        calculate_toss_sync(success_seq, is_right)
    )

    lr_diff = np.mean(calculate_impact_left_right(target_seq, is_right)) - np.mean(
        calculate_impact_left_right(success_seq, is_right)
    )

    weight_lr_diff = np.mean(calculate_weight_left_right(target_seq)) - np.mean(
        calculate_weight_left_right(success_seq)
    )

    # ============================
    # weakness判定
    # ============================

    weakness = {
        "elbow_angle": "too_bent" if elbow_diff < -20 else "ok",
        "impact_height": "low" if impact_h_diff < -0.15 else "ok",
        "shoulder_angle": "too_open" if shoulder_diff > 15 else "ok",
        "waist_rotation": "insufficient" if waist_rot_diff < -20 else "ok",
        "body_sway": "unstable" if sway_diff > 0.03 else "ok",
        "impact_forward": "too_back" if impact_f_diff < -0.1 else "ok",
        "toss_sync": "out_of_sync" if abs(toss_diff) > 0.2 else "ok",
        "impact_left_right": "unstable" if abs(lr_diff) > 0.05 else "ok",
        "weight_left_right": "unbalanced" if abs(weight_lr_diff) > 0.03 else "ok",
    }

    focus = pick_focus(weakness)

    # ============================
    # メニューは短く1個だけ
    # ============================

    menu = [f"{FOCUS_LABELS[focus]}を改善する素振り練習"]

    # ============================
    # インパクト画像生成（ズレない）
    # ============================

    out_dir = os.path.join(BASE_DIR, "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    user_idx  = detect_best_contact_frame(target_norm)
    ideal_idx = detect_best_contact_frame(success_norm)

    lid = FOCUS_MARK_LANDMARK[focus]

    # pixel座標で描画する（絶対ズレない）
    ux, uy = target_pixel[user_idx][lid]
    ix, iy = success_pixel[ideal_idx][lid]

    # ideal画像
    ideal_img = save_frame(success_path, ideal_idx, os.path.join(out_dir, "ideal.png"))
    if ideal_img is not None:
        cv2.circle(ideal_img, (ix, iy), 18, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(out_dir, "ideal.png"), ideal_img)

    # user画像
    user_img = save_frame(file_path, user_idx, os.path.join(out_dir, "user.png"))
    if user_img is not None:
        cv2.circle(user_img, (ux, uy), 18, (0, 0, 255), -1)
        cv2.circle(user_img, (ix, iy), 18, (0, 255, 0), -1)
        cv2.arrowedLine(user_img, (ux, uy), (ix, iy), (255, 255, 255), 4)
        cv2.imwrite(os.path.join(out_dir, "user.png"), user_img)

    # ============================
    # AI文章（短く）
    # ============================

    ai_text = f"改善ポイントは「{FOCUS_LABELS[focus]}」です。まず1つだけ意識しましょう！"

    return {
        "diagnosis": {
            "player": {"age": 13, "hand": "right", "serve_score": score},
            "weakness": weakness,
            "raw_values": {
                "elbow_angle_diff": float(elbow_diff),
                "impact_height_diff": float(impact_h_diff),
                "shoulder_angle_diff": float(shoulder_diff),
                "waist_rotation_diff": float(waist_rot_diff),
                "body_sway_diff": float(sway_diff),
                "impact_forward_diff": float(impact_f_diff),
                "toss_sync_diff": float(toss_diff),
                "impact_left_right_diff": float(lr_diff),
                "weight_left_right_diff": float(weight_lr_diff),
            },
        },
        "menu": menu,
        "ai_text": ai_text,
        "ideal_image": "/outputs/ideal.png",
        "user_image": "/outputs/user.png",
        "focus_label": FOCUS_LABELS[focus],
        "message": FOCUS_MESSAGES[focus],
    }
