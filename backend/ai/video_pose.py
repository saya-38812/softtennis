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
# 🎯メイン改善（強く出す3つ）
# ================================

MAIN_FOCUS = ["impact_height", "elbow_angle", "body_sway"]

FOCUS_LABELS = {
    "impact_height": "打点の高さ",
    "elbow_angle": "肘の角度",
    "body_sway": "体軸のブレ",

    # 参考指標（軽く表示）
    "shoulder_angle": "肩の開き（参考）",
    "waist_rotation": "腰の回転（参考）",
    "impact_forward": "打点の前後（参考）",
    "toss_sync": "トスのタイミング（参考）",
    "impact_left_right": "打点の左右（参考）",
    "weight_left_right": "体重バランス（参考）",
}

FOCUS_MESSAGES = {
    "impact_height": "打点が低いです。もっと高い位置で当てましょう。",
    "elbow_angle": "肘が曲がりすぎています。インパクトで伸ばしましょう。",
    "body_sway": "体の軸がブレています。頭の位置を安定させましょう。",

    # 参考メッセージ
    "shoulder_angle": "体が開き気味かもしれません（参考）。",
    "waist_rotation": "腰の回転が少し弱い可能性があります（参考）。",
}

# 赤丸や描画対象（右利き固定）
FOCUS_LANDMARK = {
    "impact_height": 16,   # 右手首
    "elbow_angle": 14,    # 右肘
    "body_sway": 24,      # 右腰
}

# ================================
# Utility
# ================================

def to_pixel(p, w, h):
    return int(p[0] * w), int(p[1] * h)


def save_frame(video_path, idx, out_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    cv2.imwrite(out_path, frame)
    return frame


def smooth(x, w=5):
    return np.convolve(x, np.ones(w) / w, mode="same")


# ================================
# 🎯インパクト推定（右手首最高点）
# ================================

def detect_contact_frame(landmarks_3d):
    """
    接触フレーム推定：
    ・右手首が最高点 → 下降開始の直後
    """
    n = len(landmarks_3d)
    if n < 15:
        return int(n * 0.7)

    WRIST = 16

    wrist_y = np.array([landmarks_3d[i][WRIST][1] for i in range(n)])
    wrist_y = smooth(wrist_y, 5)

    peak = int(np.argmin(wrist_y))

    search_end = min(n - 1, peak + 8)
    best = peak

    for i in range(peak + 1, search_end):
        if wrist_y[i] - wrist_y[i - 1] > 0:
            best = i
            break

    return min(n - 1, best + 1)


# ================================
# 🎨図解描画ルール
# ================================

def draw_focus(frame, focus, ux, uy, ix, iy):
    """
    focusごとに描画を変える
    """

    # 打点高さ → 横ライン
    if focus == "impact_height":
        cv2.line(frame, (0, iy), (frame.shape[1], iy), (0, 255, 0), 3)
        cv2.line(frame, (0, uy), (frame.shape[1], uy), (0, 0, 255), 3)

    # 肘角度 → ターゲットマーク
    elif focus == "elbow_angle":
        cv2.circle(frame, (ux, uy), 25, (0, 0, 255), 3)
        cv2.circle(frame, (ux, uy), 5, (0, 0, 255), -1)

        cv2.circle(frame, (ix, iy), 25, (0, 255, 0), 3)
        cv2.circle(frame, (ix, iy), 5, (0, 255, 0), -1)

    # 体軸ブレ → 縦ライン
    elif focus == "body_sway":
        cv2.line(frame, (ix, 0), (ix, frame.shape[0]), (0, 255, 0), 3)
        cv2.line(frame, (ux, 0), (ux, frame.shape[0]), (0, 0, 255), 3)

    # その他 → 小さめ丸だけ
    else:
        cv2.circle(frame, (ux, uy), 15, (0, 0, 255), -1)
        cv2.circle(frame, (ix, iy), 15, (0, 255, 0), -1)


# ================================
# メイン解析
# ================================

def analyze_video(file_path):
    BASE_DIR = os.path.dirname(__file__)
    success_path = os.path.join(BASE_DIR, "success.mp4")

    # 骨格抽出
    success_3d = extract_pose_landmarks(success_path)
    target_3d = extract_pose_landmarks(file_path)

    success_seq = normalize_pose(success_3d)
    target_seq = normalize_pose(target_3d)

    if len(success_seq) == 0 or len(target_seq) == 0:
        return {"menu": ["基本フォーム練習"], "ai_text": "解析できませんでした"}

    # ----------------
    # スコア計算
    # ----------------
    dists = []
    for t in target_seq:
        d = np.linalg.norm(success_seq - t, axis=(1, 2))
        dists.append(np.min(d))

    score = int(max(0, min(100, 100 - np.mean(dists) * 28)))

    # ----------------
    # 指標計算（全部残す）
    # ----------------
    is_right = True

    shoulder_diff = np.mean(calculate_shoulder_angle(target_seq, is_right))
    elbow_diff = np.mean(calculate_elbow_angle(target_seq, is_right))
    sway_diff = np.mean(calculate_body_sway(target_seq))
    impact_h_diff = np.mean(calculate_impact_height(target_seq, is_right))

    # ----------------
    # weakness判定
    # ----------------
    weakness = {
        "impact_height": "low" if impact_h_diff < -0.15 else "ok",
        "elbow_angle": "too_bent" if elbow_diff < -20 else "ok",
        "body_sway": "unstable" if sway_diff > 0.03 else "ok",

        # 参考指標
        "shoulder_angle": "too_open" if shoulder_diff > 15 else "ok",
    }

    # ----------------
    # focus決定（メイン3つだけ）
    # ----------------
    focus = "impact_height"
    for k in MAIN_FOCUS:
        if weakness.get(k) != "ok":
            focus = k
            break

    # ----------------
    # メニュー（1個だけ）
    # ----------------
    menu = [f"{FOCUS_LABELS[focus]}を改善する練習を1つだけやりましょう"]

    # ----------------
    # 図解生成
    # ----------------
    out_dir = os.path.join(BASE_DIR, "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    user_idx = detect_contact_frame(target_3d)
    ideal_idx = detect_contact_frame(success_3d)

    lid = FOCUS_LANDMARK[focus]

    cap = cv2.VideoCapture(file_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    ux, uy = to_pixel(target_3d[user_idx][lid], w, h)
    ix, iy = to_pixel(success_3d[ideal_idx][lid], w, h)

    # ideal
    ideal_img = save_frame(success_path, ideal_idx, os.path.join(out_dir, "ideal.png"))
    if ideal_img is not None:
        draw_focus(ideal_img, focus, ix, iy, ix, iy)
        cv2.imwrite(os.path.join(out_dir, "ideal.png"), ideal_img)

    # user
    user_img = save_frame(file_path, user_idx, os.path.join(out_dir, "user.png"))
    if user_img is not None:
        draw_focus(user_img, focus, ux, uy, ix, iy)
        cv2.imwrite(os.path.join(out_dir, "user.png"), user_img)

    # ----------------
    # AI文章（短く）
    # ----------------
    ai_text = f"改善ポイントは「{FOCUS_LABELS[focus]}」です。まず1つだけ意識しましょう！"

    return {
        "diagnosis": {
            "player": {"age": 13, "hand": "right", "serve_score": score},
            "weakness": weakness,
        },
        "menu": menu,
        "ai_text": ai_text,
        "ideal_image": "/outputs/ideal.png",
        "user_image": "/outputs/user.png",
        "focus_label": FOCUS_LABELS[focus],
        "message": FOCUS_MESSAGES.get(focus, "フォームを改善しましょう！"),
    }
