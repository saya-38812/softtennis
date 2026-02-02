import os
import numpy as np
import cv2
import logging
import time   # ←追加（キャッシュ対策）

from .video_pose_analyzer import extract_pose_landmarks
from .normalize_pose import normalize_pose

from .angle_utils import (
    calculate_elbow_angle,
    calculate_body_sway,
    calculate_impact_height,
)

logging.basicConfig(level=logging.INFO)

# ==============================
# MVPで強く出す改善ポイント（3つだけ）
# ==============================
MAIN_FOCUS = ["impact_height", "elbow_angle", "body_sway"]

FOCUS_LABELS = {
    "impact_height": "打点の高さ",
    "elbow_angle": "肘の角度",
    "body_sway": "体軸のブレ",
}

FOCUS_MESSAGES = {
    "impact_height": "打点が低いです。もっと高い位置で当てましょう。",
    "elbow_angle": "肘が曲がりすぎています。インパクトで伸ばしましょう。",
    "body_sway": "体の軸がブレています。頭の位置を安定させましょう。",
}

# 描画対象ランドマーク（右利き固定）
FOCUS_LANDMARK = {
    "impact_height": 16,  # 手首
    "elbow_angle": 14,   # 肘
    "body_sway": 24,     # 腰
}

# ==============================
# ✅腕が一番上の瞬間で固定（ズレない）
# ==============================
def detect_top_arm_frame(norm_landmarks):

    n = len(norm_landmarks)
    if n < 10:
        return int(n * 0.7)

    WRIST = 16
    wrist_y = np.array([norm_landmarks[i][WRIST][1] for i in range(n)])

    best = int(np.argmin(wrist_y))
    return best


# ==============================
# ✅フレーム保存（必要な1枚だけ）
# ==============================
def save_frame(video_path, idx, out_path):

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    cv2.imwrite(out_path, frame)
    return frame


# ==============================
# ✅描画ルール（MVP最終版）
# ==============================
def draw_focus(frame, focus, ux, uy, ix, iy):

    h, w = frame.shape[:2]

    # --------------------------
    # ① 打点高さ → 横ライン
    # --------------------------
    if focus == "impact_height":
        cv2.line(frame, (0, iy), (w, iy), (0, 255, 0), 4)
        cv2.line(frame, (0, uy), (w, uy), (0, 0, 255), 4)

    # --------------------------
    # ② 肘角度 → ターゲットマーク
    # --------------------------
    elif focus == "elbow_angle":

        cv2.circle(frame, (ux, uy), 28, (0, 0, 255), 3)
        cv2.circle(frame, (ux, uy), 6, (0, 0, 255), -1)

        cv2.circle(frame, (ix, iy), 28, (0, 255, 0), 3)
        cv2.circle(frame, (ix, iy), 6, (0, 255, 0), -1)

        cv2.arrowedLine(frame, (ux, uy), (ix, iy),
                        (255, 255, 255), 3, tipLength=0.3)

    # --------------------------
    # ③ 体軸ブレ → 縦ライン
    # --------------------------
    elif focus == "body_sway":

        cv2.line(frame, (ix, 0), (ix, h), (0, 255, 0), 4)
        cv2.line(frame, (ux, 0), (ux, h), (0, 0, 255), 4)


# ==============================
# ✅メイン解析
# ==============================
def analyze_video(file_path):

    BASE_DIR = os.path.dirname(__file__)
    success_path = os.path.join(BASE_DIR, "success.mp4")

    # --------------------------
    # 骨格抽出（軽量版）
    # --------------------------
    success = extract_pose_landmarks(success_path)
    target  = extract_pose_landmarks(file_path)

    success_norm  = success["norm"]
    target_norm   = target["norm"]

    success_pixel = success["pixel"]
    target_pixel  = target["pixel"]

    if len(success_norm) == 0 or len(target_norm) == 0:
        return {"menu": ["基本フォーム練習"], "ai_text": "解析できませんでした"}

    # --------------------------
    # 正規化（診断用）
    # --------------------------
    success_seq = normalize_pose(success_norm)
    target_seq  = normalize_pose(target_norm)

    # --------------------------
    # 指標計算（3つだけ）
    # --------------------------
    elbow_val  = np.mean(calculate_elbow_angle(target_seq, True))
    impact_val = np.mean(calculate_impact_height(target_seq, True))
    sway_val   = np.mean(calculate_body_sway(target_seq))

    weakness = {
        "impact_height": "low" if impact_val < -0.15 else "ok",
        "elbow_angle": "too_bent" if elbow_val < -20 else "ok",
        "body_sway": "unstable" if sway_val > 0.03 else "ok",
    }

    # --------------------------
    # focus決定（優先順）
    # --------------------------
    focus = "impact_height"
    for k in MAIN_FOCUS:
        if weakness[k] != "ok":
            focus = k
            break

    # --------------------------
    # 腕最高点フレームで固定
    # --------------------------
    user_idx  = detect_top_arm_frame(target_norm)
    ideal_idx = detect_top_arm_frame(success_norm)

    lid = FOCUS_LANDMARK[focus]

    ux, uy = target_pixel[user_idx][lid]
    ix, iy = success_pixel[ideal_idx][lid]

    # --------------------------
    # 保存先
    # --------------------------
    out_dir = os.path.join(BASE_DIR, "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    user_path  = os.path.join(out_dir, "user.png")
    ideal_path = os.path.join(out_dir, "ideal.png")

    # --------------------------
    # 必要な1枚だけ保存して描画
    # --------------------------
    user_img  = save_frame(file_path, user_idx, user_path)
    ideal_img = save_frame(success_path, ideal_idx, ideal_path)

    if user_img is not None:
        draw_focus(user_img, focus, ux, uy, ix, iy)
        cv2.imwrite(user_path, user_img)

    if ideal_img is not None:
        draw_focus(ideal_img, focus, ix, iy, ix, iy)
        cv2.imwrite(ideal_path, ideal_img)

    # --------------------------
    # ✅キャッシュ対策：毎回URLを変える
    # --------------------------
    cache_buster = int(time.time())

    # --------------------------
    # 結果返却
    # --------------------------
    return {
        "diagnosis": {
            "weakness": weakness,
        },
        "menu": [f"{FOCUS_LABELS[focus]}を改善する練習を1つだけやりましょう"],
        "ai_text": f"改善ポイントは「{FOCUS_LABELS[focus]}」です。",

        # 👇ここが最重要
        "ideal_image": f"/outputs/ideal.png?v={cache_buster}",
        "user_image":  f"/outputs/user.png?v={cache_buster}",

        "focus_label": FOCUS_LABELS[focus],
        "message": FOCUS_MESSAGES[focus],
    }
