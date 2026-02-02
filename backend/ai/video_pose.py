import os
import numpy as np
import cv2
import logging

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
# ✅腕が一番上の瞬間で固定する（MVP最強）
# ==============================
def detect_top_arm_frame(norm_landmarks):
    """
    MVP最適解：
    ・右手首が最も高い瞬間（y最小）
    → インパクト推定を捨ててズレを消す
    """

    n = len(norm_landmarks)
    if n < 10:
        return int(n * 0.7)

    WRIST = 16

    wrist_y = np.array([
        norm_landmarks[i][WRIST][1]
        for i in range(n)
    ])

    best = int(np.argmin(wrist_y))
    return best


# ==============================
# ✅ユーザー側描画（赤＋緑＋矢印）
# ==============================
def draw_user(frame, focus, ux, uy, ix, iy):

    # --------------------------
    # ① 打点高さ → 丸＋矢印（ライン廃止）
    # --------------------------
    if focus == "impact_height":

        cv2.circle(frame, (ux, uy), 18, (0, 0, 255), -1)   # あなた（赤）
        cv2.circle(frame, (ix, iy), 18, (0, 255, 0), -1)   # 理想（緑）

        cv2.arrowedLine(
            frame,
            (ux, uy),
            (ix, iy),
            (255, 255, 255),
            3,
            tipLength=0.3
        )

    # --------------------------
    # ② 肘角度 → ターゲットマーク
    # --------------------------
    elif focus == "elbow_angle":

        # あなた（赤）
        cv2.circle(frame, (ux, uy), 28, (0, 0, 255), 3)
        cv2.circle(frame, (ux, uy), 6, (0, 0, 255), -1)

        # 理想（緑）
        cv2.circle(frame, (ix, iy), 28, (0, 255, 0), 3)
        cv2.circle(frame, (ix, iy), 6, (0, 255, 0), -1)

        cv2.arrowedLine(frame, (ux, uy), (ix, iy),
                        (255, 255, 255), 3, tipLength=0.3)

    # --------------------------
    # ③ 体軸ブレ → 縦ライン
    # --------------------------
    elif focus == "body_sway":

        h, w = frame.shape[:2]

        # 理想軸（緑）
        cv2.line(frame, (ix, 0), (ix, h), (0, 255, 0), 4)

        # あなた軸（赤）
        cv2.line(frame, (ux, 0), (ux, h), (0, 0, 255), 4)


# ==============================
# ✅理想側描画（緑だけ）
# ==============================
def draw_ideal(frame, focus, ix, iy):

    if focus == "impact_height":
        cv2.circle(frame, (ix, iy), 18, (0, 255, 0), -1)

    elif focus == "elbow_angle":
        cv2.circle(frame, (ix, iy), 28, (0, 255, 0), 3)
        cv2.circle(frame, (ix, iy), 6, (0, 255, 0), -1)

    elif focus == "body_sway":
        h, w = frame.shape[:2]
        cv2.line(frame, (ix, 0), (ix, h), (0, 255, 0), 4)


# ==============================
# ✅メイン解析（MVP完成版）
# ==============================
def analyze_video(file_path):

    BASE_DIR = os.path.dirname(__file__)
    success_path = os.path.join(BASE_DIR, "success.mp4")

    # --------------------------
    # 骨格抽出（norm + pixel + frames）
    # --------------------------
    success = extract_pose_landmarks(success_path)
    target  = extract_pose_landmarks(file_path)

    success_norm  = success["norm"]
    target_norm   = target["norm"]

    success_pixel = success["pixel"]
    target_pixel  = target["pixel"]

    success_frames = success["frames"]
    target_frames  = target["frames"]

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
    # ✅腕最高点フレームで固定（ズレない）
    # --------------------------
    user_idx  = detect_top_arm_frame(target_norm)
    ideal_idx = detect_top_arm_frame(success_norm)

    lid = FOCUS_LANDMARK[focus]

    ux, uy = target_pixel[user_idx][lid]
    ix, iy = success_pixel[ideal_idx][lid]

    # --------------------------
    # フレームを直接描画（絶対ズレない）
    # --------------------------
    user_img  = target_frames[user_idx].copy()
    ideal_img = success_frames[ideal_idx].copy()

    draw_user(user_img, focus, ux, uy, ix, iy)
    draw_ideal(ideal_img, focus, ix, iy)

    # --------------------------
    # 保存
    # --------------------------
    out_dir = os.path.join(BASE_DIR, "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, "user.png"), user_img)
    cv2.imwrite(os.path.join(out_dir, "ideal.png"), ideal_img)

    # --------------------------
    # 結果返却
    # --------------------------
    return {
        "diagnosis": {
            "weakness": weakness,
        },
        "menu": [f"{FOCUS_LABELS[focus]}を改善する練習を1つだけやりましょう"],
        "ai_text": f"改善ポイントは「{FOCUS_LABELS[focus]}」です。",
        "ideal_image": "/outputs/ideal.png",
        "user_image": "/outputs/user.png",
        "focus_label": FOCUS_LABELS[focus],
        "message": FOCUS_MESSAGES[focus],
    }
