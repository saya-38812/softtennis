import os
import numpy as np
import cv2
import logging
import uuid
import glob

from .video_pose_analyzer import extract_pose_landmarks
from .normalize_pose import normalize_pose

from .angle_utils import (
    calculate_elbow_angle,
    calculate_body_sway,
    calculate_impact_height,
)

logging.basicConfig(level=logging.INFO)

# ==============================
# 改善フォーカス（3つだけ）
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

FOCUS_LANDMARK = {
    "impact_height": 16,
    "elbow_angle": 14,
    "body_sway": 24,
}

# ==============================
# 腕が一番上の瞬間を使う
# ==============================
def detect_contact_frame(norm_landmarks):

    n = len(norm_landmarks)
    if n < 10:
        return int(n * 0.7)

    WRIST = 16
    wrist_y = np.array([norm_landmarks[i][WRIST][1] for i in range(n)])

    return int(np.argmin(wrist_y))


# ==============================
# 描画ルール
# ==============================
def draw_focus(frame, focus, ux, uy, ix, iy):

    h, w = frame.shape[:2]

    if focus == "impact_height":

        pass

    elif focus == "elbow_angle":

        cv2.circle(frame, (ux, uy), 28, (0, 0, 255), 3)
        cv2.circle(frame, (ix, iy), 28, (0, 255, 0), 3)

        cv2.arrowedLine(frame, (ux, uy), (ix, iy),
                        (255, 255, 255), 3, tipLength=0.3)

    elif focus == "body_sway":

        cv2.line(frame, (ix, 0), (ix, h), (0, 255, 0), 4)
        cv2.line(frame, (ux, 0), (ux, h), (0, 0, 255), 4)


# ==============================
# メイン解析
# ==============================
def analyze_video(file_path):

    BASE_DIR = os.path.dirname(__file__)
    success_path = os.path.join(BASE_DIR, "success.mp4")

    # --------------------------
    # 骨格抽出
    # --------------------------
    success = extract_pose_landmarks(success_path)
    target  = extract_pose_landmarks(file_path)

    success_norm = success["norm"]
    target_norm  = target["norm"]

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

    # focus決定
    focus = "impact_height"
    for k in MAIN_FOCUS:
        if weakness[k] != "ok":
            focus = k
            break

    # --------------------------
    # フレーム取得
    # --------------------------
    user_idx  = detect_contact_frame(target_norm)
    ideal_idx = detect_contact_frame(success_norm)

    lid = FOCUS_LANDMARK[focus]

    ux, uy = target_pixel[user_idx][lid]
    ix, iy = success_pixel[ideal_idx][lid]

    user_img  = target_frames[user_idx].copy()
    ideal_img = success_frames[ideal_idx].copy()

    draw_focus(user_img, focus, ux, uy, ix, iy)
    draw_focus(ideal_img, focus, ix, iy, ix, iy)

    # --------------------------
    # 保存（UUIDでキャッシュ防止）
    # --------------------------
    out_dir = os.path.join(BASE_DIR, "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    uid = str(uuid.uuid4())

    user_file  = f"user_{uid}.png"
    ideal_file = f"ideal_{uid}.png"

    cv2.imwrite(os.path.join(out_dir, user_file), user_img)
    cv2.imwrite(os.path.join(out_dir, ideal_file), ideal_img)

    # --------------------------
    # ✅古い画像を削除（50枚超えたら整理）
    # --------------------------
    files = glob.glob(out_dir + "/*.png")
    if len(files) > 50:
        for f in files[:20]:
            os.remove(f)

    # --------------------------
    # 結果返却
    # --------------------------
    return {
        "diagnosis": {"weakness": weakness},
        "menu": [f"{FOCUS_LABELS[focus]}を改善する練習を1つだけやりましょう"],
        "ai_text": f"改善ポイントは「{FOCUS_LABELS[focus]}」です。",
        "ideal_image": f"/outputs/{ideal_file}",
        "user_image": f"/outputs/{user_file}",
        "focus_label": FOCUS_LABELS[focus],
        "message": FOCUS_MESSAGES[focus],
    }
