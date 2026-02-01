import os
import logging
import numpy as np
import cv2

from .video_pose_analyzer import extract_pose_landmarks
from .normalize_pose import normalize_pose
from .coach_ai_utils import safe_mean
from .angle_utils import calculate_elbow_angle

logging.basicConfig(level=logging.INFO)

# ================================
# MVP設定
# ================================

FOCUS_LABELS = {
    "elbow_angle": "肘の角度",
    "impact_height": "打点の高さ",
}

FOCUS_MESSAGES = {
    "elbow_angle": "肘が曲がりすぎています。インパクトでしっかり伸ばしましょう。",
    "impact_height": "打点が低いです。できるだけ高い位置で当てましょう。",
}

FOCUS_PRIORITY = ["elbow_angle", "impact_height"]

FOCUS_MARK_LANDMARK = {
    "elbow_angle": 14,     # 右肘
    "impact_height": 16,   # 右手首
}

# ================================
# Utility
# ================================

def pick_focus(weakness):
    for k in FOCUS_PRIORITY:
        if weakness.get(k) != "ok":
            return k
    return "elbow_angle"


def to_pixel(p, w, h):
    return int(p[0] * w), int(p[1] * h)


def save_frame(video_path, idx, out_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(out_path, frame)
        return frame
    return None


def smooth(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode="same")


# ================================
# 本気インパクト推定（完成版）
# ================================

def detect_impact_frame_strict(landmarks_3d):
    """
    MediaPipeのみで限界まで精度を上げたインパクト推定
    条件：
    ・手首速度
    ・打点高さ
    ・肘伸展
    """

    n = len(landmarks_3d)
    if n < 15:
        return int(n * 0.7)

    WRIST, ELBOW, SHOULDER = 16, 14, 12

    speeds, heights, elbows = [], [], []

    for i in range(1, n):
        prev = landmarks_3d[i-1][WRIST]
        curr = landmarks_3d[i][WRIST]

        # 速度
        speeds.append(np.linalg.norm(curr - prev))

        # 高さ（y小さい＝高い）
        heights.append(-curr[1])

        # 肘角度
        s = landmarks_3d[i][SHOULDER]
        e = landmarks_3d[i][ELBOW]
        w = landmarks_3d[i][WRIST]

        v1 = s - e
        v2 = w - e
        cos = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        elbows.append(angle)

    speeds = smooth(np.array(speeds))
    heights = smooth(np.array(heights))
    elbows = smooth(np.array(elbows))

    # 正規化
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-6)

    score = (
        norm(speeds) * 0.5 +
        norm(heights) * 0.3 +
        norm(elbows) * 0.2
    )

    # 後半のみ探索
    start = int(len(score) * 0.4)
    impact_idx = start + np.argmax(score[start:])

    return impact_idx


# ================================
# メイン解析
# ================================

def analyze_video(file_path):
    BASE_DIR = os.path.dirname(__file__)
    success_path = os.path.join(BASE_DIR, "success.mp4")

    success_3d = extract_pose_landmarks(success_path)
    target_3d = extract_pose_landmarks(file_path)

    success_seq = normalize_pose(success_3d)
    target_seq = normalize_pose(target_3d)

    if len(success_seq) == 0 or len(target_seq) == 0:
        return {"menu": ["基本フォーム練習"], "ai_text": "解析できませんでした"}

    # ----------------
    # スコア
    # ----------------
    dists = []
    for t in target_seq:
        d = np.linalg.norm(success_seq - t, axis=(1,2))
        dists.append(np.min(d))

    score = int(max(0, min(100, 100 - np.mean(dists)*28)))

    impact_height = safe_mean(success_seq - target_seq, 0, 1)

    success_elbow = calculate_elbow_angle(success_seq, True)
    target_elbow = calculate_elbow_angle(target_seq, True)
    elbow_diff = np.mean(target_elbow) - np.mean(success_elbow)

    weakness = {
        "elbow_angle": "too_bent" if elbow_diff < -20 else "ok",
        "impact_height": "low" if impact_height < -0.15 else "ok",
    }

    focus = pick_focus(weakness)

    # ----------------
    # インパクト画像
    # ----------------
    out_dir = os.path.join(BASE_DIR, "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    user_idx = detect_impact_frame_strict(target_3d)
    ideal_idx = detect_impact_frame_strict(success_3d)

    lid = FOCUS_MARK_LANDMARK[focus]

    cap = cv2.VideoCapture(file_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    ux, uy = to_pixel(target_3d[user_idx][lid], w, h)
    ix, iy = to_pixel(success_3d[ideal_idx][lid], w, h)

    ideal_img = save_frame(success_path, ideal_idx, os.path.join(out_dir, "ideal.png"))
    if ideal_img is not None:
        cv2.circle(ideal_img, (ix, iy), 18, (0,255,0), -1)
        cv2.imwrite(os.path.join(out_dir, "ideal.png"), ideal_img)

    user_img = save_frame(file_path, user_idx, os.path.join(out_dir, "user.png"))
    if user_img is not None:
        cv2.circle(user_img, (ux, uy), 18, (0,0,255), -1)
        cv2.circle(user_img, (ix, iy), 18, (0,255,0), -1)
        cv2.arrowedLine(user_img, (ux, uy), (ix, iy), (255,255,255), 4)
        cv2.imwrite(os.path.join(out_dir, "user.png"), user_img)

    return {
        "diagnosis": {
            "player": {"age": 13, "hand": "right", "serve_score": score},
            "weakness": weakness,
        },
        "menu": ["肘を高く固定する素振り練習"],
        "ai_text": f"改善ポイントは「{FOCUS_LABELS[focus]}」です。まず1点に集中しましょう。",
        "ideal_image": "/outputs/ideal.png",
        "user_image": "/outputs/user.png",
        "focus_label": FOCUS_LABELS[focus],
        "message": FOCUS_MESSAGES[focus],
    }
