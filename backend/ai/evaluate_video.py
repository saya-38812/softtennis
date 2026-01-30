import numpy as np
import json
import os
from .video_pose_analyzer import extract_pose_landmarks
from .normalize_pose import normalize_pose
from .coach_ai_utils import generate_menu, safe_mean

def analyze_video(target_file_path):
    print("解析中...")

    # === 成功フォーム取得 ===
    success_path = "./backend/ai/success.mp4"
    if not os.path.exists(success_path):
        raise FileNotFoundError(f"{success_path} が存在しません")

    success_landmarks = extract_pose_landmarks(success_path)
    if success_landmarks is None or len(success_landmarks) == 0:
        raise ValueError("成功フォームのランドマークが取得できませんでした")

    success_seq = normalize_pose(success_landmarks)
    print("成功フォーム shape:", np.array(success_seq).shape)

    # === 対象フォーム取得 ===
    if not os.path.exists(target_file_path):
        raise FileNotFoundError(f"{target_file_path} が存在しません")

    target_landmarks = extract_pose_landmarks(target_file_path)
    if target_landmarks is None or len(target_landmarks) == 0:
        raise ValueError("対象フォームのランドマークが取得できませんでした")

    target_seq = normalize_pose(target_landmarks)
    print("対象フォーム shape:", np.array(target_seq).shape)

    # === フレーム対応最短距離評価 ===
    frame_scores = []
    frame_diffs  = []

    for t in target_seq:
        dists = np.linalg.norm(success_seq - t, axis=(1,2))
        min_idx = np.argmin(dists)
        frame_scores.append(dists[min_idx])
        frame_diffs.append(success_seq[min_idx] - t)

    frame_diffs = np.array(frame_diffs)
    mean_dist = np.mean(frame_scores)
    score = 100 - mean_dist * 28
    score = int(max(0, min(100, score)))

    # === 弱点評価 ===
    impact_height = safe_mean(frame_diffs, 0, 1)
    weight_transfer = safe_mean(frame_diffs, 1, 0)
    body_open = safe_mean(frame_diffs, 2, 2)

    mean_diff_values = np.array([impact_height, weight_transfer, body_open])
    menu = generate_menu(mean_diff_values)

    result = {
        "player": {
            "age": 13,
            "hand": "right",
            "serve_score": int(score)
        },
        "weakness": {
            "impact_height": "low" if impact_height < -0.15 else "ok",
            "weight_transfer": "poor" if weight_transfer < -0.15 else "ok",
            "body_open": "good" if body_open > -0.1 else "poor"
        },
        "raw_values": {
            "impact_diff": round(float(impact_height), 3),
            "weight_diff": round(float(weight_transfer), 3),
            "open_diff": round(float(body_open), 3)
        }
    }

    with open("player_diagnosis.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("診断JSONを書き出しました")
    return result, menu
