import json
import numpy as np
from .normalize_pose import normalize_pose

SUCCESS_DB = "success_angles.json"

def register_success(pose_landmarks_list):
    """
    pose_landmarks_list:
    動画1本分の mediapipe landmarks（フレームごと）
    """

    all_angles = []

    for lm in pose_landmarks_list:
        angles = normalize_pose(lm)
        all_angles.append(list(angles.values()))

    mean_angles = np.mean(all_angles, axis=0).tolist()

    with open(SUCCESS_DB, "w") as f:
        json.dump(mean_angles, f, indent=2)

    print("✅ 成功フォーム（角度）を保存しました")
