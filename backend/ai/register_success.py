import numpy as np
from .video_pose_analyzer import extract_pose_landmarks
from .normalize_pose import normalize_pose

landmarks = extract_pose_landmarks("./backend/ai/success.mp4")
norm = normalize_pose(landmarks)

np.save("success_pose.npy", norm)
print("✅ 成功フォーム登録完了")
