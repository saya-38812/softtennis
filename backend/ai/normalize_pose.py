import numpy as np


def normalize_pose(poses: np.ndarray) -> np.ndarray:
    """
    骨格配列を安定して正規化する（最強版）

    Input:
        poses: [F,33,3]

    Output:
        norm_pose: [F,33,2]
    """

    poses = np.array(poses)

    # ================================
    # shapeチェック
    # ================================
    if poses.ndim != 3 or poses.shape[1] != 33:
        raise ValueError(
            f"normalize_pose に渡された poses の形状が不正です: {poses.shape}"
        )

    # ================================
    # pelvis中心（腰の中心）
    # ================================
    pelvis = (poses[:, 23] + poses[:, 24]) / 2
    poses = poses - pelvis[:, None, :]

    # ================================
    # スケール計算（安定版）
    # 肩幅を基準にする（サーブで一番ブレにくい）
    # ================================
    left_shoulder = poses[:, 11]
    right_shoulder = poses[:, 12]

    scale = np.linalg.norm(left_shoulder - right_shoulder, axis=1)

    # ================================
    # scaleが0のフレームを除外
    # ================================
    valid = scale > 1e-6

    if np.sum(valid) < 5:
        raise ValueError("骨格が安定して検出できませんでした（scaleが小さすぎます）")

    poses = poses[valid]
    scale = scale[valid]

    # ================================
    # 正規化
    # ================================
    poses = poses / scale[:, None, None]

    # xyだけ返す
    return poses[:, :, :2]
