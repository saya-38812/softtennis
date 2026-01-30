import numpy as np

def normalize_pose(poses: np.ndarray) -> np.ndarray:
    """
    指定されたポーズ配列を原点中心＆身長スケールでXY座標に正規化
    入力: [frame数, landmark数, 3(xyz)]
    出力: [frame数, landmark数, 2(xy)]
    """
    poses = np.array(poses)
    if poses.ndim != 3 or poses.shape[1] == 0:
        raise ValueError(f"normalize_pose に渡された poses の形状が想定と違います: {poses.shape}")
    # 骨盤中心座標計算
    pelvis = (poses[:,23] + poses[:,24]) / 2
    poses = poses - pelvis[:,None,:]
    # 身長スケール
    scale = np.linalg.norm(poses[:,11] - poses[:,27], axis=1)
    poses = poses / scale[:,None,None]
    return poses[:,:,:2]
