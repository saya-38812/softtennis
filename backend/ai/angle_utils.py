import numpy as np
from typing import Tuple, Optional

# Mediapipe Pose Landmark インデックス
# 右利きを想定（右腕がラケットを持つ側）
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
NOSE = 0

def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    3点から角度を計算（p2を頂点とする角度）
    入力: p1, p2, p3 (2D座標 [x, y])
    出力: 角度（度）
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    # ベクトルの長さを計算
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # 内積から角度を計算
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 数値誤差対策
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def calculate_shoulder_angle(poses: np.ndarray, is_right_handed: bool = True) -> np.ndarray:
    """
    肩の開き角度を計算
    肩-肩-骨盤中心の角度で測定
    入力: poses [frame数, landmark数, 2(xy)]
    出力: [frame数] の角度配列（度）
    """
    if poses.shape[0] == 0:
        return np.array([])
    
    # 骨盤中心を計算（23: 左股関節, 24: 右股関節）
    pelvis_center = (poses[:, 23] + poses[:, 24]) / 2
    
    # 右利きの場合：右肩-左肩-骨盤中心の角度
    # 左利きの場合：左肩-右肩-骨盤中心の角度
    if is_right_handed:
        angles = []
        for i in range(len(poses)):
            angle = calculate_angle(
                poses[i, RIGHT_SHOULDER],
                poses[i, LEFT_SHOULDER],
                pelvis_center[i]
            )
            angles.append(angle)
    else:
        angles = []
        for i in range(len(poses)):
            angle = calculate_angle(
                poses[i, LEFT_SHOULDER],
                poses[i, RIGHT_SHOULDER],
                pelvis_center[i]
            )
            angles.append(angle)
    
    return np.array(angles)

def calculate_elbow_angle(poses: np.ndarray, is_right_handed: bool = True) -> np.ndarray:
    """
    肘の角度を計算
    肩-肘-手首の角度で測定
    入力: poses [frame数, landmark数, 2(xy)]
    出力: [frame数] の角度配列（度）
    """
    if poses.shape[0] == 0:
        return np.array([])
    
    if is_right_handed:
        # 右腕の肘角度
        angles = []
        for i in range(len(poses)):
            angle = calculate_angle(
                poses[i, RIGHT_SHOULDER],
                poses[i, RIGHT_ELBOW],
                poses[i, RIGHT_WRIST]
            )
            angles.append(angle)
    else:
        # 左腕の肘角度
        angles = []
        for i in range(len(poses)):
            angle = calculate_angle(
                poses[i, LEFT_SHOULDER],
                poses[i, LEFT_ELBOW],
                poses[i, LEFT_WRIST]
            )
            angles.append(angle)
    
    return np.array(angles)

def calculate_wrist_angle(poses: np.ndarray, is_right_handed: bool = True) -> np.ndarray:
    """
    手首角度を計算
    手首の曲がり具合を評価（過伸展を検出）
    肘-手首-手首の延長方向の角度で測定
    入力: poses [frame数, landmark数, 2(xy)]
    出力: [frame数] の角度配列（度）
    """
    if poses.shape[0] == 0:
        return np.array([])
    
    if is_right_handed:
        # 右腕の手首角度
        # 手首の曲がり具合を測定するため、肘-手首のベクトルと手首の延長方向の角度を計算
        angles = []
        for i in range(len(poses)):
            # 肘から手首へのベクトル
            elbow_to_wrist = poses[i, RIGHT_WRIST] - poses[i, RIGHT_ELBOW]
            # 手首の延長方向（手首から少し先の点）
            # 手首から肘方向への逆方向に延長
            wrist_extension = poses[i, RIGHT_WRIST] - elbow_to_wrist * 0.2
            
            angle = calculate_angle(
                poses[i, RIGHT_ELBOW],
                poses[i, RIGHT_WRIST],
                wrist_extension
            )
            angles.append(angle)
    else:
        # 左腕の手首角度
        angles = []
        for i in range(len(poses)):
            elbow_to_wrist = poses[i, LEFT_WRIST] - poses[i, LEFT_ELBOW]
            wrist_extension = poses[i, LEFT_WRIST] - elbow_to_wrist * 0.2
            
            angle = calculate_angle(
                poses[i, LEFT_ELBOW],
                poses[i, LEFT_WRIST],
                wrist_extension
            )
            angles.append(angle)
    
    return np.array(angles)

def calculate_shoulder_tilt(poses: np.ndarray) -> np.ndarray:
    """
    肩の高さ差を計算
    左右の肩のY座標の差（絶対値）
    入力: poses [frame数, landmark数, 2(xy)]
    出力: [frame数] の高さ差配列
    """
    if poses.shape[0] == 0:
        return np.array([])
    
    # 左右の肩のY座標の差
    left_shoulder_y = poses[:, LEFT_SHOULDER, 1]
    right_shoulder_y = poses[:, RIGHT_SHOULDER, 1]
    tilt = np.abs(left_shoulder_y - right_shoulder_y)
    
    return tilt

def calculate_waist_rotation(poses: np.ndarray, is_right_handed: bool = True) -> np.ndarray:
    """
    腰の回転角度を計算
    骨盤（左右の股関節）を結ぶ線の角度を測定（初期フレームからの相対角度）
    入力: poses [frame数, landmark数, 2(xy)]
    出力: [frame数] の角度配列（度）
    """
    if poses.shape[0] == 0:
        return np.array([])
    
    rotations = []
    # 初期フレーム（トップスイング時）の骨盤の向きを基準とする
    initial_hip_vector = poses[0, RIGHT_HIP] - poses[0, LEFT_HIP]
    initial_hip_norm = np.linalg.norm(initial_hip_vector)
    
    if initial_hip_norm == 0:
        # 初期フレームが無効な場合は前方向を基準とする
        initial_hip_vector = np.array([0.0, 1.0])
        initial_hip_norm = 1.0
    
    for i in range(len(poses)):
        # 現在フレームの左右の股関節を結ぶベクトル
        hip_vector = poses[i, RIGHT_HIP] - poses[i, LEFT_HIP]
        
        # ベクトルの長さを計算
        hip_norm = np.linalg.norm(hip_vector)
        if hip_norm == 0:
            rotations.append(0.0)
            continue
        
        # 初期フレームからの回転角度を計算
        cos_angle = np.dot(hip_vector, initial_hip_vector) / (hip_norm * initial_hip_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        # 回転方向を判定（外積の符号を使用）
        cross_product = hip_vector[0] * initial_hip_vector[1] - hip_vector[1] * initial_hip_vector[0]
        if cross_product < 0:
            angle_deg = -angle_deg
        
        rotations.append(angle_deg)
    
    return np.array(rotations)

def calculate_waist_rotation_speed(waist_rotations: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """
    腰回転速度を計算
    フレーム間の腰の回転角度の変化率（度/秒）
    入力: waist_rotations [frame数] の角度配列, fps フレームレート
    出力: [frame数-1] の速度配列（度/秒）
    """
    if len(waist_rotations) < 2:
        return np.array([])
    
    # フレーム間の角度差を計算
    angle_diffs = np.diff(waist_rotations)
    
    # 角度の不連続性を考慮（-180度と180度の境界を跨ぐ場合）
    angle_diffs = np.where(angle_diffs > 180, angle_diffs - 360, angle_diffs)
    angle_diffs = np.where(angle_diffs < -180, angle_diffs + 360, angle_diffs)
    
    # 度/秒に変換
    speeds = np.abs(angle_diffs) * fps
    
    return speeds

def calculate_body_sway(poses: np.ndarray) -> np.ndarray:
    """
    体軸ブレ量を計算
    インパクト時の左右ブレを測定（骨盤中心から頭頂部へのベクトルのX座標成分）
    入力: poses [frame数, landmark数, 2(xy)]
    出力: [frame数] のブレ量配列（X座標の絶対値）
    """
    if poses.shape[0] == 0:
        return np.array([])
    
    # 骨盤中心を計算
    pelvis_center = (poses[:, LEFT_HIP] + poses[:, RIGHT_HIP]) / 2
    
    # 頭頂部の目安として肩の中心を使用（より正確には鼻の位置を使用可能）
    # 肩の中心
    shoulder_center = (poses[:, LEFT_SHOULDER] + poses[:, RIGHT_SHOULDER]) / 2
    
    # 体軸ベクトル（骨盤中心から肩の中心へ）
    body_axis = shoulder_center - pelvis_center
    
    # X座標成分の絶対値（左右ブレ量）
    sway = np.abs(body_axis[:, 0])
    
    return sway

def calculate_impact_height(poses: np.ndarray, is_right_handed: bool = True) -> np.ndarray:
    """
    打点高さを計算
    インパクト時の手首のY座標（高さ）
    入力: poses [frame数, landmark数, 2(xy)]
    出力: [frame数] の高さ配列
    """
    if poses.shape[0] == 0:
        return np.array([])
    
    if is_right_handed:
        # 右腕の手首のY座標（高さ）
        heights = poses[:, RIGHT_WRIST, 1]
    else:
        # 左腕の手首のY座標（高さ）
        heights = poses[:, LEFT_WRIST, 1]
    
    return heights

def calculate_impact_forward(poses: np.ndarray, is_right_handed: bool = True) -> np.ndarray:
    """
    打点前後位置を計算
    インパクト時の手首のY座標（前後方向、ネット方向が正）
    入力: poses [frame数, landmark数, 2(xy)]
    出力: [frame数] の前後位置配列
    """
    if poses.shape[0] == 0:
        return np.array([])
    
    if is_right_handed:
        # 右腕の手首のY座標（前後方向）
        forward_positions = poses[:, RIGHT_WRIST, 1]
    else:
        # 左腕の手首のY座標（前後方向）
        forward_positions = poses[:, LEFT_WRIST, 1]
    
    return forward_positions

def calculate_toss_sync(poses: np.ndarray, is_right_handed: bool = True) -> np.ndarray:
    """
    トスとのタイミング差を計算
    トス時（初期フレーム）とインパクト時の手首位置の変化を測定
    入力: poses [frame数, landmark数, 2(xy)]
    出力: [frame数] のタイミング差配列（フレーム数での差分）
    """
    if poses.shape[0] == 0:
        return np.array([])
    
    if is_right_handed:
        # 右腕の手首の初期位置（トス時）
        initial_wrist = poses[0, RIGHT_WRIST]
        # 各フレームでの手首位置
        wrist_positions = poses[:, RIGHT_WRIST]
    else:
        # 左腕の手首の初期位置（トス時）
        initial_wrist = poses[0, LEFT_WRIST]
        # 各フレームでの手首位置
        wrist_positions = poses[:, LEFT_WRIST]
    
    # 初期位置からの距離（タイミング差の指標）
    distances = np.linalg.norm(wrist_positions - initial_wrist, axis=1)
    
    return distances

def calculate_impact_left_right(poses: np.ndarray, is_right_handed: bool = True) -> np.ndarray:
    """
    打点左右偏差を計算
    インパクト時の手首のX座標（体の中心からの左右偏差）
    入力: poses [frame数, landmark数, 2(xy)]
    出力: [frame数] の左右偏差配列（X座標の絶対値）
    """
    if poses.shape[0] == 0:
        return np.array([])
    
    # 体の中心（骨盤中心）を計算
    pelvis_center = (poses[:, LEFT_HIP] + poses[:, RIGHT_HIP]) / 2
    
    if is_right_handed:
        # 右腕の手首のX座標
        wrist_x = poses[:, RIGHT_WRIST, 0]
    else:
        # 左腕の手首のX座標
        wrist_x = poses[:, LEFT_WRIST, 0]
    
    # 体の中心からの左右偏差（絶対値）
    left_right_deviation = np.abs(wrist_x - pelvis_center[:, 0])
    
    return left_right_deviation

def calculate_weight_left_right(poses: np.ndarray) -> np.ndarray:
    """
    体重左右分布を計算
    インパクト時の左右の股関節のY座標の差（体重の偏り）
    入力: poses [frame数, landmark数, 2(xy)]
    出力: [frame数] の左右分布配列（左右の股関節のY座標の差の絶対値）
    """
    if poses.shape[0] == 0:
        return np.array([])
    
    # 左右の股関節のY座標（高さ）
    left_hip_y = poses[:, LEFT_HIP, 1]
    right_hip_y = poses[:, RIGHT_HIP, 1]
    
    # 左右の高さ差（体重の偏りを示す）
    weight_distribution = np.abs(left_hip_y - right_hip_y)
    
    return weight_distribution

def get_angle_at_frames(angles: np.ndarray, frame_indices: list) -> np.ndarray:
    """
    指定されたフレームインデックスでの角度を取得
    入力: angles [frame数], frame_indices [インデックスのリスト]
    出力: [指定フレーム数] の角度配列
    """
    if len(angles) == 0:
        return np.array([])
    
    valid_indices = [idx for idx in frame_indices if 0 <= idx < len(angles)]
    if len(valid_indices) == 0:
        return np.array([])
    
    return angles[valid_indices]

