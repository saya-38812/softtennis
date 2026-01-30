import numpy as np
import logging
from typing import List

logging.basicConfig(level=logging.INFO)

def generate_menu(mean_diff: np.ndarray, 
                  shoulder_angle_diff: float = 0.0,
                  elbow_angle_diff: float = 0.0,
                  wrist_angle_diff: float = 0.0,
                  shoulder_tilt_diff: float = 0.0,
                  waist_rotation_diff: float = 0.0,
                  waist_speed_diff: float = 0.0,
                  body_sway_diff: float = 0.0,
                  impact_forward_diff: float = 0.0,
                  toss_sync_diff: float = 0.0,
                  impact_left_right_diff: float = 0.0,
                  weight_left_right_diff: float = 0.0) -> List[str]:
    """
    mean_diff: np.array([impact_height, weight_transfer, body_open])
    新しい角度指標もオプションで受け取る
    戻り値: 推奨練習メニューのリスト（日本語・難易度付き）
    """
    menu = []
    # インパクトの高さ
    if mean_diff[0] < -0.15:
        menu.append("インパクト高さ練習（強化）")
    elif mean_diff[0] < -0.05:
        menu.append("インパクト高さ練習（軽め）")
    # 体重移動
    if mean_diff[1] < -0.15:
        menu.append("体重移動練習（強化）")
    elif mean_diff[1] < -0.05:
        menu.append("体重移動練習（軽め）")
    # 体の開き/閉じ
    if mean_diff[2] > 0.1:
        menu.append("体の開き練習（強化）")
    elif mean_diff[2] > 0.05:
        menu.append("体の開き練習（軽め）")
    elif mean_diff[2] < -0.1:
        menu.append("体の閉じ方練習（強化）")
    elif mean_diff[2] < -0.05:
        menu.append("体の閉じ方練習（軽め）")
    
    # 新しい角度指標に基づくメニュー
    # 肩の開き角度
    if shoulder_angle_diff > 15:
        menu.append("肩の開きタイミング練習（早すぎる）")
    elif shoulder_angle_diff < -15:
        menu.append("肩の開きタイミング練習（遅すぎる）")
    
    # 肘の角度
    if elbow_angle_diff > 20:
        menu.append("肘の使い方練習（伸びすぎ）")
    elif elbow_angle_diff < -20:
        menu.append("肘の使い方練習（曲がりすぎ）")
    
    # 手首角度
    if wrist_angle_diff > 30:
        menu.append("手首の使い方練習（過伸展注意）")
    elif wrist_angle_diff < -30:
        menu.append("手首の使い方練習（曲げすぎ）")
    
    # 肩の高さ差
    if abs(shoulder_tilt_diff) > 0.05:
        menu.append("肩のバランス練習（左右の高さ調整）")
    
    # 腰の回転角度
    if waist_rotation_diff < -20:
        menu.append("腰の回転練習（強化）")
    elif waist_rotation_diff < -10:
        menu.append("腰の回転練習（軽め）")
    elif waist_rotation_diff > 20:
        menu.append("腰の回転制御練習（回転しすぎ）")
    
    # 腰回転速度
    if waist_speed_diff < -50:
        menu.append("腰回転速度向上練習（強化）")
    elif waist_speed_diff < -20:
        menu.append("腰回転速度向上練習（軽め）")
    
    # 体軸ブレ量
    if body_sway_diff > 0.03:
        menu.append("体軸安定性練習（左右ブレ改善）")
    
    # 打点前後位置
    if impact_forward_diff < -0.1:
        menu.append("打点前後位置練習（後ろ過ぎ）")
    elif impact_forward_diff > 0.1:
        menu.append("打点前後位置練習（前過ぎ）")
    
    # トスとのタイミング差
    if abs(toss_sync_diff) > 0.2:
        menu.append("トスとのタイミング練習（同期改善）")
    
    # 打点左右偏差
    if impact_left_right_diff > 0.05:
        menu.append("打点左右位置練習（左右ブレ改善）")
    
    # 体重左右分布
    if weight_left_right_diff > 0.03:
        menu.append("体重左右バランス練習（左右分布改善）")
    
    logging.info(f"mean_diff_values: {mean_diff}")
    logging.info(f"generated menu: {menu}")
    return menu

def safe_mean(frame_diffs: np.ndarray, landmark_idx: int, coord_idx: int) -> float:
    if frame_diffs.shape[1] > landmark_idx and frame_diffs.shape[2] > coord_idx:
        return np.mean(frame_diffs[:, landmark_idx, coord_idx])
    else:
        return 0.0
