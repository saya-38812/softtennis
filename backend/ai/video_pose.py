import numpy as np
from .coach_ai_utils import generate_menu, safe_mean
from .video_pose_analyzer import extract_pose_landmarks
from .normalize_pose import normalize_pose
from .angle_utils import (
    calculate_shoulder_angle,
    calculate_elbow_angle,
    calculate_wrist_angle,
    calculate_shoulder_tilt,
    calculate_waist_rotation,
    calculate_waist_rotation_speed,
    calculate_body_sway,
    calculate_impact_height,
    calculate_impact_forward,
    calculate_toss_sync,
    calculate_impact_left_right,
    calculate_weight_left_right
)
import os
import logging

logging.basicConfig(level=logging.INFO)  # ログ出力用

from .coach_generator import generate_ai_menu

def analyze_video(file_path):
    BASE_DIR = os.path.dirname(__file__)
    success_path = os.path.join(BASE_DIR, "success.mp4")

    # landmark取得（元の3Dデータも保持）
    success_landmarks_3d = extract_pose_landmarks(success_path)
    target_landmarks_3d = extract_pose_landmarks(file_path)
    
    success_seq = normalize_pose(success_landmarks_3d)
    target_seq  = normalize_pose(target_landmarks_3d)

    if len(target_seq) == 0 or len(success_seq) == 0:
        # 解析できなかった場合
        menu_default = ["基本フォーム練習", "肩の回転練習", "体重移動練習"]
        return {
            "diagnosis": {
                "player": {"age": 13, "hand": "right", "serve_score": 0},
                "weakness": {
                    "impact_height": "ok", "weight_transfer": "ok", "body_open": "ok",
                    "shoulder_angle": "ok", "elbow_angle": "ok", "wrist_angle": "ok", "shoulder_tilt": "ok",
                    "waist_rotation": "ok", "waist_speed": "ok", "body_sway": "ok",
                    "impact_forward": "ok", "toss_sync": "ok", "impact_left_right": "ok", "weight_left_right": "ok"
                },
                "raw_values": {
                    "impact_diff": 0, "weight_diff": 0, "open_diff": 0,
                    "shoulder_angle_diff": 0, "elbow_angle_diff": 0, "wrist_angle_diff": 0, "shoulder_tilt_diff": 0,
                    "waist_rotation_diff": 0, "waist_speed_diff": 0, "body_sway_diff": 0,
                    "impact_forward_diff": 0, "toss_sync_diff": 0, "impact_left_right_diff": 0, "weight_left_right_diff": 0
                }
            },
            "menu": menu_default
        }

    frame_scores = []
    frame_diffs = []

    for t in target_seq:
        dists = np.linalg.norm(success_seq - t, axis=(1,2))
        min_idx = np.argmin(dists)
        frame_scores.append(dists[min_idx])
        frame_diffs.append(success_seq[min_idx] - t)

    frame_diffs = np.array(frame_diffs)
    mean_dist = np.mean(frame_scores)
    score = int(max(0, min(100, 100 - mean_dist * 28)))

    # 弱点評価（既存指標）
    impact_height   = safe_mean(frame_diffs, 0, 1)
    weight_transfer = safe_mean(frame_diffs, 1, 0)
    body_open       = safe_mean(frame_diffs, 2, 0)

    # 新しい角度指標の計算
    # 右利きを想定（player情報から取得できる場合は動的に変更可能）
    is_right_handed = True
    
    # 成功フォームと対象フォームの角度を計算（既に正規化済みの2D座標を使用）
    # 肩の開き角度（トップスイング～インパクト）
    success_shoulder_angles = calculate_shoulder_angle(success_seq, is_right_handed)
    target_shoulder_angles = calculate_shoulder_angle(target_seq, is_right_handed)
    
    # 肘の角度（トップスイング～インパクト）
    success_elbow_angles = calculate_elbow_angle(success_seq, is_right_handed)
    target_elbow_angles = calculate_elbow_angle(target_seq, is_right_handed)
    
    # 手首角度（インパクト時）
    success_wrist_angles = calculate_wrist_angle(success_seq, is_right_handed)
    target_wrist_angles = calculate_wrist_angle(target_seq, is_right_handed)
    
    # 肩の高さ差（トス→インパクト）
    success_shoulder_tilt = calculate_shoulder_tilt(success_seq)
    target_shoulder_tilt = calculate_shoulder_tilt(target_seq)
    
    # 腰の回転角度（トップスイング～インパクト）
    success_waist_rotations = calculate_waist_rotation(success_seq, is_right_handed)
    target_waist_rotations = calculate_waist_rotation(target_seq, is_right_handed)
    
    # 腰回転速度（トップ→インパクト）
    # フレームレートは30fpsを想定（実際の動画から取得可能な場合は動的に変更）
    fps = 30.0
    success_waist_speeds = calculate_waist_rotation_speed(success_waist_rotations, fps)
    target_waist_speeds = calculate_waist_rotation_speed(target_waist_rotations, fps)
    
    # 体軸ブレ量（インパクト時）
    success_body_sway = calculate_body_sway(success_seq)
    target_body_sway = calculate_body_sway(target_seq)
    
    # 打点高さ（インパクト時）
    success_impact_heights = calculate_impact_height(success_seq, is_right_handed)
    target_impact_heights = calculate_impact_height(target_seq, is_right_handed)
    
    # 打点前後位置（インパクト時）
    success_impact_forward = calculate_impact_forward(success_seq, is_right_handed)
    target_impact_forward = calculate_impact_forward(target_seq, is_right_handed)
    
    # トスとのタイミング差（トス→インパクト）
    success_toss_sync = calculate_toss_sync(success_seq, is_right_handed)
    target_toss_sync = calculate_toss_sync(target_seq, is_right_handed)
    
    # 打点左右偏差（インパクト時）
    success_impact_left_right = calculate_impact_left_right(success_seq, is_right_handed)
    target_impact_left_right = calculate_impact_left_right(target_seq, is_right_handed)
    
    # 体重左右分布（インパクト時）
    success_weight_left_right = calculate_weight_left_right(success_seq)
    target_weight_left_right = calculate_weight_left_right(target_seq)
    
    # インパクト時（後半30%のフレーム）とトップスイング時（前半30%のフレーム）を特定
    total_frames = len(target_seq)
    impact_start = int(total_frames * 0.7)  # 後半30%
    top_swing_end = int(total_frames * 0.3)  # 前半30%
    toss_end = int(total_frames * 0.2)  # トス時（最初の20%）
    
    # 肩の開き角度：トップスイングとインパクト時の平均
    if len(success_shoulder_angles) > 0 and len(target_shoulder_angles) > 0:
        success_top = np.mean(success_shoulder_angles[:top_swing_end]) if top_swing_end > 0 else np.mean(success_shoulder_angles)
        success_impact = np.mean(success_shoulder_angles[impact_start:]) if impact_start < len(success_shoulder_angles) else np.mean(success_shoulder_angles)
        success_shoulder_angle_avg = (success_top + success_impact) / 2
        
        target_top = np.mean(target_shoulder_angles[:top_swing_end]) if top_swing_end > 0 else np.mean(target_shoulder_angles)
        target_impact = np.mean(target_shoulder_angles[impact_start:]) if impact_start < len(target_shoulder_angles) else np.mean(target_shoulder_angles)
        target_shoulder_angle_avg = (target_top + target_impact) / 2
        
        shoulder_angle_diff = target_shoulder_angle_avg - success_shoulder_angle_avg
    else:
        shoulder_angle_diff = 0.0
    
    # 肘の角度：トップスイング～インパクト時の平均
    if len(success_elbow_angles) > 0 and len(target_elbow_angles) > 0:
        success_elbow_avg = np.mean(success_elbow_angles[top_swing_end:impact_start]) if impact_start > top_swing_end else np.mean(success_elbow_angles)
        target_elbow_avg = np.mean(target_elbow_angles[top_swing_end:impact_start]) if impact_start > top_swing_end else np.mean(target_elbow_angles)
        elbow_angle_diff = target_elbow_avg - success_elbow_avg
    else:
        elbow_angle_diff = 0.0
    
    # 手首角度：インパクト時の平均
    if len(success_wrist_angles) > 0 and len(target_wrist_angles) > 0:
        success_wrist_avg = np.mean(success_wrist_angles[impact_start:]) if impact_start < len(success_wrist_angles) else np.mean(success_wrist_angles)
        target_wrist_avg = np.mean(target_wrist_angles[impact_start:]) if impact_start < len(target_wrist_angles) else np.mean(target_wrist_angles)
        wrist_angle_diff = target_wrist_avg - success_wrist_avg
    else:
        wrist_angle_diff = 0.0
    
    # 肩の高さ差：トス→インパクト時の平均
    if len(success_shoulder_tilt) > 0 and len(target_shoulder_tilt) > 0:
        success_tilt_avg = np.mean(success_shoulder_tilt[impact_start:]) if impact_start < len(success_shoulder_tilt) else np.mean(success_shoulder_tilt)
        target_tilt_avg = np.mean(target_shoulder_tilt[impact_start:]) if impact_start < len(target_shoulder_tilt) else np.mean(target_shoulder_tilt)
        shoulder_tilt_diff = target_tilt_avg - success_tilt_avg
    else:
        shoulder_tilt_diff = 0.0
    
    # 腰の回転角度：トップスイング～インパクト時の平均
    if len(success_waist_rotations) > 0 and len(target_waist_rotations) > 0:
        success_waist_avg = np.mean(success_waist_rotations[top_swing_end:impact_start]) if impact_start > top_swing_end else np.mean(success_waist_rotations)
        target_waist_avg = np.mean(target_waist_rotations[top_swing_end:impact_start]) if impact_start > top_swing_end else np.mean(target_waist_rotations)
        waist_rotation_diff = target_waist_avg - success_waist_avg
    else:
        waist_rotation_diff = 0.0
    
    # 腰回転速度：トップ→インパクト時の平均
    if len(success_waist_speeds) > 0 and len(target_waist_speeds) > 0:
        # 速度はフレーム間の差分なので、インデックスを調整
        speed_impact_start = max(0, impact_start - 1) if impact_start > 0 else 0
        speed_top_end = max(0, top_swing_end - 1) if top_swing_end > 0 else 0
        
        if speed_impact_start > speed_top_end:
            success_speed_avg = np.mean(success_waist_speeds[speed_top_end:speed_impact_start]) if speed_impact_start <= len(success_waist_speeds) else np.mean(success_waist_speeds)
            target_speed_avg = np.mean(target_waist_speeds[speed_top_end:speed_impact_start]) if speed_impact_start <= len(target_waist_speeds) else np.mean(target_waist_speeds)
        else:
            success_speed_avg = np.mean(success_waist_speeds) if len(success_waist_speeds) > 0 else 0.0
            target_speed_avg = np.mean(target_waist_speeds) if len(target_waist_speeds) > 0 else 0.0
        
        waist_speed_diff = target_speed_avg - success_speed_avg
    else:
        waist_speed_diff = 0.0
    
    # 体軸ブレ量：インパクト時の平均
    if len(success_body_sway) > 0 and len(target_body_sway) > 0:
        success_sway_avg = np.mean(success_body_sway[impact_start:]) if impact_start < len(success_body_sway) else np.mean(success_body_sway)
        target_sway_avg = np.mean(target_body_sway[impact_start:]) if impact_start < len(target_body_sway) else np.mean(target_body_sway)
        body_sway_diff = target_sway_avg - success_sway_avg
    else:
        body_sway_diff = 0.0
    
    # 打点高さ：インパクト時の平均
    if len(success_impact_heights) > 0 and len(target_impact_heights) > 0:
        success_height_avg = np.mean(success_impact_heights[impact_start:]) if impact_start < len(success_impact_heights) else np.mean(success_impact_heights)
        target_height_avg = np.mean(target_impact_heights[impact_start:]) if impact_start < len(target_impact_heights) else np.mean(target_impact_heights)
        impact_height_new_diff = target_height_avg - success_height_avg
    else:
        impact_height_new_diff = 0.0
    
    # 打点前後位置：インパクト時の平均
    if len(success_impact_forward) > 0 and len(target_impact_forward) > 0:
        success_forward_avg = np.mean(success_impact_forward[impact_start:]) if impact_start < len(success_impact_forward) else np.mean(success_impact_forward)
        target_forward_avg = np.mean(target_impact_forward[impact_start:]) if impact_start < len(target_impact_forward) else np.mean(target_impact_forward)
        impact_forward_diff = target_forward_avg - success_forward_avg
    else:
        impact_forward_diff = 0.0
    
    # トスとのタイミング差：トス→インパクト時の平均
    if len(success_toss_sync) > 0 and len(target_toss_sync) > 0:
        # トス時（最初の20%）とインパクト時（後半30%）のタイミング差
        success_toss_avg = np.mean(success_toss_sync[:toss_end]) if toss_end > 0 else np.mean(success_toss_sync)
        success_impact_sync_avg = np.mean(success_toss_sync[impact_start:]) if impact_start < len(success_toss_sync) else np.mean(success_toss_sync)
        success_sync_diff = success_impact_sync_avg - success_toss_avg
        
        target_toss_avg = np.mean(target_toss_sync[:toss_end]) if toss_end > 0 else np.mean(target_toss_sync)
        target_impact_sync_avg = np.mean(target_toss_sync[impact_start:]) if impact_start < len(target_toss_sync) else np.mean(target_toss_sync)
        target_sync_diff = target_impact_sync_avg - target_toss_avg
        
        toss_sync_diff = target_sync_diff - success_sync_diff
    else:
        toss_sync_diff = 0.0
    
    # 打点左右偏差：インパクト時の平均
    if len(success_impact_left_right) > 0 and len(target_impact_left_right) > 0:
        success_left_right_avg = np.mean(success_impact_left_right[impact_start:]) if impact_start < len(success_impact_left_right) else np.mean(success_impact_left_right)
        target_left_right_avg = np.mean(target_impact_left_right[impact_start:]) if impact_start < len(target_impact_left_right) else np.mean(target_impact_left_right)
        impact_left_right_diff = target_left_right_avg - success_left_right_avg
    else:
        impact_left_right_diff = 0.0
    
    # 体重左右分布：インパクト時の平均
    if len(success_weight_left_right) > 0 and len(target_weight_left_right) > 0:
        success_weight_avg = np.mean(success_weight_left_right[impact_start:]) if impact_start < len(success_weight_left_right) else np.mean(success_weight_left_right)
        target_weight_avg = np.mean(target_weight_left_right[impact_start:]) if impact_start < len(target_weight_left_right) else np.mean(target_weight_left_right)
        weight_left_right_diff = target_weight_avg - success_weight_avg
    else:
        weight_left_right_diff = 0.0

    mean_diff_values = np.array([impact_height, weight_transfer, body_open])

    menu = generate_menu(
        mean_diff_values,
        shoulder_angle_diff=shoulder_angle_diff,
        elbow_angle_diff=elbow_angle_diff,
        wrist_angle_diff=wrist_angle_diff,
        shoulder_tilt_diff=shoulder_tilt_diff,
        waist_rotation_diff=waist_rotation_diff,
        waist_speed_diff=waist_speed_diff,
        body_sway_diff=body_sway_diff,
        impact_forward_diff=impact_forward_diff,
        toss_sync_diff=toss_sync_diff,
        impact_left_right_diff=impact_left_right_diff,
        weight_left_right_diff=weight_left_right_diff
    )

    # --- menu が空や None の場合は必ずデフォルト配列に置き換える ---
    if not isinstance(menu, list) or len(menu) == 0:
        logging.info("menu が空のためデフォルトメニューを使用します")
        menu = ["基本フォーム練習", "肩の回転練習", "体重移動練習"]

    logging.info(f"mean_diff_values: {mean_diff_values}")
    logging.info(f"generated menu: {menu}")

    diagnosis = {
        "player": {
            "age": 13,
            "hand": "right",
            "serve_score": score
        },
        "weakness": {
            "impact_height": "low" if impact_height < -0.15 else "ok",
            "weight_transfer": "poor" if weight_transfer < -0.15 else "ok",
            "body_open": "good" if body_open > -0.1 else "poor",
            "shoulder_angle": "too_open" if shoulder_angle_diff > 15 else ("too_closed" if shoulder_angle_diff < -15 else "ok"),
            "elbow_angle": "too_straight" if elbow_angle_diff > 20 else ("too_bent" if elbow_angle_diff < -20 else "ok"),
            "wrist_angle": "overextended" if wrist_angle_diff > 30 else ("too_flexed" if wrist_angle_diff < -30 else "ok"),
            "shoulder_tilt": "unbalanced" if abs(shoulder_tilt_diff) > 0.05 else "ok",
            "waist_rotation": "insufficient" if waist_rotation_diff < -20 else ("excessive" if waist_rotation_diff > 20 else "ok"),
            "waist_speed": "too_slow" if waist_speed_diff < -50 else ("ok" if waist_speed_diff < 50 else "ok"),
            "body_sway": "unstable" if body_sway_diff > 0.03 else "ok",
            "impact_forward": "too_back" if impact_forward_diff < -0.1 else ("too_forward" if impact_forward_diff > 0.1 else "ok"),
            "toss_sync": "out_of_sync" if abs(toss_sync_diff) > 0.2 else "ok",
            "impact_left_right": "unstable" if impact_left_right_diff > 0.05 else "ok",
            "weight_left_right": "unbalanced" if weight_left_right_diff > 0.03 else "ok"
        },
        "raw_values": {
            "impact_diff": round(float(impact_height), 3),
            "weight_diff": round(float(weight_transfer), 3),
            "open_diff": round(float(body_open), 3),
            "shoulder_angle_diff": round(float(shoulder_angle_diff), 2),
            "elbow_angle_diff": round(float(elbow_angle_diff), 2),
            "wrist_angle_diff": round(float(wrist_angle_diff), 2),
            "shoulder_tilt_diff": round(float(shoulder_tilt_diff), 4),
            "waist_rotation_diff": round(float(waist_rotation_diff), 2),
            "waist_speed_diff": round(float(waist_speed_diff), 2),
            "body_sway_diff": round(float(body_sway_diff), 4),
            "impact_forward_diff": round(float(impact_forward_diff), 4),
            "toss_sync_diff": round(float(toss_sync_diff), 4),
            "impact_left_right_diff": round(float(impact_left_right_diff), 4),
            "weight_left_right_diff": round(float(weight_left_right_diff), 4)
        }
    }
        # menu を必ずリストへ変換（防御的プログラミング）
    if not isinstance(menu, list):
        menu = [str(menu)] if menu is not None else []

    # menu を必ずリストに保証する
    if not isinstance(menu, list):
        menu = [str(menu)] if menu is not None else []
    ai_text = generate_ai_menu(diagnosis)
    return {
        "diagnosis": diagnosis,
        "menu": menu,
        "ai_text": ai_text
    }
