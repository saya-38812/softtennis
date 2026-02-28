"""
MVP用サーブ解析モジュール。
既存の video_pose / normalize_pose / angle_utils を利用し、
スコア・フィードバック・メトリクスのみを返す軽量API用。
既存の解析機能（analyze_video, POST /analyze）は変更しない。
"""
import logging
import numpy as np

from .video_pose import detect_impact_frame
from .video_pose_analyzer import extract_pose_landmarks
from .normalize_pose import normalize_pose
from .angle_utils import (
    calculate_elbow_angle,
    calculate_body_sway,
    calculate_impact_height,
)

logger = logging.getLogger(__name__)

# MVP スコア用しきい値
ELBOW_IDEAL_MIN = 90.0
ELBOW_IDEAL_MAX = 130.0
IMPACT_HEIGHT_THRESHOLD = -0.35   # 正規化Y: これより大きいと「トスが低い」
BODY_SWAY_THRESHOLD = 0.12       # これより大きいと「体が早く開く」


def analyze_video(video_path: str) -> dict:
    """
    動画を解析し、MVP用のスコア・フィードバック・メトリクスを返す。
    既存の detect_impact_frame / extract_pose_landmarks / normalize_pose / angle_utils を使用。
    """
    # 1. インパクトフレーム検出
    try:
        impact_index = detect_impact_frame(video_path)
    except Exception as e:
        logger.warning(f"detect_impact_frame failed: {e}")
        impact_index = 0

    # 2. インパクト前後の骨格抽出（既存ロジック）
    diag = extract_pose_landmarks(video_path, impact_index, range_sec=1.0)
    norm = diag.get("norm")
    if norm is None or len(norm) == 0:
        return _error_result("動画から骨格を検出できませんでした。")

    # 3. 正規化
    try:
        seq = normalize_pose(norm)
    except Exception as e:
        logger.warning(f"normalize_pose failed: {e}")
        return _error_result("ポーズの正規化に失敗しました。")

    if len(seq) == 0:
        return _error_result("有効なポーズフレームがありませんでした。")

    # 4. メトリクス計算（既存関数を利用）
    elbow_vals = calculate_elbow_angle(seq, is_right_handed=True)
    sway_vals = calculate_body_sway(seq)
    impact_heights = calculate_impact_height(seq, is_right_handed=True)

    n = len(seq)
    # インパクト付近のウィンドウで代表値を取得
    window = max(2, n // 5)
    mid = n // 2
    lo = max(0, mid - window)
    hi = min(n, mid + window + 1)

    if len(elbow_vals) > 0:
        elbow_angle = float(np.median(elbow_vals[lo:hi])) if hi > lo else float(np.median(elbow_vals))
    else:
        elbow_angle = 0.0

    if len(sway_vals) > 0:
        sway_slice = sway_vals[max(0, lo):min(hi, len(sway_vals))]
        body_sway = float(np.median(sway_slice)) if len(sway_slice) > 0 else 0.0
    else:
        body_sway = 0.0

    if len(impact_heights) > 0:
        # 手首Yの中央値（正規化座標）。表示用は「高さ」として 1.0 + (-y) で 1.x に
        ih_median = float(np.median(impact_heights[lo:hi])) if hi > lo else float(np.median(impact_heights))
        impact_height = round(1.0 + (-ih_median), 1)  # 高いほど良いので 1.0〜2.0 程度
    else:
        impact_height = 0.0
        ih_median = 0.0

    # 5. スコア（100点満点、ペナルティ方式）
    score = 100
    feedback = []

    if ih_median > IMPACT_HEIGHT_THRESHOLD:
        score -= 10
        feedback.append("Toss is too low")

    if elbow_angle < ELBOW_IDEAL_MIN or elbow_angle > ELBOW_IDEAL_MAX:
        score -= 10
        feedback.append("Elbow angle outside ideal range (90°–130°)")

    if body_sway > BODY_SWAY_THRESHOLD:
        score -= 10
        feedback.append("Body opens too early")

    # 日本語フィードバック（既存UIとの整合）
    if not feedback:
        feedback.append("Form is within target ranges.")

    score = max(0, min(100, score))

    return {
        "score": int(score),
        "feedback": feedback,
        "metrics": {
            "elbow_angle": round(elbow_angle, 1),
            "body_sway": round(body_sway, 2),
            "impact_height": impact_height,
        },
    }


def _error_result(message: str) -> dict:
    return {
        "score": 0,
        "feedback": [message],
        "metrics": {
            "elbow_angle": 0.0,
            "body_sway": 0.0,
            "impact_height": 0.0,
        },
    }
