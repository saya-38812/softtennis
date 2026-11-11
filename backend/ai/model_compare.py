"""
お手本動画（success.mp4）と比較してサーブのフォームを採点するモジュール（Colab 用）

video_pose.py の analyze_video と同じ考え方で採点するが、
画像・動画ファイルを書き出さず、「結果の辞書」だけを返すようにしたもの。
（Colab では結果をスプレッドシートに書き込むだけなので、ファイル出力は不要）

採点の考え方：
  お手本の値を「90点」、明らかに悪い値を「0点」として、その間を直線でつないで点数にする。
  5つの指標のうち、一番点数が低いものを「今日の改善ポイント」にする。

使い方：
  from ai.model_compare import analyze_serve
  result = analyze_serve("clip.mp4")
"""
import logging
import os
from typing import Dict, Optional

import numpy as np

from .video_pose import (
    FOCUS_LABELS,
    _IDEAL_DEFAULTS,
    build_score_funcs,
    detect_contact_frame,
    detect_impact_frame,
    get_dynamic_advice,
)
from .video_pose_analyzer import extract_pose_landmarks
from .normalize_pose import normalize_pose
from .angle_utils import (
    calculate_body_sway,
    calculate_elbow_angle,
    calculate_impact_height,
    calculate_waist_rotation,
    calculate_waist_rotation_speed,
    calculate_weight_left_right,
)
from .coach_templates import get_coaching_from_template

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
DEFAULT_MODEL_VIDEO = os.path.join(BASE_DIR, "success.mp4")

# 骨格を取り出す範囲（インパクトの前後 何秒か）
# お手本と依頼者で範囲が違うと、同じフォームでも値が変わってしまうので、必ず同じ値を使う。
# （元の video_pose.py ではお手本1秒・依頼者2秒になっていて、お手本同士でも90点にならなかった）
POSE_RANGE_SEC = 2.0

# お手本の代表値は毎回計算すると遅いので、1回目の結果をここに覚えておく
_ideal_cache: Dict[str, Dict[str, float]] = {}


# ============================================================
# 指標の計算（お手本・依頼者の両方で同じ計算を使う）
# ============================================================

def _compute_metrics(norm_landmarks: np.ndarray) -> Optional[Dict[str, float]]:
    """
    骨格データ（インパクト前後のフレーム）から、5つの指標の代表値を計算する。

    お手本と依頼者で「同じ物差し」を使わないと比較にならないので、
    両方ともこの関数で計算する。
    骨格がうまく取れていなければ None を返す。
    """
    if norm_landmarks is None or len(norm_landmarks) == 0:
        return None
    seq = normalize_pose(norm_landmarks)
    n = len(seq)
    if n == 0:
        return None

    # インパクト（手首が一番高い瞬間）の前後だけを見る
    contact = min(detect_contact_frame(norm_landmarks), n - 1)
    window = max(3, n // 5)
    lo = max(0, contact - window)
    hi = min(n, contact + window + 1)

    impact_vals = calculate_impact_height(seq, True)
    elbow_vals = calculate_elbow_angle(seq, True)
    sway_vals = calculate_body_sway(seq)
    waist_speeds = calculate_waist_rotation_speed(calculate_waist_rotation(seq, True))
    weight_vals = calculate_weight_left_right(seq)

    # 打点の高さ：インパクト付近で手首が一番高い値（Y は上ほどマイナス）
    impact = float(np.min(impact_vals[lo:hi])) if len(impact_vals) > 0 else 0.0
    # 肘の角度：インパクト付近で一番伸びた値
    elbow = float(np.max(elbow_vals[lo:hi])) if len(elbow_vals) > 0 else 0.0
    # 体軸のブレ：インパクト付近の中央値（小さいほど安定）
    sway_slice = sway_vals[max(0, min(lo, len(sway_vals) - 1)):min(hi, len(sway_vals))]
    sway = float(np.median(sway_slice)) if len(sway_slice) > 0 else 0.0
    # 腰の回転速度：一番鋭く回った瞬間
    speed_slice = waist_speeds[max(0, lo - 1):min(len(waist_speeds), hi)]
    waist = float(np.max(speed_slice)) if len(speed_slice) > 0 else 0.0
    # 体重移動：前半と後半の差 + 最大値
    if len(weight_vals) > 1:
        half = n // 2
        weight = abs(float(np.mean(weight_vals[half:])) - float(np.mean(weight_vals[:half]))) + float(
            np.max(weight_vals)
        )
    else:
        weight = 0.0

    return {
        "impact_height": impact,
        "elbow_angle": elbow,
        "body_sway": sway,
        "waist_speed": waist,
        "weight_transfer": weight,
    }


def get_ideal_values(model_video: str = DEFAULT_MODEL_VIDEO) -> Dict[str, float]:
    """お手本動画の代表値を返す（2回目以降はキャッシュを使う）"""
    if model_video in _ideal_cache:
        return _ideal_cache[model_video]

    logger.info("お手本動画を解析中（初回のみ）: %s", model_video)
    impact_idx = detect_impact_frame(model_video)
    diag = extract_pose_landmarks(model_video, impact_idx, range_sec=POSE_RANGE_SEC)
    ideal = _compute_metrics(diag.get("norm"))
    if ideal is None:
        # お手本が解析できないことは普通ないが、念のため実測済みの固定値を使う
        logger.warning("お手本動画を解析できなかったので、既定値を使います")
        ideal = dict(_IDEAL_DEFAULTS)

    _ideal_cache[model_video] = ideal
    return ideal


# ============================================================
# メイン：サーブ1本を採点する
# ============================================================

def analyze_serve(video_path: str, model_video: str = DEFAULT_MODEL_VIDEO) -> dict:
    """
    サーブ1本分の動画（クリップ）を、お手本と比較して採点する。

    戻り値（成功時）：
      {
        "status": "ok",
        "overall_score": 72,                 # 5指標の平均（0〜100）
        "scores": {"elbow_angle": 85.2, ...},  # 指標ごとの点数（0〜100、お手本=90点）
        "raw": {"elbow_angle": 168.3, ...},    # 指標の実測値（デバッグ・学習用）
        "ideal": {"elbow_angle": 175.0, ...},  # お手本の実測値
        "focus_key": "impact_height",
        "focus_label": "打点の高さ",
        "ai_text": "トスをもう少し高く上げて…",  # 改善ポイントへのアドバイス
        "practice": "トスだけを10回連続で…",    # おすすめの練習
        "comparison": {...}                    # 画面で「お手本 vs あなた」を出すための値
      }
    失敗時は {"status": "error", "message": "..."}
    """
    ideal = get_ideal_values(model_video)
    score_funcs = build_score_funcs(ideal)

    # ---- 1. 骨格を抽出（インパクト前後 ±POSE_RANGE_SEC 秒） ----
    try:
        impact_idx = detect_impact_frame(video_path)
    except Exception as e:  # 動画が壊れているなど
        logger.warning("インパクト検出に失敗: %s", e)
        impact_idx = 0
    diag = extract_pose_landmarks(video_path, impact_idx, range_sec=POSE_RANGE_SEC)
    raw = _compute_metrics(diag.get("norm"))
    if raw is None:
        return {
            "status": "error",
            "message": "動画から骨格を検出できませんでした。全身が映るように、少し離れて撮影してください。",
        }

    # ---- 2. 0〜100点に換算 ----
    scores = {}
    for key, val in raw.items():
        score_input = abs(val) if key == "impact_height" else val
        scores[key] = score_funcs[key](score_input)

    # ---- 3. 2D の骨格では正しく測れない指標の補正（video_pose.py と同じルール） ----
    # 腰の回転：カメラの向きによっては回転が映らない。お手本の1/20未満なら「測れなかった」とみなす
    if raw["waist_speed"] < ideal["waist_speed"] / 20:
        scores["waist_speed"] = 60.0  # 中立の点数（改善ポイントに選ばれないようにする）
    # 体軸のブレ：骨格検出の細かい揺れで悪く見えやすいので、下限を25点にする
    scores["body_sway"] = max(scores["body_sway"], 25.0)
    # 体重移動：お手本の1/5未満なら測れなかったとみなす
    if raw["weight_transfer"] < ideal["weight_transfer"] / 5:
        scores["weight_transfer"] = 60.0

    # ---- 4. 一番点数が低い指標を改善ポイントにする ----
    focus = min(scores, key=scores.get)
    label_ideal, label_user, value_ideal, value_user, advice = get_dynamic_advice(
        focus, raw[focus], scores[focus]
    )
    coaching = get_coaching_from_template(focus)

    return {
        "status": "ok",
        "overall_score": int(round(sum(scores.values()) / len(scores))),
        "scores": {k: round(v, 1) for k, v in scores.items()},
        "raw": {k: round(v, 4) for k, v in raw.items()},
        "ideal": {k: round(v, 4) for k, v in ideal.items()},
        "focus_key": focus,
        "focus_label": FOCUS_LABELS[focus],
        "ai_text": advice,
        "cue": coaching.get("ai_text", ""),  # 身体感覚のイメージ（例：ボールを空に押し上げる）
        "practice": coaching.get("practice", ""),
        "comparison": {
            "label_ideal": label_ideal,
            "label_user": label_user,
            "value_ideal": value_ideal,
            "value_user": value_user,
        },
    }
