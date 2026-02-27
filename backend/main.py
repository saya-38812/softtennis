from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import shutil
import os
import traceback
import uuid
import json
import numpy as np
from dotenv import load_dotenv

from ai.video_pose import analyze_video, FOCUS_LABELS, FOCUS_MESSAGES, FOCUS_LANDMARK
from ai.coach_generator import generate_menu_detail, generate_coaching_message
from player_store import load_last_score, save_score
from session_store import append_score, load_scores, clear_session

# ============================
# 環境変数読み込み
# ============================
load_dotenv()

# ============================
# FastAPI起動
# ============================
app = FastAPI()

# ============================
# CORS（フロント許可）
# ============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# ディレクトリ準備
# ============================
BASE_DIR = os.path.dirname(__file__)

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# 静的ファイル（画像）
# ============================
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# ============================
# リクエストモデル
# ============================
class MenuDetailRequest(BaseModel):
    menu_name: str
    diagnosis: dict

# ============================
# improvement計算関数
# ============================
def calc_improvement(last: dict, current: dict):
    """
    前回と今回のスコアの差分を計算
    
    Args:
        last: 前回のスコア（impact_height, elbow_angle, body_sway）
        current: 今回のスコア（impact_height, elbow_angle, body_sway）
    
    Returns:
        dict: 差分（current - last）
    """
    return {
        "impact_height": current["impact_height"] - last["impact_height"],
        "elbow_angle": current["elbow_angle"] - last["elbow_angle"],
        "body_sway": current["body_sway"] - last["body_sway"],
    }


def generate_improvement_message(improvement: dict):
    """
    改善メッセージを生成
    
    Args:
        improvement: 差分（impact_height, elbow_angle, body_sway）
    
    Returns:
        str: 改善メッセージ
    """
    # 各項目の改善度を計算
    # impact_height: +が改善（高くなっている）
    # elbow_angle: +が改善（大きくなっている）
    # body_sway: -が改善（小さくなっている）
    
    improvements = {}
    
    # impact_height: +が改善
    if improvement["impact_height"] > 0:
        improvements["impact_height"] = improvement["impact_height"]
    
    # elbow_angle: +が改善
    if improvement["elbow_angle"] > 0:
        improvements["elbow_angle"] = improvement["elbow_angle"]
    
    # body_sway: -が改善（絶対値で評価）
    if improvement["body_sway"] < 0:
        improvements["body_sway"] = abs(improvement["body_sway"])
    
    # 最も改善した項目を選ぶ
    if not improvements:
        return "前回と比較して大きな変化はありません"
    
    best_key = max(improvements, key=improvements.get)
    best_value = improvements[best_key]
    
    # メッセージ生成
    if best_key == "impact_height":
        # %表示（前回値に対する改善率を計算するため、絶対値で%表示）
        percent = int(best_value * 100)
        return f"前回より打点が{percent}%高くなっています"
    
    elif best_key == "elbow_angle":
        # 数値表示
        return f"前回より肘の角度が{best_value:.1f}度改善しています"
    
    elif best_key == "body_sway":
        # 数値表示
        return f"前回より体軸のブレが{best_value:.2f}改善しています"
    
    return "前回と比較して改善が見られます"


# ============================
# 統計処理関数
# ============================
def remove_outliers_iqr(values):
    """
    IQRで外れ値を除去（1回のみ）
    
    Args:
        values: 数値のリスト
    
    Returns:
        list: 外れ値を除去したリスト
    """
    if len(values) < 4:
        return values
    
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    return [v for v in values if lower_bound <= v <= upper_bound]


def compute_session_result(scores: list):
    """
    セッションのスコアから最終診断を計算
    
    Args:
        scores: スコアのリスト（各要素は impact_height, elbow_angle, body_sway を持つ）
    
    Returns:
        dict: 診断結果（diagnosis, menu, ai_text, ideal_image, user_image, focus_label, message）
    """
    if not scores:
        return None
    
    # 各指標を抽出
    impact_heights = [s["impact_height"] for s in scores]
    elbow_angles = [s["elbow_angle"] for s in scores]
    body_sways = [s["body_sway"] for s in scores]
    
    # 外れ値除去（IQRで1回）
    impact_heights_clean = remove_outliers_iqr(impact_heights)
    elbow_angles_clean = remove_outliers_iqr(elbow_angles)
    body_sways_clean = remove_outliers_iqr(body_sways)
    
    # 平均値計算
    mean_impact = np.mean(impact_heights_clean) if impact_heights_clean else np.mean(impact_heights)
    mean_elbow = np.mean(elbow_angles_clean) if elbow_angles_clean else np.mean(elbow_angles)
    mean_sway = np.mean(body_sways_clean) if body_sways_clean else np.mean(body_sways)
    
    # 弱点判定（既存ルール使用）
    weakness = {
        "impact_height": "low" if mean_impact < -0.15 else "ok",
        "elbow_angle": "too_bent" if mean_elbow < -20 else "ok",
        "body_sway": "unstable" if mean_sway > 0.03 else "ok",
    }
    
    # 一番悪い指標を選ぶ（既存ルール使用）
    abs_scores = {
        "impact_height": abs(mean_impact),
        "elbow_angle": abs(mean_elbow),
        "body_sway": abs(mean_sway),
    }
    
    focus = max(abs_scores, key=abs_scores.get)
    
    # 最終スコア（平均値）
    final_scores = {
        "impact_height": abs(mean_impact),
        "elbow_angle": abs(mean_elbow),
        "body_sway": abs(mean_sway),
    }
    
    # 弱点ラベルを生成（focus に基づく）
    weakness_label_map = {
        "impact_height": "打点が低い",
        "elbow_angle": "肘が曲がりすぎている",
        "body_sway": "体軸がブレている",
    }
    
    weakness_label = weakness_label_map.get(focus, "打点が低い")
    
    # 身体感覚コーチング生成
    coaching = generate_coaching_message(weakness_label)
    
    # 最後の動画の画像を使用（簡易実装）
    # 実際には最後の動画の解析結果から画像を取得する必要があるが、
    # ここでは既存のanalyze_videoの結果を使用する想定
    # 画像は後で設定される
    
    return {
        "diagnosis": {
            "weakness": weakness,
            "scores": final_scores,
        },
        "ai_text": coaching["ai_text"],
        "practice": coaching["practice"],
        "focus_label": FOCUS_LABELS[focus],
        "mean_scores": {
            "impact_height": mean_impact,
            "elbow_angle": mean_elbow,
            "body_sway": mean_sway,
        },
    }

# ============================
# 動画解析API
# ============================
@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...)):

    path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # -------------------------
        # session id
        # -------------------------
        session_id = request.headers.get("X-Session-ID")
        if not session_id:
            session_id = str(uuid.uuid4())

        # -------------------------
        # 動画解析
        # -------------------------
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = analyze_video(path)
        diagnosis = result.get("diagnosis", {})
        scores = diagnosis.get("scores", {})

        current_score = {
            "impact_height": float(scores.get("impact_height", 0.0)),
            "elbow_angle": float(scores.get("elbow_angle", 0.0)),
            "body_sway": float(scores.get("body_sway", 0.0)),
        }

        # -------------------------
        # 保存
        # -------------------------
        append_score(session_id, current_score)
        all_scores = load_scores(session_id)
        count = len(all_scores)

        # -------------------------
        # 3本未満：収集中
        # -------------------------
        if count < 3:
            return Response(
                content=json.dumps({
                    "status": "collecting",
                    "count": count
                }),
                media_type="application/json",
                headers={"X-Session-ID": session_id}
            )

        # -------------------------
        # 平均算出（リアルタイム）
        # -------------------------
        mean_scores = {
            key: sum(s[key] for s in all_scores) / len(all_scores)
            for key in current_score.keys()
        }

        # -------------------------
        # 改善計算
        # -------------------------
        last_score = load_last_score()

        if last_score is None:
            improvement = None
            improvement_message = "基準を作りました"
        else:
            improvement = calc_improvement(last_score, mean_scores)
            improvement_message = generate_improvement_message(improvement)

        save_score(mean_scores)

        # -------------------------
        # レスポンス
        # -------------------------
        response_data = {
            "status": "complete",
            "scores": mean_scores,
            "improvement": improvement,
            "improvement_message": improvement_message,
            "user_image": result.get("user_image", ""),
            "ideal_image": result.get("ideal_image", ""),
        }

        return Response(
            content=json.dumps(response_data),
            media_type="application/json",
            headers={"X-Session-ID": session_id}
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(path):
            os.remove(path)


# ============================
# 練習メニュー詳細API
# ============================
@app.post("/menu-detail")
async def get_menu_detail(request: MenuDetailRequest):

    try:
        detail = generate_menu_detail(request.menu_name, request.diagnosis)
        return {"status": "ok", "detail": detail}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"詳細生成エラー: {str(e)}"
        )

# ============================
# API専用構成
# ============================
# React（フロントエンド）はVercelで配信するため、ここでの build 配信は不要になりました。
