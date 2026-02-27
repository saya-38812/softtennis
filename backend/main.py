from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import shutil
import os
import traceback
import uuid
import json
import asyncio
import queue
import threading
import logging
import numpy as np
from dotenv import load_dotenv

from ai import video_pose as _vp
from ai.video_pose import analyze_video, FOCUS_LABELS, FOCUS_LANDMARK
from ai.coach_generator import generate_menu_detail, generate_coaching_message
from player_store import load_last_score, save_score, clear_player_data
from session_store import append_score, load_scores, clear_session, clear_all_sessions

# ============================
# 環境変数読み込み
# ============================
load_dotenv()

# ============================
# FastAPI起動
# ============================
app = FastAPI()

# ============================
# ヘルスチェック（Render / ロードバランサ用）
# ============================
@app.get("/")
async def health_check():
    return {"status": "ok"}

@app.head("/")
async def health_check_head():
    return Response(status_code=200)

# ============================
# 起動時にお手本データをバックグラウンドで事前解析
# サーバーの応答開始を妨げずに、初回リクエスト高速化
# ============================
_warmup_done = threading.Event()

def _warmup_cache():
    """お手本動画を事前に解析してキャッシュに載せる"""
    try:
        logging.info("Warmup: お手本データの事前解析を開始...")
        from ai.video_pose import SUCCESS_CACHE, detect_impact_frame, detect_contact_frame, build_score_funcs
        from ai.video_pose import SCORE_FUNCS as _sf
        from ai.video_pose_analyzer import extract_pose_landmarks
        from ai.normalize_pose import normalize_pose
        from ai.angle_utils import (
            calculate_elbow_angle, calculate_body_sway,
            calculate_impact_height, calculate_waist_rotation,
            calculate_waist_rotation_speed, calculate_weight_left_right,
        )

        if "success" in SUCCESS_CACHE:
            logging.info("Warmup: キャッシュ済み、スキップ")
            return

        success_path = os.path.join(os.path.dirname(__file__), "ai", "success.mp4")
        if not os.path.exists(success_path):
            logging.warning(f"Warmup: お手本動画が見つかりません: {success_path}")
            return

        import ai.video_pose as vp_mod
        if "success" not in vp_mod.SUCCESS_CACHE:
            s_impact_idx = vp_mod.detect_impact_frame(success_path)
            s_diag = extract_pose_landmarks(success_path, s_impact_idx, range_sec=1.0)
            import cv2 as cv2_warmup
            cap_s = cv2_warmup.VideoCapture(success_path)
            cap_s.set(cv2_warmup.CAP_PROP_POS_FRAMES, s_impact_idx)
            ret, s_img_orig = cap_s.read()
            cap_s.release()

            s_norm = s_diag["norm"]
            s_seq = normalize_pose(s_norm)
            s_contact = detect_contact_frame(s_norm)
            s_n = len(s_seq)
            s_impact_local = min(s_contact, s_n - 1)
            s_win = max(3, s_n // 5)
            s_lo = max(0, s_impact_local - s_win)
            s_hi = min(s_n, s_impact_local + s_win + 1)

            s_impact_vals = calculate_impact_height(s_seq, True)
            s_elbow_vals = calculate_elbow_angle(s_seq, True)
            s_sway_vals = calculate_body_sway(s_seq)
            s_waist_rots = calculate_waist_rotation(s_seq, True)
            s_waist_spds = calculate_waist_rotation_speed(s_waist_rots)
            s_weight_vals = calculate_weight_left_right(s_seq)

            s_iv = float(np.min(s_impact_vals[s_lo:s_hi])) if s_hi > s_lo and len(s_impact_vals) > 0 else -0.5
            s_ev = float(np.max(s_elbow_vals[s_lo:s_hi])) if s_hi > s_lo and len(s_elbow_vals) > 0 else 170.0
            s_sway_lo = max(0, min(s_lo, len(s_sway_vals) - 1))
            s_sway_hi = min(s_hi, len(s_sway_vals))
            s_sway_slice = s_sway_vals[s_sway_lo:s_sway_hi] if s_sway_hi > s_sway_lo else s_sway_vals
            s_sv = float(np.mean(s_sway_slice)) if len(s_sway_slice) > 0 else 0.05
            sp_lo = max(0, s_lo - 1)
            sp_hi = min(len(s_waist_spds), s_hi)
            s_wv = float(np.max(s_waist_spds[sp_lo:sp_hi])) if sp_hi > sp_lo and len(s_waist_spds) > 0 else 4800.0
            if len(s_weight_vals) > 1:
                s_half = s_n // 2
                s_wt = abs(float(np.mean(s_weight_vals[:s_half])) - float(np.mean(s_weight_vals[s_half:]))) + float(np.max(s_weight_vals))
            else:
                s_wt = 0.15

            ideal_representative = {
                "impact_height": s_iv, "elbow_angle": s_ev,
                "body_sway": s_sv, "waist_speed": s_wv, "weight_transfer": s_wt,
            }

            vp_mod.SCORE_FUNCS = build_score_funcs(ideal_representative)
            vp_mod.SUCCESS_CACHE["success"] = {
                "impact_index": s_impact_idx, "diag": s_diag,
                "norm": s_norm, "pixel": s_diag["pixel"],
                "seq": s_seq, "img_orig": s_img_orig,
                "ideal_vals": ideal_representative,
            }
            logging.info("Warmup: お手本データの事前解析完了")
    except Exception as e:
        logging.warning(f"Warmup failed (non-critical): {e}")
    finally:
        _warmup_done.set()

@app.on_event("startup")
async def startup_event():
    t = threading.Thread(target=_warmup_cache, daemon=True)
    t.start()

# ============================
# CORS（フロント許可）
# ============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-ID"],
)

# ============================
# ディレクトリ準備 (Reload監視を避けるため、backend配下から外す)
# ============================
BASE_DIR = os.path.dirname(__file__)
DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "server_storage"))

UPLOAD_DIR = os.path.join(DATA_ROOT, "uploads")
OUTPUT_DIR = os.path.join(DATA_ROOT, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# 動画ストリーミング（Range リクエスト対応）
# ============================
@app.get("/outputs/{filename:path}")
async def serve_output_file(filename: str, request: Request):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    file_size = os.path.getsize(file_path)
    ext = os.path.splitext(filename)[1].lower()
    content_type = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(ext, "application/octet-stream")

    range_header = request.headers.get("range")

    if range_header and ext == ".mp4":
        range_spec = range_header.replace("bytes=", "")
        parts = range_spec.split("-")
        start = int(parts[0])
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)
        content_length = end - start + 1

        def iter_chunk():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(1024 * 1024, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_chunk(),
            status_code=206,
            media_type=content_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
            },
        )

    def iter_file():
        with open(file_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                yield chunk

    return StreamingResponse(
        iter_file(),
        media_type=content_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        },
    )

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
        "waist_speed": current.get("waist_speed", 0.0) - last.get("waist_speed", 0.0),
        "weight_transfer": current.get("weight_transfer", 0.0) - last.get("weight_transfer", 0.0),
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
    
    elif best_key == "waist_speed":
        return f"前回より腰のキレが改善しています"
    
    elif best_key == "weight_transfer":
        return f"前回より体重移動がスムーズになっています"
    
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
    セッションのスコアから最終診断を計算（SCORE_FUNCS で統一的に 0-100 正規化）
    """
    if not scores:
        return None

    keys = ["impact_height", "elbow_angle", "body_sway", "waist_speed", "weight_transfer"]
    raw_lists = {k: [s.get(k, 0.0) for s in scores] for k in keys}

    # 外れ値除去（IQR）
    clean = {}
    for k in keys:
        cleaned = remove_outliers_iqr(raw_lists[k])
        clean[k] = cleaned if cleaned else raw_lists[k]

    mean_scores = {k: float(np.mean(clean[k])) for k in keys}

    # 統一スコアリング（0-100）— モジュール変数を参照し、お手本解析後の最新関数を使う
    normalized = {}
    for k in keys:
        score_input = abs(mean_scores[k]) if k == "impact_height" else mean_scores[k]
        normalized[k] = _vp.SCORE_FUNCS[k](score_input)
    weakness = {k: ("ok" if v >= 60 else "low") for k, v in normalized.items()}
    focus = min(normalized, key=normalized.get)

    weakness_label_map = {
        "impact_height": "打点が低い",
        "elbow_angle": "肘が曲がりすぎている",
        "body_sway": "体軸がブレている",
        "waist_speed": "腰の回転が遅い",
        "weight_transfer": "体重移動が小さい",
    }
    weakness_label = weakness_label_map.get(focus, "打点が低い")
    coaching = generate_coaching_message(weakness_label)

    return {
        "diagnosis": {
            "weakness": weakness,
            "scores": mean_scores,
            "normalized_scores": {k: round(v, 1) for k, v in normalized.items()},
        },
        "ai_text": coaching["ai_text"],
        "practice": coaching["practice"],
        "focus_label": FOCUS_LABELS[focus],
        "mean_scores": mean_scores,
        "normalized_scores": {k: round(v, 1) for k, v in normalized.items()},
    }

# ============================
# 動画解析API（SSE で進捗を送信）
# ============================
@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...)):

    path = os.path.join(UPLOAD_DIR, file.filename)

    session_id = request.headers.get("X-Session-ID")
    if not session_id:
        session_id = str(uuid.uuid4())

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    progress_q: queue.Queue = queue.Queue()

    def _run_analysis():
        """バックグラウンドスレッドで重い解析を実行"""
        try:
            def on_progress(pct: int):
                progress_q.put(("progress", pct))

            result = analyze_video(path, progress_cb=on_progress)
            progress_q.put(("done", result))
        except Exception as e:
            traceback.print_exc()
            progress_q.put(("error", str(e)))

    thread = threading.Thread(target=_run_analysis, daemon=True)
    thread.start()

    async def event_stream():
        last_pct = -1
        try:
            while True:
                try:
                    msg_type, payload = progress_q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.3)
                    continue

                if msg_type == "progress":
                    pct = payload
                    if pct != last_pct:
                        last_pct = pct
                        yield f"data: {json.dumps({'type': 'progress', 'percent': pct})}\n\n"

                elif msg_type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'message': str(payload)})}\n\n"
                    break

                elif msg_type == "done":
                    result = payload

                    if result.get("status") == "error" or result.get("ai_text") == "解析できませんでした":
                        yield f"data: {json.dumps({'type': 'error', 'message': result.get('ai_text', '動画の解析に失敗しました。')})}\n\n"
                        break

                    diagnosis = result.get("diagnosis", {})
                    scores = diagnosis.get("scores", {})

                    current_score = {
                        "impact_height": float(scores.get("impact_height", 0.0)),
                        "elbow_angle": float(scores.get("elbow_angle", 0.0)),
                        "body_sway": float(scores.get("body_sway", 0.0)),
                        "waist_speed": float(scores.get("waist_speed", 0.0)),
                        "weight_transfer": float(scores.get("weight_transfer", 0.0)),
                    }

                    append_score(session_id, current_score)
                    all_scores = load_scores(session_id)
                    count = len(all_scores)

                    session_result = compute_session_result(all_scores)
                    session_result["user_video"] = result.get("user_video", "")
                    session_result["user_image"] = result.get("user_image", "")
                    session_result["ideal_image"] = result.get("ideal_image", "")
                    session_result["status"] = "complete"

                    last_score_data = load_last_score()
                    mean_scores = session_result["mean_scores"]

                    if last_score_data is None:
                        improvement = None
                        improvement_message = "基準を作りました"
                    else:
                        improvement = calc_improvement(last_score_data, mean_scores)
                        improvement_message = generate_improvement_message(improvement)

                    save_score(mean_scores)

                    response_data = {
                        "type": "result",
                        "status": "complete",
                        "session_id": session_id,
                        "scores": mean_scores,
                        "normalized_scores": session_result.get("normalized_scores", {}),
                        "improvement": improvement,
                        "improvement_message": improvement_message,
                        "ai_text": session_result["ai_text"],
                        "practice": session_result["practice"],
                        "user_image": result.get("user_image", ""),
                        "user_video": result.get("user_video", ""),
                        "ideal_image": result.get("ideal_image", ""),
                        "focus_label": session_result["focus_label"],
                        "comparison": result.get("comparison", {}),
                        "count": count,
                    }

                    yield f"data: {json.dumps(response_data)}\n\n"
                    break
        finally:
            if os.path.exists(path):
                os.remove(path)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "X-Session-ID": session_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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
# ============================
# データリセットAPI
# ============================
@app.post("/reset-all")
async def reset_all():
    """
    全てのセッションデータ、プレイヤー履歴、アップロード動画、解析結果を削除する
    """
    try:
        # セッションとプレイヤーデータ
        clear_all_sessions()
        clear_player_data()
        
        # アップロード・出力ディレクトリの掃除
        for d in [UPLOAD_DIR, OUTPUT_DIR]:
            if os.path.exists(d):
                import shutil
                shutil.rmtree(d)
                os.makedirs(d, exist_ok=True)
                
        return {"status": "ok", "message": "全てのデータをリセットしました"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
