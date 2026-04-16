"""
MVP: ソフトテニス サーブ練習アプリ API
- Practice Session / Serve Tracking / Session Result / Form Analysis
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import os
import json
import uuid
import logging
from typing import Optional

from tracker_store import (
    start_session as store_start_session,
    append_serve,
    undo_last_serve,
    get_session as store_get_session,
    list_sessions as store_list_sessions,
    delete_session as store_delete_session,
    save_session_video,
    save_serve_analysis,
    set_session_tracking_video,
    set_session_source,
    get_session_video_path,
    get_outputs_dir,
    ensure_sample_sessions,
)
from ai.serve_analysis import analyze_video as analyze_clip
from ai.coach_templates import get_coaching_from_template
from ai.clip_extract import extract_clip
from ai.video_pose_analyzer import extract_pose_landmarks
from ai.video_renderer import render_analyzed_video
from ai.video_pose import detect_impact_frame

# ============================
# WebM → MP4 変換（Render など OpenCV が WebM を読めない環境向け）
# imageio-ffmpeg にバンドルされた FFmpeg を使う
# ============================
def _convert_webm_to_mp4(webm_path: str) -> str:
    """
    WebM ファイルを MP4 に変換して返す。
    変換に失敗した場合は元のパスをそのまま返す。
    """
    try:
        import subprocess
        import imageio_ffmpeg
        mp4_path = webm_path.replace(".webm", ".mp4")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        result = subprocess.run(
            [
                ffmpeg_exe, "-y",
                "-i", webm_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-an",  # 音声なし（サーブ練習なので不要）
                mp4_path,
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode == 0 and os.path.isfile(mp4_path):
            logging.info(f"WebM→MP4変換成功: {mp4_path}")
            return mp4_path
        else:
            logging.warning(f"WebM→MP4変換失敗: {result.stderr.decode(errors='replace')}")
            return webm_path
    except Exception as e:
        logging.warning(f"WebM→MP4変換エラー: {e}")
        return webm_path


# ============================
# FastAPI
# ============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 起動時にルートを表示（404 確認用）
    for r in app.routes:
        if hasattr(r, "path") and hasattr(r, "methods") and "/api" in r.path:
            print(f"  {list(r.methods)} {r.path}")
    # ローカル確認用: セッションが0件ならサンプルを自動作成
    try:
        existing = store_list_sessions()
        if not existing:
            created = ensure_sample_sessions()
            if created:
                print(f"  [seed] Sample sessions created: {created}")
    except Exception as e:
        logging.warning(f"Sample session seed skipped: {e}")
    yield

app = FastAPI(title="サーブノート API", description="サーブ練習・フォーム解析", lifespan=lifespan)

@app.get("/")
async def health_check():
    return {"status": "ok"}


@app.get("/api/health")
async def api_health():
    """API が正しく動いているか確認。404 のときはバックエンド再起動を。"""
    return {"status": "ok", "api": "sessions/start, serves, analysis"}


@app.post("/seed-sample")
@app.post("/api/seed-sample")
async def api_seed_sample():
    """サンプルセッションを追加（録画なしで結果画面を確認する用）。既にある場合は何もしない。"""
    created = ensure_sample_sessions()
    return {"created": created, "message": "サンプルデータを追加しました" if created else "サンプルは既にあります"}


# ============================
# CORS
# ============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# 出力ディレクトリ（data/outputs）
# ============================
OUTPUT_DIR = get_outputs_dir()

# ============================
# 動画ストリーミング（Range 対応）
# ============================
@app.get("/outputs/{filename:path}")
async def serve_output_file(filename: str, request: Request):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    file_size = os.path.getsize(file_path)
    ext = os.path.splitext(filename)[1].lower()
    content_type = {
        ".mp4": "video/mp4", ".webm": "video/webm",
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
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
        headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
    )

# ============================
# リクエストモデル
# ============================
class ServeRecordRequest(BaseModel):
    session_id: str
    result: str  # "IN" | "OUT" | "FAULT"
    timestamp_sec: Optional[float] = None

# ============================
# API: セッション・サーブ（静的パスを先に登録し /api/sessions/{id} より前に置く）
# ============================

@app.post("/api/start-session")
async def api_sessions_start():
    """セッション開始。session_id を返す。（/api/sessions/start は {session_id} と衝突するため別パス）"""
    return store_start_session()


@app.post("/api/serves")
async def api_serves_record(body: ServeRecordRequest):
    """サーブ1本を記録（IN / OUT / FAULT）。"""
    if body.result not in ("IN", "OUT", "FAULT"):
        raise HTTPException(status_code=400, detail="result must be IN, OUT, or FAULT")
    serve = append_serve(body.session_id, body.result, timestamp_sec=body.timestamp_sec)
    if serve is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True, "serve": serve}


@app.post("/api/serves/undo")
async def api_serves_undo(body: dict):
    """最後の1本を取り消す。body: { "session_id": "..." }"""
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    ok = undo_last_serve(session_id)
    return {"ok": ok}


@app.post("/api/sessions/end")
async def api_sessions_end(body: dict):
    """セッション終了。集計結果を返す。"""
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    s = store_get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return s


@app.get("/api/sessions")
async def api_sessions_list():
    """セッション一覧（履歴）。"""
    return store_list_sessions()


@app.get("/api/sessions/{session_id}/video")
async def api_sessions_video(session_id: str, request: Request):
    """セッションの録画動画をストリーミング（結果画面で表示用）。"""
    video_path = get_session_video_path(session_id)
    if not video_path or not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    file_size = os.path.getsize(video_path)
    ext = os.path.splitext(video_path)[1].lower()
    content_type = "video/webm" if ext == ".webm" else "video/mp4"
    range_header = request.headers.get("range")
    if range_header:
        range_spec = range_header.replace("bytes=", "")
        parts = range_spec.split("-")
        start = int(parts[0])
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)
        content_length = end - start + 1
        def iter_chunk():
            with open(video_path, "rb") as f:
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
        with open(video_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                yield chunk
    return StreamingResponse(
        iter_file(),
        media_type=content_type,
        headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
    )


@app.get("/api/sessions/{session_id}")
async def api_sessions_get(session_id: str):
    """セッション詳細を取得。"""
    s = store_get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return s


@app.delete("/api/sessions/{session_id}")
async def api_sessions_delete(session_id: str):
    """セッション1件を削除。"""
    ok = store_delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


# ============================
# フォーム解析（1本のクリップ → scores, focus_label, ai_text, practice）
# コーチングはテンプレートから選択（AIは使わない）
# ============================

def _analyze_clip_to_result(clip_path: str) -> Optional[dict]:
    """クリップ1本を解析し、{ scores, focus_label, ai_text, practice } を返す。"""
    try:
        result = analyze_clip(clip_path)
        metrics = result.get("metrics", {})
        if not metrics:
            return None
        elbow = metrics.get("elbow_angle", 0)
        body_sway = metrics.get("body_sway", 1.0)
        impact_height = metrics.get("impact_height", 0)
        focus_key = "body_sway"
        if body_sway <= 0.15 and (elbow < 90 or elbow > 130):
            focus_key = "elbow_angle"
        elif body_sway <= 0.15 and 90 <= elbow <= 130 and impact_height < 1.5:
            focus_key = "impact_height"
        coaching = get_coaching_from_template(focus_key)
        return {
            "scores": metrics,
            "focus_label": coaching["focus_label"],
            "ai_text": coaching["ai_text"],
            "practice": coaching["practice"],
        }
    except Exception as e:
        logging.warning(f"Clip analysis failed: {e}")
        return None


@app.post("/api/analysis")
async def api_analysis(
    session_id: str = Form(...),
    serve_events: str = Form(...),
    video: UploadFile = File(..., alias="video"),
):
    """
    動画＋サーブタイムスタンプを送信し、各サーブの timestamp ±1秒 のクリップを解析。
    multipart: video, session_id, serve_events (JSON: [{ id, result, timestamp_sec }])
    返却: セッション詳細（analysis 付き）。
    """
    if not store_get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        events = json.loads(serve_events)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="serve_events must be valid JSON array")

    if len(events) == 1 and (events[0].get("timestamp_sec") == 0 or events[0].get("timestamp_sec") is None):
        set_session_source(session_id, "upload")

    if not video.filename or not video.filename.lower().endswith((".mp4", ".webm", ".mov", ".avi")):
        raise HTTPException(status_code=400, detail="動画ファイル（mp4/webm/mov/avi）を指定してください")

    video_bytes = await video.read()
    ext = ".webm" if video.filename and video.filename.lower().endswith(".webm") else ".mp4"
    save_session_video(session_id, video_bytes, ext=ext)
    video_path = get_session_video_path(session_id)
    if not video_path:
        raise HTTPException(status_code=500, detail="Failed to save video")

    # WebM は OpenCV が読めない環境（Render など）があるので MP4 に変換する
    if video_path.endswith(".webm"):
        video_path = _convert_webm_to_mp4(video_path)

    clip_paths = []
    tracking_video_done = False
    first_ts = None
    try:
        for ev in events:
            serve_id = ev.get("id")
            ts = ev.get("timestamp_sec")
            if serve_id is None or ts is None:
                continue
            if first_ts is None:
                first_ts = float(ts)
            analysis = None
            clip_path = None
            try:
                clip_path = extract_clip(video_path, float(ts), range_sec=2.0)
                clip_paths.append(clip_path)
                analysis = _analyze_clip_to_result(clip_path)
            except Exception as e:
                logging.warning(f"Clip extract or analysis failed for serve {serve_id}: {e}")
            if analysis:
                save_serve_analysis(session_id, serve_id, analysis)
            else:
                # 解析失敗時もプレースホルダーを保存し、フロントで「解析できませんでした」を表示
                save_serve_analysis(session_id, serve_id, {
                    "scores": {},
                    "focus_label": "解析できませんでした",
                    "ai_text": "動画の切り出しまたは骨格検出に失敗しました。照明・画角を確認してください。",
                    "practice": "再度お試しください。",
                })
        # トラッキング動画はセッション動画を直接使って生成（クリップに依存しない）
        if not tracking_video_done and first_ts is not None and os.path.isfile(video_path):
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()
                # タイムスタンプが有効なときはその時刻を中心に（タップ遅れ約0.4秒を補正）、そうでないときは動画からインパクト検出
                if first_ts >= 1.0:
                    tap_delay_sec = 0.4
                    impact_index = max(0, int((first_ts - tap_delay_sec) * fps))
                else:
                    try:
                        detected = detect_impact_frame(video_path)
                        offset_frames = int(0.25 * fps)
                        impact_index = max(0, detected - offset_frames)
                    except Exception:
                        impact_index = int(1.5 * fps)
                    impact_index = max(0, impact_index)
                diag = extract_pose_landmarks(video_path, impact_index=impact_index, range_sec=1.5)
                pixel = diag.get("pixel")
                start_frame = diag.get("start_frame", 0)
                if pixel is not None and len(pixel) > 0:
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    tracking_filename = f"tracking_{session_id}.mp4"
                    out_path = os.path.join(OUTPUT_DIR, tracking_filename)
                    # レンダラーの互換のため list に変換
                    landmarks_list = pixel.tolist() if hasattr(pixel, "tolist") else list(pixel)
                    n_frames = len(landmarks_list)
                    if render_analyzed_video(
                        video_path,
                        landmarks_list,
                        out_path,
                        impact_frame=start_frame + n_frames // 2,
                        start_frame=start_frame,
                    ):
                        set_session_tracking_video(session_id, tracking_filename)
                        tracking_video_done = True
                    else:
                        logging.warning("render_analyzed_video returned False")
                else:
                    logging.warning("No pose landmarks for tracking video")
            except Exception as e:
                logging.warning(f"Tracking video render failed: {e}", exc_info=True)
        return store_get_session(session_id)
    finally:
        for p in clip_paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

# ============================
# 開発用: データリセット・サンプル投入
# ============================
@app.post("/reset-all")
async def reset_all():
    """全セッション・出力を削除（開発用）。"""
    try:
        import shutil
        from tracker_store import DATA_ROOT
        if os.path.exists(DATA_ROOT):
            shutil.rmtree(DATA_ROOT)
        os.makedirs(os.path.join(DATA_ROOT, "sessions"), exist_ok=True)
        os.makedirs(os.path.join(DATA_ROOT, "videos"), exist_ok=True)
        os.makedirs(os.path.join(DATA_ROOT, "outputs"), exist_ok=True)
        return {"status": "ok", "message": "データをリセットしました"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


