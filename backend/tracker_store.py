"""
セッション・サーブ・解析結果の一元管理（MVP）。
server_storage/data/sessions/{id}/session.json
server_storage/data/videos/{id}.mp4（または .webm）
server_storage/data/outputs/（生成物）
"""
import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Literal

BASE_DIR = os.path.dirname(__file__)
DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "server_storage", "data"))
SESSIONS_DIR = os.path.join(DATA_ROOT, "sessions")
VIDEOS_DIR = os.path.join(DATA_ROOT, "videos")
OUTPUTS_DIR = os.path.join(DATA_ROOT, "outputs")

ServeResult = Literal["IN", "OUT", "FAULT"]


def _ensure_dirs():
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _session_dir(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, session_id)


def _session_file(session_id: str) -> str:
    return os.path.join(_session_dir(session_id), "session.json")


def _video_path(session_id: str, ext: str = ".mp4") -> str:
    return os.path.join(VIDEOS_DIR, f"{session_id}{ext}")


def start_session() -> dict:
    """新規セッションを開始。session_id と created_at を返す。"""
    _ensure_dirs()
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"
    session_dir = _session_dir(session_id)
    os.makedirs(session_dir, exist_ok=True)
    data = {
        "id": session_id,
        "created_at": now,
        "video_path": None,
        "serves": [],
        "analysis": {},
    }
    with open(_session_file(session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {
        "session_id": session_id,
        "date": now,
    }


def append_serve(
    session_id: str,
    result: ServeResult,
    timestamp_sec: Optional[float] = None,
) -> Optional[dict]:
    """
    サーブ1本を記録。
    result: "IN" | "OUT" | "FAULT"
    timestamp_sec: 録画内の経過秒。
    """
    path = _session_file(session_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    serve_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"
    serve = {
        "id": serve_id,
        "session_id": session_id,
        "result": result,
        "timestamp_sec": timestamp_sec,
        "timestamp": now,
    }
    data.setdefault("serves", []).append(serve)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return serve


def undo_last_serve(session_id: str) -> bool:
    """最後の1本を取り消す。"""
    path = _session_file(session_id)
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    serves = data.get("serves", [])
    if not serves:
        return False
    data["serves"] = serves[:-1]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return True


def get_session(session_id: str) -> Optional[dict]:
    """セッション1件を取得。集計・analysis を含む。"""
    path = _session_file(session_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    serves = data.get("serves", [])
    in_count = sum(1 for s in serves if s.get("result") == "IN")
    out_count = sum(1 for s in serves if s.get("result") == "OUT")
    fault_count = sum(1 for s in serves if s.get("result") == "FAULT")

    data["total_attempts"] = len(serves)
    data["in_count"] = in_count
    data["out_count"] = out_count
    data["fault_count"] = fault_count
    data["date"] = data.get("created_at", data.get("date", ""))
    return data


def save_session_video(session_id: str, video_bytes: bytes, ext: str = ".mp4") -> str:
    """セッション録画を data/videos/{session_id}{ext} に保存。session.json の video_path を更新。"""
    _ensure_dirs()
    path = _video_path(session_id, ext)
    with open(path, "wb") as f:
        f.write(video_bytes)
    # session.json に video_path を相対で保存（videos/{id}.mp4）
    rel_path = f"videos/{session_id}{ext}"
    session_path = _session_file(session_id)
    if os.path.exists(session_path):
        with open(session_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["video_path"] = rel_path
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def get_session_video_path(session_id: str) -> Optional[str]:
    """セッションの録画ファイル絶対パス。.webm / .mp4 のいずれかがあれば返す。"""
    for ext in (".webm", ".mp4"):
        path = _video_path(session_id, ext)
        if os.path.isfile(path):
            return path
    return None


def save_serve_analysis(session_id: str, serve_id: str, analysis: dict) -> None:
    """指定サーブの解析結果を保存。analysis: { scores, focus_label, ai_text, practice }"""
    path = _session_file(session_id)
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("analysis", {})[serve_id] = analysis
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_session_tracking_video(session_id: str, output_filename: str) -> None:
    """セッションにトラッキング動画の出力ファイル名を保存。get_session で返る。"""
    path = _session_file(session_id)
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["tracking_video"] = output_filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_session_source(session_id: str, source: str) -> None:
    """セッションの出所を保存（例: "upload" = アップロード画面から）。get_session で返る。"""
    path = _session_file(session_id)
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["source"] = source
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def delete_session(session_id: str) -> bool:
    """セッション1件を削除（session ディレクトリと動画ファイル）。"""
    import shutil
    session_dir = _session_dir(session_id)
    ok = False
    if os.path.isdir(session_dir):
        try:
            shutil.rmtree(session_dir)
            ok = True
        except OSError:
            pass
    for ext in (".webm", ".mp4"):
        vp = _video_path(session_id, ext)
        if os.path.isfile(vp):
            try:
                os.remove(vp)
            except OSError:
                pass
    return ok


def list_sessions() -> List[dict]:
    """セッション一覧を日付降順で返す。"""
    _ensure_dirs()
    sessions = []
    for name in os.listdir(SESSIONS_DIR):
        session_dir = os.path.join(SESSIONS_DIR, name)
        if not os.path.isdir(session_dir):
            continue
        s = get_session(name)
        if s:
            sessions.append(s)
    sessions.sort(key=lambda x: x.get("date", ""), reverse=True)
    return sessions


# サンプルセッション用固定ID（録画なしで結果画面を確認する用）
SAMPLE_PRACTICE_ID = "sample-practice-001"
SAMPLE_UPLOAD_ID = "sample-upload-001"


def _make_sample_analysis() -> dict:
    """テンプレートに合わせたサンプル解析データを1件返す。"""
    from ai.coach_templates import get_coaching_from_template
    coaching = get_coaching_from_template("body_sway")
    return {
        "scores": {
            "impact_height": 1.2,
            "elbow_angle": 105,
            "body_sway": 0.12,
            "waist_speed": 0.8,
            "weight_transfer": 0.7,
        },
        "focus_label": coaching["focus_label"],
        "ai_text": coaching["ai_text"],
        "practice": coaching["practice"],
    }


def ensure_sample_sessions() -> List[str]:
    """
    サンプルセッションがなければ作成する（録画なしでUI確認用）。
    作成した session_id のリストを返す。
    """
    _ensure_dirs()
    created = []
    # 練習フロー用サンプル（1本 IN、解析あり）
    if not os.path.exists(_session_file(SAMPLE_PRACTICE_ID)):
        serve_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat() + "Z"
        data = {
            "id": SAMPLE_PRACTICE_ID,
            "created_at": now,
            "video_path": None,
            "tracking_video": None,
            "serves": [
                {
                    "id": serve_id,
                    "session_id": SAMPLE_PRACTICE_ID,
                    "result": "IN",
                    "timestamp_sec": 1.5,
                    "timestamp": now,
                },
            ],
            "analysis": {serve_id: _make_sample_analysis()},
        }
        os.makedirs(_session_dir(SAMPLE_PRACTICE_ID), exist_ok=True)
        with open(_session_file(SAMPLE_PRACTICE_ID), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        created.append(SAMPLE_PRACTICE_ID)
    # アップロードフロー用サンプル（1本、解析あり）
    if not os.path.exists(_session_file(SAMPLE_UPLOAD_ID)):
        serve_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat() + "Z"
        data = {
            "id": SAMPLE_UPLOAD_ID,
            "created_at": now,
            "video_path": None,
            "tracking_video": None,
            "source": "upload",
            "serves": [
                {
                    "id": serve_id,
                    "session_id": SAMPLE_UPLOAD_ID,
                    "result": "IN",
                    "timestamp_sec": 0,
                    "timestamp": now,
                },
            ],
            "analysis": {serve_id: _make_sample_analysis()},
        }
        os.makedirs(_session_dir(SAMPLE_UPLOAD_ID), exist_ok=True)
        with open(_session_file(SAMPLE_UPLOAD_ID), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        created.append(SAMPLE_UPLOAD_ID)
    return created


def get_outputs_dir() -> str:
    """生成物出力ディレクトリ（data/outputs）の絶対パス。"""
    _ensure_dirs()
    return OUTPUTS_DIR
