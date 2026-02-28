"""
Serve Tracker MVP: セッション単位のサーブ記録（IN/OUT/FAULT）と速度・フォーム用ストア。
既存の session_store（フォームスコア用）とは別に、server_storage/tracker_sessions に保存する。
"""
import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Literal

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "server_storage", "tracker_sessions"))

ServeResult = Literal["IN", "OUT", "FAULT"]


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _session_path(session_id: str) -> str:
    return os.path.join(DATA_DIR, f"{session_id}.json")


def start_session() -> dict:
    """新規セッションを開始し、session_id とメタデータを返す。"""
    _ensure_dir()
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"
    data = {
        "id": session_id,
        "date": now,
        "serves": [],
    }
    path = _session_path(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {
        "session_id": session_id,
        "date": now,
    }


def append_serve(session_id: str, result: ServeResult, speed_kmh: Optional[float] = None) -> Optional[dict]:
    """
    サーブ1本を記録する。
    result: "IN" | "OUT" | "FAULT"
    speed_kmh: オプション。速度解析結果（km/h）
    戻り値: 追加後のサーブ記録（id, session_id, result, speed, timestamp）または None
    """
    path = _session_path(session_id)
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
        "speed": speed_kmh,
        "timestamp": now,
    }
    data["serves"] = data.get("serves", []) + [serve]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return serve


def undo_last_serve(session_id: str) -> bool:
    """最後の1本を取り消す。成功なら True。"""
    path = _session_path(session_id)
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
    """セッション1件を取得。集計済みの total_attempts, in_count, out_count, fault_count, avg_speed, max_speed を含む。"""
    path = _session_path(session_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    serves = data.get("serves", [])
    in_count = sum(1 for s in serves if s.get("result") == "IN")
    out_count = sum(1 for s in serves if s.get("result") == "OUT")
    fault_count = sum(1 for s in serves if s.get("result") == "FAULT")
    speeds = [s["speed"] for s in serves if s.get("speed") is not None and isinstance(s["speed"], (int, float))]
    avg_speed = float(sum(speeds) / len(speeds)) if speeds else None
    max_speed = float(max(speeds)) if speeds else None
    data["total_attempts"] = len(serves)
    data["in_count"] = in_count
    data["out_count"] = out_count
    data["fault_count"] = fault_count
    data["avg_speed"] = round(avg_speed, 1) if avg_speed is not None else None
    data["max_speed"] = round(max_speed, 1) if max_speed is not None else None
    return data


def delete_session(session_id: str) -> bool:
    """セッション1件を削除する。成功なら True。"""
    path = _session_path(session_id)
    if not os.path.exists(path):
        return False
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def list_sessions() -> List[dict]:
    """セッション一覧を日付降順で返す。各要素は get_session と同じ集計付き。"""
    _ensure_dir()
    ids = []
    for name in os.listdir(DATA_DIR):
        if name.endswith(".json"):
            ids.append(name[:-5])
    sessions = []
    for sid in ids:
        s = get_session(sid)
        if s:
            sessions.append(s)
    sessions.sort(key=lambda x: x.get("date", ""), reverse=True)
    return sessions
