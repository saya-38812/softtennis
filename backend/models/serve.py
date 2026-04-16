"""
Serve データモデル（MVP）
"""
from typing import Literal, Optional

# Serve: id, session_id, result (IN / OUT / FAULT), timestamp
# 実体は tracker_store の session.json 内 serves[] で管理
ServeResult = Literal["IN", "OUT", "FAULT"]

def serve_schema():
    """Serve のスキーマ説明（id, session_id, result, timestamp, timestamp_sec）"""
    pass
