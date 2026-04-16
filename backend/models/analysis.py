"""
Analysis データモデル（MVP）
"""
from typing import Dict

# Analysis: serve_id, scores, focus_label, ai_text, practice
# 実体は tracker_store の session.json 内 analysis[serve_id] で管理
def analysis_schema():
    """Analysis のスキーマ説明（scores, focus_label, ai_text, practice）"""
    pass
