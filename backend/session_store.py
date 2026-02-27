import os
import json
from typing import Dict, List, Optional

# データディレクトリ
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data", "sessions")


def get_session_dir(session_id: str) -> str:
    """
    セッションディレクトリのパスを取得
    
    Args:
        session_id: セッションID
    
    Returns:
        str: セッションディレクトリのパス
    """
    return os.path.join(DATA_DIR, session_id)


def get_scores_file(session_id: str) -> str:
    """
    スコアファイルのパスを取得
    
    Args:
        session_id: セッションID
    
    Returns:
        str: スコアファイルのパス
    """
    return os.path.join(get_session_dir(session_id), "scores.json")


def append_score(session_id: str, score_dict: Dict[str, float]) -> None:
    """
    スコアを追加保存
    
    Args:
        session_id: セッションID
        score_dict: スコア辞書（impact_height, elbow_angle, body_sway）
    """
    try:
        # ディレクトリが無ければ作成
        session_dir = get_session_dir(session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # 既存のスコアを読み込む
        scores_file = get_scores_file(session_id)
        scores = []
        
        if os.path.exists(scores_file):
            try:
                with open(scores_file, "r", encoding="utf-8") as f:
                    scores = json.load(f)
            except Exception:
                scores = []
        
        # 新しいスコアを追加
        scores.append(score_dict)
        
        # 保存
        with open(scores_file, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
    
    except Exception:
        # 例外は握りつぶす（MVP）
        pass


def load_scores(session_id: str) -> List[Dict[str, float]]:
    """
    セッションのスコア一覧を読み込む
    
    Args:
        session_id: セッションID
    
    Returns:
        list: スコアのリスト
    """
    try:
        scores_file = get_scores_file(session_id)
        
        if not os.path.exists(scores_file):
            return []
        
        with open(scores_file, "r", encoding="utf-8") as f:
            scores = json.load(f)
        
        # 必要なキーが揃っているか確認
        required_keys = ["impact_height", "elbow_angle", "body_sway"]
        valid_scores = []
        
        for score in scores:
            if all(key in score for key in required_keys):
                valid_scores.append({
                    "impact_height": float(score["impact_height"]),
                    "elbow_angle": float(score["elbow_angle"]),
                    "body_sway": float(score["body_sway"]),
                })
        
        return valid_scores
    
    except Exception:
        # JSON壊れていたら空リスト
        return []


def clear_session(session_id: str) -> None:
    """
    セッションをクリア（ディレクトリごと削除）
    
    Args:
        session_id: セッションID
    """
    try:
        session_dir = get_session_dir(session_id)
        
        if os.path.exists(session_dir):
            import shutil
            shutil.rmtree(session_dir)
    
    except Exception:
        # 例外は握りつぶす（MVP）
        pass

