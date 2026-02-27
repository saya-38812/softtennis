import os
import json
from typing import Dict, Optional

# プレイヤーID（固定）
PLAYER_ID = "local_player"

# データディレクトリ (Reload監視を避けるため、backend配下から外す)
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "server_storage", "players"))
PLAYER_FILE = os.path.join(DATA_DIR, f"{PLAYER_ID}.json")


def load_last_score() -> Optional[Dict[str, float]]:
    """
    前回のスコアを読み込む
    
    Returns:
        dict | None: 前回のスコア（impact_height, elbow_angle, body_sway）またはNone
    """
    try:
        # ディレクトリが無ければ作成
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # ファイルが無ければNoneを返す
        if not os.path.exists(PLAYER_FILE):
            return None
        
        # JSON読み込み
        with open(PLAYER_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 必要なキーが揃っているか確認
        required_keys = ["impact_height", "elbow_angle", "body_sway"]
        if not all(key in data for key in required_keys):
            return None
        
        return {
            "impact_height": float(data["impact_height"]),
            "elbow_angle": float(data["elbow_angle"]),
            "body_sway": float(data["body_sway"]),
            "waist_speed": float(data.get("waist_speed", 0.0)),
            "weight_transfer": float(data.get("weight_transfer", 0.0)),
        }
    
    except Exception:
        # JSON壊れていたらNone扱い
        return None


def save_score(score_dict: Dict[str, float]) -> None:
    """
    スコアを保存する
    
    Args:
        score_dict: スコア辞書（impact_height, elbow_angle, body_sway）
    """
    try:
        # ディレクトリが無ければ作成
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # JSON保存
        with open(PLAYER_FILE, "w", encoding="utf-8") as f:
            json.dump(score_dict, f, ensure_ascii=False, indent=2)
    
    except Exception:
        # 例外は握りつぶす（MVP）
        pass


def clear_player_data() -> None:
    """
    プレイヤーデータを削除する
    """
    try:
        if os.path.exists(PLAYER_FILE):
            os.remove(PLAYER_FILE)
    except Exception:
        pass

