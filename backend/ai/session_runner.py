"""
1回分の練習（セッション）をまとめて解析するモジュール（Colab 用）

Colab ノートブックは「Drive から動画を取る」「スプシに書く」などの入出力だけを担当し、
「動画をどう切って、どう解析するか」はこのファイルにまとめる。
こうしておくと、Google のサービスがなくても（ローカル PC でも）解析部分だけテストできる。

使い方：
  from ai.session_runner import analyze_session
  result = analyze_session("video.mp4", mode="serve_tracking",
                           serves=[{"result": "IN", "timestamp_sec": 5.2}, ...])
"""
import logging
import os
from collections import Counter
from typing import Dict, List, Optional

from .clip_extract import extract_clip
from .model_compare import analyze_serve

logger = logging.getLogger(__name__)

# ============================================================
# 切り出しの設定
# ============================================================

# IN/FAULT のタップは「ボールが入ったか見えてから」押すので、実際に打った瞬間より遅れる。
# そこで、タップした時刻より CLIP_TAP_DELAY_SEC 秒前を中心に切り出す。
# （例：5.2秒にタップ → 4.2秒を中心に前後 CLIP_RANGE_SEC 秒 = 1.7〜6.7秒を切り出す）
CLIP_TAP_DELAY_SEC = 1.0
# 中心から前後何秒を切り出すか
CLIP_RANGE_SEC = 2.5


def analyze_session(video_path: str, mode: str, serves: Optional[List[dict]] = None) -> dict:
    """
    セッション1件分を解析して、スプシの result_json に入れる辞書を返す。

    Args:
        video_path: 動画ファイルのパス（mp4 推奨）
        mode: "serve_tracking"（IN/FAULT 記録）または "form_only"（フォームだけ）
        serves: serve_tracking のときのサーブ一覧 [{"result": "IN", "timestamp_sec": 1.5}, ...]

    Returns:
        {
          "version": 1,
          "mode": "...",
          "serves": [ {"index": 0, "result": "IN", "timestamp_sec": 1.5, "status": "ok", ...採点結果}, ... ],
          "summary": {...}   # 全体のまとめ（平均点・よく出た改善ポイントなど）
        }
    """
    if mode == "form_only":
        # 動画全体がサーブ1本分。インパクトは analyze_serve の中で自動検出される
        serve_results = [_analyze_one(video_path, index=0, result=None, timestamp_sec=None)]
    else:
        serve_results = []
        for index, serve in enumerate(serves or []):
            ts = float(serve.get("timestamp_sec") or 0.0)
            serve_results.append(_analyze_clip(video_path, index, serve.get("result"), ts))

    return {
        "version": 1,  # 結果の形を将来変えたときに、アプリ側で見分けるための番号
        "mode": mode,
        "serves": serve_results,
        "summary": _summarize(serve_results),
    }


# ============================================================
# 内部処理
# ============================================================

def _analyze_clip(video_path: str, index: int, result: Optional[str], timestamp_sec: float) -> dict:
    """タップした時刻の周辺を切り出して、サーブ1本を解析する"""
    center = max(0.0, timestamp_sec - CLIP_TAP_DELAY_SEC)
    clip_path = None
    try:
        clip_path = extract_clip(video_path, center_sec=center, range_sec=CLIP_RANGE_SEC)
        return _analyze_one(clip_path, index, result, timestamp_sec)
    except Exception as e:
        logger.warning("サーブ%d の切り出しに失敗: %s", index + 1, e)
        return _base(index, result, timestamp_sec) | {
            "status": "error",
            "message": f"動画の切り出しに失敗しました（{e}）",
        }
    finally:
        # 切り出した一時ファイルは、成功・失敗にかかわらず必ず消す（ディスクを圧迫しないように）
        if clip_path and os.path.exists(clip_path):
            os.remove(clip_path)


def _analyze_one(video_path: str, index: int, result: Optional[str], timestamp_sec: Optional[float]) -> dict:
    """動画（またはクリップ）1本を採点して、サーブ番号などの情報を付けて返す"""
    try:
        analysis = analyze_serve(video_path)
    except Exception as e:
        logger.warning("サーブ%d の解析に失敗: %s", index + 1, e)
        analysis = {"status": "error", "message": f"解析中にエラーが起きました（{e}）"}
    # 「何本目・IN/FAULT・何秒」の情報 + 採点結果 を1つの辞書にまとめる
    return _base(index, result, timestamp_sec) | analysis


def _base(index: int, result: Optional[str], timestamp_sec: Optional[float]) -> dict:
    """どのサーブの結果かを表す共通の情報"""
    return {"index": index, "result": result, "timestamp_sec": timestamp_sec}


def _summarize(serve_results: List[dict]) -> Dict:
    """全サーブの結果をまとめる（アプリやレポートの一番上に出す用）"""
    ok = [r for r in serve_results if r.get("status") == "ok"]
    summary = {
        "analyzed_count": len(ok),
        "error_count": len(serve_results) - len(ok),
    }
    if not ok:
        return summary

    summary["average_score"] = round(sum(r["overall_score"] for r in ok) / len(ok), 1)

    # 一番多く「改善ポイント」に選ばれた指標 ＝ この日の一番の課題
    focus_counts = Counter(r["focus_key"] for r in ok)
    top_key, top_count = focus_counts.most_common(1)[0]
    summary["top_focus_key"] = top_key
    summary["top_focus_label"] = next(r["focus_label"] for r in ok if r["focus_key"] == top_key)
    summary["top_focus_count"] = top_count

    # IN と FAULT で平均点を分けて出す（将来の「入る姿勢 / フォルトしがちな姿勢」分析の第一歩）
    for label in ("IN", "FAULT"):
        group = [r for r in ok if r.get("result") == label]
        if group:
            summary[f"average_score_{label.lower()}"] = round(
                sum(r["overall_score"] for r in group) / len(group), 1
            )
    return summary
