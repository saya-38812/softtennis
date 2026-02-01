import json
import os
from pathlib import Path
from openai import OpenAI
from typing import Dict, List, Tuple
from dotenv import load_dotenv

# ============================
# 環境変数ロード
# ============================
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


# ============================
# 改善優先度の自動計算
# ============================
def calculate_improvement_priority(diagnosis: Dict) -> List[Tuple[str, str, float]]:
    """
    診断データから改善優先度を自動計算
    戻り値: [(改善項目名, 説明, 重要度スコア)] を重要度順で返す
    """

    raw = diagnosis.get("raw_values", {})
    weak = diagnosis.get("weakness", {})

    improvements = []

    # 体重移動（最重要）
    if weak.get("weight_transfer") != "ok":
        score = abs(raw.get("weight_diff", 0)) * 10
        improvements.append(("体重移動", "前足への体重移動が不足しています", score))

    # インパクト高さ
    if weak.get("impact_height") != "ok":
        score = abs(raw.get("impact_diff", 0)) * 8
        improvements.append(("インパクト高さ", "打点が低くなっています", score))

    # 腰の回転
    if weak.get("waist_rotation") != "ok":
        score = abs(raw.get("waist_rotation_diff", 0)) * 0.5
        improvements.append(("腰の回転", "腰の回転が不足しています", score))

    # トス同期
    if weak.get("toss_sync") != "ok":
        score = abs(raw.get("toss_sync_diff", 0)) * 5
        improvements.append(("トスのタイミング", "トスと打点のタイミングがズレています", score))

    # 肩の開き
    if weak.get("shoulder_angle") != "ok":
        score = abs(raw.get("shoulder_angle_diff", 0)) * 0.4
        improvements.append(("肩の開き", "肩が早く開きすぎています", score))

    # ソートして返す
    improvements.sort(key=lambda x: x[2], reverse=True)
    return improvements


# ============================
# メニュー詳細生成（短く）
# ============================
def generate_menu_detail(menu_name: str, diagnosis: Dict) -> str:
    """
    練習メニューの詳細を短く生成（最大6行）
    """

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "OPENAI_API_KEYが設定されていません"

        client = OpenAI(api_key=api_key)

        prompt = f"""
あなたはソフトテニスのプロコーチです。
中学生向けに、次の練習メニューを短く説明してください。

【練習メニュー】
{menu_name}

【ルール】
- 最大6行
- 箇条書きのみ
- 回数を必ず書く
- 長文は禁止

出力例：
・目的：○○
・やり方：○○
・回数：10回×3
・注意：○○

【診断データ】
{json.dumps(diagnosis, ensure_ascii=False)}
"""

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは簡潔なプロコーチです。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )

        return res.choices[0].message.content.strip()

    except Exception as e:
        return f"メニュー詳細生成エラー: {e}"


# ============================
# AIコーチ（最優先1つだけ）
# ============================
def generate_ai_menu(diagnosis: Dict) -> str:
    """
    最優先の改善点を1つだけ提示する短いAIアドバイス
    """

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "OPENAI_API_KEYが設定されていません"

        client = OpenAI(api_key=api_key)

        # 優先度を計算
        priorities = calculate_improvement_priority(diagnosis)

        # 最優先を1つだけ取得
        if priorities:
            top_name, top_desc, _ = priorities[0]
        else:
            top_name, top_desc = "フォーム維持", "大きな崩れはありません"

        prompt = f"""
あなたはソフトテニスのプロコーチです。
対象は中学生の右利きプレイヤーです。

改善点は1つだけに絞ってください。

【最優先改善点】
{top_name}：{top_desc}

【ルール】
- 最大5行
- 1行は短く
- 必ず回数を書く
- 一般論は禁止

出力形式（厳守）：

改善ポイント：○○
今日の練習：○○
回数：10回×3セット
コツ：○○
一言：○○
"""

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは簡潔なソフトテニスのプロコーチです。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )

        return res.choices[0].message.content.strip()

    except Exception as e:
        return f"AIコーチ生成エラー: {e}"
