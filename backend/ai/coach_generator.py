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


# ============================
# 身体感覚コーチング生成
# ============================
def generate_coaching_message(weakness_label: str) -> dict:
    """
    身体感覚の指示を生成（2行のみ）
    
    Args:
        weakness_label: 弱点ラベル（例：「打点が低い」「体軸がブレている」「肘が曲がりすぎている」）
    
    Returns:
        dict: {"ai_text": "...", "practice": "..."}
    """
    
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                "ai_text": "身体感覚で動きを改善しましょう",
                "practice": "サーブを10回繰り返してください"
            }
        
        client = OpenAI(api_key=api_key)
        
        # 弱点ラベルを日本語に変換
        label_map = {
            "impact_height": "打点が低い",
            "elbow_angle": "肘が曲がりすぎている",
            "body_sway": "体軸がブレている",
        }
        
        # もしキーが渡された場合は変換
        if weakness_label in label_map:
            weakness_label = label_map[weakness_label]
        
        prompt = f"""以下の問題を身体感覚の比喩で指導してください。

【問題】
{weakness_label}

【出力形式】
必ず2行で出力してください。
1行目：動作イメージ（身体感覚の比喩）
2行目：具体的な単一練習（1つだけ）

【例】
ボールを"打つ"ではなく"空に押し上げる"感覚で振ってください
タオルを上に放り投げる動きを10回繰り返しましょう

【禁止事項】
- 関節名、角度、数値説明は禁止
- 長文解説は禁止
- 複数の提案は禁止
- 3行以上は禁止
- 専門用語は禁止
"""
        
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """あなたはジュニア選手を指導するスポーツコーチです。
専門用語を使わず、身体感覚の比喩で指導してください。
フォームの説明は禁止です。
理解させるのではなく「動かす」ことが目的です。
1回で実行できる指示のみ出してください。

出力は必ず2行：
1行目：動作イメージ
2行目：具体的な単一練習

禁止：
関節名
角度
数値説明
長文解説
複数提案"""
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        
        content = res.choices[0].message.content.strip()
        
        # 改行で2行に分割
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        
        # 2行に分割
        if len(lines) >= 2:
            ai_text = lines[0]
            practice = lines[1]
        elif len(lines) == 1:
            # 1行しかない場合は分割を試みる
            parts = lines[0].split("。", 1)
            if len(parts) >= 2:
                ai_text = parts[0] + "。"
                practice = parts[1]
            else:
                ai_text = lines[0]
                practice = "サーブを10回繰り返しましょう"
        else:
            ai_text = "身体感覚で動きを改善しましょう"
            practice = "サーブを10回繰り返しましょう"
        
        return {
            "ai_text": ai_text,
            "practice": practice
        }
    
    except Exception as e:
        print(f"コーチング生成エラー: {e}")
        return {
            "ai_text": "身体感覚で動きを改善しましょう",
            "practice": "サーブを10回繰り返してください"
        }