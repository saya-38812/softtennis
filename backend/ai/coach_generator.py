import json
import os
from pathlib import Path
from openai import OpenAI
from typing import Dict, List, Tuple
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む（backendディレクトリの.envファイルを明示的に指定）
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

def calculate_improvement_priority(diagnosis: Dict) -> List[Tuple[str, str, float, str]]:
    """
    診断データから改善優先度を自動計算
    戻り値: [(改善項目名, 説明, 重要度スコア, 現在の値), ...] を優先度順（重要度が高い順）で返す
    """
    raw_values = diagnosis.get("raw_values", {})
    weakness = diagnosis.get("weakness", {})
    
    # 各改善項目の重要度を計算（数値の絶対値と影響度を考慮）
    improvements = []
    
    # 体重移動（最重要：パワー伝達の基盤）
    if weakness.get("weight_transfer") != "ok":
        weight_diff = abs(raw_values.get("weight_diff", 0))
        improvements.append(("体重移動", "前足への体重移動が不足しています", weight_diff * 10, f"weight_diff={raw_values.get('weight_diff', 0)}"))
    
    # インパクト高さ（サーブの成功率に直結）
    if weakness.get("impact_height") != "ok":
        impact_diff = abs(raw_values.get("impact_diff", 0))
        improvements.append(("インパクト高さ", "打点の高さが適切ではありません", impact_diff * 8, f"impact_diff={raw_values.get('impact_diff', 0)}"))
    
    # 腰の回転（パワー生成の要）
    if weakness.get("waist_rotation") != "ok":
        waist_rot = abs(raw_values.get("waist_rotation_diff", 0))
        improvements.append(("腰の回転", "腰の回転が不足または過剰です", waist_rot * 0.5, f"waist_rotation_diff={raw_values.get('waist_rotation_diff', 0)}度"))
    
    # 体の開き（フォームの安定性）
    if weakness.get("body_open") != "ok":
        open_diff = abs(raw_values.get("open_diff", 0))
        improvements.append(("体の開き", "体の開きタイミングが適切ではありません", open_diff * 7, f"open_diff={raw_values.get('open_diff', 0)}"))
    
    # 肩の開き角度
    if weakness.get("shoulder_angle") != "ok":
        shoulder_diff = abs(raw_values.get("shoulder_angle_diff", 0))
        improvements.append(("肩の開き", "肩の開き角度が適切ではありません", shoulder_diff * 0.4, f"shoulder_angle_diff={raw_values.get('shoulder_angle_diff', 0)}度"))
    
    # 肘の角度
    if weakness.get("elbow_angle") != "ok":
        elbow_diff = abs(raw_values.get("elbow_angle_diff", 0))
        improvements.append(("肘の使い方", "肘の角度が適切ではありません", elbow_diff * 0.3, f"elbow_angle_diff={raw_values.get('elbow_angle_diff', 0)}度"))
    
    # 腰回転速度
    if weakness.get("waist_speed") != "ok":
        waist_speed = abs(raw_values.get("waist_speed_diff", 0))
        improvements.append(("腰回転速度", "腰の回転速度が不足しています", waist_speed * 0.1, f"waist_speed_diff={raw_values.get('waist_speed_diff', 0)}度/秒"))
    
    # トスとのタイミング
    if weakness.get("toss_sync") != "ok":
        toss_sync = abs(raw_values.get("toss_sync_diff", 0))
        improvements.append(("トスとのタイミング", "トスとのタイミングが合っていません", toss_sync * 5, f"toss_sync_diff={raw_values.get('toss_sync_diff', 0)}"))
    
    # 打点前後位置
    if weakness.get("impact_forward") != "ok":
        impact_fwd = abs(raw_values.get("impact_forward_diff", 0))
        improvements.append(("打点前後位置", "打点の前後位置が適切ではありません", impact_fwd * 6, f"impact_forward_diff={raw_values.get('impact_forward_diff', 0)}"))
    
    # その他の項目（優先度は低め）
    if weakness.get("wrist_angle") != "ok":
        wrist_diff = abs(raw_values.get("wrist_angle_diff", 0))
        improvements.append(("手首の使い方", "手首の角度が適切ではありません", wrist_diff * 0.2, f"wrist_angle_diff={raw_values.get('wrist_angle_diff', 0)}度"))
    
    if weakness.get("shoulder_tilt") != "ok":
        tilt_diff = abs(raw_values.get("shoulder_tilt_diff", 0))
        improvements.append(("肩のバランス", "左右の肩の高さが不均衡です", tilt_diff * 4, f"shoulder_tilt_diff={raw_values.get('shoulder_tilt_diff', 0)}"))
    
    if weakness.get("body_sway") != "ok":
        sway_diff = abs(raw_values.get("body_sway_diff", 0))
        improvements.append(("体軸の安定性", "体軸が左右にブレています", sway_diff * 3, f"body_sway_diff={raw_values.get('body_sway_diff', 0)}"))
    
    # 重要度スコアでソート（高い順）
    improvements.sort(key=lambda x: x[2], reverse=True)
    
    return improvements

def generate_menu_detail(menu_name: str, diagnosis: Dict) -> str:
    """
    特定の練習メニューの詳細な練習方法を生成
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEYが環境変数に設定されていません")
        client = OpenAI(api_key=api_key)
        prompt = f"""
あなたはソフトテニスのプロコーチです。13歳の右利きプレイヤーのサーブフォーム解析データをもとに、以下の練習メニューの詳細な練習方法を説明してください。

【練習メニュー名】
{menu_name}

【選手の解析データ】
{json.dumps(diagnosis, ensure_ascii=False)}

以下の項目を含めて、具体的で実践的な練習方法を説明してください：

    具体的な練習方法
   - 動作の順序、タイミング、回数、セット数、時間などを具体的に示してください。
   - コートでの練習方法を詳しく説明してください。

    自宅でできる練習
   - コートがなくてもできる練習方法を説明してください。
   - 必要な道具があれば明記してください。

    目標数値・目安
   - 改善の定量的な目標を示してください（例：角度、移動距離、タイミングなど）。

    注意点・コツ
   - 練習時の注意点や上達のコツを説明してください。

出力は読みやすく、箇条書きや見出しを使って構造化してください。初心者にも分かりやすく、かつ上級者も参考にできる内容にしてください。
"""
        res = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"練習メニュー詳細の生成中にエラーが発生しました: {e}"

def generate_ai_menu(diagnosis: Dict) -> str:
    """
    OpenAI APIを使い自然文アドバイスを生成
    例外発生時は詳細なエラーメッセージを返す
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEYが環境変数に設定されていません")
        client = OpenAI(api_key=api_key)
        
        # 改善優先度を自動計算
        priorities = calculate_improvement_priority(diagnosis)
        
        # 優先度リストを文字列に変換
        priority_text = ""
        if priorities:
            priority_text = "\n【改善優先度リスト】（重要度順）\n"
            for idx, (name, desc, score, value) in enumerate(priorities, 1):
                priority_text += f"{idx}. {name}: {desc} ({value})\n"
        else:
            priority_text = "\n【改善優先度リスト】\n現在、特に改善が必要な項目はありません。現在のフォームを維持しながら、さらなる精度向上を目指してください。\n"
        
        player_info = diagnosis.get("player", {})
        age = player_info.get("age", 13)
        hand = player_info.get("hand", "right")
        
        prompt = f"""
この選手のデータです。
{json.dumps(diagnosis, ensure_ascii=False)}

{priority_text}

あなたはソフトテニスのプロコーチです。{age}歳の{hand}利きプレイヤーのサーブフォーム解析データをもとに、毎日15分でできるサーブ改善プログラムを作成してください。

**重要：上記の改善優先度リストの順序を必ず守ってください。簡潔でわかりやすいアドバイスを提示してください。**

条件：
1. 改善優先度リストの順序通りに、改善点ごとに「改善点・理由・具体的練習法・自宅練習・目標数値・効果測定法」をセットで2セットだけ提示する。追加はなし。
2. 練習は短時間（15分以内）で効果が出るように、効率的に組み合わせる。
3. 自宅練習や簡易チェック法も必ず含める。
4. 練習法は、動作順序・タイミング・回数・秒数など、再現性のある具体的な指示を含める。
5. 目標数値を示す（例：腰回転角度〇度、体重移動〇％など）、改善の定量的目安を必ず提示。
6. 語り口は論理的・簡潔に、やさしく、初心者にも分かりやすく、かつ上級者も参考にできるレベルで。

出力形式の例：
---
【改善した方がいいところ】ウェイトトランスファー（体重移動）
- 理由：weight_diff=-0.054のため、前足への体重移動が不足している。パワー伝達のため最優先改善。
- 具体的練習法：トス→後ろ脚に体重を乗せ→ボール頂点で前脚に体重移動を爆発的に行う。スローで10回、3セット。
- 自宅練習：片足立ちで前後交互に体重移動、ゴムバンドを使って抵抗を付加。1セット30秒×2。
- 目標数値：前脚に体重70〜75％移動させる。
- 効果測定法：動画撮影で体重移動タイミングを確認。専用解析装置でweight_diff改善を確認。

【次に意識するといいところ】ボディオープン（体の開き）
- 理由：open_diff=-0.058。さらに安定したフォームで打点での力を伝えるため改善。
- 具体的練習法：胸をネットに平行に保ちつつ、腰45°・肩30°の開きでラケットを振る。スローで10回、3セット。
- 自宅練習：壁ドリル・ミラー前シャドウスイングで開き角度確認。
- 目標数値：腰45°・肩30°の開きで安定した打点。
- 効果測定法：動画で腰・肩角度をチェック、スピン量やラケットフェイス向きの変化を確認。
---

上記フォーマットで、改善優先度リストの順序通りに、毎日15分でできる効率的プログラムを作ってください。
"""
        res = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"AIコーチ生成時にエラーが発生しました: {e}"
