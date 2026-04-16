"""
解析結果用のコーチングメッセージテンプレート。
メトリクスごとに cue（イメージ）と practice（練習）を固定文から1つずつ選んで返す。
"""
import random
from typing import Dict, List

# メトリクスキー → 表示ラベル
FOCUS_LABELS: Dict[str, str] = {
    "impact_height": "トス高さ",
    "elbow_angle": "肘の余裕",
    "body_sway": "体の安定",
    "waist_speed": "腰のキレ",
    "weight_transfer": "体重移動",
}

# メトリクスごとの cue（身体感覚のイメージ）と practice（練習）
COACH_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "impact_height": {
        "cues": [
            "ボールを「空に押し上げる」イメージで腕を伸ばしてください",
            "トスを頭の上ではなく「もう一段高い空」に上げる感覚です",
            "打点を「背伸びして届く高さ」に作りましょう",
            "ボールを取りに行くのではなく「落ちてくるのを待つ」感覚です",
        ],
        "practices": [
            "トスだけを10回連続で上げる練習をしてください",
            "ラケットなしでトス→キャッチを10回繰り返してください",
            "壁に向かってトスだけを10回行い高さを安定させてください",
        ],
    },
    "elbow_angle": {
        "cues": [
            "肘を少し外に張って「弓を引く」形を作りましょう",
            "腕を伸ばすより「肘をたたむ感覚」で振ってください",
            "ラケットを背中に担ぐイメージで準備してください",
            "腕ではなく「肘から先がムチのように動く」感覚です",
        ],
        "practices": [
            "タオルを持ってサーブ動作を10回行ってください",
            "肘を意識してシャドースイングを10回行ってください",
            "ラケットを持たずに腕だけでサーブ動作を10回行ってください",
        ],
    },
    "body_sway": {
        "cues": [
            "頭の位置を動かさず「軸を一本立てる」イメージです",
            "体が横に流れないよう「柱の周りで回る」感覚です",
            "打つ瞬間は「頭が静止している」状態を意識してください",
            "上半身をぶらさず「その場で回転する」イメージです",
        ],
        "practices": [
            "足を揃えてサーブ動作を10回行ってください",
            "壁に背中を近づけてシャドースイングを10回行ってください",
            "鏡を見ながら体のブレを確認して10回スイングしてください",
        ],
    },
    "waist_speed": {
        "cues": [
            "腰から先に回すイメージで体を回転させましょう",
            "「腰 → 肩 → 腕」の順番で動く感覚を作ってください",
            "下半身でボールを押し出すイメージです",
            "腰を素早く切り替える意識で振りましょう",
        ],
        "practices": [
            "腰だけを回すシャドースイングを10回行ってください",
            "下半身を意識してサーブ動作を10回繰り返してください",
            "軽くジャンプして着地からサーブ動作を10回行ってください",
        ],
    },
    "weight_transfer": {
        "cues": [
            "前足に体重を乗せながら打つ感覚です",
            "ボールを「前に押し出す」イメージで振りましょう",
            "後ろ足から前足へ体重を流してください",
            "前に踏み込みながら打つ感覚を作りましょう",
        ],
        "practices": [
            "前に一歩踏み込みながらサーブ動作を10回行ってください",
            "ステップ→スイングを10回繰り返してください",
            "前足に体重を乗せる練習を10回行ってください",
        ],
    },
}


def get_coaching_from_template(focus_key: str) -> Dict[str, str]:
    """
    メトリクスキーに応じてテンプレートから cue と practice を1つずつ選んで返す。

    Args:
        focus_key: impact_height / elbow_angle / body_sway / waist_speed / weight_transfer

    Returns:
        {"focus_label": "表示名", "ai_text": "cue文", "practice": "練習文"}
    """
    if focus_key not in COACH_TEMPLATES:
        focus_key = "body_sway"
    t = COACH_TEMPLATES[focus_key]
    cue = random.choice(t["cues"]) if t["cues"] else "身体感覚で動きを改善しましょう"
    practice = random.choice(t["practices"]) if t["practices"] else "サーブを10回繰り返してください"
    label = FOCUS_LABELS.get(focus_key, "体の安定")
    return {
        "focus_label": label,
        "ai_text": cue,
        "practice": practice,
    }
