# ソフトテニス サーブ診断アプリ

AIを活用したソフトテニスのサーブフォーム分析・コーチングアプリケーションです。動画をアップロードするだけで、MediaPipeによる骨格検出とOpenAI APIによる詳細なアドバイスを提供します。

## 概要

このアプリケーションは、ソフトテニスプレイヤーのサーブフォームを動画から自動解析し、以下の機能を提供します：

- **骨格検出**: MediaPipeを使用した高精度なポーズ検出
- **フォーム分析**: 成功フォームとの比較による詳細な診断
- **弱点特定**: 13項目以上の指標による包括的な分析
- **練習メニュー提案**: 診断結果に基づいた個別化された練習メニュー
- **AIコーチング**: OpenAI GPT-4による詳細なアドバイスと改善プログラム

## 主な機能

### 1. 動画解析
- MP4形式のサーブ動画をアップロード
- MediaPipeによるリアルタイム骨格検出
- 成功フォーム（`success.mp4`）との比較分析

### 2. 診断項目
以下の13項目以上の指標を分析します：

- **基本指標**
  - インパクト高さ
  - 体重移動
  - 体の開き

- **角度指標**
  - 肩の開き角度
  - 肘の角度
  - 手首の角度
  - 肩の高さ差（左右バランス）

- **動的指標**
  - 腰の回転角度
  - 腰回転速度
  - 体軸ブレ量

- **位置・タイミング指標**
  - 打点前後位置
  - トスとのタイミング
  - 打点左右偏差
  - 体重左右分布

### 3. AIコーチング
- 改善優先度の自動計算
- 毎日15分でできる効率的な練習プログラム
- 具体的な練習方法、自宅練習、目標数値、効果測定法の提示
- 練習メニューの詳細説明

## 技術スタック

### バックエンド
- **Python 3.x**
- **FastAPI**: RESTful APIサーバー
- **MediaPipe**: ポーズ検出・骨格分析
- **OpenCV**: 動画処理
- **NumPy**: 数値計算
- **OpenAI API**: GPT-4によるコーチング生成
- **Uvicorn**: ASGIサーバー

### フロントエンド
- **React 19**: UIフレームワーク
- **Axios**: HTTPクライアント
- **CSS3**: モダンなUIデザイン

## セットアップ

### 前提条件
- Python 3.8以上
- Node.js 16以上
- OpenAI APIキー

### バックエンドのセットアップ

1. リポジトリをクローン
```bash
git clone <repository-url>
cd soft-tennis
```

2. バックエンドディレクトリに移動
```bash
cd backend
```

3. 仮想環境を作成・有効化
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

4. 依存パッケージをインストール
```bash
pip install -r requirements.txt
```

5. 環境変数を設定
`backend`ディレクトリに`.env`ファイルを作成し、以下を記述：
```env
OPENAI_API_KEY=your_openai_api_key_here
```

6. MediaPipeモデルファイルの確認
`backend/ai/models/pose_landmarker_full.task`が存在することを確認してください。

7. 成功フォーム動画の確認
`backend/ai/success.mp4`が存在することを確認してください。

### フロントエンドのセットアップ

1. フロントエンドディレクトリに移動
```bash
cd frontend
```

2. 依存パッケージをインストール
```bash
npm install
```

## 実行方法

### バックエンドサーバーの起動

```bash
cd backend
# 仮想環境を有効化
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# サーバーを起動
uvicorn main:app --reload --port 8000
```

サーバーは `http://127.0.0.1:8000` で起動します。

### フロントエンドの起動

別のターミナルで：

```bash
cd frontend
npm start
```

ブラウザで `http://localhost:3000` が自動的に開きます。

## プロジェクト構造

```
soft-tennis/
├── backend/
│   ├── main.py                 # FastAPIメインサーバー
│   ├── requirements.txt         # Python依存パッケージ
│   ├── .env                     # 環境変数（要作成）
│   ├── uploads/                 # アップロードファイル一時保存
│   └── ai/
│       ├── video_pose.py        # メイン解析ロジック
│       ├── video_pose_analyzer.py  # 骨格検出
│       ├── normalize_pose.py    # ポーズ正規化
│       ├── angle_utils.py       # 角度計算ユーティリティ
│       ├── coach_ai_utils.py    # 練習メニュー生成
│       ├── coach_generator.py   # AIコーチング生成
│       ├── evaluate_video.py    # 動画評価（旧版）
│       ├── success.mp4          # 成功フォーム動画
│       └── models/
│           └── pose_landmarker_full.task  # MediaPipeモデル
│
├── frontend/
│   ├── src/
│   │   ├── App.js              # メインコンポーネント
│   │   ├── App.css             # スタイル
│   │   └── index.js            # エントリーポイント
│   ├── package.json            # Node.js依存パッケージ
│   └── public/                 # 静的ファイル
│
└── README.md                   # このファイル
```

## APIエンドポイント

### POST `/analyze`
動画ファイルをアップロードして解析を実行します。

**リクエスト:**
- Content-Type: `multipart/form-data`
- Body: `file` (動画ファイル)

**レスポンス:**
```json
{
  "status": "ok",
  "diagnosis": {
    "player": {
      "age": 13,
      "hand": "right",
      "serve_score": 85
    },
    "weakness": {
      "impact_height": "ok",
      "weight_transfer": "poor",
      ...
    },
    "raw_values": {
      "impact_diff": 0.023,
      "weight_diff": -0.054,
      ...
    }
  },
  "menu": ["体重移動練習（強化）", "腰の回転練習（軽め）"],
  "ai_text": "AIコーチからのアドバイス..."
}
```

### POST `/menu-detail`
練習メニューの詳細な練習方法を取得します。

**リクエスト:**
```json
{
  "menu_name": "体重移動練習（強化）",
  "diagnosis": { ... }
}
```

**レスポンス:**
```json
{
  "status": "ok",
  "detail": "具体的な練習方法の説明..."
}
```

## 使用方法

1. **動画の準備**
   - サーブの動画をMP4形式で準備
   - プレイヤー全体が映っている動画が推奨

2. **動画のアップロード**
   - フロントエンドの「動画ファイルを選択」ボタンから動画を選択
   - 「解析を開始」ボタンをクリック

3. **結果の確認**
   - 診断結果が表示されます
   - 推奨練習メニューがリスト表示されます
   - AIコーチからのアドバイスが表示されます

4. **練習メニューの詳細確認**
   - 各メニューの「詳細を見る」ボタンをクリック
   - 具体的な練習方法、自宅練習、目標数値などが表示されます

## 注意事項

- 動画ファイルは解析完了後に自動的に削除されます
- OpenAI APIの使用にはAPIキーが必要です（有料）
- 解析には数秒から数分かかる場合があります
- 動画の品質が低い場合、骨格検出が失敗する可能性があります

## トラブルシューティング

### 骨格検出が失敗する場合
- 動画の解像度を確認してください
- プレイヤーが動画内に完全に映っているか確認してください
- 照明条件を改善してください

### APIエラーが発生する場合
- `.env`ファイルに`OPENAI_API_KEY`が正しく設定されているか確認してください
- APIキーの有効性を確認してください
- ネットワーク接続を確認してください

### サーバーが起動しない場合
- 仮想環境が正しく有効化されているか確認してください
- 必要なパッケージがすべてインストールされているか確認してください
- ポート8000が使用可能か確認してください

## ライセンス

このプロジェクトのライセンス情報については、リポジトリのLICENSEファイルを参照してください。

## 貢献

プルリクエストやイシューの報告を歓迎します。改善提案やバグ報告もお気軽にどうぞ。

## 今後の開発予定

- [ ] 複数の成功フォームパターンのサポート
- [ ] 左利きプレイヤーの対応
- [ ] 動画の前処理機能（トリミング、速度調整など）
- [ ] 解析結果の履歴保存機能
- [ ] モバイルアプリ対応
- [ ] リアルタイム解析機能

