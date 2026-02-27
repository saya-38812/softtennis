# サーブノート — ソフトテニス サーブ診断アプリ

AIを活用したソフトテニスのサーブフォーム分析・コーチングアプリケーションです。動画をアップロードするだけで、MediaPipeによる骨格検出とOpenAI APIによる身体感覚コーチングを提供します。

## 概要

サーブ動画を1本アップロードすると、AIが骨格を検出してフォームを数値化し、5つの指標で100点満点のスコアと改善アドバイスを即座に返します。前回の診断結果との比較による改善度フィードバックも自動で表示されます。

## 主な機能

### 1. 動画解析
- MP4形式のサーブ動画をアップロード（1本ずつ即時解析）
- OpenCVによるフレーム間差分からインパクトフレーム（動き量最大の瞬間）を自動検出
- MediaPipeによるインパクト前後 ±1.5秒の骨格抽出（33ランドマーク）
- 骨格付き解析動画（MP4）の自動生成・再生（スロー再生対応）

### 2. 診断項目（5項目）

| 指標 | 内部キー | 計測方法 | 判定基準（理想） |
|---|---|---|---|
| **トス高さ** | `impact_height` | 手首のY座標（中央値） | Y ≤ -2.2 |
| **肘の余裕** | `elbow_angle` | 肩-肘-手首の角度（中央値） | ≥ 135° |
| **体の安定** | `body_sway` | 骨盤-肩中心のX偏差（中央値） | ≤ 0.15 |
| **腰のキレ** | `waist_speed` | 腰回転の角速度（中央値） | ≥ 300°/s |
| **体重移動** | `weight_transfer` | 左右股関節のY座標差（中央値） | ≥ 0.02 |

5指標を重み付きスコアリングし、**最も改善が必要な1つの指標**を自動選定します。

### 3. スコア表示
- 各指標を0〜100点に正規化し、5指標の平均を **Overall Score** として表示
- 改善ポイント（focus_label）に応じた理想値 vs ユーザー値の比較を表示
- 70点未満の指標はオレンジで警告表示

### 4. 視覚的比較
- インパクトフレームの成功フォーム画像とユーザーフォーム画像を自動生成
- 改善ポイントに応じた描画ガイドライン
  - トス高さ：横ライン（理想の高さ vs 実際の高さ）
  - 体の安定 / 腰のキレ：縦ライン（理想軸 vs 現在軸）
  - 肘の余裕：ターゲットマーク
- 骨格オーバーレイ動画（ネオン風スケルトン + インパクトポイント強調）

### 5. AIコーチング
- OpenAI GPT-4o-mini による身体感覚コーチングを自動生成
- **ai_text**：1行の動作イメージ（比喩表現・専門用語なし）
- **practice**：1行の具体的な単一練習（回数指示付き）

### 6. セッション管理と改善度フィードバック
- セッションIDでスコアを蓄積し、複数本のサーブから外れ値除去（IQR）＋平均で最終診断を算出
- 前回セッションとの差分を自動計算し、改善メッセージを生成
  - 例：「前回より打点が10%高くなっています」「前回より肘の角度が5.0度改善しています」
- 初回セッションは「基準を作りました」と表示

### 7. データリセット
- 全セッション・プレイヤー履歴・出力ファイルを一括削除するリセット機能

## 技術スタック

### バックエンド
- **Python** + **FastAPI**（RESTful API）
- **MediaPipe** Pose Landmarker（`pose_landmarker_full.task`）
- **OpenCV**（動画読み込み・骨格描画・MP4書き出し）
- **NumPy**（数値計算・統計処理）
- **OpenAI API**（GPT-4o-mini によるコーチング生成）
- **Uvicorn**（ASGIサーバー）

### フロントエンド
- **React 19** + **Vite 6**
- **Axios**（API通信）
- **CSS3**（ダークテーマ UI、アニメーション付き）

### デプロイ構成
- バックエンド：Render
- フロントエンド：Vercel
- API Base は環境変数 `VITE_API_BASE` またはホスト名で自動切り替え

## セットアップ

### 前提条件
- Python 3.8以上
- Node.js 16以上
- OpenAI APIキー

### バックエンド

```bash
git clone <repository-url>
cd soft-tennis/backend

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

`.env` ファイルを `backend/` に作成：

```env
OPENAI_API_KEY=your_openai_api_key_here
```

以下のファイルが存在することを確認：
- `backend/ai/models/pose_landmarker_full.task`（MediaPipeモデル）
- `backend/ai/success.mp4`（お手本フォーム動画）

### フロントエンド

```bash
cd soft-tennis/frontend
npm install
```

## 実行方法

### バックエンド

```bash
cd backend
venv\Scripts\activate
uvicorn main:app --reload --port 8000
```

`http://127.0.0.1:8000` で起動します。

### フロントエンド

```bash
cd frontend
npm start
```

`http://localhost:3000` で起動します。

## プロジェクト構造

```
soft-tennis/
├── backend/
│   ├── main.py                     # FastAPI サーバー（APIエンドポイント、統計処理、改善計算）
│   ├── player_store.py             # プレイヤーの前回スコア読み書き
│   ├── session_store.py            # セッション単位のスコア蓄積・管理
│   ├── requirements.txt            # Python 依存パッケージ
│   ├── .env                        # 環境変数（要作成）
│   ├── build/                      # フロントエンドのビルド成果物
│   └── ai/
│       ├── video_pose.py           # メイン解析ロジック（インパクト検出・指標計算・画像生成）
│       ├── video_pose_analyzer.py  # MediaPipe による骨格抽出（インパクト周辺のみ高速処理）
│       ├── video_renderer.py       # 骨格オーバーレイ動画の生成
│       ├── normalize_pose.py       # 骨格データの正規化（腰中心・肩幅基準）
│       ├── angle_utils.py          # 角度・距離・速度の計算ユーティリティ
│       ├── coach_generator.py      # OpenAI API による身体感覚コーチング生成
│       ├── coach_ai_utils.py       # 旧練習メニュー生成ロジック
│       ├── evaluate_video.py       # 旧フレームマッチング解析（後方互換）
│       ├── register_success.py     # お手本登録スクリプト
│       ├── success_register_angle.py # お手本角度登録スクリプト
│       ├── success_angles.json     # お手本の角度データ
│       ├── player_diagnosis.json   # 診断結果出力
│       ├── success.mp4             # お手本フォーム動画
│       └── models/
│           └── pose_landmarker_full.task  # MediaPipeモデル
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # メインコンポーネント（全UI・API通信・スコア正規化）
│   │   ├── App.css                 # ダークテーマ UI スタイル
│   │   ├── main.jsx                # Vite エントリーポイント
│   │   └── index.css               # グローバルスタイル
│   ├── package.json
│   ├── vite.config.js
│   └── public/
│
├── server_storage/                 # 実行時データ（backend配下から分離してホットリロード回避）
│   ├── uploads/                    # アップロード動画の一時保存
│   ├── outputs/                    # 生成された比較画像・解析動画
│   ├── players/                    # プレイヤーの前回スコア（JSON）
│   └── sessions/                   # セッションごとのスコア蓄積（JSON）
│
└── README.md
```

## APIエンドポイント

### POST `/analyze`

サーブ動画をアップロードして解析を実行します。1本ごとに即時結果を返し、同じセッションID内ではスコアを蓄積して平均診断を更新します。

**リクエスト:**
- Content-Type: `multipart/form-data`
- Body: `file`（動画ファイル）
- Header: `X-Session-ID`（任意。省略時は自動生成しレスポンスヘッダーで返却）

**レスポンス（解析成功）:**
```json
{
  "status": "complete",
  "count": 1,
  "scores": {
    "impact_height": -2.5,
    "elbow_angle": 150.3,
    "body_sway": 0.08,
    "waist_speed": 420.0,
    "weight_transfer": 0.04
  },
  "focus_label": "打点の高さ",
  "ai_text": "ボールを空に押し上げるイメージで腕を伸ばしていきましょう",
  "practice": "タオルを上に放り投げる動きを10回繰り返しましょう",
  "user_image": "/outputs/user_1234567890.png",
  "user_video": "/outputs/user_1234567890.mp4",
  "ideal_image": "/outputs/ideal_1234567890.png",
  "improvement": {
    "impact_height": 0.10,
    "elbow_angle": 5.0,
    "body_sway": -0.02,
    "waist_speed": 30.0,
    "weight_transfer": 0.01
  },
  "improvement_message": "前回より打点が10%高くなっています"
}
```

**レスポンス（解析失敗 — 422）:**
```json
{
  "status": "error",
  "message": "動画から骨格を正しく検出できませんでした。..."
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
  "detail": "・目的：○○\n・やり方：○○\n・回数：10回×3\n・注意：○○"
}
```

### POST `/reset-all`

全てのセッションデータ・プレイヤー履歴・出力ファイルを一括削除します。

**レスポンス:**
```json
{
  "status": "ok",
  "message": "全てのデータをリセットしました"
}
```

## 使用方法

1. **動画の準備** — サーブの動画をMP4形式で用意。全身が映る構図を推奨
2. **アップロード** — トップ画面の円形ボタンをタップし、動画を1本選択
3. **自動解析** — 骨格検出・指標計算・AI コーチング生成が自動実行される
4. **結果確認**
   - 骨格オーバーレイ動画のスロー再生
   - Overall Score（100点満点）と5指標のバー表示
   - 最優先の改善ポイントと理想値 vs ユーザー値の比較
   - AI コーチの身体感覚アドバイス＋練習メニュー
   - 前回からの改善度メッセージ
5. **繰り返し** — 「もう一回撮影」ボタンで次の動画をアップロード

## 注意事項

- 動画ファイルは解析完了後に自動削除されます
- 生成された画像・動画は最新15ファイルのみ保持し、古いものは自動削除されます
- OpenAI API の使用にはAPIキーが必要です（有料）
- 現在のバージョンは**右利きプレイヤー**を想定しています
- 骨格データの正規化は腰の中心を原点、肩幅をスケール基準として実施します

## トラブルシューティング

### 骨格検出が失敗する場合
- プレイヤーの全身が動画に映っているか確認してください
- 動画の解像度・照明条件を改善してください

### APIエラーが発生する場合
- `.env` に `OPENAI_API_KEY` が正しく設定されているか確認してください
- ネットワーク接続を確認してください

### サーバーが起動しない場合
- 仮想環境が有効化されているか確認してください
- 必要なパッケージがインストールされているか確認してください
- ポート8000が使用可能か確認してください

## 解析パイプライン

```
動画アップロード
  │
  ├─ 1. インパクト検出（OpenCV フレーム間差分 → 動き量最大フレーム）
  │
  ├─ 2. 骨格抽出（MediaPipe Pose → インパクト ±1.5秒のランドマーク）
  │
  ├─ 3. 正規化（腰中心・肩幅スケール → xy座標）
  │
  ├─ 4. 指標計算（角度・距離・速度の中央値）
  │
  ├─ 5. 重み付きスコアリング → 最重要課題の選定
  │
  ├─ 6. 画像・動画生成（骨格オーバーレイ + ガイドライン描画）
  │
  ├─ 7. AIコーチング生成（GPT-4o-mini）
  │
  └─ 8. 改善度計算（前回スコアとの差分）→ レスポンス返却
```

## 今後の開発予定

- [ ] 複数の成功フォームパターンのサポート
- [ ] 左利きプレイヤーの対応
- [ ] 動画の前処理機能（トリミング、速度調整など）
- [ ] 解析結果の履歴保存・グラフ表示
- [ ] モバイルアプリ対応
- [ ] リアルタイム解析機能
