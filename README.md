# サーブノート — ソフトテニス サーブ練習アプリ（MVP）

**ソフトテニスのサーブ練習を記録し、必要に応じて動画からAIでフォーム解析できるアプリ**です。

## アプリ概要

- 練習セッションを開始し、カメラで録画しながらサーブごとに IN / OUT / FAULT を記録する
- セッション終了後に統計（Total / IN / OUT / FAULT / Accuracy）とサーブ一覧を表示
- 「フォーム解析する」で録画を送信し、各サーブの timestamp ±1秒 のクリップを MediaPipe で解析し、AIコーチング（ai_text, practice）を表示

**スポーツ練習アプリ**として設計しており、研究デモではなく実用を想定しています。

## コア機能（MVP）

| 機能 | 説明 |
|------|------|
| **Practice Session** | セッション開始 → 練習画面へ |
| **Serve Tracking** | サーブごとに IN /FAULT をタップして記録（録画内タイムスタンプを保存） |
| **Session Result** | セッション終了後に Total Serves / IN / FAULT / Accuracy と Serve 一覧を表示 |
| **Form Analysis** | セッション結果画面から「フォーム解析する」で動画＋サーブタイムスタンプを送信し、各サーブのフォーム解析結果（スコア・改善ポイント・AIコーチング）を表示 |

## ユーザーフロー

1. ホームで「練習を始める」→ セッション開始
2. 練習画面でカメラ録画が開始され、サーブを打つたびに IN / OUT / FAULT をタップ
3. 「End Session」でセッション終了 → セッション結果画面へ
4. セッション結果で統計・サーブ一覧を確認し、必要なら「フォーム解析する」をクリック
5. 解析完了後、サーブをタップしてフォーム解析（改善ポイント・AIコーチング）を表示

## 技術スタック

- **Backend**: Python, FastAPI, MediaPipe Pose Landmarker, OpenCV, OpenAI API（コーチング生成）
- **Frontend**: Next.js 14 (App Router), React 18, TypeScript, MediaRecorder API

## プロジェクト構造

```
soft-tennis/
├── backend/
│   ├── main.py                 # FastAPI（セッション・サーブ・解析 API）
│   ├── tracker_store.py        # セッション・サーブ・解析の一元管理
│   ├── models/
│   │   ├── session.py
│   │   ├── serve.py
│   │   └── analysis.py
│   ├── ai/
│   │   ├── clip_extract.py     # 動画クリップ抽出（timestamp ±1秒）
│   │   ├── video_pose.py       # インパクト検出など（serve_analysis から利用）
│   │   ├── video_pose_analyzer.py
│   │   ├── normalize_pose.py
│   │   ├── angle_utils.py
│   │   ├── serve_analysis.py   # クリップ1本のフォーム解析
│   │   ├── coach_generator.py  # OpenAI コーチング（ai_text, practice）
│   │   └── legacy/            # 旧解析コード（MVP では未使用）
│   └── requirements.txt
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx            # ホーム（練習を始める）
│   │   ├── practice/page.tsx   # 練習画面（録画 + IN/OUT/FAULT）
│   │   ├── session-result/page.tsx  # セッション結果（統計・Serve 一覧・フォーム解析）
│   │   └── history/page.tsx    # 履歴
│   ├── components/
│   ├── contexts/
│   ├── lib/
│   └── package.json
│
├── server_storage/
│   └── data/
│       ├── sessions/           # {id}/session.json
│       ├── videos/             # {id}.mp4 or .webm
│       └── outputs/             # 生成物（必要に応じて）
│
└── README.md
```

## データ構造（統一）

- **Session**: `id`, `created_at`, `video_path`, `serves[]`, `analysis{}`
- **Serve**: `id`, `session_id`, `result` (IN / OUT / FAULT), `timestamp`, `timestamp_sec`
- **Analysis** (serve ごと): `scores`, `focus_label`, `ai_text`, `practice`

## API エンドポイント

いずれもプレフィックス `/api` 付き（例: `POST /api/sessions/start`）。ベース URL は `http://localhost:8000`（未設定時）。

| メソッド | パス | 説明 |
|---------|------|------|
| POST | `/api/start-session` | セッション開始。`session_id` を返す。 |
| POST | `/api/serves` | サーブ1本を記録。Body: `{ "session_id", "result": "IN" \| "OUT" \| "FAULT", "timestamp_sec"?: number }` |
| POST | `/api/serves/undo` | 最後の1本を取り消し。Body: `{ "session_id" }` |
| POST | `/api/sessions/end` | セッション終了。集計結果を返す。Body: `{ "session_id" }` |
| POST | `/api/analysis` | フォーム解析。multipart: `video`, `session_id`, `serve_events` (JSON) |
| GET | `/api/sessions` | セッション一覧（履歴）。 |
| GET | `/api/sessions/{id}` | セッション詳細。 |
| DELETE | `/api/sessions/{id}` | セッション1件を削除。 |

## セットアップ・実行

### バックエンド

```bash
cd backend
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

`.env` に `OPENAI_API_KEY` を設定。  
MediaPipe モデル: `backend/ai/models/pose_landmarker_full.task`  
お手本動画（任意）: `backend/ai/success.mp4`

**必ず `backend` ディレクトリで**次を実行：

```bash
cd backend
uvicorn main:app --reload --port 8000
```

または `python run.py`（backend にいる状態で）。

- 起動後、ブラウザで **http://localhost:8000/api/health** を開き `{"status":"ok","api":"..."}` が返れば OK。
- **404 になる場合**:  port 8000 で**古いプロセスが動いている**可能性があります。ターミナルでそのプロセスを止めてから、上記をやり直してください（例: Ctrl+C で uvicorn を止める。別ターミナルで起動したままの backend がないか確認）。

### フロントエンド

```bash
cd frontend
npm install
npm run dev
```

環境変数 `NEXT_PUBLIC_API_BASE` で API のベース URL を指定（未設定時は `http://localhost:8000`）。

## 注意事項

- 録画はブラウザの MediaRecorder で行い、フォーム解析時に Blob をサーバへ送信します。
- OpenAI API の利用には API キー（有料）が必要です。
- 現在は**右利きプレイヤー**を想定しています。
