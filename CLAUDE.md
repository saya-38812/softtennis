# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

ソフトテニスのサーブ練習アプリ「サーブノート」。ブラウザでカメラ録画しながら IN/FAULT を記録し、MediaPipe で骨格解析してフォームフィードバックを表示する。

## 起動コマンド

### バックエンド（`backend/` ディレクトリで実行）
```powershell
cd backend
# 初回のみ: 依存関係インストール
pip install -r requirements.txt
# 起動
python run.py
# または
uvicorn main:app --reload --port 8000
```
- 確認: `GET http://localhost:8000/api/health` が `{"status":"ok"}` を返せばOK
- **必ず `backend/` ディレクトリにいる状態で起動する**（相対importのため）

### フロントエンド（`frontend/` ディレクトリで実行）
```powershell
cd frontend
npm install   # 初回のみ
npm run dev
# 型チェックのみ
npx tsc --noEmit
# プロダクションビルド
npm run build
```

### 環境変数
- `backend/.env` に `OPENAI_API_KEY=sk-...`（コーチングテンプレート使用中は不要）
- フロントの `NEXT_PUBLIC_API_BASE` は未設定時 `http://localhost:8000`

## アーキテクチャ

### データフロー
```
練習画面（MediaRecorder録画 + IN/FAULT タップ）
  → End Session → videoBlob を SessionVideoContext に保存
  → session-result 画面に遷移
  → 自動で POST /api/analysis（multipart: video + serve_events）
  → バックエンドがサーブごとにクリップ抽出 → MediaPipe 解析 → テンプレートコーチング
  → session.json に解析結果を保存、トラッキング動画を生成
  → フロントに結果を返す
```

### バックエンド構成

**`tracker_store.py`** がデータ永続化の唯一の窓口。セッション・サーブ・解析を全て `server_storage/data/` 配下の JSON + 動画ファイルで管理する。DBは使わない。

```
server_storage/data/
  sessions/{id}/session.json   # serves[], analysis{} を含む
  videos/{id}.webm             # 録画動画
  outputs/tracking_{id}.mp4   # 骨格描画済み動画
```

**AI解析パイプライン**（`backend/ai/`）:
1. `clip_extract.py` — timestamp ±2秒のクリップ切り出し（OpenCV）
2. `video_pose.py` — インパクトフレーム自動検出
3. `video_pose_analyzer.py` — MediaPipe PoseLandmarker でランドマーク抽出（モデル: `ai/models/pose_landmarker_full.task`、初回起動時に遅延ロード）
4. `serve_analysis.py` — 肘角度・体の揺れ・インパクト高さを計算してスコア化
5. `coach_templates.py` — スコアに応じたコーチングメッセージをランダム選択（OpenAI不使用）
6. `video_renderer.py` — 骨格ランドマークをオーバーレイした動画を生成

**コーチングの決定ロジック**（`main.py` の `_analyze_clip_to_result`）:
- `body_sway > 0.15` → focus: "体の安定"
- `body_sway ≤ 0.15` かつ肘角度が90〜130°外 → focus: "肘の余裕"
- `body_sway ≤ 0.15` かつ肘OK かつ `impact_height < 1.5` → focus: "トス高さ"

### フロントエンド構成

**ページ構成**（Next.js App Router）:
- `/` — ホーム（練習開始ボタン）
- `/practice` — 録画 + IN/FAULT タップ画面
- `/session-result` — 統計・フォーム解析結果（`?session_id=` で取得）
- `/history` — セッション一覧
- `/stats` — 全セッションの集計統計
- `/upload` — 既存動画をアップロードして解析

**重要な状態管理**:
- `contexts/SessionVideoContext.tsx` — 練習画面で録画した Blob を session-result に渡すための Context。ページ遷移をまたいで videoBlob を保持する。
- session-result の自動解析トリガー: `videoBlob` が存在 かつ `analysis` がまだない場合に `useEffect` で自動実行。`analysisTriggeredRef` で二重実行を防ぐ。

**共通ユーティリティ**:
- `lib/api.ts` — バックエンドAPI全呼び出し。`getApiBase()` で環境変数を参照。
- `lib/uploadResult.ts` — `ServeAnalysis` → 表示用スコア変換（`fromAnalysis`）。upload/session-result 両方から参照するため共通化済み。
- `lib/utils.ts` — `accuracyPercent(session)` など集計ヘルパー。

**コンポーネント**:
- `components/tracker/TrackerStats.tsx` — Serves/IN/FAULT/IN率 の4列統計カード
- `components/result/UploadStyleResultView.tsx` — アップロードフロー専用の全画面解析結果ビュー（session-result では使わず、直接インライン表示）
- `components/layout/LayoutWithTabs.tsx` — 全ページ共通のヘッダー＋BottomTabBar ラッパー

## 注意事項

- サーブの結果は **IN / FAULT のみ**（OUT はラリー用でサーブには存在しない）
- フォーム解析は **右利きプレイヤー** を前提とした実装
- `backend/ai/legacy/` は旧解析コードの退避場所。MVP では未使用
- サンプルデータは起動時にセッション0件なら自動作成される（`ensure_sample_sessions`）。履歴画面の「サンプルデータを読み込む」ボタンからも手動追加可能
- Next.js API Route（`/api/seed`）はフロントからバックエンドへのプロキシとして機能（CORS回避）
