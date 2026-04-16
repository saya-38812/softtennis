# MVP リファクタリング概要

## 1. 改善されたアーキテクチャ

- **単一データソース**: `tracker_store` のみ。`server_storage/data/` 配下に `sessions/`, `videos/`, `outputs/` を統一。
- **統一データモデル**: Session（id, created_at, video_path）→ Serves[]（id, session_id, result, timestamp）→ Analysis{ serve_id: { scores, focus_label, ai_text, practice } }。
- **解析API 1本化**: フォーム解析は **POST /analysis** のみ。入力は動画＋serve_events（timestamp_sec 付き）。処理は timestamp ±1秒のクリップ抽出 → MediaPipe 骨格 → 指標計算 → OpenAI コーチング（ai_text, practice）。
- **フロント**: ルートは `/`（ホーム）, `/practice`, `/session-result`, `/history` のみ。アップロード専用ページは廃止。フォーム解析はセッション結果画面の「フォーム解析する」ボタンから実行。

## 2. 新しいディレクトリ構造

```
backend/
  main.py
  tracker_store.py
  models/
    session.py, serve.py, analysis.py
  ai/
    clip_extract.py, video_pose.py, video_pose_analyzer.py,
    normalize_pose.py, angle_utils.py, serve_analysis.py, coach_generator.py
    legacy/   # evaluate_video, coach_ai_utils, register_success, success_register_angle

server_storage/
  data/
    sessions/   # {id}/session.json
    videos/     # {id}.mp4 or .webm
    outputs/    # 生成物
```

## 3. 削除・無効化した機能・コード

- **速度測定**: `/api/analyze_speed`、平均・最大速度の記録・表示を削除。
- **チャレンジモード**: フロントの `useChallengeUnlock.ts` を削除。
- **SSE 解析**: `POST /analyze`（SSE）を削除。
- **旧解析**: `POST /api/analyze_form`, `POST /api/analyze` を削除。`evaluate_video.py`, `coach_ai_utils.py`, `register_success.py`, `success_register_angle.py` は `ai/legacy/` に移動（MVP では未使用）。
- **player_store / session_store**: 削除。セッション・サーブ・解析は `tracker_store` に集約。

## 4. 主なコード修正

- **tracker_store**: 保存先を `server_storage/data/{sessions,videos,outputs}` に変更。Serve から `speed` を削除。`video_path` を session に保存。
- **main.py**: 上記削除に合わせてエンドポイントを整理。API パスを `POST /sessions/start`, `POST /serves`, `POST /serves/undo`, `POST /sessions/end`, `POST /analysis`, `GET /sessions`, `GET /sessions/{id}`, `DELETE /sessions/{id}` に統一。
- **フロント api.ts**: 上記パスに対応する `startSession`, `recordServe`, `undoServe`, `endSession`, `getSession`, `listSessions`, `deleteSession`, `runFormAnalysis` に整理。SSE・単体動画解析の関数は削除。
- **session-result**: 自動解析をやめ、「フォーム解析する」ボタンで `runFormAnalysis` を呼ぶ形に変更。
- **Practice**: 上部に Attempts / IN / OUT / FAULT / Accuracy を表示する UI に変更（TrackerStats）。

## 5. README

`README.md` を MVP 用に更新済み（アプリ概要、コア機能 4 つ、ユーザーフロー、技術スタック、プロジェクト構造、データ構造、API 一覧、セットアップ・注意事項）。
