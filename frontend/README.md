# Soft Tennis Serve Analyzer (Frontend)

Next.js + TypeScript のモバイルファースト UI です。

## ページ

- **/** — ホーム（タイトル + 「Start Analysis」→ `/upload`）
- **/upload** — 動画アップロード + 「Analyze Serve」で `POST /api/analyze` に送信
- **/result** — スコア・フィードバック・メトリクス表示

## 開発

```bash
npm install
npm run dev
```

[http://localhost:3000](http://localhost:3000) で表示。バックエンドは別途 `http://localhost:8000` で起動してください。

## 環境変数

| 変数 | 説明 |
|------|------|
| `NEXT_PUBLIC_API_BASE` | バックエンドのベースURL（未設定時は `http://localhost:8000`） |

本番で別ホストにデプロイする場合は例: `NEXT_PUBLIC_API_BASE=https://api.example.com`

## ビルド

```bash
npm run build
npm start
```
