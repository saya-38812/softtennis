FROM python:3.11-slim

# MediaPipe / OpenCV がサーバー環境で必要とするシステムライブラリをインストール
# Debian trixie では libgl1-mesa-glx → libgl1 に名前変更されている
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgles2 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存ライブラリを先にインストール（キャッシュを効かせるため）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# バックエンドのコードをコピー
COPY . .

# Render が $PORT 環境変数でポートを指定する
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
