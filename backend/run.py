#!/usr/bin/env python3
"""
バックエンドを port 8000 で起動。
必ず backend ディレクトリから実行: python run.py
"""
import os
import sys

# このファイルがある backend をカレントに
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != BACKEND_DIR:
    os.chdir(BACKEND_DIR)
    sys.path.insert(0, BACKEND_DIR)

if __name__ == "__main__":
    import uvicorn
    print("Starting server at http://127.0.0.1:8000")
    print("API: POST http://127.0.0.1:8000/api/sessions/start")
    print("Check: GET http://127.0.0.1:8000/api/health")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
