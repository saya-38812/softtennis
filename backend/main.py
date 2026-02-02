from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import shutil
import os
import traceback
from dotenv import load_dotenv

from ai.video_pose import analyze_video
from ai.coach_generator import generate_menu_detail

# ============================
# 環境変数読み込み
# ============================
load_dotenv()

# ============================
# FastAPI起動
# ============================
app = FastAPI()

# ============================
# CORS（フロント許可）
# ============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# ディレクトリ準備
# ============================
BASE_DIR = os.path.dirname(__file__)

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# 静的ファイル（画像）
# ============================
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# ============================
# リクエストモデル
# ============================
class MenuDetailRequest(BaseModel):
    menu_name: str
    diagnosis: dict

# ============================
# 動画解析API
# ============================
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # 保存
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # AI解析
        result = analyze_video(path)

        return {"status": "ok", **result}

    except Exception as e:
        print("🔥 ANALYZE ERROR:", e)
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"解析中にエラーが発生しました: {str(e)}"
        )

    finally:
        # ファイル削除
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print("⚠️ ファイル削除失敗:", e)

# ============================
# 練習メニュー詳細API
# ============================
@app.post("/menu-detail")
async def get_menu_detail(request: MenuDetailRequest):

    try:
        detail = generate_menu_detail(request.menu_name, request.diagnosis)
        return {"status": "ok", "detail": detail}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"詳細生成エラー: {str(e)}"
        )

# ============================
# React build配信（※注意）
# ============================

# RenderではAPIとReactを分ける方が安全
# どうしても同居するなら /app にする

# app.mount("/", StaticFiles(directory="build", html=True), name="static")

app.mount("/app", StaticFiles(directory="build", html=True), name="static")
