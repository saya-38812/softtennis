from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
from dotenv import load_dotenv
from ai.video_pose import analyze_video
from ai.coach_generator import generate_menu_detail

from fastapi.staticfiles import StaticFiles
# .envファイルから環境変数を読み込む（アプリケーション起動時）
load_dotenv()

app = FastAPI()
# フロントエンドからアクセスできるようにCORS設定（開発用：全許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "uploads"

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")




class MenuDetailRequest(BaseModel):
    menu_name: str
    diagnosis: dict

import traceback

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    path = f"{UPLOAD_DIR}/{file.filename}"

    try:
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = analyze_video(path)
        return {"status": "ok", **result}

    except Exception as e:
        print("🔥 ANALYZE ERROR:", e)
        traceback.print_exc()   # ←これが最重要

        raise HTTPException(
            status_code=500,
            detail="解析中にサーバー内部でエラーが発生しました"
        )

    finally:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print("⚠️ ファイル削除失敗:", e)


@app.post("/menu-detail")
async def get_menu_detail(request: MenuDetailRequest):
    """
    練習メニューの詳細な練習方法を取得
    """
    try:
        detail = generate_menu_detail(request.menu_name, request.diagnosis)
        return {"status": "ok", "detail": detail}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"詳細生成中エラー: {e}")

from fastapi.staticfiles import StaticFiles

# React buildを配信する
app.mount("/", StaticFiles(directory="build", html=True), name="static")
