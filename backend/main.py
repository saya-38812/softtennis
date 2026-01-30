from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
from dotenv import load_dotenv
from ai.video_pose import analyze_video
from ai.coach_generator import generate_menu_detail

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
os.makedirs(UPLOAD_DIR, exist_ok=True)

class MenuDetailRequest(BaseModel):
    menu_name: str
    diagnosis: dict

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    動画ファイルを受け取り、解析&コーチングアドバイスAPI
    Returns: 診断情報, 練習メニュー（リスト＆自然文）
    """
    path = f"{UPLOAD_DIR}/{file.filename}"
    try:
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = analyze_video(path)  # dict: diagnosis, menu, ai_text
        return {"status": "ok", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析中エラー: {e}")
    finally:
        # 解析完了後、成功・失敗に関わらずファイルを削除
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                # 削除失敗はログに記録するが、レスポンスには影響させない
                print(f"警告: ファイル削除に失敗しました ({path}): {e}")

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
