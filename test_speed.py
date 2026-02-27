import sys
import os
import time
import logging

# backendディレクトリをパスに追加
sys.path.append(os.path.join(os.getcwd(), "backend"))

from backend.ai.video_pose import analyze_video

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    start_time = time.time()
    video_path = "backend/ai/success.mp4"
    print(f"Testing analyze_video on {video_path}...")
    
    # 実際には analyze_video は内部で success.mp4 を参照するので、
    # 自身をテスト対象にする
    try:
        result = analyze_video(video_path)
        end_time = time.time()
        print(f"Analysis completed in {end_time - start_time:.2f} seconds")
        print(f"Status: {result.get('status', 'N/A')}")
        if "ai_text" in result:
            print(f"AI Text: {result['ai_text']}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
