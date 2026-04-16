import cv2
import numpy as np
import logging
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Any

import os
import threading

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker_full.task")

# ==============================
# Render など GPU なしサーバーで MediaPipe が必要とする
# libGLESv2.so.2 が存在しない問題をコードで解決する
# ==============================
def _ensure_gles_stub():
    """
    libGLESv2.so.2 が見つからない環境向けのスタブを作成し、
    LD_LIBRARY_PATH に追加する。
    ctypes.CDLL は dlopen() を使うので、呼び出し前に LD_LIBRARY_PATH を
    変更すれば反映される。
    """
    import ctypes.util
    # すでに libGLESv2 が見つかれば何もしない
    if ctypes.util.find_library("GLESv2"):
        return

    stub_dir = "/tmp/gl_stubs"
    stub_path = os.path.join(stub_dir, "libGLESv2.so.2")

    if not os.path.exists(stub_path):
        os.makedirs(stub_dir, exist_ok=True)
        created = False

        # 方法1: gcc でスタブの .so を作成
        try:
            import subprocess
            # MediaPipe が実際には GPU 関数を呼ばないため空の実装でよい
            c_code = b"void glGenBuffers(){} void glDeleteBuffers(){} void glBindBuffer(){}"
            result = subprocess.run(
                ["gcc", "-x", "c", "-shared", "-fPIC", "-o", stub_path, "-"],
                input=c_code,
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0 and os.path.exists(stub_path):
                created = True
                logging.info("libGLESv2 スタブを gcc で作成しました")
        except Exception as e:
            logging.debug(f"gcc スタブ作成失敗: {e}")

        # 方法2: システムにある別の GL 系ライブラリをコピーして代用
        if not created:
            import glob, shutil
            candidates = (
                glob.glob("/usr/lib/x86_64-linux-gnu/libEGL.so*")
                + glob.glob("/usr/lib/x86_64-linux-gnu/libGL.so*")
                + glob.glob("/usr/lib/*/libEGL.so*")
                + glob.glob("/usr/lib/*/libGL.so*")
            )
            for candidate in candidates:
                if os.path.isfile(candidate):
                    try:
                        shutil.copy2(candidate, stub_path)
                        created = True
                        logging.info(f"libGLESv2 スタブ: {candidate} をコピーして代用")
                        break
                    except Exception:
                        pass

        if not created:
            logging.warning("libGLESv2 スタブの作成に失敗しました。MediaPipe が動かない可能性があります。")
            return

    # LD_LIBRARY_PATH にスタブのディレクトリを追加
    current_path = os.environ.get("LD_LIBRARY_PATH", "")
    if stub_dir not in current_path:
        os.environ["LD_LIBRARY_PATH"] = f"{stub_dir}:{current_path}" if current_path else stub_dir
        logging.info(f"LD_LIBRARY_PATH に {stub_dir} を追加しました")


# ==============================
# モデルは初回使用時に1回だけロード（サーバー起動を高速化）
# ==============================
_detector = None
_detector_lock = threading.Lock()

def _get_detector():
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                # GPU なしサーバー向け: libGLESv2 スタブを準備してから MediaPipe をロード
                _ensure_gles_stub()
                logging.info("MediaPipe PoseLandmarker モデルをロード中...")
                base = python.BaseOptions(model_asset_path=MODEL_PATH)
                _detector = vision.PoseLandmarker.create_from_options(
                    vision.PoseLandmarkerOptions(base_options=base)
                )
                logging.info("MediaPipe PoseLandmarker モデルのロード完了")
    return _detector

# ==============================
# ✅骨格抽出（軽量版・インパクト周辺のみ）
# framesは絶対保存しない
# ==============================
def extract_pose_landmarks(video_path: str, impact_index: int = None, range_sec: float = 1.0, progress_cb=None):
    """
    動画から骨格ランドマークを抽出（インパクト周辺 ±range_sec のみ）

    Args:
        video_path: 動画ファイルのパス
        impact_index: インパクトフレームのインデックス（Noneの場合は全フレーム処理）

    Returns:
    {
        "norm":  [F,33,3] (0〜1正規化座標)
        "pixel": [F,33,2] (ピクセル座標)
    }

    ※ framesは返さない（Renderで落ちるため）
    """
    detector = _get_detector()

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {
            "norm": np.zeros((0, 0, 0)),
            "pixel": np.zeros((0, 0, 0)),
        }
    
    # 総フレーム数を取得（WebM等では0になることがある）
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 処理するフレーム範囲を決定
    start_frame = 0
    if impact_index is not None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        buffer = int(fps * range_sec)
        # インパクト前後 range_sec（total_frames=0のときは上限制限しない）
        start_frame = max(0, impact_index - buffer)
        end_frame = (impact_index + buffer) if total_frames <= 0 else min(total_frames - 1, impact_index + buffer)
        frame_indices = list(range(start_frame, end_frame + 1))
        logging.info(f"解析範囲: {start_frame} to {end_frame} ({len(frame_indices)} frames)")
    else:
        # 全期間（指定がない場合）
        frame_indices = None
        logging.info(f"全フレーム処理モード（後方互換）")

    all_norm: List[Any] = []
    all_pixel: List[Any] = []
    current_frame = 0

    current_frame = 0

    if frame_indices is not None:
        last_idx = -1
        n_target = len(frame_indices)
        for i_fi, idx in enumerate(frame_indices):
            if idx != last_idx + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            
            ret, frame = cap.read()
            if not ret: break
            last_idx = idx
            
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                all_norm.append([[p.x, p.y, p.z] for p in lm])
                all_pixel.append([[int(p.x * w), int(p.y * h)] for p in lm])
            else:
                all_norm.append([[0.0, 0.0, 0.0] for _ in range(33)])
                all_pixel.append([[0, 0] for _ in range(33)])

            if progress_cb and n_target > 0 and i_fi % 5 == 0:
                progress_cb(i_fi / n_target)
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                all_norm.append([[p.x, p.y, p.z] for p in lm])
                all_pixel.append([[int(p.x * w), int(p.y * h)] for p in lm])
            else:
                all_norm.append([[0.0, 0.0, 0.0] for _ in range(33)])
                all_pixel.append([[0, 0] for _ in range(33)])

            current_frame += 1
            if progress_cb and total_frames > 0 and current_frame % 5 == 0:
                progress_cb(current_frame / total_frames)

    cap.release()

    # 空なら返す
    if len(all_norm) == 0:
        return {
            "norm": np.zeros((0, 0, 0)),
            "pixel": np.zeros((0, 0, 0)),
        }

    return {
        "norm": np.array(all_norm),
        "pixel": np.array(all_pixel),
        "start_frame": start_frame
    }
