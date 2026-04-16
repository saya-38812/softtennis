"""
動画から指定秒数範囲のクリップを切り出し（timestamp ± 1秒の解析用）。
"""
import os
import tempfile
import cv2
import logging

logger = logging.getLogger(__name__)


def extract_clip(video_path: str, center_sec: float, range_sec: float = 2.0) -> str:
    """
    動画から center_sec を中心に ±range_sec のクリップを一時ファイルに出力する（デフォルト4秒）。

    Args:
        video_path: 元動画のパス
        center_sec: 中心時刻（秒）
        range_sec: 前後の範囲（秒）

    Returns:
        一時ファイルのパス。呼び出し側で削除すること。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    start_sec = max(0.0, center_sec - range_sec)
    end_sec = center_sec + range_sec
    if center_sec < range_sec:
        end_sec = max(end_sec, range_sec * 2)
    start_frame = int(start_sec * fps)
    # WebM等で total_frames が 0 のときは範囲のフレーム数を使う
    end_frame = int(end_sec * fps) if total_frames <= 0 else min(total_frames, int(end_sec * fps))

    if start_frame >= end_frame:
        cap.release()
        raise ValueError(f"Invalid clip range: {start_sec}s - {end_sec}s")

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    logger.info(f"Extracted clip: {out_path} ({start_sec:.1f}s - {end_sec:.1f}s)")
    return out_path
