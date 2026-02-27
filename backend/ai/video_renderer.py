import cv2
import numpy as np
import os
import subprocess
import logging

CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28)
]

def _get_ffmpeg_path():
    """imageio-ffmpeg にバンドルされた ffmpeg パスを取得"""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def _reencode_to_h264(src_path, dst_path, ffmpeg_path):
    """mp4v で書き出された動画を H.264 (libx264) に再エンコード"""
    cmd = [
        ffmpeg_path,
        "-y",
        "-i", src_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        dst_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120, check=True)
        return True
    except Exception as e:
        logging.warning(f"H.264 re-encode failed: {e}")
        return False


MAX_OUTPUT_WIDTH = 640

def render_analyzed_video(input_path, landmarks, output_path, impact_frame=None, start_frame=0, focus_landmark=None, progress_cb=None):
    """
    骨格付き解析動画を生成する（メモリ効率重視）
    ランドマークが存在する範囲のみ出力し、解像度を制限する
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale = min(1.0, MAX_OUTPUT_WIDTH / orig_w)
    out_w = int(orig_w * scale)
    out_h = int(orig_h * scale)

    # ランドマーク範囲のみ出力（メモリ・時間節約）
    out_start = max(0, start_frame)
    out_end = min(total_frames, start_frame + len(landmarks))

    logging.info(
        f"Render: frames {out_start}-{out_end} of {total_frames} ({(out_end-out_start)/fps:.1f}s), "
        f"output={out_w}x{out_h}, impact={impact_frame}"
    )

    tmp_path = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (out_w, out_h))
    if not out.isOpened():
        cap.release()
        return False

    if out_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, out_start)

    render_total = out_end - out_start
    for f_idx in range(out_start, out_end):
        ret, frame = cap.read()
        if not ret:
            break

        if scale < 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        internal_idx = f_idx - start_frame

        if 0 <= internal_idx < len(landmarks):
            frame_lms = landmarks[internal_idx]

            for s_idx, e_idx in CONNECTIONS:
                p1 = frame_lms[s_idx]
                p2 = frame_lms[e_idx]
                if (p1[0] == 0 and p1[1] == 0) or (p2[0] == 0 and p2[1] == 0):
                    continue
                pt1 = (int(p1[0] * scale), int(p1[1] * scale))
                pt2 = (int(p2[0] * scale), int(p2[1] * scale))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 3)

            for i_lm, p in enumerate(frame_lms):
                if p[0] == 0 and p[1] == 0: continue
                if i_lm not in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]: continue
                center = (int(p[0] * scale), int(p[1] * scale))
                cv2.circle(frame, center, 5, (255, 255, 255), -1)

            if f_idx == impact_frame:
                if focus_landmark is not None and focus_landmark < len(frame_lms):
                    fp = frame_lms[focus_landmark]
                    f_center = (int(fp[0] * scale), int(fp[1] * scale))
                    cv2.circle(frame, f_center, 20, (0, 165, 255), 2)
                    cv2.putText(frame, "Impact", (f_center[0] + 25, f_center[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        out.write(frame)

        if progress_cb and render_total > 0 and (f_idx - out_start) % 10 == 0:
            progress_cb((f_idx - out_start) / render_total)

    cap.release()
    out.release()

    ffmpeg_path = _get_ffmpeg_path()
    if ffmpeg_path and _reencode_to_h264(tmp_path, output_path, ffmpeg_path):
        os.remove(tmp_path)
        logging.info(f"H.264 video saved: {output_path}")
    else:
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(tmp_path, output_path)
        logging.warning("Fallback: saved as mp4v (may not play in browser)")

    return True
