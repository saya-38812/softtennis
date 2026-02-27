import os
import gc
import numpy as np
import cv2
import logging
import time
import glob

from .video_pose_analyzer import extract_pose_landmarks
from .normalize_pose import normalize_pose

from .angle_utils import (
    calculate_elbow_angle,
    calculate_body_sway,
    calculate_impact_height,
    calculate_waist_rotation,
    calculate_waist_rotation_speed,
    calculate_weight_left_right,
)
from .video_renderer import render_analyzed_video

logging.basicConfig(level=logging.INFO)

FOCUS_LABELS = {
    "impact_height": "打点の高さ",
    "elbow_angle": "肘の角度",
    "body_sway": "体軸のブレ",
    "waist_speed": "腰の回転速度",
    "weight_transfer": "体重移動",
}

FOCUS_LANDMARK = {
    "impact_height": 16,
    "elbow_angle": 14,
    "body_sway": 0,
    "waist_speed": 24,
    "weight_transfer": 24,
}

# ==============================
# お手本基準のスコアリング
# お手本の代表値を「90点相当」として、そこからの乖離でスコアを付ける。
# お手本データは analyze_video 初回実行時にキャッシュされた後、
# build_score_funcs() で動的に閾値を構築する。
# 初回キャッシュ前のフォールバック用にデフォルト値も保持する。
# ==============================

# フォールバック基準値（success.mp4 の実測値）
_IDEAL_DEFAULTS = {
    "impact_height": -0.58,   # 手首Y（負=高い）
    "elbow_angle": 175.0,     # 度
    "body_sway": 0.057,       # 鼻のフレーム間平均移動量（肩幅=1基準）
    "waist_speed": 4800.0,    # deg/s peak
    "weight_transfer": 0.15,  # composite (half_diff+max)
}

def _make_score_fn(ideal, zero_val, higher_is_better=True):
    """ideal を 90点、zero_val を 0点 とする線形スコア関数を返す"""
    def fn(val):
        if higher_is_better:
            return float(np.clip((val - zero_val) / (ideal - zero_val) * 90, 0, 100))
        else:
            return float(np.clip((zero_val - val) / (zero_val - ideal) * 90, 0, 100))
    return fn

def build_score_funcs(ideal_vals=None):
    """お手本の代表値からスコアリング関数群を構築"""
    iv = ideal_vals or _IDEAL_DEFAULTS

    return {
        # 打点: abs(val) が大きいほど高打点。ideal≈0.58、ゼロ付近は0点
        "impact_height": _make_score_fn(abs(iv["impact_height"]), 0.0, higher_is_better=True),
        # 肘: 大きいほど伸展。ideal≈175、90度は0点
        "elbow_angle": _make_score_fn(iv["elbow_angle"], 90.0, higher_is_better=True),
        # ブレ: 小さいほど良い。idealの4倍を0点とする
        "body_sway": _make_score_fn(iv["body_sway"], iv["body_sway"] * 4, higher_is_better=False),
        # 腰速: 大きいほど良い。ideal≈4800、0は0点
        "waist_speed": _make_score_fn(iv["waist_speed"], 0.0, higher_is_better=True),
        # 体重移動: 大きいほど良い。ideal≈0.15、0は0点
        "weight_transfer": _make_score_fn(iv["weight_transfer"], 0.0, higher_is_better=True),
    }

# 初期状態はフォールバック値で構築。お手本解析後に上書きされる。
SCORE_FUNCS = build_score_funcs()

def get_dynamic_advice(focus, val, score_100):
    """score_100 は 0-100 正規化済みスコア"""
    if focus == "impact_height":
        if score_100 >= 80:
            return "Ideal", "You", "高い打点", "Good!", "打点の高さは十分です！"
        elif score_100 >= 50:
            return "Ideal", "You", "高い打点", "もう少し", "トスをもう少し高く上げて、腕が伸びきった位置で打ちましょう"
        else:
            return "Ideal", "You", "高い打点", "低い", "トスを高く上げ、最高到達点で打つ意識を持ちましょう"

    if focus == "elbow_angle":
        deg = int(val)
        if score_100 >= 80:
            return "Ideal", "You", "165°", f"{deg}°", "肘の伸びは十分です！"
        elif score_100 >= 50:
            return "Ideal", "You", "165°", f"{deg}°", f"インパクト時にあと少し腕を伸ばして打ちましょう"
        else:
            return "Ideal", "You", "165°", f"{deg}°", "肘が曲がりすぎています。腕を大きく伸ばしてスイングしましょう"

    if focus == "body_sway":
        if score_100 >= 80:
            return "Target", "You", "安定", "Good!", "体軸は安定しています！"
        elif score_100 >= 50:
            return "Target", "You", "安定", "やや不安定", "インパクト時に頭の位置を固定する意識を持ちましょう"
        else:
            return "Target", "You", "安定", "不安定", "体が大きくブレています。下半身を安定させてスイングしましょう"

    if focus == "waist_speed":
        spd = int(val)
        if score_100 >= 80:
            return "Ideal", "You", "鋭い回転", "Good!", "腰の回転は十分鋭いです！"
        elif score_100 >= 50:
            return "Ideal", "You", "鋭い回転", "もう少し", "スイング時に腰をもっと鋭く回して威力を上げましょう"
        else:
            return "Ideal", "You", "鋭い回転", "遅い", "腰の回転が不足しています。下半身から回転を始めましょう"

    if focus == "weight_transfer":
        if score_100 >= 80:
            return "Goal", "You", "前足荷重", "Good!", "体重移動は十分です！"
        elif score_100 >= 50:
            return "Goal", "You", "前足荷重", "もう少し", "インパクト時にもう少し前足に体重を乗せましょう"
        else:
            return "Goal", "You", "前足荷重", "後ろ重心", "前足（右利きなら左足）へしっかり体重を移動させましょう"

    return "Ideal", "You", "--", "--", "今の調子で練習を続けましょう！"

# ==============================
# ✅ お手本データのキャッシュ
# ==============================
SUCCESS_CACHE = {}

# ==============================
# 軽量インパクト検出（OpenCVのみ）
# ==============================
def detect_impact_frame(video_path: str) -> int:
    """
    動き量からインパクト候補を検出（MediaPipe不使用）
    
    Args:
        video_path: 動画ファイルのパス
    
    Returns:
        int: インパクトフレームのインデックス
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        cap.release()
        return 0
    
    frame_diffs = []
    prev_gray = None
    frame_count = 0
    
    # 高速化のためサンプリング（全フレームではなく5フレームに1回とかでも良いが、
    # インパクトは一瞬なので全フレーム読む。ただしリサイズして処理を軽くする）
    while cap.isOpened():
        # 3フレームに1回サンプリング（高速化）
        if frame_count % 3 != 0:
            ret = cap.grab()
            if not ret: break
            frame_count += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break
        
        # リサイズ（超高速化）
        h, w = frame.shape[:2]
        small_w = 240
        small_h = int(h * (small_w / w))
        small_frame = cv2.resize(frame, (small_w, small_h))
        
        # グレースケール変換
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            # フレーム間差分を計算
            diff = cv2.absdiff(prev_gray, gray)
            diff_sum = np.sum(diff)
            frame_diffs.append(diff_sum)
        else:
            frame_diffs.append(0)
        
        prev_gray = gray
        frame_count += 1
    
    cap.release()
    
    if len(frame_diffs) == 0:
        return 0
    
    # フレーム間差分が最大のフレーム。ただし最初と最後の方は無視（ノイズ回避）
    n_diffs = len(frame_diffs)
    skip = 3
    if n_diffs > 20:
        search_range = frame_diffs[10:-10]
        impact_index = (int(np.argmax(search_range)) + 10) * skip
    else:
        impact_index = int(np.argmax(frame_diffs)) * skip
    
    logging.info(f"インパクト検出: フレーム {impact_index} (サンプリング {n_diffs}点)")
    
    return impact_index


# ==============================
# 腕が一番上の瞬間で固定する（後方互換用）
# ==============================
def detect_contact_frame(norm_landmarks):

    n = len(norm_landmarks)
    if n < 10:
        return int(n * 0.7)

    WRIST = 16
    wrist_y = np.array([norm_landmarks[i][WRIST][1] for i in range(n)])

    peak = int(np.argmin(wrist_y))
    return peak


# ==============================
# 描画ルール（最終版）
# ==============================
def draw_focus(frame, focus, ux, uy, ix, iy):
    if frame is None: return
    h, w = frame.shape[:2]

    # ① ターゲットマーク（共通）
    cv2.drawMarker(frame, (ux, uy), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 30, 3)
    cv2.circle(frame, (ix, iy), 35, (0, 255, 0), 3)

    # ② 指標別の追加描画
    if focus == "body_sway" or focus == "waist_speed":
        cv2.line(frame, (ix, 0), (ix, h), (0, 255, 0), 2)
        cv2.line(frame, (ux, 0), (ux, h), (0, 0, 255), 2)
    
    elif focus == "impact_height":
        cv2.line(frame, (0, iy), (w, iy), (0, 255, 0), 2)
        cv2.line(frame, (0, uy), (w, uy), (0, 0, 255), 2)

        cv2.putText(frame, "Your Axis", (ux + 10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


# ==============================
# ✅古い画像を自動削除する
# ==============================
def cleanup_old_outputs(out_dir, keep=5):
    # pngとmp4の両方を対象にする
    files = []
    for ext in ["*.png", "*.mp4"]:
        files.extend(glob.glob(os.path.join(out_dir, ext)))
    
    # 更新日時順にソート
    files.sort(key=os.path.getmtime)

    # 最新keep*2枚（画像と動画のセット想定）だけ残す。
    # ここではシンプルに全体数を制限
    max_files = keep * 3 
    if len(files) > max_files:
        for f in files[:-max_files]:
            try:
                os.remove(f)
            except:
                pass


# ==============================
# メイン解析
# ==============================
def analyze_video(file_path, progress_cb=None):

    def _report(pct):
        if progress_cb:
            progress_cb(int(pct))

    BASE_DIR = os.path.dirname(__file__)
    success_path = os.path.join(BASE_DIR, "success.mp4")

    # 1. お手本データの準備（キャッシュ活用）
    if "success" not in SUCCESS_CACHE:
        logging.info("お手本データを解析中（初回のみ）...")
        s_impact_idx = detect_impact_frame(success_path)
        s_diag = extract_pose_landmarks(success_path, s_impact_idx, range_sec=1.0)
        
        # インパクトフレームの画像を取得
        cap_s = cv2.VideoCapture(success_path)
        cap_s.set(cv2.CAP_PROP_POS_FRAMES, s_impact_idx)
        ret, s_img_orig = cap_s.read()
        cap_s.release()
        
        s_norm = s_diag["norm"]
        s_seq = normalize_pose(s_norm)
        
        # お手本の代表値を、ユーザーと同じロジックで算出
        s_contact = detect_contact_frame(s_norm)
        s_n = len(s_seq)
        s_impact_local = min(s_contact, s_n - 1)
        s_win = max(3, s_n // 5)
        s_lo = max(0, s_impact_local - s_win)
        s_hi = min(s_n, s_impact_local + s_win + 1)

        s_impact_vals = calculate_impact_height(s_seq, True)
        s_elbow_vals = calculate_elbow_angle(s_seq, True)
        s_sway_vals = calculate_body_sway(s_seq)
        s_waist_rots = calculate_waist_rotation(s_seq, True)
        s_waist_spds = calculate_waist_rotation_speed(s_waist_rots)
        s_weight_vals = calculate_weight_left_right(s_seq)

        s_iv = float(np.min(s_impact_vals[s_lo:s_hi])) if s_hi > s_lo and len(s_impact_vals) > 0 else -0.5
        s_ev = float(np.max(s_elbow_vals[s_lo:s_hi])) if s_hi > s_lo and len(s_elbow_vals) > 0 else 170.0

        s_sway_lo = max(0, min(s_lo, len(s_sway_vals) - 1))
        s_sway_hi = min(s_hi, len(s_sway_vals))
        s_sway_slice = s_sway_vals[s_sway_lo:s_sway_hi] if s_sway_hi > s_sway_lo else s_sway_vals
        s_sv = float(np.mean(s_sway_slice)) if len(s_sway_slice) > 0 else 0.05

        sp_lo = max(0, s_lo - 1)
        sp_hi = min(len(s_waist_spds), s_hi)
        s_wv = float(np.max(s_waist_spds[sp_lo:sp_hi])) if sp_hi > sp_lo and len(s_waist_spds) > 0 else 4800.0

        if len(s_weight_vals) > 1:
            s_half = s_n // 2
            s_wt = abs(float(np.mean(s_weight_vals[:s_half])) - float(np.mean(s_weight_vals[s_half:]))) + float(np.max(s_weight_vals))
        else:
            s_wt = 0.15

        ideal_representative = {
            "impact_height": s_iv,
            "elbow_angle": s_ev,
            "body_sway": s_sv,
            "waist_speed": s_wv,
            "weight_transfer": s_wt,
        }
        logging.info(f"お手本代表値: {', '.join(f'{k}={v:.4f}' for k, v in ideal_representative.items())}")

        # お手本基準でスコアリング関数を再構築
        global SCORE_FUNCS
        SCORE_FUNCS = build_score_funcs(ideal_representative)

        SUCCESS_CACHE["success"] = {
            "impact_index": s_impact_idx,
            "diag": s_diag,
            "norm": s_norm,
            "pixel": s_diag["pixel"],
            "seq": s_seq,
            "img_orig": s_img_orig,
            "ideal_vals": ideal_representative,
        }
    
    s_data = SUCCESS_CACHE["success"]
    success_impact_index = s_data["impact_index"]
    success_norm = s_data["norm"]
    success_pixel = s_data["pixel"]
    success_seq = s_data["seq"]
    ideal_img_orig = s_data["img_orig"]

    _report(10)

    # 2. ユーザー動画: まず軽量にインパクトフレームを検出
    logging.info(f"Step 1: インパクトフレーム検出中... {file_path}")
    rough_impact = detect_impact_frame(file_path)
    gc.collect()

    _report(20)

    # 3. インパクト±1秒だけMediaPipeで骨格抽出（メモリ節約）
    logging.info(f"Step 2: インパクト周辺の骨格抽出中 (±1s around frame {rough_impact})...")
    def _pose_progress(ratio):
        _report(20 + int(ratio * 40))
    target_diag = extract_pose_landmarks(file_path, rough_impact, range_sec=1.0, progress_cb=_pose_progress)

    target_norm   = target_diag["norm"]
    target_pixel  = target_diag["pixel"]
    target_full_pixel = target_pixel
    target_start_frame = target_diag.get("start_frame", 0)

    if len(target_norm) == 0:
        return {"menu": ["基本フォーム練習"], "ai_text": "解析できませんでした"}

    contact_local = detect_contact_frame(target_norm)
    target_impact_index = target_start_frame + contact_local
    logging.info(
        f"インパクト検出: 手首最高点 frame={target_impact_index} "
        f"(local={contact_local}/{len(target_norm)})"
    )

    logging.info(f"処理フレーム数: success={len(success_norm)}, target={len(target_norm)}")

    target_seq  = normalize_pose(target_norm)

    if len(target_seq) == 0:
        return {
            "menu": ["基本フォーム練習"], 
            "ai_text": "動画から骨格を正しく検出できませんでした。もう少し離れて全身が写るように撮影してください。",
            "status": "error"
        }

    gc.collect()
    _report(65)

    # 3. 指標計算
    elbow_values = calculate_elbow_angle(target_seq, True)
    impact_values = calculate_impact_height(target_seq, True)
    sway_values = calculate_body_sway(target_seq)
    waist_rotations = calculate_waist_rotation(target_seq, True)
    waist_speeds = calculate_waist_rotation_speed(waist_rotations)
    weight_transfers = calculate_weight_left_right(target_seq)

    n_frames = len(target_seq)
    # contact_local は target_norm 基準。target_seq はフレームが間引かれている可能性があるので clamp
    impact_local = min(contact_local, n_frames - 1)
    window = max(3, n_frames // 5)
    lo = max(0, impact_local - window)
    hi = min(n_frames, impact_local + window + 1)

    # 打点の高さ: インパクト瞬間の値（最も手首が高い点 = Y値が最も負の点）
    if len(impact_values) > 0:
        impact_val = float(np.min(impact_values[lo:hi])) if hi > lo else float(np.min(impact_values))
    else:
        impact_val = 0.0

    # 肘の角度: インパクト付近の最大値（最も伸びた瞬間）
    if len(elbow_values) > 0:
        elbow_val = float(np.max(elbow_values[lo:hi])) if hi > lo else float(np.max(elbow_values))
    else:
        elbow_val = 0.0

    # 体軸ブレ: インパクト付近の鼻のフレーム間平均移動量（小さいほど安定）
    if len(sway_values) > 0:
        sway_lo = max(0, min(lo, len(sway_values) - 1))
        sway_hi = min(hi, len(sway_values))
        sway_slice = sway_values[sway_lo:sway_hi] if sway_hi > sway_lo else sway_values
        sway_val = float(np.mean(sway_slice))
    else:
        sway_val = 0.0

    # 腰の回転速度: ピーク速度（最も鋭く回転した瞬間）
    if len(waist_speeds) > 0:
        speed_lo = max(0, lo - 1)
        speed_hi = min(len(waist_speeds), hi)
        waist_val = float(np.max(waist_speeds[speed_lo:speed_hi])) if speed_hi > speed_lo else float(np.max(waist_speeds))
    else:
        waist_val = 0.0

    # 体重移動: インパクト前後の変化量（前半と後半の差）
    if len(weight_transfers) > 1:
        half = n_frames // 2
        first_half = float(np.mean(weight_transfers[:half])) if half > 0 else 0.0
        second_half = float(np.mean(weight_transfers[half:])) if half < n_frames else 0.0
        weight_val = abs(second_half - first_half) + float(np.max(weight_transfers))
    else:
        weight_val = 0.0

    logging.info(
        f"指標値: impact_height={impact_val:.3f}, elbow_angle={elbow_val:.1f}, "
        f"body_sway={sway_val:.4f}, waist_speed={waist_val:.1f}, weight_transfer={weight_val:.4f}"
    )

    scores = {
        "impact_height": float(impact_val),
        "elbow_angle": float(elbow_val),
        "body_sway": float(sway_val),
        "waist_speed": float(waist_val),
        "weight_transfer": float(weight_val),
    }

    # 全指標を 0-100 に正規化してから比較（公平な土俵）
    normalized_scores = {}
    for key, val in scores.items():
        score_input = abs(val) if key == "impact_height" else val
        normalized_scores[key] = SCORE_FUNCS[key](score_input)

    # 腰の回転速度は2D投影のため、カメラアングルによっては計測不能。
    # お手本の1/20未満のピーク速度は「計測不能」として、スコア比較から除外し中立スコアを与える。
    ideal_vals = s_data.get("ideal_vals", _IDEAL_DEFAULTS)
    waist_threshold = ideal_vals["waist_speed"] / 20
    if waist_val < waist_threshold:
        logging.info(f"waist_speed={waist_val:.1f} < threshold={waist_threshold:.1f}: カメラアングルにより計測不能と判定、スコア比較から除外")
        normalized_scores["waist_speed"] = 60.0  # 中立値（改善候補にならない）

    # 体重移動も同様に、2D投影の限界がある
    wt_threshold = ideal_vals["weight_transfer"] / 5
    if weight_val < wt_threshold:
        logging.info(f"weight_transfer={weight_val:.4f} < threshold={wt_threshold:.4f}: 計測困難と判定、スコア比較から除外")
        normalized_scores["weight_transfer"] = 60.0

    logging.info(f"正規化スコア: {', '.join(f'{k}={v:.1f}' for k, v in normalized_scores.items())}")

    weakness = {k: ("ok" if v >= 60 else "low") for k, v in normalized_scores.items()}

    # 最もスコアが低い指標を改善ポイントとして選ぶ
    focus = min(normalized_scores, key=normalized_scores.get)

    # フレーム取得（抽出したフレームの中央を使用）
    extracted_user_idx  = len(target_norm) // 2 if len(target_norm) > 0 else 0
    extracted_ideal_idx = len(success_norm) // 2 if len(success_norm) > 0 else 0

    lid = FOCUS_LANDMARK[focus]

    ux, uy = target_pixel[extracted_user_idx][lid]
    ix, iy = success_pixel[extracted_ideal_idx][lid]

    # フレーム画像取得（元の動画からインパクトフレームを取得）
    cap1 = cv2.VideoCapture(file_path)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, target_impact_index)
    ret, user_img = cap1.read()
    cap1.release()

    if not ret:
        user_img = np.zeros((720, 1280, 3), dtype=np.uint8)

    ideal_img = ideal_img_orig.copy()

    # 描画
    draw_focus(user_img, focus, ux, uy, ix, iy)
    draw_focus(ideal_img, focus, ix, iy, ix, iy)

    # ==============================
    # ✅自動再起動を避けるため、backend配下から外す
    # ==============================
    # main.py と同じルールで server_storage を使用
    STORAGE_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "server_storage"))
    out_dir = os.path.join(STORAGE_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    ts = int(time.time())

    user_name  = f"user_{ts}.png"
    ideal_name = f"ideal_{ts}.png"

    # 動画生成
    user_video_name = f"user_{ts}.mp4"
    user_video_path = os.path.join(out_dir, user_video_name)
    
    _report(70)

    # 不要データを解放してからレンダリング
    del target_norm, target_seq
    del elbow_values, impact_values, sway_values, waist_rotations, waist_speeds, weight_transfers
    gc.collect()

    logging.info(f"Step 3: 解析動画をレンダリング中... {user_video_name}")
    def _render_progress(ratio):
        _report(70 + int(ratio * 25))
    render_analyzed_video(
        file_path, 
        target_full_pixel, 
        user_video_path, 
        impact_frame=target_impact_index,
        start_frame=target_start_frame,
        focus_landmark=lid,
        progress_cb=_render_progress,
    )

    cv2.imwrite(os.path.join(out_dir, user_name), user_img)
    cv2.imwrite(os.path.join(out_dir, ideal_name), ideal_img)

    # 古い画像を掃除
    cleanup_old_outputs(out_dir)

    _report(95)

    l_ideal, l_user, v_ideal, v_user, advice = get_dynamic_advice(
        focus, scores[focus], normalized_scores[focus]
    )

    # ==============================
    # 結果返却
    # ==============================
    return {
        "diagnosis": {
            "weakness": weakness,
            "scores": scores,
            "normalized_scores": {k: round(v, 1) for k, v in normalized_scores.items()},
        },
        "menu": [f"{FOCUS_LABELS[focus]}を改善する練習を1つだけやりましょう"],
        "ai_text": f"今日の最優先課題は「{FOCUS_LABELS[focus]}」です。",
        "ideal_image": f"/outputs/{ideal_name}",
        "user_image": f"/outputs/{user_name}",
        "user_video": f"/outputs/{user_video_name}",
        "focus_label": FOCUS_LABELS[focus],
        "comparison": {
            "label_ideal": l_ideal,
            "label_user": l_user,
            "value_ideal": v_ideal,
            "value_user": v_user,
            "action_tip": advice
        }
    }
