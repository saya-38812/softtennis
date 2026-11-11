"use client";

/**
 * アップロード画面（撮影済みの動画から解析を依頼する）
 *
 * 2つのモードがある：
 *
 * 【IN/FAULT を記録】
 *   1. 動画ファイルを選ぶ
 *   2. 再生しながら、サーブの結果が出たら IN / FAULT をタップ
 *      → タップした瞬間の「再生位置（秒）」をタイムスタンプとして記録
 *   3. 「記録を完了」で保存 → 結果画面へ
 *
 * 【フォームだけ解析】
 *   1. サーブ1本分の動画を選ぶ
 *   2. 「この動画で解析する」→ 結果画面へ
 *      （打った瞬間は後で Colab が自動で見つけるので、タップは不要）
 *
 * どちらも結果画面から同じように Drive へアップロードできる。
 */

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import type { ServeResult } from "@/lib/api";
import { createLocalSessionFromServes, createFormOnlySession } from "@/lib/localSession";
import { useSessionVideo } from "@/contexts/SessionVideoContext";

/** タップした1回分の記録（保存するまでは画面の中だけで持つ） */
interface TaggedServe {
  result: ServeResult;
  timestampSec: number;
}

/** 選べる再生速度 */
const PLAYBACK_RATES = [1, 1.5, 2];

/** アップロード画面のモード */
type UploadMode = "serve_tracking" | "form_only";

/** モード切り替えボタンの表示内容 */
const MODE_OPTIONS: { value: UploadMode; label: string; description: string }[] = [
  {
    value: "serve_tracking",
    label: "IN/FAULT を記録",
    description: "練習動画を再生しながら、1本ずつ IN / FAULT を付けます。",
  },
  {
    value: "form_only",
    label: "フォームだけ解析",
    description: "サーブ1本分の動画のフォームを解析します。IN / FAULT の記録は不要です。",
  },
];

/** 秒数を「1:05.3」のような見やすい形にする */
function formatTime(sec: number): string {
  const minutes = Math.floor(sec / 60);
  const seconds = (sec % 60).toFixed(1).padStart(4, "0");
  return `${minutes}:${seconds}`;
}

export default function UploadPage() {
  const router = useRouter();
  const { setVideoBlob } = useSessionVideo();
  const videoRef = useRef<HTMLVideoElement>(null);

  const [mode, setMode] = useState<UploadMode>("serve_tracking");
  const [file, setFile] = useState<File | null>(null);
  // 選んだファイルを <video> で再生するための一時URL
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [serves, setServes] = useState<TaggedServe[]>([]);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [error, setError] = useState<string | null>(null);

  // ファイルが変わったら再生用URLを作り直す。前のURLはメモリ解放する
  useEffect(() => {
    if (!file) {
      setVideoUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // 再生速度ボタンが押されたら <video> に反映する
  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = playbackRate;
  }, [playbackRate, videoUrl]);

  // ------------------------------------------------------------
  // イベント処理
  // ------------------------------------------------------------

  /** 動画ファイルが選ばれたとき */
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0] ?? null;
    setError(null);
    if (selected && !selected.type.startsWith("video/")) {
      setError("動画ファイルを選んでください");
      return;
    }
    // 別の動画に替えたら、前の動画に付けた記録は意味がなくなるのでリセット
    setFile(selected);
    setServes([]);
  };

  /** IN / FAULT ボタン：今の再生位置を記録する */
  const handleTag = (result: ServeResult) => {
    const video = videoRef.current;
    if (!video) return;
    setServes((prev) => [...prev, { result, timestampSec: video.currentTime }]);
  };

  /** 直前にタップした記録を取り消す */
  const handleUndo = () => {
    setServes((prev) => prev.slice(0, -1));
  };

  /** 再生位置を指定秒数だけ動かす（マイナスなら巻き戻し） */
  const seekBy = (seconds: number) => {
    const video = videoRef.current;
    if (!video) return;
    // 0秒より前・動画の長さより後ろに行かないようにする
    video.currentTime = Math.min(Math.max(video.currentTime + seconds, 0), video.duration || 0);
  };

  /** 記録一覧の時刻を押したとき：その場面に飛ぶ（見直し用） */
  const jumpTo = (sec: number) => {
    const video = videoRef.current;
    if (!video) return;
    // 結果が出る少し前（打つ瞬間あたり）から見られるよう、2秒前に飛ぶ
    video.currentTime = Math.max(sec - 2, 0);
    video.play().catch(() => {});
  };

  /** モードを切り替える（付けかけの IN/FAULT は混ざらないようリセット） */
  const handleModeChange = (next: UploadMode) => {
    setMode(next);
    setServes([]);
    setError(null);
  };

  /** 「フォームだけ解析」：保存して結果画面へ */
  const handleFormOnlyFinish = () => {
    if (!file) return;
    try {
      const session = createFormOnlySession();
      setVideoBlob(file);
      router.push(`/session-result?session_id=${session.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "保存に失敗しました");
    }
  };

  /** 記録を完了：保存して結果画面へ */
  const handleFinish = () => {
    if (!file || serves.length === 0) return;
    try {
      const session = createLocalSessionFromServes(serves);
      // 動画は大きいので localStorage には入れず、練習画面と同じく Context で結果画面に渡す
      setVideoBlob(file);
      router.push(`/session-result?session_id=${session.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "保存に失敗しました");
    }
  };

  // ------------------------------------------------------------
  // 表示
  // ------------------------------------------------------------

  const inCount = serves.filter((s) => s.result === "IN").length;
  const faultCount = serves.filter((s) => s.result === "FAULT").length;
  // 一覧は動画内の時刻順に並べて表示する（巻き戻してタップしても見やすいように）
  const sortedServes = [...serves].sort((a, b) => a.timestampSec - b.timestampSec);

  return (
    <main className="container" style={{ paddingTop: "1rem", paddingBottom: "5rem" }}>
      {/* ---- 0. モードを選ぶ ---- */}
      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "0.5rem" }}>
        {MODE_OPTIONS.map((option) => (
          <button
            key={option.value}
            type="button"
            className={mode === option.value ? "btn btn-primary" : "btn btn-outline"}
            style={{ flex: 1 }}
            onClick={() => handleModeChange(option.value)}
            aria-pressed={mode === option.value}
          >
            {option.label}
          </button>
        ))}
      </div>
      <p style={{ fontSize: "0.8rem", color: "var(--muted)", margin: "0 0 1rem 0" }}>
        {MODE_OPTIONS.find((o) => o.value === mode)?.description}
      </p>

      {/* ---- 1. 動画を選ぶ ---- */}
      <div className="card" style={{ padding: "0.9rem", marginBottom: "1rem" }}>
        <label style={{ display: "block", fontSize: "0.85rem", color: "var(--muted)", marginBottom: "0.5rem" }}>
          {mode === "form_only"
            ? "サーブ1本分の動画を選んでください（全身が横から映っているものがおすすめ）"
            : "撮影済みのサーブ練習の動画を選んでください"}
        </label>
        <input type="file" accept="video/*" onChange={handleFileChange} />
      </div>

      {/* ---- フォームだけ解析：動画を確認して進むだけ ---- */}
      {videoUrl && mode === "form_only" && (
        <>
          <div className="card" style={{ padding: 0, overflow: "hidden", marginBottom: "0.75rem" }}>
            <video
              src={videoUrl}
              controls
              playsInline
              style={{ width: "100%", display: "block", maxHeight: "50vh", background: "#000" }}
            />
          </div>
          <p style={{ fontSize: "0.8rem", color: "var(--muted)", margin: "0 0 1rem 0" }}>
            打った瞬間は自動で見つけます。複数本が映っている場合は「IN/FAULT を記録」を使ってください。
          </p>
          <button
            type="button"
            className="btn btn-primary btn-block"
            onClick={handleFormOnlyFinish}
            style={{ padding: "0.75rem" }}
          >
            この動画で解析する
          </button>
        </>
      )}

      {/* ---- IN/FAULT を記録：再生しながらタップ ---- */}
      {videoUrl && mode === "serve_tracking" && (
        <>
          {/* ---- 2. 再生しながらタップ ---- */}
          <p style={{ fontSize: "0.85rem", color: "var(--muted)", margin: "0 0 0.5rem 0" }}>
            再生しながら、サーブの結果が出たら IN / FAULT を押してください。
          </p>

          <div className="card" style={{ padding: 0, overflow: "hidden", marginBottom: "0.75rem" }}>
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              playsInline
              // スマホでも動画と IN/FAULT ボタンが同時に見えるよう、高さを画面の4割までにする
              style={{ width: "100%", display: "block", maxHeight: "40vh", background: "#000" }}
            />
          </div>

          {/* 再生操作：5秒戻る・再生速度 */}
          <div style={{ display: "flex", gap: "0.5rem", marginBottom: "0.75rem", flexWrap: "wrap" }}>
            <button type="button" className="btn btn-outline" onClick={() => seekBy(-5)}>
              ⏪ 5秒戻る
            </button>
            {PLAYBACK_RATES.map((rate) => (
              <button
                key={rate}
                type="button"
                className={rate === playbackRate ? "btn btn-primary" : "btn btn-outline"}
                onClick={() => setPlaybackRate(rate)}
              >
                {rate}x
              </button>
            ))}
          </div>

          {/* IN / FAULT / Undo
              position: sticky で、スクロールしても画面下（タブバーのすぐ上）に張り付かせる。
              動画を見ながら押す画面なので、ボタンが画面外に出ないようにするため */}
          <div
            style={{
              position: "sticky",
              bottom: "calc(72px + 0.5rem)", // 72px = 下のタブバーの高さ
              zIndex: 10,
              display: "flex",
              gap: "0.5rem",
              padding: "0.5rem",
              marginBottom: "1rem",
              background: "var(--surface)",
              borderRadius: 12,
              boxShadow: "0 4px 16px rgba(0,0,0,0.4)",
            }}
          >
            <button
              type="button"
              className="btn btn-outline"
              onClick={handleUndo}
              disabled={serves.length === 0}
              title="直前の記録を取り消す"
            >
              Undo
            </button>
            <button type="button" className="btn btn-success" style={{ flex: 1 }} onClick={() => handleTag("IN")}>
              IN
            </button>
            <button type="button" className="btn btn-danger" style={{ flex: 1 }} onClick={() => handleTag("FAULT")}>
              FAULT
            </button>
          </div>

          {/* ---- 記録の一覧 ---- */}
          <div className="card" style={{ padding: "0.9rem", marginBottom: "1rem" }}>
            <div style={{ fontWeight: 600, marginBottom: "0.5rem" }}>
              記録 {serves.length}本（IN {inCount} / FAULT {faultCount}）
            </div>
            {sortedServes.length === 0 ? (
              <p className="text-muted" style={{ fontSize: "0.85rem", margin: 0 }}>まだ記録がありません</p>
            ) : (
              <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
                {sortedServes.map((s, i) => (
                  <li key={`${s.timestampSec}-${i}`}>
                    {/* 押すとその場面に飛んで見直せる */}
                    <button
                      type="button"
                      onClick={() => jumpTo(s.timestampSec)}
                      style={{
                        border: "none",
                        borderRadius: 6,
                        padding: "0.25rem 0.5rem",
                        fontSize: "0.8rem",
                        cursor: "pointer",
                        background: s.result === "IN" ? "rgba(63,185,80,0.2)" : "rgba(248,81,73,0.2)",
                        color: s.result === "IN" ? "var(--success)" : "var(--error)",
                      }}
                    >
                      #{i + 1} {s.result} {formatTime(s.timestampSec)}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* ---- 3. 完了 ---- */}
          <button
            type="button"
            className="btn btn-primary btn-block"
            onClick={handleFinish}
            disabled={serves.length === 0}
            style={{ padding: "0.75rem" }}
          >
            記録を完了する
          </button>
        </>
      )}

      {error && (
        <p style={{ color: "var(--error)", fontSize: "0.875rem", marginTop: "1rem" }}>{error}</p>
      )}
    </main>
  );
}
