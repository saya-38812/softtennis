"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import type { TrackerSession, ServeResult } from "@/lib/api";
// サーブの記録はサーバーに送らず、ブラウザ内（localStorage）に保存する
import {
  getLocalSession,
  addLocalServe,
  undoLocalServe,
} from "@/lib/localSession";
import { TrackerStats } from "@/components/tracker/TrackerStats";
import { useSessionVideo } from "@/contexts/SessionVideoContext";

function PracticeContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const sessionId = searchParams.get("session_id");
  const { setVideoBlob } = useSessionVideo();

  const [session, setSession] = useState<TrackerSession | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [recording, setRecording] = useState<ServeResult | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const recordingStartTimeRef = useRef<number>(0);

  // ブラウザ内に保存されたセッションを読み込む
  // （localStorage の読み込みは一瞬で終わるので async は不要）
  const fetchSession = useCallback(() => {
    if (!sessionId) return;
    const data = getLocalSession(sessionId);
    if (data) {
      setSession(data);
    } else {
      setError("セッションが見つかりません。ホームから始めてください。");
    }
    setLoading(false);
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId) {
      setError("No session. Start from home.");
      setLoading(false);
      return;
    }
    fetchSession();
  }, [sessionId, fetchSession]);

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        startRecording(stream);
      } catch (e) {
        if (!cancelled) setCameraError("カメラを起動できませんでした");
      }
    };
    startCamera();
    return () => {
      cancelled = true;
      stopRecording();
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, [sessionId]);

  function startRecording(stream: MediaStream) {
    if (mediaRecorderRef.current?.state === "recording") return;
    chunksRef.current = [];
    const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? "video/webm;codecs=vp9"
      : "video/webm";
    const mr = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 2500000 });
    mr.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    mr.start(1000);
    mediaRecorderRef.current = mr;
    recordingStartTimeRef.current = Date.now();
    setIsRecording(true);
  }

  function stopRecording(): Promise<Blob | null> {
    return new Promise((resolve) => {
      const mr = mediaRecorderRef.current;
      if (!mr || mr.state !== "recording") {
        resolve(null);
        return;
      }
      mr.onstop = () => {
        const blob = chunksRef.current.length
          ? new Blob(chunksRef.current, { type: mr.mimeType })
          : null;
        mediaRecorderRef.current = null;
        setIsRecording(false);
        resolve(blob);
      };
      mr.stop();
    });
  }

  const getTimestampSec = useCallback(() => {
    return (Date.now() - recordingStartTimeRef.current) / 1000;
  }, []);

  // IN / FAULT ボタンが押されたときの処理
  const handleRecord = (result: ServeResult) => {
    if (!sessionId || recording) return;
    setRecording(result);
    // 録画開始から何秒後に押したか（後で動画を1本ずつ切り出すときの目印になる）
    const timestamp_sec = getTimestampSec();
    try {
      // 保存した結果（件数を数え直した最新のセッション）をそのまま画面に反映
      const updated = addLocalServe(sessionId, result, timestamp_sec);
      setSession(updated);
    } catch (e) {
      setError(e instanceof Error ? e.message : "記録に失敗しました");
    } finally {
      setRecording(null);
    }
  };

  // 直前のサーブを取り消す
  const handleUndo = () => {
    if (!sessionId || (session?.total_attempts ?? 0) === 0) return;
    try {
      const updated = undoLocalServe(sessionId);
      setSession(updated);
    } catch (e) {
      setError(e instanceof Error ? e.message : "取り消しに失敗しました");
    }
  };

  const handleEndSession = async () => {
    if (!sessionId) return;
    const blob = await stopRecording();
    if (blob) setVideoBlob(blob);
    router.push(`/session-result?session_id=${sessionId}`);
  };

  if (loading && !session) {
    return (
      <main className="container" style={{ paddingTop: "2rem", textAlign: "center" }}>
        <p className="text-muted">Loading…</p>
      </main>
    );
  }

  if (error && !sessionId) {
    return (
      <main className="container" style={{ paddingTop: "2rem" }}>
        <div className="card text-center">
          <p style={{ color: "var(--error)" }}>{error}</p>
          <Link href="/" className="btn btn-primary">Home</Link>
        </div>
      </main>
    );
  }

  const total = session?.total_attempts ?? 0;

  return (
    <main className="container practice-screen" style={{ paddingTop: "0.5rem", paddingBottom: "1rem" }}>
      <TrackerStats session={session} />
      {isRecording && (
        <p style={{ fontSize: "0.75rem", color: "var(--muted)", marginBottom: "0.5rem" }}>
          ● 録画中
        </p>
      )}
      <div className="card" style={{ padding: "0.5rem", marginBottom: "1rem" }}>
        <div style={{ aspectRatio: "4/3", background: "var(--bg)", borderRadius: 8, overflow: "hidden", position: "relative" }}>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
          {cameraError && (
            <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "var(--surface)", color: "var(--muted)" }}>
              {cameraError}
            </div>
          )}
        </div>
      </div>

      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem", flexWrap: "wrap" }}>
        <button
          type="button"
          className="btn btn-success"
          style={{ flex: 1, minWidth: 80 }}
          onClick={() => handleRecord("IN")}
          disabled={!!recording}
        >
          IN
        </button>
        <button
          type="button"
          className="btn btn-danger"
          style={{ flex: 1, minWidth: 80 }}
          onClick={() => handleRecord("FAULT")}
          disabled={!!recording}
        >
          FAULT
        </button>
      </div>

      <div style={{ display: "flex", gap: "0.5rem" }}>
        <button type="button" className="btn btn-outline" style={{ flex: 1 }} onClick={handleUndo} disabled={total === 0}>
          Undo
        </button>
        <button type="button" className="btn btn-primary" style={{ flex: 1 }} onClick={handleEndSession}>
          End Session
        </button>
      </div>

      {error && (
        <p style={{ color: "var(--error)", fontSize: "0.875rem", marginTop: "0.5rem" }}>{error}</p>
      )}

      <p className="text-center" style={{ marginTop: "1rem" }}>
        <Link href="/">← Home</Link>
      </p>
    </main>
  );
}

export default function PracticePage() {
  return (
    <Suspense fallback={
      <main className="container" style={{ paddingTop: "2rem", textAlign: "center" }}>
        <p className="text-muted">Loading…</p>
      </main>
    }>
      <PracticeContent />
    </Suspense>
  );
}
