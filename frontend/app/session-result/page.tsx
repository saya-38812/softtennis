"use client";

import { Suspense, useEffect, useState, useCallback, useRef } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { getSession, runFormAnalysis, type TrackerSession, type ServeAnalysis } from "@/lib/api";
import { fromAnalysis } from "@/lib/uploadResult";
import { TrackerStats } from "@/components/tracker/TrackerStats";
import { useSessionVideo } from "@/contexts/SessionVideoContext";

// ============================================================
// サーブ結果のバッジ色
// ============================================================
function resultBadge(result: string) {
  const base: React.CSSProperties = {
    fontSize: "0.75rem",
    fontWeight: 700,
    padding: "0.2rem 0.5rem",
    borderRadius: 4,
    display: "inline-block",
  };
  if (result === "IN") return { ...base, background: "rgba(63,185,80,0.2)", color: "var(--success)" };
  if (result === "FAULT") return { ...base, background: "rgba(248,81,73,0.2)", color: "var(--error)" };
  return { ...base, background: "rgba(210,153,34,0.2)", color: "var(--warning)" }; // OUT
}

// ============================================================
// 解析結果をカード形式で表示するコンポーネント
// ============================================================
function AnalysisPanel({ analysis }: { analysis: ServeAnalysis }) {
  const data = fromAnalysis(analysis);

  const barColor = (key: string, value: number) => {
    if (key === "impact_height") return "var(--warning)";
    if (key === "waist_speed") return "#38bdf8";
    return value >= 60 ? "var(--success)" : "var(--warning)";
  };

  return (
    <>
      {/* スコアと改善ポイント */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "0.75rem",
          marginBottom: "1rem",
        }}
      >
        <div className="card" style={{ padding: "1rem" }}>
          <div style={{ fontSize: "0.7rem", color: "var(--muted)", marginBottom: "0.35rem", letterSpacing: "0.05em" }}>
            OVERALL SCORE
          </div>
          <p style={{ fontSize: "1.75rem", fontWeight: 700, color: "var(--success)", margin: 0 }}>
            {data.overallScore} / 100
          </p>
        </div>
        <div className="card" style={{ padding: "1rem" }}>
          <div style={{ fontSize: "0.7rem", color: "var(--muted)", marginBottom: "0.35rem", letterSpacing: "0.05em" }}>
            改善ポイント
          </div>
          <p style={{ fontSize: "1rem", fontWeight: 700, color: "#fff", margin: "0 0 0.25rem 0" }}>
            {data.focusLabel}
          </p>
          <p style={{ fontSize: "0.8rem", color: "var(--accent)", lineHeight: 1.4, margin: 0 }}>
            {data.focusAdvice}
          </p>
        </div>
      </div>

      {/* 技術分析バー */}
      <div style={{ marginBottom: "1rem" }}>
        <p style={{ fontSize: "0.7rem", color: "var(--muted)", letterSpacing: "0.08em", marginBottom: "0.75rem" }}>
          TECHNICAL BREAKDOWN
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
          {data.technical.map(({ key, label, value }) => (
            <div key={key} style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
              <span style={{ fontSize: "0.875rem", color: "var(--text)", minWidth: 100 }}>{label}</span>
              <div
                style={{
                  flex: 1,
                  height: 8,
                  background: "var(--bg)",
                  borderRadius: 4,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${value}%`,
                    height: "100%",
                    background: barColor(key, value),
                    borderRadius: 4,
                    transition: "width 0.3s ease",
                  }}
                />
              </div>
              <span
                style={{
                  fontSize: "0.875rem",
                  fontWeight: 600,
                  color: barColor(key, value),
                  minWidth: 28,
                }}
              >
                {value}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* AI コーチ */}
      <div
        className="card"
        style={{ padding: "1rem", marginBottom: "1rem", border: "1px solid var(--success)" }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.5rem" }}>
          <span style={{ fontSize: "0.7rem", letterSpacing: "0.05em", color: "var(--muted)" }}>
            AI COACH
          </span>
          <span style={{ fontSize: "0.7rem", color: "var(--warning)" }}>⚡ ONLINE</span>
        </div>
        {data.aiText && (
          <p style={{ fontSize: "0.9rem", color: "var(--text)", lineHeight: 1.6, margin: 0 }}>
            {data.aiText}
          </p>
        )}
        {data.practice.length > 0 && (
          <ul style={{ margin: "0.5rem 0 0 1.25rem", padding: 0, fontSize: "0.875rem", color: "var(--muted)" }}>
            {data.practice.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
        )}
      </div>
    </>
  );
}

// ============================================================
// メインコンテンツ
// ============================================================
function SessionResultContent() {
  const searchParams = useSearchParams();
  const sessionId = searchParams.get("session_id");
  const { videoBlob, clearVideo } = useSessionVideo();

  const [session, setSession] = useState<TrackerSession | null>(null);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // 複数サーブがある場合、どのサーブの解析を表示するか
  const [selectedServeId, setSelectedServeId] = useState<string | null>(null);
  const analysisTriggeredRef = useRef(false);

  const loadSession = useCallback(async () => {
    if (!sessionId) return;
    try {
      const data = await getSession(sessionId);
      setSession(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load session");
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId) {
      setLoading(false);
      setError("No session");
      return;
    }
    loadSession();
  }, [sessionId, loadSession]);

  // 録画＋タイムスタンプが揃っていれば自動でフォーム解析を実行
  useEffect(() => {
    if (!sessionId || !session || loading) return;
    if (!videoBlob || !session.serves?.length || analyzing || analysisTriggeredRef.current) return;
    const hasAnalysis = session.analysis && Object.keys(session.analysis).length > 0;
    if (hasAnalysis) {
      clearVideo();
      return;
    }
    const hasTimestamps = session.serves.some((s) => typeof s.timestamp_sec === "number");
    if (!hasTimestamps) {
      clearVideo();
      return;
    }
    analysisTriggeredRef.current = true;
    setAnalyzing(true);
    setError(null);
    const serveEvents = session.serves
      .filter((s) => typeof s.timestamp_sec === "number")
      .map((s) => ({ id: s.id, result: s.result, timestamp_sec: s.timestamp_sec! }));
    runFormAnalysis(sessionId, videoBlob, serveEvents)
      .then((updated) => {
        setSession(updated);
        clearVideo();
        setTimeout(() => {
          getSession(sessionId).then(setSession).catch(() => {});
        }, 300);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "姿勢解析に失敗しました"))
      .finally(() => setAnalyzing(false));
  }, [sessionId, videoBlob, session, loading, analyzing, clearVideo]);

  // 手動でフォーム解析を実行するボタン用ハンドラ
  const handleFormAnalysis = async () => {
    if (!sessionId || !videoBlob || !session?.serves?.length) return;
    setAnalyzing(true);
    setError(null);
    try {
      const serveEvents = session.serves
        .filter((s) => typeof s.timestamp_sec === "number")
        .map((s) => ({ id: s.id, result: s.result, timestamp_sec: s.timestamp_sec! }));
      const updated = await runFormAnalysis(sessionId, videoBlob, serveEvents);
      setSession(updated);
      clearVideo();
      getSession(sessionId).then(setSession).catch(() => {});
    } catch (e) {
      setError(e instanceof Error ? e.message : "解析に失敗しました");
    } finally {
      setAnalyzing(false);
    }
  };

  // ローディング中
  if (loading) {
    return (
      <main className="container" style={{ paddingTop: "2rem", textAlign: "center" }}>
        <p className="text-muted">Loading…</p>
      </main>
    );
  }

  // セッションが見つからない場合
  if (error && !session) {
    return (
      <main className="container" style={{ paddingTop: "2rem" }}>
        <div className="card text-center">
          <p style={{ color: "var(--error)" }}>{error || "Session not found"}</p>
          <Link href="/" className="btn btn-primary">Home</Link>
        </div>
      </main>
    );
  }

  if (!session) return null;

  // 表示に使うデータを整理
  const analysisMap = session.analysis ?? {};
  const serves = session.serves ?? [];
  const servesWithAnalysis = serves.filter((s) => analysisMap[s.id]);
  const hasAnalysis = servesWithAnalysis.length > 0;

  // 選択中のサーブ（デフォルトは最初の解析済みサーブ）
  const activeServeId = selectedServeId ?? servesWithAnalysis[0]?.id ?? null;
  const activeAnalysis = activeServeId ? analysisMap[activeServeId] : null;

  // 動画 URL（トラッキング動画があればそちらを優先）
  const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
  const trackingVideoUrl = session.tracking_video
    ? `${apiBase}/outputs/${session.tracking_video}`
    : null;
  const sessionVideoUrl =
    session.video_path && sessionId
      ? `${apiBase}/api/sessions/${sessionId}/video`
      : null;
  const videoUrl = trackingVideoUrl ?? sessionVideoUrl;

  // フォーム解析ボタンを表示する条件
  const canAnalyze =
    !!videoBlob &&
    serves.length > 0 &&
    serves.some((s) => typeof s.timestamp_sec === "number");

  return (
    <main className="container" style={{ paddingTop: "1rem", paddingBottom: "5rem" }}>

      {/* セッション統計（常に表示） */}
      <div style={{ marginBottom: "1rem" }}>
        <TrackerStats session={session} />
      </div>

      {/* 解析動画プレイヤー */}
      {videoUrl && (
        <div
          className="card"
          style={{ padding: 0, marginBottom: "1rem", overflow: "hidden" }}
        >
          <video
            src={videoUrl}
            controls
            autoPlay
            muted
            playsInline
            loop
            style={{ width: "100%", display: "block" }}
          />
          <div style={{ padding: "0.4rem 0.75rem", fontSize: "0.75rem", color: "var(--muted)" }}>
            {session.tracking_video ? "骨格トラッキング付き解析動画" : "セッション録画"}
          </div>
        </div>
      )}

      {/* 解析中インジケーター */}
      {analyzing && (
        <div
          className="card"
          style={{ padding: "1.5rem", textAlign: "center", marginBottom: "1rem" }}
        >
          <p className="text-muted">フォームを解析しています…</p>
        </div>
      )}

      {/* 解析結果セクション */}
      {hasAnalysis && !analyzing && (
        <>
          {/* 複数サーブがある場合はサーブ切り替えボタンを表示 */}
          {serves.length > 1 && (
            <div
              style={{
                display: "flex",
                gap: "0.5rem",
                flexWrap: "wrap",
                marginBottom: "1rem",
              }}
            >
              {serves.map((s, i) => {
                const hasA = !!analysisMap[s.id];
                const isActive = s.id === activeServeId;
                return (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => {
                      if (hasA) setSelectedServeId(s.id);
                    }}
                    style={{
                      padding: "0.4rem 0.75rem",
                      borderRadius: 8,
                      border: isActive
                        ? "2px solid var(--accent)"
                        : "2px solid transparent",
                      background: isActive
                        ? "rgba(88,166,255,0.1)"
                        : "var(--surface)",
                      color: hasA ? "var(--text)" : "var(--muted)",
                      cursor: hasA ? "pointer" : "default",
                      display: "flex",
                      alignItems: "center",
                      gap: "0.4rem",
                      fontSize: "0.875rem",
                    }}
                  >
                    <span style={resultBadge(s.result)}>{s.result}</span>
                    <span>#{i + 1}</span>
                  </button>
                );
              })}
            </div>
          )}

          {/* 選択中のサーブの解析パネル */}
          {activeAnalysis && <AnalysisPanel analysis={activeAnalysis} />}
        </>
      )}

      {/* 解析なし・解析ボタン表示 */}
      {!hasAnalysis && !analyzing && (
        <div style={{ marginBottom: "1rem" }}>
          {serves.length === 0 ? (
            <p className="text-muted">サーブが記録されていません。</p>
          ) : canAnalyze ? (
            <button
              type="button"
              className="btn btn-primary btn-block"
              onClick={handleFormAnalysis}
              style={{ padding: "0.75rem" }}
            >
              フォーム解析する
            </button>
          ) : (
            <p className="text-muted" style={{ fontSize: "0.875rem" }}>
              フォーム解析を行うには、セッション終了時に録画が必要です。
            </p>
          )}
        </div>
      )}

      {/* エラー表示 */}
      {error && (
        <p style={{ color: "var(--error)", fontSize: "0.875rem", marginBottom: "1rem" }}>
          {error}
        </p>
      )}

      {/* ナビゲーション */}
      <div style={{ display: "flex", gap: "1rem", justifyContent: "center", marginTop: "1rem" }}>
        <Link href="/history" style={{ color: "var(--accent)" }}>← 履歴へ</Link>
        <Link href="/" style={{ color: "var(--muted)" }}>ホームへ</Link>
      </div>
    </main>
  );
}

export default function SessionResultPage() {
  return (
    <Suspense
      fallback={
        <main className="container" style={{ paddingTop: "2rem", textAlign: "center" }}>
          <p className="text-muted">Loading…</p>
        </main>
      }
    >
      <SessionResultContent />
    </Suspense>
  );
}
