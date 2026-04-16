"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { Video, Hourglass } from "lucide-react";
import {
  startSession,
  recordServe,
  runFormAnalysis,
} from "@/lib/api";
import { fromAnalysis, type UploadResultData } from "@/lib/uploadResult";

/** 結果表示用（videoUrl を追加したローカル型） */
interface DisplayResult extends UploadResultData {
  videoUrl: string | null;
}

const SAMPLE_RESULT: DisplayResult = {
  videoUrl: null,
  overallScore: 78,
  improvementMessage: "前回より打点が高くなっています",
  focusLabel: "体の安定",
  focusAdvice: "スイング中に体がブレやすいです。下半身を安定させて打ちましょう。",
  technical: [
    { key: "impact_height", label: "トス高さ", value: 82 },
    { key: "elbow_angle", label: "肘の余裕", value: 75 },
    { key: "body_sway", label: "体の安定", value: 65 },
    { key: "waist_speed", label: "腰のキレ", value: 85 },
    { key: "weight_transfer", label: "体重移動", value: 83 },
  ],
  aiText: "打点は良くなっています。体軸をキープする意識で振るとさらに安定します。",
  practice: ["タオルを上に放り投げる動きを10回繰り返しましょう"],
};

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DisplayResult | null>(null);
  const progressIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("動画を選択してください");
      return;
    }
    setError(null);
    setResult(null);
    setLoading(true);
    setProgress(0);

    if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);
    progressIntervalRef.current = setInterval(() => {
      setProgress((p) => {
        if (p >= 95) return p;
        return p + Math.random() * 8 + 2;
      });
    }, 600);

    try {
      const { session_id } = await startSession();
      const { serve } = await recordServe(session_id, "IN", 0);
      if (!serve || typeof serve !== "object" || !("id" in serve)) {
        throw new Error("サーブの記録に失敗しました");
      }
      const serveId = (serve as { id: string }).id;
      const blob = new Blob([await file.arrayBuffer()], { type: file.type });
      const session = await runFormAnalysis(session_id, blob, [
        { id: serveId, result: "IN", timestamp_sec: 0 },
      ]);
      const analysisMap = session.analysis ?? {};
      const analysis = analysisMap[serveId] ?? null;
      if (analysis) {
        const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
        let videoUrl: string | null = null;
        if (session.tracking_video) {
          videoUrl = `${apiBase}/outputs/${session.tracking_video}`;
        } else if (session.video_path && session.id) {
          videoUrl = `${apiBase}/api/sessions/${session.id}/video`;
        }
        const displayResult: DisplayResult = { ...fromAnalysis(analysis), videoUrl };
        setResult(displayResult);
      } else {
        setError("解析結果を取得できませんでした");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "解析に失敗しました");
    } finally {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }
      setLoading(false);
      setProgress(100);
    }
  };

  const handleAnalyzeAnother = () => {
    setResult(null);
    setFile(null);
    setError(null);
  };

  const showSampleResult = () => {
    setResult(SAMPLE_RESULT);
    setError(null);
  };

  if (loading) {
    return <AnalyzingScreen progress={progress} />;
  }

  if (result !== null) {
    return <ResultScreen data={result} onRecordAgain={handleAnalyzeAnother} />;
  }

  return (
    <main
      className="container"
      style={{
        paddingTop: "2rem",
        minHeight: "60vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <form
        onSubmit={handleSubmit}
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          width: "100%",
        }}
      >
        <label
          htmlFor="video-upload"
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            cursor: "pointer",
          }}
        >
          <span
            style={{
              width: 160,
              height: 160,
              borderRadius: "50%",
              border: "3px dashed #2dd4bf",
              background: "transparent",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              marginBottom: "1.5rem",
            }}
          >
            <Video size={56} strokeWidth={1.5} className="upload-video-icon" />
          </span>
          <span
            style={{
              fontSize: "1.25rem",
              fontWeight: 700,
              color: "#fff",
              marginBottom: "0.5rem",
            }}
          >
            動画をアップロード
          </span>
          <span style={{ fontSize: "0.9rem", color: "var(--muted)" }}>
            サーブの動画を1本選んでください
          </span>
          <input
            id="video-upload"
            type="file"
            accept="video/*"
            className="sr-only"
            onChange={(e) => {
              setFile(e.target.files?.[0] ?? null);
              setError(null);
            }}
            disabled={loading}
          />
        </label>
        {file && (
          <p style={{ fontSize: "0.875rem", color: "var(--muted)", marginTop: "0.5rem" }}>
            {file.name}
          </p>
        )}
        {error && (
          <p
            style={{
              color: "var(--error)",
              fontSize: "0.875rem",
              marginTop: "0.75rem",
            }}
          >
            {error}
          </p>
        )}
        <button
          type="submit"
          className="btn btn-primary"
          disabled={loading || !file}
          style={{ marginTop: "1.5rem", padding: "0.75rem 2rem" }}
        >
          {loading ? "解析中..." : "解析する"}
        </button>
      </form>
      <p className="text-muted" style={{ fontSize: "0.8rem", marginTop: "2rem" }}>
        <button
          type="button"
          onClick={showSampleResult}
          style={{
            background: "none",
            border: "none",
            color: "var(--accent)",
            cursor: "pointer",
            textDecoration: "underline",
          }}
        >
          サンプル結果を表示
        </button>
      </p>
    </main>
  );
}

function ResultScreen({
  data,
  onRecordAgain,
}: {
  data: DisplayResult;
  onRecordAgain: () => void;
}) {
  const {
    videoUrl,
    overallScore,
    improvementMessage,
    focusLabel,
    focusAdvice,
    technical,
    aiText,
    practice,
  } = data;

  const barColor = (key: string, value: number) => {
    if (key === "impact_height") return "var(--warning)";
    if (key === "waist_speed") return "#38bdf8";
    return value >= 60 ? "var(--success)" : "var(--warning)";
  };

  return (
    <div className="result-screen-wrap">
      <div className="result-bg" aria-hidden />
      <main
        className="container result-content"
        style={{ paddingTop: "1rem", paddingBottom: "2rem" }}
      >
        {videoUrl && (
          <div
            className="card"
            style={{
              padding: 0,
              marginBottom: "1rem",
              overflow: "hidden",
            }}
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
            <div
              style={{
                padding: "0.5rem 0.75rem",
                fontSize: "0.75rem",
                color: "var(--muted)",
              }}
            >
              骨格トラッキング付き解析動画
            </div>
          </div>
        )}

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "0.75rem",
            marginBottom: "1rem",
          }}
        >
          <div className="card" style={{ padding: "1rem" }}>
            <div
              style={{
                fontSize: "0.7rem",
                color: "var(--muted)",
                marginBottom: "0.35rem",
                letterSpacing: "0.05em",
              }}
            >
              OVERALL SCORE
            </div>
            <p
              style={{
                fontSize: "1.75rem",
                fontWeight: 700,
                color: "var(--success)",
                margin: "0 0 0.5rem 0",
              }}
            >
              {overallScore} / 100
            </p>
            {improvementMessage && (
              <div
                style={{
                  border: "1px solid var(--success)",
                  borderRadius: 8,
                  padding: "0.5rem 0.6rem",
                  background: "rgba(63,185,80,0.08)",
                }}
              >
                <span
                  style={{
                    fontSize: "0.8rem",
                    color: "var(--success)",
                  }}
                >
                  {improvementMessage}
                </span>
              </div>
            )}
          </div>
          <div className="card" style={{ padding: "1rem" }}>
            <div
              style={{
                fontSize: "0.7rem",
                color: "var(--muted)",
                marginBottom: "0.35rem",
                letterSpacing: "0.05em",
              }}
            >
              改善ポイント
            </div>
            <p
              style={{
                fontSize: "1rem",
                fontWeight: 700,
                color: "#fff",
                margin: "0 0 0.5rem 0",
              }}
            >
              {focusLabel}
            </p>
            <p
              style={{
                fontSize: "0.8rem",
                color: "var(--accent)",
                lineHeight: 1.4,
                margin: 0,
              }}
            >
              {focusAdvice}
            </p>
          </div>
        </div>

        <div style={{ marginBottom: "1.25rem" }}>
          <h2
            style={{
              fontSize: "0.7rem",
              color: "var(--muted)",
              letterSpacing: "0.08em",
              marginBottom: "0.75rem",
            }}
          >
            TECHNICAL BREAKDOWN
          </h2>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "0.6rem",
            }}
          >
            {technical.map(({ key, label, value }) => (
              <div
                key={key}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.75rem",
                }}
              >
                <span
                  style={{
                    fontSize: "0.875rem",
                    color: "var(--text)",
                    minWidth: 100,
                  }}
                >
                  {label}
                </span>
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

        <div
          className="card"
          style={{
            padding: "1rem",
            marginBottom: "1.5rem",
            border: "1px solid var(--success)",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
              marginBottom: "0.5rem",
            }}
          >
            <span
              style={{
                fontSize: "0.7rem",
                letterSpacing: "0.05em",
                color: "var(--muted)",
              }}
            >
              AI COACH
            </span>
            <span
              style={{
                fontSize: "0.7rem",
                color: "var(--warning)",
                display: "flex",
                alignItems: "center",
                gap: "0.25rem",
              }}
            >
              ⚡ ONLINE
            </span>
          </div>
          {aiText && (
            <p
              style={{
                fontSize: "0.9rem",
                color: "var(--text)",
                lineHeight: 1.6,
                margin: 0,
              }}
            >
              {aiText}
            </p>
          )}
          {practice.length > 0 && (
            <ul
              style={{
                margin: "0.5rem 0 0 1.25rem",
                padding: 0,
                fontSize: "0.875rem",
                color: "var(--muted)",
              }}
            >
              {practice.map((p, i) => (
                <li key={i}>{p}</li>
              ))}
            </ul>
          )}
        </div>

        <button
          type="button"
          onClick={onRecordAgain}
          className="btn btn-primary btn-block"
          style={{ padding: "0.9rem" }}
        >
          もう一回撮影
        </button>

        <p style={{ marginTop: "1rem", textAlign: "center" }}>
          <Link href="/" style={{ color: "var(--accent)" }}>
            ← ホームへ
          </Link>
        </p>
      </main>
    </div>
  );
}

function AnalyzingScreen({ progress: progressProp }: { progress?: number }) {
  const [localProgress, setLocalProgress] = useState(0);
  useEffect(() => {
    if (typeof progressProp === "number" && progressProp >= 0) return;
    const t = setInterval(() => {
      setLocalProgress((p) => (p >= 99 ? 99 : p + 1));
    }, 800);
    return () => clearInterval(t);
  }, [progressProp]);
  const progress =
    typeof progressProp === "number" && progressProp >= 0
      ? progressProp
      : localProgress;

  return (
    <div className="analyzing-screen-wrap">
      <div className="analyzing-bg" aria-hidden />
      <main
        className="container analyzing-content"
        style={{
          paddingTop: "2rem",
          minHeight: "60vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            width: 160,
            height: 160,
            borderRadius: "50%",
            border: "3px dashed #22d3ee",
            background: "transparent",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: "1.5rem",
            boxShadow:
              "0 0 24px rgba(34, 211, 238, 0.4), 0 0 48px rgba(34, 211, 238, 0.2)",
          }}
        >
          <Hourglass
            size={56}
            strokeWidth={1.5}
            className="analyzing-hourglass-icon"
          />
        </div>
        <p
          style={{
            fontSize: "1.1rem",
            fontWeight: 600,
            color: "#fff",
            marginBottom: "0.35rem",
          }}
        >
          AIが解析しています...
        </p>
        <p
          style={{
            fontSize: "0.9rem",
            color: "var(--muted)",
            marginBottom: "1.25rem",
          }}
        >
          骨格の動きを検出しています
        </p>
        <div style={{ width: "100%", maxWidth: 280, marginBottom: "1rem" }}>
          <div
            style={{
              height: 10,
              borderRadius: 5,
              background: "var(--surface)",
              overflow: "hidden",
              border: "1px solid rgba(34, 211, 238, 0.3)",
            }}
          >
            <div
              style={{
                height: "100%",
                width: `${progress}%`,
                borderRadius: 5,
                background: "linear-gradient(90deg, #22d3ee, #0ea5e9)",
                boxShadow: "0 0 12px rgba(34, 211, 238, 0.5)",
                transition: "width 0.4s ease-out",
              }}
            />
          </div>
        </div>
        <p
          style={{
            fontSize: "2.5rem",
            fontWeight: 700,
            color: "#22d3ee",
            textShadow:
              "0 0 20px rgba(34, 211, 238, 0.8), 0 0 40px rgba(34, 211, 238, 0.4)",
          }}
        >
          {Math.round(progress)}%
        </p>
      </main>
    </div>
  );
}
