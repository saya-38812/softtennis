"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { Video, Hourglass } from "lucide-react";
import {
  analyzeServe,
  analyzeServeStreaming,
  type AnalyzeResponse,
  type AnalyzeStreamResult,
} from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

/** 結果画面用の統一データ（SSE 結果 or MVP 結果から変換） */
export interface DisplayResult {
  videoUrl: string | null;
  userImageUrl: string | null;
  idealImageUrl: string | null;
  overallScore: number;
  improvementMessage: string;
  focusLabel: string;
  focusAdvice: string;
  technical: { label: string; value: number; key: string }[];
  aiText: string;
  practice: string[];
}

const TECHNICAL_KEYS: { key: string; label: string }[] = [
  { key: "impact_height", label: "トス高さ" },
  { key: "elbow_angle", label: "肘の余裕" },
  { key: "body_sway", label: "体の安定" },
  { key: "waist_speed", label: "腰のキレ" },
  { key: "weight_transfer", label: "体重移動" },
];

function fromStreamResult(r: AnalyzeStreamResult): DisplayResult {
  const ns = r.normalized_scores ?? {};
  const technical = TECHNICAL_KEYS.map(({ key, label }) => ({
    label,
    value: Math.round(ns[key] ?? 0),
    key,
  }));
  const overallScore = technical.length
    ? Math.round(technical.reduce((a, t) => a + t.value, 0) / technical.length)
    : 0;
  return {
    videoUrl: r.user_video ? `${API_BASE}${r.user_video}` : null,
    userImageUrl: r.user_image ? `${API_BASE}${r.user_image}` : null,
    idealImageUrl: r.ideal_image ? `${API_BASE}${r.ideal_image}` : null,
    overallScore,
    improvementMessage: r.improvement_message ?? "",
    focusLabel: r.focus_label ?? "体軸のブレ",
    focusAdvice: r.comparison?.action_tip ?? "スイング中に体を安定させましょう。",
    technical,
    aiText: r.ai_text ?? "",
    practice: Array.isArray(r.practice) ? r.practice : typeof r.practice === "string" ? [r.practice] : [],
  };
}

const SAMPLE_MVP: AnalyzeResponse = {
  score: 85,
  feedback: [
    "前回より打点が高くなっています。",
    "スイング中に体がブレやすいです。下半身を安定させて打ちましょう。",
  ],
  metrics: {
    elbow_angle: 118,
    body_sway: 0.35,
    impact_height: 1.92,
  },
};

function getTechnicalFromMetrics(metrics: AnalyzeResponse["metrics"]) {
  const elbow = Math.round(Math.max(0, Math.min(100, (metrics.elbow_angle - 85) * 2.5)));
  const bodyStability = Math.round(Math.max(0, Math.min(100, 100 - metrics.body_sway * 190)));
  const toss = Math.round(Math.max(0, Math.min(100, (metrics.impact_height - 1) * 55)));
  return [
    { key: "impact_height", label: "トス高さ", value: Math.min(98, Math.max(50, toss)) },
    { key: "elbow_angle", label: "肘の余裕", value: Math.min(95, Math.max(50, elbow)) },
    { key: "body_sway", label: "体の安定", value: Math.max(20, bodyStability) },
    { key: "waist_speed", label: "腰のキレ", value: 98 },
    { key: "weight_transfer", label: "体重移動", value: 98 },
  ];
}

function fromMvpResult(r: AnalyzeResponse): DisplayResult {
  const technical = getTechnicalFromMetrics(r.metrics);
  const weakest = technical.reduce((a, b) => (a.value <= b.value ? a : b));
  const improvementLabels: Record<string, string> = {
    body_sway: "スイング中に体がブレやすいです。下半身を安定させて打ちましょう。",
    elbow_angle: "肘の角度に余裕を持たせ、ラケットをスムーズに振りましょう。",
    impact_height: "トスを安定した高さに上げ、打点を一定にしましょう。",
    waist_speed: "腰の回転を意識してスイングにキレを出しましょう。",
    weight_transfer: "前足への体重移動を意識して打ちましょう。",
  };
  return {
    videoUrl: null,
    userImageUrl: null,
    idealImageUrl: null,
    overallScore: r.score,
    improvementMessage: r.feedback[0] ?? "",
    focusLabel: weakest.label,
    focusAdvice: improvementLabels[weakest.key] ?? "",
    technical,
    aiText: r.feedback[1] ?? "フォームを確認して繰り返し練習しましょう。",
    practice: r.feedback.length > 2 ? r.feedback.slice(2) : [],
  };
}

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DisplayResult | null>(null);

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

    const applyStreamResult = (streamResult: AnalyzeStreamResult) => {
      setResult(fromStreamResult(streamResult));
    };

    try {
      const streamResult = await analyzeServeStreaming(file, (ev) => {
        if (ev.type === "progress") setProgress(ev.percent);
        if (ev.type === "error") setError(ev.message);
        if (ev.type === "result") applyStreamResult(ev.result);
      });
      if (streamResult) applyStreamResult(streamResult);
      else if (!error) setError("解析に失敗しました");
    } catch {
      setError(null);
      try {
        const mvpData = await analyzeServe(file);
        setResult(fromMvpResult(mvpData));
      } catch (err) {
        setError(err instanceof Error ? err.message : "解析に失敗しました");
      }
    } finally {
      setLoading(false);
      setProgress(0);
    }
  };

  const handleAnalyzeAnother = () => {
    setResult(null);
    setFile(null);
    setError(null);
  };

  const showSampleResult = () => {
    setResult(fromMvpResult(SAMPLE_MVP));
    setError(null);
  };

  if (loading) {
    return <AnalyzingScreen progress={progress} />;
  }

  if (result !== null) {
    return <ResultScreen data={result} onRecordAgain={handleAnalyzeAnother} />;
  }

  // アップロード前: 円形破線 + カメラアイコン + 文言（画像どおり）
  return (
    <main className="container" style={{ paddingTop: "2rem", minHeight: "60vh", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "100%" }}>
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
          <span style={{ fontSize: "1.25rem", fontWeight: 700, color: "#fff", marginBottom: "0.5rem" }}>
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
          <p style={{ color: "var(--error)", fontSize: "0.875rem", marginTop: "0.75rem" }}>
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
        <button type="button" onClick={showSampleResult} style={{ background: "none", border: "none", color: "var(--accent)", cursor: "pointer", textDecoration: "underline" }}>
          サンプル結果を表示
        </button>
      </p>
    </main>
  );
}

function ResultScreen({ data, onRecordAgain }: { data: DisplayResult; onRecordAgain: () => void }) {
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
      <main className="container result-content" style={{ paddingTop: "1rem", paddingBottom: "2rem" }}>
      {videoUrl && (
        <div className="card" style={{ padding: 0, marginBottom: "1rem", overflow: "hidden" }}>
          <video
            src={videoUrl}
            controls
            autoPlay
            muted
            playsInline
            loop
            style={{ width: "100%", display: "block" }}
          />
          <div style={{ padding: "0.5rem 0.75rem", fontSize: "0.75rem", color: "var(--muted)" }}>
            骨格トラッキング付き解析動画
          </div>
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", marginBottom: "1rem" }}>
        <div className="card" style={{ padding: "1rem" }}>
          <div style={{ fontSize: "0.7rem", color: "var(--muted)", marginBottom: "0.35rem", letterSpacing: "0.05em" }}>
            OVERALL SCORE
          </div>
          <p style={{ fontSize: "1.75rem", fontWeight: 700, color: "var(--success)", margin: "0 0 0.5rem 0" }}>
            {overallScore} / 100
          </p>
          {improvementMessage && (
            <div style={{ border: "1px solid var(--success)", borderRadius: 8, padding: "0.5rem 0.6rem", background: "rgba(63,185,80,0.08)" }}>
              <span style={{ fontSize: "0.8rem", color: "var(--success)" }}>{improvementMessage}</span>
            </div>
          )}
        </div>
        <div className="card" style={{ padding: "1rem" }}>
          <div style={{ fontSize: "0.7rem", color: "var(--muted)", marginBottom: "0.35rem", letterSpacing: "0.05em" }}>
            改善ポイント
          </div>
          <p style={{ fontSize: "1rem", fontWeight: 700, color: "#fff", margin: "0 0 0.5rem 0" }}>
            {focusLabel}
          </p>
          <div style={{ fontSize: "0.75rem", color: "var(--muted)", marginBottom: "0.25rem" }}>
            Target <span style={{ marginLeft: "0.25rem" }}>You</span>
          </div>
          <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginBottom: "0.35rem" }}>
            <span style={{ fontSize: "0.8rem", color: "var(--muted)" }}>安定</span>
            <span style={{ fontSize: "0.8rem", color: "var(--warning)", fontWeight: 600 }}>改善しよう</span>
          </div>
          <p style={{ fontSize: "0.8rem", color: "var(--accent)", lineHeight: 1.4, margin: 0 }}>
            {focusAdvice}
          </p>
        </div>
      </div>

      <div style={{ marginBottom: "1.25rem" }}>
        <h2 style={{ fontSize: "0.7rem", color: "var(--muted)", letterSpacing: "0.08em", marginBottom: "0.75rem" }}>
          TECHNICAL BREAKDOWN
        </h2>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
          {technical.map(({ key, label, value }) => (
            <div key={key} style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
              <span style={{ fontSize: "0.875rem", color: "var(--text)", minWidth: 100 }}>{label}</span>
              <div style={{ flex: 1, height: 8, background: "var(--bg)", borderRadius: 4, overflow: "hidden" }}>
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
              <span style={{ fontSize: "0.875rem", fontWeight: 600, color: barColor(key, value), minWidth: 28 }}>{value}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="card" style={{ padding: "1rem", marginBottom: "1.5rem", border: "1px solid var(--success)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.5rem" }}>
          <span style={{ fontSize: "0.7rem", letterSpacing: "0.05em", color: "var(--muted)" }}>AI COACH</span>
          <span style={{ fontSize: "0.7rem", color: "var(--warning)", display: "flex", alignItems: "center", gap: "0.25rem" }}>
            ⚡ ONLINE
          </span>
        </div>
        {aiText && <p style={{ fontSize: "0.9rem", color: "var(--text)", lineHeight: 1.6, margin: 0 }}>{aiText}</p>}
        {practice.length > 0 && (
          <ul style={{ margin: "0.5rem 0 0 1.25rem", padding: 0, fontSize: "0.875rem", color: "var(--muted)" }}>
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
  const progress = typeof progressProp === "number" && progressProp >= 0 ? progressProp : localProgress;

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
            boxShadow: "0 0 24px rgba(34, 211, 238, 0.4), 0 0 48px rgba(34, 211, 238, 0.2)",
          }}
        >
          <Hourglass size={56} strokeWidth={1.5} className="analyzing-hourglass-icon" />
        </div>
        <p style={{ fontSize: "1.1rem", fontWeight: 600, color: "#fff", marginBottom: "0.35rem" }}>
          AIが解析しています...
        </p>
        <p style={{ fontSize: "0.9rem", color: "var(--muted)", marginBottom: "1.25rem" }}>
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
            textShadow: "0 0 20px rgba(34, 211, 238, 0.8), 0 0 40px rgba(34, 211, 238, 0.4)",
          }}
        >
          {progress}%
        </p>
      </main>
    </div>
  );
}
