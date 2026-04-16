"use client";

import Link from "next/link";
import type { UploadResultData } from "@/lib/uploadResult";

interface UploadStyleResultViewProps {
  data: UploadResultData;
  videoUrl?: string | null;
  /** 下部に表示する要素（例: もう一回撮影ボタン or ホームへリンク）。未指定なら「ホームへ」リンク */
  children?: React.ReactNode;
}

export function UploadStyleResultView({ data, videoUrl, children }: UploadStyleResultViewProps) {
  const {
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
                <span style={{ fontSize: "0.8rem", color: "var(--success)" }}>
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
          <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
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

        {children ?? (
          <p style={{ marginTop: "1rem", textAlign: "center" }}>
            <Link href="/" style={{ color: "var(--accent)" }}>
              ← ホームへ
            </Link>
          </p>
        )}
      </main>
    </div>
  );
}
