"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import type { AnalyzeResponse } from "@/lib/api";

const RESULT_KEY = "serveAnalysisResult";

export default function ResultPage() {
  const [data, setData] = useState<AnalyzeResponse | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const raw = sessionStorage.getItem(RESULT_KEY);
    if (raw) {
      try {
        setData(JSON.parse(raw) as AnalyzeResponse);
      } catch {
        setData(null);
      }
    }
  }, []);

  if (data === null) {
    return (
      <main className="container" style={{ paddingTop: "2rem" }}>
        <div className="card text-center">
          <p className="text-muted mb-2">結果が見つかりませんでした。</p>
          <Link href="/upload" className="btn btn-primary">
            もう一度解析する
          </Link>
        </div>
        <p className="text-center mt-2">
          <Link href="/">ホームに戻る</Link>
        </p>
      </main>
    );
  }

  const { score, feedback, metrics } = data;

  return (
    <main className="container" style={{ paddingTop: "2rem" }}>
      <div className="card">
        <h1 style={{ fontSize: "1.25rem", marginBottom: "1rem" }}>解析結果</h1>
        <p style={{ fontSize: "1.5rem", fontWeight: 700, marginBottom: "1rem" }}>
          Serve Score: {score} / 100
        </p>

        <h2 style={{ fontSize: "1rem", marginBottom: "0.5rem" }}>Feedback</h2>
        <ul style={{ margin: "0 0 1rem 1.25rem", padding: 0 }}>
          {feedback.map((item, i) => (
            <li key={i} style={{ marginBottom: "0.25rem" }}>
              {item}
            </li>
          ))}
        </ul>

        <h2 style={{ fontSize: "1rem", marginBottom: "0.5rem" }}>Metrics</h2>
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontSize: "0.9rem",
          }}
        >
          <thead>
            <tr style={{ borderBottom: "1px solid var(--muted)" }}>
              <th style={{ textAlign: "left", padding: "0.5rem 0" }}>Metric</th>
              <th style={{ textAlign: "right", padding: "0.5rem 0" }}>Value</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: "1px solid var(--surface)" }}>
              <td style={{ padding: "0.5rem 0" }}>elbow_angle</td>
              <td style={{ textAlign: "right", padding: "0.5rem 0" }}>
                {metrics.elbow_angle}
              </td>
            </tr>
            <tr style={{ borderBottom: "1px solid var(--surface)" }}>
              <td style={{ padding: "0.5rem 0" }}>body_sway</td>
              <td style={{ textAlign: "right", padding: "0.5rem 0" }}>
                {metrics.body_sway}
              </td>
            </tr>
            <tr style={{ borderBottom: "1px solid var(--surface)" }}>
              <td style={{ padding: "0.5rem 0" }}>impact_height</td>
              <td style={{ textAlign: "right", padding: "0.5rem 0" }}>
                {metrics.impact_height}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="text-center">
        <Link href="/upload">別の動画を解析する</Link>
        {" · "}
        <Link href="/">ホームに戻る</Link>
      </p>
    </main>
  );
}
