"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { startTrackerSession } from "@/lib/api";

function PlayIcon() {
  return (
    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="32" cy="32" r="28" fill="rgba(88, 166, 255, 0.15)" stroke="var(--accent)" strokeWidth="2" />
      <path d="M26 22v20l18-10-18-10z" fill="var(--accent)" />
    </svg>
  );
}

export default function HomePage() {
  const router = useRouter();
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleStartPractice = async () => {
    setStarting(true);
    setError(null);
    try {
      const { session_id } = await startTrackerSession();
      router.push(`/practice?session_id=${session_id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start session");
    } finally {
      setStarting(false);
    }
  };

  return (
    <div className="home-screen-wrap">
      <div className="home-bg" aria-hidden />
      <main
        className="container home-content"
        style={{
          paddingTop: "1.5rem",
          paddingBottom: "2rem",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          minHeight: "60vh",
        }}
      >
        <div style={{ width: "100%", maxWidth: 360, display: "flex", flexDirection: "column", alignItems: "center" }}>
          <label
            htmlFor="start-practice"
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              cursor: starting ? "not-allowed" : "pointer",
            }}
          >
            <span
              style={{
                width: 160,
                height: 160,
                borderRadius: "50%",
                border: "3px dashed var(--accent)",
                background: "transparent",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                marginBottom: "1.5rem",
                boxShadow: "0 0 24px rgba(88, 166, 255, 0.2)",
              }}
            >
              <PlayIcon />
            </span>
            <span style={{ fontSize: "1.25rem", fontWeight: 700, color: "#fff", marginBottom: "0.5rem" }}>
              練習を始める
            </span>
            <span style={{ fontSize: "0.9rem", color: "var(--muted)", textAlign: "center", marginBottom: "1rem" }}>
              サーブ練習をセッションで記録。IN/OUT/FAULT・フォームを可視化します。
            </span>
          </label>
          <button
            id="start-practice"
            type="button"
            onClick={handleStartPractice}
            disabled={starting}
            className="btn btn-primary btn-block"
            style={{ marginBottom: "0.5rem", padding: "0.75rem 2rem" }}
          >
            {starting ? "Starting…" : "練習を始める"}
          </button>


          {error && (
            <p style={{ color: "var(--error)", fontSize: "0.875rem", marginTop: "1rem", textAlign: "center" }}>
              {error}
            </p>
          )}
        </div>
      </main>
    </div>
  );
}
