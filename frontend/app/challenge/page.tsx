"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { startTrackerSession } from "@/lib/api";
import { getDailyChallenge } from "@/lib/constants";

export default function ChallengePage() {
  const router = useRouter();
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const challenge = getDailyChallenge();

  const handleStart = async () => {
    setStarting(true);
    setError(null);
    try {
      const { session_id } = await startTrackerSession();
      const params = new URLSearchParams({ session_id, challenge: challenge.id });
      router.push(`/practice?${params.toString()}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start session");
    } finally {
      setStarting(false);
    }
  };

  return (
    <div className="challenge-screen-wrap">
      <main
        className="container"
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
          <div
            className="card"
            style={{
              padding: "1.5rem 1.25rem",
              marginBottom: "1rem",
              width: "100%",
              borderLeft: "4px solid var(--accent)",
              background: "rgba(88, 166, 255, 0.08)",
            }}
          >
            <h2 style={{ fontSize: "1rem", fontWeight: 700, color: "#fff", marginBottom: "1rem" }}>
              今日のチャレンジ
            </h2>
            <ul style={{ margin: "0 0 1.25rem 1.25rem", padding: 0, fontSize: "0.95rem", color: "var(--muted)" }}>
              <li>{challenge.minServes}本サーブ</li>
              <li>IN率 {challenge.minAccuracy}%以上</li>
            </ul>
            <button
              type="button"
              className="btn btn-primary btn-block"
              onClick={handleStart}
              disabled={starting}
              style={{ padding: "0.75rem" }}
            >
              {starting ? "Starting…" : "チャレンジを始める"}
            </button>
          </div>

          <p className="text-muted" style={{ fontSize: "0.8rem", marginTop: "0.5rem" }}>
            <Link href="/" style={{ color: "var(--accent)", textDecoration: "underline" }}>
              ホームで練習を始める
            </Link>
          </p>

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
