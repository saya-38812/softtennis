"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { getTrackerSession, type TrackerSession } from "@/lib/api";
import { CHALLENGES } from "@/lib/constants";
import { accuracyPercent, checkChallengeClear } from "@/lib/utils";

function SessionResultContent() {
  const searchParams = useSearchParams();
  const sessionId = searchParams.get("session_id");
  const challengeId = searchParams.get("challenge");
  const challenge = challengeId ? CHALLENGES.find((c) => c.id === challengeId) ?? null : null;

  const [session, setSession] = useState<TrackerSession | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) {
      setLoading(false);
      setError("No session");
      return;
    }
    getTrackerSession(sessionId)
      .then(setSession)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed"))
      .finally(() => setLoading(false));
  }, [sessionId]);

  const challengeResult = checkChallengeClear(session, challenge);

  if (loading) {
    return (
      <main className="container" style={{ paddingTop: "2rem", textAlign: "center" }}>
        <p className="text-muted">Loading…</p>
      </main>
    );
  }

  if (error || !session) {
    return (
      <main className="container" style={{ paddingTop: "2rem" }}>
        <div className="card text-center">
          <p style={{ color: "var(--error)" }}>{error || "Session not found"}</p>
          <Link href="/" className="btn btn-primary">Home</Link>
        </div>
      </main>
    );
  }

  const total = session.total_attempts;
  const inCount = session.in_count;
  const outCount = session.out_count;
  const faultCount = session.fault_count;
  const accuracy = accuracyPercent(session);

  return (
    <main className="container" style={{ paddingTop: "1rem" }}>
      <p className="text-muted" style={{ fontSize: "0.875rem", marginBottom: "1rem" }}>
        {new Date(session.date).toLocaleString("ja-JP")}
      </p>
      {challenge && challengeResult != null && (
        <div
          className="card"
          style={{
            marginBottom: "1rem",
            borderLeft: `4px solid ${challengeResult.clear ? "var(--success)" : "var(--warning)"}`,
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: "0.25rem" }}>
            {challengeResult.clear ? "クリア！" : "クリアならず"}
          </div>
          <div className="text-muted" style={{ fontSize: "0.875rem", marginBottom: "0.5rem" }}>
            今日のチャレンジ
          </div>
          <ul style={{ margin: 0, paddingLeft: "1.25rem", fontSize: "0.875rem" }}>
            <li className={challengeResult.servesOk ? "" : "text-muted"}>
              {challengeResult.servesOk ? "✓" : "✗"} 本数: {total} / {challenge.minServes}本
            </li>
            <li className={challengeResult.accuracyOk ? "" : "text-muted"}>
              {challengeResult.accuracyOk ? "✓" : "✗"} IN率: {accuracy}% / {challenge.minAccuracy}%以上
            </li>
          </ul>
        </div>
      )}
      <div className="card">
        <h2 style={{ fontSize: "1rem", marginBottom: "0.5rem" }}>Total Serves: {total}</h2>
        <p style={{ marginBottom: "1rem" }}>
          IN: <strong>{inCount}</strong> &nbsp; OUT: <strong>{outCount}</strong> &nbsp; FAULT: <strong>{faultCount}</strong>
        </p>
        <p style={{ fontSize: "1.25rem", marginBottom: "1rem" }}>
          IN率: <strong>{accuracy}%</strong>
        </p>
        <p className="text-muted" style={{ fontSize: "0.9rem" }}>
          動画アップロードでフォーム解析が利用できます。（Upload Video）
        </p>
      </div>

      <div style={{ display: "flex", gap: "0.5rem", justifyContent: "center", flexWrap: "wrap" }}>
        <Link href="/practice" className="btn btn-outline">
          New Practice
        </Link>
        <Link href="/" className="btn btn-primary">Home</Link>
      </div>
    </main>
  );
}

export default function SessionResultPage() {
  return (
    <Suspense fallback={
      <main className="container" style={{ paddingTop: "2rem", textAlign: "center" }}>
        <p className="text-muted">Loading…</p>
      </main>
    }>
      <SessionResultContent />
    </Suspense>
  );
}
