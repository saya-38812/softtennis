"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { listSessions, type TrackerSession } from "@/lib/api";
import { accuracyPercent } from "@/lib/utils";

/** 全セッションを集計 */
function aggregate(sessions: TrackerSession[]) {
  let total = 0;
  let inSum = 0;
  let outSum = 0;
  let faultSum = 0;
  for (const s of sessions) {
    total += s.total_attempts ?? 0;
    inSum += s.in_count ?? 0;
    outSum += s.out_count ?? 0;
    faultSum += s.fault_count ?? 0;
  }
  const successRate = total > 0 ? Math.round((inSum / total) * 100) : 0;
  return { total, inSum, outSum, faultSum, successRate, sessionCount: sessions.length };
}

export default function StatsPage() {
  const [sessions, setSessions] = useState<TrackerSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listSessions()
      .then(setSessions)
      .catch((e) => setError(e instanceof Error ? e.message : "読み込み失敗"))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <main className="container" style={{ paddingTop: "2rem", textAlign: "center" }}>
        <p className="text-muted">読み込み中…</p>
      </main>
    );
  }

  if (error) {
    return (
      <main className="container" style={{ paddingTop: "2rem", textAlign: "center" }}>
        <p style={{ color: "var(--error)" }}>{error}</p>
      </main>
    );
  }

  if (sessions.length === 0) {
    return (
      <main className="container" style={{ paddingTop: "2rem", textAlign: "center" }}>
        <div className="card">
          <p className="text-muted">まだセッションがありません</p>
          <Link href="/" className="btn btn-primary" style={{ display: "inline-flex", marginTop: "1rem" }}>
            練習を始める
          </Link>
        </div>
      </main>
    );
  }

  const stats = aggregate(sessions);

  return (
    <main className="container" style={{ paddingTop: "1rem", paddingBottom: "5rem" }}>
      <p style={{ fontSize: "0.7rem", color: "var(--muted)", letterSpacing: "0.08em", marginBottom: "1rem" }}>
        OVERALL STATS
      </p>

      {/* 全体統計 */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", marginBottom: "1rem" }}>
        <div className="card" style={{ padding: "1rem" }}>
          <div style={{ fontSize: "0.7rem", color: "var(--muted)", marginBottom: "0.25rem" }}>セッション数</div>
          <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "var(--accent)" }}>
            {stats.sessionCount}
          </div>
        </div>
        <div className="card" style={{ padding: "1rem" }}>
          <div style={{ fontSize: "0.7rem", color: "var(--muted)", marginBottom: "0.25rem" }}>総サーブ数</div>
          <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "var(--text)" }}>
            {stats.total}
          </div>
        </div>
        <div className="card" style={{ padding: "1rem" }}>
          <div style={{ fontSize: "0.7rem", color: "var(--muted)", marginBottom: "0.25rem" }}>IN率</div>
          <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "var(--success)" }}>
            {stats.successRate}%
          </div>
        </div>
        <div className="card" style={{ padding: "1rem" }}>
          <div style={{ fontSize: "0.7rem", color: "var(--muted)", marginBottom: "0.25rem" }}>FAULT 数</div>
          <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "var(--error)" }}>
            {stats.faultSum}
          </div>
        </div>
      </div>

      {/* IN / OUT / FAULT の内訳バー */}
      <div className="card" style={{ padding: "1rem", marginBottom: "1rem" }}>
        <p style={{ fontSize: "0.7rem", color: "var(--muted)", letterSpacing: "0.08em", marginBottom: "0.75rem" }}>
          BREAKDOWN
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
          {[
            { label: "IN", value: stats.inSum, color: "var(--success)" },
            { label: "FAULT", value: stats.faultSum, color: "var(--error)" },
          ].map(({ label, value, color }) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
              <span style={{ fontSize: "0.875rem", color: "var(--text)", minWidth: 52 }}>{label}</span>
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
                    width: `${stats.total > 0 ? (value / stats.total) * 100 : 0}%`,
                    height: "100%",
                    background: color,
                    borderRadius: 4,
                    transition: "width 0.4s ease",
                  }}
                />
              </div>
              <span style={{ fontSize: "0.875rem", fontWeight: 600, color, minWidth: 28 }}>{value}</span>
            </div>
          ))}
        </div>
      </div>

      {/* 最近のセッション一覧（最大5件） */}
      <p style={{ fontSize: "0.7rem", color: "var(--muted)", letterSpacing: "0.08em", marginBottom: "0.5rem" }}>
        RECENT SESSIONS
      </p>
      <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        {sessions.slice(0, 5).map((s) => (
          <Link
            key={s.id}
            href={`/session-result?session_id=${s.id}`}
            style={{
              display: "block",
              padding: "0.75rem 1rem",
              background: "var(--surface)",
              borderRadius: 12,
              color: "inherit",
              textDecoration: "none",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "0.875rem" }}>
                {new Date(s.date).toLocaleDateString("ja-JP")}
              </span>
              <span style={{ fontSize: "0.875rem", color: "var(--success)", fontWeight: 600 }}>
                IN率 {accuracyPercent(s)}%
              </span>
            </div>
            <div style={{ fontSize: "0.8rem", color: "var(--muted)", marginTop: "0.25rem" }}>
              {s.total_attempts}本 &nbsp; IN {s.in_count} / FAULT {s.fault_count}
            </div>
          </Link>
        ))}
      </div>
    </main>
  );
}
