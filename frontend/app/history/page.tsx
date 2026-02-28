"use client";

import Link from "next/link";
import { SessionList } from "@/components/history/SessionList";
import { useSessionList } from "@/hooks/useSessionList";

export default function HistoryPage() {
  const { sessions, loading, error, deleteSession } = useSessionList();

  return (
    <main className="container" style={{ paddingTop: "1rem" }}>
      {error && (
          <p style={{ color: "var(--error)", marginBottom: "1rem", fontSize: "0.875rem" }}>
            {error}
          </p>
        )}
        {loading ? (
          <p className="text-muted" style={{ textAlign: "center", padding: "2rem" }}>
            Loading…
          </p>
        ) : sessions.length === 0 ? (
          <div className="card text-center">
            <p className="text-muted">まだセッションがありません。練習を開始して記録しましょう。</p>
            <Link href="/" className="btn btn-primary" style={{ marginTop: "1rem" }}>
              ホームへ
            </Link>
          </div>
        ) : (
          <SessionList sessions={sessions} onDelete={deleteSession} />
        )}
    </main>
  );
}
