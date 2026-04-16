"use client";

import { useState } from "react";
import Link from "next/link";
import { SessionList } from "@/components/history/SessionList";
import { useSessionList } from "@/hooks/useSessionList";
import { resetAll, seedSample } from "@/lib/api";

export default function HistoryPage() {
  const { sessions, loading, error, deleteSession, refresh } = useSessionList();
  const [deletingAll, setDeletingAll] = useState(false);
  const [loadingSample, setLoadingSample] = useState(false);

  const handleDeleteAll = async () => {
    if (typeof window !== "undefined" && !window.confirm("すべてのセッション・動画を削除しますか？この操作は取り消せません。")) return;
    setDeletingAll(true);
    try {
      await resetAll();
      await refresh();
    } catch (e) {
      if (typeof window !== "undefined") window.alert(e instanceof Error ? e.message : "全削除に失敗しました");
    } finally {
      setDeletingAll(false);
    }
  };

  const handleLoadSample = async () => {
    setLoadingSample(true);
    try {
      const { created, message } = await seedSample();
      await refresh();
      if (typeof window !== "undefined") {
        if (created.length > 0) {
          window.alert("サンプルデータを読み込みました。履歴からタップして結果を確認できます。");
        } else {
          window.alert(message || "サンプルは既に読み込まれています。");
        }
      }
    } catch (e) {
      if (typeof window !== "undefined") window.alert(e instanceof Error ? e.message : "サンプルの読み込みに失敗しました");
    } finally {
      setLoadingSample(false);
    }
  };

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
        <div className="card text-center" style={{ padding: "1.5rem" }}>
          <p className="text-muted">まだセッションがありません。練習を開始して記録しましょう。</p>
          <p className="text-muted" style={{ fontSize: "0.875rem", marginTop: "0.5rem" }}>
            録画ができない環境では、サンプルデータで結果画面を確認できます。
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem", marginTop: "1.25rem", alignItems: "center" }}>
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleLoadSample}
              disabled={loadingSample}
            >
              {loadingSample ? "読み込み中…" : "サンプルデータを読み込む"}
            </button>
            <Link href="/" className="btn btn-outline">
              ホームへ
            </Link>
          </div>
        </div>
      ) : (
        <>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.5rem", marginBottom: "0.75rem" }}>
            <button
              type="button"
              className="btn btn-outline"
              style={{ fontSize: "0.875rem", padding: "0.5rem 1rem" }}
              onClick={handleLoadSample}
              disabled={loadingSample}
            >
              {loadingSample ? "読み込み中…" : "サンプルデータを読み込む"}
            </button>
            <button
              type="button"
              className="btn btn-outline"
              style={{ fontSize: "0.875rem", padding: "0.5rem 1rem", color: "var(--error)", borderColor: "var(--error)" }}
              onClick={handleDeleteAll}
              disabled={deletingAll}
            >
              {deletingAll ? "削除中…" : "全削除"}
            </button>
          </div>
          <SessionList sessions={sessions} onDelete={deleteSession} />
        </>
      )}
    </main>
  );
}
