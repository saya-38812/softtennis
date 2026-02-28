"use client";

import Link from "next/link";
import type { TrackerSession } from "@/lib/api";
import { accuracyPercent } from "@/lib/utils";

interface SessionListProps {
  sessions: TrackerSession[];
  onDelete?: (sessionId: string) => void;
}

export function SessionList({ sessions, onDelete }: SessionListProps) {
  const handleDelete = (e: React.MouseEvent, sessionId: string) => {
    e.preventDefault();
    e.stopPropagation();
    if (typeof window !== "undefined" && !window.confirm("このセッションを削除しますか？")) return;
    onDelete?.(sessionId);
  };

  if (sessions.length === 0) return null;

  return (
    <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
      {sessions.map((s) => {
        const accuracy = accuracyPercent(s);
        return (
          <li key={s.id} style={{ marginBottom: "0.75rem" }}>
            <div style={{ display: "flex", alignItems: "stretch", gap: "0.5rem" }}>
              <Link
                href={`/session-result?session_id=${s.id}`}
                style={{
                  flex: 1,
                  display: "block",
                  padding: "1rem",
                  background: "var(--surface)",
                  borderRadius: 12,
                  color: "inherit",
                  textDecoration: "none",
                }}
              >
                <div style={{ fontWeight: 600 }}>
                  {new Date(s.date).toLocaleString("ja-JP")}
                </div>
                <div style={{ fontSize: "0.875rem", color: "var(--muted)", marginTop: "0.25rem" }}>
                  {s.total_attempts}本 &nbsp; IN {s.in_count} / OUT {s.out_count} &nbsp; IN率 {accuracy}%
                </div>
              </Link>
              {onDelete && (
                <button
                  type="button"
                  className="btn btn-outline"
                  style={{ alignSelf: "center", padding: "0.5rem 0.75rem", fontSize: "0.875rem" }}
                  onClick={(e) => handleDelete(e, s.id)}
                  title="削除"
                >
                  削除
                </button>
              )}
            </div>
          </li>
        );
      })}
    </ul>
  );
}
