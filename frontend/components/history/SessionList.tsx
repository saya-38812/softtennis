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
                  {/* 日時は created_at（時刻まで入った値）を優先して表示する。
                      date（"YYYY-MM-DD"）を new Date() に渡すと UTC の0時と解釈され、
                      日本時間では「9:00:00」と表示されてしまうため */}
                  {new Date(s.created_at ?? s.date).toLocaleString("ja-JP")}
                </div>
                {/* アップロード画面でも IN/FAULT を記録するようになったので、件数は常に表示する */}
                <div style={{ fontSize: "0.875rem", color: "var(--muted)", marginTop: "0.25rem" }}>
                  {s.source === "upload" && <span>📁 アップロード &nbsp;</span>}
                  {/* フォームだけ解析のときは本数・IN率に意味がないので出さない */}
                  {s.mode === "form_only" ? (
                    <span>📐 フォーム解析のみ</span>
                  ) : (
                    <span>
                      {s.total_attempts}本 &nbsp; IN {s.in_count} / FAULT {s.fault_count ?? 0} &nbsp; IN率 {accuracy}%
                    </span>
                  )}
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
