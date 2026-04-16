"use client";

import type { TrackerSession } from "@/lib/api";
import { accuracyPercent } from "@/lib/utils";

interface TrackerStatsProps {
  session: TrackerSession | null;
  className?: string;
}

export function TrackerStats({ session, className }: TrackerStatsProps) {
  if (!session) return null;
  const total = session.total_attempts ?? 0;
  const inCount = session.in_count ?? 0;
  const faultCount = session.fault_count ?? 0;
  const accuracy = accuracyPercent(session);

  return (
    <div className={`card ${className ?? ""}`} style={{ padding: "0.75rem" }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: "0.5rem",
          fontSize: "0.85rem",
          textAlign: "center",
        }}
      >
        <div>
          <div style={{ color: "var(--muted)", fontSize: "0.7rem" }}>Serves</div>
          <strong>{total}</strong>
        </div>
        <div>
          <div style={{ color: "var(--muted)", fontSize: "0.7rem" }}>IN</div>
          <strong style={{ color: "var(--success)" }}>{inCount}</strong>
        </div>
        <div>
          <div style={{ color: "var(--muted)", fontSize: "0.7rem" }}>FAULT</div>
          <strong style={{ color: "var(--error)" }}>{faultCount}</strong>
        </div>
        <div>
          <div style={{ color: "var(--muted)", fontSize: "0.7rem" }}>IN率</div>
          <strong>{accuracy}%</strong>
        </div>
      </div>
    </div>
  );
}
