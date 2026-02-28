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
  const outCount = session.out_count ?? 0;
  const faultCount = session.fault_count ?? 0;
  const accuracy = accuracyPercent(session);

  return (
    <div className={`card ${className ?? ""}`} style={{ padding: "0.75rem" }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", fontSize: "0.85rem" }}>
        <div>本数: <strong>{total}</strong></div>
        <div>IN率: <strong>{accuracy}%</strong></div>
        <div style={{ gridColumn: "1 / -1" }}>IN <strong>{inCount}</strong> / OUT <strong>{outCount}</strong> / FAULT <strong>{faultCount}</strong></div>
      </div>
    </div>
  );
}
