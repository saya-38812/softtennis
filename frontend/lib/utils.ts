/**
 * セッション統計・表示用ユーティリティ
 */

import type { TrackerSession } from "./api";

export function accuracyPercent(session: TrackerSession | null): number {
  if (!session) return 0;
  const total = session.total_attempts ?? 0;
  if (total === 0) return 0;
  return Math.round((session.in_count / total) * 100);
}

export function faultRatePercent(session: TrackerSession | null): number {
  if (!session) return 0;
  const total = session.total_attempts ?? 0;
  if (total === 0) return 0;
  return Math.round(((session.fault_count ?? 0) / total) * 100);
}

export function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}

export function clampScore(value: number): number {
  return Math.min(98, Math.max(5, Math.round(Number(value) || 0)));
}
