/**
 * セッション統計・チャレンジ判定・表示用ユーティリティ
 */

import type { TrackerSession } from "./api";
import type { ChallengeDef } from "./constants";

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

export interface ChallengeCheckResult {
  clear: boolean;
  servesOk: boolean;
  accuracyOk: boolean;
}

export function checkChallengeClear(
  session: TrackerSession | null,
  challenge: ChallengeDef | null
): ChallengeCheckResult | null {
  if (!challenge || !session) return null;
  const total = session.total_attempts ?? 0;
  const acc = accuracyPercent(session);
  const servesOk = total >= challenge.minServes;
  const accuracyOk = acc >= challenge.minAccuracy;
  return {
    clear: servesOk && accuracyOk,
    servesOk,
    accuracyOk,
  };
}

export function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}

export function clampScore(value: number): number {
  return Math.min(98, Math.max(5, Math.round(Number(value) || 0)));
}
