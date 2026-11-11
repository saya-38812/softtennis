"use client";

/**
 * お手本比較の解析結果を表示する部品
 *
 * 表示する内容：
 *   1. まとめ：平均点・一番多かった改善ポイント・IN と FAULT の平均点
 *   2. サーブの切り替えボタン（#1 IN 85点 …）
 *   3. 選んだサーブの詳細：総合点・改善ポイントとアドバイス・5指標の点数バー
 *
 * 点数は Colab 側で「お手本 = 90点」として計算済みなので、ここでは計算し直さずにそのまま表示する。
 * （旧方式の lib/uploadResult.ts の fromAnalysis は使わない）
 */

import { useState } from "react";
import type { MetricKey, ModelAnalysisResult, ModelServeResult } from "@/lib/api";

/** 指標の表示名と並び順（Python 側の FOCUS_LABELS と同じ名前） */
const METRICS: { key: MetricKey; label: string }[] = [
  { key: "impact_height", label: "打点の高さ" },
  { key: "elbow_angle", label: "肘の角度" },
  { key: "body_sway", label: "体軸のブレ" },
  { key: "waist_speed", label: "腰の回転速度" },
  { key: "weight_transfer", label: "体重移動" },
];

/** お手本の点数（この線を超えていれば、お手本と同じくらいできている） */
const MODEL_SCORE = 90;

/** 点数に応じた色：お手本並み → 緑、まずまず → 黄、要改善 → 赤 */
function scoreColor(score: number): string {
  if (score >= 80) return "var(--success)";
  if (score >= 50) return "var(--warning)";
  return "var(--error)";
}

/** IN / FAULT の小さなバッジ */
function ResultBadge({ result }: { result: string | null }) {
  if (!result) return null;
  const isIn = result === "IN";
  return (
    <span
      style={{
        fontSize: "0.7rem",
        fontWeight: 700,
        padding: "0.1rem 0.4rem",
        borderRadius: 4,
        background: isIn ? "rgba(63,185,80,0.2)" : "rgba(248,81,73,0.2)",
        color: isIn ? "var(--success)" : "var(--error)",
      }}
    >
      {result}
    </span>
  );
}

// ============================================================
// 1. まとめ
// ============================================================
function SummaryCard({ result }: { result: ModelAnalysisResult }) {
  const s = result.summary;
  if (s.analyzed_count === 0) {
    return (
      <div className="card" style={{ padding: "1rem", marginBottom: "1rem" }}>
        <p style={{ color: "var(--error)", margin: 0 }}>
          解析できたサーブがありませんでした。全身が横から映るように撮影して、もう一度お試しください。
        </p>
      </div>
    );
  }
  return (
    <div className="card" style={{ padding: "1rem", marginBottom: "1rem" }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: "0.5rem", marginBottom: "0.5rem" }}>
        <span style={{ fontSize: "0.75rem", color: "var(--muted)" }}>
          {result.mode === "form_only" ? "フォームの点数" : "平均点"}
        </span>
        <span style={{ fontSize: "2rem", fontWeight: 700, color: scoreColor(s.average_score ?? 0) }}>
          {s.average_score}
        </span>
        <span style={{ fontSize: "0.8rem", color: "var(--muted)" }}>/ 100（お手本 = {MODEL_SCORE}）</span>
      </div>

      {s.top_focus_label && (
        <p style={{ margin: "0 0 0.25rem 0", fontSize: "0.9rem" }}>
          今日の課題：<strong>{s.top_focus_label}</strong>
          {result.mode !== "form_only" && (
            <span className="text-muted" style={{ fontSize: "0.8rem" }}>
              （{s.analyzed_count}本中 {s.top_focus_count}本）
            </span>
          )}
        </p>
      )}

      {/* IN と FAULT の平均点の比較（両方あるときだけ） */}
      {s.average_score_in !== undefined && s.average_score_fault !== undefined && (
        <p style={{ margin: 0, fontSize: "0.85rem", color: "var(--muted)" }}>
          IN のときの平均 <strong style={{ color: "var(--success)" }}>{s.average_score_in}</strong>点 ／ FAULT のときの平均{" "}
          <strong style={{ color: "var(--error)" }}>{s.average_score_fault}</strong>点
        </p>
      )}

      {s.error_count > 0 && (
        <p style={{ margin: "0.25rem 0 0 0", fontSize: "0.8rem", color: "var(--warning)" }}>
          ※ {s.error_count}本は解析できませんでした
        </p>
      )}
    </div>
  );
}

// ============================================================
// 3. サーブ1本の詳細
// ============================================================
function ServeDetail({ serve }: { serve: ModelServeResult }) {
  if (serve.status !== "ok" || !serve.scores) {
    return (
      <div className="card" style={{ padding: "1rem", marginBottom: "1rem" }}>
        <p style={{ color: "var(--error)", margin: 0, fontSize: "0.9rem" }}>
          このサーブは解析できませんでした。{serve.message}
        </p>
      </div>
    );
  }
  const scores = serve.scores;
  return (
    <>
      {/* 改善ポイントとアドバイス */}
      <div className="card" style={{ padding: "1rem", marginBottom: "1rem", border: "1px solid var(--accent)" }}>
        <div style={{ fontSize: "0.7rem", color: "var(--muted)", letterSpacing: "0.05em", marginBottom: "0.35rem" }}>
          改善ポイント（総合 {serve.overall_score}点）
        </div>
        <p style={{ fontSize: "1.1rem", fontWeight: 700, margin: "0 0 0.4rem 0" }}>{serve.focus_label}</p>
        {serve.ai_text && <p style={{ margin: "0 0 0.5rem 0", lineHeight: 1.6 }}>{serve.ai_text}</p>}
        {serve.cue && (
          <p style={{ margin: "0 0 0.5rem 0", fontSize: "0.875rem", color: "var(--accent)" }}>💡 {serve.cue}</p>
        )}
        {serve.practice && (
          <p style={{ margin: 0, fontSize: "0.875rem", color: "var(--muted)" }}>🏋️ おすすめ練習：{serve.practice}</p>
        )}
      </div>

      {/* 5指標の点数バー（縦線がお手本の位置） */}
      <div style={{ marginBottom: "1rem" }}>
        <p style={{ fontSize: "0.7rem", color: "var(--muted)", letterSpacing: "0.08em", marginBottom: "0.75rem" }}>
          お手本との比較（縦線 = お手本の {MODEL_SCORE}点）
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
          {METRICS.map(({ key, label }) => {
            const value = Math.round(scores[key] ?? 0);
            const isFocus = key === serve.focus_key;
            return (
              <div key={key} style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                <span style={{ fontSize: "0.85rem", minWidth: 96, fontWeight: isFocus ? 700 : 400 }}>
                  {isFocus ? "▶ " : ""}
                  {label}
                </span>
                <div style={{ flex: 1, height: 8, background: "var(--bg)", borderRadius: 4, position: "relative" }}>
                  <div
                    style={{
                      width: `${Math.min(value, 100)}%`,
                      height: "100%",
                      background: scoreColor(value),
                      borderRadius: 4,
                    }}
                  />
                  {/* お手本の位置を示す縦線 */}
                  <div
                    style={{
                      position: "absolute",
                      left: `${MODEL_SCORE}%`,
                      top: -3,
                      bottom: -3,
                      width: 2,
                      background: "var(--text)",
                      opacity: 0.6,
                    }}
                  />
                </div>
                <span style={{ fontSize: "0.85rem", fontWeight: 600, color: scoreColor(value), minWidth: 28, textAlign: "right" }}>
                  {value}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </>
  );
}

// ============================================================
// 全体
// ============================================================
export function ModelAnalysisView({ result }: { result: ModelAnalysisResult }) {
  // 最初は「解析できた最初のサーブ」を選んでおく
  const firstOk = result.serves.find((s) => s.status === "ok") ?? result.serves[0];
  const [selectedIndex, setSelectedIndex] = useState<number>(firstOk?.index ?? 0);
  const selected = result.serves.find((s) => s.index === selectedIndex) ?? firstOk;

  return (
    <>
      <SummaryCard result={result} />

      {/* 2. サーブの切り替え（2本以上あるときだけ） */}
      {result.serves.length > 1 && (
        <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap", marginBottom: "1rem" }}>
          {result.serves.map((s) => {
            const active = s.index === selected?.index;
            return (
              <button
                key={s.index}
                type="button"
                onClick={() => setSelectedIndex(s.index)}
                style={{
                  padding: "0.35rem 0.6rem",
                  borderRadius: 8,
                  border: active ? "2px solid var(--accent)" : "2px solid transparent",
                  background: active ? "rgba(88,166,255,0.1)" : "var(--surface)",
                  color: "var(--text)",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: "0.35rem",
                  fontSize: "0.8rem",
                }}
              >
                <span>#{s.index + 1}</span>
                <ResultBadge result={s.result} />
                <span style={{ color: s.status === "ok" ? scoreColor(s.overall_score ?? 0) : "var(--muted)" }}>
                  {s.status === "ok" ? `${s.overall_score}点` : "—"}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {selected && <ServeDetail serve={selected} />}
    </>
  );
}
