import type { ServeAnalysis } from "./api";

export interface UploadResultData {
  overallScore: number;
  improvementMessage: string;
  focusLabel: string;
  focusAdvice: string;
  technical: { label: string; value: number; key: string }[];
  aiText: string;
  practice: string[];
}

export function fromAnalysis(a: ServeAnalysis): UploadResultData {
  const scores = a.scores ?? {};
  const clamp = (v: number) => Math.min(98, Math.max(5, Math.round(v)));
  const e = scores.elbow_angle ?? 0;
  const elbow = clamp(e >= 90 && e <= 130 ? 70 + (e - 90) * 0.5 : Math.max(30, 100 - Math.abs(e - 110)));
  const sway = clamp(100 - Math.min(100, (scores.body_sway ?? 0) * 250));
  const ih = scores.impact_height ?? 0;
  const toss = clamp(Math.min(100, Math.max(0, (ih - 0.3) * 60)));
  const technical = [
    { key: "impact_height", label: "トス高さ", value: toss },
    { key: "elbow_angle", label: "肘の余裕", value: elbow },
    { key: "body_sway", label: "体の安定", value: sway },
    { key: "waist_speed", label: "腰のキレ", value: 70 },
    { key: "weight_transfer", label: "体重移動", value: 70 },
  ];
  const overallScore = Math.round(
    technical.reduce((acc, t) => acc + t.value, 0) / technical.length
  );
  return {
    overallScore,
    improvementMessage: "",
    focusLabel: a.focus_label ?? "体軸のブレ",
    focusAdvice: a.ai_text ?? "スイング中に体を安定させましょう。",
    technical,
    aiText: a.ai_text ?? "",
    practice: a.practice ? [a.practice] : [],
  };
}
