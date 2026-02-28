/**
 * アプリ全体で使う定数
 */

export const APP_NAME = "サーブノート";
export const APP_SUBTITLE = "Serve Tracker & Analysis";

export const TAB_IDS = ["home", "challenge", "history", "upload"] as const;
export type TabId = (typeof TAB_IDS)[number];

export const TAB_CONFIG: { id: TabId; label: string }[] = [
  { id: "home", label: "ホーム" },
  { id: "challenge", label: "チャレンジ" },
  { id: "history", label: "履歴" },
  { id: "upload", label: "アップロード" },
];

/** チャレンジ条件は本数＋IN率のみ（サーブ特化・IN/OUTだけ） */
export interface ChallengeDef {
  id: string;
  name: string;
  nameEn: string;
  minServes: number;
  minAccuracy: number;
}

export const CHALLENGES: ChallengeDef[] = [
  { id: "beginner", name: "初級", nameEn: "Beginner", minServes: 20, minAccuracy: 50 },
  { id: "intermediate", name: "中級", nameEn: "Intermediate", minServes: 30, minAccuracy: 65 },
  { id: "advanced", name: "上級", nameEn: "Advanced", minServes: 40, minAccuracy: 75 },
];

/**
 * 将来の指標候補（実装時はバックエンド対応が必要なものあり）
 * - 速度（平均/最大）: 計測保留中
 * - OUT率上限: コントロール目標（OUT ○%以下）
 * - 連続IN数: セッション内の最大連続IN（安定性）
 * - ゾーン狙い: サービスボックス分割で「左上に○本」など
 * - ファーストサーブ率: 1本目IN率（1打目/2打目の区別が必要）
 */

/** 日付に応じて今日のチャレンジのインデックスを返す（0〜2）。同じ日は同じ値。 */
export function getDailyChallengeIndex(): number {
  const today = new Date();
  const dayKey = Math.floor(
    new Date(today.getFullYear(), today.getMonth(), today.getDate()).getTime() / 86400000
  );
  return dayKey % CHALLENGES.length;
}

/** 今日のチャレンジ1つを返す（日替わりでローテーション）。 */
export function getDailyChallenge(): ChallengeDef {
  return CHALLENGES[getDailyChallengeIndex()];
}
