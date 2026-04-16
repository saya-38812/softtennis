/**
 * アプリ全体で使う定数
 */

export const APP_NAME = "サーブノート";
export const APP_SUBTITLE = "サーブ練習 & フォーム解析";

export const TAB_IDS = ["home", "upload", "stats", "history"] as const;
export type TabId = (typeof TAB_IDS)[number];

export const TAB_CONFIG: { id: TabId; label: string }[] = [
  { id: "home", label: "ホーム" },
  { id: "upload", label: "アップロード" },
  { id: "stats", label: "統計" },
  { id: "history", label: "履歴" },
];
