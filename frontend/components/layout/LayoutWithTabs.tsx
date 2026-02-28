"use client";

import { usePathname } from "next/navigation";
import { BottomTabBar } from "./BottomTabBar";
import { AppHeader } from "./AppHeader";
import { APP_NAME, APP_SUBTITLE } from "@/lib/constants";

const TAB_BAR_HEIGHT = "72px";

const HEADER_TITLES: Record<string, { title: string; subtitle?: string }> = {
  "/": { title: APP_NAME, subtitle: APP_SUBTITLE },
  "/challenge": { title: "チャレンジ", subtitle: "今日のチャレンジ" },
  "/history": { title: "履歴", subtitle: "Past practices" },
  "/upload": { title: "アップロード", subtitle: "動画をアップロード" },
  "/result": { title: "結果", subtitle: "解析結果" },
  "/session-result": { title: "セッション結果", subtitle: "" },
  "/practice": { title: "練習", subtitle: "サーブ記録" },
};

function getHeaderForPath(pathname: string | null) {
  if (!pathname) return { title: APP_NAME, subtitle: APP_SUBTITLE };
  return HEADER_TITLES[pathname] ?? { title: APP_NAME, subtitle: APP_SUBTITLE };
}

export function LayoutWithTabs({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { title, subtitle } = getHeaderForPath(pathname);

  return (
    <div
      className="layout-with-tabs"
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        paddingBottom: `calc(${TAB_BAR_HEIGHT} + env(safe-area-inset-bottom, 0px))`,
      }}
    >
      <header className="layout-header">
        <div className="layout-header-inner">
          <AppHeader title={title} subtitle={subtitle || undefined} />
        </div>
      </header>
      <div className="layout-content">
        {children}
      </div>
      <BottomTabBar />
    </div>
  );
}
