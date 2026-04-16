import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "サーブノート — ソフトテニス サーブ練習",
  description: "ソフトテニスのサーブ練習を記録し、録画からAIでフォーム解析できるアプリ",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

import { LayoutWithTabs } from "@/components/layout";
import { SessionVideoProvider } from "@/contexts/SessionVideoContext";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ja">
      <body style={{ margin: 0, minHeight: "100vh", background: "#0f1419", color: "#e6edf3" }}>
        <SessionVideoProvider>
          <LayoutWithTabs>{children}</LayoutWithTabs>
        </SessionVideoProvider>
      </body>
    </html>
  );
}
