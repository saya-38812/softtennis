import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Soft Tennis Serve Analyzer",
  description: "Upload a serve video and get AI-based feedback on your form.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

import { LayoutWithTabs } from "@/components/layout";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ja">
      <body style={{ margin: 0, minHeight: "100vh", background: "#0f1419", color: "#e6edf3" }}>
        <LayoutWithTabs>{children}</LayoutWithTabs>
      </body>
    </html>
  );
}
