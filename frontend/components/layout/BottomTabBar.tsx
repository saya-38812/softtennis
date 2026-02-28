"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Home, Target, History, Upload, type LucideIcon } from "lucide-react";
import { TAB_CONFIG, type TabId } from "@/lib/constants";

const pathByTab: Record<TabId, string> = {
  home: "/",
  challenge: "/challenge",
  history: "/history",
  upload: "/upload",
};

const tabIcons: Record<TabId, LucideIcon> = {
  home: Home,
  challenge: Target,
  history: History,
  upload: Upload,
};

export function BottomTabBar() {
  const pathname = usePathname();
  const activeTab: TabId =
    pathname === "/challenge"
      ? "challenge"
      : pathname === "/history"
        ? "history"
        : pathname === "/upload" || pathname === "/result"
          ? "upload"
          : "home";

  return (
    <nav className="bottom-tabs" role="tablist" aria-label="メイン">
      <div className="bottom-tabs-inner">
        {TAB_CONFIG.map((tab) => {
          const href = pathByTab[tab.id];
          const isActive = activeTab === tab.id;
          const Icon = tabIcons[tab.id];
          return (
            <Link
              key={tab.id}
              href={href}
              role="tab"
              aria-selected={isActive}
              className={`tab-item ${isActive ? "active" : ""}`}
            >
              <span className="tab-icon" aria-hidden>
                <Icon size={22} className="tab-icon-svg" />
              </span>
              <span className="tab-label">{tab.label}</span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
