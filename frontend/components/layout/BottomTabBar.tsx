"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Home, Upload, BarChart2, History, type LucideIcon } from "lucide-react";
import { TAB_CONFIG, type TabId } from "@/lib/constants";

const pathByTab: Record<TabId, string> = {
  home: "/",
  upload: "/upload",
  stats: "/stats",
  history: "/history",
};

const tabIcons: Record<TabId, LucideIcon> = {
  home: Home,
  upload: Upload,
  stats: BarChart2,
  history: History,
};

function getActiveTab(pathname: string | null): TabId {
  if (pathname === "/history") return "history";
  if (pathname === "/stats") return "stats";
  if (pathname === "/upload") return "upload";
  return "home";
}

export function BottomTabBar() {
  const pathname = usePathname();
  const activeTab = getActiveTab(pathname);

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
