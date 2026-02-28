"use client";

import { CircleDot } from "lucide-react";
import { APP_NAME } from "@/lib/constants";

interface AppHeaderProps {
  title?: string;
  subtitle?: string;
  right?: React.ReactNode;
}

export function AppHeader({ title = APP_NAME, subtitle, right }: AppHeaderProps) {
  return (
    <header className="app-header">
      <div className="header-left">
        <div className="header-title-row">
          <CircleDot size={24} className="header-icon" aria-hidden />
          <h1 className="app-title">{title}</h1>
        </div>
        {subtitle && <span className="app-subtitle">{subtitle}</span>}
      </div>
      {right && <div className="header-right">{right}</div>}
    </header>
  );
}
