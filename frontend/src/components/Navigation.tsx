"use client";

import React from "react";
import Image from "next/image";
import {
  FileText,
  FlaskConical,
  Gauge,
  MessageSquareText,
  PanelLeftClose,
  PanelLeftOpen,
  Waypoints,
  Workflow,
} from "lucide-react";
import { NavTab, NAV_TABS, PAGE_META } from "@/lib/navigation";
import { useTheme } from "@/lib/theme";
import { cx } from "./ui";

export type { NavTab } from "@/lib/navigation";

const ICONS: Record<NavTab, React.ComponentType<{ className?: string }>> = {
  overview: Gauge,
  ask: MessageSquareText,
  graph: Waypoints,
  evidence: FileText,
  evaluation: FlaskConical,
  architecture: Workflow,
};

/** Official logo, rendered at fixed height with intrinsic 3:1 aspect ratio. */
export function BrandLogo({ collapsed }: { collapsed?: boolean }) {
  const { theme } = useTheme();
  const src = theme === "dark" ? "/brand/fingraph-logo-dark.png" : "/brand/fingraph-logo-light.png";
  if (collapsed) {
    // Show only the mark by windowing the unmodified artwork (no redraw, no distortion).
    return (
      <span className="block w-[26px] h-[33px] overflow-hidden relative" aria-label="FinGraph AI">
        <Image src={src} alt="FinGraph AI" width={99} height={33} priority className="absolute -left-[6px] top-0 max-w-none h-[33px] w-[99px]" />
      </span>
    );
  }
  return <Image src={src} alt="FinGraph AI" width={108} height={36} priority className="h-9 w-[108px]" />;
}

export function NavList({
  activeTab,
  onTabChange,
  collapsed,
}: {
  activeTab: NavTab;
  onTabChange: (t: NavTab) => void;
  collapsed?: boolean;
}) {
  return (
    <nav aria-label="Primary" className="px-2 space-y-0.5">
      {NAV_TABS.map((id) => {
        const Icon = ICONS[id];
        const active = activeTab === id;
        return (
          <button
            key={id}
            onClick={() => onTabChange(id)}
            aria-current={active ? "page" : undefined}
            title={collapsed ? PAGE_META[id].label : undefined}
            className={cx(
              "group relative w-full flex items-center gap-2.5 h-8 rounded text-sm transition-colors duration-150",
              collapsed ? "justify-center px-0" : "px-2.5",
              active ? "bg-surface-2 text-fg font-medium" : "text-fg-2 hover:text-fg hover:bg-surface-2/60"
            )}
          >
            {active && <span aria-hidden className="absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-full bg-accent" />}
            <Icon className={cx("w-4 h-4 shrink-0", active ? "text-accent" : "text-fg-3 group-hover:text-fg-2")} />
            {!collapsed && <span className="truncate">{PAGE_META[id].label}</span>}
          </button>
        );
      })}
    </nav>
  );
}

export function Navigation({
  activeTab,
  onTabChange,
  collapsed,
  onToggleCollapsed,
}: {
  activeTab: NavTab;
  onTabChange: (t: NavTab) => void;
  collapsed: boolean;
  onToggleCollapsed: () => void;
}) {
  return (
    <aside
      className={cx(
        "hidden lg:flex flex-col shrink-0 bg-sidebar border-r border-line-soft transition-[width] duration-200 ease-out",
        collapsed ? "w-[60px]" : "w-[224px]"
      )}
    >
      <div className={cx("h-14 flex items-center shrink-0", collapsed ? "justify-center" : "px-4")}>
        <BrandLogo collapsed={collapsed} />
      </div>
      <div className="pt-2 flex-1 overflow-y-auto scrollbar-thin">
        <NavList activeTab={activeTab} onTabChange={onTabChange} collapsed={collapsed} />
      </div>
      <div className={cx("p-2 border-t border-line-soft", collapsed && "flex justify-center")}>
        <button
          onClick={onToggleCollapsed}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          className={cx(
            "flex items-center gap-2.5 h-8 rounded text-sm text-fg-3 hover:text-fg hover:bg-surface-2/60 transition-colors",
            collapsed ? "w-8 justify-center" : "w-full px-2.5"
          )}
        >
          {collapsed ? <PanelLeftOpen className="w-4 h-4" /> : <PanelLeftClose className="w-4 h-4" />}
          {!collapsed && <span>Collapse</span>}
        </button>
      </div>
    </aside>
  );
}
