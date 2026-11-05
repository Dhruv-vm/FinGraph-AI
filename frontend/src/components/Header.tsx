"use client";

import React, { useState } from "react";
import Link from "next/link";
import { ArrowLeft, Crosshair, Home, Menu, Moon, RefreshCw, Sun, X } from "lucide-react";
import { HealthResponse, ComponentHealth } from "@/lib/types";
import { useTheme } from "@/lib/theme";
import { NavTab, NAV_TABS, PAGE_META } from "@/lib/navigation";
import { BrandLogo, ICONS } from "./Navigation";
import { useResearchContext, issuerEntity } from "@/lib/research-context";
import { CORPUS_ISSUERS, fmtDateTime, fmtInt } from "@/lib/format";
import { ApiError } from "@/lib/api";
import { IconButton, Popover, StatusDot, Status, STATUS_LABEL, cx } from "./ui";

export interface HealthState {
  data: HealthResponse | null;
  error: ApiError | null;
  loading: boolean;
  checkedAt: Date | null;
  refresh: () => void;
}

export function overallStatus(h: HealthState): Status {
  if (h.error) return "unavailable";
  if (!h.data) return "unknown";
  return (h.data.status as Status) ?? "unknown";
}

function componentRows(h: HealthResponse): { name: string; c: ComponentHealth; value: string }[] {
  return [
    { name: "Pipeline", c: h.pipeline, value: `${h.pipeline.details?.hop_mode ?? "—"} hops` },
    { name: "Qdrant", c: h.qdrant, value: `${fmtInt(h.qdrant.details?.points_count)} vectors` },
    {
      name: "Knowledge graph",
      c: h.knowledge_graph,
      value: `${fmtInt(h.knowledge_graph.details?.node_count)} nodes`,
    },
    { name: "Ollama", c: h.llm, value: String(h.llm.details?.model ?? "—") },
  ];
}

function StatusControl({ health }: { health: HealthState }) {
  const [open, setOpen] = useState(false);
  const status = overallStatus(health);
  const d = health.data;
  const down = d
    ? ([
        ["Pipeline", d.pipeline],
        ["Qdrant", d.qdrant],
        ["Knowledge graph", d.knowledge_graph],
        ["Language model", d.llm],
      ] as const).filter(([, c]) => c.status !== "healthy")
    : [];
  const label = health.error
    ? "Backend offline"
    : status === "healthy"
    ? "Operational"
    : down.length === 1
    ? `${down[0][0]} ${down[0][1].status === "unavailable" ? "offline" : "degraded"}`
    : STATUS_LABEL[status];
  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="dialog"
        className="flex items-center gap-2 h-9 px-3.5 rounded-full border border-line-soft bg-surface-2/70 text-xs text-fg-2 hover:text-fg hover:border-line transition-colors"
      >
        <StatusDot status={status} />
        <span className="hidden sm:inline">{label}</span>
      </button>
      <Popover open={open} onClose={() => setOpen(false)} className="w-[300px]">
        <div className="px-3.5 pt-3 pb-2 flex items-center justify-between">
          <div className="text-sm font-semibold">Backend status</div>
          <IconButton label="Re-check status" size="sm" onClick={health.refresh} disabled={health.loading}>
            <RefreshCw className={cx("w-3.5 h-3.5", health.loading && "animate-spin")} />
          </IconButton>
        </div>
        {health.error ? (
          <div className="px-3.5 pb-3 text-sm text-fg-2">
            <div className="flex items-center gap-2 text-fg">
              <StatusDot status="unavailable" /> FastAPI unreachable
            </div>
            <p className="text-xs text-fg-3 mt-1.5">{health.error.message}</p>
          </div>
        ) : health.data ? (
          <ul className="px-1.5 pb-1.5">
            <li className="flex items-center justify-between gap-3 px-2 py-1.5 text-sm">
              <span className="flex items-center gap-2">
                <StatusDot status="healthy" /> FastAPI
              </span>
              <span className="font-mono-num text-xs text-fg-3">v{health.data.version}</span>
            </li>
            {componentRows(health.data).map((r) => (
              <li key={r.name} className="flex items-center justify-between gap-3 px-2 py-1.5 text-sm" title={r.c.message ?? undefined}>
                <span className="flex items-center gap-2">
                  <StatusDot status={r.c.status as Status} /> {r.name}
                </span>
                <span className="font-mono-num text-xs text-fg-3 truncate">{r.value}</span>
              </li>
            ))}
          </ul>
        ) : (
          <div className="px-3.5 pb-3 text-sm text-fg-3">Checking…</div>
        )}
        <div className="px-3.5 py-2 border-t border-line-soft text-2xs text-fg-3">
          {health.checkedAt ? `Checked ${fmtDateTime(health.checkedAt.toISOString())}` : "Not checked yet"}. Refreshes every 30 seconds.
        </div>
      </Popover>
    </div>
  );
}

function ContextControl({ onNavigate }: { onNavigate: (t: NavTab) => void }) {
  const { entity, setEntity, clear } = useResearchContext();
  const [open, setOpen] = useState(false);
  return (
    <div className="relative min-w-0">
      <div
        className={cx(
          "flex items-center h-9 rounded-full border text-xs min-w-0",
          entity ? "border-accent/40 bg-accent/[0.1]" : "border-line border-dashed"
        )}
      >
        <button
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          aria-haspopup="dialog"
          className="flex items-center gap-1.5 h-full pl-3.5 pr-3 min-w-0 text-fg-2 hover:text-fg"
          title="Research context: the entity you are currently studying"
        >
          <Crosshair className={cx("w-3.5 h-3.5 shrink-0", entity ? "text-accent" : "text-fg-3")} />
          {entity ? (
            <span className="truncate text-fg font-medium max-w-[160px]">
              {entity.name}
              {entity.ticker && <span className="font-mono-num text-fg-3 font-normal ml-1.5">{entity.ticker}</span>}
            </span>
          ) : (
            <span className="hidden md:inline">Set research context</span>
          )}
        </button>
        {entity && (
          <button
            onClick={clear}
            aria-label="Clear research context"
            title="Clear research context"
            className="h-full pl-2 pr-3 text-fg-3 hover:text-fg border-l border-accent/25"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
      <Popover open={open} onClose={() => setOpen(false)} anchorClassName="right-0 md:left-0 md:right-auto top-full mt-2" className="w-[280px]">
        <div className="px-3.5 pt-3 pb-1">
          <div className="text-sm font-semibold">Research context</div>
          <p className="text-xs text-fg-3 mt-0.5">
            Opens in the knowledge graph, filters the evidence library and is offered as an optional scope in Ask.
          </p>
        </div>
        <ul className="p-1.5 grid grid-cols-2 gap-0.5">
          {CORPUS_ISSUERS.map((c) => {
            const on = entity?.nodeId === `company:${c.ticker}`;
            return (
              <li key={c.ticker}>
                <button
                  onClick={() => {
                    setEntity(issuerEntity(c.ticker, c.name));
                    setOpen(false);
                  }}
                  className={cx(
                    "w-full text-left px-3 py-2 rounded-xl text-sm flex items-baseline justify-between gap-2",
                    on ? "bg-accent/10 text-fg" : "hover:bg-surface-2 text-fg-2 hover:text-fg"
                  )}
                >
                  <span className="truncate">{c.name.replace(/,? Inc\.|Corporation|, Inc\./g, "").trim()}</span>
                  <span className="font-mono-num text-2xs text-fg-3">{c.ticker}</span>
                </button>
              </li>
            );
          })}
        </ul>
        <div className="px-3.5 py-2 border-t border-line-soft flex items-center justify-between text-xs">
          <button
            className="text-accent hover:underline"
            onClick={() => {
              setOpen(false);
              onNavigate("graph");
            }}
          >
            Find another entity in the graph
          </button>
          {entity && (
            <button className="text-fg-3 hover:text-fg" onClick={() => { clear(); setOpen(false); }}>
              Clear
            </button>
          )}
        </div>
      </Popover>
    </div>
  );
}

/** Pill navigation across the six workspace sections (the reference's top tab bar). */
function PillNav({ activeTab, onNavigate }: { activeTab: NavTab; onNavigate: (t: NavTab) => void }) {
  return (
    <nav aria-label="Primary" className="hidden md:block">
      <div className="pill-track">
      {NAV_TABS.map((id) => {
        const Icon = ICONS[id];
        const on = id === activeTab;
        return (
          <button
            key={id}
            onClick={() => onNavigate(id)}
            aria-current={on ? "page" : undefined}
            title={PAGE_META[id].label}
            className={cx(
              "group inline-flex items-center gap-2 h-9 rounded-full text-[13px] font-medium transition-all duration-200 whitespace-nowrap",
              on ? "pill-on px-4" : "px-3 text-fg-2 hover:text-fg hover:bg-surface-3/60"
            )}
          >
            <Icon className={cx("w-4 h-4 shrink-0", on ? "" : "text-fg-3 group-hover:text-fg-2")} />
            <span className={on ? "" : "hidden xl:inline"}>{PAGE_META[id].label}</span>
          </button>
        );
      })}
      </div>
    </nav>
  );
}

export function Header({
  activeTab,
  health,
  onOpenMobileNav,
  onNavigate,
}: {
  activeTab: NavTab;
  health: HealthState;
  onOpenMobileNav: () => void;
  onNavigate: (t: NavTab) => void;
}) {
  const { theme, toggleTheme } = useTheme();
  const meta = PAGE_META[activeTab];
  const compactTitle = activeTab === "graph";
  return (
    <header className="shrink-0">
      <div className="h-16 flex items-center gap-2 sm:gap-3 px-3 sm:px-5">
        <IconButton label="Open navigation" className="md:hidden" onClick={onOpenMobileNav}>
          <Menu className="w-4 h-4" />
        </IconButton>
        <Link href="/" aria-label="FinGraph AI home" title="FinGraph AI home" className="shrink-0 rounded-md">
          <BrandLogo />
        </Link>
        <div className="flex-1 min-w-0 flex justify-center">
          <PillNav activeTab={activeTab} onNavigate={onNavigate} />
        </div>
        <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
          <ContextControl onNavigate={onNavigate} />
          <StatusControl health={health} />
          <IconButton
            label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            onClick={toggleTheme}
            className="border border-line-soft"
          >
            {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </IconButton>
          <Link
            href="/"
            aria-label="Back to the FinGraph homepage"
            title="Homepage"
            className="hidden lg:inline-flex w-9 h-9 items-center justify-center rounded-full border border-line-soft text-fg-2 hover:text-fg hover:bg-surface-2 transition-colors"
          >
            <Home className="w-4 h-4" />
          </Link>
        </div>
      </div>
      <div className={cx("flex items-center gap-3 sm:gap-4 px-4 sm:px-8", compactTitle ? "pb-3" : "pt-2 pb-4 sm:pt-3 sm:pb-5")}>
        <Link
          href="/"
          aria-label="Back to the FinGraph homepage"
          title="Homepage"
          className="hidden sm:inline-flex w-10 h-10 shrink-0 items-center justify-center rounded-full border border-line text-fg-2 hover:text-fg hover:bg-surface-2 hover:-translate-x-0.5 transition-all"
        >
          <ArrowLeft className="w-4 h-4" />
        </Link>
        <div className="min-w-0">
          <h1 className={cx("font-medium tracking-[-0.035em] leading-none truncate", compactTitle ? "text-[26px]" : "text-[30px] sm:text-[40px]")}>
            {meta.title}
          </h1>
          <p className="text-sm text-fg-3 truncate mt-2 hidden sm:block">{meta.description}</p>
        </div>
      </div>
    </header>
  );
}
