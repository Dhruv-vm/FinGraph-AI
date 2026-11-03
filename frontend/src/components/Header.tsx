"use client";

import React, { useState } from "react";
import { Crosshair, Menu, Moon, RefreshCw, Sun, X } from "lucide-react";
import { HealthResponse, ComponentHealth } from "@/lib/types";
import { useTheme } from "@/lib/theme";
import { NavTab, PAGE_META } from "@/lib/navigation";
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
        className="flex items-center gap-2 h-8 px-2.5 rounded text-xs text-fg-2 hover:text-fg hover:bg-surface-2 transition-colors"
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
          "flex items-center h-8 rounded border text-xs min-w-0",
          entity ? "border-accent/35 bg-accent/[0.07]" : "border-line border-dashed"
        )}
      >
        <button
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          aria-haspopup="dialog"
          className="flex items-center gap-1.5 h-full pl-2.5 pr-2 min-w-0 text-fg-2 hover:text-fg"
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
            className="h-full px-1.5 text-fg-3 hover:text-fg border-l border-accent/25"
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
                    "w-full text-left px-2 py-1.5 rounded text-sm flex items-baseline justify-between gap-2",
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
  return (
    <header className="h-14 shrink-0 flex items-center gap-3 px-3 sm:px-5 border-b border-line-soft bg-canvas/95 backdrop-blur-sm">
      <IconButton label="Open navigation" className="lg:hidden" onClick={onOpenMobileNav}>
        <Menu className="w-4 h-4" />
      </IconButton>
      <div className="min-w-0 flex-1">
        <h1 className="text-md font-semibold leading-tight truncate">{meta.title}</h1>
        <p className="text-xs text-fg-3 truncate hidden sm:block">{meta.description}</p>
      </div>
      <div className="flex items-center gap-1 sm:gap-2 shrink-0">
        <ContextControl onNavigate={onNavigate} />
        <span aria-hidden className="w-px h-5 bg-line-soft mx-1 hidden sm:block" />
        <StatusControl health={health} />
        <IconButton label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"} onClick={toggleTheme}>
          {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </IconButton>
      </div>
    </header>
  );
}
