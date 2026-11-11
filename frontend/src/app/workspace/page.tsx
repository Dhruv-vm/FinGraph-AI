"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { Header, HealthState } from "@/components/Header";
import { BrandLogo, NavList } from "@/components/Navigation";
import { ViewErrorBoundary } from "@/components/ErrorBoundary";
import { Drawer, cx } from "@/components/ui";
import dynamic from "next/dynamic";
import { OverviewView } from "@/components/views/OverviewView";

// Every other section is split into its own chunk and downloaded the first time it is opened,
// so the Overview does not pay for the graph (React Flow) or chart (Recharts) libraries.
function ViewLoading() {
  return (
    <div className="max-w-[1320px] mx-auto px-4 sm:px-8 py-8 space-y-4" aria-busy="true" aria-label="Loading section">
      <div className="h-40 rounded-2xl bg-surface-2/70 animate-pulse" />
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="h-28 rounded-2xl bg-surface-2/60 animate-pulse" />
        ))}
      </div>
    </div>
  );
}
const AskFinGraphView = dynamic(() => import("@/components/views/AskFinGraphView").then((m) => m.AskFinGraphView), { ssr: false, loading: ViewLoading });
const GraphExplorerView = dynamic(() => import("@/components/views/GraphExplorerView").then((m) => m.GraphExplorerView), { ssr: false, loading: ViewLoading });
const EvidenceInspectorView = dynamic(() => import("@/components/views/EvidenceInspectorView").then((m) => m.EvidenceInspectorView), { ssr: false, loading: ViewLoading });
const EvaluationLabView = dynamic(() => import("@/components/views/EvaluationLabView").then((m) => m.EvaluationLabView), { ssr: false, loading: ViewLoading });
const ArchitectureView = dynamic(() => import("@/components/views/ArchitectureView").then((m) => m.ArchitectureView), { ssr: false, loading: ViewLoading });

/** Warm the other sections' chunks once the browser is idle, so switching tabs is instant. */
function usePrefetchViews() {
  useEffect(() => {
    const load = () => {
      import("@/components/views/AskFinGraphView");
      import("@/components/views/EvaluationLabView");
      import("@/components/views/ArchitectureView");
      import("@/components/views/EvidenceInspectorView");
      import("@/components/views/GraphExplorerView");
    };
    const w = window as Window & { requestIdleCallback?: (cb: () => void, o?: { timeout: number }) => number };
    const id = w.requestIdleCallback ? w.requestIdleCallback(load, { timeout: 1200 }) : window.setTimeout(load, 800);
    return () => {
      if (!w.requestIdleCallback) window.clearTimeout(id);
    };
  }, []);
}
import { api, ApiError, isAbortError } from "@/lib/api";
import { HealthResponse } from "@/lib/types";
import { NavTab, PAGE_META, useHashTab } from "@/lib/navigation";
import { ResearchContextProvider } from "@/lib/research-context";

const HEALTH_INTERVAL_MS = 30_000;

function useHealth(): HealthState {
  const [data, setData] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(false);
  const [checkedAt, setCheckedAt] = useState<Date | null>(null);
  const ctrlRef = useRef<AbortController | null>(null);

  const refresh = useCallback(async () => {
    ctrlRef.current?.abort();
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    setLoading(true);
    try {
      const h = await api.getHealth(ctrl.signal);
      setData(h);
      setError(null);
    } catch (e) {
      if (isAbortError(e)) return;
      // Keep the last good payload visible but mark the backend as unreachable.
      setError(e instanceof ApiError ? e : new ApiError(String(e), 0));
    } finally {
      if (!ctrl.signal.aborted) {
        setLoading(false);
        setCheckedAt(new Date());
      }
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = window.setInterval(refresh, HEALTH_INTERVAL_MS);
    return () => {
      window.clearInterval(t);
      ctrlRef.current?.abort();
    };
  }, [refresh]);

  return { data, error, loading, checkedAt, refresh };
}

export default function FinGraphWorkspace() {
  const [activeTab, navigate] = useHashTab();
  const health = useHealth();
  usePrefetchViews();
  const [mobileNav, setMobileNav] = useState(false);
  // Views stay mounted after first visit so in-progress work (a running query,
  // graph position, filters) survives navigation.
  const [visited, setVisited] = useState<Set<NavTab>>(() => new Set());

  useEffect(() => {
    setVisited((prev) => (prev.has(activeTab) ? prev : new Set(prev).add(activeTab)));
    document.title = `${PAGE_META[activeTab].title} — FinGraph AI`;
  }, [activeTab]);

  const go = useCallback(
    (t: NavTab) => {
      navigate(t);
      setMobileNav(false);
    },
    [navigate]
  );

  const views: Record<NavTab, React.ReactNode> = {
    overview: <OverviewView health={health} onNavigate={go} />,
    ask: <AskFinGraphView onNavigate={go} />,
    graph: <GraphExplorerView active={activeTab === "graph"} onNavigate={go} />,
    evidence: <EvidenceInspectorView active={activeTab === "evidence"} />,
    evaluation: <EvaluationLabView active={activeTab === "evaluation"} />,
    architecture: <ArchitectureView />,
  };

  return (
    <ResearchContextProvider>
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:bg-surface focus:px-3 focus:py-2 focus:rounded focus:border focus:border-line"
      >
        Skip to content
      </a>
      <div className="h-[100dvh] w-full overflow-hidden lg:p-3">
        <Drawer open={mobileNav} onClose={() => setMobileNav(false)} side="left" title={<BrandLogo />} width="w-[270px]">
          <div className="py-3">
            <NavList activeTab={activeTab} onTabChange={go} />
          </div>
        </Drawer>

        <div className="ws-frame h-full min-w-0 flex flex-col overflow-hidden">
          <Header activeTab={activeTab} health={health} onOpenMobileNav={() => setMobileNav(true)} onNavigate={go} />
          <main id="main" className="flex-1 min-h-0 relative border-t border-line-soft">
            {(Object.keys(views) as NavTab[]).map((id) =>
              visited.has(id) ? (
                <div
                  key={id}
                  // display:none for every inactive view. (visibility:hidden is not enough: React Flow
                  // sets visibility:visible on its nodes, which made the graph bleed over other pages.)
                  hidden={id !== activeTab}
                  className={cx("absolute inset-0 animate-fade-in", id === "graph" ? "overflow-hidden" : "overflow-y-auto scrollbar-thin")}
                >
                  <ViewErrorBoundary name={PAGE_META[id].title}>{views[id]}</ViewErrorBoundary>
                </div>
              ) : null
            )}
          </main>
        </div>
      </div>
    </ResearchContextProvider>
  );
}
