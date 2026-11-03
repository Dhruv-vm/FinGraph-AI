"use client";

import { useCallback, useEffect, useState } from "react";

export type NavTab = "overview" | "ask" | "graph" | "evidence" | "evaluation" | "architecture";

export const NAV_TABS: NavTab[] = ["overview", "ask", "graph", "evidence", "evaluation", "architecture"];

export const PAGE_META: Record<NavTab, { title: string; label: string; description: string }> = {
  overview: {
    title: "Overview",
    label: "Overview",
    description: "What FinGraph does, the data it uses and how it performed",
  },
  ask: {
    title: "Ask FinGraph",
    label: "Ask FinGraph",
    description: "Point-in-time questions answered from SEC filings with cited evidence",
  },
  graph: {
    title: "Knowledge graph",
    label: "Knowledge graph",
    description: "Relationships extracted from filings, with provenance for every edge",
  },
  evidence: {
    title: "Evidence explorer",
    label: "Evidence explorer",
    description: "The filing library behind every answer",
  },
  evaluation: {
    title: "Evaluation lab",
    label: "Evaluation lab",
    description: "Historical benchmark results and ablations",
  },
  architecture: {
    title: "Architecture",
    label: "Architecture",
    description: "The temporal GraphRAG pipeline, stage by stage",
  },
};

function readHash(): NavTab {
  if (typeof window === "undefined") return "overview";
  const h = window.location.hash.replace(/^#\/?/, "").split(/[?/]/)[0] as NavTab;
  return NAV_TABS.includes(h) ? h : "overview";
}

/**
 * Keeps the active tab in the URL hash (#/graph) so pages are linkable and
 * browser back/forward works, without changing the single-route app.
 */
export function useHashTab(): [NavTab, (t: NavTab) => void] {
  const [tab, setTab] = useState<NavTab>("overview");

  useEffect(() => {
    setTab(readHash());
    const onHash = () => setTab(readHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const navigate = useCallback((t: NavTab) => {
    if (readHash() !== t || !window.location.hash) {
      window.location.hash = `/${t}`;
    }
    setTab(t);
  }, []);

  return [tab, navigate];
}
