"use client";

/**
 * Shared research context: the entity an analyst is currently studying.
 *
 * It is a visible, user-controlled scope. Views may use it as a default
 * (Knowledge Graph focal entity, Evidence issuer filter, Ask suggestions) but
 * must never silently rewrite a user's query with it.
 */

import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

export interface ResearchEntity {
  /** Knowledge-graph node id, e.g. "company:NVDA" */
  nodeId: string;
  /** Display name, e.g. "NVIDIA Corporation" */
  name: string;
  /** Corpus issuer ticker when the entity is one of the 10 corpus issuers */
  ticker?: string | null;
  /** KG node type, e.g. "Company" */
  type?: string | null;
}

interface ResearchContextValue {
  entity: ResearchEntity | null;
  setEntity: (e: ResearchEntity | null) => void;
  clear: () => void;
}

const STORAGE_KEY = "fingraph-research-context";

const Ctx = createContext<ResearchContextValue>({
  entity: null,
  setEntity: () => {},
  clear: () => {},
});

export function ResearchContextProvider({ children }: { children: React.ReactNode }) {
  const [entity, setEntityState] = useState<ResearchEntity | null>(null);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as ResearchEntity;
        if (parsed && typeof parsed.nodeId === "string" && typeof parsed.name === "string") {
          setEntityState(parsed);
        }
      }
    } catch {
      /* ignore unreadable storage */
    }
  }, []);

  const setEntity = useCallback((e: ResearchEntity | null) => {
    setEntityState(e);
    try {
      if (e) localStorage.setItem(STORAGE_KEY, JSON.stringify(e));
      else localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* storage unavailable */
    }
  }, []);

  const clear = useCallback(() => setEntity(null), [setEntity]);

  const value = useMemo(() => ({ entity, setEntity, clear }), [entity, setEntity, clear]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useResearchContext() {
  return useContext(Ctx);
}

/** Build a context entity for one of the corpus issuers. */
export function issuerEntity(ticker: string, name: string): ResearchEntity {
  return { nodeId: `company:${ticker}`, name, ticker, type: "Company" };
}
