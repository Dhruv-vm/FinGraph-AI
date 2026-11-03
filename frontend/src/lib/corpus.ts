"use client";

/**
 * Session-wide caches for read-only reference data shared across views:
 * the filing manifest (joins evidence → filing metadata) and KG node labels
 * (resolves hashed node ids in graph paths). Failures are not cached.
 */

import { useEffect, useState } from "react";
import { api, ApiError } from "./api";
import { CorpusFilingsResponse, FilingItem, GraphNodeItem } from "./types";

let corpusPromise: Promise<CorpusFilingsResponse> | null = null;

export function loadCorpus(): Promise<CorpusFilingsResponse> {
  if (!corpusPromise) {
    corpusPromise = api.getCorpusFilings().catch((e) => {
      corpusPromise = null;
      throw e;
    });
  }
  return corpusPromise;
}

export function useCorpus() {
  const [data, setData] = useState<CorpusFilingsResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    let live = true;
    loadCorpus()
      .then((d) => live && (setData(d), setError(null)))
      .catch((e) => live && setError(e instanceof ApiError ? e : new ApiError(String(e), 0)));
    return () => {
      live = false;
    };
  }, [nonce]);
  return { data, error, loading: !data && !error, reload: () => setNonce((n) => n + 1) };
}

/** document_id → filing */
export function filingIndex(corpus: CorpusFilingsResponse | null): Map<string, FilingItem> {
  const m = new Map<string, FilingItem>();
  corpus?.filings.forEach((f) => m.set(f.document_id, f));
  return m;
}

const labelCache = new Map<string, Promise<GraphNodeItem | null>>();

/** Resolve a KG node id to its node record via the search endpoint (exact id match only). */
export function resolveNode(id: string): Promise<GraphNodeItem | null> {
  let p = labelCache.get(id);
  if (!p) {
    p = api
      .searchNodes(id, 5)
      .then((rows) => rows.find((r) => r.id === id) ?? null)
      .catch(() => {
        labelCache.delete(id);
        return null;
      });
    labelCache.set(id, p);
  }
  return p;
}

export function useNodeLabels(ids: string[]): Record<string, string> {
  const [labels, setLabels] = useState<Record<string, string>>({});
  const key = Array.from(new Set(ids)).sort().join("|");
  useEffect(() => {
    if (!key) return;
    let live = true;
    const unique = key.split("|");
    Promise.all(unique.map((id) => resolveNode(id).then((n) => [id, n?.label] as const))).then((pairs) => {
      if (!live) return;
      const next: Record<string, string> = {};
      pairs.forEach(([id, label]) => {
        if (label) next[id] = label;
      });
      setLabels(next);
    });
    return () => {
      live = false;
    };
  }, [key]);
  return labels;
}
