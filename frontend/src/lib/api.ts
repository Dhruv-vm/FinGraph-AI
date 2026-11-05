/**
 * Typed client for the FinGraph AI FastAPI backend.
 *
 * Integrity rule: this client never substitutes bundled or invented data when
 * a request fails. Errors propagate as ApiError so views can show an explicit
 * error, offline or not-found state.
 */

import {
  BenchmarkSummaryResponse,
  CompanyListResponse,
  CorpusChunkResponse,
  CorpusFilingsResponse,
  DemoQuestionsResponse,
  GraphNodeItem,
  HealthResponse,
  ParametricLeakageResponse,
  PerQuestionResponse,
  QueryRequest,
  LiveCompany,
  LiveFetchResponse,
  QueryResponse,
  SubgraphResponse,
} from "./types";

const BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;
  data: unknown;

  constructor(message: string, status: number, data?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data;
  }

  /** True when the backend could not be reached at all. */
  get isNetwork() {
    return this.status === 0;
  }
  get isNotFound() {
    return this.status === 404;
  }
}

export function isAbortError(err: unknown): boolean {
  return (
    typeof err === "object" &&
    err !== null &&
    "name" in err &&
    (err as { name?: string }).name === "AbortError"
  );
}

function describeDetail(detail: unknown): string | null {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // FastAPI validation errors: [{loc, msg, type}]
    return detail
      .map((d) => {
        const item = d as { loc?: unknown[]; msg?: string };
        const loc = Array.isArray(item.loc) ? item.loc.filter((p) => p !== "body").join(".") : "";
        return loc ? `${loc}: ${item.msg}` : item.msg;
      })
      .filter(Boolean)
      .join("; ");
  }
  if (detail != null) return JSON.stringify(detail);
  return null;
}

async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const isBrowser = typeof window !== "undefined";
  // In the browser, the Next.js rewrite (/api → backend) is tried first to avoid CORS;
  // the absolute URL is a fallback for deployments without the rewrite.
  const urls = isBrowser ? [endpoint, `${BASE_URL}${endpoint}`] : [`${BASE_URL}${endpoint}`];

  let lastError: unknown = null;
  for (const url of urls) {
    try {
      const res = await fetch(url, {
        ...options,
        headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
      });
      if (res.ok) return (await res.json()) as T;

      let body: unknown;
      try {
        body = await res.json();
      } catch {
        body = await res.text().catch(() => null);
      }
      const detail =
        typeof body === "object" && body !== null && "detail" in body
          ? describeDetail((body as { detail: unknown }).detail)
          : null;
      // An HTTP error from the backend is authoritative; don't retry another URL.
      throw new ApiError(detail || `HTTP ${res.status} ${res.statusText}`.trim(), res.status, body);
    } catch (err) {
      if (isAbortError(err) || err instanceof ApiError) throw err;
      lastError = err;
    }
  }
  const message = lastError instanceof Error ? lastError.message : "Network request failed";
  throw new ApiError(`Backend unreachable (${message})`, 0);
}

export const api = {
  getHealth: (signal?: AbortSignal) => fetchJson<HealthResponse>("/api/v1/health", { signal }),

  submitQuery: (req: QueryRequest, signal?: AbortSignal) =>
    fetchJson<QueryResponse>("/api/v1/query", { method: "POST", body: JSON.stringify(req), signal }),

  resolveLiveCompany: (q: string, signal?: AbortSignal) =>
    fetchJson<{ company: LiveCompany | null }>(`/api/v1/live/resolve?${new URLSearchParams({ q })}`, { signal }),

  fetchLiveFilings: (req: { query?: string; ticker?: string }, signal?: AbortSignal) =>
    fetchJson<LiveFetchResponse>("/api/v1/live/fetch", { method: "POST", body: JSON.stringify(req), signal }),

  getDemoQuestions: (signal?: AbortSignal) =>
    fetchJson<DemoQuestionsResponse>("/api/v1/benchmark/demo-questions", { signal }),

  getBenchmarkSummary: (signal?: AbortSignal) =>
    fetchJson<BenchmarkSummaryResponse>("/api/v1/benchmark/summary", { signal }),

  getPerQuestion: (signal?: AbortSignal) =>
    fetchJson<PerQuestionResponse>("/api/v1/benchmark/per-question", { signal }),

  getParametricLeakage: (signal?: AbortSignal) =>
    fetchJson<ParametricLeakageResponse>("/api/v1/benchmark/parametric-leakage", { signal }),

  getCorpusFilings: (signal?: AbortSignal) =>
    fetchJson<CorpusFilingsResponse>("/api/v1/corpus/filings", { signal }),

  getCompanies: (signal?: AbortSignal) =>
    fetchJson<CompanyListResponse>("/api/v1/graph/companies", { signal }),

  getSubgraph: (
    nodeId: string,
    opts: { limit?: number; offset?: number; filterType?: "companies" | "all"; signal?: AbortSignal } = {}
  ) => {
    const params = new URLSearchParams({
      node_id: nodeId,
      limit: String(opts.limit ?? 100),
      offset: String(opts.offset ?? 0),
      filter_type: opts.filterType ?? "companies",
    });
    return fetchJson<SubgraphResponse>(`/api/v1/graph/subgraph?${params}`, { signal: opts.signal });
  },

  searchNodes: (query: string, limit = 15, signal?: AbortSignal) => {
    const params = new URLSearchParams({ q: query, limit: String(limit) });
    return fetchJson<GraphNodeItem[]>(`/api/v1/graph/search?${params}`, { signal });
  },

  browseChunks: (
    opts: {
      company?: string;
      documentId?: string;
      documentType?: string;
      limit?: number;
      offset?: string | null;
      signal?: AbortSignal;
    } = {}
  ) => {
    const params = new URLSearchParams({ limit: String(opts.limit ?? 25) });
    if (opts.company && opts.company !== "ALL") params.append("company", opts.company);
    if (opts.documentId) params.append("document_id", opts.documentId);
    if (opts.documentType) params.append("document_type", opts.documentType);
    if (opts.offset) params.append("offset", opts.offset);
    return fetchJson<CorpusChunkResponse>(`/api/v1/evidence/chunks?${params}`, { signal: opts.signal });
  },
};
