/** Shared formatting and domain helpers. All dates render as ISO (YYYY-MM-DD). */

export const fmtInt = (n: number | null | undefined) =>
  n == null || Number.isNaN(n) ? "—" : Math.round(n).toLocaleString("en-US");

/** Ratio (0–1) → percentage string. */
export const fmtPct = (ratio: number | null | undefined, digits = 1) =>
  ratio == null || Number.isNaN(ratio) ? "—" : `${(ratio * 100).toFixed(digits)}%`;

export const fmtNum = (n: number | null | undefined, digits = 2) =>
  n == null || Number.isNaN(n) ? "—" : n.toFixed(digits);

export const fmtMs = (ms: number | null | undefined) => {
  if (ms == null || Number.isNaN(ms)) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
};

/** "2025-04-30T20:08:52+00:00" → "2025-04-30". Leaves non-dates untouched. */
export function fmtDate(value: string | null | undefined): string {
  if (!value || value === "None") return "—";
  const m = /^(\d{4}-\d{2}-\d{2})/.exec(value);
  return m ? m[1] : value;
}

/** ISO timestamp → "2025-04-30 20:08 UTC". */
export function fmtDateTime(value: string | null | undefined): string {
  if (!value || value === "None") return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return `${d.toISOString().slice(0, 10)} ${d.toISOString().slice(11, 16)} UTC`;
}

export const pluralize = (n: number, one: string, many = `${one}s`) => `${fmtInt(n)} ${n === 1 ? one : many}`;

/** "factual_single_hop" → "Factual single hop" */
export const humanize = (s: string | null | undefined) => {
  if (!s) return "—";
  const t = s.replace(/_/g, " ").toLowerCase();
  return t.charAt(0).toUpperCase() + t.slice(1);
};

/** "COMPETES_WITH" → "Competes with" */
export const relLabel = (rel: string) => humanize(rel);

export type RelFamily = "acquired" | "competes" | "supplies" | "partners" | "invested" | "risk" | "other";

export function relFamily(rel: string): RelFamily {
  const r = (rel || "").toUpperCase();
  if (r.includes("ACQUIR") || r.includes("MERGE")) return "acquired";
  if (r.includes("COMPET")) return "competes";
  if (r.includes("SUPPL") || r.includes("CUSTOMER") || r.includes("MANUFACT") || r.includes("DEPENDS")) return "supplies";
  if (r.includes("PARTNER")) return "partners";
  if (r.includes("INVEST") || r.includes("OWN")) return "invested";
  if (r.includes("RISK") || r.includes("LITIG") || r.includes("REGULAT") || r.includes("AFFECTED")) return "risk";
  return "other";
}

export const REL_FAMILY_LABEL: Record<RelFamily, string> = {
  acquired: "Acquisition",
  competes: "Competition",
  supplies: "Supply chain",
  partners: "Partnership",
  invested: "Investment",
  risk: "Risk and regulation",
  other: "Other",
};

/** CSS colour for a relationship family (reads theme tokens). */
export const relColor = (family: RelFamily, alpha = 1) => `rgb(var(--rel-${family}) / ${alpha})`;

/** KG ids like "company:580e21bb1bdf8bda" carry a hash rather than a ticker. */
export const isHashId = (raw: string) => /^[0-9a-f]{16}$/.test(raw);

export function nodeKind(id: string): string {
  const i = id.indexOf(":");
  return i > 0 ? id.slice(0, i) : "";
}

/** Short display code for a node: ticker for issuers, otherwise none. */
export function nodeTicker(id: string): string | null {
  const raw = id.slice(id.indexOf(":") + 1);
  return id.startsWith("company:") && !isHashId(raw) ? raw.toUpperCase() : null;
}

/** "doc:sec:076412151b515b96:chunk:26" → 26 */
export function chunkIndex(chunkId: string | null | undefined): number | null {
  if (!chunkId) return null;
  const m = /chunk:(\d+)$/.exec(chunkId);
  return m ? Number(m[1]) : null;
}

/** The ten issuers whose filings make up the research corpus. */
export const CORPUS_ISSUERS: { ticker: string; name: string }[] = [
  { ticker: "AAPL", name: "Apple Inc." },
  { ticker: "AMZN", name: "Amazon.com, Inc." },
  { ticker: "GOOGL", name: "Alphabet Inc." },
  { ticker: "JNJ", name: "Johnson & Johnson" },
  { ticker: "JPM", name: "JPMorgan Chase & Co." },
  { ticker: "META", name: "Meta Platforms, Inc." },
  { ticker: "MSFT", name: "Microsoft" },
  { ticker: "NVDA", name: "NVIDIA Corporation" },
  { ticker: "TSLA", name: "Tesla, Inc." },
  { ticker: "XOM", name: "Exxon Mobil Corporation" },
];

export const issuerName = (ticker: string) =>
  CORPUS_ISSUERS.find((c) => c.ticker === ticker)?.name ?? ticker;

/** Validate a YYYY-MM-DD date and convert to the ISO reference time the API expects. */
export function toReferenceTime(date: string): string | null {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return null;
  const d = new Date(`${date}T00:00:00Z`);
  if (Number.isNaN(d.getTime()) || d.toISOString().slice(0, 10) !== date) return null;
  return `${date}T00:00:00+00:00`;
}

/**
 * Human phrase for a filing's period field. For 10-K/10-Q it is the fiscal period end;
 * for an 8-K the EDGAR "period of report" is the date of the reported event.
 */
export function periodPhrase(documentType: string | null | undefined, date: string | null | undefined): string | null {
  if (!date) return null;
  return documentType === "8-K" ? `report date ${fmtDate(date)}` : `period ending ${fmtDate(date)}`;
}
