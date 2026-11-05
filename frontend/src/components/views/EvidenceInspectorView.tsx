"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, ChevronLeft, ChevronRight, Crosshair, ExternalLink, Search, X } from "lucide-react";
import { api, ApiError, isAbortError } from "@/lib/api";
import { useCorpus } from "@/lib/corpus";
import { useResearchContext } from "@/lib/research-context";
import { CORPUS_ISSUERS, chunkIndex, fmtDate, fmtDateTime, fmtInt, issuerName, periodPhrase } from "@/lib/format";
import { CorpusChunkItem, FilingItem } from "@/lib/types";
import {
  Badge,
  Button,
  Drawer,
  EmptyState,
  ErrorState,
  IconButton,
  Input,
  KeyValue,
  Note,
  ProgressBar,
  Segmented,
  Select,
  Skeleton,
  Spinner,
  cx,
} from "@/components/ui";

type FormFilter = "ALL" | "10-K" | "10-Q" | "8-K";
type Sort = "newest" | "oldest" | "issuer";

interface OpenRequest {
  documentId: string;
  chunkId: string | null;
}

function readPendingOpen(): OpenRequest | null {
  try {
    const raw = sessionStorage.getItem("fingraph-open-filing");
    if (!raw) return null;
    sessionStorage.removeItem("fingraph-open-filing");
    return JSON.parse(raw) as OpenRequest;
  } catch {
    return null;
  }
}

/** Marks every case-insensitive occurrence of `term` in `text`. */
function Highlight({ text, term }: { text: string; term: string }) {
  const q = term.trim();
  if (q.length < 2) return <>{text}</>;
  const esc = q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const parts = text.split(new RegExp(`(${esc})`, "gi"));
  return (
    <>
      {parts.map((p, i) =>
        p.toLowerCase() === q.toLowerCase() ? (
          <mark key={i} className="bg-accent/25 text-fg rounded-[2px] px-0.5">
            {p}
          </mark>
        ) : (
          <React.Fragment key={i}>{p}</React.Fragment>
        )
      )}
    </>
  );
}

/* ------------------------------------------------------------------ */
/* Chunk loading for one filing                                        */
/* ------------------------------------------------------------------ */

function useFilingChunks(documentId: string | null) {
  const [chunks, setChunks] = useState<CorpusChunkItem[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    setChunks([]);
    setTotal(null);
    setError(null);
    if (!documentId) return;
    const ctrl = new AbortController();
    setLoading(true);
    (async () => {
      try {
        let offset: string | null = null;
        const all: CorpusChunkItem[] = [];
        // Page until exhausted (filings hold up to a few hundred chunks).
        for (let i = 0; i < 10; i++) {
          const page = await api.browseChunks({ documentId, limit: 100, offset, signal: ctrl.signal });
          all.push(...page.chunks);
          if (page.total_matching != null) setTotal(page.total_matching);
          offset = page.offset ?? null;
          if (!offset || page.chunks.length === 0) break;
        }
        all.sort((a, b) => (chunkIndex(a.chunk_id) ?? 0) - (chunkIndex(b.chunk_id) ?? 0));
        if (!ctrl.signal.aborted) setChunks(all);
      } catch (e) {
        if (!isAbortError(e)) setError(e instanceof ApiError ? e : new ApiError(String(e), 0));
      } finally {
        if (!ctrl.signal.aborted) setLoading(false);
      }
    })();
    return () => ctrl.abort();
  }, [documentId, nonce]);

  return { chunks, total, loading, error, reload: () => setNonce((n) => n + 1) };
}

/* ------------------------------------------------------------------ */
/* Reader                                                              */
/* ------------------------------------------------------------------ */

function ChunkReader({
  chunk,
  filing,
  position,
  count,
  onPrev,
  onNext,
  highlight = "",
  linked = false,
}: {
  chunk: CorpusChunkItem;
  filing?: FilingItem | null;
  position: number | null;
  count: number | null;
  onPrev?: () => void;
  onNext?: () => void;
  highlight?: string;
  linked?: boolean;
}) {
  const url = chunk.source_url || filing?.source_url;
  return (
    <article className="px-5 sm:px-8 py-6">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
        <div className="text-xs text-fg-3">
          {issuerName(chunk.company)} {chunk.document_type ?? filing?.document_type}
          {position != null && count != null && <span className="num">, passage {position} of {count}</span>}
        </div>
        <div className="flex items-center gap-1">
          <IconButton label="Previous passage" size="sm" onClick={onPrev} disabled={!onPrev}>
            <ChevronLeft className="w-4 h-4" />
          </IconButton>
          <IconButton label="Next passage" size="sm" onClick={onNext} disabled={!onNext}>
            <ChevronRight className="w-4 h-4" />
          </IconButton>
        </div>
      </div>

      {linked && (
        <div className="mb-4 inline-flex items-center gap-2 rounded-md border border-accent/30 bg-accent/10 px-2.5 py-1 text-xs text-fg">
          <span className="w-1.5 h-1.5 rounded-full bg-accent" /> The passage you followed a link to
        </div>
      )}
      <div className={cx("reading whitespace-pre-wrap", linked && "border-l-2 border-accent pl-4")}>
        <Highlight text={chunk.text} term={highlight} />
      </div>

      <div className="mt-8 pt-4 border-t border-line-soft max-w-[68ch]">
        <h3 className="text-xs font-medium text-fg-2 mb-1">Provenance</h3>
        <KeyValue
          dense
          items={[
            { label: "Issuer", value: `${issuerName(chunk.company)} (${chunk.company})` },
            { label: "Form", value: chunk.document_type ?? "—" },
            { label: "Filing date", value: fmtDate(chunk.publication_date ?? filing?.publication_date), mono: true, hint: "Date filed with the SEC" },
            {
              label: filing?.document_type === "8-K" ? "Report date" : "Reporting period",
              value: filing?.fiscal_period ? fmtDate(filing.fiscal_period) : <span className="text-fg-3">Not recorded</span>,
              hint: filing?.document_type === "8-K" ? "Date of the event the 8-K reports" : "Fiscal period end the filing covers",
            },
            {
              label: "Event date",
              value: chunk.event_time ? fmtDateTime(chunk.event_time) : <span className="text-fg-3">Not recorded</span>,
            },
            {
              label: "Available from",
              value: fmtDateTime(chunk.available_time ?? filing?.available_time),
              mono: true,
              hint: "Earliest reference time that can use this passage",
            },
            { label: "Section", value: chunk.section ?? <span className="text-fg-3">Not recorded in chunk metadata</span> },
            ...(filing?.source_reference ? [{ label: "Accession no.", value: filing.source_reference, mono: true }] : []),
            { label: "Document id", value: chunk.document_id, mono: true },
            { label: "Chunk id", value: chunk.chunk_id, mono: true },
          ]}
        />
        {url && (
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 mt-3 text-sm text-accent hover:underline"
          >
            Open the filing on SEC EDGAR <ExternalLink className="w-3.5 h-3.5" />
          </a>
        )}
      </div>
    </article>
  );
}

/* ------------------------------------------------------------------ */
/* Fallback: original chunk browse (no corpus endpoint)                */
/* ------------------------------------------------------------------ */

function ChunkBrowseFallback({ company }: { company: string }) {
  const [chunks, setChunks] = useState<CorpusChunkItem[]>([]);
  const [offset, setOffset] = useState<string | null>(null);
  const [total, setTotal] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [selected, setSelected] = useState<CorpusChunkItem | null>(null);

  const load = useCallback(
    async (reset: boolean, from: string | null) => {
      setLoading(true);
      try {
        const page = await api.browseChunks({ company, limit: 50, offset: reset ? null : from });
        setChunks((c) => (reset ? page.chunks : [...c, ...page.chunks]));
        setOffset(page.offset ?? null);
        setTotal(page.total_matching ?? null);
        setError(null);
        if (reset) setSelected(page.chunks[0] ?? null);
      } catch (e) {
        setError(e instanceof ApiError ? e : new ApiError(String(e), 0));
      } finally {
        setLoading(false);
      }
    },
    [company]
  );

  useEffect(() => {
    load(true, null);
  }, [load]);

  return (
    <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[360px_1fr]">
      <div className="border-r border-line-soft overflow-y-auto scrollbar-thin">
        <Note tone="warn" className="m-3">
          Filing-level browsing needs the <code className="font-mono-num">/api/v1/corpus/filings</code> endpoint. Restart the
          FastAPI server to enable it. Showing raw passages from Qdrant meanwhile.
        </Note>
        {error && <ErrorState compact className="m-3" title="Could not load passages" error={error} />}
        <ul className="divide-y divide-line-soft">
          {chunks.map((c) => (
            <li key={c.chunk_id}>
              <button
                onClick={() => setSelected(c)}
                className={cx("w-full text-left px-4 py-2.5", selected?.chunk_id === c.chunk_id ? "bg-surface-2" : "hover:bg-surface-2/50")}
              >
                <div className="flex gap-2 text-xs text-fg-3">
                  <span className="font-mono-num text-fg-2">{c.company}</span>
                  <span>{c.document_type}</span>
                  <span className="font-mono-num ml-auto">{fmtDate(c.publication_date)}</span>
                </div>
                <div className="text-sm text-fg-2 line-clamp-2 mt-0.5">{c.text}</div>
              </button>
            </li>
          ))}
        </ul>
        <div className="p-3 text-xs text-fg-3 flex items-center justify-between">
          <span className="num">
            {fmtInt(chunks.length)} loaded{total != null ? ` of ${fmtInt(total)}` : ""}
          </span>
          {offset && (
            <Button size="sm" loading={loading} onClick={() => load(false, offset)}>
              Load more
            </Button>
          )}
        </div>
      </div>
      <div className="overflow-y-auto scrollbar-thin">
        {selected ? <ChunkReader chunk={selected} position={null} count={null} /> : <EmptyState title="Select a passage" />}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* View                                                                */
/* ------------------------------------------------------------------ */

export function EvidenceInspectorView({ active }: { active: boolean }) {
  const corpus = useCorpus();
  const { entity, clear: clearContext } = useResearchContext();
  const contextTicker = entity?.ticker && CORPUS_ISSUERS.some((c) => c.ticker === entity.ticker) ? entity.ticker : null;

  const [issuer, setIssuer] = useState<string>(contextTicker ?? "ALL");
  const [form, setForm] = useState<FormFilter>("ALL");
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [sort, setSort] = useState<Sort>("newest");
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null);
  const [selectedChunk, setSelectedChunk] = useState<string | null>(null);
  const [chunkFilter, setChunkFilter] = useState("");
  const [mobilePane, setMobilePane] = useState<"list" | "filing">("list");
  const [readerOpen, setReaderOpen] = useState(false);
  const pendingChunk = useRef<string | null>(null);
  const [linkedChunk, setLinkedChunk] = useState<string | null>(null);
  const [libQuery, setLibQuery] = useState("");

  // Follow research context changes.
  useEffect(() => {
    setIssuer(contextTicker ?? "ALL");
  }, [contextTicker]);

  // Deep link from Ask / Graph.
  const applyOpen = useCallback((req: OpenRequest) => {
    setSelectedDoc(req.documentId);
    setSelectedChunk(null);
    pendingChunk.current = req.chunkId;
    setLinkedChunk(req.chunkId);
    setMobilePane("filing");
    setChunkFilter("");
  }, []);

  useEffect(() => {
    const pending = readPendingOpen();
    if (pending) applyOpen(pending);
    const onOpen = (e: Event) => {
      const d = (e as CustomEvent<OpenRequest>).detail;
      if (d?.documentId) {
        try {
          sessionStorage.removeItem("fingraph-open-filing");
        } catch {
          /* ignore */
        }
        applyOpen(d);
      }
    };
    window.addEventListener("fingraph:open-filing", onOpen);
    return () => window.removeEventListener("fingraph:open-filing", onOpen);
  }, [applyOpen]);

  const filings = useMemo(() => {
    const all = corpus.data?.filings ?? [];
    const list = all.filter(
      (f) =>
        (issuer === "ALL" || f.company === issuer) &&
        (form === "ALL" || f.document_type === form) &&
        (!from || (f.publication_date ?? "") >= from) &&
        (!to || (f.publication_date ?? "") <= to) &&
        (!libQuery.trim() ||
          [f.company, issuerName(f.company), f.document_type, f.publication_date, f.fiscal_period, f.source_reference, f.document_id]
            .filter(Boolean)
            .join(" ")
            .toLowerCase()
            .includes(libQuery.trim().toLowerCase()))
    );
    return list.sort((a, b) =>
      sort === "issuer"
        ? a.company.localeCompare(b.company) || (b.publication_date ?? "").localeCompare(a.publication_date ?? "")
        : sort === "newest"
        ? (b.publication_date ?? "").localeCompare(a.publication_date ?? "")
        : (a.publication_date ?? "").localeCompare(b.publication_date ?? "")
    );
  }, [corpus.data, issuer, form, from, to, sort, libQuery]);

  const formCounts = useMemo(() => {
    const c: Record<string, number> = {};
    (corpus.data?.filings ?? [])
      .filter((f) => issuer === "ALL" || f.company === issuer)
      .forEach((f) => (c[f.document_type ?? "?"] = (c[f.document_type ?? "?"] ?? 0) + 1));
    return c;
  }, [corpus.data, issuer]);

  const filing = corpus.data?.filings.find((f) => f.document_id === selectedDoc) ?? null;
  const chunks = useFilingChunks(selectedDoc);

  // Select the requested (or first) chunk once a filing's chunks arrive.
  useEffect(() => {
    if (!chunks.chunks.length) return;
    const want = pendingChunk.current;
    pendingChunk.current = null;
    if (want && chunks.chunks.some((c) => c.chunk_id === want)) setSelectedChunk(want);
    else if (!selectedChunk || !chunks.chunks.some((c) => c.chunk_id === selectedChunk)) setSelectedChunk(chunks.chunks[0].chunk_id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chunks.chunks]);

  // Scroll the selected chunk row into view.
  useEffect(() => {
    if (!selectedChunk || !active) return;
    document.getElementById(`chunk-row-${selectedChunk}`)?.scrollIntoView({ block: "nearest" });
  }, [selectedChunk, active]);

  const visibleChunks = useMemo(() => {
    const q = chunkFilter.trim().toLowerCase();
    return q ? chunks.chunks.filter((c) => c.text.toLowerCase().includes(q)) : chunks.chunks;
  }, [chunks.chunks, chunkFilter]);

  const chunkPos = chunks.chunks.findIndex((c) => c.chunk_id === selectedChunk);
  const current = chunkPos >= 0 ? chunks.chunks[chunkPos] : null;
  const step = (d: number) => {
    const next = chunks.chunks[chunkPos + d];
    if (next) setSelectedChunk(next.chunk_id);
  };

  /* ---------- fallback when corpus endpoint missing ---------- */
  if (corpus.error) {
    return (
      <div className="h-full flex flex-col">
        <div className="px-5 py-3 border-b border-line-soft flex items-center gap-3">
          <Select value={issuer} onChange={(e) => setIssuer(e.target.value)} aria-label="Issuer" className="w-56">
            <option value="ALL">All issuers</option>
            {CORPUS_ISSUERS.map((c) => (
              <option key={c.ticker} value={c.ticker}>
                {c.name}
              </option>
            ))}
          </Select>
          {!corpus.error.isNotFound && <span className="text-xs text-neg">{corpus.error.message}</span>}
        </div>
        <ChunkBrowseFallback company={issuer} />
      </div>
    );
  }

  const filtersActive = issuer !== "ALL" || form !== "ALL" || from || to || libQuery;

  return (
    <div className="h-full flex flex-col lg:flex-row min-h-0">
      {/* Library */}
      <section
        aria-label="Filing library"
        className={cx(
          "flex-1 lg:flex-none lg:w-[340px] xl:w-[360px] flex flex-col min-h-0 border-r border-line-soft bg-sidebar/60",
          mobilePane === "filing" && "hidden lg:flex"
        )}
      >
        <div className="p-4 space-y-3 border-b border-line-soft">
          <Input
            inputSize="sm"
            placeholder="Search filings by company, form, date or accession"
            aria-label="Search filings"
            leading={<Search className="w-3.5 h-3.5" />}
            value={libQuery}
            onChange={(e) => setLibQuery(e.target.value)}
          />
          <div className="flex items-center gap-2">
            <Select value={issuer} onChange={(e) => setIssuer(e.target.value)} aria-label="Issuer" className="flex-1">
              <option value="ALL">All issuers</option>
              {(corpus.data?.companies ?? []).map((c) => (
                <option key={c.company} value={c.company}>
                  {issuerName(c.company)} ({c.filings})
                </option>
              ))}
            </Select>
            <Select value={sort} onChange={(e) => setSort(e.target.value as Sort)} aria-label="Sort filings" className="w-[110px]">
              <option value="newest">Newest</option>
              <option value="oldest">Oldest</option>
              <option value="issuer">Issuer</option>
            </Select>
          </div>
          {contextTicker && issuer === contextTicker && (
            <div className="flex items-center gap-1.5 text-xs text-fg-2">
              <Crosshair className="w-3.5 h-3.5 text-accent" />
              Filtered by research context
              <button className="ml-auto text-fg-3 hover:text-fg inline-flex items-center gap-0.5" onClick={() => setIssuer("ALL")}>
                <X className="w-3 h-3" /> Show all
              </button>
            </div>
          )}
          <Segmented
            label="Form type"
            value={form}
            onChange={setForm}
            className="w-full justify-between"
            options={[
              { value: "ALL", label: "All" },
              { value: "10-K", label: "10-K", count: formCounts["10-K"] ?? 0 },
              { value: "10-Q", label: "10-Q", count: formCounts["10-Q"] ?? 0 },
              { value: "8-K", label: "8-K", count: formCounts["8-K"] ?? 0 },
            ]}
          />
          <div className="grid grid-cols-2 gap-2">
            <label className="text-2xs text-fg-3">
              Filed from
              <Input type="date" inputSize="sm" value={from} onChange={(e) => setFrom(e.target.value)} className="mt-1" />
            </label>
            <label className="text-2xs text-fg-3">
              Filed to
              <Input type="date" inputSize="sm" value={to} onChange={(e) => setTo(e.target.value)} className="mt-1" />
            </label>
          </div>
          <div className="flex items-center justify-between text-xs text-fg-3">
            <span className="num">
              {corpus.data ? `${filings.length} of ${corpus.data.total_filings} filings` : "Loading filings…"}
            </span>
            {filtersActive && (
              <button
                className="text-accent hover:underline"
                onClick={() => {
                  setIssuer("ALL");
                  setForm("ALL");
                  setFrom("");
                  setTo("");
                  setLibQuery("");
                }}
              >
                Reset filters
              </button>
            )}
          </div>
        </div>
        <div className="flex-1 min-h-0 overflow-y-auto scrollbar-thin">
          {corpus.loading ? (
            <div className="p-4 space-y-3">
              {Array.from({ length: 8 }).map((_, i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))}
            </div>
          ) : filings.length === 0 ? (
            <EmptyState title="No filings match these filters">Widen the date range or choose another form type.</EmptyState>
          ) : (
            <ul>
              {filings.map((f) => {
                const on = f.document_id === selectedDoc;
                return (
                  <li key={f.document_id}>
                    <button
                      onClick={() => {
                        setSelectedDoc(f.document_id);
                        setSelectedChunk(null);
                        setChunkFilter("");
                        setLinkedChunk(null);
                        setMobilePane("filing");
                      }}
                      aria-current={on ? "true" : undefined}
                      className={cx(
                        "w-full text-left px-4 py-2.5 border-b border-line-soft grid grid-cols-[3.5rem_1fr_auto] gap-x-2 items-baseline",
                        on ? "bg-accent/[0.08]" : "hover:bg-surface-2/60"
                      )}
                    >
                      <span className="font-mono-num text-xs text-fg-2">{f.company}</span>
                      <span className="text-sm text-fg">
                        {f.document_type}
                        {f.fiscal_period && <span className="text-xs text-fg-3">, {periodPhrase(f.document_type, f.fiscal_period)}</span>}
                      </span>
                      <span className="font-mono-num text-xs text-fg-3">{fmtDate(f.publication_date)}</span>
                      <span />
                      <span className="text-2xs text-fg-3 num col-span-2">{f.chunk_count != null ? `${f.chunk_count} passages` : ""}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </section>

      {/* Filing */}
      <section
        aria-label="Filing"
        className={cx("flex-1 min-w-0 min-h-0 flex flex-col", mobilePane === "list" && "hidden lg:flex")}
      >
        {!selectedDoc ? (
          <div className="flex-1 flex items-center justify-center">
            <EmptyState title="Choose a filing">
              {corpus.data
                ? `${corpus.data.total_filings} filings by ${corpus.data.companies.length} issuers, ${fmtInt(corpus.data.total_chunks)} passages. Pick one to read its disclosures.`
                : "Pick one to read its disclosures."}
            </EmptyState>
          </div>
        ) : (
          <>
            <header className="px-4 sm:px-6 py-3 border-b border-line-soft">
              <button className="lg:hidden text-xs text-fg-2 inline-flex items-center gap-1 mb-2" onClick={() => setMobilePane("list")}>
                <ArrowLeft className="w-3.5 h-3.5" /> Filings
              </button>
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0">
                  <h2 className="text-xl font-medium tracking-tight">
                    {filing ? `${issuerName(filing.company)}, Form ${filing.document_type}` : selectedDoc}
                  </h2>
                  {filing && (
                    <p className="text-xs text-fg-3 mt-0.5 num">
                      Filed {fmtDate(filing.publication_date)}
                      {filing.fiscal_period && `, ${periodPhrase(filing.document_type, filing.fiscal_period)}`}
                      {filing.source_reference && `, accession ${filing.source_reference}`}
                    </p>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <Badge mono>{selectedDoc}</Badge>
                  {filing?.source_url && (
                    <a
                      href={filing.source_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-xs text-accent hover:underline"
                    >
                      SEC <ExternalLink className="w-3 h-3" />
                    </a>
                  )}
                </div>
              </div>
            </header>
            <ProgressBar active={chunks.loading} />
            <div className="flex-1 min-h-0 flex">
              {/* Chunk index */}
              <div className="w-full xl:w-[320px] shrink-0 xl:border-r border-line-soft flex flex-col min-h-0">
                <div className="p-3 border-b border-line-soft">
                  <Input
                    inputSize="sm"
                    placeholder="Find in this filing"
                    aria-label="Find in this filing"
                    leading={<Search className="w-3.5 h-3.5" />}
                    value={chunkFilter}
                    onChange={(e) => setChunkFilter(e.target.value)}
                  />
                  <div className="text-2xs text-fg-3 mt-1.5 num">
                    {chunks.loading
                      ? "Loading passages…"
                      : chunkFilter
                      ? `${visibleChunks.length} of ${chunks.chunks.length} passages match`
                      : `${chunks.chunks.length}${chunks.total != null && chunks.total !== chunks.chunks.length ? ` of ${chunks.total}` : ""} passages`}
                  </div>
                </div>
                <div className="flex-1 min-h-0 overflow-y-auto scrollbar-thin">
                  {chunks.error ? (
                    <ErrorState
                      compact
                      className="m-3"
                      title="Could not load passages"
                      error={chunks.error}
                      action={
                        <Button size="sm" onClick={chunks.reload}>
                          Retry
                        </Button>
                      }
                    />
                  ) : chunks.loading && !chunks.chunks.length ? (
                    <div className="flex items-center gap-2 p-4 text-sm text-fg-3">
                      <Spinner /> Loading
                    </div>
                  ) : !chunks.loading && chunks.chunks.length === 0 ? (
                    <EmptyState title="No passages returned" />
                  ) : (
                    <ol>
                      {visibleChunks.map((c) => {
                        const on = c.chunk_id === selectedChunk;
                        return (
                          <li key={c.chunk_id} id={`chunk-row-${c.chunk_id}`}>
                            <button
                              onClick={() => {
                                setSelectedChunk(c.chunk_id);
                                setReaderOpen(true);
                              }}
                              className={cx(
                                "w-full text-left px-3 py-2 grid grid-cols-[2.25rem_1fr] gap-2 border-b border-line-soft",
                                on ? "bg-accent/[0.08]" : "hover:bg-surface-2/50"
                              )}
                            >
                              <span className={cx("font-mono-num text-2xs pt-0.5", on ? "text-accent" : "text-fg-3")}>{chunkIndex(c.chunk_id)}</span>
                              <span className="text-xs text-fg-2 line-clamp-2 leading-relaxed">
                                {c.chunk_id === linkedChunk && <span className="text-accent font-medium">Linked: </span>}
                                <Highlight text={c.text} term={chunkFilter} />
                              </span>
                            </button>
                          </li>
                        );
                      })}
                    </ol>
                  )}
                </div>
              </div>
              {/* Reader (wide screens) */}
              <div className="hidden xl:block flex-1 min-w-0 overflow-y-auto scrollbar-thin">
                {current ? (
                  <ChunkReader
                    chunk={current}
                    filing={filing}
                    position={chunkPos + 1}
                    count={chunks.chunks.length}
                    onPrev={chunkPos > 0 ? () => step(-1) : undefined}
                    onNext={chunkPos < chunks.chunks.length - 1 ? () => step(1) : undefined}
                    highlight={chunkFilter}
                    linked={current.chunk_id === linkedChunk}
                  />
                ) : (
                  !chunks.loading && <EmptyState title="Select a passage" className="mt-16" />
                )}
              </div>
            </div>
          </>
        )}
      </section>

      {/* Reader (below xl) */}
      <div className="xl:hidden">
        <Drawer open={readerOpen && !!current} onClose={() => setReaderOpen(false)} title="Passage" width="w-[min(640px,100vw)]">
          {current && (
            <ChunkReader
              chunk={current}
              filing={filing}
              position={chunkPos + 1}
              count={chunks.chunks.length}
              onPrev={chunkPos > 0 ? () => step(-1) : undefined}
              onNext={chunkPos < chunks.chunks.length - 1 ? () => step(1) : undefined}
              highlight={chunkFilter}
              linked={current.chunk_id === linkedChunk}
            />
          )}
        </Drawer>
      </div>
    </div>
  );
}
