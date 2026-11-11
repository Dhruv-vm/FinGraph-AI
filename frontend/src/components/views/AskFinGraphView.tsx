"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowUpRight,
  BookOpen,
  CalendarClock,
  CheckCircle2,
  ChevronDown,
  CircleSlash,
  CloudDownload,
  Crosshair,
  ExternalLink,
  History,
  Square,
  TriangleAlert,
} from "lucide-react";
import { api, ApiError, isAbortError } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import { filingIndex, useCorpus, useNodeLabels } from "@/lib/corpus";
import { NavTab } from "@/lib/navigation";
import { useResearchContext } from "@/lib/research-context";
import {
  chunkIndex,
  fmtDate,
  fmtDateTime,
  fmtMs,
  fmtNum,
  fmtPct,
  humanize,
  relColor,
  relFamily,
  relLabel,
  periodPhrase,
  toReferenceTime,
} from "@/lib/format";
import { DemoQuestionItem, EvidenceItem, FilingItem, LiveCompany, LiveFetchResponse, QueryMode, QueryResponse } from "@/lib/types";
import {
  Badge,
  Button,
  EmptyState,
  ErrorState,
  FieldLabel,
  IconButton,
  Input,
  KeyValue,
  Note,
  ProgressBar,
  Segmented,
  Skeleton,
  Spinner,
  Tabs,
  Textarea,
  cx,
} from "@/components/ui";

/* ------------------------------------------------------------------ */
/* Outcome classification                                              */
/* ------------------------------------------------------------------ */

type OutcomeKind = "supported" | "partial" | "insufficient" | "validation" | "generation";

interface Outcome {
  kind: OutcomeKind;
  label: string;
  explanation: string;
  tone: "pos" | "warn" | "neg" | "info";
}

function classify(r: QueryResponse): Outcome {
  const v = r.validation_status as { is_valid?: boolean; errors?: string[]; warnings?: string[] };
  // The pipeline reports an LLM outage as a refusal (HTTP 200). Separate it from a genuine evidence refusal.
  const generationFailed =
    r.sufficiency.sufficient &&
    /generation failed|llm backend error|ollama execution failed|failed to connect to ollama/i.test(
      `${r.answer} ${r.reasoning_summary ?? ""}`
    );
  if (generationFailed) {
    return {
      kind: "generation",
      label: "Generation failed",
      tone: "neg",
      explanation:
        "The evidence passed the sufficiency gate, but the language model could not run, so no answer was generated. Start Ollama and run the question again.",
    };
  }
  if (r.decision === "refuse" || r.insufficient_evidence) {
    return {
      kind: "insufficient",
      label: "Insufficient evidence",
      tone: "warn",
      explanation:
        r.sufficiency.sufficient
          ? // Gate passed, but the model found the retrieved passages do not answer the question.
            "The retrieved filings were read, but they do not answer this question, so FinGraph declined rather than guess."
          : r.sufficiency.reason ||
            "The evidence gate found the retrieved filings do not support an answer, so the system declined rather than guess.",
    };
  }
  if (v?.is_valid === false || (v?.errors && v.errors.length > 0)) {
    return {
      kind: "validation",
      label: "Validation issue",
      tone: "neg",
      explanation: "The answer was generated but failed post-generation validation. Treat it as unverified.",
    };
  }
  const gaps: string[] = [];
  if (!r.sufficiency.sufficient) gaps.push("the sufficiency gate did not pass");
  if (!r.sufficiency.provenance_complete) gaps.push("provenance is incomplete");
  if (!r.temporal_validation.temporal_valid) gaps.push("temporal validation failed");
  if (v?.warnings && v.warnings.length > 0) gaps.push(`${v.warnings.length} validation warning${v.warnings.length > 1 ? "s" : ""}`);
  if (gaps.length) {
    return {
      kind: "partial",
      label: "Partially supported",
      tone: "info",
      explanation: `An answer was produced, but ${gaps.join(", ")}. Check the evidence before relying on it.`,
    };
  }
  return {
    kind: "supported",
    label: "Supported answer",
    tone: "pos",
    explanation: "Evidence passed the sufficiency gate, citations validated, and provenance is complete.",
  };
}

const OUTCOME_ICON: Record<OutcomeKind, React.ReactNode> = {
  supported: <CheckCircle2 className="w-4 h-4" />,
  partial: <TriangleAlert className="w-4 h-4" />,
  insufficient: <CircleSlash className="w-4 h-4" />,
  validation: <TriangleAlert className="w-4 h-4" />,
  generation: <TriangleAlert className="w-4 h-4" />,
};

const TONE_TEXT = { pos: "text-pos", warn: "text-warn", neg: "text-neg", info: "text-info" } as const;
const TONE_PILL = {
  pos: "border-pos/30 bg-pos/10",
  warn: "border-warn/30 bg-warn/10",
  neg: "border-neg/30 bg-neg/10",
  info: "border-info/30 bg-info/10",
} as const;

/* ------------------------------------------------------------------ */

interface Run {
  id: number;
  query: string;
  referenceDate: string;
  startedAt: number;
  result: QueryResponse | null;
  error: ApiError | null;
}

function useElapsed(running: boolean) {
  const [ms, setMs] = useState(0);
  useEffect(() => {
    if (!running) return;
    const start = performance.now();
    setMs(0);
    const t = window.setInterval(() => setMs(performance.now() - start), 200);
    return () => window.clearInterval(t);
  }, [running]);
  return ms;
}

function SourceTypeBadge({ type }: { type: string }) {
  const tone = type === "graph" ? "info" : type === "hybrid" ? "accent" : "neutral";
  return (
    <Badge tone={tone as "info" | "accent" | "neutral"} title={`Retrieved by ${type} retrieval`}>
      {type}
    </Badge>
  );
}

function ProvenanceBadge({ verified, status }: { verified: boolean; status?: string | null }) {
  if (verified)
    return (
      <Badge tone="pos" title={status ?? "verified"}>
        Verified
      </Badge>
    );
  return (
    <Badge tone="warn" title={status ?? "unverified"}>
      {status ? humanize(status) : "Unverified"}
    </Badge>
  );
}

/* ------------------------------------------------------------------ */
/* Evidence ledger                                                     */
/* ------------------------------------------------------------------ */

function EvidenceRow({
  ev,
  citeNo,
  filing,
  referenceTime,
  onOpenFiling,
}: {
  ev: EvidenceItem;
  citeNo?: number;
  filing?: FilingItem;
  referenceTime: string | null;
  onOpenFiling: () => void;
}) {
  const [open, setOpen] = useState(false);
  const afterRef =
    referenceTime && ev.available_time ? new Date(ev.available_time) > new Date(referenceTime) : false;
  const url = ev.source_url || filing?.source_url || null;
  const company = ev.source_company || filing?.company;
  const isGraph = ev.source_type === "graph";
  return (
    <li className={cx("border-b border-line-soft last:border-0", citeNo != null && "bg-accent/[0.035]")}>
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="w-full text-left px-4 py-3 grid grid-cols-[2.25rem_1fr_auto] gap-3 items-start hover:bg-surface-2/50 transition-colors"
      >
        <span className="font-mono-num text-xs text-fg-3 pt-0.5">#{ev.rank}</span>
        <span className="min-w-0">
          <span className="flex flex-wrap items-center gap-1.5 mb-1">
            {citeNo != null && (
              <Badge tone="accent" mono>
                Cited [{citeNo}]
              </Badge>
            )}
            <SourceTypeBadge type={ev.source_type} />
            {company && <span className="font-mono-num text-xs text-fg-2">{company}</span>}
            {filing?.document_type && <span className="text-xs text-fg-3">{filing.document_type}</span>}
            {ev.available_time && (
              <span className={cx("font-mono-num text-xs", afterRef ? "text-neg" : "text-fg-3")}>
                available {fmtDate(ev.available_time)}
              </span>
            )}
          </span>
          <span className={cx("block text-sm text-fg-2", !open && "line-clamp-2")}>{ev.text}</span>
        </span>
        <span className="flex items-center gap-2 pt-0.5">
          <span className="font-mono-num text-xs text-fg-3" title="Fused retrieval score">
            {fmtNum(ev.score, 3)}
          </span>
          <ChevronDown className={cx("w-4 h-4 text-fg-3 transition-transform", open && "rotate-180")} />
        </span>
      </button>
      {open && (
        <div className="px-4 pb-4 pl-[3.25rem] animate-fade-in">
          <p className="reading text-[0.9375rem] whitespace-pre-wrap">{ev.text}</p>
          <KeyValue
            dense
            className="mt-4 max-w-2xl"
            items={[
              { label: "Evidence id", value: ev.evidence_id, mono: true },
              ...(ev.document_id ? [{ label: "Document", value: ev.document_id, mono: true }] : []),
              ...(ev.chunk_id && !isGraph ? [{ label: "Chunk", value: ev.chunk_id, mono: true }] : []),
              ...(filing
                ? [
                    {
                      label: "Filing",
                      value: `${filing.company} ${filing.document_type ?? ""}, filed ${fmtDate(filing.publication_date)}${
                        filing.fiscal_period ? `, ${periodPhrase(filing.document_type, filing.fiscal_period)}` : ""
                      }`,
                    },
                  ]
                : []),
              { label: "Available from", value: fmtDateTime(ev.available_time), mono: true },
              ...(ev.event_time ? [{ label: "Event time", value: fmtDateTime(ev.event_time), mono: true }] : []),
              {
                label: "Provenance",
                value: <ProvenanceBadge verified={ev.provenance_verified} status={ev.attribution_status} />,
              },
            ]}
          />
          <div className="flex flex-wrap gap-2 mt-3">
            {ev.document_id && (
              <Button size="sm" onClick={onOpenFiling} icon={<BookOpen className="w-3.5 h-3.5" />}>
                Open in evidence explorer
              </Button>
            )}
            {url && (
              <a
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 h-7 px-2.5 text-xs rounded text-accent hover:bg-accent/10"
              >
                SEC filing <ExternalLink className="w-3.5 h-3.5" />
              </a>
            )}
          </div>
        </div>
      )}
    </li>
  );
}

/* ------------------------------------------------------------------ */
/* Result                                                              */
/* ------------------------------------------------------------------ */

type ResultTab = "evidence" | "paths" | "checks" | "sources";

function Result({
  run,
  demo,
  onOpenFiling,
}: {
  run: Run & { result: QueryResponse };
  demo: DemoQuestionItem | null;
  onOpenFiling: (documentId: string, chunkId?: string | null) => void;
}) {
  const r = run.result;
  const outcome = classify(r);
  const corpus = useCorpus();
  const filings = useMemo(() => filingIndex(corpus.data), [corpus.data]);
  const [tab, setTab] = useState<ResultTab>("evidence");
  const [showCitedOnly, setShowCitedOnly] = useState(false);

  const citeNumbers = useMemo(() => {
    const m = new Map<string, number>();
    r.citations.forEach((c, i) => m.set(c.evidence_id, i + 1));
    return m;
  }, [r.citations]);

  const paths = useMemo(
    () => (r.graph_paths || []) as { source: string; relationship: string; target: string }[][],
    [r.graph_paths]
  );
  const pathNodeIds = useMemo(() => paths.flat().flatMap((s) => [s.source, s.target]), [paths]);
  const labels = useNodeLabels(pathNodeIds);

  const ref = r.temporal_validation.reference_time;
  const afterRefCount = ref
    ? r.evidence.filter((e) => e.available_time && new Date(e.available_time) > new Date(ref)).length
    : null;
  const v = r.validation_status as { is_valid?: boolean; errors?: string[]; warnings?: string[]; cited_evidence_ids?: string[] };
  const evidenceShown = showCitedOnly ? r.evidence.filter((e) => citeNumbers.has(e.evidence_id)) : r.evidence;

  return (
    <article className="animate-rise-in" aria-live="polite">
      {/* Answer card */}
      <section className="card overflow-hidden">
        <header className="flex flex-wrap items-center justify-between gap-3 px-5 sm:px-6 py-3 border-b border-line-soft bg-surface-2/60">
          <span
            className={cx(
              "inline-flex items-center gap-1.5 h-7 px-2.5 rounded-full border text-sm font-medium",
              TONE_TEXT[outcome.tone],
              TONE_PILL[outcome.tone]
            )}
          >
            {OUTCOME_ICON[outcome.kind]}
            {outcome.label}
          </span>
          <span className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-fg-3">
            <span className="inline-flex items-center gap-1">
              <CalendarClock className="w-3.5 h-3.5" />
              {ref ? `As of ${fmtDate(ref)}` : "Whole corpus"}
            </span>
            <span className="num">{fmtMs(r.latency_ms)}</span>
            <span className="num">
              {r.retrieval_rounds} retrieval round{r.retrieval_rounds === 1 ? "" : "s"}
            </span>
            {r.decision === "answer" && (
              <span className="num" title="Model-reported confidence">
                Confidence {fmtPct(r.confidence, 0)}
              </span>
            )}
          </span>
        </header>

        <div className="px-5 sm:px-8 py-6 sm:py-7">
          <h2 className="text-sm text-fg-2 leading-snug max-w-[80ch]">{r.query}</h2>
          <p className="font-serif text-[1.5rem] sm:text-[1.625rem] leading-[1.45] text-fg mt-3 max-w-[58ch]">
            {r.answer}
            {r.decision === "answer" &&
              r.citations.map((c, i) => (
                <sup key={c.evidence_id} className="ml-0.5">
                  <button
                    className="font-sans text-[0.55em] font-medium text-accent hover:underline px-px"
                    onClick={() => {
                      setTab("evidence");
                      setShowCitedOnly(true);
                    }}
                    title={c.text_snippet ?? c.evidence_id}
                  >
                    [{i + 1}]
                  </button>
                </sup>
              ))}
          </p>
          <p className="text-sm text-fg-2 mt-4 max-w-[72ch]">{outcome.explanation}</p>
          {r.reasoning_summary && (
            <p className="text-sm text-fg-3 mt-2 max-w-[72ch]">
              <span className="text-fg-2">Model rationale:</span> {r.reasoning_summary}
            </p>
          )}
          {outcome.kind === "insufficient" &&
            (r.sufficiency.missing_entities.length > 0 || r.sufficiency.missing_relationships.length > 0) && (
              <div className="mt-3 text-sm text-fg-2 space-y-1">
                {r.sufficiency.missing_entities.length > 0 && (
                  <div>
                    Missing entities: <span className="font-mono-num text-xs">{r.sufficiency.missing_entities.join(", ")}</span>
                  </div>
                )}
                {r.sufficiency.missing_relationships.length > 0 && (
                  <div>
                    Missing relationships:{" "}
                    <span className="font-mono-num text-xs">{r.sufficiency.missing_relationships.join(", ")}</span>
                  </div>
                )}
              </div>
            )}
        </div>

        {/* Numbered citations with filing metadata */}
        {r.citations.length > 0 && (
          <div className="border-t border-line-soft px-5 sm:px-6 py-5">
            <h3 className="text-xs font-medium text-fg-2 mb-3">
              Sources cited <span className="num text-fg-3">{r.citations.length}</span>
            </h3>
            <ol className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              {r.citations.map((c, i) => {
                const f = c.document_id ? filings.get(c.document_id) : undefined;
                return (
                  <li key={c.evidence_id} className="rounded-lg border border-line-soft bg-surface-2 p-3.5 flex gap-3">
                    <span className="w-6 h-6 rounded-md bg-accent/15 text-accent font-mono-num text-xs flex items-center justify-center shrink-0">
                      {i + 1}
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-1.5 text-xs">
                        {f ? (
                          <>
                            <span className="font-mono-num text-fg">{f.company}</span>
                            <span className="text-fg-2">Form {f.document_type}</span>
                            <span className="text-fg-3 num">filed {fmtDate(f.publication_date)}</span>
                          </>
                        ) : (
                          <span className="font-mono-num text-fg-3 truncate">{c.document_id ?? c.evidence_id}</span>
                        )}
                        {c.source_type && <SourceTypeBadge type={c.source_type} />}
                      </div>
                      {c.text_snippet && <p className="text-sm text-fg-2 line-clamp-3 mt-1.5 leading-relaxed">“{c.text_snippet.trim()}…”</p>}
                      <div className="flex flex-wrap gap-3 mt-2 text-xs">
                        {c.document_id && (
                          <button className="text-accent hover:underline" onClick={() => onOpenFiling(c.document_id!, c.chunk_id)}>
                            Read in context
                          </button>
                        )}
                        {f?.source_url && (
                          <a href={f.source_url} target="_blank" rel="noopener noreferrer" className="text-accent hover:underline inline-flex items-center gap-1">
                            SEC filing <ExternalLink className="w-3 h-3" />
                          </a>
                        )}
                      </div>
                    </div>
                  </li>
                );
              })}
            </ol>
          </div>
        )}
      </section>

      {/* Supporting material */}
      <div className="card mt-5 px-4 sm:px-5 pb-5">
        <Tabs
          label="Supporting material"
          value={tab}
          onChange={setTab}
          options={[
            { value: "evidence", label: "Evidence", count: r.evidence.length },
            { value: "paths", label: "Graph paths", count: paths.length },
            { value: "checks", label: "Checks" },
            { value: "sources", label: "Sources", count: r.sources.length },
          ]}
        />

        {tab === "evidence" && (
          <div className="mt-3">
            <div className="flex items-center justify-between gap-3 mb-2 text-xs text-fg-3">
              <span>
                Ranked after hybrid retrieval, RRF fusion and reranking. {r.citations.length} of {r.evidence.length} cited.
              </span>
              <label className="inline-flex items-center gap-1.5 cursor-pointer select-none">
                <input
                  type="checkbox"
                  className="accent-[rgb(var(--accent-strong))]"
                  checked={showCitedOnly}
                  onChange={(e) => setShowCitedOnly(e.target.checked)}
                />
                Cited only
              </label>
            </div>
            {evidenceShown.length === 0 ? (
              <EmptyState title="No evidence items">
                {r.evidence.length === 0 ? "The pipeline returned no evidence for this question." : "No cited items."}
              </EmptyState>
            ) : (
              <ul className="border border-line-soft rounded-md bg-surface overflow-hidden">
                {evidenceShown.map((ev) => (
                  <EvidenceRow
                    key={ev.evidence_id}
                    ev={ev}
                    citeNo={citeNumbers.get(ev.evidence_id)}
                    filing={ev.document_id ? filings.get(ev.document_id) : undefined}
                    referenceTime={ref ?? null}
                    onOpenFiling={() => ev.document_id && onOpenFiling(ev.document_id, ev.chunk_id)}
                  />
                ))}
              </ul>
            )}
          </div>
        )}

        {tab === "paths" && (
          <div className="mt-3">
            {paths.length === 0 ? (
              <EmptyState title="No graph paths">The graph retriever did not contribute a path for this question.</EmptyState>
            ) : (
              <ol className="space-y-2">
                {paths.map((p, i) => (
                  <li key={i} className="flex flex-wrap items-center gap-1.5 text-sm bg-surface border border-line-soft rounded px-3 py-2">
                    <span className="font-mono-num text-xs text-fg-3 w-6">{i + 1}</span>
                    {p.map((s, j) => (
                      <React.Fragment key={j}>
                        {j === 0 && <span className="text-fg">{labels[s.source] ?? s.source}</span>}
                        <span
                          className="inline-flex items-center gap-1 text-xs px-1.5 h-5 rounded-sm border"
                          style={{ color: relColor(relFamily(s.relationship)), borderColor: relColor(relFamily(s.relationship), 0.35) }}
                        >
                          {relLabel(s.relationship)} →
                        </span>
                        <span className="text-fg">{labels[s.target] ?? <span className="font-mono-num text-xs">{s.target}</span>}</span>
                      </React.Fragment>
                    ))}
                    <span className="ml-auto text-xs text-fg-3">
                      {p.length} hop{p.length > 1 ? "s" : ""}
                    </span>
                  </li>
                ))}
              </ol>
            )}
          </div>
        )}

        {tab === "checks" && (
          <div className="mt-4 grid grid-cols-1 lg:grid-cols-3 gap-6">
            <section>
              <h3 className="text-sm font-semibold mb-1">Evidence sufficiency</h3>
              <p className="text-xs text-fg-3 mb-2">Gate applied before generation.</p>
              <KeyValue
                dense
                items={[
                  { label: "Result", value: r.sufficiency.sufficient ? "Sufficient" : "Insufficient" },
                  { label: "Hops required", value: <span className="num">{r.sufficiency.required_hops}</span> },
                  { label: "Hops observed", value: <span className="num">{r.sufficiency.observed_hops}</span> },
                  { label: "Evidence items", value: <span className="num">{r.sufficiency.evidence_count}</span> },
                  { label: "Provenance complete", value: r.sufficiency.provenance_complete ? "Yes" : "No" },
                ]}
              />
              <p className="text-xs text-fg-2 mt-2">{r.sufficiency.reason}</p>
            </section>
            <section>
              <h3 className="text-sm font-semibold mb-1">Temporal validity</h3>
              <p className="text-xs text-fg-3 mb-2">Point-in-time constraint on evidence.</p>
              <KeyValue
                dense
                items={[
                  { label: "Reference time", value: ref ? fmtDateTime(ref) : "None", mono: !!ref },
                  { label: "Pipeline result", value: r.temporal_validation.temporal_valid ? "Valid" : "Invalid" },
                  { label: "Context items", value: <span className="num">{r.temporal_validation.context_items_count}</span> },
                  {
                    label: "Returned evidence after reference",
                    value:
                      afterRefCount == null ? (
                        "Not applicable"
                      ) : (
                        <span className={cx("num", afterRefCount > 0 && "text-neg")}>
                          {afterRefCount} of {r.evidence.length}
                        </span>
                      ),
                    hint: "Counted in the browser from returned timestamps",
                  },
                ]}
              />
              <Note className="mt-3">
                The generator raises an error if any evidence post-dates the reference time, so a returned answer implies none
                reached generation. The API’s <code className="font-mono-num">future_items_count</code> and{" "}
                <code className="font-mono-num">feir</code> fields are constants, so they are not shown as measurements. This
                check covers retrieved evidence only, not knowledge the model learned in pre-training.
              </Note>
            </section>
            <section>
              <h3 className="text-sm font-semibold mb-1">Answer validation</h3>
              <p className="text-xs text-fg-3 mb-2">Post-generation checks on citations and format.</p>
              <KeyValue
                dense
                items={[
                  { label: "Valid", value: v?.is_valid == null ? "Not reported" : v.is_valid ? "Yes" : "No" },
                  { label: "Errors", value: <span className="num">{v?.errors?.length ?? 0}</span> },
                  { label: "Warnings", value: <span className="num">{v?.warnings?.length ?? 0}</span> },
                  { label: "Cited evidence ids", value: <span className="num">{v?.cited_evidence_ids?.length ?? r.citations.length}</span> },
                ]}
              />
              {[...(v?.errors ?? []), ...(v?.warnings ?? [])].length > 0 && (
                <ul className="mt-2 space-y-1 text-xs text-fg-2 list-disc pl-4">
                  {[...(v?.errors ?? []), ...(v?.warnings ?? [])].map((m, i) => (
                    <li key={i}>{m}</li>
                  ))}
                </ul>
              )}
            </section>
          </div>
        )}

        {tab === "sources" && (
          <div className="mt-3">
            {r.sources.length === 0 ? (
              <EmptyState title="No source documents" />
            ) : (
              <ul className="border border-line-soft rounded-md bg-surface divide-y divide-line-soft">
                {r.sources.map((s) => {
                  const f = filings.get(s);
                  return (
                    <li key={s} className="flex flex-wrap items-center gap-x-4 gap-y-1 px-4 py-2.5 text-sm">
                      <span className="font-mono-num text-xs text-fg-3 w-56 truncate">{s}</span>
                      <span className="flex-1 min-w-0 text-fg">
                        {f ? `${f.company} ${f.document_type ?? ""}, filed ${fmtDate(f.publication_date)}` : "Filing metadata unavailable"}
                      </span>
                      <button className="text-xs text-accent hover:underline" onClick={() => onOpenFiling(s)}>
                        Open
                      </button>
                      {f?.source_url && (
                        <a href={f.source_url} target="_blank" rel="noopener noreferrer" className="text-xs text-accent inline-flex items-center gap-1 hover:underline">
                          SEC <ExternalLink className="w-3 h-3" />
                        </a>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}
            <p className="text-xs text-fg-3 mt-2">
              Answered {fmtDateTime(r.timestamp)} in {r.mode === "benchmark" ? "Benchmark (Phase 7 configuration)" : r.mode === "assist" ? "Assist" : "Benchmark"} mode. Filing
              metadata joined from the corpus manifest by document id.
            </p>
          </div>
        )}
      </div>

      {/* Historical comparison */}
      {demo && (
        <section className="card mt-5 p-5">
          <h3 className="text-sm font-semibold">Recorded Phase 7 result for {demo.question_id}</h3>
          <p className="text-xs text-fg-3 mt-0.5">Historical benchmark run, shown for comparison with this live run.</p>
          <KeyValue
            className="mt-3 max-w-3xl"
            items={[
              { label: "Reference answer", value: demo.ground_truth_answer },
              { label: "Recorded answer", value: demo.cached_phase7_result.generated_answer },
              {
                label: "Recorded decision",
                value: `${humanize(demo.cached_phase7_result.decision)}, token F1 ${fmtNum(demo.cached_phase7_result.token_f1, 4)}, ${demo.cached_phase7_result.latency_sec} s`,
              },
              ...(demo.unanswerability_rationale ? [{ label: "Why unanswerable", value: demo.unanswerability_rationale }] : []),
            ]}
          />
        </section>
      )}
    </article>
  );
}

/* ------------------------------------------------------------------ */
/* View                                                                */
/* ------------------------------------------------------------------ */

function contextSuggestions(name: string): string[] {
  return [
    `What risks did ${name} disclose in its most recent filings?`,
    `Which companies does ${name} compete with?`,
    `What acquisitions has ${name} disclosed?`,
  ];
}


/* ------------------------------------------------------------------ */
/* Live SEC fetch: when the indexed filings fall short, pull the company's latest filings from EDGAR. */

const attemptedFetches = new Set<string>();

function needsMoreEvidence(r: QueryResponse): boolean {
  if (r.mode !== "assist") return false;
  const coverage = (r.validation_status as Record<string, unknown> | undefined)?.coverage;
  return r.decision === "refuse" || r.insufficient_evidence || coverage === "partial";
}

function LiveFetchPanel({
  run,
  autoFetch,
  onAutoFetchChange,
  onRerun,
}: {
  run: Run & { result: QueryResponse };
  autoFetch: boolean;
  onAutoFetchChange: (v: boolean) => void;
  onRerun: () => void;
}) {
  type Phase = "resolving" | "none" | "offer" | "fetching" | "added" | "nothing-new" | "error";
  const [phase, setPhase] = useState<Phase>("resolving");
  const [company, setCompany] = useState<LiveCompany | null>(null);
  const [result, setResult] = useState<LiveFetchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const ctrl = useRef<AbortController | null>(null);

  const doFetch = async (c: LiveCompany) => {
    attemptedFetches.add(`${c.ticker}|${run.query}`);
    ctrl.current?.abort();
    const ac = new AbortController();
    ctrl.current = ac;
    setPhase("fetching");
    try {
      const res = await api.fetchLiveFilings({ ticker: c.ticker }, ac.signal);
      setResult(res);
      if (res.added.length > 0) {
        setPhase("added");
        onRerun();
      } else {
        setPhase("nothing-new");
      }
    } catch (e) {
      if (isAbortError(e)) return;
      setError(e instanceof ApiError ? e.message : String(e));
      setPhase("error");
    }
  };

  useEffect(() => {
    const ac = new AbortController();
    setPhase("resolving");
    setResult(null);
    setError(null);
    api
      .resolveLiveCompany(run.query, ac.signal)
      .then(({ company: c }) => {
        if (!c) return setPhase("none");
        setCompany(c);
        const key = `${c.ticker}|${run.query}`;
        if (autoFetch && !attemptedFetches.has(key)) doFetch(c);
        else setPhase(attemptedFetches.has(key) ? "nothing-new" : "offer");
      })
      .catch((e) => !isAbortError(e) && setPhase("none"));
    return () => {
      ac.abort();
      ctrl.current?.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run.id]);

  if (phase === "resolving" || phase === "none" || !company) return null;
  const name = company.name.replace(/\b(INC|CORP|CORPORATION|CO|LTD|PLC)\b\.?/gi, (m) => m.charAt(0) + m.slice(1).toLowerCase());

  return (
    <div className="card p-5 mb-4 border-accent/30 relative overflow-hidden" role="status" aria-live="polite">
      <span aria-hidden className="absolute -right-16 -top-16 w-48 h-48 rounded-full bg-accent-strong/15 blur-3xl" />
      <div className="relative flex flex-wrap items-start gap-4">
        <span className="w-10 h-10 rounded-full bg-accent-strong/15 text-accent flex items-center justify-center shrink-0">
          {phase === "fetching" ? <Spinner className="w-4 h-4" /> : <CloudDownload className="w-4 h-4" />}
        </span>
        <div className="flex-1 min-w-[220px]">
          <div className="text-md font-medium text-fg">
            {phase === "fetching"
              ? `Fetching ${name}'s latest filings from SEC EDGAR…`
              : phase === "added"
              ? `Added ${result?.added.length} filing${result?.added.length === 1 ? "" : "s"} for ${name}. Asking again…`
              : phase === "nothing-new"
              ? `${name}'s latest filings are already indexed`
              : phase === "error"
              ? "Could not fetch filings from SEC EDGAR"
              : company.in_research_corpus
              ? `Check SEC EDGAR for newer ${name} filings?`
              : `${name} (${company.ticker}) isn't in the indexed filings`}
          </div>
          <p className="text-sm text-fg-2 mt-1 leading-relaxed">
            {phase === "fetching"
              ? "Downloading the latest 10-K, two 10-Qs and two 8-Ks, then splitting and indexing them. This usually takes 15–40 seconds."
              : phase === "added"
              ? (result?.added ?? []).map((f) => `${f.form} ${f.filing_date ?? ""}`.trim()).join(" · ")
              : phase === "nothing-new"
              ? "There was nothing newer to add, so this answer reflects everything FinGraph has for this company."
              : phase === "error"
              ? error
              : "FinGraph can download the company's latest 10-K, 10-Q and 8-K filings from SEC EDGAR and answer again from them. Fetched filings are kept separate from the research corpus and are not used in Benchmark mode."}
          </p>
          <label className="mt-3 inline-flex items-center gap-2 text-xs text-fg-3 cursor-pointer select-none">
            <input
              type="checkbox"
              className="accent-[rgb(var(--accent-strong))]"
              checked={autoFetch}
              onChange={(e) => onAutoFetchChange(e.target.checked)}
            />
            Fetch automatically when the indexed filings fall short
          </label>
        </div>
        {(phase === "offer" || phase === "error") && (
          <Button variant="primary" icon={<CloudDownload className="w-4 h-4" />} onClick={() => doFetch(company)}>
            {phase === "error" ? "Try again" : "Fetch from SEC"}
          </Button>
        )}
      </div>
    </div>
  );
}

export function AskFinGraphView({ onNavigate }: { onNavigate: (t: NavTab) => void }) {
  const demos = useApi((s) => api.getDemoQuestions(s), []);
  const { entity } = useResearchContext();
  const [query, setQuery] = useState("");
  const [referenceDate, setReferenceDate] = useState("");
  const [runs, setRuns] = useState<Run[]>([]);
  const [activeRunId, setActiveRunId] = useState<number | null>(null);
  const [running, setRunning] = useState(false);
  const [mode, setModeState] = useState<QueryMode>("assist");
  const [autoFetch, setAutoFetchState] = useState(true);
  useEffect(() => {
    try {
      if (localStorage.getItem("fingraph-auto-fetch") === "0") setAutoFetchState(false);
    } catch {
      /* ignore */
    }
  }, []);
  const setAutoFetch = (v: boolean) => {
    setAutoFetchState(v);
    try {
      localStorage.setItem("fingraph-auto-fetch", v ? "1" : "0");
    } catch {
      /* ignore */
    }
  };
  useEffect(() => {
    try {
      const m = localStorage.getItem("fingraph-ask-mode");
      if (m === "assist" || m === "benchmark") setModeState(m);
    } catch {
      /* ignore */
    }
  }, []);
  const setMode = (m: QueryMode) => {
    setModeState(m);
    try {
      localStorage.setItem("fingraph-ask-mode", m);
    } catch {
      /* ignore */
    }
  };
  const ctrlRef = useRef<AbortController | null>(null);
  const textRef = useRef<HTMLTextAreaElement>(null);
  const elapsed = useElapsed(running);

  useEffect(() => () => ctrlRef.current?.abort(), []);

  // Prefill requests from other views (e.g. the Overview "Run it live" action). Never auto-runs.
  useEffect(() => {
    const apply = (d: { question?: string; referenceTime?: string | null }) => {
      if (!d?.question) return;
      setQuery(d.question);
      setReferenceDate(d.referenceTime ? fmtDate(d.referenceTime) : "");
      window.setTimeout(() => textRef.current?.focus(), 50);
    };
    try {
      const raw = sessionStorage.getItem("fingraph-ask-prefill");
      if (raw) {
        sessionStorage.removeItem("fingraph-ask-prefill");
        apply(JSON.parse(raw));
      }
    } catch {
      /* ignore */
    }
    const onPrefill = (e: Event) => {
      try {
        sessionStorage.removeItem("fingraph-ask-prefill");
      } catch {
        /* ignore */
      }
      apply((e as CustomEvent).detail);
    };
    window.addEventListener("fingraph:ask-prefill", onPrefill);
    return () => window.removeEventListener("fingraph:ask-prefill", onPrefill);
  }, []);

  const refTime = referenceDate ? toReferenceTime(referenceDate) : null;
  const dateInvalid = referenceDate !== "" && refTime === null;
  const trimmed = query.trim();
  const tooShort = trimmed.length > 0 && trimmed.length < 3;
  const canRun = !running && trimmed.length >= 3 && trimmed.length <= 1000 && !dateInvalid;

  const activeRun = runs.find((r) => r.id === activeRunId) ?? null;
  const matchingDemo = (q: string) => demos.data?.questions.find((d) => d.question.trim() === q.trim()) ?? null;

  const runQuery = async (q: string, refDate: string, rt: string | null) => {
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    const id = Date.now();
    const run: Run = { id, query: q, referenceDate: refDate, startedAt: id, result: null, error: null };
    setRuns((rs) => [run, ...rs].slice(0, 12));
    setActiveRunId(id);
    setRunning(true);
    try {
      // The query is sent verbatim; mode selects the live assist settings or the benchmarked config.
      const result = await api.submitQuery({ query: q, reference_time: rt, top_k: 10, candidate_pool_size: 20, mode }, ctrl.signal);
      setRuns((rs) => rs.map((r) => (r.id === id ? { ...r, result } : r)));
    } catch (e) {
      if (isAbortError(e)) {
        setRuns((rs) => rs.filter((r) => r.id !== id));
        setActiveRunId(null);
      } else {
        const err = e instanceof ApiError ? e : new ApiError(String(e), 0);
        setRuns((rs) => rs.map((r) => (r.id === id ? { ...r, error: err } : r)));
      }
    } finally {
      setRunning(false);
      ctrlRef.current = null;
    }
  };

  const execute = async () => {
    if (!canRun) return; // duplicate-submit guard
    await runQuery(trimmed, referenceDate, refTime);
  };

  /** Re-ask a previous question (after new filings were fetched). */
  const rerun = (r: Run) => {
    if (running) return;
    runQuery(r.query, r.referenceDate, r.referenceDate ? toReferenceTime(r.referenceDate) : null);
  };

  const cancel = () => ctrlRef.current?.abort();

  const loadDemo = (d: DemoQuestionItem) => {
    setQuery(d.question);
    setReferenceDate(d.reference_time ? fmtDate(d.reference_time) : "");
    textRef.current?.focus();
  };

  const openFiling = (documentId: string, chunkId?: string | null) => {
    try {
      sessionStorage.setItem("fingraph-open-filing", JSON.stringify({ documentId, chunkId: chunkId ?? null }));
    } catch {
      /* ignore */
    }
    window.dispatchEvent(new CustomEvent("fingraph:open-filing", { detail: { documentId, chunkId: chunkId ?? null } }));
    onNavigate("evidence");
  };

  return (
    <div className="max-w-[1280px] mx-auto px-4 sm:px-8 py-6 sm:py-8 grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_300px] gap-x-8 gap-y-8">
      {/* Main column */}
      <div className="min-w-0">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            execute();
          }}
          className="card focus-within:border-accent/60 transition-colors overflow-hidden"
        >
          <div className="px-5 pt-4 flex items-center justify-between gap-3">
            <label htmlFor="ask-query" className="text-sm font-semibold text-fg">
              Ask a question about the filings
            </label>
            <span className="text-2xs text-fg-3 hidden sm:inline">Ctrl or ⌘ + Enter to run</span>
          </div>
          <Textarea
            id="ask-query"
            ref={textRef}
            rows={3}
            value={query}
            maxLength={1000}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                execute();
              }
            }}
            placeholder="Ask about acquisitions, risks, suppliers or disclosures in the SEC filings…"
            className="border-0 focus:ring-0 bg-transparent text-lg px-5 pt-2 min-h-[96px]"
          />
          <div className="flex flex-wrap items-end gap-3 px-5 pb-4 pt-3 border-t border-line-soft bg-surface-2/50">
            <div>
              <FieldLabel htmlFor="ask-ref" hint={dateInvalid ? <span className="text-neg">Invalid date</span> : undefined}>
                Answer as of
              </FieldLabel>
              <div className="flex items-center gap-1">
                <Input
                  id="ask-ref"
                  type="date"
                  inputSize="sm"
                  value={referenceDate}
                  onChange={(e) => setReferenceDate(e.target.value)}
                  className="w-40"
                  aria-describedby="ask-ref-help"
                />
                {referenceDate && (
                  <Button type="button" size="sm" variant="ghost" onClick={() => setReferenceDate("")}>
                    Clear
                  </Button>
                )}
              </div>
            </div>
            <div>
              <FieldLabel>Answer mode</FieldLabel>
              <Segmented
                label="Answer mode"
                value={mode}
                onChange={setMode}
                options={[
                  { value: "assist", label: "Assist" },
                  { value: "benchmark", label: "Benchmark" },
                ]}
              />
            </div>
            <p id="ask-ref-help" className="text-xs text-fg-3 flex-1 min-w-[180px] pb-1.5">
              {refTime
                ? `Only evidence available by ${referenceDate} 00:00 UTC is used.`
                : "No date: the whole corpus is searchable."}{" "}
              {mode === "assist"
                ? "Assist reads full passages and gives partial, cited answers where it can."
                : "Benchmark runs the evaluated Phase 7 configuration exactly."}
            </p>
            <div className="flex items-center gap-2 ml-auto">
              {running ? (
                <Button type="button" onClick={cancel} icon={<Square className="w-3.5 h-3.5" />}>
                  Cancel
                </Button>
              ) : null}
              <Button type="submit" variant="primary" className="h-9 px-5" disabled={!canRun} loading={running}>
                {running ? `Running ${(elapsed / 1000).toFixed(0)} s` : "Run query"}
              </Button>
            </div>
          </div>
          <ProgressBar active={running} />
        </form>
        <div className="flex flex-wrap justify-between gap-2 mt-2 text-xs text-fg-3">
          <span>{tooShort ? "Questions need at least 3 characters." : "Answers use only the indexed filings and knowledge graph."}</span>
          <span className="num">{query.length} / 1000</span>
        </div>

        {/* Context scope (opt-in, never applied silently) */}
        {entity && (
          <div className="mt-4 flex flex-wrap items-center gap-2 text-sm">
            <Crosshair className="w-3.5 h-3.5 text-accent" />
            <span className="text-fg-2">
              Research context: <span className="text-fg">{entity.name}</span>. It is not added to your question.
            </span>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setQuery((q) => (q.includes(entity.name) ? q : `${q.trim()}${q.trim() ? " " : ""}${entity.name}`));
                textRef.current?.focus();
              }}
            >
              Insert name
            </Button>
          </div>
        )}

        {/* Body */}
        <div className="mt-6">
          {activeRun?.result ? (
            <>
              {needsMoreEvidence(activeRun.result) && (
                <LiveFetchPanel
                  key={activeRun.id}
                  run={activeRun as Run & { result: QueryResponse }}
                  autoFetch={autoFetch}
                  onAutoFetchChange={setAutoFetch}
                  onRerun={() => rerun(activeRun)}
                />
              )}
              <Result run={activeRun as Run & { result: QueryResponse }} demo={matchingDemo(activeRun.query)} onOpenFiling={openFiling} />
            </>
          ) : activeRun?.error ? (
            <ErrorState
              title={activeRun.error.isNetwork ? "Backend unreachable" : `Query failed (HTTP ${activeRun.error.status})`}
              error={activeRun.error}
              action={
                <Button size="sm" onClick={execute} disabled={!canRun}>
                  Retry
                </Button>
              }
            />
          ) : running ? (
            <div className="card p-6" role="status">
              <div className="flex items-center gap-2 text-sm text-fg">
                <Spinner className="w-4 h-4 text-accent" /> Running the pipeline
              </div>
              <ol className="mt-4 grid grid-cols-1 sm:grid-cols-5 gap-2 text-xs text-fg-2">
                {["Entity linking", "Hybrid retrieval", "Fusion and reranking", "Sufficiency gate", "Generation and validation"].map(
                  (st, i) => (
                    <li key={st} className="rounded-md border border-line-soft bg-surface-2 px-3 py-2">
                      <span className="font-mono-num text-fg-3 mr-1.5">{i + 1}</span>
                      {st}
                    </li>
                  )
                )}
              </ol>
              <div className="mt-6 space-y-2.5">
                <Skeleton className="h-6 w-4/5" />
                <Skeleton className="h-6 w-3/5" />
              </div>
              <p className="text-xs text-fg-3 mt-5 max-w-[64ch]">
                These stages run on the server and are listed for orientation, not as live progress. Recorded Phase 7 runs averaged about
                17 seconds per question.
              </p>
            </div>
          ) : (
            <div>
              <div className="flex items-end justify-between gap-3 mb-3">
                <div>
                  <h2 className="text-xl font-medium tracking-tight">Try an audited example</h2>
                  <p className="text-sm text-fg-2 mt-0.5">
                    Benchmark questions with recorded Phase 7 results. Choosing one fills the form; run it to see the live answer.
                  </p>
                </div>
              </div>
              {demos.error ? (
                <ErrorState compact title="Examples unavailable" error={demos.error} />
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {(demos.data?.questions ?? []).map((d) => (
                    <button
                      key={d.question_id}
                      onClick={() => loadDemo(d)}
                      className="card card-interactive text-left p-4 flex flex-col"
                    >
                      <span className="flex flex-wrap items-center gap-1.5">
                        <Badge mono>{d.question_id}</Badge>
                        <span className="text-xs text-fg-3">{humanize(d.category)}</span>
                      </span>
                      <span className="block text-sm text-fg mt-2.5 leading-snug">{d.question}</span>
                      <span className="flex flex-wrap items-center gap-2 mt-auto pt-3 text-2xs text-fg-3">
                        {d.answerable ? (
                          <span className="text-pos">Expects an answer</span>
                        ) : (
                          <span className="text-warn">Expects a refusal</span>
                        )}
                        {d.reference_time && <span className="font-mono-num">as of {fmtDate(d.reference_time)}</span>}
                        <span>{d.expected_hops} hop{d.expected_hops > 1 ? "s" : ""}</span>
                      </span>
                    </button>
                  ))}
                  {demos.loading && [0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-28" />)}
                </div>
              )}
              <div className="card mt-5 p-5 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
                {[
                  ["text-pos", "Supported answer", "Passed sufficiency, validation and provenance checks."],
                  ["text-info", "Partially supported", "Answered, but a check flagged a gap."],
                  ["text-warn", "Insufficient evidence", "The evidence did not support an answer, so it declined."],
                  ["text-neg", "Validation issue or failure", "The answer failed validation, or the model could not run."],
                ].map(([tone, t, d]) => (
                  <div key={t}>
                    <div className={cx("font-medium text-sm", tone)}>{t}</div>
                    <div className="text-fg-3 mt-0.5 leading-relaxed">{d}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Rail */}
      <aside className="space-y-8 xl:pt-1">
        {entity && (
          <section>
            <h2 className="text-sm font-semibold mb-2">For {entity.name}</h2>
            <ul className="space-y-1">
              {contextSuggestions(entity.name).map((q) => (
                <li key={q}>
                  <button
                    onClick={() => {
                      setQuery(q);
                      textRef.current?.focus();
                    }}
                    className="text-left text-sm text-fg-2 hover:text-fg leading-snug py-1"
                  >
                    {q}
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}

        {activeRun && (
        <section>
          <h2 className="text-sm font-semibold">Audited examples</h2>
          <p className="text-xs text-fg-3 mt-0.5 mb-2">Benchmark questions with recorded Phase 7 results. Loading one fills the form.</p>
          {demos.error ? (
            <ErrorState compact title="Examples unavailable" error={demos.error} />
          ) : (
            <ul className="divide-y divide-line-soft border-y border-line-soft">
              {(demos.data?.questions ?? []).map((d) => (
                <li key={d.question_id}>
                  <button onClick={() => loadDemo(d)} className="w-full text-left py-2.5 group">
                    <span className="flex items-center gap-2 text-xs text-fg-3">
                      <span className="font-mono-num text-fg-2">{d.question_id}</span>
                      <span>{humanize(d.category)}</span>
                      {!d.answerable && <Badge tone="warn">Expects refusal</Badge>}
                      {d.reference_time && <span className="font-mono-num">as of {fmtDate(d.reference_time)}</span>}
                    </span>
                    <span className="block text-sm text-fg-2 group-hover:text-fg mt-0.5 leading-snug">{d.question}</span>
                  </button>
                </li>
              ))}
              {demos.loading && <li className="py-3 text-sm text-fg-3">Loading…</li>}
            </ul>
          )}
        </section>
        )}

        {runs.length > 0 && (
          <section>
            <h2 className="text-sm font-semibold flex items-center gap-1.5">
              <History className="w-3.5 h-3.5 text-fg-3" /> This session
            </h2>
            <ul className="mt-2 space-y-0.5">
              {runs.map((r) => (
                <li key={r.id}>
                  <button
                    onClick={() => setActiveRunId(r.id)}
                    className={cx(
                      "w-full text-left px-2 py-1.5 rounded text-sm",
                      r.id === activeRunId ? "bg-surface-2 text-fg" : "text-fg-2 hover:bg-surface-2/60"
                    )}
                  >
                    <span className="line-clamp-1">{r.query}</span>
                    <span className="text-2xs text-fg-3">
                      {r.result
                        ? `${classify(r.result).label}, ${fmtMs(r.result.latency_ms)}`
                        : r.error
                        ? "Failed"
                        : "Running…"}
                      {r.referenceDate && ` · as of ${r.referenceDate}`}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}

        <section className="text-xs text-fg-3 leading-relaxed">
          <h2 className="text-sm font-semibold text-fg mb-1">About these answers</h2>
          <p>
            Retrieval is limited to the indexed filings and, when a date is set, to filings public by that date. The model may still
            know later facts from pre-training; citations show what each answer actually rests on.
          </p>
          <button className="mt-2 text-accent hover:underline inline-flex items-center gap-1" onClick={() => onNavigate("architecture")}>
            How the pipeline works <ArrowUpRight className="w-3 h-3" />
          </button>
        </section>
      </aside>
    </div>
  );
}
