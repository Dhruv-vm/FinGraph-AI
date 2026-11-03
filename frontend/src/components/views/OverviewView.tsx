"use client";

import React, { useMemo, useState } from "react";
import {
  ArrowRight,
  BookOpenText,
  Building2,
  ChevronDown,
  Crosshair,
  Database,
  FileStack,
  FileText,
  Layers,
  MessageSquareText,
  Network,
  Search,
  ShieldCheck,
  TriangleAlert,
  Waypoints,
} from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import { useCorpus } from "@/lib/corpus";
import { NavTab } from "@/lib/navigation";
import { HealthState, overallStatus } from "@/components/Header";
import { useResearchContext, issuerEntity } from "@/lib/research-context";
import { CORPUS_ISSUERS, fmtDate, fmtInt, fmtNum, fmtPct } from "@/lib/format";
import { CompanyCoverage, ComponentHealth } from "@/lib/types";
import {
  Badge,
  Button,
  ErrorState,
  IconButton,
  Input,
  KeyValue,
  Skeleton,
  Status,
  StatusDot,
  STATUS_LABEL,
  cx,
} from "@/components/ui";

/** Ask another view to prefill the Ask composer (consumed by AskFinGraphView). */
export function requestAsk(question: string, referenceTime: string | null) {
  try {
    sessionStorage.setItem("fingraph-ask-prefill", JSON.stringify({ question, referenceTime }));
  } catch {
    /* ignore */
  }
  window.dispatchEvent(new CustomEvent("fingraph:ask-prefill", { detail: { question, referenceTime } }));
}

/* ------------------------------------------------------------------ */

function KpiCard({
  icon,
  label,
  value,
  detail,
  loading,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  detail?: React.ReactNode;
  loading?: boolean;
}) {
  return (
    <div className="card p-5 flex flex-col">
      <div className="flex items-center justify-between">
        <span className="text-sm text-fg-2">{label}</span>
        <span className="w-8 h-8 rounded-lg bg-accent/10 text-accent flex items-center justify-center">{icon}</span>
      </div>
      {loading ? (
        <Skeleton className="h-9 w-24 mt-3" />
      ) : (
        <div className="num text-[2rem] leading-none font-semibold tracking-tight mt-3">{value}</div>
      )}
      {detail && <div className="text-xs text-fg-3 mt-2.5 num">{detail}</div>}
    </div>
  );
}

const PIPELINE: { name: string; plain: string; icon: React.ReactNode; when: "offline" | "query" }[] = [
  { name: "SEC filings", plain: "Primary 10-K, 10-Q and 8-K filings from EDGAR", icon: <FileStack className="w-4 h-4" />, when: "offline" },
  { name: "Structure-aware chunking", plain: "Split at paragraph boundaries, each passage dated", icon: <Layers className="w-4 h-4" />, when: "offline" },
  { name: "Temporal knowledge graph", plain: "Entities and relationships, each tied to its filing", icon: <Network className="w-4 h-4" />, when: "offline" },
  { name: "Hybrid retrieval", plain: "Vector search and graph paths, fused and reranked", icon: <Search className="w-4 h-4" />, when: "query" },
  { name: "Evidence-grounded QA", plain: "Answer only if evidence suffices, with citations", icon: <ShieldCheck className="w-4 h-4" />, when: "query" },
];

function PerfTile({ value, label, detail, tone }: { value: string; label: string; detail: string; tone?: "pos" | "accent" }) {
  return (
    <div className="rounded-lg bg-surface-2 border border-line-soft p-4">
      <div className={cx("num text-2xl font-semibold tracking-tight", tone === "pos" ? "text-pos" : tone === "accent" ? "text-accent" : "text-fg")}>
        {value}
      </div>
      <div className="text-sm text-fg mt-1.5">{label}</div>
      <div className="text-xs text-fg-3 mt-0.5">{detail}</div>
    </div>
  );
}

function StatusValue({ c }: { c?: ComponentHealth | null }) {
  if (!c) return <span className="text-fg-3">—</span>;
  const s = c.status as Status;
  return (
    <span className="inline-flex items-center gap-2" title={c.message ?? undefined}>
      <StatusDot status={s} />
      {STATUS_LABEL[s] ?? c.status}
    </span>
  );
}

/* ------------------------------------------------------------------ */

export function OverviewView({ health, onNavigate }: { health: HealthState; onNavigate: (t: NavTab) => void }) {
  const corpus = useCorpus();
  const summary = useApi((s) => api.getBenchmarkSummary(s), []);
  const companies = useApi((s) => api.getCompanies(s), []);
  const demos = useApi((s) => api.getDemoQuestions(s), []);
  const plr = useApi((s) => api.getParametricLeakage(s), []);
  const { entity, setEntity } = useResearchContext();
  const [filter, setFilter] = useState("");
  const [detailsOpen, setDetailsOpen] = useState(false);

  const h = health.data;
  const status = overallStatus(health);
  const kgNodes = h?.knowledge_graph.details?.node_count as number | undefined;
  const kgEdges = h?.knowledge_graph.details?.edge_count as number | undefined;
  const points = h?.qdrant.details?.points_count as number | undefined;
  const model = (h?.llm.details?.model as string | undefined) ?? null;

  const corpusData = corpus.data;
  const filingDates = useMemo(() => {
    const ds = (corpusData?.filings.map((f) => f.publication_date).filter(Boolean) as string[] | undefined)?.sort() ?? [];
    return ds.length ? { first: ds[0], last: ds[ds.length - 1] } : null;
  }, [corpusData]);

  const degreeByTicker = useMemo(() => {
    const m = new Map<string, { degree: number; name: string }>();
    companies.data?.companies.forEach((c) => m.set(c.ticker, { degree: c.edge_count, name: c.canonical_name }));
    return m;
  }, [companies.data]);

  const coverageByTicker = useMemo(() => {
    const m = new Map<string, CompanyCoverage>();
    corpusData?.companies.forEach((c) => m.set(c.company, c));
    return m;
  }, [corpusData]);

  const issuers = useMemo(() => {
    const tickers = corpusData?.companies.map((c) => c.company) ?? CORPUS_ISSUERS.map((c) => c.ticker);
    const q = filter.trim().toLowerCase();
    return tickers
      .map((t) => ({
        ticker: t,
        name: degreeByTicker.get(t)?.name ?? CORPUS_ISSUERS.find((c) => c.ticker === t)?.name ?? t,
        degree: degreeByTicker.get(t)?.degree,
        coverage: coverageByTicker.get(t),
      }))
      .filter((r) => !q || r.ticker.toLowerCase().includes(q) || r.name.toLowerCase().includes(q))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [corpusData, degreeByTicker, coverageByTicker, filter]);

  const combined = summary.data?.configurations?.cfg5_combined;
  const p7 = summary.data;
  const example = demos.data?.questions.find((d) => d.question_id === "FSH-07") ?? demos.data?.questions[0];
  const plr83 = plr.data?.phase8_3?.closed_book?.A2_as_of?.recalled_all ?? null;

  const llmDown = h?.llm.status === "unavailable";

  return (
    <div className="max-w-[1240px] mx-auto px-4 sm:px-8 py-6 sm:py-8 space-y-6">
      {/* Hero */}
      <section className="card overflow-hidden bg-accent-wash" aria-labelledby="hero-title">
        <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
          <div className="p-6 sm:p-9">
            <div className="flex flex-wrap items-center gap-2 mb-6">
              <span className="inline-flex items-center gap-2 h-6 px-2.5 rounded-full border border-line bg-surface text-xs text-fg-2">
                <StatusDot status={status} />
                {status === "healthy"
                  ? "All services operational"
                  : health.error
                  ? "Backend offline"
                  : llmDown
                  ? "Language model offline"
                  : STATUS_LABEL[status]}
              </span>
              <span className="text-xs text-fg-3">Temporal GraphRAG research system</span>
            </div>
            <h1 id="hero-title" className="text-[2rem] sm:text-[2.5rem] leading-[1.1] font-semibold tracking-[-0.02em] max-w-[15em]">
              Financial intelligence, grounded in evidence.
            </h1>
            <p className="text-md text-fg-2 mt-4 max-w-[56ch] leading-relaxed">
              FinGraph answers questions about public companies using only their SEC filings. A temporal knowledge graph and hybrid
              GraphRAG retrieval find the evidence, every answer cites its sources, and questions can be asked as of a past date so
              later filings cannot leak in.
            </p>
            <div className="flex flex-wrap gap-2.5 mt-7">
              <Button variant="primary" className="h-10 px-5" onClick={() => onNavigate("ask")} icon={<MessageSquareText className="w-4 h-4" />}>
                Ask FinGraph
              </Button>
              <Button className="h-10 px-5" onClick={() => onNavigate("graph")} icon={<Waypoints className="w-4 h-4" />}>
                Explore knowledge graph
              </Button>
            </div>
          </div>

          {/* What happens when you ask — a real recorded example */}
          <div className="border-t lg:border-t-0 lg:border-l border-line-soft bg-surface/60 p-6 sm:p-8 flex flex-col">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium text-fg-2">What happens when you ask</span>
              <Badge mono>Recorded {example?.question_id ?? ""}</Badge>
            </div>
            {example ? (
              <>
                <div className="mt-4 rounded-lg border border-line-soft bg-surface-2 px-4 py-3">
                  <div className="text-2xs text-fg-3 mb-1">Question</div>
                  <div className="text-sm text-fg leading-snug">{example.question}</div>
                </div>
                <ol className="mt-3 space-y-2 text-xs text-fg-2">
                  <li className="flex gap-2.5">
                    <span className="font-mono-num text-accent w-4">1</span>Links the companies named to graph entities, then retrieves passages and relationships
                  </li>
                  <li className="flex gap-2.5">
                    <span className="font-mono-num text-accent w-4">2</span>Checks the evidence is sufficient before generating anything
                  </li>
                  <li className="flex gap-2.5">
                    <span className="font-mono-num text-accent w-4">3</span>Answers from the evidence and cites it
                  </li>
                </ol>
                <div className="mt-4 rounded-lg border border-pos/30 bg-pos/[0.06] px-4 py-3">
                  <div className="flex items-center gap-1.5 text-2xs text-pos mb-1">
                    <ShieldCheck className="w-3.5 h-3.5" /> Answered with {example.cached_phase7_result.citations.length} citations
                  </div>
                  <div className="font-serif text-lg text-fg leading-snug">{example.cached_phase7_result.generated_answer}</div>
                </div>
                <div className="flex flex-wrap items-center justify-between gap-2 mt-auto pt-4">
                  <span className="text-2xs text-fg-3">From the Phase 7 benchmark run, not a live query.</span>
                  <Button
                    size="sm"
                    variant="ghost"
                    icon={<ArrowRight className="w-3.5 h-3.5" />}
                    onClick={() => {
                      requestAsk(example.question, example.reference_time ?? null);
                      onNavigate("ask");
                    }}
                  >
                    Run it live
                  </Button>
                </div>
              </>
            ) : demos.error ? (
              <p className="text-sm text-fg-3 mt-4">Example unavailable: {demos.error.message}</p>
            ) : (
              <div className="space-y-3 mt-4">
                <Skeleton className="h-14 w-full" />
                <Skeleton className="h-16 w-full" />
              </div>
            )}
          </div>
        </div>
      </section>

      {health.error && !h && (
        <ErrorState
          title="The FastAPI backend is unreachable"
          error={health.error}
          action={
            <Button size="sm" onClick={health.refresh}>
              Retry
            </Button>
          }
        />
      )}

      {/* KPIs */}
      <section aria-label="Research corpus" className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard
          icon={<FileText className="w-4 h-4" />}
          label="SEC filings"
          value={fmtInt(corpusData?.total_filings)}
          loading={corpus.loading}
          detail={
            corpusData
              ? Object.entries(corpusData.by_type)
                  .sort((a, b) => b[1] - a[1])
                  .map(([t, n]) => `${n} ${t}`)
                  .join(", ")
              : corpus.error
              ? "Corpus endpoint unavailable"
              : undefined
          }
        />
        <KpiCard
          icon={<Building2 className="w-4 h-4" />}
          label="Companies"
          value={fmtInt(corpusData?.companies.length)}
          loading={corpus.loading}
          detail={filingDates ? `Filed ${fmtDate(filingDates.first)} to ${fmtDate(filingDates.last)}` : undefined}
        />
        <KpiCard
          icon={<Database className="w-4 h-4" />}
          label="Indexed passages"
          value={fmtInt(points ?? corpusData?.total_chunks)}
          loading={!h && !corpusData && !health.error}
          detail="Structure-aware chunks in Qdrant"
        />
        <KpiCard
          icon={<Network className="w-4 h-4" />}
          label="Knowledge graph entities"
          value={fmtInt(kgNodes)}
          loading={!h && !health.error}
          detail={kgEdges != null ? `${fmtInt(kgEdges)} relationships` : undefined}
        />
      </section>

      {/* Pipeline */}
      <section className="card p-5 sm:p-6" aria-labelledby="pipe-title">
        <div className="flex flex-wrap items-end justify-between gap-3 mb-5">
          <div>
            <h2 id="pipe-title" className="text-md font-semibold">
              How it works
            </h2>
            <p className="text-sm text-fg-2 mt-0.5">
              The first three steps are built once from the filings. The last two run for every question.
            </p>
          </div>
          <Button size="sm" variant="ghost" onClick={() => onNavigate("architecture")} icon={<ArrowRight className="w-3.5 h-3.5" />}>
            Full architecture
          </Button>
        </div>
        <ol className="grid grid-cols-1 md:grid-cols-5 gap-3 md:gap-0">
          {PIPELINE.map((s, i) => (
            <li key={s.name} className="relative md:pr-6 last:md:pr-0">
              <button
                onClick={() => onNavigate("architecture")}
                className="card-interactive w-full h-full text-left rounded-lg border border-line-soft bg-surface-2 p-4 hover:bg-surface-3/60"
              >
                <div className="flex items-center gap-2.5">
                  <span
                    className={cx(
                      "w-8 h-8 rounded-lg flex items-center justify-center shrink-0",
                      s.when === "query" ? "bg-accent/15 text-accent" : "bg-cyan-soft text-cyan"
                    )}
                  >
                    {s.icon}
                  </span>
                  <span className="font-mono-num text-2xs text-fg-3">{String(i + 1).padStart(2, "0")}</span>
                </div>
                <div className="text-sm font-medium text-fg mt-3 leading-snug">{s.name}</div>
                <div className="text-xs text-fg-3 mt-1 leading-relaxed">{s.plain}</div>
              </button>
              {i < PIPELINE.length - 1 && (
                <span aria-hidden className="hidden md:flex absolute right-0 top-1/2 -translate-y-1/2 w-6 justify-center text-fg-3">
                  <ArrowRight className="w-4 h-4" />
                </span>
              )}
            </li>
          ))}
        </ol>
        <div className="hidden md:grid grid-cols-5 mt-3 text-2xs text-fg-3">
          <div className="col-span-3 border-t border-cyan-soft pt-1.5 mr-6 text-cyan">Built once, offline</div>
          <div className="col-span-2 border-t border-accent/30 pt-1.5 text-accent">Runs per question</div>
        </div>
      </section>

      {/* Performance + limitations */}
      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)] gap-6">
        <section className="card p-5 sm:p-6" aria-labelledby="perf-title">
          <div className="flex flex-wrap items-start justify-between gap-3 mb-5">
            <div>
              <h2 id="perf-title" className="text-md font-semibold">
                How it performed
              </h2>
              <p className="text-sm text-fg-2 mt-0.5">
                {p7?.total_questions ?? 64}-question SEC benchmark, including {p7?.unanswerable_questions ?? 14} questions designed to be
                refused.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              <Badge tone="neutral">Historical benchmark</Badge>
              <Badge mono>Phase 7</Badge>
              {p7?.timestamp && <Badge mono>{fmtDate(p7.timestamp)}</Badge>}
            </div>
          </div>
          {summary.error ? (
            <ErrorState compact title="Benchmark summary unavailable" error={summary.error} />
          ) : !combined || !p7 ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {[0, 1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-24" />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <PerfTile
                tone="accent"
                value={fmtPct(combined.decision_accuracy, 2)}
                label="Decision accuracy"
                detail={`${Math.round(combined.decision_accuracy * (p7.total_questions ?? 0))} of ${p7.total_questions} answer-or-refuse decisions correct`}
              />
              <PerfTile
                value={`${combined.correct_answers_attempted ?? "—"} / ${p7.answerable_questions}`}
                label="Answerable questions answered"
                detail={`${combined.false_refusals} declined (false refusals)`}
              />
              <PerfTile
                tone="pos"
                value={`${combined.correct_refusals ?? "—"} / ${p7.unanswerable_questions}`}
                label="Correct refusals"
                detail="Unanswerable questions declined"
              />
              <PerfTile
                tone="pos"
                value={`${combined.hallucinations_total ?? "—"} / ${p7.unanswerable_questions}`}
                label="Hallucinations"
                detail="Unanswerable questions answered anyway"
              />
            </div>
          )}
          <p className="text-xs text-fg-3 mt-4 leading-relaxed max-w-[90ch]">
            Decision accuracy measures whether the system correctly chose to answer or to refuse. It is not factual answer accuracy:
            answer wording against the reference scored token F1 {fmtNum(combined?.answerable_token_f1 ?? null, 4)} and exact match{" "}
            {fmtPct(combined?.answerable_exact_match ?? null, 0)}.{" "}
            <button className="text-accent hover:underline" onClick={() => onNavigate("evaluation")}>
              See the full evaluation
            </button>
          </p>
        </section>

        <section className="card p-5 sm:p-6" aria-labelledby="lim-title">
          <h2 id="lim-title" className="text-md font-semibold flex items-center gap-2">
            <TriangleAlert className="w-4 h-4 text-warn" /> Known limitations
          </h2>
          <ul className="mt-4 space-y-4 text-sm">
            <li>
              <div className="text-fg">Pre-trained knowledge can leak</div>
              <div className="text-xs text-fg-3 mt-0.5 leading-relaxed">
                Date filtering controls which filings reach the model, not what it already knows.
                {plr83
                  ? ` On the Phase 8.3 held-out benchmark, with retrieval off, it recalled ${plr83.k} of ${plr83.n} later facts (PLR ${plr83.pct}%).`
                  : " This is measured separately (Phase 8.3), with retrieval switched off."}
              </div>
            </li>
            <li>
              <div className="text-fg">Answers are paraphrased</div>
              <div className="text-xs text-fg-3 mt-0.5 leading-relaxed">
                Exact match with reference answers was {fmtPct(combined?.answerable_exact_match ?? null, 0)}; answers should be read with
                their citations.
              </div>
            </li>
            <li>
              <div className="text-fg">Narrow corpus</div>
              <div className="text-xs text-fg-3 mt-0.5 leading-relaxed">
                {corpusData ? `${corpusData.companies.length} companies` : "Ten companies"}
                {filingDates ? `, filings from ${fmtDate(filingDates.first)} to ${fmtDate(filingDates.last)}` : ""}. Questions outside
                this scope are refused.
              </div>
            </li>
            <li>
              <div className="text-fg">Small local model</div>
              <div className="text-xs text-fg-3 mt-0.5 leading-relaxed">
                Generation uses {model ? <span className="font-mono-num">{model}</span> : "a 4B-parameter model"} through Ollama, chosen
                for reproducibility over fluency.
              </div>
            </li>
          </ul>
        </section>
      </div>

      {/* Secondary: system details */}
      <section className="card" aria-labelledby="details-title">
        <button
          onClick={() => setDetailsOpen((o) => !o)}
          aria-expanded={detailsOpen}
          aria-controls="system-details"
          className="w-full flex items-center justify-between gap-3 px-5 sm:px-6 py-4 text-left"
        >
          <span>
            <span id="details-title" className="text-md font-semibold block">
              System and corpus details
            </span>
            <span className="text-sm text-fg-2">Live service health, graph statistics and per-company coverage.</span>
          </span>
          <ChevronDown className={cx("w-5 h-5 text-fg-3 transition-transform shrink-0", detailsOpen && "rotate-180")} />
        </button>

        {detailsOpen && (
          <div id="system-details" className="border-t border-line-soft px-5 sm:px-6 py-5 space-y-7 animate-fade-in">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div>
                <h3 className="text-sm font-semibold mb-2">Services</h3>
                <KeyValue
                  dense
                  items={[
                    {
                      label: "FastAPI",
                      value: health.error ? (
                        <span className="inline-flex items-center gap-2">
                          <StatusDot status="unavailable" /> Unreachable
                        </span>
                      ) : h ? (
                        <span className="inline-flex items-center gap-2">
                          <StatusDot status="healthy" /> Responding
                        </span>
                      ) : (
                        "—"
                      ),
                    },
                    { label: "Pipeline", value: <StatusValue c={h?.pipeline} /> },
                    { label: "Qdrant", value: <StatusValue c={h?.qdrant} /> },
                    { label: "Knowledge graph", value: <StatusValue c={h?.knowledge_graph} /> },
                    { label: "Ollama", value: <StatusValue c={h?.llm} /> },
                    { label: "Model", value: model ?? "—", mono: true },
                    {
                      label: "Mode",
                      value: h
                        ? `${h.pipeline.details?.hop_mode} hops, ${h.pipeline.details?.balanced_fusion ? "balanced" : "standard"} RRF`
                        : "—",
                    },
                  ]}
                />
              </div>
              <div>
                <h3 className="text-sm font-semibold mb-2">Corpus</h3>
                <KeyValue
                  dense
                  items={[
                    { label: "Filings", value: <span className="num">{fmtInt(corpusData?.total_filings)}</span> },
                    { label: "Chunks (manifest)", value: <span className="num">{fmtInt(corpusData?.total_chunks)}</span> },
                    { label: "Vectors (Qdrant)", value: <span className="num">{fmtInt(points)}</span> },
                    {
                      label: "Filing dates",
                      value: filingDates ? <span className="num">{`${fmtDate(filingDates.first)} to ${fmtDate(filingDates.last)}`}</span> : "—",
                    },
                  ]}
                />
              </div>
              <div>
                <h3 className="text-sm font-semibold mb-2">Knowledge graph</h3>
                <KeyValue
                  dense
                  items={[
                    { label: "Entities", value: <span className="num">{fmtInt(kgNodes)}</span> },
                    { label: "Relationships", value: <span className="num">{fmtInt(kgEdges)}</span> },
                    {
                      label: "Company-type nodes",
                      value: <span className="num">{fmtInt(companies.data?.total_companies)}</span>,
                      hint: "Includes extracted mentions, not only the 10 issuers",
                    },
                    {
                      label: "Regulatory linking",
                      value: h ? (h.knowledge_graph.details?.regulatory_linking ? "Enabled" : "Disabled") : "—",
                    },
                  ]}
                />
              </div>
            </div>

            <div>
              <div className="flex flex-wrap items-end justify-between gap-3 mb-3">
                <div>
                  <h3 className="text-sm font-semibold">Company coverage</h3>
                  <p className="text-xs text-fg-3 mt-0.5">Choose a company to make it your research context.</p>
                </div>
                <Input
                  inputSize="sm"
                  className="w-full sm:w-56"
                  placeholder="Filter companies"
                  aria-label="Filter companies"
                  leading={<Search className="w-3.5 h-3.5" />}
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                />
              </div>
              <div className="border border-line-soft rounded-lg overflow-x-auto">
                <table className="w-full text-sm min-w-[720px]">
                  <thead>
                    <tr className="text-left text-xs text-fg-3 border-b border-line-soft bg-surface-2">
                      <th className="font-medium px-4 py-2.5">Company</th>
                      <th className="font-medium px-3 py-2.5 text-right">Filings</th>
                      <th className="font-medium px-3 py-2.5">Forms</th>
                      <th className="font-medium px-3 py-2.5">Filed</th>
                      <th className="font-medium px-3 py-2.5 text-right">Chunks</th>
                      <th className="font-medium px-3 py-2.5 text-right" title="Direct knowledge-graph edges">
                        KG edges
                      </th>
                      <th className="px-3 py-2.5">
                        <span className="sr-only">Actions</span>
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-line-soft">
                    {issuers.map((r) => {
                      const isCtx = entity?.nodeId === `company:${r.ticker}`;
                      const cov = r.coverage;
                      return (
                        <tr key={r.ticker} className={cx(isCtx ? "bg-accent/[0.06]" : "hover:bg-surface-2/60")}>
                          <td className="px-4 py-2.5">
                            <div className="flex items-center gap-2.5">
                              <span className="font-mono-num text-xs text-fg-3 w-12 shrink-0">{r.ticker}</span>
                              <span className="text-fg truncate">{r.name}</span>
                              {isCtx && <Badge tone="accent">Context</Badge>}
                            </div>
                          </td>
                          <td className="px-3 py-2.5 text-right num">{cov ? cov.filings : "—"}</td>
                          <td className="px-3 py-2.5 text-xs text-fg-2 num whitespace-nowrap">
                            {cov
                              ? Object.entries(cov.by_type)
                                  .sort()
                                  .map(([t, n]) => `${n} ${t}`)
                                  .join(", ")
                              : "—"}
                          </td>
                          <td className="px-3 py-2.5 text-xs text-fg-2 num whitespace-nowrap">
                            {cov ? `${fmtDate(cov.first_publication_date)} to ${fmtDate(cov.last_publication_date)}` : "—"}
                          </td>
                          <td className="px-3 py-2.5 text-right num">{cov ? fmtInt(cov.chunks) : "—"}</td>
                          <td className="px-3 py-2.5 text-right num">{fmtInt(r.degree)}</td>
                          <td className="px-3 py-1.5">
                            <div className="flex justify-end gap-0.5">
                              <IconButton
                                size="sm"
                                label={isCtx ? `${r.name} is the research context` : `Set ${r.name} as research context`}
                                active={isCtx}
                                onClick={() => setEntity(issuerEntity(r.ticker, r.name))}
                              >
                                <Crosshair className={cx("w-3.5 h-3.5", isCtx && "text-accent")} />
                              </IconButton>
                              <IconButton
                                size="sm"
                                label={`Open ${r.name} in the knowledge graph`}
                                onClick={() => {
                                  setEntity(issuerEntity(r.ticker, r.name));
                                  onNavigate("graph");
                                }}
                              >
                                <Waypoints className="w-3.5 h-3.5" />
                              </IconButton>
                              <IconButton
                                size="sm"
                                label={`Browse ${r.name} filings`}
                                onClick={() => {
                                  setEntity(issuerEntity(r.ticker, r.name));
                                  onNavigate("evidence");
                                }}
                              >
                                <BookOpenText className="w-3.5 h-3.5" />
                              </IconButton>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                    {issuers.length === 0 && (
                      <tr>
                        <td colSpan={7} className="px-4 py-6 text-center text-sm text-fg-3">
                          No company matches “{filter}”.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
