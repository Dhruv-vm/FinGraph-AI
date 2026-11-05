"use client";

import React, { useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Badge, IconButton, Note, cx } from "@/components/ui";

/**
 * The FinGraph research pipeline, as implemented.
 * Every file path below was checked against the repository (2026-11-03).
 * Figures marked "historical" come from results/phase7; live counts are on the Overview page.
 */

type Phase = "Corpus" | "Indexing" | "Retrieval" | "Reasoning" | "Verification";

interface Stage {
  id: string;
  name: string;
  phase: Phase;
  purpose: string;
  input: string;
  output: string;
  files: string[];
  tech: string[];
  significance: string;
  facts?: { label: string; value: string }[];
  /** "offline" stages build artefacts ahead of time; "query" stages run per question. */
  runs: "offline" | "query";
}

const STAGES: Stage[] = [
  {
    id: "ingest",
    name: "SEC filing ingestion",
    phase: "Corpus",
    runs: "offline",
    purpose: "Collect the issuers’ filings from SEC EDGAR and extract readable text while keeping the original filing metadata.",
    input: "EDGAR filing index for 10 issuers (10-K, 10-Q, 8-K)",
    output: "Canonical documents in data/processed/documents (one JSON per filing)",
    files: ["src/data/ingestion/sec_corpus.py", "src/data/ingestion/sec_content.py", "src/data/normalization/sec_filings.py", "src/data/documents/builder.py"],
    tech: ["Python", "SEC EDGAR"],
    significance: "Restricting the corpus to primary regulatory filings makes every answer traceable to a dated, public source.",
    facts: [
      { label: "Filings", value: "99 (50 8-K, 39 10-Q, 10 10-K)" },
      { label: "Issuers", value: "AAPL, AMZN, GOOGL, JNJ, JPM, META, MSFT, NVDA, TSLA, XOM" },
    ],
  },
  {
    id: "chunk",
    name: "Structure-aware chunking",
    phase: "Corpus",
    runs: "offline",
    purpose: "Split each filing into passages that follow paragraph boundaries; only oversized paragraphs are cut, with deterministic overlapping windows.",
    input: "Canonical documents",
    output: "Chunks in data/processed/chunks, each carrying its document’s metadata",
    files: ["src/data/chunking/structure_aware.py"],
    tech: ["Deterministic paragraph splitting"],
    significance: "Passages stay semantically whole, which keeps citations readable and retrieval precise.",
    facts: [
      { label: "Chunks", value: "8,181" },
      { label: "Window", value: "max 1,800 characters, 250-character overlap" },
    ],
  },
  {
    id: "temporal",
    name: "Temporal metadata",
    phase: "Corpus",
    runs: "offline",
    purpose: "Attach the times that govern point-in-time use: filing date, fiscal period, and available_time (when the filing became public).",
    input: "Documents and chunks",
    output: "available_time, publication_date, fiscal_period (and event_time where known) on every record",
    files: ["src/data/documents/schema.py", "src/data/documents/provenance.py", "src/data/validation/temporal.py", "src/data/graph/snapshot.py"],
    tech: ["ISO 8601 timestamps"],
    significance: "available_time is what the temporal barrier compares against; filing date and reporting period are kept distinct from it.",
  },
  {
    id: "extract",
    name: "Entity and relationship extraction",
    phase: "Indexing",
    runs: "offline",
    purpose: "Extract companies, products, risks and their relationships (acquired, competes with, supplies, partners with, has risk…) from each chunk, then resolve entity names.",
    input: "Chunks",
    output: "Typed entities and relations with confidence and source chunk",
    files: [
      "src/data/extraction/pipeline.py",
      "src/data/extraction/llm.py",
      "src/data/extraction/entities.py",
      "src/data/extraction/relations.py",
      "src/data/entities/resolution.py",
    ],
    tech: ["LLM extraction (Ollama or OpenRouter client)", "Entity resolution"],
    significance: "Turns unstructured disclosures into relationships the graph retriever can traverse.",
  },
  {
    id: "kg",
    name: "Temporal knowledge graph",
    phase: "Indexing",
    runs: "offline",
    purpose: "Build the graph from extracted relations, keeping each edge’s filing provenance, timestamps and attribution status.",
    input: "Resolved entities and relations",
    output: "data/processed/unified/fingraph_semantic_kg.json",
    files: ["src/data/graph/builder.py", "src/data/graph/schema.py", "src/data/graph/relationships.py", "src/data/graph/storage.py"],
    tech: ["JSON graph store", "Temporal snapshots"],
    significance:
      "Edges carry provenance: an edge extracted from another company’s filing is marked unsupported_cross_company rather than attributed to the issuer.",
    facts: [{ label: "Size", value: "10,054 nodes, 9,822 edges (live count on Overview)" }],
  },
  {
    id: "vector",
    name: "Dense vector index",
    phase: "Indexing",
    runs: "offline",
    purpose: "Embed every chunk for semantic search.",
    input: "Chunks",
    output: "Qdrant collection fingraph_chunks (local storage, data/vector_store)",
    files: ["src/retrieval/vector.py"],
    tech: ["sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)", "Qdrant"],
    significance: "Supplies the semantic half of hybrid retrieval and the passage store behind the evidence explorer.",
  },
  {
    id: "analysis",
    name: "Query analysis",
    phase: "Retrieval",
    runs: "query",
    purpose: "Identify entities, relationship intent, expected hop depth and any reference time in the question.",
    input: "Question and optional reference time",
    output: "Structured query analysis",
    files: ["src/qa/query_analysis.py"],
    tech: ["Graph-backed entity linking"],
    significance: "Hop depth and intent drive both retrieval and the sufficiency gate later.",
  },
  {
    id: "hybrid",
    name: "Vector and graph retrieval",
    phase: "Retrieval",
    runs: "query",
    purpose: "Retrieve candidate passages by vector similarity and candidate relationships by bounded graph exploration around the linked entities.",
    input: "Query analysis",
    output: "Candidate evidence pool (vector, graph and hybrid items)",
    files: ["src/retrieval/hybrid.py", "src/retrieval/graph.py", "src/retrieval/orchestrator.py", "src/qa/agent.py"],
    tech: ["Qdrant search", "Graph traversal"],
    significance: "Graph paths answer relational and multi-hop questions that text similarity alone misses.",
    facts: [{ label: "Defaults", value: "top_k 10, candidate pool 20, up to 2 agent iterations" }],
  },
  {
    id: "rrf",
    name: "Hybrid fusion with balanced RRF",
    phase: "Retrieval",
    runs: "query",
    purpose: "Merge vector and graph rankings with Reciprocal Rank Fusion; for comparative questions reserve a minimum share of evidence for each entity.",
    input: "Vector and graph rankings",
    output: "One fused ranking",
    files: ["src/retrieval/hybrid.py"],
    tech: ["Reciprocal Rank Fusion (k = 60)", "Entity-balanced allocation (min 3 per entity)"],
    significance:
      "Without balancing, one company can crowd out the other in comparisons. Historical: balanced RRF alone moved decision accuracy from 85.94% to 89.06% in Phase 7.",
  },
  {
    id: "rerank",
    name: "Deterministic reranking",
    phase: "Retrieval",
    runs: "query",
    purpose: "Re-score candidates with a transparent weighted formula and reject, not just down-rank, anything unavailable at the reference time.",
    input: "Fused ranking",
    output: "Final ordered evidence",
    files: ["src/retrieval/reranker.py"],
    tech: ["Weighted linear scoring"],
    significance: "Every ranking decision can be reproduced and explained; no stochastic reranker.",
    facts: [
      { label: "Weights", value: "RRF 0.20, entity overlap 0.25, relation intent 0.35, confidence 0.15, hop penalty 0.05" },
    ],
  },
  {
    id: "sufficiency",
    name: "Evidence sufficiency gate",
    phase: "Reasoning",
    runs: "query",
    purpose: "Decide before generation whether the evidence covers the required entities, relationships, hop depth, time constraint and provenance. If not, refuse.",
    input: "Ranked evidence and query analysis",
    output: "Sufficient or insufficient, with reasons and missing items",
    files: ["src/qa/evidence_sufficiency.py"],
    tech: ["Adaptive hop validation"],
    significance:
      "This is what produces correct refusals. Adaptive mode lets a verified direct relationship satisfy a direct question instead of demanding extra hops.",
  },
  {
    id: "generate",
    name: "Grounded generation",
    phase: "Reasoning",
    runs: "query",
    purpose: "Generate a short answer from the supplied evidence only, citing evidence ids.",
    input: "Sufficient evidence package",
    output: "Answer, confidence, cited evidence ids",
    files: ["src/qa/generator.py"],
    tech: ["Ollama", "qwen3.5:4b", "temperature 0", "JSON output mode"],
    significance: "A small local model keeps runs reproducible and private; the prompt forbids using outside knowledge.",
  },
  {
    id: "validate",
    name: "Answer validation",
    phase: "Verification",
    runs: "query",
    purpose: "Check the output structure, confidence range, that cited ids exist in the package, that factual claims are cited, and that no citation post-dates the reference time.",
    input: "Generated output and evidence package",
    output: "Validation result with errors and warnings",
    files: ["src/qa/validation.py"],
    tech: ["Deterministic validator"],
    significance: "Catches fabricated or out-of-package citations before an answer is returned.",
  },
  {
    id: "temporal-check",
    name: "Temporal verification and provenance",
    phase: "Verification",
    runs: "query",
    purpose: "The generator raises an error if any evidence in its package post-dates the reference time; returned evidence keeps document, chunk and attribution status.",
    input: "Evidence package and reference time",
    output: "Temporally valid answer with provenance, or an error",
    files: ["src/qa/generator.py", "src/retrieval/evidence.py", "src/evaluation/temporal.py", "src/evaluation/parametric_leakage.py"],
    tech: ["Hard temporal assertion", "FEIR measurement"],
    significance:
      "Controls document leakage only. What the model learned in pre-training is not filtered, so parametric leakage is measured separately with retrieval disabled (Phase 8.3).",
  },
];

const PHASES: Phase[] = ["Corpus", "Indexing", "Retrieval", "Reasoning", "Verification"];

const PHASE_PLAIN: Record<Phase, string> = {
  Corpus: "Collect SEC filings and cut them into dated passages.",
  Indexing: "Turn the passages into a knowledge graph and a search index.",
  Retrieval: "For each question, find the relevant passages and relationships.",
  Reasoning: "Check the evidence is enough, then answer only from it.",
  Verification: "Validate citations and dates before the answer is returned.",
};

export function ArchitectureView() {
  const [selectedId, setSelectedId] = useState(STAGES[0].id);
  const idx = STAGES.findIndex((s) => s.id === selectedId);
  const stage = STAGES[idx];
  const detailRef = useRef<HTMLDivElement>(null);

  // Arrow-key navigation through stages.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return;
      if (!document.getElementById("arch-root")?.offsetParent) return; // view hidden
      if (e.key === "ArrowDown" || e.key === "ArrowRight") {
        setSelectedId(STAGES[Math.min(idx + 1, STAGES.length - 1)].id);
      } else if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
        setSelectedId(STAGES[Math.max(idx - 1, 0)].id);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [idx]);

  return (
    <div id="arch-root" className="max-w-[1180px] mx-auto px-4 sm:px-8 py-8">
      <div className="mb-8 max-w-[78ch]">
        <h2 className="text-[2rem] font-medium tracking-[-0.035em] leading-[1.1]">From filing to cited answer, in {STAGES.length} stages.</h2>
        <p className="text-sm text-fg-2 mt-2">
          Six stages build the corpus, graph and index ahead of time; eight run for every question. Select a stage to see what it takes
          in, what it produces and where it lives in the code. Arrow keys step through stages.
        </p>
      </div>

      {/* Methodology diagram */}
      <section aria-label="Methodology overview" className="card p-5 sm:p-6 mb-8">
        <ol className="grid grid-cols-1 md:grid-cols-5 gap-3 md:gap-0">
          {PHASES.map((phase, i) => {
            const stages = STAGES.filter((st) => st.phase === phase);
            const on = stage.phase === phase;
            const offline = stages[0].runs === "offline";
            return (
              <li key={phase} className="relative md:pr-7 last:md:pr-0">
                <button
                  onClick={() => setSelectedId(stages[0].id)}
                  aria-pressed={on}
                  className={cx(
                    "card-interactive w-full h-full text-left rounded-lg border p-4 transition-colors",
                    on ? "border-accent/50 bg-accent/[0.07]" : "border-line-soft bg-surface-2 hover:bg-surface-3/60"
                  )}
                >
                  <div className="flex items-center justify-between">
                    <span className={cx("text-2xs font-medium", offline ? "text-cyan" : "text-accent")}>
                      {offline ? "Built once" : "Every question"}
                    </span>
                    <span className="font-mono-num text-2xs text-fg-3">
                      {STAGES.indexOf(stages[0]) + 1}–{STAGES.indexOf(stages[stages.length - 1]) + 1}
                    </span>
                  </div>
                  <div className="text-md font-semibold mt-2">{phase}</div>
                  <p className="text-xs text-fg-2 mt-1 leading-relaxed">{PHASE_PLAIN[phase]}</p>
                  <ul className="mt-3 space-y-1">
                    {stages.map((st) => (
                      <li key={st.id} className={cx("text-2xs leading-snug", st.id === selectedId ? "text-fg" : "text-fg-3")}>
                        {st.name}
                      </li>
                    ))}
                  </ul>
                </button>
                {i < PHASES.length - 1 && (
                  <span aria-hidden className="hidden md:flex absolute right-0 top-1/2 -translate-y-1/2 w-7 justify-center text-fg-3">
                    <ChevronRight className="w-4 h-4" />
                  </span>
                )}
              </li>
            );
          })}
        </ol>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,380px)_1fr] gap-8">
        {/* Stage rail */}
        <nav aria-label="Pipeline stages">
          {PHASES.map((phase) => {
            const stages = STAGES.filter((s) => s.phase === phase);
            return (
              <div key={phase} className="mb-4">
                <div className="text-xs font-medium text-fg-3 mb-1 flex items-center gap-2">
                  {phase}
                  <span className="text-2xs">{stages[0].runs === "offline" ? "built ahead of time" : "per question"}</span>
                </div>
                <ol>
                  {stages.map((s) => {
                    const n = STAGES.indexOf(s) + 1;
                    const on = s.id === selectedId;
                    const last = n === STAGES.length;
                    return (
                      <li key={s.id} className="relative">
                        {!last && (
                          <span
                            aria-hidden
                            className={cx("absolute left-[13px] top-7 -bottom-1 w-px", n < idx + 1 ? "bg-accent/50" : "bg-line")}
                          />
                        )}
                        <button
                          onClick={() => {
                            setSelectedId(s.id);
                            if (window.innerWidth < 1024) detailRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
                          }}
                          aria-current={on ? "step" : undefined}
                          className={cx(
                            "relative w-full text-left flex items-start gap-3 py-1.5 pr-2 rounded group",
                            on ? "text-fg" : "text-fg-2 hover:text-fg"
                          )}
                        >
                          <span
                            className={cx(
                              "relative z-10 w-[27px] h-[27px] shrink-0 rounded-full border flex items-center justify-center font-mono-num text-2xs transition-colors",
                              on
                                ? "bg-accent-strong border-accent-strong text-accent-fg"
                                : n < idx + 1
                                ? "bg-surface border-accent/50 text-accent"
                                : "bg-surface border-line text-fg-3 group-hover:border-fg-3"
                            )}
                          >
                            {n}
                          </span>
                          <span className={cx("pt-1 text-sm leading-snug", on && "font-medium")}>{s.name}</span>
                        </button>
                      </li>
                    );
                  })}
                </ol>
              </div>
            );
          })}
        </nav>

        {/* Stage detail */}
        <div ref={detailRef} className="min-w-0 lg:sticky lg:top-6 self-start">
          <article key={stage.id} className="card animate-fade-in">
            <header className="px-6 pt-5 pb-4 border-b border-line-soft flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 text-xs text-fg-3">
                  <span className="font-mono-num">Stage {idx + 1} of {STAGES.length}</span>
                  <Badge>{stage.phase}</Badge>
                  <Badge tone={stage.runs === "query" ? "accent" : "neutral"}>
                    {stage.runs === "query" ? "Runs per question" : "Built offline"}
                  </Badge>
                </div>
                <h3 className="text-xl font-semibold mt-2">{stage.name}</h3>
              </div>
              <div className="flex gap-1 shrink-0">
                <IconButton label="Previous stage" size="sm" disabled={idx === 0} onClick={() => setSelectedId(STAGES[idx - 1].id)}>
                  <ChevronLeft className="w-4 h-4" />
                </IconButton>
                <IconButton
                  label="Next stage"
                  size="sm"
                  disabled={idx === STAGES.length - 1}
                  onClick={() => setSelectedId(STAGES[idx + 1].id)}
                >
                  <ChevronRight className="w-4 h-4" />
                </IconButton>
              </div>
            </header>
            <div className="px-6 py-5 space-y-6">
              <p className="font-serif text-lg leading-relaxed text-fg max-w-[62ch]">{stage.purpose}</p>

              <div className="grid grid-cols-1 sm:grid-cols-[1fr_auto_1fr] gap-3 items-stretch">
                <div className="rounded border border-line-soft bg-surface-2 px-3.5 py-3">
                  <div className="text-2xs text-fg-3 mb-1">Input</div>
                  <div className="text-sm text-fg">{stage.input}</div>
                </div>
                <div aria-hidden className="hidden sm:flex items-center text-fg-3">
                  <svg width="28" height="12" viewBox="0 0 28 12">
                    <path d="M0 6h24M19 1l5 5-5 5" fill="none" stroke="currentColor" strokeWidth="1.5" />
                  </svg>
                </div>
                <div className="rounded border border-line-soft bg-surface-2 px-3.5 py-3">
                  <div className="text-2xs text-fg-3 mb-1">Output</div>
                  <div className="text-sm text-fg">{stage.output}</div>
                </div>
              </div>

              <div>
                <h4 className="text-xs font-medium text-fg-2 mb-1.5">Why it matters</h4>
                <p className="text-sm text-fg-2 max-w-[70ch] leading-relaxed">{stage.significance}</p>
              </div>

              {stage.facts && (
                <dl className="divide-y divide-line-soft border-y border-line-soft">
                  {stage.facts.map((f) => (
                    <div key={f.label} className="py-2 grid grid-cols-[8rem_1fr] gap-3">
                      <dt className="text-xs text-fg-3 pt-0.5">{f.label}</dt>
                      <dd className="text-sm text-fg num">{f.value}</dd>
                    </div>
                  ))}
                </dl>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-xs font-medium text-fg-2 mb-1.5">Implementation</h4>
                  <ul className="space-y-1">
                    {stage.files.map((f) => (
                      <li key={f} className="font-mono-num text-xs text-fg break-all">
                        {f}
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h4 className="text-xs font-medium text-fg-2 mb-1.5">Technologies</h4>
                  <ul className="flex flex-wrap gap-1.5">
                    {stage.tech.map((t) => (
                      <li key={t}>
                        <Badge>{t}</Badge>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </article>
          <Note className="mt-4">
            Configuration in use is reported live by the backend (hop mode, balanced fusion, regulatory linking) on the Overview page.
            Figures here describe the implementation; benchmark results are in the Evaluation lab.
          </Note>
        </div>
      </div>
    </div>
  );
}
