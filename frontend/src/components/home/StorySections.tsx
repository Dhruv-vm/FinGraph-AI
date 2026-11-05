"use client";

import React, { useEffect, useRef, useState } from "react";
import { Database, FileText, GitBranch, Layers, ShieldCheck } from "lucide-react";
import { Reveal, SplitText, useScrollProgress } from "./motion";

/* ======================================================================
   Section B — the solution: four methodological ideas
   ====================================================================== */

function StructureVisual() {
  const items = [
    { k: "Item 1A", t: "Risk Factors", w: "86%" },
    { k: "Item 7", t: "MD&A", w: "72%" },
    { k: "Item 8", t: "Financial Statements", w: "64%" },
  ];
  return (
    <div className="fgh-vis space-y-2.5">
      {items.map((it, i) => (
        <div key={it.k} className="fgh-vis-row" style={{ ["--i" as any]: i }}>
          <span className="font-mono text-[10px] text-[color:var(--h-blue)] w-14 shrink-0">{it.k}</span>
          <span className="flex-1 min-w-0">
            <span className="block text-[12px] text-[color:var(--h-fg-2)] truncate">{it.t}</span>
            <span className="fgh-bar mt-1.5" style={{ ["--w" as any]: it.w }} />
          </span>
        </div>
      ))}
      <div className="flex flex-wrap gap-1.5 pt-2">
        {["≤ 1,800 chars", "250 overlap", "paragraph-aware"].map((c) => (
          <span key={c} className="fgh-chip">{c}</span>
        ))}
      </div>
    </div>
  );
}

function TimeVisual() {
  const dots = [8, 22, 36, 50, 64, 78, 92];
  return (
    <div className="fgh-vis">
      <div className="relative h-16">
        <div className="absolute left-0 right-0 top-8 h-px bg-white/10" />
        {dots.map((x, i) => (
          <span key={x} className="fgh-tdot" data-future={x > 60 ? "" : undefined} style={{ left: `${x}%`, ["--i" as any]: i }} />
        ))}
        <div className="fgh-asof" style={{ left: "60%" }}>
          <span className="font-mono text-[10px]">as_of</span>
        </div>
      </div>
      <p className="font-mono text-[11px] text-[color:var(--h-fg-2)] mt-3">
        include evidence only if <span className="text-[color:var(--h-cyan)]">available_time ≤ as_of</span>
      </p>
    </div>
  );
}

function GraphVisual() {
  const preds = ["ACQUIRED", "COMPETES_WITH", "SUPPLIES", "PARTNERS_WITH", "HAS_RISK"];
  return (
    <div className="fgh-vis">
      <svg viewBox="0 0 240 90" className="w-full h-[90px]" aria-hidden="true">
        {[
          [40, 45, 120, 20],
          [40, 45, 120, 70],
          [120, 20, 200, 45],
          [120, 70, 200, 45],
          [120, 20, 120, 70],
        ].map(([a, b, c, d], i) => (
          <line key={i} x1={a} y1={b} x2={c} y2={d} className="fgh-gline" style={{ ["--i" as any]: i }} pathLength={1} />
        ))}
        {[
          [40, 45, 5],
          [120, 20, 3.5],
          [120, 70, 3.5],
          [200, 45, 5],
        ].map(([x, y, r], i) => (
          <circle key={i} cx={x} cy={y} r={r} className="fgh-gnode" style={{ ["--i" as any]: i }} />
        ))}
      </svg>
      <div className="flex flex-wrap gap-1.5 mt-2">
        {preds.map((p) => (
          <span key={p} className="fgh-chip font-mono">{p}</span>
        ))}
      </div>
    </div>
  );
}

function GateVisual() {
  return (
    <div className="fgh-vis">
      <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2 text-[11px]">
        <span className="fgh-box">Ranked evidence</span>
        <span className="fgh-gate" aria-hidden="true">
          <ShieldCheck className="w-4 h-4" />
        </span>
        <div className="space-y-1.5">
          <span className="fgh-box fgh-box--ok block">Sufficient → cited answer</span>
          <span className="fgh-box fgh-box--no block">Insufficient → refuse</span>
        </div>
      </div>
      <p className="text-[11px] text-[color:var(--h-fg-3)] mt-3">Checked before generation: entities, relationships, hop depth, time and provenance.</p>
    </div>
  );
}

const METHOD = [
  {
    n: "01",
    title: "Keep the structure of the filing",
    body: "Filings are split along paragraph boundaries, so a passage about risk factors stays whole and every citation reads like the original.",
    Visual: StructureVisual,
  },
  {
    n: "02",
    title: "Respect when information was public",
    body: "Every passage and edge carries available_time. Questions are answered as of a date, and anything published later is rejected, not just down-ranked.",
    Visual: TimeVisual,
  },
  {
    n: "03",
    title: "Connect entities across filings",
    body: "Companies, counterparties and risks become nodes; typed, dated relationships become edges that keep their source filing.",
    Visual: GraphVisual,
  },
  {
    n: "04",
    title: "Answer from evidence — or refuse",
    body: "A sufficiency gate decides whether the evidence supports an answer before the model writes anything. If it does not, FinGraph says so.",
    Visual: GateVisual,
  },
];

export function MethodSection() {
  return (
    <section id="method" aria-labelledby="method-title" className="relative py-28 lg:py-40">
      <div className="fgh-orb fgh-orb--left" data-speed="0.12" aria-hidden="true" />
      <div className="fgh-container grid lg:grid-cols-12 gap-12 lg:gap-16">
        <div className="lg:col-span-5">
          <div className="lg:sticky lg:top-32">
            <p className="fgh-eyebrow">
              <span>02</span> The solution
            </p>
            <h2 id="method-title" className="fgh-h2 mt-5">
              <SplitText text="Connect the information." />
              <br />
              <SplitText text="Understand the relationships." delay={180} className="fgh-serif text-[color:var(--h-fg-2)]" />
            </h2>
            <Reveal as="p" delay={200} className="fgh-lead mt-6 max-w-[42ch]">
              FinGraph treats a corpus of filings as one connected, time-aware body of evidence — and holds every answer to it.
            </Reveal>
          </div>
        </div>
        <ol className="lg:col-span-7 grid sm:grid-cols-2 gap-4 lg:gap-5">
          {METHOD.map((m, i) => (
            <li key={m.n} className={i % 2 === 1 ? "sm:translate-y-16" : undefined}>
              <div data-speed={i % 2 === 1 ? "0.06" : "0"}>
                <Reveal delay={i * 90} className="fgh-card fgh-card--method h-full">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-[11px] text-[color:var(--h-fg-3)]">{m.n}</span>
                  </div>
                  <h3 className="text-[19px] leading-snug font-medium tracking-tight text-[color:var(--h-fg)] mt-6">{m.title}</h3>
                  <p className="text-[14px] leading-relaxed text-[color:var(--h-fg-2)] mt-3">{m.body}</p>
                  <div className="mt-6 pt-5 border-t border-[color:var(--h-line)]">
                    <m.Visual />
                  </div>
                </Reveal>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

/* ======================================================================
   Section C — the research pipeline (horizontal, scroll-driven on desktop)
   ====================================================================== */

const PIPELINE = [
  {
    title: "SEC Filings",
    icon: FileText,
    body: "Primary filings from SEC EDGAR for 10 issuers, with original filing metadata kept.",
    facts: ["99 filings", "10-K · 10-Q · 8-K"],
  },
  {
    title: "Structure-Aware Processing",
    icon: Layers,
    body: "Paragraph-boundary chunking with deterministic overlap; filing date, fiscal period and available_time on every passage.",
    facts: ["8,181 passages", "≤ 1,800 chars"],
  },
  {
    title: "Temporal Knowledge Graph",
    icon: GitBranch,
    body: "Entities and relationships extracted per passage, resolved, and stored as time-stamped edges with filing provenance.",
    facts: ["10,054 entities", "9,822 relationships"],
  },
  {
    title: "Hybrid Retrieval",
    icon: Database,
    body: "Dense search in Qdrant (all-MiniLM-L6-v2) and bounded graph traversal, fused with balanced RRF and a deterministic reranker.",
    facts: ["RRF k = 60", "min 3 per entity"],
  },
  {
    title: "Evidence-Grounded QA",
    icon: ShieldCheck,
    body: "A sufficiency gate, then a local model at temperature 0 answers only from cited evidence — and citations are validated against the as-of date.",
    facts: ["refuse when unsupported", "validated citations"],
  },
];

export function PipelineSection({ reduced }: { reduced: boolean }) {
  const trackRef = useRef<HTMLOListElement>(null);
  const fillRef = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState(0);
  const [wide, setWide] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia("(min-width: 1024px)");
    const u = () => setWide(mq.matches);
    u();
    mq.addEventListener?.("change", u);
    return () => mq.removeEventListener?.("change", u);
  }, []);

  const horizontal = wide && !reduced;

  const wrapRef = useScrollProgress<HTMLDivElement>(
    (p) => {
      const track = trackRef.current;
      if (horizontal && track) {
        const travel = Math.max(0, track.scrollWidth - window.innerWidth + 96);
        const t = Math.min(1, Math.max(0, (p - 0.04) / 0.86));
        track.style.transform = `translate3d(${(-travel * t).toFixed(1)}px,0,0)`;
        if (fillRef.current) fillRef.current.style.transform = `scaleX(${t.toFixed(4)})`;
        setActive(Math.min(PIPELINE.length - 1, Math.floor(t * PIPELINE.length * 0.999 + 0.15)));
      } else if (track) {
        track.style.transform = "";
      }
    },
    "sticky",
    !horizontal
  );

  return (
    <section id="pipeline" aria-labelledby="pipeline-title" className="relative">
      <div ref={wrapRef} className={horizontal ? "relative h-[330vh]" : "relative"}>
        <div className={horizontal ? "sticky top-0 h-[100svh] overflow-hidden flex flex-col justify-center" : "py-24"}>
          <div className="fgh-container">
            <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-6">
              <div>
                <p className="fgh-eyebrow">
                  <span>03</span> The research pipeline
                </p>
                <h2 id="pipeline-title" className="fgh-h2 mt-5 max-w-[16ch]">
                  <SplitText text="From filing to grounded answer." />
                </h2>
              </div>
              <p className="fgh-lead max-w-[40ch]">Five stages. The first three build the index ahead of time; the last two run for every question.</p>
            </div>
            {horizontal && (
              <div className="relative h-px bg-[color:var(--h-line)] mt-12" aria-hidden="true">
                <div ref={fillRef} className="absolute inset-0 origin-left bg-gradient-to-r from-[color:var(--h-blue)] to-[color:var(--h-cyan)]" style={{ transform: "scaleX(0)" }} />
              </div>
            )}
          </div>
          <div className={horizontal ? "mt-10 pl-[max(1.5rem,calc((100vw-1240px)/2+1.5rem))]" : "fgh-container mt-12"}>
            <ol ref={trackRef} className={horizontal ? "flex gap-5 w-max will-change-transform" : "relative grid gap-4 md:grid-cols-2"}>
              {PIPELINE.map((s, i) => {
                const Icon = s.icon;
                const on = !horizontal || i <= active;
                return (
                  <li key={s.title} className={horizontal ? "w-[400px] shrink-0" : i === PIPELINE.length - 1 ? "md:col-span-2" : undefined}>
                    <Reveal delay={horizontal ? 0 : (i % 2) * 80} className="fgh-card fgh-card--stage h-full" data-on={on ? "" : undefined}>
                      <div className="flex items-center gap-3">
                        <span className="fgh-stage-num">{String(i + 1).padStart(2, "0")}</span>
                        <span className="fgh-stage-icon">
                          <Icon className="w-4 h-4" />
                        </span>
                        {i < PIPELINE.length - 1 && <span className="hidden lg:block flex-1 h-px bg-[color:var(--h-line)]" aria-hidden="true" />}
                      </div>
                      <h3 className="text-[22px] leading-tight font-medium tracking-tight text-[color:var(--h-fg)] mt-8">{s.title}</h3>
                      <p className="text-[14px] leading-relaxed text-[color:var(--h-fg-2)] mt-3">{s.body}</p>
                      <div className="flex flex-wrap gap-1.5 mt-6">
                        {s.facts.map((f) => (
                          <span key={f} className="fgh-chip">{f}</span>
                        ))}
                      </div>
                      <span className="sr-only">{i < PIPELINE.length - 1 ? `then ${PIPELINE[i + 1].title}` : "final stage"}</span>
                    </Reveal>
                  </li>
                );
              })}
            </ol>
          </div>
        </div>
      </div>
    </section>
  );
}
