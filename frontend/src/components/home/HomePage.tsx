"use client";

import "./home.css";
import React, { useCallback, useEffect, useRef, useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { ArrowRight, ArrowUpRight, Menu, X } from "lucide-react";
import { NAV_TABS, PAGE_META } from "@/lib/navigation";
import { HeroGraph } from "./HeroGraph";
import { ProblemSection } from "./ProblemSection";
import { MethodSection, PipelineSection } from "./StorySections";
import { GraphShowcase } from "./GraphShowcase";
import { HOME_FILINGS, HOME_SHOWCASE, HOME_TICKERS } from "./homeData";
import { CountUp, Reveal, SplitText, prefersReducedMotion, subscribeScroll, useParallax, useReducedMotion } from "./motion";

const WORKSPACE = "/workspace";

const SECTIONS = [
  { id: "problem", label: "Problem" },
  { id: "method", label: "Method" },
  { id: "pipeline", label: "Pipeline" },
  { id: "data", label: "Data" },
  { id: "graph", label: "Graph" },
  { id: "results", label: "Results" },
];

const fmtMonth = (d: string) => new Date(d + "T00:00:00Z").toLocaleDateString("en-US", { month: "short", year: "numeric", timeZone: "UTC" });
const FILING_RANGE = `${fmtMonth(HOME_FILINGS[0].d)} – ${fmtMonth(HOME_FILINGS[HOME_FILINGS.length - 1].d)}`;

/* ---------------------------------------------------------------- navigation into the workspace */

function useEnterWorkspace(rootRef: React.RefObject<HTMLElement>) {
  const router = useRouter();
  useEffect(() => {
    router.prefetch(WORKSPACE);
  }, [router]);
  return useCallback(
    (href: string) => (e: React.MouseEvent<HTMLAnchorElement>) => {
      if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      e.preventDefault();
      if (prefersReducedMotion()) {
        router.push(href);
        return;
      }
      rootRef.current?.setAttribute("data-leaving", "");
      window.setTimeout(() => router.push(href), 360);
    },
    [router, rootRef]
  );
}

function Logo({ className }: { className?: string }) {
  return <Image src="/brand/fingraph-logo-dark.png" alt="FinGraph AI" width={108} height={36} priority className={className ?? "h-8 w-auto"} />;
}

/* ---------------------------------------------------------------- nav */

function SiteNav({ enter }: { enter: (href: string) => (e: React.MouseEvent<HTMLAnchorElement>) => void }) {
  const [scrolled, setScrolled] = useState(false);
  const [active, setActive] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const barRef = useRef<HTMLDivElement>(null);

  useEffect(
    () =>
      subscribeScroll(() => {
        const y = window.scrollY;
        setScrolled(y > 24);
        const max = document.documentElement.scrollHeight - window.innerHeight;
        if (barRef.current) barRef.current.style.transform = `scaleX(${max > 0 ? (y / max).toFixed(4) : 0})`;
        let current: string | null = null;
        for (const s of SECTIONS) {
          const el = document.getElementById(s.id);
          if (el && el.getBoundingClientRect().top < window.innerHeight * 0.45) current = s.id;
        }
        setActive(current);
      }),
    []
  );

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <header className="fgh-nav" data-scrolled={scrolled || open ? "" : undefined} data-open={open ? "" : undefined}>
      <div className="flex items-center h-14 pl-4 pr-2 gap-2">
        <a href="#top" className="shrink-0 rounded-md" aria-label="FinGraph AI — back to top">
          <Logo className="h-[30px] w-auto" />
        </a>
        <nav aria-label="Homepage sections" className="hidden lg:flex items-center gap-0.5 mx-auto">
          {SECTIONS.map((s) => (
            <a key={s.id} href={`#${s.id}`} className="fgh-navlink" data-active={active === s.id ? "" : undefined} aria-current={active === s.id ? "true" : undefined}>
              {s.label}
            </a>
          ))}
        </nav>
        <div className="ml-auto lg:ml-0 flex items-center gap-1.5">
          <a href={`${WORKSPACE}#/overview`} onClick={enter(`${WORKSPACE}#/overview`)} className="fgh-btn fgh-btn--sm">
            <span className="hidden sm:inline">Enter Workspace</span>
            <span className="sm:hidden">Workspace</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </a>
          <button
            type="button"
            className="lg:hidden fgh-icon-btn"
            aria-expanded={open}
            aria-controls="fgh-mobile-menu"
            aria-label={open ? "Close menu" : "Open menu"}
            onClick={() => setOpen((o) => !o)}
          >
            {open ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
          </button>
        </div>
      </div>
      {open && (
        <nav id="fgh-mobile-menu" aria-label="Homepage sections" className="lg:hidden px-3 pb-3 grid grid-cols-2 gap-1">
          {SECTIONS.map((s) => (
            <a key={s.id} href={`#${s.id}`} onClick={() => setOpen(false)} className="fgh-navlink fgh-navlink--block">
              {s.label}
            </a>
          ))}
        </nav>
      )}
      <div className="fgh-progress" aria-hidden="true">
        <div ref={barRef} />
      </div>
    </header>
  );
}

/* ---------------------------------------------------------------- hero */

function Hero({ reduced, enter }: { reduced: boolean; enter: (href: string) => (e: React.MouseEvent<HTMLAnchorElement>) => void }) {
  const contentRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (reduced) return;
    return subscribeScroll(() => {
      const el = contentRef.current;
      if (!el) return;
      const y = window.scrollY;
      const vh = window.innerHeight;
      if (y > vh * 1.2) return;
      el.style.transform = `translate3d(0, ${(-y * 0.18).toFixed(1)}px, 0)`;
      el.style.opacity = Math.max(0, 1 - y / (vh * 0.75)).toFixed(3);
    });
  }, [reduced]);

  return (
    <section id="top" aria-labelledby="hero-title" className="relative h-[100svh] min-h-[640px] overflow-hidden">
      <div className="fgh-hero-grid" data-speed="-0.08" aria-hidden="true" />
      <div className="fgh-hero-glow" aria-hidden="true" />
      <HeroGraph reduced={reduced} />
      <div className="fgh-hero-vignette" aria-hidden="true" />

      <div ref={contentRef} className="relative z-10 h-full fgh-container flex flex-col justify-center pt-16">
        <div className="max-w-[920px]">
          <Reveal className="fgh-pill">
            <span className="fgh-pill-dot" aria-hidden="true" />
            <span className="sm:hidden">Temporal GraphRAG for SEC filings</span>
            <span className="hidden sm:inline">Temporal GraphRAG research platform for SEC filings</span>
          </Reveal>
          <h1 id="hero-title" className="fgh-h1 mt-7">
            <SplitText text="Understand Finance" immediate step={70} />
            <br />
            <SplitText text="Beyond Documents." immediate delay={200} step={90} className="fgh-serif fgh-grad-text" />
          </h1>
          <Reveal as="p" delay={450} className="fgh-hero-sub mt-7">
            <span>Temporal Knowledge Graphs.</span> <span>Hybrid GraphRAG.</span> <span>Evidence-Grounded Intelligence.</span>
          </Reveal>
          <Reveal delay={600} className="flex flex-wrap gap-3 mt-10">
            <a href={`${WORKSPACE}#/overview`} onClick={enter(`${WORKSPACE}#/overview`)} className="fgh-btn">
              Explore FinGraph <ArrowRight className="w-4 h-4" />
            </a>
            <a href="#pipeline" className="fgh-btn fgh-btn--ghost">
              View Research
            </a>
          </Reveal>
        </div>
      </div>

      <div className="absolute z-10 bottom-0 inset-x-0">
        <div className="fgh-container flex items-end justify-between pb-7 gap-6">
          <Reveal delay={900} className="fgh-scrollcue">
            <span className="fgh-scrollcue-line" aria-hidden="true" />
            Scroll to explore
          </Reveal>
          <Reveal delay={1000} className="hidden md:block text-right font-mono text-[11px] leading-5 text-[color:var(--h-fg-3)]">
            SEC EDGAR · {HOME_TICKERS.length} issuers
            <br />
            Filings dated {FILING_RANGE}
          </Reveal>
        </div>
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- data */

const STATS = [
  { v: 99, label: "SEC filings", note: "50 8-K · 39 10-Q · 10 10-K" },
  { v: 10, label: "Companies", note: "Ten issuers across technology, consumer, finance, energy and healthcare" },
  { v: 8181, label: "Indexed passages", note: "Structure-aware chunks, embedded for dense search" },
  { v: 10054, label: "Knowledge graph entities", note: "Connected by 9,822 typed, dated relationships" },
];

function DataSection() {
  const names = HOME_TICKERS.map((t) => ({ t, n: HOME_SHOWCASE[t]?.name ?? t }));
  return (
    <section id="data" aria-labelledby="data-title" className="relative py-28 lg:py-40 border-y border-[color:var(--h-line)] fgh-data-bg">
      <div className="fgh-container">
        <div className="grid lg:grid-cols-12 gap-6 items-end">
          <div className="lg:col-span-7">
            <p className="fgh-eyebrow">
              <span>04</span> The data
            </p>
            <h2 id="data-title" className="fgh-h2 mt-5">
              <SplitText text="A corpus small enough to audit," />
              <br />
              <SplitText text="rich enough to connect." delay={200} className="fgh-serif text-[color:var(--h-fg-2)]" />
            </h2>
          </div>
          <Reveal as="p" className="lg:col-span-5 fgh-lead">
            Every figure below is a count from the project&rsquo;s own build — the filings ingested, the passages indexed and the entities in the graph.
          </Reveal>
        </div>
        <dl className="grid grid-cols-2 lg:grid-cols-4 mt-16 border-t border-[color:var(--h-line)]">
          {STATS.map((s, i) => (
            <Reveal key={s.label} delay={i * 100} className="fgh-stat">
              <dt className="sr-only">{s.label}</dt>
              <dd>
                <CountUp value={s.v} className="fgh-stat-num" duration={1600 + i * 200} />
                <span className="block text-[15px] font-medium text-[color:var(--h-fg)] mt-4">{s.label}</span>
                <span className="block text-[13px] leading-relaxed text-[color:var(--h-fg-3)] mt-1.5 max-w-[26ch]">{s.note}</span>
              </dd>
            </Reveal>
          ))}
        </dl>
      </div>
      <div className="fgh-marquee mt-20" aria-label="Issuers in the corpus">
        <ul className="fgh-marquee-track">
          {[0, 1].map((k) =>
            names.map(({ t, n }) => (
              <li key={`${k}${t}`} aria-hidden={k === 1 ? true : undefined} className="flex items-baseline gap-3 shrink-0">
                <span className="font-mono text-[13px] text-[color:var(--h-blue-text)]">{t}</span>
                <span className="text-[22px] sm:text-[28px] tracking-tight text-[color:var(--h-fg-2)]">{n}</span>
                <span className="text-[color:var(--h-fg-3)] px-6" aria-hidden="true">
                  /
                </span>
              </li>
            ))
          )}
        </ul>
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- results */

const ABLATION = [
  { name: "Frozen Phase 6.4 baseline", v: 85.94 },
  { name: "Regulatory entity linking only", v: 87.5 },
  { name: "Balanced RRF only", v: 89.06 },
  { name: "Adaptive hop validation only", v: 89.06 },
  { name: "Combined Phase 7 configuration", v: 92.19, best: true },
];

function Dots({ total, filled, tone, cols }: { total: number; filled: number; tone: "blue" | "cyan" | "none"; cols: number }) {
  return (
    <div className="fgh-dots" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }} aria-hidden="true">
      {Array.from({ length: total }, (_, i) => (
        <span key={i} data-filled={i < filled ? tone : undefined} style={{ ["--i" as any]: i }} />
      ))}
    </div>
  );
}

function ResultsSection({ enter }: { enter: (href: string) => (e: React.MouseEvent<HTMLAnchorElement>) => void }) {
  return (
    <section id="results" aria-labelledby="results-title" className="relative py-28 lg:py-40">
      <div className="fgh-orb fgh-orb--left" data-speed="0.14" aria-hidden="true" />
      <div className="fgh-container">
        <div className="grid lg:grid-cols-12 gap-6 items-end">
          <div className="lg:col-span-7">
            <p className="fgh-eyebrow">
              <span>06</span> Research performance
            </p>
            <h2 id="results-title" className="fgh-h2 mt-5">
              <SplitText text="Answers when it can." />
              <br />
              <SplitText text="Refuses when it should." delay={200} className="fgh-serif text-[color:var(--h-fg-2)]" />
            </h2>
          </div>
          <Reveal className="lg:col-span-5">
            <span className="fgh-badge">Historical benchmark · Phase 7</span>
            <p className="fgh-lead mt-4">
              Results recorded from the Phase 7 evaluation of the combined configuration on 64 benchmark questions — 50 answerable and 14 unanswerable. Not a live measurement.
            </p>
          </Reveal>
        </div>

        <div className="grid lg:grid-cols-12 gap-4 lg:gap-5 mt-14">
          <Reveal className="fgh-card lg:col-span-5 flex flex-col">
            <div className="text-[13px] text-[color:var(--h-fg-2)]">Decision accuracy</div>
            <div className="relative mt-6 mx-auto w-[220px] h-[220px] sm:w-[260px] sm:h-[260px]">
              <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90" aria-hidden="true">
                <circle cx="60" cy="60" r="52" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="5" />
                <circle cx="60" cy="60" r="52" fill="none" stroke="url(#fgh-ring)" strokeWidth="5" strokeLinecap="round" pathLength={1} className="fgh-ring" style={{ ["--v" as any]: 0.9219 }} />
                <defs>
                  <linearGradient id="fgh-ring" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="#3D7BFF" />
                    <stop offset="100%" stopColor="#5EC8DE" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="absolute inset-0 grid place-items-center text-center">
                <div>
                  <CountUp value={92.19} decimals={2} suffix="%" className="fgh-ring-num" />
                  <div className="font-mono text-[11px] text-[color:var(--h-fg-3)] mt-1">59 of 64 decisions</div>
                </div>
              </div>
            </div>
            <p className="text-[13px] leading-relaxed text-[color:var(--h-fg-2)] mt-8">
              The share of benchmark questions where FinGraph made the right <em>decision</em>: answering an answerable question, or refusing an unanswerable one.
            </p>
            <p className="fgh-note mt-4">
              Decision accuracy is not factual answer accuracy. It does not grade whether answer text is correct; answer quality is reported separately (token F1, exact match) in the Evaluation lab.
            </p>
          </Reveal>

          <div className="lg:col-span-7 grid gap-4 lg:gap-5">
            <Reveal delay={80} className="fgh-card">
              <div className="flex flex-wrap items-baseline justify-between gap-3">
                <div>
                  <div className="fgh-metric">
                    45<span>/50</span>
                  </div>
                  <div className="text-[14px] text-[color:var(--h-fg)] mt-2">Answerable questions answered</div>
                </div>
                <div className="text-[12px] text-[color:var(--h-fg-3)] max-w-[28ch] text-right">5 answerable questions were refused (false refusals)</div>
              </div>
              <div className="mt-6">
                <Dots total={50} filled={45} tone="blue" cols={25} />
              </div>
            </Reveal>
            <div className="grid sm:grid-cols-2 gap-4 lg:gap-5">
              <Reveal delay={160} className="fgh-card">
                <div className="fgh-metric">
                  14<span>/14</span>
                </div>
                <div className="text-[14px] text-[color:var(--h-fg)] mt-2">Correct refusals</div>
                <div className="text-[12px] text-[color:var(--h-fg-3)] mt-1">Every unanswerable question was declined.</div>
                <div className="mt-6">
                  <Dots total={14} filled={14} tone="cyan" cols={7} />
                </div>
              </Reveal>
              <Reveal delay={240} className="fgh-card">
                <div className="fgh-metric">
                  0<span>/14</span>
                </div>
                <div className="text-[14px] text-[color:var(--h-fg)] mt-2">Hallucinations on unanswerable questions</div>
                <div className="text-[12px] text-[color:var(--h-fg-3)] mt-1">No answer was invented where evidence was missing.</div>
                <div className="mt-6">
                  <Dots total={14} filled={0} tone="none" cols={7} />
                </div>
              </Reveal>
            </div>
          </div>
        </div>

        <Reveal className="fgh-card mt-4 lg:mt-5">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <h3 className="text-[15px] font-medium text-[color:var(--h-fg)]">Phase 7 ablation · decision accuracy</h3>
            <a href={`${WORKSPACE}#/evaluation`} onClick={enter(`${WORKSPACE}#/evaluation`)} className="fgh-link">
              Full results in the Evaluation lab <ArrowUpRight className="w-3.5 h-3.5" />
            </a>
          </div>
          <ul className="mt-6 space-y-3.5">
            {ABLATION.map((a, i) => (
              <li key={a.name} className="grid grid-cols-[minmax(0,1fr)_auto] sm:grid-cols-[230px_minmax(0,1fr)_64px] items-center gap-x-4 gap-y-1.5">
                <span className={`text-[13px] ${a.best ? "text-[color:var(--h-fg)]" : "text-[color:var(--h-fg-2)]"}`}>{a.name}</span>
                <span className="font-mono text-[12.5px] text-right sm:order-last" style={{ color: a.best ? "var(--h-cyan)" : "var(--h-fg-2)" }}>
                  {a.v.toFixed(2)}%
                </span>
                <span className="fgh-abl col-span-2 sm:col-span-1" aria-hidden="true">
                  <span style={{ ["--w" as any]: `${a.v}%`, ["--i" as any]: i }} data-best={a.best ? "" : undefined} />
                </span>
              </li>
            ))}
          </ul>
          <p className="text-[11.5px] text-[color:var(--h-fg-3)] mt-5">Bars start at 0%. Same 64-question benchmark for every configuration.</p>
        </Reveal>
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- final CTA + footer */

function FinalCTA({ enter }: { enter: (href: string) => (e: React.MouseEvent<HTMLAnchorElement>) => void }) {
  return (
    <section aria-labelledby="cta-title" className="relative py-32 lg:py-48 overflow-hidden">
      <div className="fgh-cta-rings" data-speed="0.08" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      <div className="fgh-float" aria-hidden="true">
        {[
          [12, 30, 0.18],
          [22, 72, 0.3],
          [80, 24, 0.24],
          [88, 68, 0.12],
          [64, 86, 0.2],
          [36, 14, 0.1],
        ].map(([x, y, s], i) => (
          <span key={i} data-speed={String(s)} style={{ left: `${x}%`, top: `${y}%` }} />
        ))}
      </div>
      <div className="fgh-container relative text-center">
        <p className="fgh-eyebrow justify-center">
          <span>07</span> Workspace
        </p>
        <h2 id="cta-title" className="fgh-h2 fgh-h2--xl mt-6 mx-auto max-w-[18ch]">
          <SplitText text="Explore the intelligence behind" />{" "}
          <SplitText text="financial disclosures." delay={250} className="fgh-serif fgh-grad-text" />
        </h2>
        <Reveal as="p" delay={200} className="fgh-lead mt-7 mx-auto max-w-[52ch]">
          Ask point-in-time questions, walk the knowledge graph, inspect the evidence and review the benchmarks — in the FinGraph workspace.
        </Reveal>
        <Reveal delay={320} className="mt-10 flex justify-center">
          <a href={`${WORKSPACE}#/overview`} onClick={enter(`${WORKSPACE}#/overview`)} className="fgh-btn fgh-btn--lg">
            Enter FinGraph Workspace <ArrowRight className="w-4 h-4" />
          </a>
        </Reveal>
      </div>
    </section>
  );
}

function Footer({ enter }: { enter: (href: string) => (e: React.MouseEvent<HTMLAnchorElement>) => void }) {
  return (
    <footer className="border-t border-[color:var(--h-line)]">
      <div className="fgh-container py-14 grid md:grid-cols-12 gap-10">
        <div className="md:col-span-5">
          <Logo className="h-9 w-auto" />
          <p className="text-[13px] leading-relaxed text-[color:var(--h-fg-3)] mt-4 max-w-[44ch]">
            A research prototype for temporal knowledge graphs and hybrid GraphRAG over SEC filings. Benchmark figures on this page are historical Phase 7 results.
          </p>
        </div>
        <nav aria-label="Workspace" className="md:col-span-7 md:justify-self-end">
          <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-[color:var(--h-fg-3)]">Workspace</div>
          <ul className="grid grid-cols-2 sm:grid-cols-3 gap-x-10 gap-y-2.5 mt-4">
            {NAV_TABS.map((t) => (
              <li key={t}>
                <a href={`${WORKSPACE}#/${t}`} onClick={enter(`${WORKSPACE}#/${t}`)} className="text-[13.5px] text-[color:var(--h-fg-2)] hover:text-[color:var(--h-fg)] transition-colors">
                  {PAGE_META[t].label}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </footer>
  );
}

/* ---------------------------------------------------------------- page */

export function HomePage() {
  const rootRef = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotion();
  const enter = useEnterWorkspace(rootRef);
  useParallax(rootRef);

  // Old links pointed at the dashboard on "/" (e.g. /#/graph). Send them to the workspace.
  useEffect(() => {
    const tab = window.location.hash.replace(/^#\/?/, "").split(/[?/]/)[0];
    if ((NAV_TABS as string[]).includes(tab)) window.location.replace(`${WORKSPACE}${window.location.hash}`);
  }, []);

  // Coming back via the browser's back button must not leave the page faded out.
  useEffect(() => {
    const onShow = () => rootRef.current?.removeAttribute("data-leaving");
    window.addEventListener("pageshow", onShow);
    return () => window.removeEventListener("pageshow", onShow);
  }, []);

  return (
    <div ref={rootRef} className="fgh">
      <a href="#problem" className="fgh-skip">
        Skip to content
      </a>
      <SiteNav enter={enter} />
      <main>
        <Hero reduced={reduced} enter={enter} />
        <ProblemSection reduced={reduced} />
        <MethodSection />
        <PipelineSection reduced={reduced} />
        <DataSection />
        <GraphShowcase />
        <ResultsSection enter={enter} />
        <FinalCTA enter={enter} />
      </main>
      <Footer enter={enter} />
    </div>
  );
}
