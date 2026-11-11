"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { HOME_FILINGS, HOME_TICKERS, HUB_LINKS } from "./homeData";
import { easeInOut, range, SplitText, useScrollProgress } from "./motion";

/**
 * Section A — real filings from the corpus start scattered, then condense into
 * nodes and connect to the issuer they were filed by. Issuer-to-issuer links are
 * only drawn where the knowledge graph has a real edge between those companies.
 */

function rng(seed: number) {
  return () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

function pickFilings(perTicker: number) {
  const out: { t: string; f: string; d: string }[] = [];
  for (const t of HOME_TICKERS) {
    const mine = HOME_FILINGS.filter((f) => f.t === t);
    const k = mine.find((f) => f.f === "10-K");
    const q = mine.filter((f) => f.f !== "10-K").slice(-1)[0];
    const chosen = [k, q, mine[0]].filter(Boolean).slice(0, perTicker) as typeof mine;
    out.push(...chosen);
  }
  return out;
}

/** Issuer-to-issuer links that exist as real edges in the graph snapshot. */
const hubLinks = HUB_LINKS;

const CAPTIONS = [
  { k: "01", text: "99 filings — 50 8-K, 39 10-Q, 10 10-K. Each one is usually read on its own." },
  { k: "02", text: "Yet the same companies, counterparties and risks recur across them, filed months apart." },
  { k: "03", text: "FinGraph links every passage to the entities it mentions, in one temporal knowledge graph." },
];

export function ProblemSection({ reduced }: { reduced: boolean }) {
  const stageRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [compact, setCompact] = useState(false);
  const [caption, setCaption] = useState(0);

  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setSize({ w: r.width, h: r.height });
      setCompact(r.width < 560);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const filings = useMemo(() => pickFilings(compact ? 1 : 2), [compact]);

  const geo = useMemo(() => {
    const { w, h } = size;
    const rand = rng(compact ? 5 : 19);
    const cx = w / 2, cy = h / 2;
    const rx = w * (compact ? 0.34 : 0.33), ry = h * (compact ? 0.32 : 0.33);
    const hubs: Record<string, { x: number; y: number; a: number }> = {};
    HOME_TICKERS.forEach((t, i) => {
      const a = (i / HOME_TICKERS.length) * Math.PI * 2 - Math.PI / 2;
      hubs[t] = { x: cx + Math.cos(a) * rx, y: cy + Math.sin(a) * ry, a };
    });
    const count: Record<string, number> = {};
    const docs = filings.map((f) => {
      const hub = hubs[f.t];
      const n = (count[f.t] = (count[f.t] || 0) + 1) - 1;
      const spread = compact ? 0 : n === 0 ? -0.32 : 0.32;
      const off = Math.min(w, h) * (compact ? 0.13 : 0.12);
      return {
        f,
        sx: w * (0.08 + rand() * 0.84),
        sy: h * (0.08 + rand() * 0.84),
        rot: (rand() - 0.5) * 34,
        tx: hub.x + Math.cos(hub.a + spread) * off,
        ty: hub.y + Math.sin(hub.a + spread) * off,
        hub,
      };
    });
    return { hubs, docs };
  }, [size, filings, compact]);

  const cardRefs = useRef<(HTMLDivElement | null)[]>([]);
  const dotRefs = useRef<(HTMLDivElement | null)[]>([]);
  const lineRefs = useRef<(SVGLineElement | null)[]>([]);
  const hubRefs = useRef<(HTMLDivElement | null)[]>([]);
  const hubLineRefs = useRef<(SVGLineElement | null)[]>([]);
  const progress = useRef(reduced ? 1 : 0);
  const lastCaption = useRef(0);

  const apply = (p: number) => {
    progress.current = p;
    const move = easeInOut(range(p, 0.06, 0.55));
    const cardFade = 1 - range(p, 0.42, 0.58);
    const dotIn = range(p, 0.44, 0.6);
    const cw = compact ? 72 : 104, ch = compact ? 92 : 132;
    geo.docs.forEach((d, i) => {
      const c = cardRefs.current[i];
      if (c) {
        if (cardFade <= 0.001) {
          c.style.opacity = "0";
        } else {
          const x = lerp(d.sx, d.tx, move);
          const y = lerp(d.sy, d.ty, move);
          const s = lerp(1, 0.18, move);
          const r = lerp(d.rot, 0, move);
          c.style.transform = `translate3d(${(x - cw / 2).toFixed(1)}px, ${(y - ch / 2).toFixed(1)}px, 0) rotate(${r.toFixed(2)}deg) scale(${s.toFixed(3)})`;
          c.style.opacity = cardFade.toFixed(3);
        }
      }
      const dot = dotRefs.current[i];
      if (dot) {
        dot.style.transform = `translate3d(${(d.tx - 4).toFixed(1)}px, ${(d.ty - 4).toFixed(1)}px, 0) scale(${(0.4 + dotIn * 0.6).toFixed(3)})`;
        dot.style.opacity = dotIn.toFixed(3);
      }
      const ln = lineRefs.current[i];
      if (ln) ln.style.strokeDashoffset = (1 - range(p, 0.55 + (i % 10) * 0.012, 0.74 + (i % 10) * 0.012)).toFixed(3);
    });
    const hubIn = easeInOut(range(p, 0.5, 0.66));
    HOME_TICKERS.forEach((_, i) => {
      const h = hubRefs.current[i];
      if (h) {
        h.style.opacity = hubIn.toFixed(3);
        h.style.setProperty("--s", (0.6 + hubIn * 0.4).toFixed(3));
      }
    });
    hubLineRefs.current.forEach((ln, i) => {
      if (ln) ln.style.strokeDashoffset = (1 - range(p, 0.72 + i * 0.01, 0.9 + i * 0.01)).toFixed(3);
    });
    const nextCap = p < 0.36 ? 0 : p < 0.7 ? 1 : 2;
    if (nextCap !== lastCaption.current) {
      lastCaption.current = nextCap;
      setCaption(nextCap);
    }
  };

  const wrapRef = useScrollProgress<HTMLDivElement>(apply, "sticky", reduced);

  // Re-apply after layout changes so positions match the new stage size.
  useEffect(() => {
    apply(reduced ? 1 : progress.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geo, reduced]);

  return (
    <section id="problem" aria-labelledby="problem-title" className="relative">
      <div ref={wrapRef} className={reduced ? "relative" : "relative h-[240vh] lg:h-[280vh]"}>
        <div className={reduced ? "relative py-24" : "sticky top-0 h-[100svh] overflow-hidden"}>
          <div className="fgh-container h-full grid lg:grid-cols-12 gap-6 lg:gap-10 items-center pt-24 pb-10 lg:py-0">
            <div className="lg:col-span-5 relative z-10">
              <p className="fgh-eyebrow">
                <span>01</span> The problem
              </p>
              <h2 id="problem-title" className="fgh-h2 fgh-h2--md mt-5">
                <SplitText text="Financial intelligence is scattered across thousands of disclosures." />
              </h2>
              <div className="mt-6 lg:mt-8 relative min-h-[5.5rem]">
                {CAPTIONS.map((c, i) => (
                  <p
                    key={c.k}
                    className="fgh-caption absolute inset-x-0 top-0"
                    data-active={reduced || caption === i ? "" : undefined}
                    aria-hidden={!(reduced ? i === 2 : caption === i)}
                    style={reduced && i !== 2 ? { display: "none" } : undefined}
                  >
                    <span className="font-mono text-[11px] text-[color:var(--h-blue)] mr-3">{c.k}</span>
                    {c.text}
                  </p>
                ))}
              </div>
              <div className="hidden lg:flex gap-1.5 mt-4" aria-hidden="true">
                {CAPTIONS.map((c, i) => (
                  <span key={c.k} className="h-[2px] w-10 rounded-full transition-colors duration-500" style={{ background: caption >= i || reduced ? "var(--h-blue)" : "var(--h-line-2)" }} />
                ))}
              </div>
            </div>

            <div className="lg:col-span-7 relative h-[52svh] sm:h-[56svh] lg:h-[76svh]">
              <div ref={stageRef} className="absolute inset-0" aria-hidden="true">
                <svg className="absolute inset-0 w-full h-full" width={size.w} height={size.h}>
                  {geo.docs.map((d, i) => (
                    <line
                      key={`l${i}`}
                      ref={(el) => {
                        lineRefs.current[i] = el;
                      }}
                      x1={d.tx}
                      y1={d.ty}
                      x2={d.hub.x}
                      y2={d.hub.y}
                      pathLength={1}
                      strokeDasharray="1"
                      strokeDashoffset="1"
                      stroke="rgba(96,148,255,0.45)"
                      strokeWidth={1}
                    />
                  ))}
                  {hubLinks.map(([a, b], i) => {
                    const A = geo.hubs[a], B = geo.hubs[b];
                    if (!A || !B) return null;
                    const mx = (A.x + B.x) / 2, my = (A.y + B.y) / 2;
                    const qx = mx + (size.w / 2 - mx) * 0.55, qy = my + (size.h / 2 - my) * 0.55;
                    return (
                      <path
                        key={`h${a}${b}`}
                        ref={(el) => {
                          hubLineRefs.current[i] = el as unknown as SVGLineElement;
                        }}
                        d={`M${A.x},${A.y} Q${qx},${qy} ${B.x},${B.y}`}
                        fill="none"
                        pathLength={1}
                        strokeDasharray="1"
                        strokeDashoffset="1"
                        stroke="rgba(94,200,222,0.5)"
                        strokeWidth={1}
                      />
                    );
                  })}
                </svg>

                {geo.docs.map((d, i) => (
                  <div
                    key={`c${i}`}
                    ref={(el) => {
                      cardRefs.current[i] = el;
                    }}
                    className="fgh-doc"
                    style={{ width: compact ? 72 : 104, height: compact ? 92 : 132 }}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-[9px] sm:text-[10px] text-[color:var(--h-fg-3)]">{d.f.t}</span>
                      <span className="w-1.5 h-1.5 rounded-full bg-[color:var(--h-blue)] opacity-70" />
                    </div>
                    <div className="text-[13px] sm:text-[17px] font-semibold tracking-tight text-[color:var(--h-fg)] mt-1">{d.f.f}</div>
                    <div className="font-mono text-[8px] sm:text-[9px] text-[color:var(--h-fg-3)]">{d.f.d}</div>
                    <div className="mt-auto space-y-1">
                      <span className="block h-[3px] rounded bg-white/10 w-full" />
                      <span className="block h-[3px] rounded bg-white/10 w-4/5" />
                      <span className="block h-[3px] rounded bg-white/10 w-3/5" />
                    </div>
                  </div>
                ))}

                {geo.docs.map((d, i) => (
                  <div
                    key={`d${i}`}
                    ref={(el) => {
                      dotRefs.current[i] = el;
                    }}
                    className="absolute left-0 top-0 w-2 h-2 rounded-full bg-[color:var(--h-blue)] opacity-0"
                    style={{ boxShadow: "0 0 0 3px rgba(61,123,255,0.15)" }}
                  />
                ))}

                {HOME_TICKERS.map((t, i) => {
                  const h = geo.hubs[t];
                  if (!h) return null;
                  return (
                    <div
                      key={t}
                      ref={(el) => {
                        hubRefs.current[i] = el;
                      }}
                      className="fgh-hub"
                      style={{ left: h.x, top: h.y, opacity: 0 }}
                    >
                      {t}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
