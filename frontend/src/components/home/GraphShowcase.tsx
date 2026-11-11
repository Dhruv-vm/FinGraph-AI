"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { ArrowUpRight, ExternalLink } from "lucide-react";
import { HOME_TICKERS } from "./homeData";
import { HOME_SHOWCASE, HomeEdge, HomeNode } from "./showcaseData";
import { Reveal, SplitText, useInView } from "./motion";

/**
 * Section E — interactive neighbourhood preview built only from real edges in
 * the project's graph snapshot. Every relationship shown links back to the SEC
 * filing it was extracted from.
 */

export const REL_COLORS: Record<string, string> = {
  PARTNERS_WITH: "#5EC8DE",
  ACQUIRED: "#5EBA8C",
  COMPETES_WITH: "#E87A6E",
  INVESTED_IN: "#A896E6",
  HAS_RISK: "#E2B25C",
  AFFECTED_BY: "#D9A46C",
  SUPPLIES: "#78ACE8",
  MANUFACTURES: "#8FB8F0",
  DEPENDS_ON: "#C7A0D8",
};
const relColor = (r: string) => REL_COLORS[r] ?? "#808A9B";
const relLabel = (r: string) => r.toLowerCase().replace(/_/g, " ");
const sameRel = (a: HomeEdge | undefined, b: HomeEdge) => !!a && a.r === b.r && [a.s, a.t].sort().join() === [b.s, b.t].sort().join();
const short = (s: string, n: number) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

interface Placed extends HomeNode {
  x: number;
  y: number;
  side: "l" | "r" | "c";
}

function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, size] as const;
}

export function GraphShowcase() {
  const [ticker, setTicker] = useState<string>("NVDA");
  const data = HOME_SHOWCASE[ticker];
  const [hover, setHover] = useState<string | null>(null);
  const [selected, setSelected] = useState<string>(data.center);
  const [hoverEdge, setHoverEdge] = useState<number | null>(null);
  const [boxRef, size] = useSize<HTMLDivElement>();
  const [viewRef, inView] = useInView<HTMLDivElement>();

  useEffect(() => {
    setSelected(HOME_SHOWCASE[ticker].center);
    setHover(null);
    setHoverEdge(null);
  }, [ticker]);

  const compact = size.w > 0 && size.w < 640;

  const { placed, byId } = useMemo(() => {
    const { w, h } = size;
    const center = data.nodes[0];
    const relOf = (n: HomeNode) => data.edges.find((e) => (e.s === center.id && e.t === n.id) || (e.t === center.id && e.s === n.id))?.r ?? "~";
    const rest = data.nodes.slice(1).sort((a, b) => relOf(a).localeCompare(relOf(b)));
    const padX = compact ? 22 : 170;
    const rx = Math.max(60, w / 2 - padX);
    const ry = Math.max(60, h / 2 - (compact ? 30 : 42));
    const out: Placed[] = [{ ...center, x: w / 2, y: h / 2, side: "c" }];
    rest.forEach((n, i) => {
      const a = (i / rest.length) * Math.PI * 2 - Math.PI / 2 + 0.12;
      const k = i % 2 ? 0.8 : 1;
      const x = w / 2 + Math.cos(a) * rx * k;
      const y = h / 2 + Math.sin(a) * ry * k;
      out.push({ ...n, x, y, side: Math.cos(a) >= 0 ? "r" : "l" });
    });
    return { placed: out, byId: new Map(out.map((p) => [p.id, p])) };
  }, [data, size, compact]);

  // One drawn curve per (pair, relationship); repeated extractions of the same
  // relationship from several filings thicken the curve and are all listed in the inspector.
  const drawn = useMemo(() => {
    const m = new Map<string, { s: string; t: string; r: string; idx: number[] }>();
    data.edges.forEach((e, i) => {
      const k = [e.s, e.t].sort().join("|") + "|" + e.r;
      const g = m.get(k);
      if (g) g.idx.push(i);
      else m.set(k, { s: e.s, t: e.t, r: e.r, idx: [i] });
    });
    const list = Array.from(m.values());
    const groups = new Map<string, number[]>();
    list.forEach((d, i) => {
      const k = [d.s, d.t].sort().join("|");
      groups.set(k, [...(groups.get(k) || []), i]);
    });
    return list.map((d, i) => {
      const a = byId.get(d.s), b = byId.get(d.t);
      if (!a || !b) return { ...d, path: null as string | null };
      const g = groups.get([d.s, d.t].sort().join("|"))!;
      const k = g.indexOf(i) - (g.length - 1) / 2;
      const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
      const dx = b.x - a.x, dy = b.y - a.y;
      const len = Math.hypot(dx, dy) || 1;
      const nx = -dy / len, ny = dx / len;
      const isCenter = d.s === data.center || d.t === data.center;
      const pull = isCenter ? 0 : 0.35;
      const qx = mx + (size.w / 2 - mx) * pull + nx * k * 30;
      const qy = my + (size.h / 2 - my) * pull + ny * k * 30;
      return { ...d, path: `M${a.x.toFixed(1)},${a.y.toFixed(1)} Q${qx.toFixed(1)},${qy.toFixed(1)} ${b.x.toFixed(1)},${b.y.toFixed(1)}` };
    });
  }, [data, byId, size]);

  const active = hover ?? selected;
  const activeNode = byId.get(active) ?? placed[0];
  const incident = data.edges.map((e, i) => ({ e, i })).filter(({ e }) => e.s === active || e.t === active);
  const linked = new Set(incident.flatMap(({ e }) => [e.s, e.t]));
  const rels = Array.from(new Set(data.edges.map((e) => e.r)));

  const pick = (id: string) => setSelected(id);

  return (
    <section id="graph" aria-labelledby="graph-title" className="relative py-28 lg:py-40">
      <div className="fgh-orb fgh-orb--right" data-speed="0.1" aria-hidden="true" />
      <div className="fgh-container">
        <div className="grid lg:grid-cols-12 gap-6 items-end">
          <div className="lg:col-span-7">
            <p className="fgh-eyebrow">
              <span>05</span> Knowledge graph
            </p>
            <h2 id="graph-title" className="fgh-h2 mt-5">
              <SplitText text="Every edge, traceable to a filing." />
            </h2>
          </div>
          <Reveal as="p" className="lg:col-span-5 fgh-lead">
            A live slice of the FinGraph knowledge graph. Choose a company, then hover or tap an entity to see the relationships extracted
            from SEC filings — and where each one came from.
          </Reveal>
        </div>

        <div role="tablist" aria-label="Company" className="fgh-tabs mt-10">
          {HOME_TICKERS.map((t) => (
            <button
              key={t}
              role="tab"
              aria-selected={ticker === t}
              aria-controls="graph-panel"
              onClick={() => setTicker(t)}
              className="fgh-tab"
              data-active={ticker === t ? "" : undefined}
            >
              {t}
            </button>
          ))}
        </div>

        <div id="graph-panel" role="tabpanel" aria-label={`${data.name} neighbourhood`} ref={viewRef} className="fgh-card fgh-card--graph mt-4 grid lg:grid-cols-[1fr_340px]">
          <div ref={boxRef} className="relative h-[380px] sm:h-[480px] lg:h-[600px] min-w-0 fgh-graph-bg">
            {size.w > 0 && (
              <svg width={size.w} height={size.h} className="absolute inset-0" data-in={inView ? "" : undefined}>
                <g key={ticker}>
                  {drawn.map((d, i) => {
                    if (!d.path) return null;
                    const hot = hoverEdge !== null && d.idx.includes(hoverEdge);
                    const on = d.s === active || d.t === active || hot;
                    return (
                      <g key={i}>
                        <path
                          d={d.path}
                          className="fgh-edge"
                          pathLength={1}
                          style={{ ["--i" as any]: i, stroke: relColor(d.r), opacity: on ? 0.95 : active ? 0.14 : 0.4 }}
                          strokeWidth={(on ? 1.8 : 1) + Math.min(2, (d.idx.length - 1) * 0.5)}
                        />
                        <path
                          d={d.path}
                          stroke="transparent"
                          strokeWidth={12}
                          fill="none"
                          onMouseEnter={() => setHoverEdge(d.idx[0])}
                          onMouseLeave={() => setHoverEdge(null)}
                        />
                      </g>
                    );
                  })}
                  {placed.map((n, i) => {
                    const isCenter = i === 0;
                    const on = n.id === active;
                    const dim = active && !linked.has(n.id) && !on;
                    const showLabel = !compact || isCenter || on || (active !== data.center && linked.has(n.id));
                    // On small screens labels point toward the centre so they stay inside the canvas.
                    const toRight = compact ? n.side === "l" : n.side === "r";
                    const r = isCenter ? 9 : 5 + Math.min(3, Math.log10(Math.max(1, n.deg)));
                    return (
                      <g
                        key={n.id}
                        className="fgh-node"
                        transform={`translate(${n.x.toFixed(1)},${n.y.toFixed(1)})`}
                        style={{ opacity: dim ? 0.35 : 1 }}
                        role="button"
                        tabIndex={0}
                        aria-label={`${n.l}${isCenter ? " (selected company)" : ""}`}
                        aria-pressed={selected === n.id}
                        onMouseEnter={() => setHover(n.id)}
                        onMouseLeave={() => setHover(null)}
                        onFocus={() => setHover(n.id)}
                        onBlur={() => setHover(null)}
                        // pointerup (not click): on touch, focusing the node re-renders the SVG between
                        // press and release, which retargets the synthetic click to the container.
                        onPointerUp={(ev) => {
                          if (ev.button === 0 || ev.pointerType !== "mouse") pick(n.id);
                        }}
                        onKeyDown={(ev) => {
                          if (ev.key === "Enter" || ev.key === " ") {
                            ev.preventDefault();
                            pick(n.id);
                          }
                        }}
                      >
                        <circle r={22} fill="#000" fillOpacity={0} pointerEvents="all" />
                        <g className="fgh-node-in" style={{ ["--i" as any]: i }} pointerEvents="none">
                        {(on || isCenter) && <circle r={r + 6} fill="none" stroke={isCenter ? "rgba(96,148,255,0.5)" : "rgba(94,200,222,0.6)"} />}
                        <circle r={r} fill={isCenter ? "#3D7BFF" : on ? "#CFE6FF" : "#1B2638"} stroke={isCenter ? "#9DBCFF" : "#5E7BA8"} strokeWidth={1.2} />
                        {showLabel && (
                          <text
                            x={isCenter ? 0 : toRight ? r + 8 : -(r + 8)}
                            y={isCenter ? -(r + 12) : 4}
                            textAnchor={isCenter ? "middle" : toRight ? "start" : "end"}
                            className={isCenter ? "fgh-label fgh-label--center" : "fgh-label"}
                            data-on={on ? "" : undefined}
                          >
                            {short(n.l, compact ? 18 : 26)}
                          </text>
                        )}
                        </g>
                      </g>
                    );
                  })}
                </g>
              </svg>
            )}
            <div className="hidden sm:flex absolute left-4 bottom-4 right-4 flex-wrap gap-x-3 gap-y-1.5 pointer-events-none">
              {rels.map((r) => (
                <span key={r} className="inline-flex items-center gap-1.5 text-[10.5px] text-[color:var(--h-fg-3)]">
                  <span className="w-2.5 h-[2px] rounded" style={{ background: relColor(r) }} />
                  {relLabel(r)}
                </span>
              ))}
            </div>
          </div>

          <div className="sm:hidden flex flex-wrap gap-x-3 gap-y-1.5 px-5 py-3 border-t border-[color:var(--h-line)]">
            {rels.map((r) => (
              <span key={r} className="inline-flex items-center gap-1.5 text-[10.5px] text-[color:var(--h-fg-3)]">
                <span className="w-2.5 h-[2px] rounded" style={{ background: relColor(r) }} />
                {relLabel(r)}
              </span>
            ))}
          </div>
          <aside className="border-t lg:border-t-0 lg:border-l border-[color:var(--h-line)] p-5 sm:p-6 flex flex-col min-h-[320px] lg:max-h-[600px]">
            <div className="text-[11px] font-mono text-[color:var(--h-fg-3)] uppercase tracking-[0.12em]">Inspecting</div>
            <div className="text-[18px] font-medium tracking-tight text-[color:var(--h-fg)] mt-2 leading-snug">{activeNode.l}</div>
            <div className="text-[12px] text-[color:var(--h-fg-3)] mt-1">
              {activeNode.ty} · degree {activeNode.deg.toLocaleString("en-US")} in the full graph
            </div>
            <div className="mt-5 text-[11px] font-mono text-[color:var(--h-fg-3)] uppercase tracking-[0.12em]">
              {incident.length} relationship{incident.length === 1 ? "" : "s"} in this view
            </div>
            <ul className="mt-3 space-y-2 overflow-y-auto fgh-scroll -mr-2 pr-2 flex-1">
              {incident.map(({ e, i }) => (
                <EdgeRow key={i} e={e} nameOf={(id) => byId.get(id)?.l ?? id} hot={hoverEdge !== null && sameRel(data.edges[hoverEdge], e)} />
              ))}
            </ul>
            <a href="/workspace#/graph" className="fgh-link mt-5 self-start">
              Open the full Graph Explorer <ArrowUpRight className="w-3.5 h-3.5" />
            </a>
          </aside>
        </div>
        <p className="text-[12px] text-[color:var(--h-fg-3)] mt-4">
          Showing {data.edges.length} of {data.total_edges} relationships around {data.name} in the bundled graph snapshot. Edges are as
          extracted by the pipeline; the source filing may belong to another issuer.
        </p>
      </div>
    </section>
  );
}

function EdgeRow({ e, nameOf, hot }: { e: HomeEdge; nameOf: (id: string) => string; hot: boolean }) {
  return (
    <li className="fgh-edge-row" data-hot={hot ? "" : undefined}>
      <div className="text-[13px] leading-snug text-[color:var(--h-fg)]">
        {short(nameOf(e.s), 30)}{" "}
        <span className="font-mono text-[10.5px] px-1.5 py-0.5 rounded" style={{ color: relColor(e.r), background: `${relColor(e.r)}1a` }}>
          {e.r}
        </span>{" "}
        {short(nameOf(e.t), 30)}
      </div>
      <div className="text-[11px] text-[color:var(--h-fg-3)] mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1">
        {e.src && <span>{e.src}</span>}
        {e.pub && <span className="font-mono">{e.pub}</span>}
        {typeof e.c === "number" && <span className="font-mono">conf {e.c.toFixed(2)}</span>}
        {e.url && (
          <a href={e.url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-[color:var(--h-blue-text)] hover:underline">
            SEC filing <ExternalLink className="w-3 h-3" />
            <span className="sr-only">(opens in a new tab)</span>
          </a>
        )}
      </div>
    </li>
  );
}
