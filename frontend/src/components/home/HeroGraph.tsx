"use client";

import React, { useEffect, useRef } from "react";
import { HERO_GRAPH } from "./homeData";

/**
 * Hero visual: the real company neighbourhoods from the FinGraph knowledge graph
 * (HERO_GRAPH — every edge exists in the source graph), laid out in 3D with a
 * small force simulation and rendered on a single canvas. Unlinked "dust"
 * particles add depth but carry no edges, so no relationship is implied.
 */

type V3 = { x: number; y: number; z: number };
interface N extends V3 {
  id: string;
  label: string;
  ticker: string | null;
  center: boolean;
  vx: number;
  vy: number;
  vz: number;
}

function rng(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function layout(): { nodes: N[]; edges: [number, number][] } {
  const rand = rng(7);
  const index = new Map<string, number>();
  const nodes: N[] = HERO_GRAPH.nodes.map((n, i) => {
    index.set(n.id, i);
    const m = /^company:([A-Z]{1,5})$/.exec(n.id);
    const a = rand() * Math.PI * 2;
    const b = Math.acos(2 * rand() - 1);
    const r = 0.4 + rand() * 0.6;
    return {
      id: n.id,
      label: n.l,
      ticker: m ? m[1] : null,
      center: !!n.c,
      x: r * Math.sin(b) * Math.cos(a),
      y: r * Math.sin(b) * Math.sin(a),
      z: r * Math.cos(b),
      vx: 0,
      vy: 0,
      vz: 0,
    };
  });
  const edges = HERO_GRAPH.edges
    .map(([s, t]) => [index.get(s), index.get(t)] as [number | undefined, number | undefined])
    .filter((e): e is [number, number] => e[0] !== undefined && e[1] !== undefined);

  // Small 3D force simulation (runs once, ~50 nodes).
  for (let it = 0; it < 320; it++) {
    const alpha = 1 - it / 320;
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        let dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        const d2 = dx * dx + dy * dy + dz * dz + 0.0025;
        const f = (0.012 * alpha) / d2;
        dx *= f; dy *= f; dz *= f;
        a.vx += dx; a.vy += dy; a.vz += dz;
        b.vx -= dx; b.vy -= dy; b.vz -= dz;
      }
    }
    for (const [s, t] of edges) {
      const a = nodes[s], b = nodes[t];
      const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.001;
      const f = ((d - 0.32) / d) * 0.08 * alpha;
      a.vx += dx * f; a.vy += dy * f; a.vz += dz * f;
      b.vx -= dx * f; b.vy -= dy * f; b.vz -= dz * f;
    }
    for (const n of nodes) {
      n.vx -= n.x * 0.02 * alpha; n.vy -= n.y * 0.02 * alpha; n.vz -= n.z * 0.02 * alpha;
      n.x += n.vx; n.y += n.vy; n.z += n.vz;
      n.vx *= 0.6; n.vy *= 0.6; n.vz *= 0.6;
    }
  }
  // Normalise to a unit sphere.
  let max = 0;
  for (const n of nodes) max = Math.max(max, Math.hypot(n.x, n.y, n.z));
  for (const n of nodes) {
    n.x /= max; n.y /= max; n.z /= max;
  }
  return { nodes, edges };
}

export function HeroGraph({ reduced }: { reduced: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { nodes, edges } = layout();
    const rand = rng(11);
    let W = 0, H = 0, dpr = 1, mode: "desktop" | "tablet" | "mobile" = "desktop";
    let rectLeft = 0, rectTop = 0, winW = window.innerWidth, winH = window.innerHeight;
    let cachedScrollY = window.scrollY;
    let dust: V3[] = [];

    const updateRect = () => {
      const r = canvas.getBoundingClientRect();
      rectLeft = r.left;
      rectTop = r.top;
      return r;
    };

    const resize = () => {
      const r = updateRect();
      winW = window.innerWidth;
      winH = window.innerHeight;
      W = r.width;
      H = r.height;
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      mode = W >= 1024 ? "desktop" : W >= 640 ? "tablet" : "mobile";
      const count = mode === "desktop" ? 140 : mode === "tablet" ? 80 : 40;
      dust = Array.from({ length: count }, () => {
        const a = rand() * Math.PI * 2, b = Math.acos(2 * rand() - 1), r = 1.15 + rand() * 1.1;
        return { x: r * Math.sin(b) * Math.cos(a), y: r * Math.sin(b) * Math.sin(a) * 0.7, z: r * Math.cos(b) };
      });
    };
    resize();

    // pointer tilt
    let tx = 0, ty = 0, px = 0, py = 0;
    let hoverX = -1e4, hoverY = -1e4;
    const onPointer = (e: PointerEvent) => {
      if (e.pointerType !== "mouse") return;
      tx = (e.clientX / winW - 0.5) * 2;
      ty = (e.clientY / winH - 0.5) * 2;
      hoverX = e.clientX - rectLeft;
      hoverY = e.clientY - rectTop;
    };
    window.addEventListener("pointermove", onPointer, { passive: true });

    const pulses: { e: number; t: number; speed: number }[] = [];
    let lastSpawn = 0;
    let rot = 0.6;
    let raf = 0;
    let running = false;
    let last = performance.now();
    const proj = new Float32Array(nodes.length * 3);
    const tmp = new Float32Array(3);
    const order = nodes.map((_, i) => i);

    const draw = (now: number) => {
      const dt = Math.min(64, now - last);
      last = now;
      if (!reduced) rot += dt * 0.000055;
      px += (tx - px) * 0.04;
      py += (ty - py) * 0.04;

      const scrollP = Math.min(1, cachedScrollY / Math.max(1, H));
      const cx = mode === "desktop" ? W * 0.67 : W * 0.5;
      const cy = (mode === "desktop" ? H * 0.5 : H * 0.6) + cachedScrollY * 0.35;
      const R = (mode === "desktop" ? Math.min(W * 0.33, H * 0.46) : mode === "tablet" ? Math.min(W * 0.46, H * 0.36) : W * 0.6) * (1 + scrollP * 0.18);
      const baseAlpha = (mode === "desktop" ? 1 : mode === "tablet" ? 0.7 : 0.55) * (1 - scrollP * 0.75);

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      if (baseAlpha <= 0.01) return;

      const ry = rot + px * 0.25;
      const rx = -0.18 + py * 0.12;
      const cyR = Math.cos(ry), syR = Math.sin(ry), cxR = Math.cos(rx), sxR = Math.sin(rx);
      const D = 3.1;
      const project = (p: V3, out: Float32Array, k: number) => {
        const x1 = p.x * cyR + p.z * syR;
        const z1 = -p.x * syR + p.z * cyR;
        const y1 = p.y * cxR - z1 * sxR;
        const z2 = p.y * sxR + z1 * cxR;
        const s = D / (D - z2);
        out[k] = cx + x1 * s * R;
        out[k + 1] = cy + y1 * s * R;
        out[k + 2] = z2;
      };

      // dust (depth layer, no edges)
      for (const d of dust) {
        project(d, tmp, 0);
        const depth = (tmp[2] + 2.3) / 4.6;
        ctx.fillStyle = `rgba(148,176,230,${(0.06 + depth * 0.22) * baseAlpha})`;
        ctx.fillRect(tmp[0], tmp[1], 1.1 + depth, 1.1 + depth);
      }

      for (let i = 0; i < nodes.length; i++) project(nodes[i], proj, i * 3);

      // edges
      ctx.lineWidth = 1;
      for (const [s, t] of edges) {
        const z = (proj[s * 3 + 2] + proj[t * 3 + 2]) / 2;
        const depth = (z + 1) / 2;
        ctx.strokeStyle = `rgba(86,140,255,${(0.07 + depth * 0.26) * baseAlpha})`;
        ctx.beginPath();
        ctx.moveTo(proj[s * 3], proj[s * 3 + 1]);
        ctx.lineTo(proj[t * 3], proj[t * 3 + 1]);
        ctx.stroke();
      }

      // pulses travelling along real edges
      if (!reduced && edges.length) {
        if (now - lastSpawn > 520 && pulses.length < (mode === "mobile" ? 3 : 6)) {
          pulses.push({ e: Math.floor(rand() * edges.length), t: 0, speed: 0.00045 + rand() * 0.0004 });
          lastSpawn = now;
        }
        for (let i = pulses.length - 1; i >= 0; i--) {
          const p = pulses[i];
          p.t += dt * p.speed;
          if (p.t >= 1) {
            pulses.splice(i, 1);
            continue;
          }
          const [s, t] = edges[p.e];
          const x = proj[s * 3] + (proj[t * 3] - proj[s * 3]) * p.t;
          const y = proj[s * 3 + 1] + (proj[t * 3 + 1] - proj[s * 3 + 1]) * p.t;
          const fade = Math.sin(p.t * Math.PI);
          ctx.fillStyle = `rgba(94,200,222,${0.12 * fade * baseAlpha})`;
          ctx.beginPath();
          ctx.arc(x, y, 5, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = `rgba(180,236,248,${0.9 * fade * baseAlpha})`;
          ctx.beginPath();
          ctx.arc(x, y, 1.4, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // nodes (painter's order, reuse pre-allocated index array)
      order.sort((a, b) => proj[a * 3 + 2] - proj[b * 3 + 2]);
      let hovered = -1;
      let best = 26 * 26;
      for (const i of order) {
        const n = nodes[i];
        const x = proj[i * 3], y = proj[i * 3 + 1], z = proj[i * 3 + 2];
        const depth = (z + 1) / 2;
        const a = (0.35 + depth * 0.65) * baseAlpha;
        const r = (n.center ? 3.6 : 1.7) * (0.75 + depth * 0.5);
        if (n.center) {
          ctx.strokeStyle = `rgba(96,148,255,${a * 0.55})`;
          ctx.beginPath();
          ctx.arc(x, y, r + 4.5, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.fillStyle = n.center ? `rgba(206,222,255,${a})` : `rgba(120,160,240,${a * 0.9})`;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
        const d2 = (x - hoverX) ** 2 + (y - hoverY) ** 2;
        if (d2 < best) {
          best = d2;
          hovered = i;
        }
        if (n.center && n.ticker && mode !== "mobile") {
          ctx.font = "500 10.5px var(--font-mono), ui-monospace, monospace";
          ctx.fillStyle = `rgba(170,186,214,${a * 0.9})`;
          ctx.fillText(n.ticker, x + r + 8, y + 3.5);
        }
      }
      if (hovered >= 0 && mode === "desktop") {
        const n = nodes[hovered];
        const x = proj[hovered * 3], y = proj[hovered * 3 + 1];
        ctx.font = "500 12px var(--font-sans), ui-sans-serif, system-ui";
        const label = n.label.length > 34 ? n.label.slice(0, 33) + "…" : n.label;
        const w = ctx.measureText(label).width + 16;
        ctx.fillStyle = `rgba(10,14,22,${0.85 * baseAlpha})`;
        ctx.strokeStyle = `rgba(96,148,255,${0.45 * baseAlpha})`;
        ctx.beginPath();
        ctx.roundRect?.(x + 10, y - 26, w, 22, 6);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = `rgba(232,236,242,${baseAlpha})`;
        ctx.fillText(label, x + 18, y - 11);
      }
    };

    const loop = (now: number) => {
      draw(now);
      if (running) raf = requestAnimationFrame(loop);
    };
    const start = () => {
      if (running || reduced) return;
      running = true;
      last = performance.now();
      raf = requestAnimationFrame(loop);
    };
    const stop = () => {
      running = false;
      cancelAnimationFrame(raf);
    };

    const vis = new IntersectionObserver(([e]) => (e.isIntersecting && !document.hidden ? start() : stop()));
    vis.observe(canvas);
    const onVis = () => (document.hidden ? stop() : start());
    document.addEventListener("visibilitychange", onVis);
    const ro = new ResizeObserver(() => {
      resize();
      if (!running) draw(performance.now());
    });
    ro.observe(canvas);
    // Keep cachedScrollY fresh without calling getBoundingClientRect() in draw.
    const onScroll = () => {
      cachedScrollY = window.scrollY;
      if (!running) draw(performance.now());
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    draw(performance.now());

    return () => {
      stop();
      vis.disconnect();
      ro.disconnect();
      document.removeEventListener("visibilitychange", onVis);
      window.removeEventListener("pointermove", onPointer);
      window.removeEventListener("scroll", onScroll);
    };
  }, [reduced]);

  return <canvas ref={canvasRef} aria-hidden="true" className="absolute inset-0 w-full h-full" />;
}
