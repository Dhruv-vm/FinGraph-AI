"use client";

/**
 * Lightweight motion primitives for the public homepage.
 *
 * No animation library is installed in this project, so instead of adding
 * one these use native browser APIs: a single rAF-throttled scroll bus,
 * a shared IntersectionObserver and CSS transitions. Everything degrades to
 * a static, fully readable page under prefers-reduced-motion.
 */

import React, { useEffect, useLayoutEffect, useRef, useState } from "react";

const useIsoLayoutEffect = typeof window !== "undefined" ? useLayoutEffect : useEffect;

/* ---------------------------------------------------------------- reduced motion */

export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(mq.matches);
    update();
    mq.addEventListener?.("change", update);
    return () => mq.removeEventListener?.("change", update);
  }, []);
  return reduced;
}

export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/* ---------------------------------------------------------------- scroll bus */

type Sub = () => void;
const subs = new Set<Sub>();
let ticking = false;
let bound = false;

function flush() {
  ticking = false;
  subs.forEach((fn) => fn());
}
function schedule() {
  if (!ticking) {
    ticking = true;
    requestAnimationFrame(flush);
  }
}

/** Run `fn` once per animation frame in which the page scrolled or resized. */
export function subscribeScroll(fn: Sub): () => void {
  subs.add(fn);
  if (!bound && typeof window !== "undefined") {
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule, { passive: true });
    bound = true;
  }
  requestAnimationFrame(fn);
  return () => {
    subs.delete(fn);
  };
}

export const clamp = (v: number, a = 0, b = 1) => Math.min(b, Math.max(a, v));
/** Map p from [a,b] to [0,1], clamped. */
export const range = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
export const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);
export const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/**
 * Scroll progress of an element.
 *  - "sticky": 0 when the element's top reaches the viewport top, 1 when its bottom reaches the viewport bottom
 *    (use on a tall wrapper around a position:sticky stage).
 *  - "through": 0 when the top enters the viewport bottom, 1 when the bottom leaves the top.
 * Writes `--p` on the element and calls `onProgress` without re-rendering React.
 */
export function useScrollProgress<T extends HTMLElement>(
  onProgress: (p: number) => void,
  mode: "sticky" | "through" = "sticky",
  disabled = false
) {
  const ref = useRef<T>(null);
  const cb = useRef(onProgress);
  cb.current = onProgress;
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (disabled) {
      el.style.setProperty("--p", "1");
      cb.current(1);
      return;
    }
    let last = -1;
    return subscribeScroll(() => {
      const r = el.getBoundingClientRect();
      const vh = window.innerHeight;
      if (r.bottom < -vh || r.top > vh * 2) return; // far away — skip work
      const p =
        mode === "sticky"
          ? clamp(-r.top / Math.max(1, r.height - vh))
          : clamp((vh - r.top) / Math.max(1, vh + r.height));
      if (Math.abs(p - last) < 0.0005) return;
      last = p;
      el.style.setProperty("--p", p.toFixed(4));
      cb.current(p);
    });
  }, [mode, disabled]);
  return ref;
}

/**
 * Multi-layer parallax: every descendant with data-speed="0.2" is translated
 * relative to its nearest <section>. Off on small screens and reduced motion.
 */
export function useParallax(rootRef: React.RefObject<HTMLElement>) {
  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const mqSmall = window.matchMedia("(max-width: 767px)");
    const els = Array.from(root.querySelectorAll<HTMLElement>("[data-speed]"));
    const items = els.map((el) => ({ el, speed: parseFloat(el.dataset.speed || "0"), host: el.closest("section") || root }));
    const off = () => items.forEach(({ el }) => (el.style.transform = ""));
    const unsub = subscribeScroll(() => {
      if (mqSmall.matches || prefersReducedMotion()) return off();
      const vh = window.innerHeight;
      for (const it of items) {
        const r = it.host.getBoundingClientRect();
        if (r.bottom < -200 || r.top > vh + 200) continue;
        const d = r.top + r.height / 2 - vh / 2;
        it.el.style.transform = `translate3d(0, ${(-d * it.speed).toFixed(1)}px, 0)`;
      }
    });
    return () => {
      unsub();
      off();
    };
  }, [rootRef]);
}

/* ---------------------------------------------------------------- in-view */

let io: IntersectionObserver | null = null;
const ioCallbacks = new WeakMap<Element, (inView: boolean) => void>();

function getObserver() {
  if (!io) {
    io = new IntersectionObserver(
      (entries) => entries.forEach((e) => ioCallbacks.get(e.target)?.(e.isIntersecting)),
      { rootMargin: "0px 0px -12% 0px", threshold: 0.01 }
    );
  }
  return io;
}

export function useInView<T extends Element>(once = true): [React.RefObject<T>, boolean] {
  const ref = useRef<T>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = getObserver();
    ioCallbacks.set(el, (v) => {
      if (v) {
        setInView(true);
        if (once) obs.unobserve(el);
      } else if (!once) setInView(false);
    });
    obs.observe(el);
    return () => obs.unobserve(el);
  }, [once]);
  return [ref, inView];
}

/** Fades and lifts its children into place the first time they enter the viewport. */
export function Reveal({
  as: Tag = "div",
  delay = 0,
  className,
  children,
  ...rest
}: {
  as?: keyof JSX.IntrinsicElements;
  delay?: number;
  className?: string;
  children?: React.ReactNode;
} & React.HTMLAttributes<HTMLElement>) {
  const [ref, inView] = useInView<HTMLElement>();
  const Comp = Tag as any;
  return (
    <Comp
      ref={ref}
      data-reveal=""
      data-shown={inView ? "" : undefined}
      className={className}
      style={{ ["--d" as any]: `${delay}ms`, ...(rest.style || {}) }}
      {...rest}
    >
      {children}
    </Comp>
  );
}

/** Staggered word-by-word reveal. Screen readers get the plain sentence. */
export function SplitText({
  text,
  className,
  delay = 0,
  step = 55,
  immediate = false,
}: {
  text: string;
  className?: string;
  delay?: number;
  step?: number;
  /** Animate on mount rather than on entering the viewport (hero). */
  immediate?: boolean;
}) {
  const [ref, inView] = useInView<HTMLSpanElement>();
  const [mounted, setMounted] = useState(false);
  useIsoLayoutEffect(() => {
    if (!immediate) return;
    const id = requestAnimationFrame(() => setMounted(true));
    return () => cancelAnimationFrame(id);
  }, [immediate]);
  const shown = immediate ? mounted : inView;
  const words = text.split(" ");
  return (
    <span ref={ref} className={className} data-split="" data-shown={shown ? "" : undefined}>
      <span className="sr-only">{text}</span>
      <span aria-hidden="true">
        {words.map((w, i) => (
          <React.Fragment key={i}>
            <span className="fgh-word">
              <span style={{ transitionDelay: `${delay + i * step}ms` }}>{w}</span>
            </span>
            {i < words.length - 1 ? " " : null}
          </React.Fragment>
        ))}
      </span>
    </span>
  );
}

/* ---------------------------------------------------------------- counters */

export function CountUp({
  value,
  decimals = 0,
  duration = 1800,
  suffix = "",
  className,
}: {
  value: number;
  decimals?: number;
  duration?: number;
  suffix?: string;
  className?: string;
}) {
  const [ref, inView] = useInView<HTMLSpanElement>();
  const [shown, setShown] = useState<number | null>(null); // null → render final value (SSR / reduced motion)
  const started = useRef(false);

  useEffect(() => {
    if (prefersReducedMotion()) return;
    if (!started.current) setShown(0);
  }, []);

  useEffect(() => {
    if (!inView || started.current || prefersReducedMotion()) return;
    started.current = true;
    const t0 = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const t = clamp((now - t0) / duration);
      const e = t === 1 ? 1 : 1 - Math.pow(2, -10 * t); // ease-out expo
      setShown(value * e);
      if (t < 1) raf = requestAnimationFrame(tick);
      else setShown(null);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [inView, value, duration]);

  const v = shown ?? value;
  const text = v.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  return (
    <span ref={ref} className={className}>
      <span aria-hidden="true">
        {text}
        {suffix}
      </span>
      <span className="sr-only">
        {value.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}
        {suffix}
      </span>
    </span>
  );
}
