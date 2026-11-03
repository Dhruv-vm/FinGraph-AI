import React from "react";
import { cx } from "./cx";

/** Bordered surface. Use sparingly — prefer rules and whitespace inside it. */
export function Panel({
  className,
  children,
  as: As = "section",
  ...rest
}: React.HTMLAttributes<HTMLElement> & { as?: "section" | "div" | "aside" | "article" }) {
  return (
    <As className={cx("card", className)} {...rest}>
      {children}
    </As>
  );
}

export function PanelHeader({
  title,
  description,
  actions,
  className,
  id,
}: {
  title: React.ReactNode;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
  id?: string;
}) {
  return (
    <header className={cx("flex flex-wrap items-start justify-between gap-x-4 gap-y-2 px-4 pt-3.5 pb-3", className)}>
      <div className="min-w-0">
        <h2 id={id} className="text-sm font-semibold text-fg">
          {title}
        </h2>
        {description && <p className="text-xs text-fg-3 mt-0.5">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
    </header>
  );
}

/** Section heading used directly on the page canvas (no box). */
export function SectionHeading({
  title,
  description,
  actions,
  className,
  id,
}: {
  title: React.ReactNode;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
  id?: string;
}) {
  return (
    <div className={cx("flex flex-wrap items-end justify-between gap-3 mb-3", className)}>
      <div className="min-w-0">
        <h2 id={id} className="text-md font-semibold text-fg">
          {title}
        </h2>
        {description && <p className="text-sm text-fg-2 mt-0.5 max-w-[72ch]">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

/** Label/value ledger row list. */
export function KeyValue({
  items,
  className,
  dense,
}: {
  items: { label: React.ReactNode; value: React.ReactNode; mono?: boolean; hint?: React.ReactNode }[];
  className?: string;
  dense?: boolean;
}) {
  return (
    <dl className={cx("divide-y divide-line-soft", className)}>
      {items.map((it, i) => (
        <div key={i} className={cx("grid grid-cols-[minmax(7rem,40%)_1fr] gap-3", dense ? "py-1.5" : "py-2")}>
          <dt className="text-xs text-fg-3 pt-px">
            {it.label}
            {it.hint && <div className="text-2xs text-fg-3/80 mt-0.5">{it.hint}</div>}
          </dt>
          <dd className={cx("text-sm text-fg min-w-0 break-words", it.mono && "font-mono-num text-xs pt-0.5")}>{it.value}</dd>
        </div>
      ))}
    </dl>
  );
}

/** A single figure with label — typographic, not a card. */
export function Metric({
  label,
  value,
  detail,
  tone,
  className,
}: {
  label: React.ReactNode;
  value: React.ReactNode;
  detail?: React.ReactNode;
  tone?: "pos" | "neg" | "warn";
  className?: string;
}) {
  return (
    <div className={cx("min-w-0", className)}>
      <div className="text-xs text-fg-3">{label}</div>
      <div
        className={cx(
          "num text-2xl font-semibold tracking-tight mt-1",
          tone === "pos" ? "text-pos" : tone === "neg" ? "text-neg" : tone === "warn" ? "text-warn" : "text-fg"
        )}
      >
        {value}
      </div>
      {detail && <div className="text-xs text-fg-2 mt-1">{detail}</div>}
    </div>
  );
}

/** Container for charts: consistent padding, caption and source line. */
export function ChartFrame({
  title,
  caption,
  source,
  children,
  className,
}: {
  title: React.ReactNode;
  caption?: React.ReactNode;
  source?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <figure className={cx("card p-5 flex flex-col", className)}>
      <figcaption className="mb-3">
        <div className="text-sm font-semibold text-fg">{title}</div>
        {caption && <div className="text-xs text-fg-3 mt-1 leading-relaxed">{caption}</div>}
      </figcaption>
      <div className="flex-1 min-h-0">{children}</div>
      {source && <div className="text-2xs text-fg-3 mt-3 pt-2 border-t border-line-soft">{source}</div>}
    </figure>
  );
}
