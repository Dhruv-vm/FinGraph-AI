import React from "react";
import { cx } from "./cx";

export interface SegmentOption<T extends string> {
  value: T;
  label: React.ReactNode;
  count?: number;
}

/** Compact segmented control (radio group semantics). */
export function Segmented<T extends string>({
  value,
  onChange,
  options,
  label,
  size = "sm",
  className,
}: {
  value: T;
  onChange: (v: T) => void;
  options: SegmentOption<T>[];
  label: string;
  size?: "sm" | "md";
  className?: string;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={cx("inline-flex p-1 rounded-full bg-surface-2 border border-line-soft", className)}
    >
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button
            key={o.value}
            role="radio"
            aria-checked={on}
            onClick={() => onChange(o.value)}
            className={cx(
              "inline-flex items-center gap-1.5 rounded-full font-medium transition-all duration-200 whitespace-nowrap",
              size === "sm" ? "h-7 px-3 text-xs" : "h-8 px-4 text-sm",
              on ? "bg-accent-strong text-accent-fg shadow-glow" : "text-fg-2 hover:text-fg"
            )}
          >
            {o.label}
            {o.count != null && <span className={cx("num", on ? "text-accent-fg/70" : "text-fg-3")}>{o.count}</span>}
          </button>
        );
      })}
    </div>
  );
}

/** Underlined tabs for switching panels within a page. */
export function Tabs<T extends string>({
  value,
  onChange,
  options,
  label,
  className,
}: {
  value: T;
  onChange: (v: T) => void;
  options: SegmentOption<T>[];
  label: string;
  className?: string;
}) {
  return (
    <div role="tablist" aria-label={label} className={cx("flex flex-wrap gap-1.5", className)}>
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button
            key={o.value}
            role="tab"
            aria-selected={on}
            onClick={() => onChange(o.value)}
            className={cx(
              "h-9 px-4 rounded-full inline-flex items-center gap-1.5 text-sm transition-all duration-200 border",
              on ? "pill-on border-transparent font-medium" : "text-fg-2 border-line hover:text-fg hover:border-fg-3/50"
            )}
          >
            {o.label}
            {o.count != null && <span className={cx("num text-xs", on ? "opacity-60" : "text-fg-3")}>{o.count}</span>}
          </button>
        );
      })}
    </div>
  );
}
