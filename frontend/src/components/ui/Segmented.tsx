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
      className={cx("inline-flex p-0.5 rounded bg-surface-2 border border-line-soft", className)}
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
              "inline-flex items-center gap-1.5 rounded-sm font-medium transition-colors duration-150 whitespace-nowrap",
              size === "sm" ? "h-6 px-2 text-xs" : "h-7 px-3 text-sm",
              on ? "bg-surface text-fg shadow-[0_1px_2px_rgb(0_0_0/0.15)]" : "text-fg-2 hover:text-fg"
            )}
          >
            {o.label}
            {o.count != null && <span className="num text-fg-3">{o.count}</span>}
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
    <div role="tablist" aria-label={label} className={cx("flex gap-4 border-b border-line-soft", className)}>
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button
            key={o.value}
            role="tab"
            aria-selected={on}
            onClick={() => onChange(o.value)}
            className={cx(
              "relative -mb-px h-9 inline-flex items-center gap-1.5 text-sm transition-colors border-b-2",
              on ? "text-fg border-accent font-medium" : "text-fg-2 border-transparent hover:text-fg"
            )}
          >
            {o.label}
            {o.count != null && <span className="num text-xs text-fg-3">{o.count}</span>}
          </button>
        );
      })}
    </div>
  );
}
