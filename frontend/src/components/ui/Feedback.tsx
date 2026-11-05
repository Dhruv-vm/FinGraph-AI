import React from "react";
import { AlertTriangle, Inbox, WifiOff } from "lucide-react";
import { cx } from "./cx";

export function Spinner({ className }: { className?: string }) {
  return (
    <svg className={cx("animate-spin", className ?? "w-4 h-4")} viewBox="0 0 16 16" aria-hidden>
      <circle cx="8" cy="8" r="6.25" fill="none" stroke="currentColor" strokeOpacity="0.2" strokeWidth="1.5" />
      <path d="M14.25 8A6.25 6.25 0 0 0 8 1.75" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

/** Thin indeterminate bar for in-flight work at the top of a region. */
export function ProgressBar({ active, className }: { active: boolean; className?: string }) {
  return (
    <div
      aria-hidden
      className={cx("h-0.5 w-full overflow-hidden bg-transparent transition-opacity", active ? "opacity-100" : "opacity-0", className)}
    >
      <div className="h-full w-2/5 bg-accent animate-progress" />
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={cx("rounded-lg bg-surface-3/70 animate-pulse", className)} />;
}

export type Status = "healthy" | "degraded" | "unavailable" | "unknown";

export const STATUS_LABEL: Record<Status, string> = {
  healthy: "Operational",
  degraded: "Degraded",
  unavailable: "Unavailable",
  unknown: "Checking",
};

/** Status uses shape + colour + text, never colour alone. */
export function StatusDot({ status, className }: { status: Status; className?: string }) {
  const color =
    status === "healthy" ? "bg-pos" : status === "degraded" ? "bg-warn" : status === "unavailable" ? "bg-neg" : "bg-fg-3";
  return (
    <span
      aria-hidden
      className={cx(
        "inline-block w-2 h-2 shrink-0",
        status === "degraded" ? "rotate-45 rounded-[1px]" : status === "unavailable" ? "rounded-[1px]" : "rounded-full",
        color,
        className
      )}
    />
  );
}

type BadgeTone = "neutral" | "accent" | "pos" | "neg" | "warn" | "info";
const BADGE: Record<BadgeTone, string> = {
  neutral: "bg-surface-2 text-fg-2 border-line",
  accent: "bg-accent/10 text-accent border-accent/25",
  pos: "bg-pos/10 text-pos border-pos/25",
  neg: "bg-neg/10 text-neg border-neg/25",
  warn: "bg-warn/10 text-warn border-warn/25",
  info: "bg-info/10 text-info border-info/25",
};

export function Badge({
  tone = "neutral",
  children,
  className,
  mono,
  title,
}: {
  tone?: BadgeTone;
  children: React.ReactNode;
  className?: string;
  mono?: boolean;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={cx(
        "inline-flex items-center gap-1 h-6 px-2.5 rounded-full border text-2xs font-medium whitespace-nowrap",
        mono && "font-mono-num",
        BADGE[tone],
        className
      )}
    >
      {children}
    </span>
  );
}

export function EmptyState({
  title,
  children,
  action,
  icon,
  className,
}: {
  title: string;
  children?: React.ReactNode;
  action?: React.ReactNode;
  icon?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cx("flex flex-col items-center justify-center text-center px-6 py-10", className)}>
      <div className="text-fg-3 mb-3">{icon ?? <Inbox className="w-5 h-5" />}</div>
      <div className="text-sm font-medium text-fg">{title}</div>
      {children && <div className="text-sm text-fg-2 mt-1 max-w-[46ch]">{children}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function ErrorState({
  title,
  error,
  action,
  className,
  compact,
}: {
  title: string;
  error?: { message: string; status?: number } | null;
  action?: React.ReactNode;
  className?: string;
  compact?: boolean;
}) {
  const offline = error?.status === 0;
  return (
    <div
      role="alert"
      className={cx(
        "flex gap-3 rounded-2xl border border-neg/30 bg-neg/[0.06] text-sm",
        compact ? "px-3 py-2.5" : "px-4 py-3.5",
        className
      )}
    >
      {offline ? <WifiOff className="w-4 h-4 text-neg shrink-0 mt-0.5" /> : <AlertTriangle className="w-4 h-4 text-neg shrink-0 mt-0.5" />}
      <div className="min-w-0 flex-1">
        <div className="font-medium text-fg">{title}</div>
        {error?.message && <div className="text-fg-2 mt-0.5 break-words">{error.message}</div>}
        {offline && (
          <div className="text-fg-3 text-xs mt-1">
            Check that the FastAPI server is running on port 8000.
          </div>
        )}
        {action && <div className="mt-2.5">{action}</div>}
      </div>
    </div>
  );
}

/** Inline note used for methodology caveats and data provenance. */
export function Note({
  children,
  tone = "neutral",
  className,
  icon,
}: {
  children: React.ReactNode;
  tone?: "neutral" | "warn" | "info";
  className?: string;
  icon?: React.ReactNode;
}) {
  return (
    <div
      className={cx(
        "flex gap-2.5 text-xs leading-relaxed rounded-xl px-3.5 py-3 border",
        tone === "warn"
          ? "border-warn/30 bg-warn/[0.06] text-fg-2"
          : tone === "info"
          ? "border-info/25 bg-info/[0.06] text-fg-2"
          : "border-line-soft bg-surface-2 text-fg-2",
        className
      )}
    >
      {icon && <span className="shrink-0 mt-px text-fg-3">{icon}</span>}
      <div className="min-w-0">{children}</div>
    </div>
  );
}
