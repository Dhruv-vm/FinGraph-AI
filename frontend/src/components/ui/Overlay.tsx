"use client";

import React, { useEffect, useId, useRef } from "react";
import { X } from "lucide-react";
import { cx } from "./cx";
import { IconButton } from "./Button";

function useEscape(open: boolean, onClose: () => void) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);
}

function useFocusOnOpen(open: boolean, ref: React.RefObject<HTMLElement>) {
  useEffect(() => {
    if (!open) return;
    const prev = document.activeElement as HTMLElement | null;
    const t = window.setTimeout(() => ref.current?.focus(), 0);
    return () => {
      window.clearTimeout(t);
      prev?.focus?.();
    };
  }, [open, ref]);
}

/** Side drawer (modal on small screens). */
export function Drawer({
  open,
  onClose,
  title,
  side = "right",
  children,
  className,
  width = "w-[min(420px,92vw)]",
}: {
  open: boolean;
  onClose: () => void;
  title: React.ReactNode;
  side?: "left" | "right";
  children: React.ReactNode;
  className?: string;
  width?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const titleId = useId();
  useEscape(open, onClose);
  useFocusOnOpen(open, ref);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50">
      <div className="absolute inset-0 bg-black/40 animate-fade-in" onClick={onClose} aria-hidden />
      <div
        ref={ref}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        className={cx(
          "absolute top-0 bottom-0 bg-surface border-line shadow-overlay flex flex-col focus:outline-none",
          side === "right" ? "right-0 border-l animate-slide-in-right" : "left-0 border-r animate-slide-in-left",
          width,
          className
        )}
      >
        <div className="flex items-center justify-between gap-2 px-4 h-12 border-b border-line-soft shrink-0">
          <div id={titleId} className="text-sm font-semibold truncate">
            {title}
          </div>
          <IconButton label="Close" size="sm" onClick={onClose}>
            <X className="w-4 h-4" />
          </IconButton>
        </div>
        <div className="flex-1 overflow-y-auto scrollbar-thin">{children}</div>
      </div>
    </div>
  );
}

/** Centered dialog. */
export function Dialog({
  open,
  onClose,
  title,
  children,
  footer,
}: {
  open: boolean;
  onClose: () => void;
  title: React.ReactNode;
  children: React.ReactNode;
  footer?: React.ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const titleId = useId();
  useEscape(open, onClose);
  useFocusOnOpen(open, ref);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/45 animate-fade-in" onClick={onClose} aria-hidden />
      <div
        ref={ref}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        className="relative w-full max-w-lg bg-surface border border-line rounded-2xl shadow-overlay animate-rise-in focus:outline-none"
      >
        <div className="flex items-center justify-between px-4 h-12 border-b border-line-soft">
          <div id={titleId} className="text-sm font-semibold">
            {title}
          </div>
          <IconButton label="Close" size="sm" onClick={onClose}>
            <X className="w-4 h-4" />
          </IconButton>
        </div>
        <div className="p-4 max-h-[70vh] overflow-y-auto scrollbar-thin">{children}</div>
        {footer && <div className="px-4 py-3 border-t border-line-soft flex justify-end gap-2">{footer}</div>}
      </div>
    </div>
  );
}

/** Lightweight hover/focus tooltip (CSS only). */
export function Tooltip({ content, children, side = "top" }: { content: React.ReactNode; children: React.ReactElement; side?: "top" | "bottom" }) {
  return (
    <span className="relative inline-flex group/tt">
      {children}
      <span
        role="tooltip"
        className={cx(
          "pointer-events-none absolute left-1/2 -translate-x-1/2 z-40 w-max max-w-[260px] rounded px-2 py-1.5",
          "bg-fg text-canvas text-2xs leading-snug opacity-0 transition-opacity duration-150",
          "group-hover/tt:opacity-100 group-focus-within/tt:opacity-100",
          side === "top" ? "bottom-full mb-1.5" : "top-full mt-1.5"
        )}
      >
        {content}
      </span>
    </span>
  );
}

/** Small anchored popover controlled by the parent. */
export function Popover({
  open,
  onClose,
  anchorClassName,
  children,
  className,
}: {
  open: boolean;
  onClose: () => void;
  anchorClassName?: string;
  children: React.ReactNode;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEscape(open, onClose);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    const t = window.setTimeout(() => document.addEventListener("mousedown", onDown), 0);
    return () => {
      window.clearTimeout(t);
      document.removeEventListener("mousedown", onDown);
    };
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div
      ref={ref}
      className={cx(
        "absolute z-40 bg-surface border border-line rounded-2xl shadow-overlay animate-rise-in",
        anchorClassName ?? "right-0 top-full mt-2",
        className
      )}
    >
      {children}
    </div>
  );
}
