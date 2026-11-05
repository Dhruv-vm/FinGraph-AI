import React from "react";
import { cx } from "./cx";

const CONTROL =
  "w-full bg-surface-2 border border-line-soft rounded-xl text-fg placeholder:text-fg-3 " +
  "transition-colors duration-150 hover:border-fg-3/50 focus:outline-none focus:border-accent focus:ring-2 focus:ring-accent/20 " +
  "disabled:opacity-60";

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement> & { leading?: React.ReactNode; inputSize?: "sm" | "md" }
>(function Input({ className, leading, inputSize = "md", ...rest }, ref) {
  return (
    <div className={cx("relative flex items-center", className)}>
      {leading && <span className="absolute left-3.5 text-fg-3 pointer-events-none flex">{leading}</span>}
      <input
        ref={ref}
        className={cx(CONTROL, inputSize === "sm" ? "h-9 text-sm rounded-full" : "h-10 text-sm rounded-full", leading ? "pl-9 pr-3.5" : "px-4")}
        {...rest}
      />
    </div>
  );
});

export const Textarea = React.forwardRef<HTMLTextAreaElement, React.TextareaHTMLAttributes<HTMLTextAreaElement>>(
  function Textarea({ className, ...rest }, ref) {
    return <textarea ref={ref} className={cx(CONTROL, "px-4 py-3 text-md leading-relaxed resize-none rounded-2xl", className)} {...rest} />;
  }
);

export function Select({
  className,
  children,
  ...rest
}: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <div className={cx("relative", className)}>
      <select
        className={cx(CONTROL, "h-9 text-sm pl-4 pr-8 appearance-none cursor-pointer rounded-full")}
        {...rest}
      >
        {children}
      </select>
      <svg
        aria-hidden
        viewBox="0 0 12 12"
        className="w-3 h-3 absolute right-3.5 top-1/2 -translate-y-1/2 text-fg-3 pointer-events-none"
      >
        <path d="M3 4.5 6 7.5 9 4.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    </div>
  );
}

export function FieldLabel({ htmlFor, children, hint }: { htmlFor?: string; children: React.ReactNode; hint?: React.ReactNode }) {
  return (
    <label htmlFor={htmlFor} className="flex items-baseline justify-between gap-2 text-xs font-medium text-fg-2 mb-1.5">
      <span>{children}</span>
      {hint && <span className="font-normal text-fg-3">{hint}</span>}
    </label>
  );
}
