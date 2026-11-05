import React from "react";
import { cx } from "./cx";
import { Spinner } from "./Feedback";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md";

const VARIANTS: Record<Variant, string> = {
  primary:
    "bg-accent-strong text-accent-fg shadow-glow hover:brightness-110 hover:-translate-y-px disabled:bg-surface-3 disabled:text-fg-3 disabled:shadow-none disabled:translate-y-0",
  secondary:
    "bg-surface-2 border border-line text-fg hover:bg-surface-3 hover:border-fg-3/40 disabled:text-fg-3",
  ghost: "text-fg-2 hover:text-fg hover:bg-surface-2 disabled:text-fg-3",
  danger: "text-neg hover:bg-neg/10",
};

const SIZES: Record<Size, string> = {
  sm: "h-8 px-3.5 text-xs gap-1.5",
  md: "h-10 px-5 text-sm gap-2",
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
  icon?: React.ReactNode;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", size = "md", loading, icon, className, children, disabled, ...rest },
  ref
) {
  return (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={cx(
        "inline-flex items-center justify-center rounded-full font-medium whitespace-nowrap select-none",
        "transition-[color,background-color,border-color,transform,filter] duration-200 disabled:cursor-not-allowed",
        VARIANTS[variant],
        SIZES[size],
        className
      )}
      {...rest}
    >
      {loading ? <Spinner className="w-3.5 h-3.5" /> : icon}
      {children}
    </button>
  );
});

export interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  label: string;
  size?: "sm" | "md";
  active?: boolean;
}

/** Square icon-only button. `label` is required for accessibility and the tooltip. */
export const IconButton = React.forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  { label, size = "md", active, className, children, ...rest },
  ref
) {
  return (
    <button
      ref={ref}
      aria-label={label}
      title={label}
      className={cx(
        "inline-flex items-center justify-center rounded-full transition-colors duration-150 shrink-0",
        size === "sm" ? "w-8 h-8" : "w-9 h-9",
        active ? "pill-on" : "text-fg-2 hover:text-fg hover:bg-surface-2",
        "disabled:opacity-40 disabled:cursor-not-allowed",
        className
      )}
      {...rest}
    >
      {children}
    </button>
  );
});
