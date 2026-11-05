import type { Config } from "tailwindcss";

/**
 * FinGraph design tokens.
 * Every colour is a CSS variable holding RGB channels (see src/app/globals.css),
 * so utilities support opacity modifiers (e.g. bg-accent/10) and both themes
 * share one set of class names.
 */
const token = (name: string) => `rgb(var(--${name}) / <alpha-value>)`;

const config: Config = {
  darkMode: "class",
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: token("bg"),
        sidebar: token("sidebar"),
        frame: token("frame"),
        pill: { DEFAULT: token("pill"), fg: token("pill-fg") },
        surface: {
          DEFAULT: token("surface"),
          2: token("surface-2"),
          3: token("surface-3"),
        },
        fg: {
          DEFAULT: token("fg"),
          2: token("fg-2"),
          3: token("fg-3"),
        },
        line: {
          DEFAULT: token("line"),
          soft: token("line-soft"),
        },
        accent: {
          DEFAULT: token("accent"),
          strong: token("accent-strong"),
          fg: token("accent-fg"),
        },
        pos: token("pos"),
        neg: token("neg"),
        warn: token("warn"),
        info: token("info"),
        rel: {
          acquired: token("rel-acquired"),
          competes: token("rel-competes"),
          supplies: token("rel-supplies"),
          partners: token("rel-partners"),
          invested: token("rel-invested"),
          risk: token("rel-risk"),
          other: token("rel-other"),
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
        serif: ["Charter", "'Iowan Old Style'", "'Source Serif Pro'", "Georgia", "Cambria", "serif"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
        xs: ["0.75rem", { lineHeight: "1.125rem" }],
        sm: ["0.8125rem", { lineHeight: "1.25rem" }],
        base: ["0.875rem", { lineHeight: "1.375rem" }],
        md: ["0.9375rem", { lineHeight: "1.5rem" }],
        lg: ["1.0625rem", { lineHeight: "1.625rem" }],
        xl: ["1.25rem", { lineHeight: "1.75rem" }],
        "2xl": ["1.5rem", { lineHeight: "2rem" }],
        "3xl": ["1.875rem", { lineHeight: "2.375rem" }],
      },
      borderRadius: {
        sm: "6px",
        DEFAULT: "10px",
        md: "12px",
        lg: "16px",
        xl: "22px",
        "2xl": "26px",
        "3xl": "32px",
      },
      boxShadow: {
        overlay: "0 24px 60px -16px rgb(0 0 0 / 0.6), 0 2px 8px rgb(0 0 0 / 0.25)",
        glow: "0 10px 30px -10px rgb(var(--accent-strong) / 0.75)",
      },
      transitionTimingFunction: {
        out: "cubic-bezier(0.2, 0.8, 0.2, 1)",
      },
      keyframes: {
        "fade-in": { from: { opacity: "0" }, to: { opacity: "1" } },
        "rise-in": { from: { opacity: "0", transform: "translateY(4px)" }, to: { opacity: "1", transform: "none" } },
        "slide-in-right": { from: { transform: "translateX(100%)" }, to: { transform: "none" } },
        "slide-in-left": { from: { transform: "translateX(-100%)" }, to: { transform: "none" } },
        progress: { "0%": { transform: "translateX(-100%)" }, "100%": { transform: "translateX(250%)" } },
      },
      animation: {
        "fade-in": "fade-in 160ms ease-out",
        "rise-in": "rise-in 200ms cubic-bezier(0.2,0.8,0.2,1)",
        "slide-in-right": "slide-in-right 220ms cubic-bezier(0.2,0.8,0.2,1)",
        "slide-in-left": "slide-in-left 220ms cubic-bezier(0.2,0.8,0.2,1)",
        progress: "progress 1.4s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
export default config;
