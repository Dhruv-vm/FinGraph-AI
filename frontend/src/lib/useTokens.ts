"use client";

import { useEffect, useState } from "react";
import { useTheme } from "./theme";

const NAMES = [
  "fg", "fg-2", "fg-3", "line", "line-soft", "surface", "surface-2", "accent", "pos", "neg", "warn", "info", "graph-dot",
  "rel-acquired", "rel-competes", "rel-supplies", "rel-partners", "rel-invested", "rel-risk", "rel-other",
] as const;
export type TokenName = (typeof NAMES)[number];

/** Resolved token colours (for SVG/canvas libraries that can't read CSS variables). */
export function useTokens(): Record<TokenName, string> {
  const { theme } = useTheme();
  const [colors, setColors] = useState<Record<TokenName, string>>(
    () => Object.fromEntries(NAMES.map((n) => [n, "currentColor"])) as Record<TokenName, string>
  );
  useEffect(() => {
    const cs = getComputedStyle(document.documentElement);
    setColors(
      Object.fromEntries(NAMES.map((n) => [n, `rgb(${cs.getPropertyValue(`--${n}`).trim().split(/\s+/).join(",")})`])) as Record<
        TokenName,
        string
      >
    );
  }, [theme]);
  return colors;
}
