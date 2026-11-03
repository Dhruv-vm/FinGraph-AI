# FinGraph AI — UI/UX Design System

**Version:** 2.0 (2026-11-03): presentation-first revision (midnight navy, electric blue, restrained cyan, card surfaces)
**Source of truth:** `frontend/src/app/globals.css` (tokens) · `frontend/tailwind.config.ts` (mapping) · `frontend/src/components/ui/` (primitives)

---

## 1. Point of view

FinGraph is presented as an institutional research product: research density from financial terminals, the clarity of a modern SaaS dashboard, an interactive graph, and evidence-grounded answers.

| Register | What it contributes | Where it shows |
|---|---|---|
| **Dashboard** | Distinct card surfaces, compact KPI cards, clear primary actions, consistent 16–24 px rhythm | Overview, Evaluation, Ask examples |
| **Institutional** | Tabular figures, ISO dates, explicit sample sizes, ledgers inside cards for detail | Evaluation tables, inspectors, system details |
| **Editorial** | A serif reading voice only where people *read* evidence: answers, filing passages, stage descriptions | Ask answer, Evidence reader, Architecture |
| **Knowledge** | The graph canvas is the one place with a full relationship palette | Knowledge graph |

**Rules**

1. *Presentation first.* The first screen of each page answers a faculty question (what it does, what data, how it works, what happens on a question, how it was evaluated). Diagnostics live in collapsible sections.
2. *Blue leads, cyan supports.* Electric blue for primary actions, selection and focus; cyan only marks "built offline" stages and partnership edges. No neon glows; one soft accent wash on the hero.
3. *Every figure has a provenance.* Live values come from the API; historical results carry `Historical` + phase + run date + *n*. Nothing falls back to a hard-coded number.
4. *Separate measurements stay separate.* Decision, answer correctness, retrieval, temporal leakage and parametric leakage are never merged.
5. *Sentence case; mono only for identifiers and numeric metadata.*

## 2. Colour tokens

RGB channel triplets (`globals.css`) mapped to Tailwind (`tailwind.config.ts`), so opacity modifiers work (`bg-accent/10`). Dark is default (`html.dark`), light is `html.light`.

### 2.1 Neutrals

| Token | Tailwind | Dark | Light | Use |
|---|---|---|---|---|
| `--bg` | `bg-canvas` | `#0A0E16` midnight | `#F4F6F9` | App background |
| `--sidebar` | `bg-sidebar` | `#0C111A` | `#FAFBFC` | Navigation, library column |
| `--surface` | `bg-surface`, `.card` | `#111722` | `#FFFFFF` | Cards, inputs, inspector |
| `--surface-2` | `bg-surface-2` | `#161E2C` | `#F7F9FB` | Nested tiles, table heads |
| `--surface-3` | `bg-surface-3` | `#1E2839` | `#ECF0F5` | Hover, pressed, skeleton |
| `--fg` | `text-fg` | `#E8ECF2` | `#101828` | Primary text |
| `--fg-2` | `text-fg-2` | `#A0AABA` | `#475467` | Secondary text |
| `--fg-3` | `text-fg-3` | `#747F91` | `#667085` | Captions |
| `--line` | `border-line` | `#263042` | `#DFE4EC` | Card and control borders |
| `--line-soft` | `border-line-soft` | `#1C2433` | `#EAEDF2` | Dividers inside cards |
| `--shadow-card` | `.card` | inset highlight + soft drop | 1–3 px drop | Layered surfaces |

### 2.2 Brand and semantic

| Token | Usage | Dark | Light |
|---|---|---|---|
| `--accent` (`text-accent`) | Links, selection, focus, primary metrics | `#6094FF` | `#155EEF` |
| `--accent-strong` (`bg-accent-strong`) | Primary buttons, active stage | `#155EEF` | `#155EEF` |
| `--cyan` (`.text-cyan`, `.bg-cyan-soft`) | Secondary emphasis: offline stages | `#5EC8DE` | `#0884A0` |
| `--pos` | Supported, operational, refusals that were correct | `#5EBA8C` | `#168050` |
| `--neg` | Errors, incorrect, leakage, generation failure | `#E87A6E` | `#BE3E32` |
| `--warn` | Insufficient evidence, unverified provenance, limitations | `#E2B25C` | `#A0680A` |
| `--info` | Partially supported, graph-sourced evidence | `#78ACE8` | `#2563B4` |

> `cyan` is defined as CSS utilities rather than a Tailwind colour so the running `next dev` process (which caches `tailwind.config.ts`) did not need a restart.

### 2.3 Relationship palette (graph only)

| Family | Relationships grouped | Token |
|---|---|---|
| Acquisition | `ACQUIRED`, `MERGED…` | `--rel-acquired` |
| Competition | `COMPETES_WITH` | `--rel-competes` |
| Supply chain | `SUPPLIES`, `MANUFACTURES`, `DEPENDS_ON` | `--rel-supplies` |
| Partnership | `PARTNERS_WITH` | `--rel-partners` |
| Investment | `INVESTED_IN`, `OWNS…` | `--rel-invested` |
| Risk and regulation | `HAS_RISK`, `AFFECTED_BY`, `REGULAT…` | `--rel-risk` |
| Other | everything else | `--rel-other` |

Mapping: `lib/format.ts → relFamily()`. Colour is never the only signal (legend and inspector name every relationship).

### 2.4 Status

`StatusDot`: circle = operational, diamond = degraded, square = unavailable. The header names the failing service when only one is down ("Language model offline").

## 3. Typography

| Role | Family | Notes |
|---|---|---|
| UI sans | **Geist** (bundled `app/fonts/GeistVF.woff`, `--font-sans`) | All interface text |
| Mono | **Geist Mono** (`--font-mono`) | Ids, tickers, timestamps, numeric metadata only |
| Reading serif | Charter → Iowan Old Style → Source Serif → Georgia | Answers, passages, page leads. System fonts: no download |

Fonts load through `next/font/local` — no network dependency.

**Scale** (rem / line-height) — `2xs 0.6875/1rem`, `xs 0.75/1.125`, `sm 0.8125/1.25`, `base 0.875/1.375`, `md 0.9375/1.5`, `lg 1.0625/1.625`, `xl 1.25/1.75`, `2xl 1.5/2`, `3xl 1.875/2.375`. Body is `base` (14 px) for density; reading text is 17 px serif.

**Weights** — 400 body, 500 emphasis/labels, 600 headings. No 700+.

**Figures** — `.num` applies tabular numerals; every numeric column and metric uses it.

**Dates** — always ISO `YYYY-MM-DD`; timestamps `YYYY-MM-DD HH:MM UTC` (`fmtDate`, `fmtDateTime`). Filing date, reporting period, event time and availability time are always labelled separately.

## 4. Shape, space, elevation, motion

- **Radii by hierarchy:** 4 px badges, 6 px controls, 8 px nested tiles, 12 px cards (`.card` → `rounded-xl`), full round for status pills and graph nodes.
- **Spacing:** 4 px base. Page gutters 16 px (mobile) / 32 px (desktop). Content max widths per page (1180–1240 px) rather than full-bleed, except the graph.
- **Elevation:** cards use `--shadow-card` (subtle, layered); `shadow-overlay` is reserved for floating elements (popovers, drawers, dialogs, canvas controls).
- **Motion:** 150 ms colour transitions; 160–220 ms fade/rise/slide for overlays and newly revealed results; a 2 px indeterminate progress bar for in-flight requests. No ambient loops. `prefers-reduced-motion` collapses all durations.

## 5. Components (`components/ui`)

| Component | File | Purpose |
|---|---|---|
| `Button` | `Button.tsx` | `primary` / `secondary` / `ghost` / `danger`, `sm`/`md`, `loading`, `icon` |
| `IconButton` | `Button.tsx` | Square icon action; `label` is required (aria-label + tooltip) |
| `Input`, `Textarea`, `Select`, `FieldLabel` | `Field.tsx` | Token-styled controls with focus ring; `Input` supports a leading icon |
| `Segmented`, `Tabs` | `Segmented.tsx` | Radio-group segmented control; underlined tabs with counts |
| `Panel`, `PanelHeader`, `SectionHeading` | `Panel.tsx` | Card surface (`.card`) and heading patterns |
| `.card`, `.card-interactive` | `globals.css` | Card surface with layered shadow; hover border for clickable cards |
| `KeyValue` | `Panel.tsx` | Ledger rows (label / value / hint); `mono` for identifiers |
| `Metric` | `Panel.tsx` | Typographic figure with label and detail |
| `ChartFrame` | `Panel.tsx` | Chart container with title, caption and source line |
| `Badge` | `Feedback.tsx` | Small tonal label (`neutral/accent/pos/neg/warn/info`) |
| `StatusDot`, `STATUS_LABEL` | `Feedback.tsx` | Shape-coded status |
| `Spinner`, `ProgressBar`, `Skeleton` | `Feedback.tsx` | Loading states |
| `EmptyState`, `ErrorState`, `Note` | `Feedback.tsx` | Empty, error (offline-aware), and methodology notes |
| `Drawer`, `Dialog`, `Popover`, `Tooltip` | `Overlay.tsx` | Overlays with Escape-to-close, focus on open, focus restore |
| `ViewErrorBoundary` | `components/ErrorBoundary.tsx` | Isolates a crashing page |

Shell components: `Navigation` (+ `BrandLogo`, `NavList`), `Header` (status popover, research-context control, theme toggle).

## 6. Patterns

- **KPI cards for headlines, ledgers for detail.** At most four KPI cards per row, each with a label, value and one-line provenance. Detailed figures live in `KeyValue` ledgers inside cards or collapsible sections.
- **Historical labelling.** Any benchmark figure carries `Historical`, phase and run-date badges, plus *n* (e.g. "59 of 64 questions").
- **Outcome states (Ask).** Status pill in the answer card header: Supported answer (pos), Partially supported (info), Insufficient evidence (warn), Validation issue (neg), Generation failed (neg: evidence sufficient but the LLM could not run), and API errors as an error state. Icon + label + colour.
- **Counts are explicit.** Graph shows: KG nodes/edges · direct edges · connected companies · in-scope edges · loaded · shown after filters.
- **Missing data says so.** "Not recorded" in muted text; never a default value.
- **Context is visible.** The research-context chip in the header shows the entity on every page; views that use it say so ("Filtered by research context · Show all").

## 7. Accessibility

- Visible `:focus-visible` ring (2 px accent, 2 px offset) on every interactive element; skip-to-content link.
- Icon-only buttons have `aria-label`; segmented controls use `radiogroup`/`radio`; tabs use `tablist`/`tab`; nav uses `aria-current`.
- Drawers and dialogs: `role="dialog"`, `aria-modal`, labelled title, Escape closes, focus moves in and is restored.
- Results and graph counts are in `aria-live="polite"` regions.
- Status and outcomes combine shape/icon, colour and text.
- Reduced-motion respected globally.

## 8. Theming implementation

- `THEME_BOOTSTRAP_SCRIPT` (in `lib/theme.tsx`) runs in `<head>` before paint and sets `html.dark` / `html.light` from `localStorage["fingraph-theme"]` (default dark), so there is no flash.
- Charts read resolved token colours through `useTokens()` (Recharts cannot read CSS variables), recomputed on theme change.
- React Flow is themed via token-based CSS in `globals.css`; the canvas dot grid uses `--graph-dot`.
