# FinGraph AI — Product Audit and Implementation Plan

**Date:** 2026-11-03
**Scope:** `frontend/` (Next.js app) and its integration surface in `api/` (FastAPI)
**Method:** Source inspection of every frontend file and every API route/schema, plus live probing of the running backend (`127.0.0.1:8000`) and frontend dev server (`localhost:3000`) on the author's machine. Every claim below about behaviour was observed in code or against the live API; nothing is inferred from documentation alone.

---

## 1. Repository and git state at start

- Branch `main`, last commit `1f0c0b4`.
- **`frontend/` and `api/` are entirely untracked in git.** Git therefore cannot restore them. A full pre-change backup was taken before any edit:
  `archive/frontend_pre_redesign_backup_2026-11-03.tar.gz` (frontend + api, excluding `node_modules` and `.next`).
- 15 tracked research files carry uncommitted modifications (`src/qa/*`, `src/retrieval/*`, `scripts/run_fingraph.py`, …). **None of these are touched by this work.**
- No staging, committing, resetting or cleaning is performed.

## 2. Existing architecture

### 2.1 Frontend
| Item | Finding |
|---|---|
| Framework | Next.js 14.2.35 (App Router), React 18, TypeScript (strict), Tailwind 3.4 |
| Libraries | `reactflow` 11, `recharts` 3, `lucide-react` |
| Routing | **Single route** (`src/app/page.tsx`). The six "pages" are tabs held in React state; no URLs, no back-button, no deep links. |
| Theme | `lib/theme.tsx` — class-based dark mode persisted to `localStorage`. Dark default. Wraps children in `<div className={mounted ? "" : "dark"}>`, causing a light/dark flash for light-theme users and a full-tree class toggle on mount. |
| Design tokens | Two parallel, unused-in-practice systems: CSS variables in `globals.css` and `terminal.*` colours in `tailwind.config.ts`. Components hard-code hex colours (`#0E131E`, `#1E293B`, …) and Tailwind slate/blue directly. |
| Fonts | Geist woff files exist in `app/fonts/` but are **not loaded**; body uses the system stack. |
| Shared components | `Badge`, `StatCard` only. |
| API client | `lib/api.ts` — typed `fetchJson` that tries the Next rewrite (`/api/...`) then the absolute base URL. |
| Types | `lib/types.ts` — mostly synchronised with Pydantic schemas. |
| State management | Local component state only. No shared context. |

### 2.2 Backend endpoints used by the UI
| Endpoint | Source of data | Notes |
|---|---|---|
| `GET /api/v1/health` | live pipeline, Qdrant, KG, Ollama | Real. Verified healthy: 8,181 points, 10,054 nodes, 9,822 edges, `qwen3.5:4b` present. |
| `POST /api/v1/query` | live `FinGraphPipeline.answer` | Real. Serialises a lock-guarded pipeline call. |
| `GET /api/v1/benchmark/demo-questions` | hard-coded list of 5 audited questions in `routes/benchmark.py` | Static by design (cached Phase 7 results). |
| `GET /api/v1/benchmark/summary` | `results/phase7/phase7_summary.json` | Real file read. |
| `GET /api/v1/graph/companies` | KG | Real. Returns 1,093 company-typed nodes, **including extraction artefacts** ("the Company", "Firm"). |
| `GET /api/v1/graph/subgraph` | KG | Real. Paginated direct edges with `total_direct_edges`, `total_connected_companies`, `total_matching`. |
| `GET /api/v1/graph/search` | KG | Real substring search. |
| `GET /api/v1/evidence/chunks` | Qdrant scroll | Real. Only a `company` filter; no document, filing-type or date filter; no total count. |

## 3. Functionality inventory (page by page)

### 3.1 System Overview — *partially real, heavily hard-coded*
- Works: reads `health` for chunk/node/edge counts.
- **Fabricated or misleading:**
  - Decorative SVG "sparkline" with a "59/64 Audited (92.2%)" pin — the curve represents no data.
  - Range pills `10-K / 10-Q / RRF / K8 / Phase 7` change only their own highlight state.
  - Every `⋮` "more" button does nothing. Header bell shows a permanent unread dot with no notifications behind it.
  - Company table is a hard-coded list with invented sector labels and invented filing ranges ("FY2023-2025"); actual corpus filings span 2025-04-30 → 2026-09-03.
  - "S&P 500 Corporate Entity Universe · 1,093 Companies Indexed" — the 1,093 count is all company-typed KG nodes (including artefacts), not S&P 500 constituents. Supported issuers in the corpus = 10.
  - "100% Multi-hop Graph Traversal Ready", "+100% Provenance", "0 Hallucinations", "Zero Future Leakage" presented as live system facts. The historical Phase 7 run recorded 0 hallucinations and FEIR 0 **on the 64-question benchmark only**; FEIR = 0 does not establish zero parametric leakage.
  - Fallback numbers (`|| 8181`, `|| 10054`) silently display historical values when the backend is down.
  - Star toggles are cosmetic and not persisted.

### 3.2 Ask FinGraph — *real pipeline, good bones*
- Works: real `POST /query`, reference-time input, five audited demo questions, cached Phase 7 baseline comparison, sufficiency and temporal panels, citations, evidence list with source links.
- Issues:
  - Reference time is free text (`YYYY-MM-DD or ISO 8601`), no validation; malformed input reaches the backend.
  - **`future_items_count` and `feir` are constants (`0`, `0.0`) in `api/services/pipeline_service.py`.** The UI displays them as measurements ("FEIR Rate: 0.00%", "Future Leakage Count: 0"). The true guarantee comes from the hard assertion in `src/qa/generator.py` (raises if any evidence post-dates the reference time), which is not what the UI says.
  - FEIR is rendered as `feir.toFixed(2) + "%"` although the field is a ratio.
  - No request cancellation; switching tabs mid-query discards the result (component unmounts).
  - No distinction between partially supported answers, validation failure and API error beyond colour.

### 3.3 Knowledge Graph Explorer — *real data, unreadable presentation, silent fabrication path*
- Works: live subgraph, search, company/all filter, limit selector, node and edge inspection.
- Issues:
  - **Silent fallback to a 3.6 MB bundled snapshot (`lib/graphFallbackData.ts`)** whenever any graph request fails; unknown entities fall back to **Microsoft's** subgraph. A 404 for an unknown entity is therefore shown as real MSFT data. This file is also bundled into the client JS.
  - Default view renders 60 boxed nodes in three concentric rings with edge labels: overlapping and illegible (see `docs/screenshots/before/graph-dark.png`).
  - Count labels conflate three different quantities: total direct edges (2,338 for NVDA, all relationship types), company-to-company edges (250) and unique connected corporate entities (118).
  - Edge inspector never shows `attribution_status` / `provenance_verified`, although many edges are `unsupported_cross_company` (e.g. an NVDA edge sourced from a Tesla filing).
  - Backend defaults missing edge `confidence` to `0.9` (`routes/graph.py`) — a fabricated value.
  - No progressive expansion from a neighbour; pivoting replaces the graph.

### 3.4 Evidence Explorer — *real, limited*
- Works: Qdrant scroll, company filter pills, local text filter on loaded chunks, chunk reader.
- Issues: no filing-level (document) view, no filing-type or date filter, no sort, no total count, no "load more" control, `section` is always null in payloads but shown as if expected, "Available cutoff" formats dates inconsistently (`26/02/2026` vs ISO elsewhere). Header claims "Form 10-K and 10-Q" — corpus also contains 50 8-Ks.

### 3.5 Evaluation Lab — *real file, good integrity notes*
- Works: loads `phase7_summary.json`, charts decision accuracy and answerable F1 across 5 configurations, explicit note separating decision accuracy from factual accuracy.
- Issues: falls back to a hard-coded copy on failure without saying so; ignores the richer `configurations` block (refusal precision, hit@k, correct refusals, overall F1, descriptions); no category breakdown although `results/phase7/per_question_results.csv` exists; y-axis truncated at 75 % exaggerates differences without a label.

### 3.6 Architecture — *static content, several inaccuracies*
- Works: 12-stage list with detail panel.
- **Inaccurate file references:** `src/data/sec/collector.py`, `src/data/sec/parser.py`, `src/data/sec/chunker.py` do not exist. Actual: `src/data/ingestion/sec_corpus.py`, `sec_content.py`, `src/data/chunking/structure_aware.py`.
- Claims "JSON schema enforcement" (code uses Ollama `format="json"`, not a schema) and "+3.1% decision accuracy lift on comparative queries" (Phase 7 summary shows +3.12 pp overall decision accuracy for balanced RRF vs. frozen baseline, not a comparative-only figure). Stage 12 states "FEIR = 0.0% (Zero Future Evidence Infiltration Rate)" as a property rather than a benchmark observation.
- Descriptions are truncated in the list with no way to read them except selecting.

### 3.7 Shell
- Floating rounded panels with 26 px radii and heavy borders consume space; theme toggle duplicated in sidebar and header; fake profile badge; header connection pill says "CONNECTING…" forever when degraded; no mobile navigation (sidebar fixed at 256 px, overflows on phones); no collapse.

## 4. Technical limitations discovered
1. No package registry is reachable from the build environment, so **no new dependencies can be installed** — the redesign must use what is already in `node_modules`.
2. `next build` cannot run in the Linux verification VM (macOS-only SWC binary). TypeScript and ESLint run there; runtime behaviour is verified against the user's running dev server. The production build must be run on the Mac.
3. Python backend tests cannot run in the VM (macOS virtualenv).
4. New backend routes only become reachable after the user's uvicorn process reloads.

## 5. Proposed improvements

### 5.1 Design system
- One token set (CSS variables, RGB channels) for dark (default) and light themes, mapped into Tailwind (`bg-surface`, `text-fg-muted`, `border-line`, `accent`, semantic `pos/neg/warn/info`).
- Load the bundled Geist Sans / Geist Mono via `next/font/local`; system serif (Charter/Iowan/Georgia) for editorial headings and answer text.
- Reusable primitives in `components/ui/`: Button, IconButton, Input, Select, SegmentedControl/Tabs, Panel, DataTable styles, Drawer, Dialog, Tooltip, StatusDot, Spinner/Skeleton, EmptyState, ErrorState, Metric, Badge, ChartFrame, KeyValue.
- No-flash theme bootstrap script in `<head>`.

### 5.2 Shell
- Flush, compact sidebar (collapsible to icons, persisted), mobile drawer navigation.
- Header: page title + short description, research-context chip, live backend status popover sourced from `/health`, theme toggle, refresh. Remove the fake bell and profile.
- URL hash sync (`#graph`, `#evidence?…`) so tabs are linkable and back/forward works, without changing the app's single-route structure.

### 5.3 Shared research context
- `ResearchContextProvider` holding `{ nodeId, ticker, name }`, persisted per browser, shown in the header with change/clear.
- Graph opens the context entity; Evidence filters to it (visible, removable chip); Ask shows it as an *optional, opt-in* scope chip and **never rewrites the query** (the query is sent verbatim; the context is only a UI hint and suggestion source).

### 5.4 Pages
- **Overview:** executive briefing — system health ledger, corpus ledger from a real corpus endpoint, KG composition, historical Phase 7 panel with definitions and sample sizes, company universe of the 10 real issuers with per-issuer filing counts, date span and KG degree, linking into research context.
- **Ask:** analyst workspace — query composer with date picker for reference time, context chip, audited examples; result as an editorial answer with outcome state (Supported / Partially supported / Insufficient evidence / Validation issue / API error), evidence ledger, citations, graph paths, sufficiency, temporal check (with honest labelling of what the API measures), latency. AbortController + duplicate-submit guard; state survives tab switches.
- **Knowledge Graph:** canvas-first; default 24 strongest neighbours aggregated per entity (one edge per pair with multiplicity), radial layout with spacing, relationship filters, progressive "Load more relationships" and "Expand neighbour", explicit count ledger (total KG nodes/edges · direct edges · unique connected entities · loaded edges · shown after filter), collapsible inspector (drawer on small screens) with full provenance including attribution status. No fallback data — explicit not-found and offline states.
- **Evidence:** filing library (99 filings, filter by issuer/type/date, sort) → document reader listing its chunks → focused chunk reader with distinct filing date / fiscal period / availability time and SEC link.
- **Evaluation:** metric definitions, sample sizes, configuration table from the full `configurations` block, charts with honest axes, per-category decision accuracy from `per_question_results.csv`, explicit historical labelling.
- **Architecture:** corrected 12-stage pipeline with input/output, real file paths, technologies and research significance; stage navigator and connected flow.

### 5.5 Backend (additive, read-only, no research logic touched)
1. `GET /api/v1/corpus/filings` — filing manifest from `data/processed/documents/*.json` metadata (cached).
2. `GET /api/v1/evidence/chunks` — optional `document_id` and `document_type` filters (Qdrant payload filters).
3. `GET /api/v1/benchmark/per-question` — parsed `results/phase7/per_question_results.csv` (read-only).
4. `routes/graph.py` — return `confidence: null` when the edge has none, instead of `0.9`.

The query pipeline, retrieval, fusion, sufficiency, generation, validation and all `results/` files are not modified.

## 6. Files likely to change
- `frontend/src/app/{layout.tsx,page.tsx,globals.css}`, `frontend/tailwind.config.ts`
- `frontend/src/lib/{api.ts,types.ts,theme.tsx}` + new `lib/{research-context.tsx,format.ts,useApi.ts}`
- `frontend/src/components/{Header.tsx,Navigation.tsx}`, `components/common/*`, new `components/ui/*`
- `frontend/src/components/views/*.tsx` (all six)
- `api/routes/{evidence.py,graph.py,benchmark.py,__init__.py}`, new `api/routes/corpus.py`, `api/schemas/*`, `api/main.py` (router registration)
- `docs/UI_UX_DESIGN_SYSTEM.md`, `docs/UI_UX_IMPLEMENTATION_REPORT.md`, `docs/screenshots/`

`lib/graphFallbackData.ts` will no longer be imported (removed from the bundle) but is left on disk.

## 7. Implementation sequence
1. Backup (done) → 2. tokens, fonts, theme bootstrap, UI primitives → 3. shell + hash routing + research context → 4. API client cleanup (remove fallbacks, AbortSignal support) → 5. backend additive routes → 6. Overview → 7. Ask → 8. Graph → 9. Evidence → 10. Evaluation → 11. Architecture → 12. responsive/a11y pass → 13. typecheck, lint, live functional tests → 14. screenshots and reports.

## 8. Regression risks and mitigations
| Risk | Mitigation |
|---|---|
| Overwriting untracked frontend/api with no git history | Tarball backup in `archive/` before any change |
| Breaking the live query contract | `QueryRequest` / `QueryResponse` unchanged; Ask sends identical payload (`query`, `reference_time`, `top_k=10`, `candidate_pool_size=20`) |
| New backend routes not loaded | UI detects 404 and degrades with an explicit "requires backend restart" note rather than inventing data |
| Removing graph fallback exposes backend outages | Intended: explicit offline/error state instead of silently wrong data |
| Bundle/perf regressions in graph | Bounded default neighbourhood, memoised layout, no new libraries |
| Misstated research results | Every metric labelled "Historical · Phase 7 · n = …"; decision accuracy never called answer accuracy; FEIR caveat stated where shown |
