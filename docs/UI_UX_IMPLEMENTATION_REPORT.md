# FinGraph AI: UI/UX Implementation Report

**Date:** 2026-11-03
**Scope:** `frontend/` redesign (two passes) plus additive, read-only integration endpoints in `api/`
**Companion documents:** `docs/PRODUCT_AUDIT_AND_IMPLEMENTATION_PLAN.md` (audit and plan) and `docs/UI_UX_DESIGN_SYSTEM.md` (tokens, components, patterns)

This report separates what was **implemented and verified**, what was **implemented but not verified**, and what is **not done**. A test is listed as passed only if it was actually executed.

---

## 1. Existing architecture audit (summary)

The full audit is in `PRODUCT_AUDIT_AND_IMPLEMENTATION_PLAN.md`. In short:

- **Stack:** Next.js 14.2.35 (App Router), React 18, TypeScript (strict), Tailwind 3.4, React Flow 11, Recharts 3, lucide-react. FastAPI backend on `127.0.0.1:8000`, reached through the Next.js `/api` rewrite.
- **Structure:** one route with six tab views held in React state.
- **Git:** `frontend/` and `api/` are **untracked in git**. A backup was taken before any edit: `archive/frontend_pre_redesign_backup_2026-11-03.tar.gz`.

## 2. Problems discovered

| # | Problem | Severity | Resolution |
|---|---|---|---|
| 1 | The API client silently fell back to a bundled 3.6 MB graph snapshot. Unknown entities showed **Microsoft's** subgraph. | Integrity | Fallback removed. Explicit not-found and offline states. Snapshot no longer bundled. |
| 2 | `future_items_count` and `feir` are constants in `api/services/pipeline_service.py`, but the UI showed them as measurements (and formatted a ratio as a percentage). | Integrity | No longer shown as measurements. The UI explains that the generator's hard assertion enforces the guarantee, and counts post-reference evidence itself from the returned timestamps. |
| 3 | The graph API substituted `confidence = 0.9` when an edge had none. | Integrity | `api/routes/graph.py` returns `null`; the UI shows "Not recorded". |
| 4 | The Overview had a decorative sparkline, invented sector labels and filing ranges, dead buttons, a fake notification bell, and hard-coded fallback counts. | Integrity / UX | Replaced with live data, or removed. |
| 5 | The corpus was described as "10-K and 10-Q". It actually holds 50 8-K, 39 10-Q and 10 10-K. | Accuracy | Corrected from the ingestion manifest. |
| 6 | Architecture cited three non-existent files (`src/data/sec/*.py`). | Accuracy | Every path was re-verified against the repo. |
| 7 | The Phase 6 parametric-leakage result (PLR 0.80, 4 of 5 probes leaked) appeared nowhere in the UI. | Integrity | Shown in the Evaluation lab, the Overview limitations and the Architecture page. |
| 8 | **Ollama outages appeared as refusals.** With Ollama down, the backend returns HTTP 200 with `decision: "refuse"` and "Generation failed due to LLM backend error…" while the sufficiency gate passed. | UX / integrity | New **Generation failed** outcome, kept distinct from "Insufficient evidence" (found during live testing). |
| 9 | The graph default rendered 60 boxed, labelled nodes in overlapping rings. Its counts mixed up total direct edges (2,338), company edges (250) and unique companies (118). | UX | Aggregated readable neighbourhood plus an explicit count ledger. |
| 10 | Edge attribution (`unsupported_cross_company`) was never shown. | Integrity | Shown in the edge inspector with an explanation. |
| 11 | `favicon.ico` existed in both `public/` and `app/`, so Next.js returned 500. | Bug | The identical `public/favicon.ico` was moved to `archive/frontend_public_favicon/`. |
| 12 | 8-K "fiscal period" was labelled "period ending". | Accuracy | 8-Ks now show "report date". |
| 13 | No mobile navigation, no deep links, theme flash on load, tab state lost when switching tabs. | UX | Mobile drawer, URL-hash routing, pre-paint theme script, views kept mounted. |

## 3. Design direction implemented

**Pass 1** (editorial and institutional) established the token system, component library, shell, shared context and integrity fixes.

**Pass 2** (presentation-first, per the academic-demonstration brief) re-skinned to midnight navy, electric blue and restrained cyan. It adds:

- distinct card surfaces;
- an executive Overview;
- a polished Ask workspace;
- a five-way "results at a glance" evaluation dashboard;
- a methodology diagram on the Architecture page.

A faculty member can answer the brief's five questions from the first screens:

| Question | Where it is answered |
|---|---|
| What problem does FinGraph solve? | Overview hero |
| What data does it use? | Overview KPI cards (99 filings, 10 companies, 8,181 passages, 10,054 entities) |
| How does the architecture work? | Overview "How it works" five-step pipeline, then Architecture diagram and stages |
| What happens when someone asks? | Overview recorded FSH-07 walkthrough ("Run it live" prefills Ask), then the Ask page |
| How was it evaluated, and what are the limits? | Overview performance card (historical) and Known limitations; Evaluation lab |

Note: the brief mentions a visual reference image, but only the text brief was attached. The redesign follows the written description.

## 4. Design tokens

See `UI_UX_DESIGN_SYSTEM.md §2`. Tokens are RGB channels in `src/app/globals.css`, mapped in `tailwind.config.ts`. Cyan and card styles are CSS utilities (`.text-cyan`, `.bg-cyan-soft`, `.card`, `.card-interactive`) because `next dev` caches the Tailwind config.

## 5. Components

**Retained:** React Flow, Recharts, lucide icons, the `fetchJson` dual-URL strategy, all API types, `theme.tsx` storage key (`fingraph-theme`), the six views and their file names.

**Modified:**

- `app/layout.tsx`: Geist fonts via `next/font/local`, theme bootstrap.
- `app/page.tsx`: shell, hash routing, keep-mounted views, health polling with abort.
- `app/globals.css`, `tailwind.config.ts`, `lib/theme.tsx`, `lib/api.ts` (fallbacks removed, AbortSignal, new endpoints), `lib/types.ts`.
- `components/Header.tsx`, `components/Navigation.tsx`.
- `components/common/Badge.tsx` (now re-exports the token Badge).
- All six view files.

**Created:**

- `components/ui/`: `Button`, `IconButton`, `Input`, `Textarea`, `Select`, `FieldLabel`, `Segmented`, `Tabs`, `Panel`, `PanelHeader`, `SectionHeading`, `KeyValue`, `Metric`, `ChartFrame`, `Badge`, `StatusDot`, `Spinner`, `ProgressBar`, `Skeleton`, `EmptyState`, `ErrorState`, `Note`, `Drawer`, `Dialog`, `Popover`, `Tooltip`, `cx`.
- `components/ErrorBoundary.tsx`.
- `lib/research-context.tsx`, `lib/navigation.ts`, `lib/useApi.ts`, `lib/corpus.ts`, `lib/format.ts`, `lib/useTokens.ts`.

**Left on disk, unused:** `lib/graphFallbackData.ts` (no longer imported, so not bundled) and `components/common/StatCard.tsx`. They were not deleted because deleting files was out of scope.

## 6. Pages redesigned

| Page | Implemented |
|---|---|
| **Overview** | Hero (headline, description, Ask and Explore actions, live status pill). Recorded FSH-07 walkthrough with "Run it live". Four live KPI cards. Five-step pipeline (built once vs per question). Historical Phase 7 card: 92.19% (59/64), 45/50 answered, 14/14 correct refusals, 0/14 hallucinations, with the decision-accuracy caveat. Known-limitations card. Collapsible system and corpus details with a company coverage table. |
| **Ask FinGraph** | Prominent composer with "answer as of" date (validated to ISO), audited example cards, research-context hint (never inserted silently; "Insert name" is explicit). Answer card with status pill, as-of, latency, rounds and confidence; serif answer with numbered citation markers; numbered source cards with filing metadata. Tabs for evidence (expandable, provenance, SEC and in-app links), graph paths (labels resolved), checks and sources. Recorded Phase 7 comparison for audited questions. Session history, cancel, duplicate-submit guard, state kept across tabs. |
| **Knowledge graph** | Search combobox (keyboard-navigable), issuer quick-pick, Companies / All scope. Count ledger. 24 strongest neighbours by default ("Show more", "Load next 100 edges"). One aggregated edge per pair, thickness by edge count, trimmed so arrows stay visible. Relationship filters in a collapsible panel. Expand a neighbour, make it focal, set it as context. Inspector (drawer below 1024 px) with node and edge provenance including attribution status, accession number, document, chunk and SEC link. "Read source chunk" opens the Evidence explorer. |
| **Evidence explorer** | Filing library (99 filings) with search, issuer, form and date filters, sort, and research-context filter. Filing header (form, filing date, period or report date, accession, SEC link). Passages ordered by chunk index with find-in-filing highlighting. Serif reader with prev/next and a provenance ledger keeping filing date, reporting period, event date and availability separate. Linked-passage marker when arriving from Ask or the graph. Falls back to raw Qdrant browsing if the corpus endpoint is missing. |
| **Evaluation lab** | Historical, phase and run badges. Five glance cards: Decision, Answer correctness, Retrieval, Temporal leakage, Parametric leakage. Per-configuration metric groups with definitions checked against `src/evaluation/*` and deltas vs baseline. Phase 6 parametric-leakage probe table. Charts on 0-based axes. Five-way comparison table. Category chart and 64-row question table with filters. |
| **Architecture** | Methodology diagram of five phases in plain language (built once vs every question). A 14-stage navigator with purpose, input → output, why it matters, facts, verified file paths and technologies. Arrow-key navigation. |

## 7. Shared entity context

`ResearchContextProvider` (`lib/research-context.tsx`) holds `{ nodeId, name, ticker, type }`, persisted in `localStorage` (`fingraph-research-context`, wrapped in try/catch).

- **Header chip:** set, change or clear the context on every page.
- **Graph:** opens the context entity. Setting a neighbour as context does not re-centre the graph.
- **Evidence:** filters to the context issuer, shown as "Filtered by research context" with a "Show all" control.
- **Ask:** shows the context and suggested questions. It is never added to the query.
- **Overview:** company rows can set the context.

## 8. API changes

All additive and read-only; no research logic changed.

| Change | File |
|---|---|
| `GET /api/v1/corpus/filings`: filing manifest and per-issuer coverage from `data/processed/documents/*.json` and `sec_corpus_manifest.json` (cached) | `api/routes/corpus.py`, `api/schemas/corpus.py`, registered in `api/main.py` and `api/routes/__init__.py` |
| `GET /api/v1/evidence/chunks`: optional `document_id` and `document_type` filters, plus `total_matching` (Qdrant count) | `api/routes/evidence.py`, `api/schemas/evidence.py` |
| `GET /api/v1/benchmark/per-question`: parses `results/phase7/per_question_results.csv` | `api/routes/benchmark.py` |
| `GET /api/v1/benchmark/parametric-leakage`: returns the `parametric_leakage_probes` block from `results/phase6/full_benchmark/benchmark_evaluation_summary.json` | `api/routes/benchmark.py` |
| `confidence` returns `null` instead of a substituted `0.9` | `api/routes/graph.py` |

`QueryRequest` and `QueryResponse` are unchanged. Ask sends the same payload as before (`query`, `reference_time`, `top_k=10`, `candidate_pool_size=20`).

## 9. Research functionality preserved

- Not modified:
  - `src/` (chunking, temporal metadata, KG construction, retrieval, RRF, reranking, sufficiency, generation, validation);
  - `scripts/run_fingraph.py` and `api/services/pipeline_service.py`;
  - `results/phase5/`, `results/phase6/`, `results/phase7/`;
  - `FinGraph_AI_Research_and_Implementation_Audit.pdf`;
  - the 15 pre-existing uncommitted research modifications.
- No staging, committing, pushing, resetting or cleaning was performed.
- Historical figures are read from the results files through the API and labelled as historical. Decision accuracy is never presented as factual accuracy. FEIR = 0 is never presented as proof of zero parametric leakage.

## 10. Accessibility

- Visible focus ring on all controls, plus a skip-to-content link.
- `aria-label` on every icon-only button.
- Proper roles: radiogroup/radio, tablist/tab, combobox/listbox, `aria-current` on navigation and stages.
- Dialogs and drawers: labelled, Escape closes, focus moves in and is restored.
- Results and graph counts are announced through `aria-live` regions.
- Status and outcomes combine shape or icon, colour and text.
- `prefers-reduced-motion` is respected.
- Charts and graph colours come from theme tokens, so both themes stay readable.

## 11. Responsive

| Width | Behaviour |
|---|---|
| ≥ 1280 px | Ask gains its side rail; the evidence reader becomes a third column |
| ≥ 1024 px | Collapsible sidebar (state persisted), graph inspector docked |
| < 1024 px | Navigation drawer, graph inspector as a drawer, graph filter panel collapsed by default, evidence list and filing panes switch with a back control |
| < 1280 px | Evidence reader opens as a drawer |

Tables scroll horizontally inside their cards. Measured page-level horizontal overflow at 390 px: 0 px on Overview, Ask, Graph and Evidence.

## 12. Performance

- The 3.6 MB fallback snapshot was removed from the client bundle (no longer imported).
- The graph renders at most 24 neighbours plus 8 per expansion by default. Layout and elements are memoised, and edges are aggregated per pair.
- The corpus manifest and node labels are cached once per session (`lib/corpus.ts`). The backend caches the manifest with `lru_cache`.
- Every fetch accepts an AbortSignal; stale requests are aborted.
- Search is debounced at 220 ms.
- Health is polled every 30 s with abort.
- Views stay mounted after first visit, so there are no repeat fetches on tab switches.
- Bundle size was **not measured**, because the production build was not run.

## 13. Tests executed

All runs were against the live backend on `127.0.0.1:8000` and the `next dev` server on `localhost:3000`, driven by Playwright on the author's machine.

| Test | Result |
|---|---|
| `tsc --noEmit` (TypeScript strict) | **Pass**, 0 errors (final run after the last edit) |
| ESLint (`next/core-web-vitals`) | **Pass**, 0 errors, 0 warnings (baseline before redesign: 1 warning) |
| Overview: hero, KPI cards and historical card load from live endpoints | **Pass** (99 / 10 / 8,181 / 10,054; 92.19%, 45/50, 14/14, 0/14) |
| Ask: TPT-02 with reference time 2025-06-01 | **Pass**: "Insufficient evidence" in 2.3 s, missing `company:NVDA`, all returned evidence available before the cutoff |
| Ask: FSH-07 with Ollama offline | **Pass for error handling**: shown as "Generation failed" (outcome added after this test exposed the issue) |
| Ask: FSH-07 live answer through the API (before the redesign) | **Pass**: "Microsoft acquired Activision Blizzard.", 2 citations, `is_valid: true`, ~34 s |
| Ask: cancel a running query | **Pass**: run cleared, Run re-enabled |
| Ask: example prefill and "Run it live" prefill | Example prefill **pass**; Overview → Ask prefill implemented, **not separately exercised** |
| Graph: NVIDIA, Microsoft, Apple | **Pass**. Ledger matches the API (2,338 / 118 / 250; 374 / 15 / 58; 213 / 13 / 27) |
| Graph: search "Apple" → select, unknown "zzqxv" | **Pass**: Apple Inc. first; clear no-match message |
| Graph: edge inspection and provenance | **Pass**: AAPL↔GOOGL, 7 edges, confidence 0.90, verified issuer, accession, document, chunk |
| Graph: expand neighbour | **Pass**: 14 → 22 nodes |
| Graph → Evidence "Read source chunk" | **Pass**: AAPL 10-Q filed 2026-07-31, period ending 2026-06-27, linked passage marked |
| Evidence: issuer, form, search filters and reset | **Pass**: 99 → 4 → 1 → 99 |
| Evidence: find in filing | **Pass**: 13 of 54 passages, highlights rendered |
| Evaluation: configuration switch | **Pass**: baseline 85.94% (55/64); 64 question rows |
| Evaluation: parametric leakage section | **Not verified live**: endpoint returns 404 until the backend restarts; the UI shows the restart note |
| Architecture: phase card and next stage | **Pass** |
| Themes: dark, light, persistence | **Pass**: screenshots in both; light class persisted across reload |
| Responsive: 1440, 820 and 390 px | **Pass**: screenshots; 0 px horizontal overflow at 390 px; mobile drawer has 6 links |
| Console and network | Only error: `404 /api/v1/benchmark/parametric-leakage` (pending restart). Remaining warning: React Flow `nodeTypes` notice under development Strict Mode. Favicon 500 and React Flow zero-size warning fixed. |
| Backend Python tests (`tests/test_api_*.py`) | **Not run**: the verification VM cannot load the macOS virtualenv. New routes were syntax-checked only. |
| `next build` (production) | **Not run**: see §14 |

## 14. Production build result

**Not executed.** The Linux verification VM cannot load the macOS-only SWC binary in `node_modules`, and running `next build` on the user's machine alongside the active `next dev` would overwrite the dev server's `.next` directory. To run it:

```bash
cd frontend
# stop `npm run dev` first
npm run build
```

## 15. Screenshots

All in `docs/screenshots/`.

**`before/`:** all six pages in dark and light, original UI.

**`after/` (v2, final):**

| Variant | Files |
|---|---|
| Dark, 1440 px | `v2-{overview,ask,graph,evidence,evaluation,architecture}-dark.png` |
| Light, 1440 px | `v2-{overview,ask,graph,evidence,evaluation,architecture}-light.png` |
| Tablet, 820 px | `v2-{overview,ask,graph,evidence}-tablet.png` |
| Mobile, 390 px | `v2-{overview,ask,graph,evidence}-mobile.png`, `v2-nav-mobile.png` |

Flows: `v2-ask-tpt02-refusal-dark.png`, `v2-ask-fsh07-dark.png` (Ollama offline), `v2-graph-aapl-edge-dark.png`, `v2-graph-expanded-dark.png`, `v2-evidence-linked-dark.png`, `v2-evidence-search-dark.png`.

**`after/` (other):** `v1-*` are first-pass screenshots, kept for comparison. `wip-overview-dark.png` is an intermediate capture of the Tailwind-config cache error and is not the final UI.

## 16. Remaining limitations

1. **Live FSH-07 answer not re-verified in the final UI.** Ollama was offline during final testing. Start Ollama and run FSH-07 from Ask to confirm the supported-answer card.
2. **Backend restart needed** for `/api/v1/benchmark/parametric-leakage`.
3. **Production build and backend tests not run** (§13, §14).
4. **Graph nodes are not draggable.** They are positioned deterministically, and dragging was disabled rather than left half-working.
5. **Chunk `section` metadata is null** in the corpus payloads, so it is shown as "Not recorded".
6. **Node aliases and temporal fields are absent** on KG node records. The inspector says so and shows temporal data on edges.
7. **"Company-type nodes" (1,093) includes extraction artefacts** such as "the Company". It is labelled accordingly, and the KG itself was not cleaned (research data).
8. **Pipeline stage list in Ask while running is descriptive**, not live progress, and is labelled that way. The API does not stream stages.
9. **The visual reference image in the brief was not received.**

## 17. Post-review fixes (reported by the user)

| Issue | Cause | Fix | Verified |
|---|---|---|---|
| The graph appeared over other pages. | The inactive graph view used `visibility: hidden`, but React Flow sets `visibility: visible` on its nodes, so they leaked through. | Every inactive view uses `display: none` (`app/page.tsx`). React Flow re-fits when the graph is reopened. | After visiting the graph, 0 graph nodes render on Overview, Ask, Evaluation or Architecture, at 1440 px and 390 px. The graph renders again on return. |
| Pages could not be scrolled to the end. | The leaked graph sat over the page and captured the mouse wheel (React Flow zooms on scroll). On phones and tablets, the Evidence filing library could not shrink, so the list overflowed instead of scrolling. | The graph-leak fix above. The library panel now fills and scrolls (`EvidenceInspectorView.tsx`). | Overview, Ask, Evaluation and Architecture reach their last element at 1440 px and 390 px. The Evidence library reaches filing 99 of 99 at 1440, 820 and 390 px. |
| Two files had not reached the Mac. | Some file transfers reported success but delivered a stale copy: `api/routes/benchmark.py` (missing the parametric-leakage route) and `lib/navigation.ts`. | Both written directly on the device. All changed frontend, API and doc files checked by MD5 against the source copies. | Checksums match. TypeScript and ESLint re-run on the device copy: 0 errors, 0 warnings. |

The parametric-leakage endpoint now exists in `api/routes/benchmark.py`. It needs one more FastAPI restart to load.
