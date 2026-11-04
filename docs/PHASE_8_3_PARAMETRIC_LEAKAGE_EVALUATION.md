# Phase 8.3 — Controlled Parametric Leakage Evaluation of Qwen3.5:4B

**Run date:** 2026-11-03 / 2026-11-04 · **Model under test:** `qwen3.5:4b` via Ollama (the production FinGraph generator)
**Artefacts:** `results/phase8_3/` · **Scripts:** `scripts/phase8_3/` · Phase 5–7 results were not modified.

## 1. Summary

| Measure | Result | 95% Wilson CI |
|---|---|---|
| Valid future facts tested | **142** (84 before the model's public release, 58 after) | — |
| **PLR** — closed book, "as-of" framing (A2): recalled / valid × 100 | **0 / 142 = 0.0%** | 0.0 – 2.63% |
| PLR, pre-release window only (the facts that *could* be in training data) | **0 / 84 = 0.0%** | 0.0 – 4.37% |
| Future fact recall — closed book, no date framing (A1) | **0 / 142 = 0.0%** | 0.0 – 2.63% |
| Forced-estimate hits within 1% (A3), pre-release vs post-release control | 0 / 84 vs 1 / 58 | Fisher p = 0.41 |
| Grounded pipeline (retrieval on, future evidence excluded) | <<G_SUMMARY>> | |

**Interpretation.** On this benchmark, no parametric leakage was detected for Qwen3.5:4B: in no condition did it
reproduce a held-out financial figure, and its forced guesses were no more accurate for facts that pre-date its
release than for facts it cannot have seen. This is a measurement on 142 facts of one type (reported financial figures
of ten US large caps); it is **not** a universal leakage rate for the model, and it does **not** establish zero
leakage — the upper 95% bound for the pre-release window is 4.37%.

**Correction of an earlier figure.** The PLR = 0.8 (4/5) shown in earlier builds of the Evaluation Lab came from the
Phase 6 probe set and was reported as measured with Llama 3.1 8B. It must not be attributed to Qwen3.5:4B (§2). The UI
now shows Phase 8.3 as the parametric result and keeps the Phase 6 probes in a labelled, collapsed section.

## 2. Audit of the existing parametric evaluation

| Finding | Evidence |
|---|---|
| The Phase 6 PLR probes are **manual labels, not model outputs.** All five records have `answer: null`, `verification_method: "manual"`, `reason: "Ground truth label provided (manual)."` | `results/phase6/full_benchmark/benchmark_evaluation_summary.json → parametric_leakage_probes.details.plr_results` |
| No model name, digest or decoding settings are stored with the probes. | same file |
| The evaluator returns the preset label without calling the LLM when one is present (`future_information_present` set). | `src/evaluation/parametric_leakage.py` |
| n = 5, all at reference 2024-01-01, all events from 2024 (NVIDIA Blackwell, Rubin, Run:ai; Gemini 1.5 …) — events well inside the training window of 2025-era models. | same file |
| Production generator: `qwen3.5:4b`, temperature 0, `think=False`, `format=json`, `num_predict=2048`. | `src/qa/generator.py` |

Conclusion: the Phase 6 number cannot be reproduced from stored artefacts and does not describe the production model.
Phase 8.3 is a new, separately stored measurement; the Phase 6 files are unchanged.

## 3. Benchmark (PLB-8.3)

File: `results/phase8_3/benchmark/parametric_leakage_benchmark.json` (sha256 `f692f917…e92304`, deterministic build by
`scripts/phase8_3/build_benchmark.py`).

- **Facts:** 142 = 49 total revenue + 49 net income + 44 diluted EPS, from **49 periodic SEC filings** (10-Q / 10-K) of
  AAPL, AMZN, GOOGL, JNJ, JPM, META, MSFT, NVDA, TSLA, XOM (XOM EPS omitted: not stated in a form that could be verified
  verbatim). Fiscal periods ending 2025-03-31 … 2026-07-26; filings published 2025-04-30 … 2026-08-26.
- **Ground truth:** each value was read from the face of the income statement and recorded in an explicit table in
  `build_benchmark.py`; the build **asserts the value appears verbatim in the filing text** and stores the accession,
  EDGAR URL and the verbatim table context with every fact. Values were cross-checked against prior-year comparatives
  (and, for JPM, quarterly sums equalling the fiscal year). No value was generated or estimated.
- **Temporal split:** `reference_time` = fiscal period end (00:00 UTC). Every fact became public only with its filing,
  23–49 days later, so `public_availability_time > reference_time` for all 142 facts (asserted at build time).
- **Stratification by model release.** Qwen3.5-4B's public release is 2026-03-02 (OpenCode model registry); its
  training cutoff is not published.
  - `pre_release` (84 facts, filed before 2026-03-02) — could be in the training data; this is where leakage could appear.
  - `post_release` (58 facts, filed after) — cannot be in the weights; a **control** for lucky guesses.
- **No duplicates:** one fact per (company, period, metric). Questions name the company, the metric label used in the
  filing (e.g. "total net sales", "net income attributable to common stockholders") and the exact period.

## 4. Protocol

### 4.1 Closed book (Experiment A)

Exact production model (`qwen3.5:4b`, digest `sha256:2a654d98…eefd`, Q4_K_M, 4.66 B params, Ollama 0.33.3), called
directly on `/api/chat`. **Retrieval, knowledge graph, documents, tools, browsing and conversation history are all
disabled**: one fresh system + user turn per question. Decoding: temperature 0, seed 0, `num_predict` 256,
`think: false`, `format: json` (same as production apart from the shorter output budget). Full prompts: `config/evaluation_config.json`.

| Condition | System framing | Measures |
|---|---|---|
| **A1 unconstrained** | answer from own knowledge; say if unknown | future fact recall |
| **A2 as-of** | "Today's date is {reference_date}"; use only information public by then; say if not yet public | **PLR** |
| **A3 forced estimate** | must give a best numeric estimate; no refusal | latent knowledge, compared with the post-release control |

A3 exists because abstention can hide knowledge: if the weights contained a figure, a forced estimate should land on
it more often in the pre-release window than in the control.

### 4.2 Scoring rules (`scripts/phase8_3/score.py`, stdlib, deterministic)

1. Parse the JSON `value`; if absent, fall back to every number in the answer sentence (conservative: favours detecting leakage).
2. Revenue / net income ($ M): **recalled** if |pred − truth| ≤ 1% × |truth| (0.5% and 2% reported as sensitivity).
   Values given in billions or dollars are rescaled when that lands within tolerance.
3. Diluted EPS: **exact to the reported cent** (|pred − truth| ≤ $0.005).
4. Abstentions (null value) are **not recalled**.
5. A matching value counts as recalled **even if the model also says the figure was not yet public**.
6. PLR = recalled / valid facts × 100; Wilson score 95% intervals; Fisher exact two-sided test for pre vs post (`stats.py`).
7. Manual review of all 426 closed-book responses (`scoring/manual_inspection_closed_book.md`). No score was overridden
   (`scoring/manual_overrides.json` is empty).

### 4.3 Grounded pipeline (Experiment B)

Each question is sent to the unchanged production endpoint `POST /api/v1/query` with `reference_time` = the fact's
reference time, `top_k` 10, `candidate_pool_size` 20. Because every fact became public after its reference time, the
correct behaviour is a **refusal / insufficient evidence**, and no evidence available after `reference_time` should enter
the context or citations. Scored per question: refusal vs answer, whether an answer states the future figure
(unsupported future claim), and FEIR (share of context / cited evidence with `available_time > reference_time`).

## 5. Closed-book results

### 5.1 Headline (426 responses, 0 request errors, 0 JSON parse failures)

| | A1 unconstrained | A2 as-of (PLR) | A3 forced estimate |
|---|---|---|---|
| Recalled, all (within 1% / exact cent) | 0 / 142 (0.0%, CI 0–2.63) | **0 / 142 (0.0%, CI 0–2.63)** | 1 / 142 (0.7%, CI 0.12–3.88) |
| Recalled, pre-release window | 0 / 84 (CI 0–4.37) | **0 / 84 (CI 0–4.37)** | 0 / 84 (CI 0–4.37) |
| Recalled, post-release control | 0 / 58 (CI 0–6.21) | 0 / 58 (CI 0–6.21) | 1 / 58 (1.7%, CI 0.30–9.14) |
| Not recalled | 142 | 142 | 141 |
| Abstained | 142 (100%) | 142 (100%, all `not_yet_public`) | 0 |
| Pre-release within 0.5% / 1% / 2% | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 3 |
| Median relative error of stated values | — | — | 0.32 |

The single A3 hit (XOM, quarter ended 2026-06-30, net income 14,500 vs 14,525 $M) is in the **control** window, so it
is a round-number guess, not memorisation.

### 5.2 By company and category (A2, pre-release window)

Every company scored 0 recalled: AAPL, AMZN, GOOGL, JNJ, META, MSFT, NVDA, TSLA 0/9 each (CI 0–29.9%); JPM, XOM 0/6
(CI 0–39.0%). By category: revenue 0/29 (CI 0–11.7%), net income 0/29 (CI 0–11.7%), diluted EPS 0/26 (CI 0–12.9%).
By filing quarter (all windows): 2025-Q2 0/3, Q3 0/23, Q4 0/29, 2026-Q1 0/29, Q2 0/29, Q3 0/29. Identical for A1.
Per-group CIs are wide; these tables show the absence of concentrated leakage, not per-company rates.

### 5.3 Latent-knowledge probe (A3): pre-release vs control

| Test (A3) | Pre-release | Post-release control | Fisher p |
|---|---|---|---|
| Within 1% (primary rule) | 0 / 84 | 1 / 58 | 0.41 |
| Within 2% | 3 / 84 | 2 / 58 | 1.00 |
| Relative error ≤ 5% | 6 / 84 | 5 / 58 | 0.76 |
| Revenue, ≤ 10% | 12 / 29 | 4 / 20 | 0.14 |
| Net income, ≤ 10% | 4 / 29 | 1 / 20 | 0.64 |
| Diluted EPS, ≤ 10% | 3 / 26 | 3 / 18 | 0.68 |
| Median relative error | 0.30 | 0.34 | — |

No test separates the windows. The revenue ≤ 10% difference is the largest (not significant) and is expected even
without leakage: revenue grows smoothly, and an extrapolation from the model's last known figures is naturally closer
for periods nearer its training data.

### 5.4 Manual inspection

- No A1 or A2 response contains a figure (only dates/years).
- 268 / 284 abstentions cite the knowledge cutoff or that the period is in the future. 16 give a **false** reason
  (e.g. "Tesla has no fiscal quarter ending June 30", "J&J's fiscal year ends March 1"). They remain abstentions but
  show unreliable fiscal-calendar reasoning in closed-book mode.
- A3 answers present themselves as extrapolations ("based on current growth trends"); one mislabels META Q2 2025 as Q1.

## 6. Grounded-pipeline results

<<GROUNDED>>

## 7. Parametric vs retrieval-based temporal leakage

| | Retrieval-based (FEIR) | Parametric (PLR) |
|---|---|---|
| Channel | future documents enter the context or citations | future facts surface from the model weights |
| Control in FinGraph | `available_time ≤ reference_time` filter on every evidence item, before ranking and citation | none at the architecture level; the generator is instructed to answer only from cited evidence and refuses on insufficient evidence |
| Measured in | Phase 6–7 (FEIR = 0 at every stage) and §6 here | §5 here (closed book) |
| Result in this phase | <<FEIR_LINE>> | PLR 0/142 (CI 0–2.63%); pre-release 0/84 (CI 0–4.37%) |

The two are independent: a zero FEIR shows the retriever keeps future documents out, but says nothing about what the
generator already knows. Phase 8.3 measures the second channel directly, with retrieval switched off, and then checks
that in the full pipeline the grounding requirement turns "no admissible evidence" into a refusal rather than an
answer from memory.

## 8. Limitations

- **Finite, narrow benchmark.** 142 facts, one fact type (reported revenue, net income, EPS), ten very large US
  companies. Results may not transfer to other fact types (products, M&A, executives, macro data), smaller firms, or
  qualitative information. This is not a universal leakage rate.
- **"No leakage detected" ≠ "no leakage".** With 0/84 in the pre-release window the 95% upper bound is 4.37%.
- **Unknown training cutoff.** The split uses the public release date (2026-03-02); the true data cutoff is earlier and
  unpublished, so part of the "pre-release" window may also be unseen, which makes the window a conservative superset.
- **Exact-figure recall is a high bar.** A model could know a direction ("revenue grew") without the number; such
  qualitative leakage is not measured.
- **Abstention may hide knowledge.** A3 addresses this, but a model trained to refuse could still under-report.
- **Single deterministic sample** (temperature 0, seed 0), **Q4_K_M quantisation**, and a 256-token budget (production
  uses 2048; JSON answers were < 100 tokens, all `done_reason: stop`).
- Grounded run uses the live local corpus and index; results depend on that corpus snapshot.

## 9. Reproduction

```bash
python3 scripts/phase8_3/build_benchmark.py            # rebuilds the benchmark, asserts verbatim + availability
python3 scripts/phase8_3/run_closed_book.py            # A1–A3 against local Ollama (requires `ollama` client)
# or in-browser runner used for this run: scripts/phase8_3/run_in_browser.js (plbRun(cfg,'closed'|'grounded'))
python3 scripts/phase8_3/score.py results/phase8_3     # scoring + metrics/final_metrics.json
python3 scripts/phase8_3/stats.py results/phase8_3     # metrics/statistical_tests.json
```

| Artefact | Path (under `results/phase8_3/`) |
|---|---|
| Benchmark with sources | `benchmark/parametric_leakage_benchmark.json` |
| Config / model version | `config/evaluation_config.json`, `config/model_version.json` |
| Raw responses | `raw/closed_book.jsonl`, `raw/grounded.jsonl` |
| Run logs | `logs/closed_book_run_log.json`, `logs/grounded_run_log.json` |
| Scored items | `scoring/closed_book_scored.jsonl`, `scoring/grounded_scored.jsonl` |
| Manual review | `scoring/manual_inspection_closed_book.md`, `scoring/manual_overrides.json` |
| Metrics | `metrics/final_metrics.json`, `metrics/statistical_tests.json` |
