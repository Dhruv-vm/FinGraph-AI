# Contributing to FinGraph AI

Thank you for your interest in contributing to FinGraph AI. This project is a research prototype investigating temporal knowledge graphs and hybrid GraphRAG for multi-hop financial question answering over SEC filings.

To maintain research rigor, reproducibility, and software quality, all contributions must adhere to the guidelines set out below.

---

## 1. Development Environment Setup

### Prerequisites
- **Operating System:** macOS (Apple Silicon recommended) or Linux (x86_64 / arm64).
- **Python:** 3.12.x strictly.
- **Ollama:** Local Ollama runtime with `qwen3.5:4b` installed.
- **Git:** Standard git version control.

### Setup Steps
```bash
# 1. Clone the repository
git clone https://github.com/Dhruv-vm/FinGraph-AI.git
cd FinGraph-AI

# 2. Set up the virtual environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Pull the required local model
ollama pull qwen3.5:4b

# 4. Verify test suite passes (177 tests)
.venv/bin/pytest tests/ -q
```

---

## 2. Coding Conventions & Code Quality

- **PEP 8 & Formatting:** Follow standard PEP 8 naming and indentation conventions (4 spaces).
- **Type Annotations:** All public functions, dataclasses, and method signatures must include strict Python type hints (`from typing import Optional, Any, ...` or Python 3.12 syntax `str | None`).
- **Data Structures:** Prefer typed `@dataclass` models (e.g., `RetrievalEvidence`, `QAEvidencePackage`, `QAAnswer`) or Pydantic models for structured validation.
- **Deterministic LLM Invocations:** All LLM calls must use deterministic decoding (`temperature=0.0`, `think=False`). Never introduce stochastic sampling in benchmarks or retrieval components.
- **No Hardcoded Benchmark Results:** Do not write code or tests that bypass retrieval to return pre-computed benchmark answers. All evaluations must flow through the actual retrieval and QA pipeline.
- **Documentation Integrity:** Preserve all existing comments, docstrings, and mathematical justifications when refactoring code.

---

## 3. Testing Standards

FinGraph AI maintains a rigorous test suite of **177 tests** spanning unit testing, schema verification, temporal gate enforcement, and pipeline integration.

### Running Tests
```bash
# Run the complete test suite
.venv/bin/pytest tests/

# Run targeted test suites
.venv/bin/pytest tests/test_evaluation_metrics.py tests/test_evaluation_pipeline.py -q
.venv/bin/pytest tests/test_qa_evidence_sufficiency.py -q
.venv/bin/pytest tests/test_hybrid_retriever.py -q
```

### Test Coverage Expectations
- Any change to `src/retrieval/` must include tests verifying that point-in-time constraints (`t_avail <= t_ref`) are respected and that future edges or chunks receive $-\infty$ or are filtered.
- Any change to `src/qa/evidence_sufficiency.py` must include regression tests for compound relationships (e.g., `SUPPLIES` + `HAS_RISK`), ensuring disconnected evidence is rejected.
- Any change to `src/evaluation/` must test edge cases such as empty query sets, zero temporal queries (null pre-filter rate), and missing ground-truth chunks without throwing unhandled exceptions.

---

## 4. Contributing Benchmark Questions

The benchmark dataset lives in `benchmark/questions.json` and must adhere to the verified schema defined in `benchmark/README.md`.

### Benchmark Contribution Protocol
1. **Verification Against Chunks:** Every candidate question must cite verified `relevant_chunk_ids` present in `data/processed/chunks/`.
2. **Verification Against Knowledge Graph:** Relational questions must have verified paths in `data/processed/unified/fingraph_semantic_kg.json`.
3. **Point-in-Time Constraint:** Temporal questions must provide an unambiguous `reference_time` in ISO 8601 format (e.g., `2026-08-21T00:00:00+00:00`).
4. **Refusal Cases:** We actively welcome unanswerable refusal test cases. These must have:
   - `"answerable": false`
   - `"unanswerability_rationale"` detailing why the question cannot be answered from the corpus (e.g., post-cutoff event, entity outside the 10-company universe, or missing relationship chain).
5. **No Synthetic Shortcuts:** Never use unverified synthetic questions. All questions must reflect authentic regulatory SEC disclosure queries.

---

## 5. Artifact Isolation & Reproducibility Rules

To prevent accidental data contamination and maintain research reproducibility:

- **Canonical Results Isolation:** Do not overwrite canonical evaluation artifacts in `results/phase5/benchmark_evaluation_summary.json` or `results/phase5/per_question_results.csv` in regular PRs.
- **Untracked Experiment Outputs:** Keep temporary experimental outputs in new subdirectories (e.g., `results/experiments/<experiment_id>/`).
- **Research Audit Documents:** Keep external audits such as `FinGraph_AI_Research_and_Implementation_Audit.pdf` untracked.
- **No Large File Commits:** Do not commit raw SEC filings (`data/raw/sec_filings/*.html`), large parquet files, or local vector database checkpoints.
- **Deterministic Re-runs:** When submitting an evaluation change, document the exact command line, model version, and platform used.

---

## 6. Pull Request Checklist

Before submitting a pull request, ensure you have completed the following steps:

- [ ] Ran `.venv/bin/pytest tests/` and confirmed all **177 tests pass**.
- [ ] Confirmed `git diff --check` reports zero whitespace or formatting errors.
- [ ] Verified that no secrets, credentials, or absolute local system paths (`/Users/...` or `/home/...`) are present in the diff.
- [ ] Verified that `results/` and any PDF audit files remain untracked or properly isolated.
- [ ] Verified that existing documentation and docstrings have been maintained or updated to reflect code changes.
- [ ] Provided a clear PR description detailing:
  1. The motivation for the change.
  2. The specific components modified.
  3. The exact test commands executed and their output.
