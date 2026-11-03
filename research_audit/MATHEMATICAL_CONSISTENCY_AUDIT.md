# FinGraph AI — Mathematical Consistency and Rigor Audit

**Audit Date:** 2026-11-01  
**Auditor:** Mathematical Modelling Specialist  

---

## 1. Normalization Bounds and Range Invariants

Every mathematical function was audited for value domain safety:

| Function / Metric | Theoretical Domain | Implemented Range | Boundary Condition Handling | Audit Verdict |
|---|---|---|---|---|
| **Cosine Similarity** | $[-1.0, 1.0]$ | $[0.0, 1.0]$ (via all-MiniLM) | MiniLM vectors are unit-normalized; non-negative in practice | **SOUND** |
| **Reciprocal Rank Fusion** | $(0, |\mathcal{M}|/k]$ | $(0, 2/61] pprox (0, 0.0328]$ | Monotonically decreasing with rank; no division by zero ($k=60$) | **SOUND** |
| **FEIR Infiltration Rate** | $[0.0, 1.0]$ | $[0.0, 1.0]$ or `None` | If $|\mathcal{E}| = 0$, returns `None` (avoids 0/0 error) | **SOUND** |
| **Token F1 Score** | $[0.0, 1.0]$ | $[0.0, 1.0]$ | If $|\mathcal{P}| = 0$ and $|\mathcal{G}| = 0$, returns 1.0; if sum=0 returns 0.0 | **SOUND** |
| **Exact Match** | $\{0.0, 1.0\}$ | $\{0.0, 1.0\}$ | Binary indicator | **SOUND** |
| **Decision Accuracy** | $[0.0, 1.0]$ | $[0.0, 1.0]$ | Ratio of $(TP + TN) / N$ | **SOUND** |

---

## 2. Parameter Leakage and Lookahead Checks

### Temporal Lookahead Guard:
In `src/data/graph/snapshot.py`, snapshot filtering evaluates:
$$t_{	ext{available}} \le t_{	ext{as\_of}}$$
- **Audit Finding:** The timestamp parser (`parse_datetime`) converts naive date strings into UTC timezone-aware objects.
- **Microsecond Infiltration Check:** All SEC filing publication timestamps are set to end-of-day or explicit EDGAR acceptance seconds. Chunks published at `2025-06-30T16:05:00Z` are excluded from snapshots set to `2025-06-30T00:00:00Z`. This prevents intraday data leakage.

### Walk-Forward Split Guard in Predictive Experiment:
In `archive/old_ml_experiments/models/baseline.py:chronological_split`:
- Splitting is performed purely chronologically on sorted index timestamps.
- Standard scaler statistics ($\mu, \sigma$) are computed **strictly on the training window** ($N_{	ext{train}} = 283$) and applied to the test window ($N_{	ext{test}} = 71$) without test-set re-estimation.
- No lookahead leakage detected.

---

## 3. Notation Discrepancies and Code Realities

1. **RRF Constant $k$:**
   - In theoretical IR literature, $k$ is often written as $k=60$. In `src/retrieval/hybrid.py`, $k=60$ is hardcoded as default, matching literature standard.
2. **Entity Quota $k_{	ext{min}}$:**
   - In comparative queries, $k_{	ext{min}} = 3$. With $K = 8$ and 2 target entities, $3 + 3 = 6$ slots are reserved, leaving $8 - 6 = 2$ slots for global rank leaders. This ratio ($75\%$ reserved, $25\%$ open) guarantees dual-entity representation while permitting high-scoring contextual evidence.
