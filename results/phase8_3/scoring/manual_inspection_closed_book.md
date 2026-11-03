# Phase 8.3 — manual inspection of closed-book responses

Inspected: all 426 raw responses (`raw/closed_book.jsonl`). No score was changed (`manual_overrides.json` is empty).

1. **A1 / A2 contain no figures.** Every A1 and A2 response has `value: null`; a scan of the answer
   sentences found no numbers other than dates/years. All 142 A2 responses have `status: "not_yet_public"`.
2. **Abstention reasons.** 268 of 284 A1/A2 abstentions cite the knowledge cutoff or that the period is in the
   future. 16 give a *false* reason — e.g. "Tesla's fiscal year ends in December, so there is no fiscal quarter
   ending June 30, 2025", "Johnson & Johnson's fiscal year ends on March 1st". These are abstentions (not
   recalled) but show unreliable reasoning about fiscal calendars. Not leakage.
3. **A3 forced estimates are framed as extrapolations** ("based on current growth trends"). Near hits:
   AMZN Q3-2025 revenue 178,000 vs 180,169 (1.2%), META Q2-2025 revenue 46,800 vs 47,516 (1.5%, mislabelled
   "Q1"), GOOGL FY-2025 revenue 410,000 vs 402,836 (1.8%) — all outside the 1% primary tolerance.
   The only primary-tolerance hit, XOM Q2-2026 net income 14,500 vs 14,525 (0.17%), is in the
   post-release control window, i.e. it cannot come from training data: a round-number guess.
4. **Pre-release vs control (A3).** within 2%: 3/84 vs 2/58 (Fisher two-sided p = 1.0);
   within 5%: 6/84 vs 5/58 (p = 0.76); revenue within 10%: 12/29 vs 4/20 (p = 0.14, not significant).
