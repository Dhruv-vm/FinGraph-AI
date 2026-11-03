"""Phase 8.3 — score raw closed-book and grounded responses (stdlib only).

    python3 scripts/phase8_3/score.py results/phase8_3

Inputs  : benchmark/parametric_leakage_benchmark.json, raw/closed_book.jsonl, raw/grounded.jsonl,
          scoring/manual_overrides.json (optional, documented manual decisions)
Outputs : scoring/closed_book_scored.jsonl, scoring/grounded_scored.jsonl, metrics/final_metrics.json

Scoring rules (closed book)
  * Parse the model's JSON. If it fails, fall back to extracting numbers from the raw text.
  * A response RECALLS the fact if the reported value matches ground truth within tolerance:
      revenue / net income ($ millions): |pred - truth| <= 1% of |truth| (primary; 0.5% and 2% reported)
      diluted EPS ($/share)            : exact to the reported cent (|pred - truth| <= $0.005);
                                         sensitivity runs use max(tol x |truth|, $0.005)
    Values the model states in billions are converted (x1000) when the response says "billion" or the
    value is < 1/100 of truth and x1000 lands within tolerance; this is recorded per item.
  * Abstentions (status unknown / not_yet_public with null value) are NOT recalled.
  * A recalled value counts even if the model also claims it was not yet public (it still leaked).
PLR = recalled / valid facts tested x 100, with a Wilson 95% interval.
"""
import json, math, re, sys
from collections import defaultdict
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, centre - half) * 100, 2), round(min(1.0, centre + half) * 100, 2)]


def rate(k, n):
    return {"k": k, "n": n, "pct": round(100 * k / n, 2) if n else None, "wilson95_pct": wilson(k, n)}


NUM_RE = re.compile(r"-?\$?\s*\(?-?[0-9][0-9,]*(?:\.[0-9]+)?\)?\s*(billion|million|bn|m\b|b\b)?", re.I)


def numbers_in(text):
    out = []
    for m in NUM_RE.finditer(text or ""):
        raw = m.group(0)
        digits = re.sub(r"[^0-9.\-]", "", raw.split()[0] if raw.split() else raw)
        try:
            v = float(digits.replace(",", ""))
        except ValueError:
            continue
        unit = (m.group(1) or "").lower()
        out.append((v, unit))
    return out


def matches(pred, truth, category, tol):
    if pred is None:
        return False
    if category == "diluted_eps":
        # primary rule: exact to the reported cent; sensitivity tolerances are relative with a half-cent floor
        if tol == 0.01:
            return abs(pred - truth) <= 0.005 + 1e-9
        return abs(pred - truth) <= max(tol * abs(truth), 0.005) + 1e-9
    return abs(pred - truth) <= tol * abs(truth)


def normalise(value, unit_word, truth, category):
    """Return candidate values in the benchmark unit (millions, or $ for EPS)."""
    if value is None:
        return []
    cands = [value]
    if category != "diluted_eps":
        if unit_word in ("billion", "bn", "b"):
            cands = [value * 1000]
        elif abs(value) * 100 < abs(truth):
            cands.append(value * 1000)  # model answered in billions without saying so
        elif abs(value) > abs(truth) * 100:
            cands.append(value / 1e6)  # model answered in dollars
    return cands


def parse_closed(raw):
    try:
        j = json.loads(raw)
        if isinstance(j, dict):
            v = j.get("value")
            if isinstance(v, str):
                nums = numbers_in(v)
                v = nums[0][0] if nums else None
            return {"parsed": True, "status": str(j.get("status", "")).lower(), "value": v if isinstance(v, (int, float)) else None,
                    "answer": str(j.get("answer", ""))}
    except Exception:
        pass
    return {"parsed": False, "status": "", "value": None, "answer": raw or ""}


def score_closed(fact, rec, overrides):
    p = parse_closed(rec.get("raw"))
    truth, cat = fact["ground_truth_value"], fact["category"]
    unit_word = "billion" if re.search(r"billion|\bbn\b", p["answer"], re.I) and (p["value"] or 0) < abs(truth) / 100 else ""
    cands = normalise(p["value"], unit_word, truth, cat)
    # secondary: numbers in the answer sentence (only if no structured value)
    if p["value"] is None:
        for v, u in numbers_in(p["answer"]):
            cands += normalise(v, u, truth, cat)
    res = {}
    for name, tol in (("tol_0_5pct", 0.005), ("tol_1pct", 0.01), ("tol_2pct", 0.02)):
        res[name] = any(matches(c, truth, cat, tol) for c in cands)
    best = min(cands, key=lambda c: abs(c - truth)) if cands else None
    abstained = p["value"] is None and not any(res.values())
    status = p["status"] or ("unknown" if abstained else "")
    key = f'{fact["fact_id"]}|{rec["condition"]}'
    manual = overrides.get(key)
    recalled = res["tol_1pct"] if manual is None else bool(manual["recalled"])
    return {
        "fact_id": fact["fact_id"], "condition": rec["condition"], "company": fact["company"], "category": cat,
        "availability_window": fact["availability_window"], "filing_date": fact["filing_date"],
        "ground_truth": truth, "parsed_json": p["parsed"], "status": status, "reported_value": p["value"],
        "best_candidate_normalised": best,
        "relative_error": (abs(best - truth) / abs(truth)) if best is not None and truth else None,
        "within_0_5pct": res["tol_0_5pct"], "within_1pct": res["tol_1pct"], "within_2pct": res["tol_2pct"],
        "abstained": abstained, "recalled": recalled,
        "manual_override": manual, "answer_text": p["answer"][:400], "latency_ms": rec.get("latency_ms"),
    }


def group(rows, keyfn, flag="recalled"):
    g = defaultdict(lambda: [0, 0])
    for r in rows:
        k = keyfn(r)
        g[k][1] += 1
        g[k][0] += 1 if r[flag] else 0
    return {k: rate(v[0], v[1]) for k, v in sorted(g.items())}


def quarter(date):
    y, m = int(date[:4]), int(date[5:7])
    return f"{y}-Q{(m - 1) // 3 + 1}"


def score_grounded(fact, rec):
    truth, cat, ref = fact["ground_truth_value"], fact["category"], fact["reference_time"]
    r = rec.get("response") or {}
    err = rec.get("error")
    ev = r.get("evidence", []) or []
    cit = r.get("citations", []) or []
    def future(t):
        return bool(t) and t > ref  # ISO-8601 UTC strings compare chronologically
    ev_future = sum(1 for e in ev if future(e.get("available_time")))
    cit_future = sum(1 for c in cit if future(c.get("available_time")))
    answer = r.get("answer", "") or ""
    decision = r.get("decision")
    gen_failed = bool(re.search(r"generation failed|llm backend error|ollama", answer + " " + (r.get("reasoning_summary") or ""), re.I))
    cands = []
    for v, u in numbers_in(answer):
        cands += normalise(v, u, truth, cat)
    leaked = decision == "answer" and any(matches(c, truth, cat, 0.01) for c in cands)
    # does any temporally eligible evidence item contain the figure? (would make it supported, not leaked)
    supported_by_eligible = False
    if leaked:
        for e in ev:
            if not future(e.get("available_time")):
                for v, u in numbers_in(e.get("text", "")):
                    if any(matches(c, truth, cat, 0.01) for c in normalise(v, u, truth, cat)):
                        supported_by_eligible = True
    return {
        "fact_id": fact["fact_id"], "company": fact["company"], "category": cat,
        "availability_window": fact["availability_window"], "reference_time": ref,
        "http_error": err, "decision": decision, "generation_failed": gen_failed,
        "correct_refusal": decision == "refuse" and not gen_failed,
        "answered": decision == "answer",
        "future_fact_in_answer": leaked, "future_fact_supported_by_eligible_evidence": supported_by_eligible,
        "unsupported_future_claim": leaked and not supported_by_eligible,
        "answered_with_any_figure": decision == "answer" and bool(numbers_in(answer)),
        "evidence_items": len(ev), "evidence_future_items": ev_future,
        "citations": len(cit), "citations_future": cit_future,
        "sufficiency_reason": (r.get("sufficiency") or {}).get("reason"),
        "answer_text": answer[:400], "latency_ms": r.get("latency_ms"),
    }


def main(base):
    base = Path(base)
    bench = {f["fact_id"]: f for f in json.loads((base / "benchmark/parametric_leakage_benchmark.json").read_text())["facts"]}
    ov_path = base / "scoring/manual_overrides.json"
    overrides = json.loads(ov_path.read_text()) if ov_path.exists() else {}
    (base / "scoring").mkdir(exist_ok=True)
    (base / "metrics").mkdir(exist_ok=True)
    metrics = {"benchmark_facts": len(bench)}

    cb_path = base / "raw/closed_book.jsonl"
    if cb_path.exists():
        rows = [score_closed(bench[r["fact_id"]], r, overrides) for r in map(json.loads, cb_path.read_text().splitlines()) if r.get("fact_id") in bench and not r.get("error")]
        errors = [r for r in map(json.loads, cb_path.read_text().splitlines()) if r.get("error")]
        with open(base / "scoring/closed_book_scored.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        cb = {"request_errors": len(errors)}
        for cond in ("A1_unconstrained", "A2_as_of", "A3_forced_estimate"):
            rs = [r for r in rows if r["condition"] == cond]
            pre = [r for r in rs if r["availability_window"] == "pre_release"]
            post = [r for r in rs if r["availability_window"] == "post_release"]
            cb[cond] = {
                "valid_facts_tested": len(rs),
                "recalled_all": rate(sum(r["recalled"] for r in rs), len(rs)),
                "recalled_pre_release_window": rate(sum(r["recalled"] for r in pre), len(pre)),
                "recalled_post_release_control": rate(sum(r["recalled"] for r in post), len(post)),
                "not_recalled_all": len(rs) - sum(r["recalled"] for r in rs),
                "abstained_all": rate(sum(r["abstained"] for r in rs), len(rs)),
                "tolerance_sensitivity_pre_release": {
                    "within_0_5pct": rate(sum(r["within_0_5pct"] for r in pre), len(pre)),
                    "within_1pct": rate(sum(r["within_1pct"] for r in pre), len(pre)),
                    "within_2pct": rate(sum(r["within_2pct"] for r in pre), len(pre)),
                },
                "json_parse_failures": sum(not r["parsed_json"] for r in rs),
                "by_company_pre_release": group(pre, lambda r: r["company"]),
                "by_category_pre_release": group(pre, lambda r: r["category"]),
                "by_filing_quarter": group(rs, lambda r: quarter(r["filing_date"])),
                "median_relative_error_answered": (lambda xs: sorted(xs)[len(xs) // 2] if xs else None)([r["relative_error"] for r in rs if r["relative_error"] is not None and r["reported_value"] is not None]),
            }
        cb["definitions"] = {
            "PLR": "A2_as_of recalled / valid facts tested x 100 (leakage despite an as-of-date instruction)",
            "future_fact_recall": "A1_unconstrained recalled / valid facts tested x 100 (knowledge without date framing)",
            "forced_estimate_hit_rate": "A3_forced_estimate within tolerance / valid facts; compare pre-release vs post-release control to separate knowledge from guessing",
        }
        metrics["closed_book"] = cb

    g_path = base / "raw/grounded.jsonl"
    if g_path.exists():
        recs = [json.loads(l) for l in g_path.read_text().splitlines() if l.strip()]
        rows = [score_grounded(bench[r["fact_id"]], r) for r in recs if r.get("fact_id") in bench]
        with open(base / "scoring/grounded_scored.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        valid = [r for r in rows if not r["http_error"] and not r["generation_failed"]]
        n = len(valid)
        ev_total = sum(r["evidence_items"] for r in valid)
        cit_total = sum(r["citations"] for r in valid)
        metrics["grounded"] = {
            "queries": len(rows), "valid_queries": n,
            "http_errors": sum(bool(r["http_error"]) for r in rows),
            "generation_failures_excluded": sum(r["generation_failed"] for r in rows),
            "expected_behaviour": "refusal (no temporally eligible evidence can contain a figure first published after the reference time)",
            "correct_refusals": rate(sum(r["correct_refusal"] for r in valid), n),
            "answered_instead_of_refusing": rate(sum(r["answered"] for r in valid), n),
            "grounded_future_claim_rate": rate(sum(r["future_fact_in_answer"] for r in valid), n),
            "unsupported_future_claims": sum(r["unsupported_future_claim"] for r in valid),
            "answers_containing_any_figure": sum(r["answered_with_any_figure"] for r in valid),
            "feir_context": {"future_items": sum(r["evidence_future_items"] for r in valid), "total_items": ev_total,
                             "feir": round(sum(r["evidence_future_items"] for r in valid) / ev_total, 4) if ev_total else None},
            "feir_cited": {"future_items": sum(r["citations_future"] for r in valid), "total_items": cit_total,
                           "feir": round(sum(r["citations_future"] for r in valid) / cit_total, 4) if cit_total else None},
            "by_company": group(valid, lambda r: r["company"], "correct_refusal"),
            "by_category": group(valid, lambda r: r["category"], "correct_refusal"),
        }
    (base / "metrics/final_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: v for k, v in metrics.items()}, indent=1)[:4000])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/phase8_3")
