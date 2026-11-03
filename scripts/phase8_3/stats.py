"""Phase 8.3 — pre-release window vs post-release control tests on the A3 forced-estimate condition.

    python3 scripts/phase8_3/stats.py results/phase8_3

If Qwen3.5-4B had memorised pre-release figures, its forced estimates for the pre-release window should be
closer to the truth than for the post-release control (which cannot be in its training data).
Two-sided Fisher exact tests on 2x2 tables (stdlib). Writes metrics/statistical_tests.json.
"""
import json, sys
from math import comb
from pathlib import Path


def fisher_two_sided(a, b, c, d):
    """Table [[a, b], [c, d]] — rows: window; cols: hit / miss."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    def p(x):
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)
    p_obs = p(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p_obs * (1 + 1e-7)))


def main(root):
    root = Path(root)
    rows = [json.loads(l) for l in open(root / "scoring/closed_book_scored.jsonl")]
    a3 = [r for r in rows if r["condition"] == "A3_forced_estimate"]
    out = {"condition": "A3_forced_estimate", "test": "Fisher exact, two-sided", "tests": []}

    def add(name, pred, subset=lambda r: True):
        pre = [r for r in a3 if r["availability_window"] == "pre_release" and subset(r)]
        post = [r for r in a3 if r["availability_window"] == "post_release" and subset(r)]
        a, c = sum(map(pred, pre)), sum(map(pred, post))
        out["tests"].append({"name": name, "pre_release": [a, len(pre)], "post_release_control": [c, len(post)],
                             "p_value": round(fisher_two_sided(a, len(pre) - a, c, len(post) - c), 4)})

    def within(t):
        return lambda r: r["relative_error"] is not None and r["relative_error"] <= t
    add("within 1% (primary recall rule; EPS exact cent)", lambda r: r["recalled"])
    add("within 2%", lambda r: r["within_2pct"])
    add("relative error <= 5%", within(0.05))
    add("revenue, relative error <= 10%", within(0.10), lambda r: r["category"] == "revenue")
    add("net income, relative error <= 10%", within(0.10), lambda r: r["category"] == "net_income")
    add("diluted EPS, relative error <= 10%", within(0.10), lambda r: r["category"] == "diluted_eps")
    med = {}
    for w in ("pre_release", "post_release"):
        xs = sorted(r["relative_error"] for r in a3 if r["availability_window"] == w and r["reported_value"] is not None)
        med[w] = round(xs[len(xs) // 2], 4) if xs else None
    out["median_relative_error"] = med
    (root / "metrics/statistical_tests.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/phase8_3")
