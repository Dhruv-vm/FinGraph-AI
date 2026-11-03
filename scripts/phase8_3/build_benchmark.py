"""Phase 8.3 — build the held-out parametric leakage benchmark (PLB-8.3).

Every ground-truth value below was extracted from the issuer's own 10-Q / 10-K text in
data/processed/documents and verified manually (see docs/PHASE_8_3_PARAMETRIC_LEAKAGE_EVALUATION.md §3).
This script re-verifies each value against the filing text (the formatted figure must occur in the
same filing) and writes results/phase8_3/benchmark/parametric_leakage_benchmark.json.

Temporal split: reference_time = 00:00 UTC on the fiscal period end date. A period's results cannot be
public before the period ends, so every fact's public availability (earnings release or filing) is
strictly later than its reference time.
"""
import glob, json, re
from datetime import datetime

MODEL_RELEASE_DATE = "2026-03-02"  # Qwen3.5-4B public release (OpenCode model registry), used to stratify

NAMES = {"AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon", "META": "Meta Platforms",
         "NVDA": "NVIDIA", "TSLA": "Tesla", "JPM": "JPMorgan Chase", "XOM": "Exxon Mobil", "JNJ": "Johnson & Johnson"}
REV_LABEL = {"AAPL": "total net sales", "MSFT": "total revenue", "GOOGL": "total revenues", "AMZN": "total net sales",
             "META": "total revenue", "NVDA": "total revenue", "TSLA": "total revenues", "JPM": "total net revenue",
             "XOM": "total revenues and other income", "JNJ": "sales to customers (total revenue)"}
NI_LABEL = {"TSLA": "net income attributable to common stockholders", "XOM": "net income attributable to ExxonMobil",
            "JNJ": "net earnings"}

# (company, fiscal_period_end, form): (revenue $M, net income $M, diluted EPS $ or None)
VERIFIED = {
    ("MSFT", "2025-03-31", "10-Q"): (70066, 25824, 3.46),
    ("TSLA", "2025-06-30", "10-Q"): (22496, 1172, 0.33),
    ("JNJ", "2025-06-29", "10-Q"): (23743, 5537, 2.29),
    ("GOOGL", "2025-06-30", "10-Q"): (96428, 28196, 2.31),
    ("META", "2025-06-30", "10-Q"): (47516, 18337, 7.14),
    ("AMZN", "2025-06-30", "10-Q"): (167702, 18164, 1.68),
    ("AAPL", "2025-06-28", "10-Q"): (94036, 23434, 1.57),
    ("XOM", "2025-06-30", "10-Q"): (81506, 7082, None),
    ("NVDA", "2025-07-27", "10-Q"): (46743, 26422, 1.08),
    ("JNJ", "2025-09-28", "10-Q"): (23993, 5152, 2.12),
    ("TSLA", "2025-09-30", "10-Q"): (28095, 1373, 0.39),
    ("MSFT", "2025-09-30", "10-Q"): (77673, 27747, 3.72),
    ("GOOGL", "2025-09-30", "10-Q"): (102346, 34979, 2.87),
    ("META", "2025-09-30", "10-Q"): (51242, 2709, 1.05),
    ("AMZN", "2025-09-30", "10-Q"): (180169, 21187, 1.95),
    ("AAPL", "2025-09-27", "10-K"): (416161, 112010, 7.46),
    ("XOM", "2025-09-30", "10-Q"): (85294, 7548, None),
    ("JPM", "2025-09-30", "10-Q"): (46427, 14393, 5.07),
    ("NVDA", "2025-10-26", "10-Q"): (57006, 31910, 1.30),
    ("MSFT", "2025-12-31", "10-Q"): (81273, 38458, 5.16),
    ("META", "2025-12-31", "10-K"): (200966, 60458, 23.49),
    ("TSLA", "2025-12-31", "10-K"): (94827, 3794, 1.08),
    ("AAPL", "2025-12-27", "10-Q"): (143756, 42097, 2.84),
    ("GOOGL", "2025-12-31", "10-K"): (402836, 132170, 10.81),
    ("AMZN", "2025-12-31", "10-K"): (716924, 77670, 7.17),
    ("JNJ", "2025-12-28", "10-K"): (94193, 26804, 11.03),
    ("JPM", "2025-12-31", "10-K"): (182447, 57048, 20.02),
    ("XOM", "2025-12-31", "10-K"): (332238, 28844, None),
    ("NVDA", "2026-01-25", "10-K"): (215938, 120067, 4.90),
    # ---- control window: public after the model's release ----
    ("JNJ", "2026-03-29", "10-Q"): (24062, 5235, 2.14),
    ("TSLA", "2026-03-31", "10-Q"): (22387, 477, 0.13),
    ("MSFT", "2026-03-31", "10-Q"): (82886, 31778, 4.27),
    ("AMZN", "2026-03-31", "10-Q"): (181519, 30255, 2.78),
    ("META", "2026-03-31", "10-Q"): (56311, 26773, 10.44),
    ("GOOGL", "2026-03-31", "10-Q"): (109896, 62578, 5.11),
    ("AAPL", "2026-03-28", "10-Q"): (111184, 29578, 2.01),
    ("JPM", "2026-03-31", "10-Q"): (49836, 16494, 5.94),
    ("XOM", "2026-03-31", "10-Q"): (85138, 4183, None),
    ("NVDA", "2026-04-26", "10-Q"): (81615, 58321, 2.39),
    ("GOOGL", "2026-06-30", "10-Q"): (119796, 112193, 9.11),
    ("TSLA", "2026-06-30", "10-Q"): (28236, 1114, 0.32),
    ("JNJ", "2026-06-28", "10-Q"): (25310, 5534, 2.27),
    ("MSFT", "2026-06-30", "10-K"): (331839, 133749, 17.95),
    ("META", "2026-06-30", "10-Q"): (60801, 15848, 6.18),
    ("AAPL", "2026-06-27", "10-Q"): (109417, 29789, 2.02),
    ("AMZN", "2026-06-30", "10-Q"): (200606, 62647, 5.75),
    ("XOM", "2026-06-30", "10-Q"): (116017, 14525, None),
    ("JPM", "2026-06-30", "10-Q"): (57347, 21155, 7.70),
    ("NVDA", "2026-07-26", "10-Q"): (96221, 59688, 2.46),
}


def fmt_m(v):
    return f"{v:,}"


def period_phrase(form, end):
    d = datetime.strptime(end, "%Y-%m-%d").strftime("%B %-d, %Y")
    return (f"fiscal year ended {d}" if form == "10-K" else f"fiscal quarter (three months) ended {d}"), d


def snippet(text, needle, pad=140):
    i = text.find(needle)
    return text[max(0, i - pad): i + len(needle) + 40].replace("\n", " ") if i >= 0 else None


def main():
    docs = {}
    for f in glob.glob("data/processed/documents/*.json"):
        d = json.load(open(f))
        if d["document_type"] in ("10-Q", "10-K"):
            docs[(d["company"], d["fiscal_period"], d["document_type"])] = d
    assert set(VERIFIED) == set(docs), "benchmark must cover exactly the corpus periodic filings"

    facts, problems = [], []
    for key in sorted(VERIFIED, key=lambda k: (docs[k]["publication_date"], k[0])):
        co, end, form = key
        d = docs[key]
        rev, ni, eps = VERIFIED[key]
        phrase, end_h = period_phrase(form, end)
        window = "pre_release" if d["publication_date"] < MODEL_RELEASE_DATE else "post_release"
        for metric, value, unit, label, needle in (
            ("revenue", rev, "USD_millions", REV_LABEL[co], fmt_m(rev)),
            ("net_income", ni, "USD_millions", NI_LABEL.get(co, "net income"), fmt_m(ni)),
            ("diluted_eps", eps, "USD_per_share", "diluted earnings per share", f"{eps:.2f}" if eps is not None else None),
        ):
            if value is None:
                continue
            ctx = snippet(d["text"], needle)
            if ctx is None:
                problems.append((key, metric, needle))
            unit_txt = "in millions of US dollars" if unit == "USD_millions" else "in US dollars per share"
            facts.append({
                "fact_id": f"PLB-{co}-{end}-{metric}",
                "company": co, "company_name": NAMES[co], "category": metric, "metric_label": label,
                "form": form, "fiscal_period_end": end, "period_phrase": phrase,
                "ground_truth_value": value, "unit": unit,
                "reference_time": f"{end}T00:00:00+00:00",
                "public_availability_time": d["available_time"], "filing_date": d["publication_date"],
                "availability_window": window,
                "question": f"What was {NAMES[co]}'s {label} for the {phrase}, as reported in its SEC filing? Give the figure {unit_txt}.",
                "source": {"document_id": d["document_id"], "accession": d["source_reference"], "url": d["source_url"],
                           "form": form, "verbatim_context": ctx},
            })
    assert not problems, problems
    # temporal split sanity
    for f in facts:
        assert f["public_availability_time"] > f["reference_time"], f["fact_id"]
    ids = [f["fact_id"] for f in facts]
    assert len(ids) == len(set(ids))
    out = {
        "benchmark": "FinGraph PLB-8.3 parametric leakage benchmark",
        "created": "2026-11-03",
        "model_release_date_used_for_stratification": MODEL_RELEASE_DATE,
        "temporal_split": "reference_time = fiscal period end 00:00 UTC; fact public only after period end",
        "n_facts": len(facts),
        "n_pre_release": sum(f["availability_window"] == "pre_release" for f in facts),
        "n_post_release": sum(f["availability_window"] == "post_release" for f in facts),
        "facts": facts,
    }
    json.dump(out, open("results/phase8_3/benchmark/parametric_leakage_benchmark.json", "w"), indent=1)
    print(out["n_facts"], "facts;", out["n_pre_release"], "pre-release;", out["n_post_release"], "post-release")


if __name__ == "__main__":
    main()
