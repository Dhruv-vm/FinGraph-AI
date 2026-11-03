"""Phase 8.3 — extract candidate revenue / net income facts from corpus 10-Q/10-K filings.

Read-only over data/processed/documents. Prints the matched income-statement text so every
value can be verified manually before it enters the benchmark.
"""
import glob, json, re, sys

LABELS = {
    "AAPL": ("Total net sales", "Net income"),
    "MSFT": ("Total revenue", "Net income"),
    "GOOGL": ("Revenues", "Net income"),
    "AMZN": ("Total net sales", "Net income"),
    "META": ("Revenue", "Net income"),
    "NVDA": ("Revenue", "Net income"),
    "TSLA": ("Total revenues", "Net income attributable to common stockholders"),
    "JPM": ("Total net revenue", "Net income"),
    "XOM": ("Total revenues and other income", "Net income attributable to ExxonMobil"),
    "JNJ": ("Sales to customers", "Net earnings"),
}
ANCHORS = [
    r"CONSOLIDATED STATEMENTS? OF (INCOME|OPERATIONS|EARNINGS)",
    r"Consolidated Statements? of (Income|Operations|Earnings)",
    r"INCOME STATEMENTS",
    r"STATEMENTS? OF (INCOME|OPERATIONS|EARNINGS)",
]
NUM = r"\$?\s*\(?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+(?:\.[0-9]+)?)\s*\)?"


def find_statement(text, label):
    # Use the occurrence of the label that is followed by numbers and sits after a statement heading.
    starts = []
    for a in ANCHORS:
        for m in re.finditer(a, text):
            starts.append(m.start())
    starts = sorted(set(starts))
    for s in starts:
        window = text[s : s + 6000]
        m = re.search(re.escape(label) + r"\s*" + NUM + r"\s*" + NUM, window)
        if m:
            return s, window, m
    return None, None, None


def main(out_path):
    rows = []
    for f in sorted(glob.glob("data/processed/documents/*.json")):
        d = json.load(open(f))
        if d["document_type"] not in ("10-Q", "10-K"):
            continue
        rev_label, ni_label = LABELS[d["company"]]
        for kind, label in (("revenue", rev_label), ("net_income", ni_label)):
            s, window, m = find_statement(d["text"], label)
            row = {
                "company": d["company"], "document_type": d["document_type"], "document_id": d["document_id"],
                "publication_date": d["publication_date"], "available_time": d["available_time"],
                "fiscal_period": d["fiscal_period"], "source_url": d["source_url"],
                "source_reference": d["source_reference"], "metric": kind, "label": label,
            }
            if m:
                ctx_start = max(0, m.start() - 260)
                row.update({
                    "current_value_raw": m.group(1), "comparative_value_raw": m.group(2),
                    "header_context": window[: 260].replace("\n", " "),
                    "match_context": window[ctx_start : m.end() + 40].replace("\n", " "),
                })
            rows.append(row)
    json.dump(rows, open(out_path, "w"), indent=1)
    print(len(rows), "candidate rows;", sum(1 for r in rows if "current_value_raw" in r), "matched")


if __name__ == "__main__":
    main(sys.argv[1])
