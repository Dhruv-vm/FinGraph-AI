"""Phase 8.3 — extract revenue and net income for the reporting period from corpus 10-Q/10-K text.

Method (read-only over data/processed/documents):
  1. Find the income-statement line item for the issuer-specific label.
  2. Read the column-year header that immediately precedes it (the last run of 4-digit years).
  3. Take the first value whose column year equals the fiscal-period year (current-period column).
Every extracted value is printed with its context and must be checked manually before use.
"""
import glob, json, re, sys

NUMTOK = r"(?:\(\s*)?-?[0-9]{1,3}(?:,[0-9]{3})+(?:\s*\))?|(?:\(\s*)?[0-9]+(?:\.[0-9]+)?(?:\s*\))?"

# label regex, and an anchor that must appear before the label (statement heading / row context)
SPEC = {
    "AAPL": {"revenue": r"Total net sales", "net_income": r"Net income"},
    "MSFT": {"revenue": r"Total revenue", "net_income": r"Net income"},
    "GOOGL": {"revenue": r"Revenues", "net_income": r"Net income"},
    "AMZN": {"revenue": r"Total net sales", "net_income": r"Net income(?: \(loss\))?"},
    "META": {"revenue": r"Revenue", "net_income": r"Net income"},
    "NVDA": {"revenue": r"Revenue", "net_income": r"Net income"},
    "TSLA": {"revenue": r"Total revenues", "net_income": r"Net income attributable to common stockholders"},
    "JPM": {"revenue": r"Total net revenue", "net_income": r"Net income"},
    "XOM": {"revenue": r"Total revenues and other income", "net_income": r"Net income(?: \(loss\))? attributable to ExxonMobil"},
    "JNJ": {"revenue": r"Sales to customers(?: \(Note \d+\))?", "net_income": r"Net earnings"},
}
HEADINGS = r"(?i)(consolidated statements? of (income|operations|earnings)|income statements|statements? of (income|operations|earnings)|selected income statement data)"


def parse_num(tok):
    neg = "(" in tok
    v = float(tok.replace("(", "").replace(")", "").replace(",", "").strip())
    return -v if neg else v


def candidates(text, label):
    for h in re.finditer(HEADINGS, text):
        win = text[h.start(): h.start() + 9000]
        m = re.search(r"(?<![A-Za-z])" + label + r"\s*(?:\$\s*)?((?:(?:" + NUMTOK + r")\s*(?:%\s*)?(?:\$\s*)?){2,8})", win)
        if not m:
            continue
        before = win[: m.start()]
        # last run of >=2 four-digit years in the header region before the label
        runs = re.findall(r"((?:\b20\d\d\b[^0-9A-Za-z]{0,40}(?:[A-Z][a-z]{2,8}\.? \d{1,2}, )?){2,})", before[-1500:])
        years = re.findall(r"20\d\d", runs[-1]) if runs else []
        nums = [parse_num(t) for t in re.findall(NUMTOK, m.group(1))]
        yield h.group(0), years, nums, win[max(0, m.start() - 160): m.end()].replace("\n", " ")


def main(out):
    facts = []
    for f in sorted(glob.glob("data/processed/documents/*.json")):
        d = json.load(open(f))
        if d["document_type"] not in ("10-Q", "10-K"):
            continue
        fy = d["fiscal_period"][:4]
        for metric, label in SPEC[d["company"]].items():
            chosen = None
            for head, years, nums, ctx in candidates(d["text"], label):
                if len(years) >= 2 and fy in years[:2] and len(nums) >= 2:
                    idx = years.index(fy) if years.index(fy) < 2 else None
                    # JNJ statements interleave a % of sales column after each value
                    vals = [n for n in nums if not (d["company"] == "JNJ" and n < 100 and n != int(n))]
                    if idx is not None and len(vals) > idx:
                        chosen = {"value": vals[idx], "other": vals[1 - idx], "years": years[:4], "heading": head, "context": ctx}
                        break
            facts.append({**{k: d[k] for k in ("company", "document_type", "document_id", "publication_date", "available_time", "fiscal_period", "source_url", "source_reference")}, "metric": metric, "label": label, "extraction": chosen})
    json.dump(facts, open(out, "w"), indent=1)
    ok = [x for x in facts if x["extraction"]]
    print(f"{len(ok)}/{len(facts)} extracted")
    for x in facts:
        e = x["extraction"]
        print(f'{x["company"]:5} {x["document_type"]} {x["fiscal_period"]} {x["metric"][:3]} ', end="")
        print(f'{e["value"]:>12,.0f}  other={e["other"]:>12,.0f} yrs={e["years"]}' if e else "  --- NOT FOUND")


if __name__ == "__main__":
    main(sys.argv[1])
