"""Phase 8.3 — locate diluted EPS for the reporting period (printed for manual verification)."""
import glob, json, re
NUM = r"\(?\s*-?[0-9]+\.[0-9]{2}\s*\)?"
rows = []
for f in sorted(glob.glob("data/processed/documents/*.json")):
    d = json.load(open(f))
    if d["document_type"] not in ("10-Q", "10-K"):
        continue
    t = d["text"]
    best = None
    for h in re.finditer(r"(?i)(consolidated statements? of (income|operations|earnings)|income statements|selected income statement data)", t):
        win = t[h.start(): h.start() + 12000]
        m = re.search(r"(?i)diluted(?: earnings per share| net income per (?:common )?share| net income per share)?(?: \(Note \d+\))?(?: \(a\))?\s*\$?\s*((?:" + NUM + r"\s*\$?\s*){2,6})", win)
        if m and re.search(r"(?i)per (common )?share|earnings per share", win[max(0, m.start()-600): m.start()]):
            vals = [float(x.replace("(", "-").replace(")", "").replace(" ", "")) for x in re.findall(NUM, m.group(1))]
            best = {"values": vals, "context": win[max(0, m.start()-140): m.end()].replace("\n", " ")}
            break
    rows.append({"company": d["company"], "document_type": d["document_type"], "fiscal_period": d["fiscal_period"], "document_id": d["document_id"], "eps": best})
json.dump(rows, open("results/phase8_3/benchmark/eps_candidates.json", "w"), indent=1)
for r in rows:
    e = r["eps"]
    print(f'{r["company"]:5} {r["document_type"]} {r["fiscal_period"]} ', e["values"] if e else "NOT FOUND", "|", (e["context"][-150:] if e else ""))
