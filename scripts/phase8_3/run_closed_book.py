"""Phase 8.3 Experiment A — closed-book (retrieval-disabled) runner, Python reference implementation.

Equivalent to run_in_browser.js (which was used for the recorded run because Ollama listens only on the
author's machine). Requires the project venv (`ollama` package) and a running Ollama with qwen3.5:4b.

    python scripts/phase8_3/run_closed_book.py results/phase8_3/raw/closed_book_rerun.jsonl
"""
import json, sys, time
from pathlib import Path
import ollama

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((Path(__file__).parent / "config.json").read_text())
BENCH = json.loads((ROOT / "results/phase8_3/benchmark/parametric_leakage_benchmark.json").read_text())


def main(out_path):
    client = ollama.Client(host=CFG["ollama_host"])
    with open(out_path, "w") as out:
        for fact in BENCH["facts"]:
            for cond in CFG["condition_order"]:
                system = CFG["system_prompts"][cond].replace("{reference_date}", fact["reference_time"][:10])
                t0 = time.time()
                res = client.chat(
                    model=CFG["model"],
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": fact["question"]}],
                    format=CFG["format"], options=CFG["decoding"], think=CFG["think"],
                )
                out.write(json.dumps({"fact_id": fact["fact_id"], "condition": cond, "system": system,
                                      "user": fact["question"], "raw": res.message.content,
                                      "latency_ms": round((time.time() - t0) * 1000)}) + "\n")
                out.flush()


if __name__ == "__main__":
    main(sys.argv[1])
