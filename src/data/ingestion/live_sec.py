"""
On-demand SEC EDGAR ingestion for the live app ("Assist" mode).

When a question cannot be answered from the indexed filings, the app can fetch the
named company's latest filings from EDGAR, chunk them with the same structure-aware
chunker, and add them to the vector index tagged ``corpus = "live"``.

Live filings are kept separate from the controlled research corpus:
  - benchmark-mode retrieval excludes them (see VectorStore.include_live);
  - documents and chunks are written to data/processed/live/, not data/processed/{documents,chunks};
  - they are not added to the knowledge graph (no LLM extraction), so they are reached by
    passage retrieval only.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.chunking.structure_aware import chunk_document
from src.data.ingestion.sec import SEC_SUBMISSIONS_URL, SEC_TICKER_URL, fetch_json
from src.data.ingestion.sec_content import build_sec_document_with_content
from src.data.normalization.sec_filings import normalize_filing

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TICKER_CACHE = PROJECT_ROOT / "data" / "raw" / "sec" / "company_tickers_exchange.json"
LIVE_DIR = PROJECT_ROOT / "data" / "processed" / "live"
LIVE_MANIFEST = LIVE_DIR / "live_manifest.json"
CORPUS_DOCS_DIR = PROJECT_ROOT / "data" / "processed" / "documents"

# Latest filings fetched per request (kept small so a fetch takes well under a minute).
# DEF 14A (proxy statement) carries ownership tables ("Security Ownership of Certain Beneficial
# Owners and Management"), executive pay and board details, which 10-K/10-Q/8-K do not.
DEFAULT_FORMS: dict[str, int] = {"10-K": 1, "10-Q": 2, "8-K": 2, "DEF 14A": 1}

_NAME_SUFFIX = re.compile(
    r"[,.]?\s*\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|holdings?|group|n\.?v|s\.?a|ag|llc|lp|the)\b\.?",
    re.I,
)
# Upper-case words that are also tickers but almost never mean the company in a question.
_TICKER_STOP = {
    "A", "I", "AI", "IT", "ON", "OR", "AND", "ARE", "ALL", "ANY", "CAN", "FOR", "HAS", "NOW", "ONE", "SEC",
    "CEO", "CFO", "USA", "US", "EU", "UK", "Q1", "Q2", "Q3", "Q4", "FY", "EPS", "GDP", "ESG", "IPO", "ETF",
    "AM", "PM", "TV", "GO", "SO", "BE", "DO", "NEW", "BIG", "WELL", "LOW", "HIGH", "REAL", "TRUE", "KEY",
}


def _simplify(name: str) -> str:
    n = name.lower().replace("&", " and ").replace(".com", "")
    n = _NAME_SUFFIX.sub(" ", n)
    n = re.sub(r"[^a-z0-9 ]+", " ", n)
    return re.sub(r"\s+", " ", n).strip()


_index: dict[str, Any] | None = None


def _load_tickers() -> dict[str, Any]:
    """SEC's company/ticker list, cached on disk after the first download."""
    global _index
    if _index is not None:
        return _index
    if TICKER_CACHE.exists():
        data = json.loads(TICKER_CACHE.read_text(encoding="utf-8"))
    else:
        data = fetch_json(SEC_TICKER_URL)
        TICKER_CACHE.parent.mkdir(parents=True, exist_ok=True)
        TICKER_CACHE.write_text(json.dumps(data), encoding="utf-8")
    fields = data.get("fields", [])
    rows = [dict(zip(fields, r)) for r in data.get("data", [])]
    by_ticker: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for r in rows:
        t = str(r.get("ticker") or "").upper()
        if not t or t in by_ticker:
            continue
        rec = {"ticker": t, "cik": str(r.get("cik")).zfill(10), "name": r.get("name"), "exchange": r.get("exchange")}
        by_ticker[t] = rec
        key = _simplify(str(r.get("name") or ""))
        # First listing wins (common stock is listed before other share classes).
        if key and key not in by_name:
            by_name[key] = rec
    _index = {"by_ticker": by_ticker, "by_name": by_name}
    return _index


def resolve_company(query: str, allow_network: bool = True) -> dict[str, Any] | None:
    """Find the SEC registrant a question refers to, by company name or explicit ticker."""
    if not allow_network and not TICKER_CACHE.exists() and _index is None:
        return None
    try:
        idx = _load_tickers()
    except Exception:
        return None
    words = re.findall(r"[a-z0-9&]+", query.lower().replace("'s", ""))
    # Longest name n-gram first ("johnson and johnson" before "johnson").
    for n in (4, 3, 2, 1):
        for i in range(len(words) - n + 1):
            key = _simplify(" ".join(words[i : i + n]))
            if len(key) >= 3 and key in idx["by_name"]:
                return idx["by_name"][key]
    for tok in re.findall(r"\b[A-Z]{2,5}\b", query):
        if tok not in _TICKER_STOP and tok in idx["by_ticker"]:
            return idx["by_ticker"][tok]
    return None


def _load_manifest() -> dict[str, Any]:
    if LIVE_MANIFEST.exists():
        return json.loads(LIVE_MANIFEST.read_text(encoding="utf-8"))
    return {"filings": []}


def live_companies() -> dict[str, str]:
    """ticker -> company name for every company with live filings."""
    out: dict[str, str] = {}
    for f in _load_manifest().get("filings", []):
        out[f["ticker"]] = f.get("company_name") or f["ticker"]
    return out


def _known_accessions() -> set[str]:
    known = {f["accession_number"] for f in _load_manifest().get("filings", [])}
    if CORPUS_DOCS_DIR.exists():
        for p in CORPUS_DOCS_DIR.glob("*.json"):
            try:
                ref = json.loads(p.read_text(encoding="utf-8")).get("source_reference")
            except Exception:
                continue
            if ref:
                known.add(ref)
    return known


def fetch_live_filings(
    company: dict[str, Any],
    vector_store: Any,
    forms: dict[str, int] | None = None,
    delay_seconds: float = 0.15,
) -> dict[str, Any]:
    """Download the company's latest filings that are not indexed yet and add them to the index."""
    forms = forms or DEFAULT_FORMS
    ticker, cik = company["ticker"], company["cik"]
    submissions = fetch_json(SEC_SUBMISSIONS_URL.format(cik=cik))
    recent = submissions.get("filings", {}).get("recent", {})
    records: list[dict[str, Any]] = []
    for i, form in enumerate(recent.get("form", [])):
        if form not in forms:
            continue
        raw = {k: v[i] for k, v in recent.items() if i < len(v)}
        records.append(normalize_filing(ticker=ticker, cik=cik, record=raw))

    known = _known_accessions()
    selected: list[dict[str, Any]] = []
    skipped_known = 0
    for form, limit in forms.items():
        rows = sorted(
            (r for r in records if r.get("form") == form),
            key=lambda r: (r.get("filing_date") or "", r.get("acceptance_datetime") or ""),
            reverse=True,
        )[:limit]
        for r in rows:
            if r.get("accession_number") in known:
                skipped_known += 1
            else:
                selected.append(r)

    manifest = _load_manifest()
    added: list[dict[str, Any]] = []
    errors: list[str] = []
    now = datetime.now(timezone.utc).isoformat()
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    for n, rec in enumerate(selected):
        try:
            rec = dict(rec, ingested_at=now)
            doc = build_sec_document_with_content(rec, save_raw=False)
            chunks = [c.to_dict() for c in chunk_document(doc)]
            for c in chunks:
                c["corpus"] = "live"
            indexed = vector_store.upsert_chunks(chunks) if chunks else 0
            safe = doc.document_id.replace(":", "_")
            (LIVE_DIR / f"{safe}.json").write_text(
                json.dumps({"document": doc.to_dict(), "chunks": chunks}, ensure_ascii=False), encoding="utf-8"
            )
            entry = {
                "ticker": ticker,
                "company_name": company.get("name"),
                "cik": cik,
                "form": doc.document_type,
                "filing_date": doc.publication_date,
                "fiscal_period": doc.fiscal_period,
                "accession_number": doc.source_reference,
                "document_id": doc.document_id,
                "source_url": doc.source_url,
                "chunks": indexed,
                "fetched_at": now,
            }
            manifest["filings"].append(entry)
            added.append(entry)
        except Exception as exc:  # keep going: one bad filing should not block the rest
            errors.append(f'{rec.get("form")} {rec.get("accession_number")}: {exc}')
        if n < len(selected) - 1:
            time.sleep(delay_seconds)
    LIVE_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"company": company, "added": added, "already_indexed": skipped_known, "errors": errors}
