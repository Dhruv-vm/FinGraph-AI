"""
Read-only SEC corpus manifest endpoint.

Reads document-level metadata produced by the ingestion pipeline. It does not
modify the corpus, chunking, or any research artefact.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from api.schemas.corpus import CompanyCoverage, CorpusFilingsResponse, FilingItem

logger = logging.getLogger("fingraph_api.routes.corpus")

router = APIRouter(prefix="/corpus", tags=["Corpus"])

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = ROOT / "data" / "processed" / "documents"
MANIFEST_PATH = ROOT / "data" / "processed" / "sec_corpus_manifest.json"

_META_FIELDS = (
    "document_id",
    "company",
    "document_type",
    "title",
    "publication_date",
    "fiscal_period",
    "available_time",
    "source_url",
    "source_reference",
)


def _clean(value):
    if value is None:
        return None
    text = str(value)
    return None if text in ("", "None", "null") else text


@lru_cache(maxsize=1)
def _load_corpus() -> CorpusFilingsResponse:
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(str(DOCUMENTS_DIR))

    chunk_counts: dict[str, int] = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for entry in (manifest.get("completed") or {}).values():
            doc_id = entry.get("document_id")
            if doc_id is not None:
                chunk_counts[doc_id] = int(entry.get("chunk_count") or 0)

    filings: list[FilingItem] = []
    for path in sorted(DOCUMENTS_DIR.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Skipping unreadable document %s: %s", path.name, exc)
            continue
        fields = {k: _clean(raw.get(k)) for k in _META_FIELDS}
        if not fields["document_id"] or not fields["company"]:
            continue
        filings.append(
            FilingItem(**fields, chunk_count=chunk_counts.get(fields["document_id"]))
        )

    filings.sort(key=lambda f: (f.publication_date or "", f.company), reverse=True)

    per_company: dict[str, list[FilingItem]] = defaultdict(list)
    for f in filings:
        per_company[f.company].append(f)

    companies: list[CompanyCoverage] = []
    for company, items in per_company.items():
        dates = sorted(d for d in (i.publication_date for i in items) if d)
        companies.append(
            CompanyCoverage(
                company=company,
                filings=len(items),
                chunks=sum(i.chunk_count or 0 for i in items),
                by_type=dict(Counter(i.document_type or "unknown" for i in items)),
                first_publication_date=dates[0] if dates else None,
                last_publication_date=dates[-1] if dates else None,
            )
        )
    companies.sort(key=lambda c: c.company)

    return CorpusFilingsResponse(
        total_filings=len(filings),
        total_chunks=sum(f.chunk_count or 0 for f in filings),
        by_type=dict(Counter(f.document_type or "unknown" for f in filings)),
        companies=companies,
        filings=filings,
    )


@router.get(
    "/filings",
    response_model=CorpusFilingsResponse,
    summary="List SEC filings in the research corpus with per-issuer coverage",
)
def get_corpus_filings() -> CorpusFilingsResponse:
    try:
        return _load_corpus()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Corpus documents directory not found: {exc}",
        ) from exc
