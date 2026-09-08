from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.data.normalization.sec_filings import load_recent_filings
from src.data.ingestion.sec import resolve_ciks


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sec_corpus.yaml"


class SECCorpusError(RuntimeError):
    """Raised when SEC corpus configuration or selection is invalid."""


def load_corpus_config(
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise SECCorpusError(f"Corpus configuration not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise SECCorpusError("SEC corpus configuration must be a mapping.")

    return config


def select_filings(
    ticker: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    ticker = ticker.upper()

    companies = {
        str(company).upper()
        for company in config.get("companies", [])
    }

    if ticker not in companies:
        raise SECCorpusError(
            f"{ticker} is not included in the controlled SEC corpus."
        )

    forms = set(config.get("forms", []))
    limits = config.get("limits", {})

    if not forms:
        raise SECCorpusError("No SEC forms configured.")

    cik_map = resolve_ciks()

    if ticker not in cik_map:
        raise SECCorpusError(
            f"CIK not found for {ticker}."
        )

    cik = cik_map[ticker]
    records = load_recent_filings(cik=cik, ticker=ticker)

    selected: list[dict[str, Any]] = []

    for form in forms:
        form_records = [
            record
            for record in records
            if record.get("form") == form
        ]

        # Normalize newest-first ordering explicitly.
        form_records.sort(
            key=lambda record: (
                record.get("filing_date") or "",
                record.get("acceptance_datetime") or "",
            ),
            reverse=True,
        )

        limit = int(limits.get(form, 0))

        if limit <= 0:
            continue

        selected.extend(form_records[:limit])

    # Keep corpus ordering deterministic.
    selected.sort(
        key=lambda record: (
            record.get("ticker", ""),
            record.get("form", ""),
            record.get("filing_date", ""),
            record.get("accession_number", ""),
        )
    )

    return selected


def select_corpus(
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, list[dict[str, Any]]]:
    config = load_corpus_config(config_path)

    result: dict[str, list[dict[str, Any]]] = {}

    for ticker in config.get("companies", []):
        ticker = str(ticker).upper()
        result[ticker] = select_filings(ticker, config)

    return result
