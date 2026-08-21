"""
SEC EDGAR ingestion adapter.

Retrieves public company submission metadata from data.sec.gov
and stores the raw responses for reproducible downstream processing.
"""

from __future__ import annotations
import gzip
import csv
import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "company_universe.csv"
RAW_SEC_PATH = PROJECT_ROOT / "data" / "raw" / "sec"

SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


class SECIngestionError(Exception):
    """Raised when SEC ingestion fails."""


def get_user_agent() -> str:
    """Return the configured SEC User-Agent."""

    load_dotenv(PROJECT_ROOT / ".env")

    user_agent = os.getenv("SEC_USER_AGENT")

    if not user_agent:
        raise SECIngestionError(
            "SEC_USER_AGENT is missing. "
            "Add it to .env before running ingestion."
        )

    return user_agent

def fetch_json(url: str) -> dict:
    """Fetch JSON from a SEC endpoint with defensive validation."""

    user_agent = get_user_agent()

    request = Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Host": url.split("/")[2],
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            status = response.status
            content_type = response.headers.get("Content-Type", "")
            content_encoding = response.headers.get("Content-Encoding", "")
            body = response.read()

    except Exception as exc:
        raise SECIngestionError(
            f"SEC request failed: {url}\n"
            f"Reason: {exc}"
        ) from exc

    if status != 200:
        preview = body[:500].decode("utf-8", errors="replace")

        raise SECIngestionError(
            f"SEC returned HTTP {status}.\n"
            f"Content-Type: {content_type}\n"
            f"Response preview:\n{preview}"
        )

    # SEC may return gzip-compressed JSON.
    if "gzip" in content_encoding.lower():
        try:
            body = gzip.decompress(body)
        except OSError as exc:
            raise SECIngestionError(
                "SEC response was marked as gzip but could not be decompressed."
            ) from exc

    if "json" not in content_type.lower():
        preview = body[:500].decode("utf-8", errors="replace")

        raise SECIngestionError(
            "SEC returned a non-JSON response.\n"
            f"Content-Type: {content_type}\n"
            f"Response preview:\n{preview}"
        )

    try:
        return json.loads(body)

    except json.JSONDecodeError as exc:
        preview = body[:500].decode("utf-8", errors="replace")

        raise SECIngestionError(
            "SEC response could not be decoded as JSON.\n"
            f"Response preview:\n{preview}"
        ) from exc

def load_company_universe() -> list[dict[str, str]]:
    """Load the controlled company universe."""

    if not CONFIG_PATH.exists():
        raise SECIngestionError(
            f"Company universe not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def resolve_ciks() -> dict[str, str]:
    """
    Load ticker -> CIK mappings from the controlled company universe.

    The CIK values are version-controlled as part of the research
    configuration so ingestion does not repeatedly query the SEC
    ticker mapping endpoint.
    """

    universe = load_company_universe()

    resolved: dict[str, str] = {}

    for company in universe:
        ticker = company["ticker"].strip().upper()
        cik = company["cik"].strip().zfill(10)

        if not ticker:
            raise SECIngestionError(
                "Company universe contains an empty ticker."
            )

        if not cik.isdigit() or len(cik) != 10:
            raise SECIngestionError(
                f"Invalid CIK for {ticker}: {cik}"
            )

        resolved[ticker] = cik

    if len(resolved) != len(universe):
        raise SECIngestionError(
            "Duplicate ticker detected in company universe."
        )

    return resolved

def save_json(data: dict, path: Path) -> None:
    """Save JSON using deterministic formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def ingest_company(ticker: str, cik: str) -> Path:
    """Download and store SEC submission history for one company."""

    url = SEC_SUBMISSIONS_URL.format(cik=cik)

    data = fetch_json(url)

    output_path = RAW_SEC_PATH / f"{ticker.lower()}_submissions.json"

    save_json(data, output_path)

    return output_path


def run_ingestion() -> None:
    """Run SEC ingestion for the complete controlled universe."""

    print("Resolving SEC CIK identifiers...")

    cik_map = resolve_ciks()

    print(f"Resolved {len(cik_map)} companies.")

    for index, (ticker, cik) in enumerate(
        cik_map.items(),
        start=1,
    ):
        print(
            f"[{index}/{len(cik_map)}] "
            f"Fetching {ticker} (CIK {cik})..."
        )

        output_path = ingest_company(ticker, cik)

        print(f"  Saved: {output_path}")

        # Keep requests conservative.
        time.sleep(0.2)

    print("\nSEC ingestion complete.")


if __name__ == "__main__":
    run_ingestion()