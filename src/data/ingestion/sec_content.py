"""
SEC filing-content ingestion.

Builds canonical SEC filing URLs from normalized filing metadata,
downloads filing HTML, and extracts readable text while preserving
the original filing metadata for downstream temporal/provenance use.
"""

from __future__ import annotations

import html
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from src.data.documents.builder import build_sec_document
from src.data.documents.schema import Document


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_SEC_CONTENT_DIR = PROJECT_ROOT / "data" / "raw" / "sec" / "filings"

SEC_ARCHIVES_URL = (
    "https://www.sec.gov/Archives/edgar/data"
)


class SECContentError(Exception):
    """Raised when SEC filing-content ingestion fails."""


def get_user_agent() -> str:
    """Return the configured SEC User-Agent."""

    load_dotenv(PROJECT_ROOT / ".env")

    import os

    user_agent = os.getenv("SEC_USER_AGENT")

    if not user_agent:
        raise SECContentError(
            "SEC_USER_AGENT is missing. "
            "Add it to .env before downloading filings."
        )

    return user_agent


def build_filing_url(
    cik: str,
    accession_number: str,
    primary_document: str,
) -> str:
    """
    Build the canonical SEC Archives URL for a filing.

    SEC archive paths use the numeric CIK without leading zeros and
    the accession number without hyphens.
    """

    if not cik:
        raise SECContentError("CIK is required.")

    if not accession_number:
        raise SECContentError("Accession number is required.")

    if not primary_document:
        raise SECContentError("Primary document is required.")

    cik_numeric = str(int(cik))
    accession_path = accession_number.replace("-", "")

    return (
        f"{SEC_ARCHIVES_URL}/"
        f"{cik_numeric}/"
        f"{accession_path}/"
        f"{primary_document}"
    )


def fetch_html(url: str) -> str:
    """Download an SEC filing HTML document."""

    request = Request(
        url,
        headers={
            "User-Agent": get_user_agent(),
            "Accept": "text/html,application/xhtml+xml",
        },
    )

    try:
        with urlopen(request, timeout=60) as response:
            status = response.status
            body = response.read()
            content_type = response.headers.get("Content-Type", "")
    except Exception as exc:
        raise SECContentError(
            f"SEC filing request failed: {url}\n"
            f"Reason: {exc}"
        ) from exc

    if status != 200:
        raise SECContentError(
            f"SEC filing returned HTTP {status}: {url}"
        )

    if not body:
        raise SECContentError(
            f"SEC filing returned an empty response: {url}"
        )

    if "html" not in content_type.lower():
        # Some SEC responses may omit a useful content type.
        decoded = body.decode("utf-8", errors="replace")

        if "<html" not in decoded.lower():
            raise SECContentError(
                "SEC response does not appear to contain HTML: "
                f"{url}"
            )

        return decoded

    return body.decode("utf-8", errors="replace")

def html_to_text(document_html: str) -> str:
    """Convert SEC filing HTML into readable visible filing text.

    Removes scripts, styles, hidden XBRL metadata, and other non-visible
    content while preserving the visible filing structure.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(document_html, "html.parser")

    # Remove non-content elements.
    for tag in soup(["script", "style", "noscript", "svg", "head"]):
        tag.decompose()

    # Remove Inline XBRL header/resources.
    for tag in soup.find_all(
        lambda tag: tag.name and tag.name.lower() in {
            "ix:header",
            "ix:hidden",
            "ix:references",
            "ix:resources",
        }
    ):
        tag.decompose()

    # Remove elements explicitly hidden through HTML/CSS.
    for tag in soup.find_all(True):
        attrs = tag.attrs or {}

        if "hidden" in attrs:
            tag.decompose()
            continue

        if attrs.get("aria-hidden") == "true":
            tag.decompose()
            continue

        style = attrs.get("style", "")
        style_normalized = style.replace(" ", "").lower()

        if (
            "display:none" in style_normalized
            or "visibility:hidden" in style_normalized
        ):
            tag.decompose()

    # Add line breaks around common block elements.
    for tag in soup.find_all(
        [
            "p",
            "div",
            "section",
            "article",
            "header",
            "footer",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "li",
            "tr",
            "table",
            "br",
        ]
    ):
        tag.insert_before("\n")
        tag.insert_after("\n")

    text = soup.get_text(" ", strip=True)

    # Normalize Unicode whitespace.
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse horizontal whitespace.
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)

    return text.strip()

def download_filing(
    record: dict[str, object],
    save_raw: bool = True,
) -> tuple[str, str]:
    """
    Download one filing.

    Returns:
        (filing_url, extracted_text)
    """

    cik = str(record.get("cik") or "")
    accession = str(record.get("accession_number") or "")
    primary_document = str(record.get("primary_document") or "")

    url = build_filing_url(
        cik=cik,
        accession_number=accession,
        primary_document=primary_document,
    )

    document_html = fetch_html(url)

    if save_raw:
        RAW_SEC_CONTENT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        ticker = str(record.get("ticker") or "unknown").lower()
        safe_accession = accession.replace("-", "")

        output_path = (
            RAW_SEC_CONTENT_DIR
            / f"{ticker}_{safe_accession}_{primary_document}"
        )

        output_path.write_text(
            document_html,
            encoding="utf-8",
        )

    text = html_to_text(document_html)

    if not text:
        raise SECContentError(
            f"SEC filing produced no readable text: {url}"
        )

    return url, text


def build_sec_document_with_content(
    record: dict[str, object],
    save_raw: bool = True,
) -> Document:
    """
    Download filing content and build the canonical Document.

    The original normalized metadata is retained while adding the
    actual filing text and canonical SEC source URL.
    """

    url, text = download_filing(
        record=record,
        save_raw=save_raw,
    )

    enriched_record = dict(record)
    enriched_record["source_url"] = url
    enriched_record["text"] = text

    document = build_sec_document(enriched_record)

    document.metadata["content_ingested"] = True
    document.metadata["content_format"] = "html"
    document.metadata["content_characters"] = len(text)

    return document


def ingest_filings(
    records: list[dict[str, object]],
    delay_seconds: float = 0.2,
    limit: int | None = None,
) -> list[Document]:
    """
    Download a controlled set of SEC filings.

    `limit` is useful during development so the pipeline can be
    validated on a small sample before scaling to the full corpus.
    """

    selected_records = records if limit is None else records[:limit]

    documents: list[Document] = []

    for index, record in enumerate(selected_records, start=1):
        ticker = record.get("ticker", "UNKNOWN")
        accession = record.get("accession_number", "UNKNOWN")

        print(
            f"[{index}/{len(selected_records)}] "
            f"Downloading {ticker} {accession}..."
        )

        document = build_sec_document_with_content(record)

        documents.append(document)

        if index < len(selected_records):
            time.sleep(delay_seconds)

    return documents
