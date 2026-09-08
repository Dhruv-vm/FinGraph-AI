from __future__ import annotations

import json
import time
from pathlib import Path

from src.data.ingestion.sec_corpus import select_corpus
from src.data.ingestion.sec_content import download_filing
from src.data.documents.builder import build_sec_document
from src.data.chunking.structure_aware import chunk_document


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENT_DIR = PROJECT_ROOT / "data/processed/documents"
CHUNK_DIR = PROJECT_ROOT / "data/processed/chunks"
MANIFEST_PATH = PROJECT_ROOT / "data/processed/sec_corpus_manifest.json"

DELAY_SECONDS = 0.3


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {
            "completed": {},
            "failed": {},
        }

    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(manifest: dict) -> None:
    save_json(manifest, MANIFEST_PATH)


def main() -> None:
    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    corpus = select_corpus()
    manifest = load_manifest()

    total = sum(len(records) for records in corpus.values())
    processed = 0

    print("=" * 70)
    print("FinGraph AI — SEC Corpus Content Ingestion")
    print("=" * 70)
    print(f"Selected filings: {total}")
    print()

    for ticker, filings in corpus.items():
        print(f"\n[{ticker}] {len(filings)} filings")

        for record in filings:
            accession = record["accession_number"]
            key = f"{ticker}:{accession}"

            if key in manifest["completed"]:
                print(f"  SKIP {accession} — already processed")
                processed += 1
                continue

            try:
                print(
                    f"  [{processed + 1}/{total}] "
                    f"{record['form']} "
                    f"{record['filing_date']} "
                    f"{accession}"
                )

                source_url, text = download_filing(
                    record,
                    save_raw=True,
                )

                enriched_record = dict(record)
                enriched_record["source_url"] = source_url
                enriched_record["text"] = text

                document = build_sec_document(enriched_record)

                chunks = chunk_document(document)

                document_path = (
                    DOCUMENT_DIR
                    / f"{document.document_id.replace(':', '_')}.json"
                )

                chunk_path = (
                    CHUNK_DIR
                    / f"{document.document_id.replace(':', '_')}.json"
                )

                save_json(
                    document.to_dict(),
                    document_path,
                )

                save_json(
                    [chunk.to_dict() for chunk in chunks],
                    chunk_path,
                )

                manifest["completed"][key] = {
                    "document_id": document.document_id,
                    "document_path": str(
                        document_path.relative_to(PROJECT_ROOT)
                    ),
                    "chunk_path": str(
                        chunk_path.relative_to(PROJECT_ROOT)
                    ),
                    "chunk_count": len(chunks),
                    "characters": len(document.text),
                }

                save_manifest(manifest)

                print(
                    f"    OK — {len(text):,} chars, "
                    f"{len(chunks)} chunks"
                )

            except Exception as exc:
                manifest["failed"][key] = {
                    "error": str(exc),
                    "form": record.get("form"),
                    "filing_date": record.get("filing_date"),
                    "accession_number": accession,
                }

                save_manifest(manifest)

                print(f"    FAILED — {exc}")

            processed += 1

            time.sleep(DELAY_SECONDS)

    print()
    print("=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    print(f"Completed: {len(manifest['completed'])}")
    print(f"Failed:    {len(manifest['failed'])}")
    print(f"Manifest:  {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
