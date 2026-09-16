from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Make the repository root importable when this script is run directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.extraction.llm import OllamaLLMClient
from src.data.extraction.pipeline import ExtractionPipeline


DEFAULT_CHUNK_DIR = Path("data/processed/chunks")
DEFAULT_OUTPUT_DIR = Path("data/processed/extractions")
DEFAULT_FAILED_LOG = Path("data/processed/extraction_failures.jsonl")


def load_chunks(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError(f"Chunk file must contain a JSON list: {path}")

    if not data:
        raise ValueError(f"Chunk file is empty: {path}")

    return data


def save_chunk_result(
    output_dir: Path,
    document_id: str,
    result: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_id = document_id.replace("/", "_")
    output_path = output_dir / f"{safe_id}.json"

    existing: list[dict[str, Any]] = []

    if output_path.exists():
        existing = json.loads(
            output_path.read_text(encoding="utf-8")
        )

        if not isinstance(existing, list):
            raise ValueError(
                f"Extraction output must contain a JSON list: {output_path}"
            )

    chunk_id = result.get("chunk_id")

    # Prevent duplicate results when resuming.
    existing = [
        item for item in existing
        if item.get("chunk_id") != chunk_id
    ]

    existing.append(result)

    existing.sort(
        key=lambda item: (
            item.get("chunk_index") is None,
            item.get("chunk_index", 0),
        )
    )

    output_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output_path


def load_completed_chunks(
    output_dir: Path,
    document_id: str,
) -> set[str]:
    safe_id = document_id.replace("/", "_")
    output_path = output_dir / f"{safe_id}.json"

    if not output_path.exists():
        return set()

    data = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    if not isinstance(data, list):
        raise ValueError(
            f"Extraction output must contain a JSON list: {output_path}"
        )

    return {
        str(item["chunk_id"])
        for item in data
        if item.get("chunk_id")
    }


def log_failure(
    failure_log: Path,
    *,
    chunk: dict[str, Any],
    error: Exception,
    attempts: int,
) -> None:
    failure_log.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "chunk_id": chunk.get("chunk_id"),
        "document_id": chunk.get("document_id"),
        "chunk_index": chunk.get("chunk_index"),
        "attempts": attempts,
        "error_type": type(error).__name__,
        "error": str(error),
    }

    with failure_log.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def extract_with_retry(
    pipeline: ExtractionPipeline,
    chunk: dict[str, Any],
    *,
    max_retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    """Extract a chunk, retrying transient/model-output failures."""
    attempts = 0

    while True:
        attempts += 1

        try:
            return pipeline.extract_chunk(chunk)

        except ValueError as exc:
            message = str(exc)

            retryable = (
                "LLM response is not valid JSON" in message
                or "OpenRouter returned an empty response" in message
            )

            if not retryable:
                raise

            if attempts > max_retries:
                raise

            delay = retry_delay * (2 ** (attempts - 1))

            print(
                f"  Retry {attempts}/{max_retries} "
                f"after {delay:.1f}s..."
            )

            time.sleep(delay)

        except (TypeError, KeyError, AssertionError):
            # Deterministic application/schema errors should not be retried.
            raise

        except Exception:
            if attempts > max_retries:
                raise

            delay = retry_delay * (2 ** (attempts - 1))

            print(
                f"  Retry {attempts}/{max_retries} "
                f"after {delay:.1f}s..."
            )

            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run resumable LLM extraction over processed "
            "financial document chunks."
        )
    )

    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=DEFAULT_CHUNK_DIR,
        help="Directory containing processed chunk JSON files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for extraction outputs.",
    )

    parser.add_argument(
        "--failure-log",
        type=Path,
        default=DEFAULT_FAILED_LOG,
        help="JSONL file for failed chunks.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of new chunks to process.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries after an extraction failure.",
    )

    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Initial retry delay in seconds.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero.")

    if args.max_retries < 0:
        raise ValueError("--max-retries cannot be negative.")

    if args.retry_delay < 0:
        raise ValueError("--retry-delay cannot be negative.")

    chunk_dir = args.chunk_dir

    if not chunk_dir.exists():
        raise FileNotFoundError(
            f"Chunk directory does not exist: {chunk_dir}"
        )

    chunk_files = sorted(chunk_dir.glob("*.json"))

    if not chunk_files:
        raise FileNotFoundError(
            f"No chunk JSON files found in {chunk_dir}"
        )

    client = OllamaLLMClient()

    pipeline = ExtractionPipeline(
        client=client,
        output_dir=args.output_dir,
    )

    batch_start_time = time.perf_counter()
    attempted = 0
    processed = 0
    skipped = 0
    failed = 0

    print("=== FinGraph AI Batch Extraction ===")
    print(f"Chunk files: {len(chunk_files)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max retries: {args.max_retries}")
    print()

    for chunk_file in chunk_files:
        chunks = load_chunks(chunk_file)

        document_id = str(chunks[0]["document_id"])

        for chunk in chunks:
            if str(chunk["document_id"]) != document_id:
                raise ValueError(
                    f"Mixed document IDs in {chunk_file}"
                )

        completed = load_completed_chunks(
            args.output_dir,
            document_id,
        )

        for chunk in chunks:
            chunk_id = str(chunk["chunk_id"])

            # Resume support.
            if chunk_id in completed:
                skipped += 1
                continue

            # Stop after requested number of NEW chunks.
            if (
                args.limit is not None
                and attempted >= args.limit
            ):
                print()
                print("=== LIMIT REACHED ===")
                print(f"Attempted: {attempted}")
                print(f"Processed: {processed}")
                print(f"Skipped:   {skipped}")
                print(f"Failed:    {failed}")
                return

            attempted += 1

            try:
                result = extract_with_retry(
                    pipeline,
                    chunk,
                    max_retries=args.max_retries,
                    retry_delay=args.retry_delay,
                )

                output_path = save_chunk_result(
                    args.output_dir,
                    document_id,
                    result,
                )

                processed += 1

                elapsed = time.perf_counter() - batch_start_time
                avg_time = elapsed / attempted if attempted else 0

                print(
                    f"[OK] {processed} | "
                    f"{chunk_id} | "
                    f"entities="
                    f"{len(result.get('entities', []))} | "
                    f"relations="
                    f"{len(result.get('relations', []))} | "
                    f"time={elapsed:.1f}s | "
                    f"avg={avg_time:.1f}s/chunk"
                )

            except Exception as exc:
                failed += 1

                log_failure(
                    args.failure_log,
                    chunk=chunk,
                    error=exc,
                    attempts=args.max_retries + 1,
                )

                print(
                    f"[FAILED] {chunk_id} | "
                    f"{type(exc).__name__}: {exc}"
                )

    total_elapsed = time.perf_counter() - batch_start_time
    avg_time = total_elapsed / attempted if attempted else 0

    print()
    print("=== BATCH COMPLETE ===")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Total time: {total_elapsed / 60:.2f} minutes")
    print(f"Average:    {avg_time:.2f} seconds/chunk")


if __name__ == "__main__":
    main()
