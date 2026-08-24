from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from src.data.graph.schema import TemporalGraph


def _parse_datetime(
    value: str | datetime | None,
) -> datetime | None:
    """Parse ISO date/datetime values safely."""

    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
        except (ValueError, TypeError):
            return None

    # Normalize naive datetimes to UTC.
    if dt.tzinfo is None:
        dt = dt.replace(
            tzinfo=timezone.utc
        )

    return dt


def extract_sec_features(
    snapshot: TemporalGraph,
    company_id: str,
    as_of: str | datetime | None = None,
) -> dict[str, Any]:
    """
    Extract point-in-time SEC V3 features.

    Temporal semantics:

        event_time
            = filing date

        available_time
            = SEC acceptance datetime

        snapshot.as_of
            = point-in-time cutoff

    SEC information is only allowed to contribute if it was
    available by the snapshot date.

    V3 includes:
    - filing counts
    - recent filing activity
    - filing velocity
    - filing acceleration
    - filing composition
    - filing recency
    """

    ticker = company_id.replace(
        "company:",
        "",
    ).upper()

    # ========================================================
    # SNAPSHOT DATE
    # ========================================================
    #
    # Priority:
    #
    #   1. Explicit as_of
    #   2. snapshot.as_of
    #
    # Never derive as_of from SEC filings.
    # ========================================================

    if as_of is not None:

        snapshot_date = _parse_datetime(
            as_of
        )

    else:

        snapshot_date = _parse_datetime(
            snapshot.as_of
        )

    # ========================================================
    # SEC FILINGS
    # ========================================================

    filings = [
        node
        for node in snapshot.nodes
        if (
            node.node_type == "sec_filing"
            and str(
                node.properties.get("ticker", "")
            ).upper() == ticker
        )
    ]

    # ========================================================
    # TEMPORAL FILING RECORDS
    # ========================================================
    #
    # event_time:
    #     filing_date
    #
    # available_time:
    #     acceptance_datetime
    #
    # The snapshot should already contain only information
    # available by snapshot.as_of.
    #
    # We nevertheless validate availability here as an
    # additional point-in-time safety check.
    # ========================================================

    filing_records: list[dict[str, Any]] = []

    for node in filings:

        filing_date = _parse_datetime(
            node.event_time
        )

        available_date = _parse_datetime(
            node.available_time
        )

        if filing_date is None:
            continue

        # ----------------------------------------------------
        # Point-in-time availability protection
        # ----------------------------------------------------

        if snapshot_date is not None:

            if available_date is None:
                continue

            if available_date > snapshot_date:
                continue

        filing_records.append(
            {
                "date": filing_date,
                "available_date": available_date,
                "form": node.properties.get(
                    "form"
                ),
            }
        )

    # ========================================================
    # SORT FILINGS NEWEST -> OLDEST
    # ========================================================

    filing_records.sort(
        key=lambda record: record["date"],
        reverse=True,
    )

    # ========================================================
    # ORIGINAL SEC FEATURES
    # ========================================================

    forms = Counter(
        record["form"]
        for record in filing_records
    )

    sec_filing_count = len(
        filing_records
    )

    sec_10k_count = forms.get(
        "10-K",
        0,
    )

    sec_10q_count = forms.get(
        "10-Q",
        0,
    )

    sec_8k_count = forms.get(
        "8-K",
        0,
    )

    sec_other_count = sum(
        count
        for form, count in forms.items()
        if form not in {
            "10-K",
            "10-Q",
            "8-K",
        }
    )

    # ========================================================
    # RECENT ACTIVITY
    # ========================================================

    sec_filings_5d = 0
    sec_filings_20d = 0
    sec_filings_60d = 0

    sec_10k_recent = 0
    sec_10q_recent = 0
    sec_8k_recent = 0

    if snapshot_date is not None:

        for record in filing_records:

            filing_date = record["date"]

            days_ago = (
                snapshot_date - filing_date
            ).days

            # A filing date after the snapshot should never
            # contribute to a historical temporal window.
            if days_ago < 0:
                continue

            form = record["form"]

            # -------------------------
            # Filing windows
            # -------------------------

            if days_ago <= 5:
                sec_filings_5d += 1

            if days_ago <= 20:
                sec_filings_20d += 1

            if days_ago <= 60:
                sec_filings_60d += 1

            # -------------------------
            # Filing composition
            # -------------------------

            if days_ago <= 60:

                if form == "10-K":
                    sec_10k_recent += 1

                elif form == "10-Q":
                    sec_10q_recent += 1

                elif form == "8-K":
                    sec_8k_recent += 1

    # ========================================================
    # FILING VELOCITY
    # ========================================================

    sec_filing_velocity = (
        sec_filings_20d / 20.0
    )

    # ========================================================
    # FILING ACCELERATION
    #
    # Compare the recent 5-day filing rate against the
    # broader 20-day filing rate.
    # ========================================================

    recent_rate = (
        sec_filings_5d / 5.0
    )

    baseline_rate = (
        sec_filings_20d / 20.0
    )

    sec_filing_acceleration = (
        recent_rate - baseline_rate
    )

    # ========================================================
    # FILING COMPOSITION
    # ========================================================

    if sec_filings_60d > 0:

        sec_8k_ratio = (
            sec_8k_recent
            / sec_filings_60d
        )

        sec_10q_ratio = (
            sec_10q_recent
            / sec_filings_60d
        )

    else:

        sec_8k_ratio = 0.0
        sec_10q_ratio = 0.0

    # ========================================================
    # FILING RECENCY
    # ========================================================

    days_since_last_filing = None
    days_since_last_10k = None
    days_since_last_10q = None
    days_since_last_8k = None

    if snapshot_date is not None:

        for record in filing_records:

            filing_date = record["date"]

            days_ago = (
                snapshot_date - filing_date
            ).days

            if days_ago < 0:
                continue

            form = record["form"]

            # -------------------------
            # Latest filing
            # -------------------------

            if days_since_last_filing is None:

                days_since_last_filing = (
                    days_ago
                )

            # -------------------------
            # Latest 10-K
            # -------------------------

            if (
                form == "10-K"
                and days_since_last_10k is None
            ):

                days_since_last_10k = (
                    days_ago
                )

            # -------------------------
            # Latest 10-Q
            # -------------------------

            if (
                form == "10-Q"
                and days_since_last_10q is None
            ):

                days_since_last_10q = (
                    days_ago
                )

            # -------------------------
            # Latest 8-K
            # -------------------------

            if (
                form == "8-K"
                and days_since_last_8k is None
            ):

                days_since_last_8k = (
                    days_ago
                )

            # Stop once all recency values exist.
            if (
                days_since_last_filing is not None
                and days_since_last_10k is not None
                and days_since_last_10q is not None
                and days_since_last_8k is not None
            ):
                break

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "company_id": company_id,

        # ====================================================
        # SEC V1
        # ====================================================

        "sec_filing_count": sec_filing_count,

        "sec_10k_count": sec_10k_count,

        "sec_10q_count": sec_10q_count,

        "sec_8k_count": sec_8k_count,

        "sec_other_count": sec_other_count,

        # ====================================================
        # SEC V3 - Recent Activity
        # ====================================================

        "sec_filings_5d": sec_filings_5d,

        "sec_filings_20d": sec_filings_20d,

        "sec_filings_60d": sec_filings_60d,

        "sec_10k_recent": sec_10k_recent,

        "sec_10q_recent": sec_10q_recent,

        "sec_8k_recent": sec_8k_recent,

        # ====================================================
        # SEC V3 - Dynamics
        # ====================================================

        "sec_filing_velocity": (
            sec_filing_velocity
        ),

        "sec_filing_acceleration": (
            sec_filing_acceleration
        ),

        # ====================================================
        # SEC V3 - Composition
        # ====================================================

        "sec_8k_ratio": sec_8k_ratio,

        "sec_10q_ratio": sec_10q_ratio,

        # ====================================================
        # SEC V3 - Recency
        # ====================================================

        "days_since_last_filing": (
            days_since_last_filing
        ),

        "days_since_last_10k": (
            days_since_last_10k
        ),

        "days_since_last_10q": (
            days_since_last_10q
        ),

        "days_since_last_8k": (
            days_since_last_8k
        ),
    }