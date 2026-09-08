from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from src.data.graph.schema import TemporalGraph


# ============================================================
# DATETIME HELPERS
# ============================================================

def _parse_datetime(
    value: str | datetime | None,
) -> datetime | None:
    """
    Parse ISO date/datetime values safely.

    Naive datetimes are normalized to UTC.
    """

    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(
                str(value).replace("Z", "+00:00")
            )
        except (ValueError, TypeError):
            return None

    if dt.tzinfo is None:
        dt = dt.replace(
            tzinfo=timezone.utc
        )

    return dt


# ============================================================
# SEC FEATURE EXTRACTION
# ============================================================

def extract_sec_features(
    snapshot: TemporalGraph,
    company_id: str,
    as_of: str | datetime | None = None,
) -> dict[str, Any]:
    """
    Extract point-in-time SEC features.

    Temporal semantics
    ------------------
    event_time:
        Filing event date.

    available_time:
        SEC acceptance/publication datetime.

    snapshot.as_of:
        Point-in-time cutoff.

    Only filings available by the snapshot cutoff are allowed
    to contribute to the feature vector.

    Feature groups
    --------------
    1. Filing counts
    2. Recent filing activity
    3. Filing velocity
    4. Filing acceleration
    5. Filing composition
    6. Filing recency

    Notes
    -----
    Cumulative filing counts are retained for compatibility
    with the existing dataset, but rolling-window features are
    preferred for downstream predictive modeling because
    cumulative counts naturally increase over time.
    """

    # ========================================================
    # COMPANY / TICKER
    # ========================================================

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
    # Never infer the cutoff from SEC filings.
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
    # FIND SEC FILINGS
    # ========================================================

    filings = [
        node
        for node in snapshot.nodes
        if (
            node.node_type == "sec_filing"
            and str(
                node.properties.get(
                    "ticker",
                    "",
                )
            ).upper() == ticker
        )
    ]

    # ========================================================
    # BUILD TEMPORAL FILING RECORDS
    # ========================================================

    filing_records: list[
        dict[str, Any]
    ] = []

    for node in filings:

        filing_date = _parse_datetime(
            node.event_time
        )

        available_date = _parse_datetime(
            node.available_time
        )

        # Cannot safely use a filing without an event date.
        if filing_date is None:
            continue

        # ----------------------------------------------------
        # Point-in-time availability protection
        # ----------------------------------------------------

        if snapshot_date is not None:

            # If availability is unknown, reject the record
            # rather than risking temporal leakage.
            if available_date is None:
                continue

            # Information published after the prediction
            # snapshot cannot contribute.
            if available_date > snapshot_date:
                continue

            # Event dates after the snapshot cannot contribute.
            if filing_date > snapshot_date:
                continue

        form = str(
            node.properties.get(
                "form",
                "",
            )
        ).upper()

        filing_records.append(
            {
                "date": filing_date,
                "available_date": available_date,
                "form": form,
            }
        )

    # ========================================================
    # SORT NEWEST -> OLDEST
    # ========================================================

    filing_records.sort(
        key=lambda record: record["date"],
        reverse=True,
    )

    # ========================================================
    # ORIGINAL / CUMULATIVE SEC FEATURES
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
    # ROLLING SEC ACTIVITY
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

            # Future event protection.
            if days_ago < 0:
                continue

            form = record["form"]

            # ------------------------------------------------
            # 5-day activity
            # ------------------------------------------------

            if days_ago <= 5:
                sec_filings_5d += 1

            # ------------------------------------------------
            # 20-day activity
            # ------------------------------------------------

            if days_ago <= 20:
                sec_filings_20d += 1

            # ------------------------------------------------
            # 60-day activity
            # ------------------------------------------------

            if days_ago <= 60:
                sec_filings_60d += 1

            # ------------------------------------------------
            # Recent filing composition
            #
            # Composition is measured over the same 60-day
            # window as the corresponding activity count.
            # ------------------------------------------------

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
    #
    # Average filings per day over the 20-day window.
    # ========================================================

    sec_filing_velocity = (
        sec_filings_20d / 20.0
    )

    # ========================================================
    # FILING ACCELERATION
    # ========================================================
    #
    # Compare the 5-day filing rate against the 20-day
    # filing rate.
    #
    # Positive:
    #     recent filing activity is increasing.
    #
    # Negative:
    #     recent filing activity is decreasing.
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
    #
    # IMPORTANT:
    #
    # We calculate recency from EVENT DATE, not acceptance
    # time. This represents how recently the filing event
    # occurred relative to the prediction snapshot.
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

            # ------------------------------------------------
            # Most recent filing
            # ------------------------------------------------

            if days_since_last_filing is None:
                days_since_last_filing = days_ago

            # ------------------------------------------------
            # Most recent 10-K
            # ------------------------------------------------

            if (
                form == "10-K"
                and days_since_last_10k is None
            ):
                days_since_last_10k = days_ago

            # ------------------------------------------------
            # Most recent 10-Q
            # ------------------------------------------------

            if (
                form == "10-Q"
                and days_since_last_10q is None
            ):
                days_since_last_10q = days_ago

            # ------------------------------------------------
            # Most recent 8-K
            # ------------------------------------------------

            if (
                form == "8-K"
                and days_since_last_8k is None
            ):
                days_since_last_8k = days_ago

            # Stop once all values have been found.
            if (
                days_since_last_filing is not None
                and days_since_last_10k is not None
                and days_since_last_10q is not None
                and days_since_last_8k is not None
            ):
                break

    # ========================================================
    # RETURN FEATURE VECTOR
    # ========================================================

    return {

        # ====================================================
        # METADATA
        # ====================================================

        "company_id": company_id,

        # ====================================================
        # SEC V1 / CUMULATIVE
        # ====================================================

        "sec_filing_count": (
            sec_filing_count
        ),

        "sec_10k_count": (
            sec_10k_count
        ),

        "sec_10q_count": (
            sec_10q_count
        ),

        "sec_8k_count": (
            sec_8k_count
        ),

        "sec_other_count": (
            sec_other_count
        ),

        # ====================================================
        # SEC V3 - RECENT ACTIVITY
        # ====================================================

        "sec_filings_5d": (
            sec_filings_5d
        ),

        "sec_filings_20d": (
            sec_filings_20d
        ),

        "sec_filings_60d": (
            sec_filings_60d
        ),

        "sec_10k_recent": (
            sec_10k_recent
        ),

        "sec_10q_recent": (
            sec_10q_recent
        ),

        "sec_8k_recent": (
            sec_8k_recent
        ),

        # ====================================================
        # SEC V3 - DYNAMICS
        # ====================================================

        "sec_filing_velocity": (
            sec_filing_velocity
        ),

        "sec_filing_acceleration": (
            sec_filing_acceleration
        ),

        # ====================================================
        # SEC V3 - COMPOSITION
        # ====================================================

        "sec_8k_ratio": (
            sec_8k_ratio
        ),

        "sec_10q_ratio": (
            sec_10q_ratio
        ),

        # ====================================================
        # SEC V3 - RECENCY
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