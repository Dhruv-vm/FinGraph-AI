from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path(
    "data/processed/unified/aapl_training_dataset.json"
)

SNAPSHOT_TIME = datetime(
    2026,
    8,
    21,
    tzinfo=timezone.utc,
)


# ============================================================
# COLUMN DEFINITIONS
# ============================================================

# Metadata columns.
# These identify the observation but are NOT ML features.
METADATA_COLUMNS = {
    "entity_id",
    "ticker",
    "as_of",
}


# Target columns.
# These must never be included in the feature matrix.
TARGET_COLUMNS = {
    "target_return",
    "target_direction",
}


# ============================================================
# DATETIME
# ============================================================

def parse_datetime(
    value: Any,
) -> datetime | None:
    """Convert a date/time value into UTC-aware datetime."""

    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
    else:
        value = str(value).strip()

        if not value:
            return None

        if len(value) == 10:
            value = f"{value}T00:00:00+00:00"

        dt = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

    if dt.tzinfo is None:
        dt = dt.replace(
            tzinfo=timezone.utc
        )
    else:
        dt = dt.astimezone(
            timezone.utc
        )

    return dt


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(
    path: Path,
) -> list[dict[str, Any]]:

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Expected ML dataset to be a list."
        )

    return data


# ============================================================
# VALIDATION
# ============================================================

def validate_features(
    dataset: list[dict[str, Any]],
) -> None:

    print("=" * 70)
    print("ML FEATURE-GENERATION VALIDATION")
    print("=" * 70)

    print(
        f"Rows: {len(dataset)}"
    )

    if not dataset:
        raise ValueError(
            "Dataset is empty."
        )

    df = pd.DataFrame(dataset)

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = {
        "entity_id",
        "ticker",
        "as_of",
        "target_return",
        "target_direction",
    }

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    print()
    print("1. REQUIRED COLUMNS")
    print("-" * 70)

    if missing:
        print(
            "Missing:",
            missing,
        )

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("Status: PASS")

    # --------------------------------------------------------
    # AS_OF VALIDATION
    # --------------------------------------------------------

    df["as_of"] = pd.to_datetime(
        df["as_of"],
        errors="coerce",
        utc=True,
    )

    invalid_as_of = int(
        df["as_of"].isna().sum()
    )

    print()
    print("2. AS_OF VALIDATION")
    print("-" * 70)

    print(
        f"Invalid as_of: {invalid_as_of}"
    )

    if invalid_as_of:
        raise ValueError(
            "Invalid as_of values detected."
        )

    print(
        f"First snapshot: {df['as_of'].min()}"
    )

    print(
        f"Last snapshot: {df['as_of'].max()}"
    )

    # --------------------------------------------------------
    # CHRONOLOGICAL ORDER
    # --------------------------------------------------------

    chronological = (
        df["as_of"]
        .is_monotonic_increasing
    )

    print()
    print("3. CHRONOLOGICAL ORDER")
    print("-" * 70)

    print(
        f"Chronological: {chronological}"
    )

    if not chronological:
        raise ValueError(
            "ML dataset is not chronologically ordered."
        )

    # --------------------------------------------------------
    # TARGET VALIDATION
    # --------------------------------------------------------

    print()
    print("4. TARGET VALIDATION")
    print("-" * 70)

    print("Target direction distribution:")

    print(
        df["target_direction"]
        .value_counts()
        .sort_index()
    )

    # Direction must be binary.
    invalid_direction = df[
        ~df["target_direction"]
        .isin([0, 1])
    ]

    print(
        f"Invalid target_direction values: "
        f"{len(invalid_direction)}"
    )

    if len(invalid_direction):
        raise ValueError(
            "target_direction contains values "
            "other than 0/1."
        )

    # target_return must be numeric.
    target_return_numeric = pd.to_numeric(
        df["target_return"],
        errors="coerce",
    )

    invalid_target_return = (
        target_return_numeric.isna()
        & df["target_return"].notna()
    )

    print(
        f"Invalid target_return values: "
        f"{int(invalid_target_return.sum())}"
    )

    if invalid_target_return.any():
        raise ValueError(
            "target_return contains non-numeric values."
        )

    df["target_return"] = target_return_numeric

    # --------------------------------------------------------
    # FEATURE COLUMN SELECTION
    # --------------------------------------------------------

    # Only actual predictive variables belong here.
    #
    # Excluded:
    #   entity_id        -> identifier
    #   ticker           -> identifier
    #   as_of            -> timestamp / metadata
    #   target_return    -> prediction target
    #   target_direction -> prediction target
    feature_columns = [
        column
        for column in df.columns
        if column not in (
            METADATA_COLUMNS
            | TARGET_COLUMNS
        )
    ]

    print()
    print("5. FEATURE VALIDATION")
    print("-" * 70)

    print(
        f"ML feature columns: "
        f"{len(feature_columns)}"
    )

    print()
    print("Metadata columns excluded:")

    for column in sorted(
        METADATA_COLUMNS
    ):
        print(
            f"- {column}"
        )

    print()
    print("Target columns excluded:")

    for column in sorted(
        TARGET_COLUMNS
    ):
        print(
            f"- {column}"
        )

    # --------------------------------------------------------
    # NUMERIC FEATURE VALIDATION
    # --------------------------------------------------------

    non_numeric = []

    for column in feature_columns:

        converted = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        original_non_null = (
            df[column].notna()
        )

        converted_non_null = (
            converted.notna()
        )

        if (
            converted_non_null.sum()
            != original_non_null.sum()
        ):
            non_numeric.append(
                column
            )

        df[column] = converted

    print()
    print(
        "Non-numeric ML features:",
        non_numeric,
    )

    if non_numeric:
        raise ValueError(
            "Non-numeric ML feature columns: "
            f"{non_numeric}"
        )

    print(
        "Numeric feature validation: PASS"
    )

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    print()
    print("6. MISSING FEATURE VALUES")
    print("-" * 70)

    missing_counts = (
        df[feature_columns]
        .isna()
        .sum()
    )

    total_missing = int(
        missing_counts.sum()
    )

    print(
        f"Total missing values: "
        f"{total_missing}"
    )

    if total_missing:

        print()
        print("Features with missing values:")

        print(
            missing_counts[
                missing_counts > 0
            ]
        )

    else:

        print(
            "Missing feature values: NONE"
        )

    # Missing values are not automatically leakage.
    # They may legitimately occur before enough history exists.

    # --------------------------------------------------------
    # CONSTANT FEATURES
    # --------------------------------------------------------

    print()
    print("7. CONSTANT FEATURES")
    print("-" * 70)

    constant_features = [
        column
        for column in feature_columns
        if df[column].nunique(
            dropna=True
        ) <= 1
    ]

    print(
        f"Count: {len(constant_features)}"
    )

    if constant_features:

        for column in constant_features:
            unique_values = (
                df[column]
                .dropna()
                .unique()
                .tolist()
            )

            print(
                f"- {column}: "
                f"{unique_values}"
            )

    else:

        print(
            "No constant ML features detected."
        )

    # --------------------------------------------------------
    # FEATURE TEMPORAL COVERAGE
    # --------------------------------------------------------

    print()
    print("8. FEATURE TEMPORAL COVERAGE")
    print("-" * 70)

    for column in feature_columns:

        non_null = int(
            df[column]
            .notna()
            .sum()
        )

        print(
            f"{column:<30}"
            f"{non_null:>5}/{len(df)}"
        )

    # --------------------------------------------------------
    # SNAPSHOT BOUNDARY
    # --------------------------------------------------------

    print()
    print("9. SNAPSHOT BOUNDARY")
    print("-" * 70)

    future_rows = df[
        df["as_of"]
        > SNAPSHOT_TIME
    ]

    print(
        f"Rows after snapshot: "
        f"{len(future_rows)}"
    )

    if len(future_rows):

        raise ValueError(
            "Dataset contains snapshots after "
            f"{SNAPSHOT_TIME.isoformat()}."
        )

    print(
        f"Snapshot boundary: "
        f"{SNAPSHOT_TIME.isoformat()}"
    )

    # --------------------------------------------------------
    # DUPLICATE SNAPSHOTS
    # --------------------------------------------------------

    print()
    print("10. DUPLICATE SNAPSHOTS")
    print("-" * 70)

    duplicate_count = int(
        df["as_of"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate as_of rows: "
        f"{duplicate_count}"
    )

    if duplicate_count:

        print(
            "WARNING: duplicate snapshot dates detected."
        )

    # --------------------------------------------------------
    # ENTITY CONSISTENCY
    # --------------------------------------------------------

    print()
    print("11. ENTITY CONSISTENCY")
    print("-" * 70)

    entity_count = int(
        df["entity_id"]
        .nunique()
    )

    ticker_count = int(
        df["ticker"]
        .nunique()
    )

    print(
        f"Unique entity_id values: "
        f"{entity_count}"
    )

    print(
        f"Unique ticker values   : "
        f"{ticker_count}"
    )

    if entity_count != 1:

        print(
            "WARNING: dataset contains "
            f"{entity_count} entities."
        )

    if ticker_count != 1:

        print(
            "WARNING: dataset contains "
            f"{ticker_count} tickers."
        )

    # --------------------------------------------------------
    # TARGET COVERAGE
    # --------------------------------------------------------

    print()
    print("12. TARGET COVERAGE")
    print("-" * 70)

    missing_target_direction = int(
        df["target_direction"]
        .isna()
        .sum()
    )

    missing_target_return = int(
        df["target_return"]
        .isna()
        .sum()
    )

    print(
        f"Missing target_direction: "
        f"{missing_target_direction}"
    )

    print(
        f"Missing target_return   : "
        f"{missing_target_return}"
    )

    if missing_target_direction:

        raise ValueError(
            "Missing target_direction values detected."
        )

    if missing_target_return:

        raise ValueError(
            "Missing target_return values detected."
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "ML FEATURE-GENERATION VALIDATION SUMMARY"
    )
    print("=" * 70)

    print(
        f"Rows                  : {len(df)}"
    )

    print(
        f"ML features           : "
        f"{len(feature_columns)}"
    )

    print(
        f"Metadata columns      : "
        f"{len(METADATA_COLUMNS)}"
    )

    print(
        f"Target columns        : "
        f"{len(TARGET_COLUMNS)}"
    )

    print(
        f"Missing feature vals  : "
        f"{total_missing}"
    )

    print(
        f"Constant features     : "
        f"{len(constant_features)}"
    )

    print(
        f"Future snapshots      : "
        f"{len(future_rows)}"
    )

    print(
        f"Duplicate snapshots   : "
        f"{duplicate_count}"
    )

    print(
        f"Unique entities       : "
        f"{entity_count}"
    )

    print(
        f"Unique tickers        : "
        f"{ticker_count}"
    )

    print()
    print(
        "ML FEATURE-GENERATION VALIDATION: PASSED"
    )
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    dataset = load_dataset(
        DATASET_PATH
    )

    validate_features(
        dataset
    )


if __name__ == "__main__":
    main()