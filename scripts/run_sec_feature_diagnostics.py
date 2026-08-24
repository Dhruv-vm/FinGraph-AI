from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified"
    / "aapl_training_dataset.json"
)


SEC_FEATURES = [
    # Original SEC features
    "sec_filing_count",
    "sec_10k_count",
    "sec_10q_count",
    "sec_8k_count",

    # Recent activity
    "sec_filings_5d",
    "sec_filings_20d",
    "sec_filings_60d",
    "sec_10k_recent",
    "sec_10q_recent",
    "sec_8k_recent",

    # Dynamics
    "sec_filing_velocity",
    "sec_filing_acceleration",
    "sec_8k_ratio",
    "sec_10q_ratio",

    # Recency
    "days_since_last_filing",
    "days_since_last_10k",
    "days_since_last_10q",
    "days_since_last_8k",
]


def main() -> None:

    # ========================================================
    # LOAD DATASET
    # ========================================================

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    df = pd.DataFrame(dataset)

    print("=" * 70)
    print("FINGRAPH V3 SEC FEATURE DIAGNOSTICS")
    print("=" * 70)

    print(f"Rows: {len(df)}")

    # ========================================================
    # REQUIRED COLUMN CHECK
    # ========================================================

    required_columns = [
        "as_of",
        "target_direction",
        *SEC_FEATURES,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ========================================================
    # PARSE SNAPSHOT DATE
    # ========================================================

    df["as_of"] = pd.to_datetime(
        df["as_of"],
        errors="coerce",
        utc=True,
    )

    invalid_as_of = int(
        df["as_of"].isna().sum()
    )

    print()
    print("=" * 70)
    print("AS_OF VALIDATION")
    print("=" * 70)

    print(
        "Invalid as_of:",
        invalid_as_of,
    )

    if invalid_as_of > 0:

        print()
        print("BAD as_of VALUES:")

        print(
            df.loc[
                df["as_of"].isna()
            ].to_string(
                index=False
            )
        )

        raise ValueError(
            "Invalid as_of values found. "
            "Fix the dataset before running diagnostics."
        )

    # ========================================================
    # SORT CHRONOLOGICALLY
    # ========================================================

    df = (
        df.sort_values(
            "as_of"
        )
        .reset_index(drop=True)
    )

    print()
    print(
        "First snapshot:",
        df["as_of"].iloc[0],
    )

    print(
        "Last snapshot:",
        df["as_of"].iloc[-1],
    )

    # ========================================================
    # CONVERT SEC FEATURES TO NUMERIC
    # ========================================================

    for feature in SEC_FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    # ========================================================
    # DATA QUALITY
    # ========================================================

    print()
    print("=" * 70)
    print("DATA QUALITY")
    print("=" * 70)

    for feature in SEC_FEATURES:

        series = df[feature]

        print()
        print(feature)
        print("-" * len(feature))

        print(
            "Missing:",
            int(series.isna().sum()),
            f"({series.isna().mean() * 100:.2f}%)",
        )

        print(
            "Unique:",
            series.nunique(
                dropna=True
            ),
        )

        print(
            "Min:",
            series.min(),
        )

        print(
            "Max:",
            series.max(),
        )

        print(
            "Mean:",
            series.mean(),
        )

        print(
            "Std:",
            series.std(),
        )

    # ========================================================
    # CONSTANT FEATURES
    # ========================================================

    print()
    print("=" * 70)
    print("CONSTANT FEATURES")
    print("=" * 70)

    constants = [
        feature
        for feature in SEC_FEATURES
        if df[feature].nunique(
            dropna=True
        ) <= 1
    ]

    if constants:

        for feature in constants:
            print(
                "-",
                feature,
            )

    else:

        print("None")

    # ========================================================
    # TARGET VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("TARGET VALIDATION")
    print("=" * 70)

    print(
        df["target_direction"]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # CORRELATION WITH TARGET
    # ========================================================

    print()
    print("=" * 70)
    print("CORRELATION WITH TARGET_DIRECTION")
    print("=" * 70)

    correlations = []

    for feature in SEC_FEATURES:

        feature_series = df[
            feature
        ]

        target_series = pd.to_numeric(
            df["target_direction"],
            errors="coerce",
        )

        valid = (
            feature_series.notna()
            & target_series.notna()
        )

        if (
            valid.sum() < 2
            or feature_series[valid].nunique() <= 1
            or target_series[valid].nunique() <= 1
        ):
            correlation = float("nan")

        else:

            correlation = (
                feature_series[valid]
                .corr(
                    target_series[valid]
                )
            )

        correlations.append(
            (
                feature,
                correlation,
            )
        )

    # Sort valid correlations first
    correlations.sort(
        key=lambda item: (
            pd.isna(item[1]),
            -abs(item[1])
            if not pd.isna(item[1])
            else 0,
        )
    )

    for feature, correlation in correlations:

        if pd.isna(correlation):

            print(
                f"{feature:<30} nan"
            )

        else:

            print(
                f"{feature:<30} "
                f"{correlation:.4f}"
            )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    split_index = int(
        len(df) * 0.8
    )

    train_df = df.iloc[
        :split_index
    ].copy()

    test_df = df.iloc[
        split_index:
    ].copy()

    print()
    print("=" * 70)
    print("TEMPORAL TRAIN / TEST SPLIT")
    print("=" * 70)

    print(
        f"Train rows: {len(train_df)}"
    )

    print(
        f"Test rows:  {len(test_df)}"
    )

    print(
        f"Train period: "
        f"{train_df['as_of'].iloc[0].date()} "
        f"-> "
        f"{train_df['as_of'].iloc[-1].date()}"
    )

    print(
        f"Test period:  "
        f"{test_df['as_of'].iloc[0].date()} "
        f"-> "
        f"{test_df['as_of'].iloc[-1].date()}"
    )

    # ========================================================
    # TRAIN / TEST DISTRIBUTION
    # ========================================================

    print()
    print("=" * 70)
    print("TRAIN vs TEST DISTRIBUTION")
    print("=" * 70)

    for feature in SEC_FEATURES:

        train = train_df[feature]
        test = test_df[feature]

        print()
        print(feature)

        print(
            f"Train: "
            f"mean={train.mean():.4f}, "
            f"std={train.std():.4f}, "
            f"min={train.min():.4f}, "
            f"max={train.max():.4f}"
        )

        print(
            f"Test:  "
            f"mean={test.mean():.4f}, "
            f"std={test.std():.4f}, "
            f"min={test.min():.4f}, "
            f"max={test.max():.4f}"
        )

    # ========================================================
    # RECENCY-SPECIFIC CHECK
    # ========================================================

    RECENCY_FEATURES = [
        "days_since_last_filing",
        "days_since_last_10k",
        "days_since_last_10q",
        "days_since_last_8k",
    ]

    print()
    print("=" * 70)
    print("RECENCY VALIDATION")
    print("=" * 70)

    for feature in RECENCY_FEATURES:

        series = df[feature]

        print()
        print(feature)

        print(
            "Unique:",
            series.nunique(
                dropna=True
            ),
        )

        print(
            "Min:",
            series.min(),
        )

        print(
            "Max:",
            series.max(),
        )

        print(
            "Mean:",
            series.mean(),
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    missing_feature_count = sum(
        df[feature]
        .isna()
        .any()
        for feature in SEC_FEATURES
    )

    print()
    print("=" * 70)
    print("SEC FEATURE DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print(
        "Total SEC features:",
        len(SEC_FEATURES),
    )

    print(
        "Constant features:",
        len(constants),
    )

    print(
        "Features with missing values:",
        missing_feature_count,
    )

    print()
    print("Top correlations:")

    valid_correlations = [
        (
            feature,
            correlation,
        )
        for feature, correlation
        in correlations
        if not pd.isna(correlation)
    ]

    for feature, correlation in valid_correlations[:5]:

        print(
            f"- {feature}: "
            f"{correlation:.4f}"
        )

    print()
    print("=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()