from __future__ import annotations

import os
import sys
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Allow imports from project root
# ---------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET_PATH = (
    "data/processed/unified/"
    "aapl_training_dataset.json"
)

SEC_FEATURES = [
    "sec_filing_count",
    "sec_10k_count",
    "sec_10q_count",
    "sec_8k_count",
]

MARKET_FEATURES = [
    "return_5d",
    "return_20d",
    "return_60d",
    "price_vs_20d_ma",
    "price_vs_50d_ma",
    "momentum_20d",
    "volatility_20d",
]

MACRO_FEATURES = [
    "dgs10",
    "dgs3mo",
]


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

def load_dataset(
    path: str,
) -> pd.DataFrame:
    """Load the saved FinGraph training dataset."""

    df = pd.read_json(path)

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    required = [
        *SEC_FEATURES,
        "target_direction",
        "as_of",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["as_of"] = pd.to_datetime(
        df["as_of"],
        errors="coerce",
    )

    if df["as_of"].isna().any():
        raise ValueError(
            "Invalid values found in as_of."
        )

    df = (
        df.sort_values("as_of")
        .reset_index(drop=True)
    )

    for column in [
        *SEC_FEATURES,
        *MARKET_FEATURES,
        *MACRO_FEATURES,
        "target_direction",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


# ---------------------------------------------------------
# Section 1: SEC distribution
# ---------------------------------------------------------

def print_sec_distribution(
    df: pd.DataFrame,
) -> None:
    print()
    print("=" * 60)
    print("SEC FEATURE DISTRIBUTION")
    print("=" * 60)

    for feature in SEC_FEATURES:
        series = df[feature].dropna()

        print()
        print(feature)

        print(
            "unique:",
            series.nunique(),
        )

        print(
            "min:",
            series.min(),
        )

        print(
            "max:",
            series.max(),
        )

        print(
            "mean:",
            series.mean(),
        )

        print(
            "std:",
            series.std(),
        )


# ---------------------------------------------------------
# Section 2: SEC vs target
# ---------------------------------------------------------

def print_sec_vs_target(
    df: pd.DataFrame,
) -> None:
    print()
    print("=" * 60)
    print("SEC FEATURES VS TARGET")
    print("=" * 60)

    for feature in SEC_FEATURES:
        up = df.loc[
            df["target_direction"] == 1,
            feature,
        ].mean()

        down = df.loc[
            df["target_direction"] == 0,
            feature,
        ].mean()

        difference = up - down

        print(
            f"{feature:<22}"
            f"UP = {up:.6f} "
            f"DOWN = {down:.6f} "
            f"DIFF = {difference:+.6f}"
        )


# ---------------------------------------------------------
# Section 3: SEC availability
# ---------------------------------------------------------

def print_sec_availability(
    df: pd.DataFrame,
) -> None:
    print()
    print("=" * 60)
    print("SEC FEATURE AVAILABILITY")
    print("=" * 60)

    for feature in SEC_FEATURES:
        nonzero = (
            df[feature]
            .fillna(0)
            .ne(0)
            .sum()
        )

        total = len(df)

        print(
            f"{feature:<22}"
            f"{nonzero}/{total} "
            f"({nonzero / total:.2%})"
        )


# ---------------------------------------------------------
# Section 4: SEC vs target correlation
# ---------------------------------------------------------

def print_correlations(
    df: pd.DataFrame,
) -> None:
    print()
    print("=" * 60)
    print("SEC FEATURE / TARGET CORRELATION")
    print("=" * 60)

    for feature in SEC_FEATURES:
        correlation = df[
            [feature, "target_direction"]
        ].corr().iloc[0, 1]

        print(
            f"{feature:<22}"
            f"{correlation:+.6f}"
        )


# ---------------------------------------------------------
# Section 5: Temporal SEC behavior
# ---------------------------------------------------------

def print_temporal_behavior(
    df: pd.DataFrame,
) -> None:
    print()
    print("=" * 60)
    print("SEC TEMPORAL BEHAVIOR")
    print("=" * 60)

    print()
    print(
        "First 10 chronological rows:"
    )

    print(
        df[
            [
                "as_of",
                *SEC_FEATURES,
                "target_direction",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print(
        "Last 10 chronological rows:"
    )

    print(
        df[
            [
                "as_of",
                *SEC_FEATURES,
                "target_direction",
            ]
        ]
        .tail(10)
        .to_string(index=False)
    )


# ---------------------------------------------------------
# Section 6: SEC-only baseline
# ---------------------------------------------------------

def run_baseline(
    df: pd.DataFrame,
    features: list[str],
    name: str,
) -> dict[str, Any]:

    clean = df.dropna(
        subset=features + [
            "target_direction"
        ]
    ).copy()

    split = int(
        len(clean) * 0.8
    )

    train = clean.iloc[
        :split
    ]

    test = clean.iloc[
        split:
    ]

    X_train = train[
        features
    ]

    X_test = test[
        features
    ]

    y_train = train[
        "target_direction"
    ]

    y_test = test[
        "target_direction"
    ]

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=0.1,
        solver="liblinear",
        random_state=42,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    predictions = model.predict(
        X_test_scaled
    )

    probabilities = (
        model.predict_proba(
            X_test_scaled
        )[:, 1]
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    if y_test.nunique() > 1:
        roc_auc = roc_auc_score(
            y_test,
            probabilities,
        )
    else:
        roc_auc = None

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(
        "Features:",
        len(features),
    )

    print(
        "Train:",
        len(train),
    )

    print(
        "Test:",
        len(test),
    )

    print(
        "Accuracy:",
        accuracy,
    )

    print(
        "Precision:",
        precision,
    )

    print(
        "Recall:",
        recall,
    )

    print(
        "F1:",
        f1,
    )

    print(
        "ROC-AUC:",
        roc_auc,
    )

    print()
    print("Probability:")
    print(
        "Min:",
        probabilities.min(),
    )
    print(
        "Max:",
        probabilities.max(),
    )
    print(
        "Mean:",
        probabilities.mean(),
    )

    print()
    print("Coefficients:")

    coefficients = pd.DataFrame(
        {
            "feature": features,
            "coefficient": model.coef_[0],
        }
    )

    coefficients[
        "abs_coefficient"
    ] = coefficients[
        "coefficient"
    ].abs()

    coefficients = coefficients.sort_values(
        "abs_coefficient",
        ascending=False,
    )

    print(
        coefficients[
            [
                "feature",
                "coefficient",
            ]
        ].to_string(
            index=False
        )
    )

    return {
        "name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    df = load_dataset(
        DATASET_PATH
    )

    print()
    print("=" * 60)
    print("FINGRAPH SEC FEATURE DIAGNOSTIC")
    print("=" * 60)

    print(
        "Rows:",
        len(df),
    )

    print(
        "First:",
        df["as_of"].min(),
    )

    print(
        "Last:",
        df["as_of"].max(),
    )

    print()
    print("Target distribution:")
    print(
        df[
            "target_direction"
        ].value_counts()
        .sort_index()
    )

    # ---------------------------------------------
    # Diagnostics
    # ---------------------------------------------

    print_sec_distribution(
        df
    )

    print_sec_vs_target(
        df
    )

    print_sec_availability(
        df
    )

    print_correlations(
        df
    )

    print_temporal_behavior(
        df
    )

    # ---------------------------------------------
    # Baselines
    # ---------------------------------------------

    results = []

    results.append(
        run_baseline(
            df,
            SEC_FEATURES,
            "SEC ONLY BASELINE",
        )
    )

    results.append(
        run_baseline(
            df,
            MARKET_FEATURES,
            "MARKET ONLY BASELINE",
        )
    )

    results.append(
        run_baseline(
            df,
            MARKET_FEATURES
            + SEC_FEATURES,
            "MARKET + SEC BASELINE",
        )
    )

    results.append(
        run_baseline(
            df,
            MARKET_FEATURES
            + SEC_FEATURES
            + MACRO_FEATURES,
            "MARKET + SEC + MACRO BASELINE",
        )
    )

    # ---------------------------------------------
    # Final comparison
    # ---------------------------------------------

    print()
    print("=" * 60)
    print("SEC ABLATION COMPARISON")
    print("=" * 60)

    comparison = pd.DataFrame(
        results
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print("=" * 60)
    print("SEC DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()