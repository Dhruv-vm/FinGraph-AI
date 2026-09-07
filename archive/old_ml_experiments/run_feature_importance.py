from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


DATASET_PATH = (
    ROOT
    / "data"
    / "processed"
    / "unified"
    / "aapl_training_dataset.json"
)


FEATURE_COLUMNS = [
    # Market V2
    "return_5d",
    "return_20d",
    "return_60d",
    "price_vs_20d_ma",
    "price_vs_50d_ma",
    "momentum_20d",
    "volatility_20d",

    # SEC
    "sec_filing_count",
    "sec_10k_count",
    "sec_10q_count",
    "sec_8k_count",

    # Macro
    "dgs10",
    "dgs3mo",
]


def main():

    print("=" * 60)
    print("FINGRAPH V2 FEATURE IMPORTANCE")
    print("=" * 60)

    df = pd.read_json(DATASET_PATH)

    df["as_of"] = pd.to_datetime(
        df["as_of"],
        errors="coerce",
    )

    df = (
        df.sort_values("as_of")
        .reset_index(drop=True)
    )

    split_index = int(
        len(df) * 0.8
    )

    train_df = df.iloc[
        :split_index
    ].copy()

    test_df = df.iloc[
        split_index:
    ].copy()

    X_train = (
        train_df[FEATURE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    y_train = train_df[
        "target_direction"
    ]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
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

    coefficients = (
        model.coef_[0]
    )

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "coefficient": coefficients,
            "abs_coefficient": abs(
                coefficients
            ),
        }
    )

    importance = (
        importance
        .sort_values(
            "abs_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print()
    print("TRAIN ROWS:", len(train_df))
    print("TEST ROWS:", len(test_df))

    print()
    print("=" * 60)
    print("FEATURE COEFFICIENTS")
    print("=" * 60)

    for _, row in importance.iterrows():

        print(
            f"{row['feature']:<25}"
            f"{row['coefficient']:+.6f}"
        )

    print()
    print("=" * 60)
    print("TOP FEATURES")
    print("=" * 60)

    print(
        importance[
            [
                "feature",
                "coefficient",
                "abs_coefficient",
            ]
        ].head(10).to_string(
            index=False
        )
    )

    print()
    print("=" * 60)
    print("TRAIN FEATURE MEANS BY TARGET")
    print("=" * 60)

    means = (
        train_df
        .groupby("target_direction")[
            FEATURE_COLUMNS
        ]
        .mean()
        .T
    )

    means.columns = [
        "DOWN",
        "UP",
    ]

    means["difference"] = (
        means["UP"]
        - means["DOWN"]
    )

    print(
        means.to_string()
    )

    print()
    print("=" * 60)
    print("FEATURE STANDARD DEVIATIONS")
    print("=" * 60)

    print(
        train_df[
            FEATURE_COLUMNS
        ]
        .std()
        .sort_values(
            ascending=False
        )
        .to_string()
    )


if __name__ == "__main__":
    main()