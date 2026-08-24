from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from project root when running:
# python scripts/run_threshold_analysis.py
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


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


def load_dataset() -> pd.DataFrame:
    """Load and chronologically sort the V2 dataset."""

    df = pd.read_json(DATASET_PATH)

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

    return df


def main() -> None:

    print("=" * 50)
    print("FINGRAPH V2 THRESHOLD ANALYSIS")
    print("=" * 50)

    df = load_dataset()

    train_size = int(len(df) * 0.8)

    train_df = df.iloc[
        :train_size
    ].copy()

    test_df = df.iloc[
        train_size:
    ].copy()

    print(f"Rows: {len(df)}")
    print(f"Train: {len(train_df)}")
    print(f"Test: {len(test_df)}")

    print()
    print("TRAIN TARGET")
    print(train_df["target_direction"].value_counts())

    print()
    print("TEST TARGET")
    print(test_df["target_direction"].value_counts())

    # -------------------------
    # Prepare features
    # -------------------------

    X_train = (
        train_df[FEATURE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    X_test = (
        test_df[FEATURE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    y_train = train_df[
        "target_direction"
    ]

    y_test = test_df[
        "target_direction"
    ]

    # -------------------------
    # Scale using TRAIN only
    # -------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # -------------------------
    # Train baseline
    # -------------------------

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

    probabilities = model.predict_proba(
        X_test_scaled
    )[:, 1]

    # -------------------------
    # ROC-AUC is threshold-independent
    # -------------------------

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print()
    print("=" * 50)
    print("THRESHOLD-INDEPENDENT RESULT")
    print("=" * 50)

    print(
        f"ROC-AUC: {auc:.4f}"
    )

    print(
        f"Probability range: "
        f"{probabilities.min():.4f}"
        f" -> "
        f"{probabilities.max():.4f}"
    )

    # -------------------------
    # Threshold analysis
    # -------------------------

    thresholds = [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]

    results = []

    print()
    print("=" * 90)
    print("THRESHOLD ANALYSIS")
    print("=" * 90)

    print(
        f"{'Threshold':>10} "
        f"{'Accuracy':>10} "
        f"{'Balanced':>10} "
        f"{'Precision':>10} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'UP %':>10}"
    )

    print("-" * 90)

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        balanced = balanced_accuracy_score(
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

        up_percentage = (
            predictions.mean()
            * 100
        )

        results.append(
            {
                "threshold": threshold,
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "up_percentage": up_percentage,
            }
        )

        print(
            f"{threshold:10.2f} "
            f"{accuracy:10.4f} "
            f"{balanced:10.4f} "
            f"{precision:10.4f} "
            f"{recall:10.4f} "
            f"{f1:10.4f} "
            f"{up_percentage:9.2f}%"
        )

    # -------------------------
    # Best threshold by
    # balanced accuracy
    # -------------------------

    result_df = pd.DataFrame(
        results
    )

    best_balanced = result_df.loc[
        result_df[
            "balanced_accuracy"
        ].idxmax()
    ]

    best_f1 = result_df.loc[
        result_df["f1"].idxmax()
    ]

    print()
    print("=" * 50)
    print("BEST THRESHOLDS")
    print("=" * 50)

    print(
        "Best Balanced Accuracy:"
    )

    print(
        f"Threshold: "
        f"{best_balanced['threshold']:.2f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{best_balanced['balanced_accuracy']:.4f}"
    )

    print(
        f"Accuracy: "
        f"{best_balanced['accuracy']:.4f}"
    )

    print(
        f"F1: "
        f"{best_balanced['f1']:.4f}"
    )

    print(
        f"UP %: "
        f"{best_balanced['up_percentage']:.2f}%"
    )

    print()
    print("Best F1:")

    print(
        f"Threshold: "
        f"{best_f1['threshold']:.2f}"
    )

    print(
        f"F1: "
        f"{best_f1['f1']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{best_f1['balanced_accuracy']:.4f}"
    )

    print(
        f"UP %: "
        f"{best_f1['up_percentage']:.2f}%"
    )

    # -------------------------
    # Confusion matrix for
    # best balanced threshold
    # -------------------------

    threshold = float(
        best_balanced["threshold"]
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print()
    print("=" * 50)
    print(
        f"CONFUSION MATRIX @ "
        f"{threshold:.2f}"
    )
    print("=" * 50)

    print(cm)

    print()
    print(
        "Predicted distribution:"
    )

    print(
        pd.Series(
            predictions
        ).value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()