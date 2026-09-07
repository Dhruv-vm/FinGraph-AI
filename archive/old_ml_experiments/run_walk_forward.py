from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path(
    "data/processed/unified/aapl_training_dataset.json"
)

MIN_TRAIN_SIZE = 140
TEST_SIZE = 30
STEP_SIZE = 30
RANDOM_STATE = 42


# ============================================================
# FEATURE GROUPS
# ============================================================

MARKET_FEATURES = [
    "return_5d",
    "return_20d",
    "return_60d",
    "price_vs_20d_ma",
    "price_vs_50d_ma",
    "momentum_20d",
    "volatility_20d",
]

SEC_V3_FEATURES = [
    "sec_filings_5d",
    "sec_filings_20d",
    "sec_filings_60d",
    "sec_10k_recent",
    "sec_10q_recent",
    "sec_8k_recent",
    "sec_filing_velocity",
    "sec_filing_acceleration",
    "sec_8k_ratio",
    "sec_10q_ratio",
    "days_since_last_10k",
    "days_since_last_10q",
]

MACRO_FEATURES = [
    "dgs10",
    "dgs3mo",
]

NEWS_V2_FEATURES = [
    "news_event_count",
    "average_sentiment",
    "positive_news_count",
    "negative_news_count",
    "neutral_news_count",
    "unique_news_sources",
    "sentiment_std",
    "positive_ratio",
    "negative_ratio",
    "neutral_ratio",
    "sentiment_strength",
    "source_diversity",
]


FEATURE_CONFIGS = {
    "MARKET ONLY": MARKET_FEATURES,
    "MARKET + SEC V3": (
        MARKET_FEATURES
        + SEC_V3_FEATURES
    ),
    "MARKET + SEC V3 + MACRO": (
        MARKET_FEATURES
        + SEC_V3_FEATURES
        + MACRO_FEATURES
    ),
    "MARKET + SEC V3 + NEWS": (
        MARKET_FEATURES
        + SEC_V3_FEATURES
        + NEWS_V2_FEATURES
    ),
}


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(
    path: Path,
) -> pd.DataFrame:

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
            "Expected dataset to be a list."
        )

    df = pd.DataFrame(data)

    if "as_of" not in df.columns:
        raise ValueError(
            "Dataset is missing 'as_of'."
        )

    if "target_direction" not in df.columns:
        raise ValueError(
            "Dataset is missing 'target_direction'."
        )

    df["as_of"] = pd.to_datetime(
        df["as_of"],
        utc=True,
        errors="coerce",
    )

    if df["as_of"].isna().any():
        raise ValueError(
            "Invalid as_of values detected."
        )

    df = df.sort_values(
        "as_of"
    ).reset_index(drop=True)

    return df


# ============================================================
# MODEL
# ============================================================

def create_model() -> Pipeline:

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


# ============================================================
# WALK-FORWARD SPLITS
# ============================================================

def generate_splits(
    n_rows: int,
) -> list[tuple[int, int, int, int]]:

    splits = []

    train_end = MIN_TRAIN_SIZE

    while train_end + TEST_SIZE <= n_rows:

        test_start = train_end
        test_end = (
            test_start
            + TEST_SIZE
        )

        splits.append(
            (
                0,
                train_end,
                test_start,
                test_end,
            )
        )

        train_end += STEP_SIZE

    return splits


# ============================================================
# SINGLE CONFIGURATION
# ============================================================

def evaluate_configuration(
    df: pd.DataFrame,
    name: str,
    features: list[str],
) -> dict[str, Any]:

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"{name}: missing features: "
            f"{missing_features}"
        )

    X = df[features].copy()
    y = df["target_direction"].astype(int)

    splits = generate_splits(
        len(df)
    )

    if not splits:
        raise ValueError(
            "Not enough rows for "
            "walk-forward evaluation."
        )

    fold_results = []

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"Features: {len(features)}"
    )

    print(
        f"Folds: {len(splits)}"
    )

    for fold_number, (
        train_start,
        train_end,
        test_start,
        test_end,
    ) in enumerate(
        splits,
        start=1,
    ):

        X_train = X.iloc[
            train_start:train_end
        ]

        y_train = y.iloc[
            train_start:train_end
        ]

        X_test = X.iloc[
            test_start:test_end
        ]

        y_test = y.iloc[
            test_start:test_end
        ]

        model = create_model()

        model.fit(
            X_train,
            y_train,
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        predictions = (
            probabilities >= 0.50
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

        if y_test.nunique() == 2:
            roc_auc = roc_auc_score(
                y_test,
                probabilities,
            )
        else:
            roc_auc = np.nan

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )

        train_start_date = (
            df.iloc[
                train_start
            ]["as_of"]
        )

        train_end_date = (
            df.iloc[
                train_end - 1
            ]["as_of"]
        )

        test_start_date = (
            df.iloc[
                test_start
            ]["as_of"]
        )

        test_end_date = (
            df.iloc[
                test_end - 1
            ]["as_of"]
        )

        fold_result = {
            "fold": fold_number,
            "train_rows": (
                train_end
                - train_start
            ),
            "test_rows": (
                test_end
                - test_start
            ),
            "train_start": (
                train_start_date.isoformat()
            ),
            "train_end": (
                train_end_date.isoformat()
            ),
            "test_start": (
                test_start_date.isoformat()
            ),
            "test_end": (
                test_end_date.isoformat()
            ),
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": (
                cm.tolist()
            ),
        }

        fold_results.append(
            fold_result
        )

        print(
            f"Fold {fold_number:02d} | "
            f"Train {train_end:3d} | "
            f"Test {test_start}:{test_end} | "
            f"Accuracy={accuracy:.4f} | "
            f"Balanced={balanced:.4f} | "
            f"F1={f1:.4f} | "
            f"ROC-AUC={roc_auc:.4f}"
        )

    # --------------------------------------------------------
    # AGGREGATE
    # --------------------------------------------------------

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    summary = {}

    for metric in metric_names:

        values = np.array(
            [
                result[metric]
                for result in fold_results
                if not np.isnan(
                    result[metric]
                )
            ],
            dtype=float,
        )

        if len(values) == 0:
            summary[
                f"{metric}_mean"
            ] = np.nan

            summary[
                f"{metric}_std"
            ] = np.nan

        else:
            summary[
                f"{metric}_mean"
            ] = float(
                values.mean()
            )

            summary[
                f"{metric}_std"
            ] = float(
                values.std(
                    ddof=1
                )
                if len(values) > 1
                else 0.0
            )

    return {
        "name": name,
        "features": features,
        "folds": fold_results,
        "summary": summary,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 80)
    print("FINGRAPH WALK-FORWARD EVALUATION")
    print("=" * 80)

    df = load_dataset(
        DATASET_PATH
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"First snapshot: "
        f"{df['as_of'].min().isoformat()}"
    )

    print(
        f"Last snapshot: "
        f"{df['as_of'].max().isoformat()}"
    )

    print(
        f"Minimum training size: "
        f"{MIN_TRAIN_SIZE}"
    )

    print(
        f"Test window: "
        f"{TEST_SIZE}"
    )

    print(
        f"Step size: "
        f"{STEP_SIZE}"
    )

    results = []

    for name, features in (
        FEATURE_CONFIGS.items()
    ):

        result = evaluate_configuration(
            df,
            name,
            features,
        )

        results.append(
            result
        )

    # ========================================================
    # COMPARISON
    # ========================================================

    print()
    print("=" * 100)
    print("WALK-FORWARD COMPARISON")
    print("=" * 100)

    comparison_rows = []

    for result in results:

        summary = result[
            "summary"
        ]

        comparison_rows.append(
            {
                "name": result["name"],
                "features": len(
                    result["features"]
                ),
                "accuracy_mean": (
                    summary[
                        "accuracy_mean"
                    ]
                ),
                "accuracy_std": (
                    summary[
                        "accuracy_std"
                    ]
                ),
                "balanced_mean": (
                    summary[
                        "balanced_accuracy_mean"
                    ]
                ),
                "balanced_std": (
                    summary[
                        "balanced_accuracy_std"
                    ]
                ),
                "precision_mean": (
                    summary[
                        "precision_mean"
                    ]
                ),
                "recall_mean": (
                    summary[
                        "recall_mean"
                    ]
                ),
                "f1_mean": (
                    summary["f1_mean"]
                ),
                "f1_std": (
                    summary["f1_std"]
                ),
                "roc_auc_mean": (
                    summary[
                        "roc_auc_mean"
                    ]
                ),
                "roc_auc_std": (
                    summary[
                        "roc_auc_std"
                    ]
                ),
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        160,
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    # ========================================================
    # BEST CONFIGURATION
    # ========================================================

    best_index = (
        comparison[
            "roc_auc_mean"
        ].idxmax()
    )

    best = comparison.loc[
        best_index
    ]

    print()
    print("=" * 80)
    print("BEST WALK-FORWARD CONFIGURATION")
    print("=" * 80)

    print(
        f"Model: {best['name']}"
    )

    print(
        f"Mean ROC-AUC: "
        f"{best['roc_auc_mean']:.4f}"
    )

    print(
        f"ROC-AUC Std: "
        f"{best['roc_auc_std']:.4f}"
    )

    print(
        f"Mean Accuracy: "
        f"{best['accuracy_mean']:.4f}"
    )

    print(
        f"Mean Balanced Accuracy: "
        f"{best['balanced_mean']:.4f}"
    )

    print(
        f"Mean F1: "
        f"{best['f1_mean']:.4f}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_path = Path(
        "data/processed/unified/"
        "aapl_walk_forward_results.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable_results = {
        "dataset": str(
            DATASET_PATH
        ),
        "rows": len(df),
        "min_train_size": (
            MIN_TRAIN_SIZE
        ),
        "test_size": TEST_SIZE,
        "step_size": STEP_SIZE,
        "results": results,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            serializable_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Results saved: {output_path}"
    )

    print()
    print(
        "=" * 80
    )

    print(
        "WALK-FORWARD EVALUATION COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()