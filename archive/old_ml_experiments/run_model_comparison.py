from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
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

RESULTS_PATH = Path(
    "data/processed/unified/aapl_model_comparison.json"
)

MIN_TRAIN_SIZE = 140
TEST_WINDOW = 30
STEP_SIZE = 30

TARGET = "target_direction"

# Current best feature family from walk-forward analysis
FEATURES = [
    # Market
    "latest_daily_return",
    "return_5d",
    "return_20d",
    "return_60d",
    "price_vs_20d_ma",
    "price_vs_50d_ma",
    "volatility_20d",

    # SEC V3
    "sec_filing_count",
    "sec_10k_count",
    "sec_10q_count",
    "sec_8k_count",
    "sec_filings_5d",
    "sec_filings_20d",
    "sec_filings_60d",
    "sec_10k_recent",
    "sec_10q_recent",
    "sec_8k_recent",

    # Macro
    "dgs10",
    "dgs3mo",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset() -> pd.DataFrame:

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    df = pd.DataFrame(data)

    if df.empty:
        raise ValueError("Dataset is empty.")

    df["as_of"] = pd.to_datetime(
        df["as_of"],
        utc=True,
    )

    df = df.sort_values(
        "as_of"
    ).reset_index(drop=True)

    missing = [
        column
        for column in FEATURES + [TARGET]
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df


# ============================================================
# MODELS
# ============================================================

def build_models():

    return {

        "LOGISTIC_REGRESSION": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]),

        "RANDOM_FOREST": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=5,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]),

        "GRADIENT_BOOSTING": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=2,
                    random_state=42,
                ),
            ),
        ]),

        "HIST_GRADIENT_BOOSTING": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=150,
                    learning_rate=0.05,
                    max_leaf_nodes=7,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]),
    }


# ============================================================
# WALK-FORWARD SPLITS
# ============================================================

def generate_splits(
    n_rows: int,
):

    start = MIN_TRAIN_SIZE

    while start + TEST_WINDOW <= n_rows:

        train_end = start
        test_end = start + TEST_WINDOW

        yield (
            np.arange(0, train_end),
            np.arange(train_end, test_end),
        )

        start += STEP_SIZE


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    name: str,
    model,
    df: pd.DataFrame,
):

    X = df[FEATURES]
    y = df[TARGET]

    fold_results = []

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    for fold, (train_idx, test_idx) in enumerate(
        generate_splits(len(df)),
        start=1,
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

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

        # ROC-AUC requires both classes
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(
                y_test,
                probabilities,
            )
        else:
            auc = np.nan

        result = {
            "fold": fold,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "accuracy": accuracy_score(
                y_test,
                predictions,
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_test,
                predictions,
            ),
            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "roc_auc": auc,
            "brier_score": brier_score_loss(
                y_test,
                probabilities,
            ),
        }

        fold_results.append(result)

        print(
            f"Fold {fold:02d} | "
            f"Train={len(train_idx)} | "
            f"Test={len(test_idx)} | "
            f"Accuracy={result['accuracy']:.4f} | "
            f"Balanced={result['balanced_accuracy']:.4f} | "
            f"F1={result['f1']:.4f} | "
            f"ROC-AUC={result['roc_auc']:.4f} | "
            f"Brier={result['brier_score']:.4f}"
        )

    summary = {}

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "brier_score",
    ]

    for metric in metric_names:

        values = np.array([
            fold[metric]
            for fold in fold_results
            if not np.isnan(fold[metric])
        ])

        summary[f"{metric}_mean"] = float(
            values.mean()
        )

        summary[f"{metric}_std"] = float(
            values.std(ddof=1)
        )

    return {
        "model": name,
        "features": FEATURES,
        "folds": fold_results,
        "summary": summary,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FINGRAPH V3 MODEL COMPARISON")
    print("=" * 80)

    df = load_dataset()

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    print(
        f"First snapshot: "
        f"{df['as_of'].min().isoformat()}"
    )

    print(
        f"Last snapshot: "
        f"{df['as_of'].max().isoformat()}"
    )

    models = build_models()

    results = []

    for name, model in models.items():

        result = evaluate_model(
            name,
            model,
            df,
        )

        results.append(result)

    # ========================================================
    # COMPARISON
    # ========================================================

    comparison = []

    for result in results:

        summary = result["summary"]

        comparison.append({
            "model": result["model"],
            "accuracy_mean": summary[
                "accuracy_mean"
            ],
            "accuracy_std": summary[
                "accuracy_std"
            ],
            "balanced_accuracy_mean": summary[
                "balanced_accuracy_mean"
            ],
            "f1_mean": summary[
                "f1_mean"
            ],
            "roc_auc_mean": summary[
                "roc_auc_mean"
            ],
            "roc_auc_std": summary[
                "roc_auc_std"
            ],
            "brier_score_mean": summary[
                "brier_score_mean"
            ],
        })

    comparison_df = pd.DataFrame(
        comparison
    )

    comparison_df = comparison_df.sort_values(
        "roc_auc_mean",
        ascending=False,
    )

    print()
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    output = {
        "configuration": {
            "dataset": str(DATASET_PATH),
            "minimum_training_size": MIN_TRAIN_SIZE,
            "test_window": TEST_WINDOW,
            "step_size": STEP_SIZE,
            "feature_count": len(FEATURES),
            "features": FEATURES,
        },
        "comparison": comparison,
        "models": results,
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print()
    print(
        f"Results saved: {RESULTS_PATH}"
    )

    print()
    print(
        "FINGRAPH V3 MODEL COMPARISON COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()