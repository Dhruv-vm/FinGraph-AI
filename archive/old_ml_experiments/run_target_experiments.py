from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
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

OUTPUT_PATH = Path(
    "data/processed/unified/aapl_target_experiments.json"
)

MIN_TRAIN_SIZE = 140
TEST_WINDOW = 30
STEP_SIZE = 30


# ============================================================
# DATASET CONFIGURATION
# ============================================================

METADATA_COLUMNS = {
    "entity_id",
    "ticker",
    "as_of",
}

EXCLUDED_TARGET_COLUMNS = {
    "target_return",
    "target_direction",
}


# ============================================================
# LOAD DATASET
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

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    required = {
        "as_of",
        "target_return",
    }

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["as_of"] = pd.to_datetime(
        df["as_of"],
        errors="coerce",
        utc=True,
    )

    df["target_return"] = pd.to_numeric(
        df["target_return"],
        errors="coerce",
    )

    if df["as_of"].isna().any():
        raise ValueError(
            "Invalid as_of values detected."
        )

    if df["target_return"].isna().any():
        raise ValueError(
            "Invalid target_return values detected."
        )

    df = df.sort_values(
        "as_of"
    ).reset_index(drop=True)

    return df


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:

    excluded = (
        METADATA_COLUMNS
        | EXCLUDED_TARGET_COLUMNS
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    numeric_features = []

    for column in feature_columns:

        converted = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if (
            converted.notna().sum()
            == df[column].notna().sum()
        ):
            df[column] = converted
            numeric_features.append(column)

    if not numeric_features:
        raise ValueError(
            "No numeric ML features found."
        )

    return numeric_features


# ============================================================
# TARGET GENERATION
# ============================================================

def create_target(
    returns: pd.Series,
    threshold: float,
) -> pd.Series:
    """
    Binary target.

    1 = future return > threshold
    0 = future return <= threshold
    """

    return (
        returns > threshold
    ).astype(int)


def create_three_class_target(
    returns: pd.Series,
    threshold: float,
) -> pd.Series:
    """
    Three-class target.

    0 = DOWN
    1 = NEUTRAL
    2 = UP

    DOWN:
        return < -threshold

    NEUTRAL:
        -threshold <= return <= threshold

    UP:
        return > threshold
    """

    target = pd.Series(
        np.nan,
        index=returns.index,
    )

    target[
        returns < -threshold
    ] = 0

    target[
        (returns >= -threshold)
        & (returns <= threshold)
    ] = 1

    target[
        returns > threshold
    ] = 2

    return target.astype(int)


# ============================================================
# WALK-FORWARD FOLDS
# ============================================================

def generate_folds(
    n_rows: int,
) -> list[tuple[int, int, int]]:

    folds = []

    train_end = MIN_TRAIN_SIZE

    while (
        train_end + TEST_WINDOW
        <= n_rows
    ):

        test_start = train_end
        test_end = (
            test_start
            + TEST_WINDOW
        )

        folds.append(
            (
                train_end,
                test_start,
                test_end,
            )
        )

        train_end += STEP_SIZE

    return folds


# ============================================================
# MODEL
# ============================================================

def build_model() -> Pipeline:
    """
    Use Logistic Regression as the controlled
    baseline model.

    Scaling and imputation are fitted independently
    inside every walk-forward training fold.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def calculate_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | None]:

    metrics: dict[
        str,
        float | None,
    ] = {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "roc_auc": None,
    }

    if (
        len(np.unique(y_true))
        == 2
    ):
        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

    return metrics


# ============================================================
# SINGLE EXPERIMENT
# ============================================================

def run_binary_experiment(
    df: pd.DataFrame,
    feature_columns: list[str],
    name: str,
    threshold: float,
    horizon_label: str,
) -> dict[str, Any]:

    working = df.copy()

    working["experiment_target"] = (
        create_target(
            working["target_return"],
            threshold,
        )
    )

    folds = generate_folds(
        len(working)
    )

    fold_results = []

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"Horizon       : {horizon_label}"
    )

    print(
        f"Threshold     : {threshold:.4f}"
    )

    print(
        f"Folds         : {len(folds)}"
    )

    print()

    for fold_number, (
        train_end,
        test_start,
        test_end,
    ) in enumerate(
        folds,
        start=1,
    ):

        train_df = working.iloc[
            :train_end
        ]

        test_df = working.iloc[
            test_start:test_end
        ]

        X_train = train_df[
            feature_columns
        ]

        X_test = test_df[
            feature_columns
        ]

        y_train = train_df[
            "experiment_target"
        ].astype(int)

        y_test = test_df[
            "experiment_target"
        ].astype(int)

        # ----------------------------------------------------
        # Skip folds with only one training class
        # ----------------------------------------------------

        if y_train.nunique() < 2:
            print(
                f"Fold {fold_number:02d} "
                f"| SKIPPED "
                f"| training target has one class"
            )
            continue

        model = build_model()

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        metrics = calculate_binary_metrics(
            y_test.to_numpy(),
            predictions,
            probabilities,
        )

        result = {
            "fold": fold_number,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics[
                "balanced_accuracy"
            ],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "positive_rate": float(
                y_test.mean()
            ),
            "predicted_positive_rate": float(
                predictions.mean()
            ),
        }

        fold_results.append(
            result
        )

        roc_auc_text = (
            f"{metrics['roc_auc']:.4f}"
            if metrics["roc_auc"]
            is not None
            else "N/A"
        )

        print(
            f"Fold {fold_number:02d} "
            f"| Train={len(train_df):3d} "
            f"| Test={len(test_df):2d} "
            f"| Accuracy={metrics['accuracy']:.4f} "
            f"| Balanced={metrics['balanced_accuracy']:.4f} "
            f"| F1={metrics['f1']:.4f} "
            f"| ROC-AUC={roc_auc_text}"
        )

    return summarize_experiment(
        name=name,
        horizon=horizon_label,
        threshold=threshold,
        fold_results=fold_results,
    )


# ============================================================
# EXPERIMENT SUMMARY
# ============================================================

def summarize_experiment(
    name: str,
    horizon: str,
    threshold: float,
    fold_results: list[dict[str, Any]],
) -> dict[str, Any]:

    if not fold_results:
        return {
            "name": name,
            "horizon": horizon,
            "threshold": threshold,
            "folds": 0,
        }

    def mean_metric(
        key: str,
    ) -> float:

        values = [
            result[key]
            for result in fold_results
            if result[key] is not None
        ]

        if not values:
            return float("nan")

        return float(
            np.mean(values)
        )

    def std_metric(
        key: str,
    ) -> float:

        values = [
            result[key]
            for result in fold_results
            if result[key] is not None
        ]

        if not values:
            return float("nan")

        return float(
            np.std(values)
        )

    return {
        "name": name,
        "horizon": horizon,
        "threshold": threshold,
        "folds": len(fold_results),
        "accuracy_mean": mean_metric(
            "accuracy"
        ),
        "accuracy_std": std_metric(
            "accuracy"
        ),
        "balanced_accuracy_mean": mean_metric(
            "balanced_accuracy"
        ),
        "balanced_accuracy_std": std_metric(
            "balanced_accuracy"
        ),
        "f1_mean": mean_metric(
            "f1"
        ),
        "f1_std": std_metric(
            "f1"
        ),
        "roc_auc_mean": mean_metric(
            "roc_auc"
        ),
        "roc_auc_std": std_metric(
            "roc_auc"
        ),
        "positive_rate_mean": mean_metric(
            "positive_rate"
        ),
        "predicted_positive_rate_mean": mean_metric(
            "predicted_positive_rate"
        ),
        "fold_results": fold_results,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 80)
    print("FINGRAPH TARGET / HORIZON EXPERIMENTS")
    print("=" * 80)

    df = load_dataset(
        DATASET_PATH
    )

    feature_columns = get_feature_columns(
        df
    )

    print(
        f"Rows          : {len(df)}"
    )

    print(
        f"Features      : {len(feature_columns)}"
    )

    print(
        f"First snapshot: "
        f"{df['as_of'].min().isoformat()}"
    )

    print(
        f"Last snapshot : "
        f"{df['as_of'].max().isoformat()}"
    )

    print(
        f"Minimum train : {MIN_TRAIN_SIZE}"
    )

    print(
        f"Test window   : {TEST_WINDOW}"
    )

    print(
        f"Step size     : {STEP_SIZE}"
    )

    experiments = []

    # ========================================================
    # E1 — EXISTING BASELINE
    # ========================================================

    experiments.append(
        run_binary_experiment(
            df=df,
            feature_columns=feature_columns,
            name="E1: 5-DAY ZERO THRESHOLD",
            threshold=0.0,
            horizon_label="5-day",
        )
    )

    # ========================================================
    # E2 — ±1% THRESHOLD
    # ========================================================

    experiments.append(
        run_binary_experiment(
            df=df,
            feature_columns=feature_columns,
            name="E2: 5-DAY ±1% THRESHOLD",
            threshold=0.01,
            horizon_label="5-day",
        )
    )

    # ========================================================
    # E3 — ±2% THRESHOLD
    # ========================================================

    experiments.append(
        run_binary_experiment(
            df=df,
            feature_columns=feature_columns,
            name="E3: 5-DAY ±2% THRESHOLD",
            threshold=0.02,
            horizon_label="5-day",
        )
    )

    # ========================================================
    # NOTE ON HORIZONS
    # ========================================================
    #
    # The current dataset contains target_return generated
    # using the existing horizon in builder.py.
    #
    # Therefore we DO NOT fabricate 10-day or 20-day targets
    # from the existing 5-day target.
    #
    # Those experiments require rebuilding the dataset with
    # horizon_days=10 and horizon_days=20.
    #
    # This preserves experimental validity.
    # ========================================================

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print()
    print("=" * 100)
    print("TARGET EXPERIMENT COMPARISON")
    print("=" * 100)

    comparison_rows = []

    for experiment in experiments:

        comparison_rows.append(
            {
                "experiment": experiment[
                    "name"
                ],
                "horizon": experiment[
                    "horizon"
                ],
                "threshold": experiment[
                    "threshold"
                ],
                "folds": experiment[
                    "folds"
                ],
                "accuracy_mean": experiment.get(
                    "accuracy_mean"
                ),
                "balanced_accuracy_mean": experiment.get(
                    "balanced_accuracy_mean"
                ),
                "f1_mean": experiment.get(
                    "f1_mean"
                ),
                "roc_auc_mean": experiment.get(
                    "roc_auc_mean"
                ),
                "roc_auc_std": experiment.get(
                    "roc_auc_std"
                ),
                "positive_rate_mean": experiment.get(
                    "positive_rate_mean"
                ),
                "predicted_positive_rate_mean": experiment.get(
                    "predicted_positive_rate_mean"
                ),
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
                if pd.notna(x)
                else "N/A"
            ),
        )
    )

    # ========================================================
    # BEST EXPERIMENT
    # ========================================================

    valid_experiments = [
        experiment
        for experiment in experiments
        if experiment.get(
            "roc_auc_mean"
        ) is not None
        and not np.isnan(
            experiment[
                "roc_auc_mean"
            ]
        )
    ]

    if valid_experiments:

        best = max(
            valid_experiments,
            key=lambda experiment:
                experiment[
                    "roc_auc_mean"
                ],
        )

        print()
        print(
            "=" * 80
        )
        print(
            "BEST TARGET CONFIGURATION"
        )
        print(
            "=" * 80
        )

        print(
            f"Experiment    : "
            f"{best['name']}"
        )

        print(
            f"Horizon       : "
            f"{best['horizon']}"
        )

        print(
            f"Threshold     : "
            f"{best['threshold']}"
        )

        print(
            f"Mean ROC-AUC   : "
            f"{best['roc_auc_mean']:.4f}"
        )

        print(
            f"Mean Accuracy  : "
            f"{best['accuracy_mean']:.4f}"
        )

        print(
            f"Mean Balanced  : "
            f"{best['balanced_accuracy_mean']:.4f}"
        )

        print(
            f"Mean F1        : "
            f"{best['f1_mean']:.4f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "dataset": str(
            DATASET_PATH
        ),
        "rows": len(df),
        "features": feature_columns,
        "minimum_train_size": MIN_TRAIN_SIZE,
        "test_window": TEST_WINDOW,
        "step_size": STEP_SIZE,
        "experiments": experiments,
        "note": (
            "10-day and 20-day experiments "
            "require datasets rebuilt with "
            "horizon_days=10 and horizon_days=20. "
            "They are intentionally not fabricated "
            "from the existing 5-day target."
        ),
    }

    with OUTPUT_PATH.open(
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
        f"Results saved: {OUTPUT_PATH}"
    )

    print()
    print(
        "=" * 80
    )
    print(
        "TARGET / HORIZON EXPERIMENTS COMPLETE"
    )
    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()