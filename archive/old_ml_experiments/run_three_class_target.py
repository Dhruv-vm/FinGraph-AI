from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
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

RESULT_PATH = Path(
    "data/processed/unified/aapl_three_class_target.json"
)

TARGET_RETURN = "target_return"

HORIZON_DAYS = 5
THRESHOLD = 0.01

MIN_TRAIN_SIZE = 140
TEST_WINDOW = 30
STEP_SIZE = 30

# Use the same 19-feature configuration used in V3.
FEATURE_COLUMNS = [
    # Market
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
    "sec_filings_5d",
    "sec_filings_20d",
    "sec_filings_60d",
    "sec_10k_recent",
    "sec_10q_recent",
    "sec_8k_recent",
    "sec_filing_velocity",
    "sec_filing_acceleration",
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

    if not isinstance(data, list):
        raise ValueError(
            "Expected dataset to be a list."
        )

    df = pd.DataFrame(data)

    if TARGET_RETURN not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET_RETURN}"
        )

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns: {missing_features}"
        )

    df["as_of"] = pd.to_datetime(
        df["as_of"],
        utc=True,
    )

    df = df.sort_values(
        "as_of"
    ).reset_index(drop=True)

    return df


# ============================================================
# CREATE THREE-CLASS TARGET
# ============================================================

def create_three_class_target(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    returns = pd.to_numeric(
        result[TARGET_RETURN],
        errors="coerce",
    )

    if returns.isna().any():
        raise ValueError(
            "target_return contains missing/non-numeric values."
        )

    # --------------------------------------------------------
    # Class definition
    #
    # 0 = DOWN
    # 1 = NEUTRAL
    # 2 = UP
    # --------------------------------------------------------

    result["target_3class"] = np.select(
        [
            returns < -THRESHOLD,
            returns > THRESHOLD,
        ],
        [
            0,
            2,
        ],
        default=1,
    )

    return result


# ============================================================
# MODEL
# ============================================================

def build_model() -> Pipeline:

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
                "model",
                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=2,
                    random_state=42,
                ),
            ),
        ]
    )


# ============================================================
# WALK-FORWARD EVALUATION
# ============================================================

def run_walk_forward(
    df: pd.DataFrame,
) -> dict:

    X = df[FEATURE_COLUMNS].copy()
    y = df["target_3class"].astype(int)

    fold_results = []

    print("=" * 80)
    print("FINGRAPH THREE-CLASS TARGET EVALUATION")
    print("=" * 80)

    print(
        f"Rows          : {len(df)}"
    )

    print(
        f"Features      : {len(FEATURE_COLUMNS)}"
    )

    print(
        f"First snapshot: {df['as_of'].min().isoformat()}"
    )

    print(
        f"Last snapshot : {df['as_of'].max().isoformat()}"
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

    print()
    print("=" * 80)
    print("THREE-CLASS TARGET DISTRIBUTION")
    print("=" * 80)

    distribution = (
        y.value_counts()
        .sort_index()
    )

    class_names = {
        0: "DOWN",
        1: "NEUTRAL",
        2: "UP",
    }

    for class_id, count in distribution.items():

        percentage = (
            count / len(y) * 100
        )

        print(
            f"{class_names[class_id]:<10}"
            f"{count:>6}"
            f" ({percentage:>6.2f}%)"
        )

    # --------------------------------------------------------
    # Walk-forward folds
    # --------------------------------------------------------

    start = MIN_TRAIN_SIZE
    fold_number = 1

    all_true = []
    all_pred = []
    all_prob = []

    while (
        start + TEST_WINDOW
        <= len(df)
    ):

        train_start = 0
        train_end = start

        test_start = start
        test_end = (
            start + TEST_WINDOW
        )

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
        )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        balanced = balanced_accuracy_score(
            y_test,
            predictions,
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        # ----------------------------------------------------
        # Per-class F1
        # ----------------------------------------------------

        class_f1 = f1_score(
            y_test,
            predictions,
            labels=[0, 1, 2],
            average=None,
            zero_division=0,
        )

        # ----------------------------------------------------
        # ROC-AUC
        #
        # Only calculate if all three classes are present
        # in the test fold.
        # ----------------------------------------------------

        if len(
            np.unique(y_test)
        ) == 3:

            roc_auc = roc_auc_score(
                y_test,
                probabilities,
                multi_class="ovr",
                average="macro",
                labels=[0, 1, 2],
            )

        else:

            roc_auc = np.nan

        # ----------------------------------------------------
        # Store pooled predictions
        # ----------------------------------------------------

        all_true.extend(
            y_test.tolist()
        )

        all_pred.extend(
            predictions.tolist()
        )

        all_prob.extend(
            probabilities.tolist()
        )

        fold_result = {
            "fold": fold_number,
            "train_size": int(
                len(X_train)
            ),
            "test_size": int(
                len(X_test)
            ),
            "accuracy": float(
                accuracy
            ),
            "balanced_accuracy": float(
                balanced
            ),
            "macro_f1": float(
                macro_f1
            ),
            "weighted_f1": float(
                weighted_f1
            ),
            "down_f1": float(
                class_f1[0]
            ),
            "neutral_f1": float(
                class_f1[1]
            ),
            "up_f1": float(
                class_f1[2]
            ),
            "roc_auc": (
                None
                if np.isnan(roc_auc)
                else float(roc_auc)
            ),
            "test_start": df.iloc[
                test_start
            ]["as_of"].isoformat(),
            "test_end": df.iloc[
                test_end - 1
            ]["as_of"].isoformat(),
        }

        fold_results.append(
            fold_result
        )

        roc_text = (
            "N/A"
            if np.isnan(roc_auc)
            else f"{roc_auc:.4f}"
        )

        print(
            f"Fold {fold_number:02d} | "
            f"Train={len(X_train):>3} | "
            f"Test={len(X_test):>2} | "
            f"Accuracy={accuracy:.4f} | "
            f"Balanced={balanced:.4f} | "
            f"Macro-F1={macro_f1:.4f} | "
            f"ROC-AUC={roc_text}"
        )

        fold_number += 1
        start += STEP_SIZE

    # ========================================================
    # AGGREGATE RESULTS
    # ========================================================

    fold_df = pd.DataFrame(
        fold_results
    )

    print()
    print("=" * 80)
    print("THREE-CLASS WALK-FORWARD SUMMARY")
    print("=" * 80)

    print(
        f"Accuracy mean          : "
        f"{fold_df['accuracy'].mean():.4f}"
    )

    print(
        f"Accuracy std           : "
        f"{fold_df['accuracy'].std():.4f}"
    )

    print(
        f"Balanced accuracy mean : "
        f"{fold_df['balanced_accuracy'].mean():.4f}"
    )

    print(
        f"Balanced accuracy std  : "
        f"{fold_df['balanced_accuracy'].std():.4f}"
    )

    print(
        f"Macro-F1 mean          : "
        f"{fold_df['macro_f1'].mean():.4f}"
    )

    print(
        f"Macro-F1 std           : "
        f"{fold_df['macro_f1'].std():.4f}"
    )

    print(
        f"Weighted-F1 mean       : "
        f"{fold_df['weighted_f1'].mean():.4f}"
    )

    valid_auc = fold_df[
        "roc_auc"
    ].dropna()

    if len(valid_auc):

        print(
            f"ROC-AUC mean           : "
            f"{valid_auc.mean():.4f}"
        )

        print(
            f"ROC-AUC std            : "
            f"{valid_auc.std():.4f}"
        )

    else:

        print(
            "ROC-AUC mean           : N/A"
        )

    # ========================================================
    # POOLED CONFUSION MATRIX
    # ========================================================

    pooled_matrix = confusion_matrix(
        all_true,
        all_pred,
        labels=[0, 1, 2],
    )

    print()
    print("=" * 80)
    print("POOLED CONFUSION MATRIX")
    print("=" * 80)

    print(
        "             Pred DOWN  Pred NEUTRAL  Pred UP"
    )

    for index, name in enumerate(
        ["DOWN", "NEUTRAL", "UP"]
    ):

        print(
            f"Actual {name:<7} "
            f"{pooled_matrix[index, 0]:>10} "
            f"{pooled_matrix[index, 1]:>13} "
            f"{pooled_matrix[index, 2]:>9}"
        )

    # ========================================================
    # POOLED CLASSIFICATION REPORT
    # ========================================================

    print()
    print("=" * 80)
    print("POOLED CLASSIFICATION REPORT")
    print("=" * 80)

    print(
        classification_report(
            all_true,
            all_pred,
            labels=[0, 1, 2],
            target_names=[
                "DOWN",
                "NEUTRAL",
                "UP",
            ],
            zero_division=0,
        )
    )

    # ========================================================
    # PREDICTION DISTRIBUTION
    # ========================================================

    prediction_distribution = (
        pd.Series(all_pred)
        .value_counts()
        .sort_index()
    )

    print(
        "PREDICTED CLASS DISTRIBUTION"
    )

    for class_id in [0, 1, 2]:

        count = int(
            prediction_distribution.get(
                class_id,
                0,
            )
        )

        percentage = (
            count / len(all_pred) * 100
        )

        print(
            f"{class_names[class_id]:<10}"
            f"{count:>6}"
            f" ({percentage:>6.2f}%)"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output = {
        "experiment": "three_class_target",
        "horizon_days": HORIZON_DAYS,
        "threshold": THRESHOLD,
        "classes": {
            "0": "DOWN",
            "1": "NEUTRAL",
            "2": "UP",
        },
        "definition": {
            "DOWN": f"target_return < -{THRESHOLD}",
            "NEUTRAL": (
                f"-{THRESHOLD} <= target_return "
                f"<= {THRESHOLD}"
            ),
            "UP": f"target_return > {THRESHOLD}",
        },
        "dataset": {
            "rows": len(df),
            "features": FEATURE_COLUMNS,
            "first_snapshot": df[
                "as_of"
            ].min().isoformat(),
            "last_snapshot": df[
                "as_of"
            ].max().isoformat(),
        },
        "configuration": {
            "minimum_train_size": MIN_TRAIN_SIZE,
            "test_window": TEST_WINDOW,
            "step_size": STEP_SIZE,
            "model": "GradientBoostingClassifier",
        },
        "target_distribution": {
            class_names[class_id]: int(
                distribution.get(
                    class_id,
                    0,
                )
            )
            for class_id in [0, 1, 2]
        },
        "summary": {
            "accuracy_mean": float(
                fold_df["accuracy"].mean()
            ),
            "accuracy_std": float(
                fold_df["accuracy"].std()
            ),
            "balanced_accuracy_mean": float(
                fold_df[
                    "balanced_accuracy"
                ].mean()
            ),
            "balanced_accuracy_std": float(
                fold_df[
                    "balanced_accuracy"
                ].std()
            ),
            "macro_f1_mean": float(
                fold_df["macro_f1"].mean()
            ),
            "macro_f1_std": float(
                fold_df["macro_f1"].std()
            ),
            "weighted_f1_mean": float(
                fold_df["weighted_f1"].mean()
            ),
            "roc_auc_mean": (
                None
                if valid_auc.empty
                else float(
                    valid_auc.mean()
                )
            ),
            "roc_auc_std": (
                None
                if valid_auc.empty
                else float(
                    valid_auc.std()
                )
            ),
        },
        "pooled_confusion_matrix":
            pooled_matrix.tolist(),
        "folds": fold_results,
    }

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULT_PATH.open(
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
        f"Results saved: {RESULT_PATH}"
    )

    print()
    print("=" * 80)
    print(
        "THREE-CLASS TARGET EVALUATION COMPLETE"
    )
    print("=" * 80)

    return output


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    df = load_dataset()

    df = create_three_class_target(
        df
    )

    run_walk_forward(
        df
    )


if __name__ == "__main__":
    main()