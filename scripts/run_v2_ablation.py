from __future__ import annotations

import json
import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.baseline import (
    FEATURE_SETS,
    chronological_split,
    evaluate_baseline,
    prepare_dataset,
    train_baseline,
)


# ============================================================
# DATASET
# ============================================================

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified"
    / "aapl_training_dataset.json"
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    df = prepare_dataset(dataset)

    train_df, test_df = chronological_split(
        df,
        train_ratio=0.8,
    )

    print("=" * 70)
    print("FINGRAPH V3 SEC ABLATION")
    print("=" * 70)

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Train: {len(train_df)}"
    )

    print(
        f"Test: {len(test_df)}"
    )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAIN TARGET")
    print("=" * 70)

    print(
        train_df[
            "target_direction"
        ].value_counts().sort_index()
    )

    print()
    print("=" * 70)
    print("TEST TARGET")
    print("=" * 70)

    print(
        test_df[
            "target_direction"
        ].value_counts().sort_index()
    )

    # --------------------------------------------------------
    # Run feature-set experiments
    # --------------------------------------------------------

    results = []

    for name, features in FEATURE_SETS.items():

        print()
        print("=" * 70)
        print(name)
        print("=" * 70)

        print(
            f"Features: {len(features)}"
        )

        print()
        print("Feature list:")

        for feature in features:
            print(
                f"- {feature}"
            )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        scaler, model = train_baseline(
            train_df,
            features,
        )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        metrics = evaluate_baseline(
            scaler,
            model,
            test_df,
            features,
        )

        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------

        print()
        print("RESULTS")
        print("-" * 40)

        print(
            "Accuracy:",
            round(
                metrics["accuracy"],
                4,
            ),
        )

        print(
            "Precision:",
            round(
                metrics["precision"],
                4,
            ),
        )

        print(
            "Recall:",
            round(
                metrics["recall"],
                4,
            ),
        )

        print(
            "F1:",
            round(
                metrics["f1"],
                4,
            ),
        )

        print(
            "ROC-AUC:",
            round(
                metrics["roc_auc"],
                4,
            )
            if metrics["roc_auc"] is not None
            else None,
        )

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        print()
        print("Confusion Matrix:")

        for row in metrics[
            "confusion_matrix"
        ]:
            print(row)

        # ----------------------------------------------------
        # Probability diagnostics
        # ----------------------------------------------------

        print()
        print("Probability range:")

        print(
            "Min:",
            round(
                metrics["probability_min"],
                4,
            ),
        )

        print(
            "Max:",
            round(
                metrics["probability_max"],
                4,
            ),
        )

        print(
            "Mean:",
            round(
                metrics["probability_mean"],
                4,
            ),
        )

        # ----------------------------------------------------
        # Prediction distribution
        # ----------------------------------------------------

        print()
        print("Predicted distribution:")

        print(
            metrics[
                "prediction_distribution"
            ]
        )

        # ----------------------------------------------------
        # Classification report
        # ----------------------------------------------------

        print()
        print(
            metrics[
                "classification_report"
            ]
        )

        # ----------------------------------------------------
        # Save summary
        # ----------------------------------------------------

        results.append(
            {
                "name": name,
                "features": len(features),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print()
    print("=" * 70)
    print("FINGRAPH V3 ABLATION COMPARISON")
    print("=" * 70)

    import pandas as pd

    comparison = pd.DataFrame(
        results
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("V3 ABLATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()