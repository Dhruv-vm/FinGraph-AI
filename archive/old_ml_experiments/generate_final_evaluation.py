#!/usr/bin/env python3

"""
FinGraph AI - Final Evaluation Summary

Aggregates the completed experimental results into one
research-ready JSON summary.

Inputs:
    aapl_walk_forward_results.json
    aapl_model_comparison.json
    aapl_target_experiments.json
    aapl_three_class_target.json
    aapl_statistical_tests.json

Output:
    aapl_final_evaluation.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BASE = Path("data/processed/unified")

FILES = {
    "walk_forward": BASE / "aapl_walk_forward_results.json",
    "models": BASE / "aapl_model_comparison.json",
    "targets": BASE / "aapl_target_experiments.json",
    "three_class": BASE / "aapl_three_class_target.json",
    "statistics": BASE / "aapl_statistical_tests.json",
}

OUTPUT = BASE / "aapl_final_evaluation.json"


# ============================================================
# HELPERS
# ============================================================


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def r(value: Any, digits: int = 4):
    if value is None:
        return None

    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


# ============================================================
# WALK-FORWARD SUMMARY
# ============================================================


def summarize_walk_forward(
    data: dict[str, Any],
) -> dict[str, Any]:

    results = data.get("results", [])

    configurations = []

    for item in results:

        configurations.append(
            {
                "name": item.get("name"),
                "feature_count": len(
                    item.get("features", [])
                ),
                "accuracy_mean": r(
                    item.get("accuracy_mean")
                ),
                "balanced_accuracy_mean": r(
                    item.get(
                        "balanced_accuracy_mean"
                    )
                ),
                "f1_mean": r(
                    item.get("f1_mean")
                ),
                "roc_auc_mean": r(
                    item.get("roc_auc_mean")
                ),
                "accuracy_std": r(
                    item.get("accuracy_std")
                ),
                "balanced_accuracy_std": r(
                    item.get(
                        "balanced_accuracy_std"
                    )
                ),
                "f1_std": r(
                    item.get("f1_std")
                ),
                "roc_auc_std": r(
                    item.get("roc_auc_std")
                ),
            }
        )

    return {
        "configuration_count": len(
            configurations
        ),
        "configurations": configurations,
    }


# ============================================================
# MODEL SUMMARY
# ============================================================


def summarize_models(
    data: dict[str, Any],
) -> dict[str, Any]:

    comparison = data.get(
        "comparison",
        []
    )

    models = []

    for item in comparison:

        models.append(
            {
                "model": item.get("model"),
                "accuracy_mean": r(
                    item.get("accuracy_mean")
                ),
                "accuracy_std": r(
                    item.get("accuracy_std")
                ),
                "balanced_accuracy_mean": r(
                    item.get(
                        "balanced_accuracy_mean"
                    )
                ),
                "f1_mean": r(
                    item.get("f1_mean")
                ),
                "roc_auc_mean": r(
                    item.get("roc_auc_mean")
                ),
                "roc_auc_std": r(
                    item.get("roc_auc_std")
                ),
                "brier_score_mean": r(
                    item.get(
                        "brier_score_mean"
                    )
                ),
            }
        )

    return {
        "model_count": len(models),
        "models": models,
    }


# ============================================================
# TARGET SUMMARY
# ============================================================


def summarize_targets(
    data: dict[str, Any],
) -> dict[str, Any]:

    experiments = data.get(
        "experiments",
        []
    )

    targets = []

    for item in experiments:

        targets.append(
            {
                "name": item.get("name"),
                "horizon": item.get(
                    "horizon"
                ),
                "threshold": item.get(
                    "threshold"
                ),
                "folds": item.get(
                    "folds"
                ),
                "accuracy_mean": r(
                    item.get(
                        "accuracy_mean"
                    )
                ),
                "balanced_accuracy_mean": r(
                    item.get(
                        "balanced_accuracy_mean"
                    )
                ),
                "f1_mean": r(
                    item.get("f1_mean")
                ),
                "roc_auc_mean": r(
                    item.get(
                        "roc_auc_mean"
                    )
                ),
                "accuracy_std": r(
                    item.get(
                        "accuracy_std"
                    )
                ),
                "balanced_accuracy_std": r(
                    item.get(
                        "balanced_accuracy_std"
                    )
                ),
                "f1_std": r(
                    item.get("f1_std")
                ),
                "roc_auc_std": r(
                    item.get(
                        "roc_auc_std"
                    )
                ),
            }
        )

    # Metric-specific winners.
    winners = {}

    if targets:

        for metric in [
            "accuracy_mean",
            "balanced_accuracy_mean",
            "f1_mean",
            "roc_auc_mean",
        ]:

            valid = [
                item
                for item in targets
                if item.get(metric)
                is not None
            ]

            if valid:

                best = max(
                    valid,
                    key=lambda x: x[metric],
                )

                winners[metric] = {
                    "experiment": best[
                        "name"
                    ],
                    "value": best[
                        metric
                    ],
                }

    return {
        "experiment_count": len(
            targets
        ),
        "experiments": targets,
        "metric_winners": winners,
    }


# ============================================================
# THREE-CLASS SUMMARY
# ============================================================


def summarize_three_class(
    data: dict[str, Any],
) -> dict[str, Any]:

    # Keep the summary intentionally
    # close to the generated result file.

    summary = {}

    possible_keys = [
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "weighted_f1_mean",
        "weighted_f1_std",
        "roc_auc_mean",
        "roc_auc_std",
    ]

    for key in possible_keys:

        if key in data:
            summary[key] = r(
                data[key]
            )

    if "class_distribution" in data:
        summary[
            "class_distribution"
        ] = data[
            "class_distribution"
        ]

    if "pooled_confusion_matrix" in data:
        summary[
            "pooled_confusion_matrix"
        ] = data[
            "pooled_confusion_matrix"
        ]

    if "classification_report" in data:
        summary[
            "classification_report"
        ] = data[
            "classification_report"
        ]

    # Some versions may store the summary
    # in nested form.

    if "summary" in data:
        summary["generated_summary"] = data[
            "summary"
        ]

    return summary


# ============================================================
# STATISTICAL SUMMARY
# ============================================================


def summarize_statistics(
    data: dict[str, Any],
) -> dict[str, Any]:

    summary = data.get(
        "summary",
        {},
    )

    target_tests = (
        data.get(
            "target_experiments",
            {},
        )
        .get(
            "friedman",
            []
        )
    )

    model_tests = (
        data.get(
            "models",
            {},
        )
        .get(
            "friedman",
            []
        )
    )

    walk_tests = (
        data.get(
            "walk_forward",
            {},
        )
        .get(
            "friedman",
            []
        )
    )

    return {
        "total_pairwise_comparisons": summary.get(
            "total_pairwise_comparisons"
        ),
        "total_friedman_tests": summary.get(
            "total_friedman_tests"
        ),
        "significant_pairwise_after_holm": summary.get(
            "significant_pairwise_after_holm"
        ),
        "walk_forward_friedman": walk_tests,
        "model_friedman": model_tests,
        "target_friedman": target_tests,
    }


# ============================================================
# RESEARCH FINDINGS
# ============================================================


def derive_findings(
    walk: dict[str, Any],
    models: dict[str, Any],
    targets: dict[str, Any],
    statistics: dict[str, Any],
) -> dict[str, Any]:

    findings = []

    # --------------------------------------------------------
    # Best model by metrics
    # --------------------------------------------------------

    model_list = models["models"]

    if model_list:

        best_f1 = max(
            model_list,
            key=lambda x: x["f1_mean"]
            if x["f1_mean"] is not None
            else float("-inf"),
        )

        best_auc = max(
            model_list,
            key=lambda x: x["roc_auc_mean"]
            if x["roc_auc_mean"] is not None
            else float("-inf"),
        )

        best_accuracy = max(
            model_list,
            key=lambda x: x["accuracy_mean"]
            if x["accuracy_mean"] is not None
            else float("-inf"),
        )

        findings.append(
            "Gradient/model performance should "
            "be interpreted using multiple metrics "
            "rather than a single score."
        )

        findings.append(
            f'Highest model F1: '
            f'{best_f1["model"]} '
            f'({best_f1["f1_mean"]:.4f}).'
        )

        findings.append(
            f'Highest model ROC-AUC: '
            f'{best_auc["model"]} '
            f'({best_auc["roc_auc_mean"]:.4f}).'
        )

        findings.append(
            f'Highest model accuracy: '
            f'{best_accuracy["model"]} '
            f'({best_accuracy["accuracy_mean"]:.4f}).'
        )

    # --------------------------------------------------------
    # Target experiments
    # --------------------------------------------------------

    target_friedman = (
        statistics[
            "target_friedman"
        ]
    )

    for test in target_friedman:

        metric = test.get(
            "metric"
        )

        p = test.get(
            "p_value"
        )

        if (
            metric == "f1"
            and p is not None
            and p < 0.05
        ):

            findings.append(
                "Target definition has a "
                "statistically significant "
                f"effect on F1 "
                f"(Friedman p={p:.6f})."
            )

        elif (
            metric == "balanced_accuracy"
            and p is not None
            and p < 0.05
        ):

            findings.append(
                "Target definition has a "
                "statistically significant "
                "effect on balanced accuracy."
            )

    # --------------------------------------------------------
    # Holm result
    # --------------------------------------------------------

    significant = statistics.get(
        "significant_pairwise_after_holm"
    )

    if significant == 0:

        findings.append(
            "No pairwise comparison remained "
            "significant after Holm correction "
            "at alpha=0.05."
        )

    return {
        "findings": findings
    }


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print("=" * 80)
    print(
        "FINGRAPH AI FINAL EVALUATION SUMMARY"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    walk_data = load_json(
        FILES["walk_forward"]
    )

    model_data = load_json(
        FILES["models"]
    )

    target_data = load_json(
        FILES["targets"]
    )

    three_class_data = load_json(
        FILES["three_class"]
    )

    statistics_data = load_json(
        FILES["statistics"]
    )

    print()
    print("Input files loaded successfully.")

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    walk = summarize_walk_forward(
        walk_data
    )

    models = summarize_models(
        model_data
    )

    targets = summarize_targets(
        target_data
    )

    three_class = summarize_three_class(
        three_class_data
    )

    statistics = summarize_statistics(
        statistics_data
    )

    findings = derive_findings(
        walk,
        models,
        targets,
        statistics,
    )

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    dataset_summary = {
        "rows": walk_data.get(
            "rows",
            model_data.get(
                "rows"
            ),
        ),
        "minimum_train_size": (
            walk_data.get(
                "min_train_size",
                model_data.get(
                    "configuration",
                    {},
                ).get(
                    "minimum_training_size"
                ),
            )
        ),
        "test_window": (
            walk_data.get(
                "test_size",
                model_data.get(
                    "configuration",
                    {},
                ).get(
                    "test_window"
                ),
            )
        ),
        "step_size": (
            walk_data.get(
                "step_size",
                model_data.get(
                    "configuration",
                    {},
                ).get(
                    "step_size"
                ),
            )
        ),
    }

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    output = {
        "project": "FinGraph AI",
        "evaluation_type": (
            "Temporal walk-forward "
            "financial prediction evaluation"
        ),
        "dataset": dataset_summary,
        "feature_ablation": walk,
        "model_comparison": models,
        "target_experiments": targets,
        "three_class_evaluation": three_class,
        "statistical_significance": statistics,
        "research_findings": findings,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 80)

    print(
        f'Rows                    : '
        f'{dataset_summary["rows"]}'
    )

    print(
        f'Feature configurations  : '
        f'{walk["configuration_count"]}'
    )

    print(
        f'Models evaluated        : '
        f'{models["model_count"]}'
    )

    print(
        f'Target experiments      : '
        f'{targets["experiment_count"]}'
    )

    print(
        f'Pairwise statistical tests: '
        f'{statistics["total_pairwise_comparisons"]}'
    )

    print(
        f'Friedman tests          : '
        f'{statistics["total_friedman_tests"]}'
    )

    print(
        f'Significant after Holm  : '
        f'{statistics["significant_pairwise_after_holm"]}'
    )

    print()
    print("RESEARCH FINDINGS")
    print("-" * 80)

    for finding in findings["findings"]:
        print(f"- {finding}")

    print()
    print(
        f"Results saved: {OUTPUT}"
    )

    print("=" * 80)
    print(
        "FINAL EVALUATION SUMMARY COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()