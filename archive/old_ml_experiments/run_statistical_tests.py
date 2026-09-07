#!/usr/bin/env python3

"""
FinGraph AI - Statistical Significance Analysis

Performs statistical comparison of:

1. Walk-forward feature configurations
2. ML model configurations
3. Target/horizon experiments

Methods:
- Paired Wilcoxon signed-rank tests
- Holm multiple-comparison correction
- Friedman tests

All comparisons use identical temporal folds wherever possible.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


# ============================================================
# CONFIGURATION
# ============================================================

WALK_FORWARD_PATH = Path(
    "data/processed/unified/aapl_walk_forward_results.json"
)

MODEL_PATH = Path(
    "data/processed/unified/aapl_model_comparison.json"
)

TARGET_PATH = Path(
    "data/processed/unified/aapl_target_experiments.json"
)

OUTPUT_PATH = Path(
    "data/processed/unified/aapl_statistical_tests.json"
)

ALPHA = 0.05


# ============================================================
# HELPERS
# ============================================================


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected dictionary in {path}"
        )

    return data


def safe_float(value: Any) -> float | None:
    """Convert a value to float safely."""

    if value is None:
        return None

    if isinstance(value, str):
        if value.upper() in {
            "N/A",
            "NA",
            "NONE",
            "NAN",
        }:
            return None

    try:
        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float | None:
    """Calculate mean."""

    if not values:
        return None

    return float(np.mean(values))


def std(values: list[float]) -> float | None:
    """Calculate sample standard deviation."""

    if len(values) < 2:
        return None

    return float(np.std(values, ddof=1))


# ============================================================
# EXTRACT FOLD METRICS
# ============================================================


def extract_fold_metrics(
    folds: list[dict[str, Any]],
    metric: str,
) -> dict[int, float]:
    """
    Extract:

        fold_number -> metric_value

    Missing / invalid metric values are skipped.
    """

    result: dict[int, float] = {}

    for index, fold in enumerate(folds):

        if not isinstance(fold, dict):
            continue

        fold_number = fold.get(
            "fold",
            index + 1,
        )

        try:
            fold_number = int(fold_number)
        except (TypeError, ValueError):
            fold_number = index + 1

        value = safe_float(
            fold.get(metric)
        )

        if value is not None:
            result[fold_number] = value

    return result


# ============================================================
# PAIRED DATA
# ============================================================


def paired_values(
    folds_a: list[dict[str, Any]],
    folds_b: list[dict[str, Any]],
    metric: str,
) -> tuple[list[float], list[float], list[int]]:
    """
    Match two configurations using identical fold numbers.
    """

    a = extract_fold_metrics(
        folds_a,
        metric,
    )

    b = extract_fold_metrics(
        folds_b,
        metric,
    )

    common = sorted(
        set(a).intersection(b)
    )

    values_a = [a[i] for i in common]
    values_b = [b[i] for i in common]

    return values_a, values_b, common


# ============================================================
# WILCOXON
# ============================================================


def paired_wilcoxon(
    values_a: list[float],
    values_b: list[float],
) -> dict[str, Any]:
    """
    Paired Wilcoxon signed-rank test.

    Returns N/A when the test cannot be performed.
    """

    if len(values_a) != len(values_b):
        return {
            "n": 0,
            "statistic": None,
            "p_value": None,
        }

    if len(values_a) < 2:
        return {
            "n": len(values_a),
            "statistic": None,
            "p_value": None,
        }

    differences = np.asarray(
        values_a,
        dtype=float,
    ) - np.asarray(
        values_b,
        dtype=float,
    )

    # If every difference is zero, Wilcoxon has no useful
    # ranking information.
    if np.allclose(
        differences,
        0.0,
    ):
        return {
            "n": len(values_a),
            "statistic": 0.0,
            "p_value": 1.0,
        }

    try:

        statistic, p_value = wilcoxon(
            values_a,
            values_b,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )

        return {
            "n": len(values_a),
            "statistic": float(statistic),
            "p_value": float(p_value),
        }

    except ValueError:

        return {
            "n": len(values_a),
            "statistic": None,
            "p_value": None,
        }


# ============================================================
# HOLM CORRECTION
# ============================================================


def holm_correction(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Apply Holm-Bonferroni correction across all
    valid pairwise comparisons.
    """

    valid = [
        result
        for result in results
        if result.get("p_value") is not None
    ]

    if not valid:
        return results

    ordered = sorted(
        valid,
        key=lambda x: x["p_value"],
    )

    total = len(ordered)

    adjusted_values: list[float] = []

    running_max = 0.0

    for rank, result in enumerate(
        ordered,
        start=1,
    ):

        adjusted = (
            result["p_value"]
            * (total - rank + 1)
        )

        adjusted = min(
            float(adjusted),
            1.0,
        )

        running_max = max(
            running_max,
            adjusted,
        )

        adjusted_values.append(
            running_max
        )

    for result, adjusted in zip(
        ordered,
        adjusted_values,
    ):

        result["holm_adjusted_p_value"] = (
            adjusted
        )

        result["significant"] = (
            adjusted < ALPHA
        )

    for result in results:

        if result.get("p_value") is None:

            result["holm_adjusted_p_value"] = None
            result["significant"] = False

    return results


# ============================================================
# PAIRWISE TESTS
# ============================================================


def run_pairwise_tests(
    configurations: list[dict[str, Any]],
    name_key: str,
    folds_key: str,
    metrics: list[str],
) -> list[dict[str, Any]]:
    """
    Run paired Wilcoxon tests for every pair of configurations
    and every requested metric.
    """

    results: list[dict[str, Any]] = []

    for config_a, config_b in combinations(
        configurations,
        2,
    ):

        name_a = config_a.get(
            name_key,
            "UNKNOWN_A",
        )

        name_b = config_b.get(
            name_key,
            "UNKNOWN_B",
        )

        folds_a = config_a.get(
            folds_key,
            [],
        )

        folds_b = config_b.get(
            folds_key,
            [],
        )

        if not isinstance(
            folds_a,
            list,
        ):
            continue

        if not isinstance(
            folds_b,
            list,
        ):
            continue

        for metric in metrics:

            values_a, values_b, common_folds = (
                paired_values(
                    folds_a,
                    folds_b,
                    metric,
                )
            )

            test = paired_wilcoxon(
                values_a,
                values_b,
            )

            results.append(
                {
                    "configuration_a": name_a,
                    "configuration_b": name_b,
                    "metric": metric,
                    "folds_compared": common_folds,
                    "n": test["n"],
                    "statistic": test[
                        "statistic"
                    ],
                    "p_value": test[
                        "p_value"
                    ],
                    "mean_a": mean(
                        values_a
                    ),
                    "mean_b": mean(
                        values_b
                    ),
                    "mean_difference": (
                        mean(
                            [
                                a - b
                                for a, b in zip(
                                    values_a,
                                    values_b,
                                )
                            ]
                        )
                    ),
                }
            )

    return results


# ============================================================
# FRIEDMAN TEST
# ============================================================


def run_friedman_tests(
    configurations: list[dict[str, Any]],
    name_key: str,
    folds_key: str,
    metrics: list[str],
) -> list[dict[str, Any]]:
    """
    Friedman test across multiple configurations.

    Only common fold numbers shared by ALL configurations
    are included.
    """

    results: list[dict[str, Any]] = []

    if len(configurations) < 3:
        return results

    for metric in metrics:

        fold_maps = []

        for config in configurations:

            folds = config.get(
                folds_key,
                [],
            )

            fold_maps.append(
                extract_fold_metrics(
                    folds,
                    metric,
                )
            )

        if not fold_maps:
            continue

        common_folds = set(
            fold_maps[0]
        )

        for mapping in fold_maps[1:]:

            common_folds &= set(
                mapping
            )

        common_folds = sorted(
            common_folds
        )

        # Friedman requires at least two
        # matched observations.
        if len(common_folds) < 2:
            results.append(
                {
                    "metric": metric,
                    "configurations": [
                        config.get(
                            name_key
                        )
                        for config in configurations
                    ],
                    "folds_compared": common_folds,
                    "n": len(common_folds),
                    "statistic": None,
                    "p_value": None,
                }
            )

            continue

        samples = []

        for mapping in fold_maps:

            samples.append(
                [
                    mapping[fold]
                    for fold in common_folds
                ]
            )

        try:

            statistic, p_value = (
                friedmanchisquare(
                    *samples
                )
            )

            results.append(
                {
                    "metric": metric,
                    "configurations": [
                        config.get(
                            name_key
                        )
                        for config in configurations
                    ],
                    "folds_compared": common_folds,
                    "n": len(common_folds),
                    "statistic": float(
                        statistic
                    ),
                    "p_value": float(
                        p_value
                    ),
                    "significant": (
                        float(p_value)
                        < ALPHA
                    ),
                }
            )

        except ValueError:

            results.append(
                {
                    "metric": metric,
                    "configurations": [
                        config.get(
                            name_key
                        )
                        for config in configurations
                    ],
                    "folds_compared": common_folds,
                    "n": len(common_folds),
                    "statistic": None,
                    "p_value": None,
                    "significant": False,
                }
            )

    return results


# ============================================================
# WALK-FORWARD
# ============================================================


def process_walk_forward(
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    Process feature-ablation walk-forward results.
    """

    configurations = data.get(
        "results",
        [],
    )

    if not isinstance(
        configurations,
        list,
    ):
        configurations = []

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
    ]

    pairwise = run_pairwise_tests(
        configurations,
        name_key="name",
        folds_key="folds",
        metrics=metrics,
    )

    friedman = run_friedman_tests(
        configurations,
        name_key="name",
        folds_key="folds",
        metrics=metrics,
    )

    return {
        "configuration_count": len(
            configurations
        ),
        "configurations": [
            config.get("name")
            for config in configurations
        ],
        "pairwise": pairwise,
        "friedman": friedman,
    }


# ============================================================
# MODELS
# ============================================================


def process_models(
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    Process model-comparison results.

    Important:
    Model results use:

        models -> folds

    rather than:

        comparison -> folds
    """

    configurations = data.get(
        "models",
        [],
    )

    if not isinstance(
        configurations,
        list,
    ):
        configurations = []

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "brier_score",
    ]

    pairwise = run_pairwise_tests(
        configurations,
        name_key="model",
        folds_key="folds",
        metrics=metrics,
    )

    friedman = run_friedman_tests(
        configurations,
        name_key="model",
        folds_key="folds",
        metrics=metrics,
    )

    return {
        "configuration_count": len(
            configurations
        ),
        "configurations": [
            config.get("model")
            for config in configurations
        ],
        "pairwise": pairwise,
        "friedman": friedman,
    }


# ============================================================
# TARGET EXPERIMENTS
# ============================================================


def process_target_experiments(
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    Process target/horizon experiments.

    IMPORTANT:

    The target JSON stores fold-level results under:

        experiments[*]["fold_results"]

    This is different from the walk-forward and model files.

    This function explicitly uses fold_results.
    """

    configurations = data.get(
        "experiments",
        [],
    )

    if not isinstance(
        configurations,
        list,
    ):
        configurations = []

    # Normalize fold_results -> folds.
    normalized: list[dict[str, Any]] = []

    for experiment in configurations:

        if not isinstance(
            experiment,
            dict,
        ):
            continue

        fold_results = experiment.get(
            "fold_results",
            [],
        )

        if not isinstance(
            fold_results,
            list,
        ):
            fold_results = []

        normalized.append(
            {
                "name": experiment.get(
                    "name",
                    "UNKNOWN",
                ),
                "folds": fold_results,
            }
        )

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
    ]

    pairwise = run_pairwise_tests(
        normalized,
        name_key="name",
        folds_key="folds",
        metrics=metrics,
    )

    friedman = run_friedman_tests(
        normalized,
        name_key="name",
        folds_key="folds",
        metrics=metrics,
    )

    return {
        "configuration_count": len(
            normalized
        ),
        "configurations": [
            config["name"]
            for config in normalized
        ],
        "pairwise": pairwise,
        "friedman": friedman,
    }


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print("=" * 80)
    print(
        "FINGRAPH AI STATISTICAL SIGNIFICANCE ANALYSIS"
    )
    print("=" * 80)

    print()
    print(
        "Purpose: determine whether observed performance differences "
        "are statistically meaningful."
    )

    print(
        "Method: paired Wilcoxon tests across identical "
        "walk-forward temporal folds."
    )

    print()

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    walk_data = load_json(
        WALK_FORWARD_PATH
    )

    model_data = load_json(
        MODEL_PATH
    )

    target_data = load_json(
        TARGET_PATH
    )

    print("=" * 80)
    print("INPUT FILES")
    print("=" * 80)

    print(
        f"Walk-forward : {WALK_FORWARD_PATH}"
    )

    print(
        f"Models       : {MODEL_PATH}"
    )

    print(
        f"Targets      : {TARGET_PATH}"
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    walk = process_walk_forward(
        walk_data
    )

    models = process_models(
        model_data
    )

    targets = process_target_experiments(
        target_data
    )

    # --------------------------------------------------------
    # Print structure
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("INPUT STRUCTURE VALIDATION")
    print("=" * 80)

    print(
        "Walk-forward configurations :",
        walk["configuration_count"],
    )

    print(
        "Models                      :",
        models["configuration_count"],
    )

    print(
        "Target experiments          :",
        targets["configuration_count"],
    )

    # --------------------------------------------------------
    # Walk-forward
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("WALK-FORWARD STATISTICAL TESTS")
    print("=" * 80)

    print(
        "Detected configurations:",
        walk["configuration_count"],
    )

    for name in walk["configurations"]:
        print(f"  {name}")

    print(
        "Pairwise comparisons:",
        len(walk["pairwise"]),
    )

    print(
        "Friedman tests:",
        len(walk["friedman"]),
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("MODEL STATISTICAL TESTS")
    print("=" * 80)

    print(
        "Detected configurations:",
        models["configuration_count"],
    )

    for name in models["configurations"]:
        print(f"  {name}")

    print(
        "Pairwise comparisons:",
        len(models["pairwise"]),
    )

    print(
        "Friedman tests:",
        len(models["friedman"]),
    )

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "TARGET_EXPERIMENTS STATISTICAL TESTS"
    )
    print("=" * 80)

    print(
        "Detected configurations:",
        targets["configuration_count"],
    )

    for name in targets["configurations"]:
        print(f"  {name}")

    print(
        "Pairwise comparisons:",
        len(targets["pairwise"]),
    )

    print(
        "Friedman tests:",
        len(targets["friedman"]),
    )

    # --------------------------------------------------------
    # Holm correction
    # --------------------------------------------------------

    all_pairwise = (
        walk["pairwise"]
        + models["pairwise"]
        + targets["pairwise"]
    )

    all_pairwise = holm_correction(
        all_pairwise
    )

    # Split corrected results back.
    walk_count = len(
        walk["pairwise"]
    )

    model_count = len(
        models["pairwise"]
    )

    walk["pairwise"] = all_pairwise[
        :walk_count
    ]

    models["pairwise"] = all_pairwise[
        walk_count:
        walk_count + model_count
    ]

    targets["pairwise"] = all_pairwise[
        walk_count + model_count:
    ]

    # --------------------------------------------------------
    # Friedman
    # --------------------------------------------------------

    all_friedman = (
        walk["friedman"]
        + models["friedman"]
        + targets["friedman"]
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("STATISTICAL TEST SUMMARY")
    print("=" * 80)

    print(
        "Total pairwise comparisons:",
        len(all_pairwise),
    )

    print(
        "Total Friedman tests       :",
        len(all_friedman),
    )

    # --------------------------------------------------------
    # Significant pairwise
    # --------------------------------------------------------

    significant_pairwise = [
        result
        for result in all_pairwise
        if result.get(
            "significant",
            False,
        )
    ]

    print()
    print("=" * 80)
    print(
        "SIGNIFICANT RESULTS "
        "(HOLM-ADJUSTED)"
    )
    print("=" * 80)

    if not significant_pairwise:

        print(
            "No pairwise comparisons remained "
            "significant after Holm correction "
            "at alpha=0.05."
        )

    else:

        for result in significant_pairwise:

            print(
                f'{result["configuration_a"]} vs '
                f'{result["configuration_b"]} | '
                f'{result["metric"]} | '
                f'p={result["p_value"]:.6f} | '
                f'Holm p='
                f'{result["holm_adjusted_p_value"]:.6f}'
            )

    # --------------------------------------------------------
    # Friedman
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FRIEDMAN TESTS")
    print("=" * 80)

    for group_name, group in [
        (
            "walk_forward",
            walk["friedman"],
        ),
        (
            "models",
            models["friedman"],
        ),
        (
            "target_experiments",
            targets["friedman"],
        ),
    ]:

        for result in group:

            p_value = result.get(
                "p_value"
            )

            if p_value is None:

                p_text = "N/A"

            else:

                p_text = f"{p_value:.6f}"

            print(
                f"{group_name} | "
                f'{result["metric"]} | '
                f"p={p_text}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {
        "configuration": {
            "alpha": ALPHA,
            "multiple_comparison_correction": (
                "Holm-Bonferroni"
            ),
            "pairwise_test": (
                "Wilcoxon signed-rank"
            ),
            "omnibus_test": (
                "Friedman"
            ),
        },
        "inputs": {
            "walk_forward": str(
                WALK_FORWARD_PATH
            ),
            "models": str(
                MODEL_PATH
            ),
            "target_experiments": str(
                TARGET_PATH
            ),
        },
        "walk_forward": walk,
        "models": models,
        "target_experiments": targets,
        "summary": {
            "total_pairwise_comparisons": len(
                all_pairwise
            ),
            "total_friedman_tests": len(
                all_friedman
            ),
            "significant_pairwise_after_holm": len(
                significant_pairwise
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print(
        "STATISTICAL ANALYSIS COMPLETE"
    )
    print("=" * 80)

    print(
        "Pairwise results :",
        len(all_pairwise),
    )

    print(
        "Friedman tests   :",
        len(all_friedman),
    )

    print(
        f"Results saved    : {OUTPUT_PATH}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()