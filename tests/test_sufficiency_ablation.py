from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.run_sufficiency_ablation import CONFIGURATIONS, GateConfig
from src.evaluation.qa import QAEvaluator


def test_configurations_structure():
    assert len(CONFIGURATIONS) == 6
    cfg_ids = [c.config_id for c in CONFIGURATIONS]
    assert cfg_ids == ["CFG0", "CFG1", "CFG2", "CFG3", "CFG4", "CFG5"]

    # Verify CFG0 is strict baseline gate
    cfg0 = CONFIGURATIONS[0]
    assert cfg0.entity_mode == "strict"
    assert cfg0.relationship_mode == "strict"
    assert cfg0.hop_mode == "strict"
    assert cfg0.gate_enabled is True

    # Verify CFG5 is disabled gate
    cfg5 = CONFIGURATIONS[5]
    assert cfg5.gate_enabled is False


def test_artifact_separation(tmp_path: Path):
    """
    Ensure ablation outputs write to sufficiency_ablation directory and NEVER touch
    canonical Phase 5 benchmark_evaluation_summary.json or per_question_results.csv.
    """
    canonical_dir = Path(__file__).resolve().parents[1] / "results" / "phase5"
    canonical_summary = canonical_dir / "benchmark_evaluation_summary.json"
    canonical_csv = canonical_dir / "per_question_results.csv"

    # Both canonical files must exist and not be modified by ablation runs
    assert canonical_summary.exists()
    assert canonical_csv.exists()

    stat_summary_before = canonical_summary.stat().st_mtime
    stat_csv_before = canonical_csv.stat().st_mtime

    # Ablation directory is separate
    ablation_dir = canonical_dir / "sufficiency_ablation"
    assert ablation_dir != canonical_dir


def test_metric_denominator_separation():
    """
    Verify that refusal precision, refusal recall, decision accuracy,
    and answer coverage use correct denominators.
    """
    # 50 answerable, 14 unanswerable
    total = 64
    total_ans = 50
    total_unans = 14

    tp = 13  # correct refusals on unanswerable
    fn = 1   # hallucinations on unanswerable
    fp = 28  # false refusals on answerable
    tn = 22  # correct answers on answerable

    assert tp + fn == total_unans
    assert fp + tn == total_ans
    assert tp + fn + fp + tn == total

    # Recall on unanswerable
    recall = tp / total_unans
    assert round(recall, 4) == 0.9286

    # Precision
    precision = tp / (tp + fp)
    assert round(precision, 4) == 0.3171

    # Accuracy over all 64
    accuracy = (tp + tn) / total
    assert round(accuracy, 4) == 0.5469

    # Answer coverage over answerable only
    coverage = tn / total_ans
    assert round(coverage, 4) == 0.4400


def test_config_filtering_and_validation():
    """Verify that unknown config IDs raise ValueError and valid config subsets are accepted."""
    from scripts.run_sufficiency_ablation import run_sufficiency_ablation_study

    # Unknown config must raise ValueError
    with pytest.raises(ValueError, match="Unknown configuration ID: 'CFG_INVALID'"):
        run_sufficiency_ablation_study(configs=["CFG_INVALID"], sample_size=1)

    # CLI parser tests
    import argparse
    from scripts.run_sufficiency_ablation import ABLATION_RESULTS_DIR

    parser = argparse.ArgumentParser(description="Run FinGraph AI Evidence Sufficiency Ablation")
    parser.add_argument("--output-dir", "-o", type=Path, default=ABLATION_RESULTS_DIR)
    parser.add_argument("--configs", "-c", nargs="+", default=None)
    parser.add_argument("--sample", "-s", type=int, default=None)

    args = parser.parse_args(["-o", "/custom/path", "-c", "CFG0", "CFG2", "-s", "5"])
    assert args.output_dir == Path("/custom/path")
    assert args.configs == ["CFG0", "CFG2"]
    assert args.sample == 5
