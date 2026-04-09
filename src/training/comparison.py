"""Comparison helpers for baseline and deep-model reports."""

from __future__ import annotations

import json
from pathlib import Path


def save_model_comparison_report(reports_dir: Path | str = Path("reports")) -> Path:
    """Read baseline and deep-model reports and save a comparison artifact."""

    reports_path = Path(reports_dir)
    baseline_report_path = reports_path / "baseline_metrics.json"
    deep_report_path = reports_path / "deep_model_metrics.json"

    if not baseline_report_path.exists():
        raise FileNotFoundError(
            f"Baseline report not found at {baseline_report_path}. "
            "Run the baseline experiment before creating the comparison report."
        )
    if not deep_report_path.exists():
        raise FileNotFoundError(
            f"Deep-model report not found at {deep_report_path}. "
            "Run the deep-model experiment before creating the comparison report."
        )

    with baseline_report_path.open("r", encoding="utf-8") as baseline_file:
        baseline_report = json.load(baseline_file)
    with deep_report_path.open("r", encoding="utf-8") as deep_file:
        deep_report = json.load(deep_file)

    baseline_validation_rmse = float(
        baseline_report["baselines"]["user_item_bias"]["validation"]["rmse"]
    )
    baseline_test_rmse = float(
        baseline_report["baselines"]["user_item_bias"]["test"]["rmse"]
    )
    deep_validation_rmse = float(deep_report["validation"]["rmse"])
    deep_test_rmse = float(deep_report["test"]["rmse"])

    comparison_report = {
        "baseline_user_item_bias": {
            "validation_rmse": baseline_validation_rmse,
            "test_rmse": baseline_test_rmse,
        },
        "deep_model": {
            "validation_rmse": deep_validation_rmse,
            "test_rmse": deep_test_rmse,
        },
        "rmse_delta": {
            "validation": deep_validation_rmse - baseline_validation_rmse,
            "test": deep_test_rmse - baseline_test_rmse,
        },
        "better_model": {
            "validation": _select_better_model(
                baseline_validation_rmse, deep_validation_rmse
            ),
            "test": _select_better_model(baseline_test_rmse, deep_test_rmse),
        },
        "summary": {
            "validation": _build_summary_line(
                split_name="validation",
                baseline_rmse=baseline_validation_rmse,
                deep_rmse=deep_validation_rmse,
            ),
            "test": _build_summary_line(
                split_name="test",
                baseline_rmse=baseline_test_rmse,
                deep_rmse=deep_test_rmse,
            ),
        },
    }

    comparison_report_path = reports_path / "model_comparison.json"
    with comparison_report_path.open("w", encoding="utf-8") as comparison_file:
        json.dump(comparison_report, comparison_file, indent=2)

    return comparison_report_path


def _select_better_model(baseline_rmse: float, deep_rmse: float) -> str:
    """Return the model name with the lower RMSE."""

    return "deep_model" if deep_rmse < baseline_rmse else "baseline_user_item_bias"


def _build_summary_line(
    split_name: str, baseline_rmse: float, deep_rmse: float
) -> str:
    """Create a short human-readable RMSE comparison summary."""

    delta = deep_rmse - baseline_rmse
    better_model = _select_better_model(baseline_rmse, deep_rmse)
    if better_model == "deep_model":
        return (
            f"On {split_name}, the deep model is better by {abs(delta):.4f} RMSE "
            f"({deep_rmse:.4f} vs {baseline_rmse:.4f})."
        )
    return (
        f"On {split_name}, the baseline user-item bias model is better by "
        f"{abs(delta):.4f} RMSE ({baseline_rmse:.4f} vs {deep_rmse:.4f})."
    )
