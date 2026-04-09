"""Comparison helpers for baseline, deep-model, and hybrid-model reports."""

from __future__ import annotations

import json
from pathlib import Path


def save_model_comparison_report(reports_dir: Path | str = Path("reports")) -> Path:
    """Read available model reports and save a comparison artifact."""

    reports_path = Path(reports_dir)
    baseline_report_path = reports_path / "baseline_metrics.json"
    deep_report_path = reports_path / "deep_model_metrics.json"
    hybrid_report_path = reports_path / "hybrid_model_metrics.json"

    if not baseline_report_path.exists():
        raise FileNotFoundError(
            f"Baseline report not found at {baseline_report_path}. "
            "Run the baseline experiment before creating the comparison report."
        )
    if not deep_report_path.exists() and not hybrid_report_path.exists():
        raise FileNotFoundError(
            "No model report found for comparison. "
            f"Expected at least one of {deep_report_path} or {hybrid_report_path}."
        )

    baseline_report = _load_json(baseline_report_path)
    deep_report = _load_optional_json(deep_report_path)
    hybrid_report = _load_optional_json(hybrid_report_path)

    baseline_scores = _extract_baseline_scores(baseline_report)
    comparison_report: dict[str, object] = {
        "available_models": ["baseline_user_item_bias"],
        "baseline_user_item_bias": baseline_scores,
        "comparisons": {},
    }
    model_scores: dict[str, dict[str, float]] = {
        "baseline_user_item_bias": baseline_scores
    }

    if deep_report is not None:
        deep_scores = _extract_model_scores(deep_report)
        comparison_report["available_models"].append("deep_model")
        comparison_report["deep_model"] = deep_scores
        comparison_report["rmse_delta"] = {
            "validation": deep_scores["validation_rmse"]
            - baseline_scores["validation_rmse"],
            "test": deep_scores["test_rmse"] - baseline_scores["test_rmse"],
        }
        comparison_report["better_model"] = {
            "validation": _select_better_model(
                baseline_scores["validation_rmse"],
                deep_scores["validation_rmse"],
            ),
            "test": _select_better_model(
                baseline_scores["test_rmse"],
                deep_scores["test_rmse"],
            ),
        }
        comparison_report["summary"] = {
            "validation": _build_summary_line(
                split_name="validation",
                baseline_rmse=baseline_scores["validation_rmse"],
                deep_rmse=deep_scores["validation_rmse"],
            ),
            "test": _build_summary_line(
                split_name="test",
                baseline_rmse=baseline_scores["test_rmse"],
                deep_rmse=deep_scores["test_rmse"],
            ),
        }
        comparison_report["comparisons"]["deep_vs_baseline"] = (
            _build_pairwise_comparison(
                left_model_name="baseline_user_item_bias",
                left_scores=baseline_scores,
                right_model_name="deep_model",
                right_scores=deep_scores,
            )
        )
        model_scores["deep_model"] = deep_scores

    if hybrid_report is not None:
        hybrid_scores = _extract_model_scores(hybrid_report)
        comparison_report["available_models"].append("hybrid_model")
        comparison_report["hybrid_model"] = hybrid_scores
        comparison_report["comparisons"]["hybrid_vs_baseline"] = (
            _build_pairwise_comparison(
                left_model_name="baseline_user_item_bias",
                left_scores=baseline_scores,
                right_model_name="hybrid_model",
                right_scores=hybrid_scores,
            )
        )
        model_scores["hybrid_model"] = hybrid_scores

        if "deep_model" in model_scores:
            comparison_report["comparisons"]["hybrid_vs_deep"] = (
                _build_pairwise_comparison(
                    left_model_name="deep_model",
                    left_scores=model_scores["deep_model"],
                    right_model_name="hybrid_model",
                    right_scores=hybrid_scores,
                )
            )

    comparison_report["best_model"] = {
        "validation": _select_best_model(model_scores, split_name="validation_rmse"),
        "test": _select_best_model(model_scores, split_name="test_rmse"),
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


def _load_json(path: Path) -> dict[str, object]:
    """Load a JSON file from disk."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_optional_json(path: Path) -> dict[str, object] | None:
    """Load a JSON file when it exists."""

    if not path.exists():
        return None
    return _load_json(path)


def _extract_baseline_scores(report: dict[str, object]) -> dict[str, float]:
    """Extract RMSE scores from the baseline report."""

    baselines = report["baselines"]
    user_item_bias = baselines["user_item_bias"]
    return {
        "validation_rmse": float(user_item_bias["validation"]["rmse"]),
        "test_rmse": float(user_item_bias["test"]["rmse"]),
    }


def _extract_model_scores(report: dict[str, object]) -> dict[str, float]:
    """Extract validation and test RMSE scores from a model report."""

    return {
        "validation_rmse": float(report["validation"]["rmse"]),
        "test_rmse": float(report["test"]["rmse"]),
    }


def _build_pairwise_comparison(
    left_model_name: str,
    left_scores: dict[str, float],
    right_model_name: str,
    right_scores: dict[str, float],
) -> dict[str, object]:
    """Build a compact RMSE comparison between two models."""

    validation_delta = (
        right_scores["validation_rmse"] - left_scores["validation_rmse"]
    )
    test_delta = right_scores["test_rmse"] - left_scores["test_rmse"]
    return {
        "left_model": left_model_name,
        "right_model": right_model_name,
        "validation": {
            "left_rmse": left_scores["validation_rmse"],
            "right_rmse": right_scores["validation_rmse"],
            "rmse_delta": validation_delta,
            "better_model": _select_lower_rmse_model(
                left_model_name,
                left_scores["validation_rmse"],
                right_model_name,
                right_scores["validation_rmse"],
            ),
            "summary": _build_generic_summary_line(
                split_name="validation",
                left_model_name=left_model_name,
                left_rmse=left_scores["validation_rmse"],
                right_model_name=right_model_name,
                right_rmse=right_scores["validation_rmse"],
            ),
        },
        "test": {
            "left_rmse": left_scores["test_rmse"],
            "right_rmse": right_scores["test_rmse"],
            "rmse_delta": test_delta,
            "better_model": _select_lower_rmse_model(
                left_model_name,
                left_scores["test_rmse"],
                right_model_name,
                right_scores["test_rmse"],
            ),
            "summary": _build_generic_summary_line(
                split_name="test",
                left_model_name=left_model_name,
                left_rmse=left_scores["test_rmse"],
                right_model_name=right_model_name,
                right_rmse=right_scores["test_rmse"],
            ),
        },
    }


def _select_lower_rmse_model(
    left_model_name: str,
    left_rmse: float,
    right_model_name: str,
    right_rmse: float,
) -> str:
    """Return the name of the model with the lower RMSE."""

    return right_model_name if right_rmse < left_rmse else left_model_name


def _build_generic_summary_line(
    split_name: str,
    left_model_name: str,
    left_rmse: float,
    right_model_name: str,
    right_rmse: float,
) -> str:
    """Create a readable pairwise RMSE summary."""

    better_model = _select_lower_rmse_model(
        left_model_name,
        left_rmse,
        right_model_name,
        right_rmse,
    )
    delta = abs(right_rmse - left_rmse)
    better_label = _display_name(better_model)
    worse_label = _display_name(
        right_model_name if better_model == left_model_name else left_model_name
    )
    better_rmse = min(left_rmse, right_rmse)
    worse_rmse = max(left_rmse, right_rmse)
    return (
        f"On {split_name}, the {better_label} is better by {delta:.4f} RMSE "
        f"({better_rmse:.4f} vs {worse_rmse:.4f}) compared with the {worse_label}."
    )


def _display_name(model_name: str) -> str:
    """Return a human-readable model label."""

    labels = {
        "baseline_user_item_bias": "baseline user-item bias model",
        "deep_model": "deep model",
        "hybrid_model": "hybrid model",
    }
    return labels.get(model_name, model_name.replace("_", " "))


def _select_best_model(
    model_scores: dict[str, dict[str, float]],
    split_name: str,
) -> str:
    """Return the best available model for a specific split."""

    return min(model_scores.items(), key=lambda item: item[1][split_name])[0]
