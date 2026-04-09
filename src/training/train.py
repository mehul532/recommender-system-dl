"""Training and evaluation entrypoints for baseline recommenders."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data import DatasetConfig, ensure_processed_movielens
from src.models import PopularityRecommender, UserItemBiasRecommender
from src.training.evaluation import evaluate_precision_at_k, evaluate_rmse


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for baseline recommender experiments."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    reports_dir: Path = Path("reports")
    top_k: int = 10
    relevance_threshold: float = 4.0
    bias_iterations: int = 10
    user_reg: float = 10.0
    item_reg: float = 10.0


def run_baseline_experiments(config: TrainingConfig | None = None) -> dict[str, object]:
    """Fit baseline recommenders, evaluate them, and save a JSON report."""

    active_config = config or TrainingConfig()
    data = ensure_processed_movielens(active_config.dataset)

    popularity_model = PopularityRecommender().fit(
        ratings=data.ratings_train,
        movies=data.movies,
    )
    bias_model = UserItemBiasRecommender(
        iterations=active_config.bias_iterations,
        user_reg=active_config.user_reg,
        item_reg=active_config.item_reg,
    ).fit(data.ratings_train)

    validation_precision = evaluate_precision_at_k(
        model=popularity_model,
        eval_ratings=data.ratings_val,
        seen_ratings=data.ratings_train,
        top_k=active_config.top_k,
        relevance_threshold=active_config.relevance_threshold,
    )
    test_seen_ratings = pd.concat(
        [data.ratings_train, data.ratings_val],
        ignore_index=True,
    )
    test_precision = evaluate_precision_at_k(
        model=popularity_model,
        eval_ratings=data.ratings_test,
        seen_ratings=test_seen_ratings,
        top_k=active_config.top_k,
        relevance_threshold=active_config.relevance_threshold,
    )
    validation_rmse = evaluate_rmse(
        model=bias_model,
        ratings=data.ratings_val,
    )
    test_rmse = evaluate_rmse(
        model=bias_model,
        ratings=data.ratings_test,
    )

    precision_metric_name = f"precision_at_{active_config.top_k}"
    report = {
        "config": {
            "top_k": active_config.top_k,
            "relevance_threshold": active_config.relevance_threshold,
            "bias_iterations": active_config.bias_iterations,
            "user_reg": active_config.user_reg,
            "item_reg": active_config.item_reg,
        },
        "data_summary": {
            "train_ratings": int(len(data.ratings_train)),
            "validation_ratings": int(len(data.ratings_val)),
            "test_ratings": int(len(data.ratings_test)),
            "num_users": int(len(data.users)),
            "num_movies": int(len(data.movies)),
        },
        "baselines": {
            "popularity": {
                "validation": {
                    precision_metric_name: float(validation_precision["precision"]),
                    "evaluated_users": int(validation_precision["evaluated_users"]),
                    "users_with_relevant_items": int(
                        validation_precision["users_with_relevant_items"]
                    ),
                },
                "test": {
                    precision_metric_name: float(test_precision["precision"]),
                    "evaluated_users": int(test_precision["evaluated_users"]),
                    "users_with_relevant_items": int(
                        test_precision["users_with_relevant_items"]
                    ),
                },
            },
            "user_item_bias": {
                "validation": {
                    "rmse": float(validation_rmse["rmse"]),
                    "count": int(validation_rmse["count"]),
                },
                "test": {
                    "rmse": float(test_rmse["rmse"]),
                    "count": int(test_rmse["count"]),
                },
            },
        },
    }

    report_path = _save_report(report, active_config.reports_dir)
    _print_summary(report=report, report_path=report_path, top_k=active_config.top_k)
    return report


def train_model(config: TrainingConfig | None = None) -> dict[str, object]:
    """Backward-compatible wrapper for the baseline runner."""

    return run_baseline_experiments(config)


def _save_report(report: dict[str, object], reports_dir: Path) -> Path:
    """Save the baseline metrics report to disk."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "baseline_metrics.json"
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    return report_path


def _print_summary(report: dict[str, object], report_path: Path, top_k: int) -> None:
    """Print a readable baseline experiment summary."""

    data_summary = report["data_summary"]
    baselines = report["baselines"]
    precision_metric_name = f"precision_at_{top_k}"

    popularity_validation = baselines["popularity"]["validation"]
    popularity_test = baselines["popularity"]["test"]
    bias_validation = baselines["user_item_bias"]["validation"]
    bias_test = baselines["user_item_bias"]["test"]

    print("Baseline evaluation complete")
    print(f"train ratings: {data_summary['train_ratings']}")
    print(f"validation ratings: {data_summary['validation_ratings']}")
    print(f"test ratings: {data_summary['test_ratings']}")
    print(f"users: {data_summary['num_users']}")
    print(f"movies: {data_summary['num_movies']}")
    print(
        "popularity "
        f"Precision@{top_k} validation: "
        f"{popularity_validation[precision_metric_name]:.4f}"
    )
    print(
        "popularity "
        f"Precision@{top_k} test: "
        f"{popularity_test[precision_metric_name]:.4f}"
    )
    print(f"user-item bias RMSE validation: {bias_validation['rmse']:.4f}")
    print(f"user-item bias RMSE test: {bias_test['rmse']:.4f}")
    print(f"saved report: {report_path}")


def main() -> None:
    """Run the default baseline experiment command."""

    run_baseline_experiments()


if __name__ == "__main__":
    main()
