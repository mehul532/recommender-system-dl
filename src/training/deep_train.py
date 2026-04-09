"""Training entrypoint for the simple PyTorch deep recommender."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.data import DatasetConfig, ensure_processed_movielens
from src.models import DeepRecommender
from src.training.evaluation import evaluate_rmse


@dataclass(frozen=True)
class DeepTrainingConfig:
    """Configuration for deep recommender training."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    reports_dir: Path = Path("reports")
    models_dir: Path = Path("models")
    checkpoint_name: str = "deep_recommender.pt"
    embedding_dim: int = 64
    hidden_dim: int = 128
    batch_size: int = 1024
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


def run_deep_experiments(config: DeepTrainingConfig | None = None) -> dict[str, object]:
    """Train the deep recommender, evaluate RMSE, and save artifacts."""

    active_config = config or DeepTrainingConfig()
    data = ensure_processed_movielens(active_config.dataset)

    num_users = int(data.users["user_idx"].max()) + 1
    num_movies = int(data.movies["movie_idx"].max()) + 1

    recommender = DeepRecommender(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=active_config.embedding_dim,
        hidden_dim=active_config.hidden_dim,
        batch_size=active_config.batch_size,
        epochs=active_config.epochs,
        learning_rate=active_config.learning_rate,
        weight_decay=active_config.weight_decay,
    ).fit(
        train_ratings=data.ratings_train,
        val_ratings=data.ratings_val,
    )

    checkpoint_path = active_config.models_dir / active_config.checkpoint_name
    recommender.save_checkpoint(checkpoint_path)

    validation_metrics = evaluate_rmse(recommender, data.ratings_val)
    test_metrics = evaluate_rmse(recommender, data.ratings_test)

    report = {
        "config": {
            "embedding_dim": active_config.embedding_dim,
            "hidden_dim": active_config.hidden_dim,
            "batch_size": active_config.batch_size,
            "epochs": active_config.epochs,
            "learning_rate": active_config.learning_rate,
            "weight_decay": active_config.weight_decay,
        },
        "data_summary": {
            "train_ratings": int(len(data.ratings_train)),
            "validation_ratings": int(len(data.ratings_val)),
            "test_ratings": int(len(data.ratings_test)),
            "num_users": int(len(data.users)),
            "num_movies": int(len(data.movies)),
        },
        "validation": {"rmse": float(validation_metrics["rmse"])},
        "test": {"rmse": float(test_metrics["rmse"])},
        "best_epoch": int(recommender.best_epoch),
        "checkpoint_path": str(checkpoint_path),
    }

    report_path = _save_report(report, active_config.reports_dir)
    _print_summary(report, checkpoint_path, report_path)
    return report


def _save_report(report: dict[str, object], reports_dir: Path) -> Path:
    """Save deep-model metrics to a JSON report."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "deep_model_metrics.json"
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    return report_path


def _print_summary(
    report: dict[str, object], checkpoint_path: Path, report_path: Path
) -> None:
    """Print a concise summary of deep-model training results."""

    data_summary = report["data_summary"]
    validation_metrics = report["validation"]
    test_metrics = report["test"]

    print("Deep model training complete")
    print(f"train ratings: {data_summary['train_ratings']}")
    print(f"validation ratings: {data_summary['validation_ratings']}")
    print(f"test ratings: {data_summary['test_ratings']}")
    print(f"users: {data_summary['num_users']}")
    print(f"movies: {data_summary['num_movies']}")
    print(f"validation RMSE: {validation_metrics['rmse']:.4f}")
    print(f"test RMSE: {test_metrics['rmse']:.4f}")
    print(f"best epoch: {report['best_epoch']}")
    print(f"saved checkpoint: {checkpoint_path}")
    print(f"saved report: {report_path}")


def main() -> None:
    """Run the deep-model training command with default settings."""

    run_deep_experiments()


if __name__ == "__main__":
    main()
