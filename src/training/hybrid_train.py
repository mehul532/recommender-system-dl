"""Training entrypoint for the simple genre-aware hybrid recommender."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from src.data import DatasetConfig, ensure_processed_movielens
from src.models import HybridDeepRecommender
from src.training.comparison import save_model_comparison_report
from src.training.evaluation import evaluate_rmse


@dataclass(frozen=True)
class HybridTrainingConfig:
    """Configuration for hybrid recommender training."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    reports_dir: Path = Path("reports")
    models_dir: Path = Path("models")
    checkpoint_name: str = "hybrid_recommender.pt"
    embedding_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.1
    batch_size: int = 1024
    epochs: int = 10
    early_stopping_patience: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


def run_hybrid_experiments(config: HybridTrainingConfig | None = None) -> dict[str, object]:
    """Train the hybrid recommender, evaluate RMSE, and save artifacts."""

    active_config = config or HybridTrainingConfig()
    data = ensure_processed_movielens(active_config.dataset)

    num_users = int(data.users["user_idx"].max()) + 1
    num_movies = int(data.movies["movie_idx"].max()) + 1

    recommender = HybridDeepRecommender(
        num_users=num_users,
        num_movies=num_movies,
        movie_genre_features=data.genre_features,
        embedding_dim=active_config.embedding_dim,
        hidden_dim=active_config.hidden_dim,
        dropout=active_config.dropout,
        batch_size=active_config.batch_size,
        epochs=active_config.epochs,
        early_stopping_patience=active_config.early_stopping_patience,
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
            "dropout": active_config.dropout,
            "batch_size": active_config.batch_size,
            "epochs": active_config.epochs,
            "early_stopping_patience": active_config.early_stopping_patience,
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
        "stopped_early": bool(recommender.stopped_early),
        "checkpoint_path": str(checkpoint_path),
    }

    report_path = _save_report(report, active_config.reports_dir)
    comparison_report_path = save_model_comparison_report(active_config.reports_dir)
    _print_summary(report, checkpoint_path, report_path, comparison_report_path)
    return report


def _save_report(report: dict[str, object], reports_dir: Path) -> Path:
    """Save hybrid-model metrics to a JSON report."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "hybrid_model_metrics.json"
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    return report_path


def _print_summary(
    report: dict[str, object],
    checkpoint_path: Path,
    report_path: Path,
    comparison_report_path: Path,
) -> None:
    """Print a concise summary of hybrid-model training results."""

    data_summary = report["data_summary"]
    validation_metrics = report["validation"]
    test_metrics = report["test"]

    print("Hybrid model training complete")
    print(f"train ratings: {data_summary['train_ratings']}")
    print(f"validation ratings: {data_summary['validation_ratings']}")
    print(f"test ratings: {data_summary['test_ratings']}")
    print(f"users: {data_summary['num_users']}")
    print(f"movies: {data_summary['num_movies']}")
    print(f"validation RMSE: {validation_metrics['rmse']:.4f}")
    print(f"test RMSE: {test_metrics['rmse']:.4f}")
    print(f"best epoch: {report['best_epoch']}")
    print(f"stopped early: {report['stopped_early']}")
    print(f"saved checkpoint: {checkpoint_path}")
    print(f"saved report: {report_path}")
    print(f"saved comparison report: {comparison_report_path}")


def main() -> None:
    """Run the hybrid-model training command with default settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    args = parser.parse_args()

    default_config = HybridTrainingConfig()
    config = HybridTrainingConfig(
        dataset=default_config.dataset,
        reports_dir=default_config.reports_dir,
        models_dir=default_config.models_dir,
        checkpoint_name=default_config.checkpoint_name,
        embedding_dim=args.embedding_dim or default_config.embedding_dim,
        hidden_dim=args.hidden_dim or default_config.hidden_dim,
        dropout=args.dropout if args.dropout is not None else default_config.dropout,
        batch_size=args.batch_size or default_config.batch_size,
        epochs=args.epochs or default_config.epochs,
        early_stopping_patience=(
            args.early_stopping_patience
            if args.early_stopping_patience is not None
            else default_config.early_stopping_patience
        ),
        learning_rate=(
            args.learning_rate
            if args.learning_rate is not None
            else default_config.learning_rate
        ),
        weight_decay=(
            args.weight_decay
            if args.weight_decay is not None
            else default_config.weight_decay
        ),
    )
    run_hybrid_experiments(config)


if __name__ == "__main__":
    main()
