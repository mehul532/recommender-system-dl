"""Tests for the simple genre-aware hybrid recommender."""

from __future__ import annotations

import json

import pandas as pd
import torch

from src.data import DatasetConfig, preprocess_movielens_1m
from src.models import HybridDeepRecommender
from src.training import HybridTrainingConfig, run_hybrid_experiments


def test_hybrid_recommender_predict_returns_bounded_scores() -> None:
    """Return one bounded prediction per input rating row."""

    ratings = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1],
            "movie_idx": [0, 1, 0, 1],
            "rating": [5.0, 3.0, 4.0, 2.0],
        }
    )
    genre_features = pd.DataFrame(
        {
            "movie_id": [10, 20],
            "movie_idx": [0, 1],
            "genre_action": [1, 0],
            "genre_comedy": [0, 1],
        }
    )

    model = HybridDeepRecommender(
        num_users=2,
        num_movies=2,
        movie_genre_features=genre_features,
        embedding_dim=8,
        hidden_dim=16,
        dropout=0.2,
        batch_size=2,
        epochs=2,
        early_stopping_patience=2,
    ).fit(train_ratings=ratings, val_ratings=ratings)

    predictions = model.predict(ratings)

    assert len(predictions) == len(ratings)
    assert predictions.between(1.0, 5.0).all()
    assert model.genre_feature_columns == ["genre_action", "genre_comedy"]
    assert isinstance(model.model.mlp[2], torch.nn.Dropout)
    assert model.model.mlp[2].p == 0.2


def test_run_hybrid_experiments_saves_checkpoint_report_and_comparison(
    tmp_path,
    capsys,
) -> None:
    """Train the hybrid model end-to-end on synthetic processed data."""

    dataset_config = _write_sample_movielens_files(
        tmp_path,
        create_processed_dir=False,
    )
    preprocess_movielens_1m(dataset_config)
    _write_baseline_report_fixture(tmp_path / "reports")
    _write_deep_report_fixture(tmp_path / "reports")

    training_config = HybridTrainingConfig(
        dataset=dataset_config,
        reports_dir=tmp_path / "reports",
        models_dir=tmp_path / "models",
        checkpoint_name="hybrid_recommender_test.pt",
        embedding_dim=8,
        hidden_dim=16,
        dropout=0.2,
        batch_size=4,
        epochs=8,
        early_stopping_patience=2,
        learning_rate=1e-3,
        weight_decay=1e-5,
    )

    report = run_hybrid_experiments(training_config)
    captured = capsys.readouterr().out
    report_path = training_config.reports_dir / "hybrid_model_metrics.json"
    checkpoint_path = training_config.models_dir / training_config.checkpoint_name
    comparison_report_path = training_config.reports_dir / "model_comparison.json"

    assert checkpoint_path.exists()
    assert report_path.exists()
    assert comparison_report_path.exists()

    with report_path.open("r", encoding="utf-8") as report_file:
        saved_report = json.load(report_file)
    with comparison_report_path.open("r", encoding="utf-8") as comparison_file:
        comparison_report = json.load(comparison_file)

    assert saved_report == report
    assert saved_report["data_summary"]["train_ratings"] == 16
    assert saved_report["config"]["dropout"] == 0.2
    assert saved_report["config"]["early_stopping_patience"] == 2
    assert "rmse" in saved_report["validation"]
    assert "rmse" in saved_report["test"]
    assert saved_report["checkpoint_path"] == str(checkpoint_path)
    assert saved_report["best_epoch"] >= 1
    assert "stopped_early" in saved_report

    assert comparison_report["baseline_user_item_bias"]["validation_rmse"] == 0.91
    assert comparison_report["deep_model"]["validation_rmse"] == 0.92
    assert (
        comparison_report["hybrid_model"]["validation_rmse"]
        == saved_report["validation"]["rmse"]
    )
    assert "hybrid_model" in comparison_report["available_models"]
    assert "hybrid_vs_baseline" in comparison_report["comparisons"]
    assert "hybrid_vs_deep" in comparison_report["comparisons"]
    assert "best_model" in comparison_report

    assert "Hybrid model training complete" in captured
    assert "validation RMSE:" in captured
    assert "test RMSE:" in captured
    assert "saved comparison report:" in captured
    assert f"saved checkpoint: {checkpoint_path}" in captured
    assert f"saved report: {report_path}" in captured
    assert f"saved comparison report: {comparison_report_path}" in captured


def _write_sample_movielens_files(
    tmp_path,
    create_processed_dir: bool = True,
) -> DatasetConfig:
    """Create a small synthetic MovieLens 1M-style dataset."""

    raw_dir = tmp_path / "data" / "raw" / "ml-1m"
    processed_dir = tmp_path / "data" / "processed" / "ml-1m"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if create_processed_dir:
        processed_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "users.dat").write_text(
        "\n".join(
            [
                "1::F::25::10::48067",
                "2::M::35::16::12345",
            ]
        ),
        encoding="utf-8",
    )

    (raw_dir / "movies.dat").write_text(
        "\n".join(
            [
                "1::Toy Story (1995)::Animation|Children's|Comedy",
                "2::Jumanji (1995)::Adventure|Children's|Fantasy",
                "3::Grumpier Old Men (1995)::Comedy|Romance",
                "4::Waiting to Exhale (1995)::Comedy|Drama",
                "5::Father of the Bride Part II (1995)::Comedy",
                "6::Heat (1995)::Action|Crime|Thriller",
                "7::Sabrina (1995)::Comedy|Romance",
                "8::Tom and Huck (1995)::Adventure|Children's",
                "9::Sudden Death (1995)::Action",
                "10::GoldenEye (1995)::Action|Adventure|Thriller|Sci-Fi",
            ]
        ),
        encoding="latin-1",
    )

    ratings_lines = []
    for user_id, timestamp_start in [(1, 1_000_000_000), (2, 1_000_001_000)]:
        for offset, movie_id in enumerate(range(1, 11), start=1):
            rating = 3 + (movie_id % 3)
            timestamp = timestamp_start + offset
            ratings_lines.append(f"{user_id}::{movie_id}::{rating}::{timestamp}")

    (raw_dir / "ratings.dat").write_text(
        "\n".join(ratings_lines),
        encoding="utf-8",
    )

    return DatasetConfig(raw_dir=raw_dir, processed_dir=processed_dir)


def _write_baseline_report_fixture(reports_dir) -> None:
    """Create a minimal baseline report fixture for comparison testing."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    baseline_report = {
        "baselines": {
            "user_item_bias": {
                "validation": {"rmse": 0.91},
                "test": {"rmse": 0.93},
            }
        }
    }
    with (reports_dir / "baseline_metrics.json").open("w", encoding="utf-8") as report_file:
        json.dump(baseline_report, report_file, indent=2)


def _write_deep_report_fixture(reports_dir) -> None:
    """Create a minimal deep-model report fixture for comparison testing."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    deep_report = {
        "validation": {"rmse": 0.92},
        "test": {"rmse": 0.94},
    }
    with (reports_dir / "deep_model_metrics.json").open("w", encoding="utf-8") as report_file:
        json.dump(deep_report, report_file, indent=2)
