"""Tests for baseline recommenders and the baseline runner."""

from __future__ import annotations

import json

import pandas as pd

from src.data import (
    DatasetConfig,
    ensure_processed_movielens,
    load_processed_movielens,
    preprocess_movielens_1m,
)
from src.models import PopularityRecommender, UserItemBiasRecommender
from src.training import TrainingConfig, run_baseline_experiments


def test_load_processed_movielens_reads_saved_outputs(tmp_path) -> None:
    """Load processed MovieLens outputs with stable dtypes and tables."""

    config = _write_sample_movielens_files(tmp_path)
    preprocess_movielens_1m(config)

    processed = load_processed_movielens(config)

    assert len(processed.ratings_train) == 16
    assert len(processed.ratings_val) == 2
    assert len(processed.ratings_test) == 2
    assert len(processed.users) == 2
    assert len(processed.movies) == 10
    assert len(processed.genre_features) == 10
    assert pd.api.types.is_integer_dtype(processed.ratings_train["user_idx"])
    assert pd.api.types.is_integer_dtype(processed.ratings_train["movie_idx"])
    assert pd.api.types.is_string_dtype(processed.movies["title"])
    assert processed.metadata["num_ratings"] == 20


def test_ensure_processed_movielens_auto_runs_preprocessing(tmp_path, capsys) -> None:
    """Auto-run preprocessing when processed files are missing."""

    config = _write_sample_movielens_files(tmp_path, create_processed_dir=False)

    processed = ensure_processed_movielens(config)
    captured = capsys.readouterr().out

    assert len(processed.ratings_train) == 16
    assert (config.processed_dir / "ratings_train.csv").exists()
    assert "Processed MovieLens data not found. Running preprocessing first." in captured


def test_popularity_recommender_excludes_seen_movies_and_uses_tie_breakers() -> None:
    """Rank by interaction count, then average rating, then movie index."""

    ratings = pd.DataFrame(
        {
            "user_idx": [0, 1, 0, 1, 2],
            "movie_idx": [0, 0, 1, 1, 2],
            "rating": [5.0, 4.0, 4.0, 3.0, 5.0],
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": [1, 2, 3, 4],
            "movie_idx": [0, 1, 2, 3],
        }
    )

    model = PopularityRecommender().fit(ratings=ratings, movies=movies)
    recommendations = model.recommend(user_idx=0, seen_movie_idxs={0}, top_k=3)

    assert recommendations == [1, 2, 3]


def test_user_item_bias_recommender_predicts_bounded_scores() -> None:
    """Learn simple user and item effects while staying in the rating range."""

    ratings = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1],
            "movie_idx": [0, 1, 0, 1],
            "rating": [5.0, 1.0, 4.0, 2.0],
        }
    )

    model = UserItemBiasRecommender(iterations=20, user_reg=1.0, item_reg=1.0).fit(
        ratings
    )

    high_score = model.predict_score(user_idx=0, movie_idx=0)
    low_score = model.predict_score(user_idx=0, movie_idx=1)

    assert 1.0 <= high_score <= 5.0
    assert 1.0 <= low_score <= 5.0
    assert high_score > low_score


def test_run_baseline_experiments_saves_report_and_prints_summary(
    tmp_path, capsys
) -> None:
    """Run the full baseline workflow from raw data to saved metrics."""

    dataset_config = _write_sample_movielens_files(
        tmp_path,
        create_processed_dir=False,
    )
    training_config = TrainingConfig(
        dataset=dataset_config,
        reports_dir=tmp_path / "reports",
        top_k=10,
        relevance_threshold=4.0,
        bias_iterations=10,
        user_reg=10.0,
        item_reg=10.0,
    )

    report = run_baseline_experiments(training_config)
    captured = capsys.readouterr().out
    report_path = training_config.reports_dir / "baseline_metrics.json"

    assert report_path.exists()
    with report_path.open("r", encoding="utf-8") as report_file:
        saved_report = json.load(report_file)

    assert saved_report == report
    assert saved_report["data_summary"]["train_ratings"] == 16
    assert "precision_at_10" in saved_report["baselines"]["popularity"]["validation"]
    assert "precision_at_10" in saved_report["baselines"]["popularity"]["test"]
    assert "rmse" in saved_report["baselines"]["user_item_bias"]["validation"]
    assert "rmse" in saved_report["baselines"]["user_item_bias"]["test"]

    assert "Baseline evaluation complete" in captured
    assert "train ratings: 16" in captured
    assert "validation ratings: 2" in captured
    assert "test ratings: 2" in captured
    assert "popularity Precision@10 validation:" in captured
    assert "user-item bias RMSE test:" in captured
    assert f"saved report: {report_path}" in captured


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
