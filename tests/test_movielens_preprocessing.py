"""Tests for MovieLens 1M loading and preprocessing."""

from __future__ import annotations

import json
import re

import pandas as pd

from src.data import (
    DatasetConfig,
    load_movies,
    load_ratings,
    load_users,
    preprocess_movielens_1m,
)


def test_loaders_parse_movielens_dat_files(tmp_path) -> None:
    """Load raw MovieLens-style files with the expected separators and dtypes."""

    config = _write_sample_movielens_files(tmp_path)

    ratings = load_ratings(config)
    movies = load_movies(config)
    users = load_users(config)

    assert list(ratings.columns) == ["user_id", "movie_id", "rating", "timestamp"]
    assert list(movies.columns) == ["movie_id", "title", "genres"]
    assert list(users.columns) == [
        "user_id",
        "gender",
        "age",
        "occupation",
        "zip_code",
    ]

    assert ratings.shape == (20, 4)
    assert movies.shape == (10, 3)
    assert users.shape == (2, 5)

    assert pd.api.types.is_integer_dtype(ratings["user_id"])
    assert pd.api.types.is_integer_dtype(ratings["movie_id"])
    assert pd.api.types.is_float_dtype(ratings["rating"])
    assert pd.api.types.is_integer_dtype(ratings["timestamp"])
    assert pd.api.types.is_string_dtype(movies["title"])
    assert pd.api.types.is_string_dtype(users["zip_code"])
    assert movies.loc[0, "title"] == "Toy Story (1995)"


def test_preprocess_movielens_1m_runs_end_to_end(tmp_path, capsys) -> None:
    """Run the full preprocessing flow and verify outputs and metadata."""

    config = _write_sample_movielens_files(tmp_path)

    result = preprocess_movielens_1m(config)
    captured = capsys.readouterr().out

    train_ratings = result["ratings_train"]
    val_ratings = result["ratings_val"]
    test_ratings = result["ratings_test"]
    users = result["users"]
    movies = result["movies"]
    genre_features = result["genre_features"]
    metadata = result["metadata"]
    processed_dir = result["processed_dir"]

    assert len(train_ratings) == 16
    assert len(val_ratings) == 2
    assert len(test_ratings) == 2
    assert len(train_ratings) + len(val_ratings) + len(test_ratings) == 20
    assert _rating_keys(train_ratings).isdisjoint(_rating_keys(val_ratings))
    assert _rating_keys(train_ratings).isdisjoint(_rating_keys(test_ratings))
    assert _rating_keys(val_ratings).isdisjoint(_rating_keys(test_ratings))

    assert users["user_idx"].tolist() == [0, 1]
    assert movies["movie_idx"].tolist() == list(range(10))
    assert genre_features["movie_idx"].tolist() == list(range(10))
    assert "genre_action" in genre_features.columns
    assert "genre_childrens" in genre_features.columns
    assert "genre_sci_fi" in genre_features.columns

    assert metadata["num_ratings"] == 20
    assert metadata["num_users"] == 2
    assert metadata["num_movies"] == 10
    assert metadata["train_count"] == 16
    assert metadata["val_count"] == 2
    assert metadata["test_count"] == 2

    assert (processed_dir / "ratings_train.csv").exists()
    assert (processed_dir / "ratings_val.csv").exists()
    assert (processed_dir / "ratings_test.csv").exists()
    assert (processed_dir / "users.csv").exists()
    assert (processed_dir / "movies.csv").exists()
    assert (processed_dir / "movie_genre_features.csv").exists()
    assert (processed_dir / "metadata.json").exists()

    with (processed_dir / "metadata.json").open("r", encoding="utf-8") as metadata_file:
        saved_metadata = json.load(metadata_file)
    assert saved_metadata["genre_column_map"]["Sci-Fi"] == "genre_sci_fi"

    all_ratings = pd.concat(
        [train_ratings, val_ratings, test_ratings],
        ignore_index=True,
    )
    assert all_ratings.groupby("user_id")["user_idx"].nunique().eq(1).all()
    assert all_ratings.groupby("movie_id")["movie_idx"].nunique().eq(1).all()

    movie_genre_rows = movies[["movie_id", "movie_idx", "genres"]].merge(
        genre_features,
        on=["movie_id", "movie_idx"],
        how="inner",
    )
    assert len(movie_genre_rows) == len(movies)
    feature_columns = [
        column
        for column in genre_features.columns
        if column not in {"movie_id", "movie_idx"}
    ]
    for row in movie_genre_rows.to_dict(orient="records"):
        expected_columns = {
            f"genre_{_slugify_genre_name(genre)}"
            for genre in _parse_genres(row["genres"])
        }
        actual_columns = {
            column for column in feature_columns if row[column] == 1
        }
        assert actual_columns == expected_columns

    assert "MovieLens 1M preprocessing complete" in captured
    assert "ratings shape: (20, 6)" in captured
    assert "train ratings: 16" in captured
    assert "validation ratings: 2" in captured
    assert "test ratings: 2" in captured


def _rating_keys(ratings: pd.DataFrame) -> set[tuple[int, int, int]]:
    """Return a stable key for each processed rating row."""

    return set(
        ratings[["user_id", "movie_id", "timestamp"]].itertuples(
            index=False,
            name=None,
        )
    )


def _parse_genres(genres: str) -> list[str]:
    """Split a MovieLens genre string into individual genre names."""

    if not genres or genres == "(no genres listed)":
        return []
    return [genre.strip() for genre in str(genres).split("|") if genre.strip()]


def _slugify_genre_name(name: str) -> str:
    """Mirror the feature-name slugification used by preprocessing."""

    normalized = name.lower().replace("'", "")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def _write_sample_movielens_files(tmp_path) -> DatasetConfig:
    """Create a small synthetic MovieLens 1M-style dataset."""

    raw_dir = tmp_path / "data" / "raw" / "ml-1m"
    processed_dir = tmp_path / "data" / "processed" / "ml-1m"
    raw_dir.mkdir(parents=True, exist_ok=True)
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
