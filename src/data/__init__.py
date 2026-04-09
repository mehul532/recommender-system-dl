"""Data loading utilities for the recommender scaffold."""

from src.data.dataset import (
    DatasetConfig,
    ProcessedMovieLensData,
    ensure_processed_movielens,
    load_movies,
    load_processed_movielens,
    load_ratings,
    load_users,
    preprocess_movielens_1m,
)

__all__ = [
    "DatasetConfig",
    "ProcessedMovieLensData",
    "ensure_processed_movielens",
    "load_ratings",
    "load_movies",
    "load_processed_movielens",
    "load_users",
    "preprocess_movielens_1m",
]
