"""Data loading utilities for the recommender scaffold."""

from src.data.dataset import (
    DatasetConfig,
    load_movies,
    load_ratings,
    load_users,
    preprocess_movielens_1m,
)

__all__ = [
    "DatasetConfig",
    "load_ratings",
    "load_movies",
    "load_users",
    "preprocess_movielens_1m",
]
