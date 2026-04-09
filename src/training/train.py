"""Minimal training flow for the recommender scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.data import DatasetConfig, load_movies, load_ratings, load_users
from src.models import HybridRecommender


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration for the starter project."""

    dataset: DatasetConfig = DatasetConfig()
    model_output_path: Path = Path("models/hybrid_recommender.pt")


def train_model(config: TrainingConfig | None = None) -> HybridRecommender:
    """Wire the loaded pandas datasets into the placeholder model."""

    active_config = config or TrainingConfig()
    ratings = load_ratings(active_config.dataset)
    movies = load_movies(active_config.dataset)
    users = load_users(active_config.dataset)

    model = HybridRecommender()
    model.fit(ratings=ratings, movies=movies, users=users)
    return model
