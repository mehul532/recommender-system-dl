"""Minimal prediction helpers for the recommender scaffold."""

from __future__ import annotations

from pathlib import Path

from src.models import HybridRecommender, Recommendation


def load_model(model_path: str | Path | None = None) -> HybridRecommender:
    """Return a placeholder model instance for future loading logic."""

    _ = model_path
    model = HybridRecommender()
    model.is_trained = True
    return model


def recommend_for_user(
    user_id: int,
    model: HybridRecommender | None = None,
    top_k: int = 10,
) -> list[Recommendation]:
    """Return top-k placeholder recommendations for a user."""

    active_model = model or load_model()
    return active_model.recommend(user_id=user_id, top_k=top_k)
