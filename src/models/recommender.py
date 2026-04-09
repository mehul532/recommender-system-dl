"""Minimal recommender model scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Recommendation:
    """A single ranked recommendation result."""

    movie_id: int
    score: float


@dataclass
class HybridRecommender:
    """Starter interface for a hybrid recommender model."""

    name: str = "hybrid-recommender"
    is_trained: bool = False
    training_summary: dict[str, int] = field(default_factory=dict)

    def fit(
        self,
        ratings: list[dict[str, object]],
        movies: list[dict[str, object]],
        users: list[dict[str, object]],
    ) -> None:
        """Record basic dataset counts as placeholder training state."""

        self.training_summary = {
            "ratings": len(ratings),
            "movies": len(movies),
            "users": len(users),
        }
        self.is_trained = True

    def predict_score(self, user_id: int, movie_id: int) -> float:
        """Return a placeholder score until real inference is implemented."""

        _ = (user_id, movie_id)
        return 0.0

    def recommend(self, user_id: int, top_k: int = 10) -> list[Recommendation]:
        """Return placeholder recommendations for the requested user."""

        _ = user_id
        return [
            Recommendation(movie_id=movie_id, score=0.0)
            for movie_id in range(1, max(top_k, 0) + 1)
        ]
