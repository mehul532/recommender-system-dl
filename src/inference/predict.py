"""Inference helpers for the recommender demo app."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from src.data import DatasetConfig, ProcessedMovieLensData, ensure_processed_movielens
from src.models import (
    DeepRecommender,
    HybridDeepRecommender,
    PopularityRecommender,
    Recommendation,
)


DEFAULT_DATASET_CONFIG = DatasetConfig()
DEFAULT_MODEL_PATHS = {
    "deep": Path("models/deep_recommender.pt"),
    "hybrid": Path("models/hybrid_recommender.pt"),
}
MODEL_FAMILIES = {"baseline_popularity", "deep", "hybrid"}
LoadedModel = Union[PopularityRecommender, DeepRecommender, HybridDeepRecommender]


def load_model(
    model_family: str,
    model_path: str | Path | None = None,
    data: ProcessedMovieLensData | None = None,
) -> LoadedModel:
    """Load a trained model family for inference."""

    active_data = data or ensure_processed_movielens(DEFAULT_DATASET_CONFIG)

    if model_family == "baseline_popularity":
        return PopularityRecommender().fit(
            ratings=active_data.ratings_train,
            movies=active_data.movies,
        )

    checkpoint_path = Path(model_path) if model_path is not None else _default_model_path(
        model_family
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint for '{model_family}' not found at {checkpoint_path}. "
            "Train the model before using it in the app."
        )

    num_users = int(active_data.users["user_idx"].max()) + 1
    num_movies = int(active_data.movies["movie_idx"].max()) + 1

    if model_family == "deep":
        return DeepRecommender.load_checkpoint(
            path=checkpoint_path,
            num_users=num_users,
            num_movies=num_movies,
        )
    if model_family == "hybrid":
        return HybridDeepRecommender.load_checkpoint(
            path=checkpoint_path,
            num_users=num_users,
            num_movies=num_movies,
            movie_genre_features=active_data.genre_features,
        )

    raise ValueError(
        f"Unknown model family '{model_family}'. Expected one of: "
        f"{', '.join(sorted(MODEL_FAMILIES))}."
    )


def recommend_for_user(
    user_id: int,
    model_family: str,
    top_k: int = 10,
    model: LoadedModel | None = None,
    data: ProcessedMovieLensData | None = None,
) -> list[Recommendation]:
    """Return top-k recommendations for a user from the selected model family."""

    active_data = data or ensure_processed_movielens(DEFAULT_DATASET_CONFIG)
    active_model = model or load_model(
        model_family=model_family,
        data=active_data,
    )
    if top_k <= 0:
        return []

    user_idx = _lookup_user_idx(active_data, user_id)
    seen_movie_idxs = _get_seen_movie_idxs(active_data, user_idx)

    if model_family == "baseline_popularity":
        return _recommend_with_popularity(
            model=active_model,
            data=active_data,
            user_idx=user_idx,
            seen_movie_idxs=seen_movie_idxs,
            top_k=top_k,
        )

    if model_family in {"deep", "hybrid"}:
        return _recommend_with_rating_model(
            model=active_model,
            data=active_data,
            user_id=user_id,
            user_idx=user_idx,
            seen_movie_idxs=seen_movie_idxs,
            top_k=top_k,
        )

    raise ValueError(
        f"Unknown model family '{model_family}'. Expected one of: "
        f"{', '.join(sorted(MODEL_FAMILIES))}."
    )


def _default_model_path(model_family: str) -> Path:
    """Return the default checkpoint path for a model family."""

    if model_family not in DEFAULT_MODEL_PATHS:
        raise ValueError(
            f"No default checkpoint path configured for model family '{model_family}'."
        )
    return DEFAULT_MODEL_PATHS[model_family]


def _lookup_user_idx(data: ProcessedMovieLensData, user_id: int) -> int:
    """Map a raw user ID to the processed contiguous user index."""

    matched_users = data.users.loc[data.users["user_id"] == user_id, ["user_idx"]]
    if matched_users.empty:
        raise ValueError(f"User ID {user_id} was not found in the processed dataset.")
    return int(matched_users.iloc[0]["user_idx"])


def _get_seen_movie_idxs(
    data: ProcessedMovieLensData,
    user_idx: int,
) -> set[int]:
    """Return all movies already rated by the selected user across all splits."""

    all_ratings = pd.concat(
        [data.ratings_train, data.ratings_val, data.ratings_test],
        ignore_index=True,
    )
    user_history = all_ratings.loc[all_ratings["user_idx"] == user_idx, "movie_idx"]
    return set(user_history.astype("int64").tolist())


def _recommend_with_popularity(
    model: LoadedModel,
    data: ProcessedMovieLensData,
    user_idx: int,
    seen_movie_idxs: set[int],
    top_k: int,
) -> list[Recommendation]:
    """Recommend unseen movies using the popularity baseline."""

    if not isinstance(model, PopularityRecommender):
        raise TypeError("Popularity recommendations require a PopularityRecommender.")

    movie_id_lookup = dict(
        data.movies[["movie_idx", "movie_id"]].itertuples(index=False, name=None)
    )
    score_lookup = dict(
        model.ranked_movies[["movie_idx", "interaction_count"]].itertuples(
            index=False,
            name=None,
        )
    )
    recommended_movie_idxs = model.recommend(
        user_idx=user_idx,
        seen_movie_idxs=seen_movie_idxs,
        top_k=top_k,
    )
    return [
        Recommendation(
            movie_id=int(movie_id_lookup[movie_idx]),
            score=float(score_lookup.get(movie_idx, 0)),
        )
        for movie_idx in recommended_movie_idxs
    ]


def _recommend_with_rating_model(
    model: LoadedModel,
    data: ProcessedMovieLensData,
    user_id: int,
    user_idx: int,
    seen_movie_idxs: set[int],
    top_k: int,
) -> list[Recommendation]:
    """Recommend unseen movies by scoring all unseen candidates."""

    candidate_movies = data.movies.loc[
        ~data.movies["movie_idx"].isin(seen_movie_idxs),
        ["movie_id", "movie_idx"],
    ].copy()
    if candidate_movies.empty:
        return []

    candidate_ratings = pd.DataFrame(
        {
            "user_id": user_id,
            "movie_id": candidate_movies["movie_id"].to_numpy(),
            "rating": 0.0,
            "timestamp": 0,
            "user_idx": user_idx,
            "movie_idx": candidate_movies["movie_idx"].to_numpy(),
        }
    )
    predictions = model.predict(candidate_ratings)
    scored_candidates = candidate_movies.assign(score=predictions.to_numpy())
    top_candidates = scored_candidates.sort_values(
        ["score", "movie_idx"],
        ascending=[False, True],
    ).head(top_k)

    return [
        Recommendation(movie_id=int(row.movie_id), score=float(row.score))
        for row in top_candidates.itertuples(index=False)
    ]
