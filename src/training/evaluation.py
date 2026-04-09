"""Evaluation helpers for baseline recommenders."""

from __future__ import annotations

from math import sqrt

import pandas as pd

from src.models import PopularityRecommender, UserItemBiasRecommender


def evaluate_rmse(
    model: UserItemBiasRecommender,
    ratings: pd.DataFrame,
) -> dict[str, float | int]:
    """Evaluate RMSE for explicit rating prediction."""

    if ratings.empty:
        return {"rmse": 0.0, "count": 0}

    predictions = model.predict(ratings)
    errors = ratings["rating"].astype("float64") - predictions.astype("float64")
    rmse = sqrt(float((errors**2).mean()))
    return {"rmse": rmse, "count": int(len(ratings))}


def evaluate_precision_at_k(
    model: PopularityRecommender,
    eval_ratings: pd.DataFrame,
    seen_ratings: pd.DataFrame,
    top_k: int = 10,
    relevance_threshold: float = 4.0,
) -> dict[str, float | int]:
    """Evaluate Precision@k for held-out recommendation ranking."""

    if eval_ratings.empty:
        return {
            "precision": 0.0,
            "evaluated_users": 0,
            "users_with_relevant_items": 0,
        }

    seen_by_user = {
        int(user_idx): set(group["movie_idx"].astype("int64").tolist())
        for user_idx, group in seen_ratings.groupby("user_idx")
    }
    relevant_ratings = eval_ratings[eval_ratings["rating"] >= relevance_threshold]
    relevant_by_user = {
        int(user_idx): set(group["movie_idx"].astype("int64").tolist())
        for user_idx, group in relevant_ratings.groupby("user_idx")
    }

    user_precisions: list[float] = []
    evaluated_users = sorted(eval_ratings["user_idx"].astype("int64").unique().tolist())
    users_with_relevant_items = 0

    for user_idx in evaluated_users:
        relevant_items = relevant_by_user.get(user_idx, set())
        if relevant_items:
            users_with_relevant_items += 1

        recommended_items = model.recommend(
            user_idx=user_idx,
            seen_movie_idxs=seen_by_user.get(user_idx, set()),
            top_k=top_k,
        )
        hits = len(set(recommended_items) & relevant_items)
        user_precisions.append(hits / top_k)

    precision = sum(user_precisions) / len(user_precisions)
    return {
        "precision": precision,
        "evaluated_users": int(len(evaluated_users)),
        "users_with_relevant_items": int(users_with_relevant_items),
    }
