"""Simple baseline recommenders for MovieLens experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PopularityRecommender:
    """Recommend the globally most popular unseen movies."""

    ranked_movies: pd.DataFrame = field(default_factory=pd.DataFrame)

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> "PopularityRecommender":
        """Rank movies by interaction count, then mean rating, then movie index."""

        movie_stats = ratings.groupby("movie_idx", as_index=False).agg(
            interaction_count=("rating", "size"),
            mean_rating=("rating", "mean"),
        )
        ranked_movies = movies[["movie_id", "movie_idx"]].merge(
            movie_stats,
            on="movie_idx",
            how="left",
        )
        ranked_movies["interaction_count"] = (
            ranked_movies["interaction_count"].fillna(0).astype("int64")
        )
        ranked_movies["mean_rating"] = (
            ranked_movies["mean_rating"].fillna(0.0).astype("float64")
        )
        self.ranked_movies = ranked_movies.sort_values(
            ["interaction_count", "mean_rating", "movie_idx"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        return self

    def recommend(
        self,
        user_idx: int,
        seen_movie_idxs: set[int] | None = None,
        top_k: int = 10,
    ) -> list[int]:
        """Return the top-k unseen movie indices for a user."""

        _ = user_idx
        seen_movie_idxs = seen_movie_idxs or set()
        recommended_movie_idxs: list[int] = []

        for row in self.ranked_movies.itertuples(index=False):
            movie_idx = int(row.movie_idx)
            if movie_idx in seen_movie_idxs:
                continue
            recommended_movie_idxs.append(movie_idx)
            if len(recommended_movie_idxs) >= max(top_k, 0):
                break

        return recommended_movie_idxs


@dataclass
class UserItemBiasRecommender:
    """Predict ratings from a global mean plus user and item biases."""

    iterations: int = 10
    user_reg: float = 10.0
    item_reg: float = 10.0
    min_rating: float = 1.0
    max_rating: float = 5.0
    global_mean: float = 0.0
    user_biases: dict[int, float] = field(default_factory=dict)
    item_biases: dict[int, float] = field(default_factory=dict)

    def fit(self, ratings: pd.DataFrame) -> "UserItemBiasRecommender":
        """Fit regularized user and item biases from training ratings."""

        if ratings.empty:
            raise ValueError("Cannot fit a bias baseline with no ratings.")

        self.global_mean = float(ratings["rating"].mean())
        self.user_biases = {
            int(user_idx): 0.0
            for user_idx in sorted(ratings["user_idx"].drop_duplicates().tolist())
        }
        self.item_biases = {
            int(movie_idx): 0.0
            for movie_idx in sorted(ratings["movie_idx"].drop_duplicates().tolist())
        }

        user_groups = {
            int(user_idx): group[["movie_idx", "rating"]]
            for user_idx, group in ratings.groupby("user_idx")
        }
        item_groups = {
            int(movie_idx): group[["user_idx", "rating"]]
            for movie_idx, group in ratings.groupby("movie_idx")
        }

        for _ in range(self.iterations):
            for user_idx, user_ratings in user_groups.items():
                residual_sum = 0.0
                for row in user_ratings.itertuples(index=False):
                    residual_sum += (
                        float(row.rating)
                        - self.global_mean
                        - self.item_biases.get(int(row.movie_idx), 0.0)
                    )
                self.user_biases[user_idx] = residual_sum / (
                    self.user_reg + len(user_ratings)
                )

            for movie_idx, movie_ratings in item_groups.items():
                residual_sum = 0.0
                for row in movie_ratings.itertuples(index=False):
                    residual_sum += (
                        float(row.rating)
                        - self.global_mean
                        - self.user_biases.get(int(row.user_idx), 0.0)
                    )
                self.item_biases[movie_idx] = residual_sum / (
                    self.item_reg + len(movie_ratings)
                )

        return self

    def predict_score(self, user_idx: int, movie_idx: int) -> float:
        """Predict a bounded rating for a user and movie index."""

        score = (
            self.global_mean
            + self.user_biases.get(user_idx, 0.0)
            + self.item_biases.get(movie_idx, 0.0)
        )
        return min(self.max_rating, max(self.min_rating, score))

    def predict(self, ratings: pd.DataFrame) -> pd.Series:
        """Predict ratings for every row in a DataFrame."""

        predictions = [
            self.predict_score(int(row.user_idx), int(row.movie_idx))
            for row in ratings[["user_idx", "movie_idx"]].itertuples(index=False)
        ]
        return pd.Series(predictions, index=ratings.index, dtype="float64")
