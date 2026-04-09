"""MovieLens 1M loading and preprocessing utilities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


RATINGS_FILENAME = "ratings.dat"
MOVIES_FILENAME = "movies.dat"
USERS_FILENAME = "users.dat"


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset locations for raw and processed MovieLens files."""

    raw_dir: Path = Path("data/raw/ml-1m")
    processed_dir: Path = Path("data/processed/ml-1m")

    @property
    def data_dir(self) -> Path:
        """Backward-compatible alias for the raw data directory."""

        return self.raw_dir


def load_ratings(config: DatasetConfig | None = None) -> pd.DataFrame:
    """Load MovieLens 1M ratings into a typed DataFrame."""

    active_config = config or DatasetConfig()
    ratings_path = _require_file(active_config.raw_dir / RATINGS_FILENAME)
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        keep_default_na=False,
    )
    ratings = ratings.astype(
        {
            "user_id": "int64",
            "movie_id": "int64",
            "rating": "float32",
            "timestamp": "int64",
        }
    )
    return ratings.sort_values(["user_id", "timestamp", "movie_id"]).reset_index(
        drop=True
    )


def load_movies(config: DatasetConfig | None = None) -> pd.DataFrame:
    """Load MovieLens 1M movie metadata into a typed DataFrame."""

    active_config = config or DatasetConfig()
    movies_path = _require_file(active_config.raw_dir / MOVIES_FILENAME)
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        header=None,
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
        keep_default_na=False,
    )
    movies = movies.astype({"movie_id": "int64"})
    movies["title"] = movies["title"].astype("string")
    movies["genres"] = movies["genres"].astype("string")
    return movies.sort_values("movie_id").reset_index(drop=True)


def load_users(config: DatasetConfig | None = None) -> pd.DataFrame:
    """Load MovieLens 1M user metadata into a typed DataFrame."""

    active_config = config or DatasetConfig()
    users_path = _require_file(active_config.raw_dir / USERS_FILENAME)
    users = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        keep_default_na=False,
    )
    users = users.astype(
        {
            "user_id": "int64",
            "gender": "string",
            "age": "int64",
            "occupation": "int64",
            "zip_code": "string",
        }
    )
    return users.sort_values("user_id").reset_index(drop=True)


def preprocess_movielens_1m(
    config: DatasetConfig | None = None,
) -> dict[str, pd.DataFrame | dict[str, object] | Path]:
    """Run the full MovieLens 1M preprocessing pipeline."""

    active_config = config or DatasetConfig()
    ratings = load_ratings(active_config)
    movies = load_movies(active_config)
    users = load_users(active_config)

    _validate_rating_references(ratings=ratings, movies=movies, users=users)

    user_mapping = _build_index_mapping(users["user_id"], "user_id", "user_idx")
    movie_mapping = _build_index_mapping(movies["movie_id"], "movie_id", "movie_idx")

    users_processed = (
        users.merge(user_mapping, on="user_id", how="left")
        .astype({"user_idx": "int64"})
        .sort_values(["user_idx", "user_id"])
        .reset_index(drop=True)
    )
    movies_processed = (
        movies.merge(movie_mapping, on="movie_id", how="left")
        .astype({"movie_idx": "int64"})
        .sort_values(["movie_idx", "movie_id"])
        .reset_index(drop=True)
    )
    ratings_processed = (
        ratings.merge(user_mapping, on="user_id", how="left")
        .merge(movie_mapping, on="movie_id", how="left")
        .astype({"user_idx": "int64", "movie_idx": "int64"})
        .sort_values(["user_id", "timestamp", "movie_id"])
        .reset_index(drop=True)
    )

    genre_features, genre_column_map = _build_genre_features(movies_processed)
    train_ratings, val_ratings, test_ratings = _split_ratings_by_user(
        ratings_processed
    )
    metadata = _build_metadata(
        raw_dir=active_config.raw_dir,
        processed_dir=active_config.processed_dir,
        ratings=ratings_processed,
        train_ratings=train_ratings,
        val_ratings=val_ratings,
        test_ratings=test_ratings,
        users=users_processed,
        movies=movies_processed,
        genre_features=genre_features,
        genre_column_map=genre_column_map,
    )

    _save_processed_outputs(
        processed_dir=active_config.processed_dir,
        users=users_processed,
        movies=movies_processed,
        genre_features=genre_features,
        train_ratings=train_ratings,
        val_ratings=val_ratings,
        test_ratings=test_ratings,
        metadata=metadata,
    )
    _print_summary(
        ratings=ratings_processed,
        users=users_processed,
        movies=movies_processed,
        genre_features=genre_features,
        train_ratings=train_ratings,
        val_ratings=val_ratings,
        test_ratings=test_ratings,
        processed_dir=active_config.processed_dir,
    )

    return {
        "ratings": ratings_processed,
        "ratings_train": train_ratings,
        "ratings_val": val_ratings,
        "ratings_test": test_ratings,
        "users": users_processed,
        "movies": movies_processed,
        "genre_features": genre_features,
        "metadata": metadata,
        "processed_dir": active_config.processed_dir,
    }


def _require_file(path: Path) -> Path:
    """Ensure an expected raw dataset file exists."""

    if not path.exists():
        raise FileNotFoundError(
            f"Expected MovieLens 1M file at {path}. "
            f"Place {RATINGS_FILENAME}, {MOVIES_FILENAME}, and {USERS_FILENAME} "
            "under data/raw/ml-1m/."
        )
    return path


def _validate_rating_references(
    ratings: pd.DataFrame, movies: pd.DataFrame, users: pd.DataFrame
) -> None:
    """Validate that all ratings reference known users and movies."""

    missing_user_ids = sorted(set(ratings["user_id"]) - set(users["user_id"]))
    missing_movie_ids = sorted(set(ratings["movie_id"]) - set(movies["movie_id"]))

    if missing_user_ids:
        preview = ", ".join(str(user_id) for user_id in missing_user_ids[:5])
        raise ValueError(f"ratings.dat contains unknown user IDs: {preview}")
    if missing_movie_ids:
        preview = ", ".join(str(movie_id) for movie_id in missing_movie_ids[:5])
        raise ValueError(f"ratings.dat contains unknown movie IDs: {preview}")


def _build_index_mapping(
    ids: pd.Series, raw_id_column: str, index_column: str
) -> pd.DataFrame:
    """Create a stable contiguous index mapping from sorted raw IDs."""

    unique_ids = sorted(ids.drop_duplicates().tolist())
    mapping = pd.DataFrame({raw_id_column: unique_ids})
    mapping[index_column] = range(len(mapping))
    return mapping.astype({raw_id_column: "int64", index_column: "int64"})


def _build_genre_features(
    movies: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Create a reusable multi-hot genre feature table."""

    genre_lists = movies["genres"].fillna("").map(_parse_genres)
    unique_genres = sorted({genre for genres in genre_lists for genre in genres})
    genre_column_map = {
        genre: f"genre_{_slugify_name(genre)}" for genre in unique_genres
    }

    genre_features = movies[["movie_id", "movie_idx"]].copy()
    for genre, column_name in genre_column_map.items():
        genre_features[column_name] = genre_lists.map(
            lambda genres: int(genre in genres)
        ).astype("int8")

    return genre_features, genre_column_map


def _parse_genres(genres: str) -> list[str]:
    """Split a MovieLens genre string into individual genre names."""

    if not genres or genres == "(no genres listed)":
        return []
    return [genre.strip() for genre in str(genres).split("|") if genre.strip()]


def _slugify_name(name: str) -> str:
    """Convert a label into a student-friendly snake_case feature name."""

    normalized = name.lower().replace("'", "")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def _split_ratings_by_user(
    ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ratings into per-user chronological train/validation/test sets."""

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for user_id, user_ratings in ratings.groupby("user_id", sort=False):
        user_ratings = user_ratings.sort_values(["timestamp", "movie_id"]).reset_index(
            drop=True
        )
        rating_count = len(user_ratings)
        if rating_count < 3:
            raise ValueError(
                f"User {user_id} has only {rating_count} ratings. "
                "At least 3 ratings are required for train/validation/test splitting."
            )

        holdout_count = max(1, int(rating_count * 0.1))
        train_end = rating_count - (2 * holdout_count)
        val_end = rating_count - holdout_count

        train_parts.append(user_ratings.iloc[:train_end])
        val_parts.append(user_ratings.iloc[train_end:val_end])
        test_parts.append(user_ratings.iloc[val_end:])

    train_ratings = pd.concat(train_parts, ignore_index=True)
    val_ratings = pd.concat(val_parts, ignore_index=True)
    test_ratings = pd.concat(test_parts, ignore_index=True)
    return train_ratings, val_ratings, test_ratings


def _build_metadata(
    raw_dir: Path,
    processed_dir: Path,
    ratings: pd.DataFrame,
    train_ratings: pd.DataFrame,
    val_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    genre_features: pd.DataFrame,
    genre_column_map: dict[str, str],
) -> dict[str, object]:
    """Assemble a compact preprocessing summary."""

    return {
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "num_ratings": int(len(ratings)),
        "num_users": int(len(users)),
        "num_movies": int(len(movies)),
        "train_count": int(len(train_ratings)),
        "val_count": int(len(val_ratings)),
        "test_count": int(len(test_ratings)),
        "ratings_columns": ratings.columns.tolist(),
        "user_columns": users.columns.tolist(),
        "movie_columns": movies.columns.tolist(),
        "genre_feature_columns": genre_features.columns.tolist(),
        "genre_column_map": genre_column_map,
    }


def _save_processed_outputs(
    processed_dir: Path,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    genre_features: pd.DataFrame,
    train_ratings: pd.DataFrame,
    val_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    """Save processed tables and metadata under the configured output directory."""

    processed_dir.mkdir(parents=True, exist_ok=True)
    users.to_csv(processed_dir / "users.csv", index=False)
    movies.to_csv(processed_dir / "movies.csv", index=False)
    genre_features.to_csv(processed_dir / "movie_genre_features.csv", index=False)
    train_ratings.to_csv(processed_dir / "ratings_train.csv", index=False)
    val_ratings.to_csv(processed_dir / "ratings_val.csv", index=False)
    test_ratings.to_csv(processed_dir / "ratings_test.csv", index=False)

    metadata_path = processed_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def _print_summary(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    genre_features: pd.DataFrame,
    train_ratings: pd.DataFrame,
    val_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    processed_dir: Path,
) -> None:
    """Print a compact summary of preprocessing results."""

    print("MovieLens 1M preprocessing complete")
    print(f"ratings shape: {ratings.shape}")
    print(f"users shape: {users.shape}")
    print(f"movies shape: {movies.shape}")
    print(f"genre features shape: {genre_features.shape}")
    print(f"train ratings: {len(train_ratings)}")
    print(f"validation ratings: {len(val_ratings)}")
    print(f"test ratings: {len(test_ratings)}")
    print(f"saved to: {processed_dir}")


def main() -> None:
    """Run preprocessing with default paths."""

    preprocess_movielens_1m()


if __name__ == "__main__":
    main()
