"""Smoke tests for the starter project scaffold."""

from src.app import run_app
from src.data import (
    DatasetConfig,
    ProcessedMovieLensData,
    ensure_processed_movielens,
    load_movies,
    load_processed_movielens,
    load_ratings,
    load_users,
    preprocess_movielens_1m,
)
from src.inference import load_model, recommend_for_user
from src.models import (
    DeepRecommender,
    HybridRecommender,
    PopularityRecommender,
    Recommendation,
    UserItemBiasRecommender,
)
from src.training import (
    DeepTrainingConfig,
    TrainingConfig,
    run_baseline_experiments,
    run_deep_experiments,
    train_model,
)


def test_public_symbols_are_importable() -> None:
    """Ensure the starter package exports remain import-safe."""

    assert DatasetConfig is not None
    assert ProcessedMovieLensData is not None
    assert DeepTrainingConfig is not None
    assert TrainingConfig is not None
    assert DeepRecommender is not None
    assert HybridRecommender is not None
    assert PopularityRecommender is not None
    assert Recommendation is not None
    assert UserItemBiasRecommender is not None
    assert callable(ensure_processed_movielens)
    assert callable(load_ratings)
    assert callable(load_movies)
    assert callable(load_processed_movielens)
    assert callable(load_users)
    assert callable(preprocess_movielens_1m)
    assert callable(run_baseline_experiments)
    assert callable(run_deep_experiments)
    assert callable(train_model)
    assert callable(load_model)
    assert callable(recommend_for_user)
    assert callable(run_app)
