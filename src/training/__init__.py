"""Training entrypoints for the recommender scaffold."""

from src.training.train import TrainingConfig, run_baseline_experiments, train_model

__all__ = ["TrainingConfig", "run_baseline_experiments", "train_model"]
