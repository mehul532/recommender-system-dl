"""Training entrypoints for the recommender scaffold."""

from src.training.deep_train import DeepTrainingConfig, run_deep_experiments
from src.training.train import TrainingConfig, run_baseline_experiments, train_model

__all__ = [
    "DeepTrainingConfig",
    "TrainingConfig",
    "run_baseline_experiments",
    "run_deep_experiments",
    "train_model",
]
