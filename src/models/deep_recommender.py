"""Simple PyTorch deep recommender for rating prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class RatingMLP(nn.Module):
    """Embed users and movies, then predict ratings with a small MLP."""

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, user_indices: torch.Tensor, movie_indices: torch.Tensor
    ) -> torch.Tensor:
        """Predict raw ratings for a batch of user and movie indices."""

        user_vectors = self.user_embedding(user_indices)
        movie_vectors = self.movie_embedding(movie_indices)
        features = torch.cat([user_vectors, movie_vectors], dim=1)
        mlp_score = self.mlp(features).squeeze(1)
        user_bias = self.user_bias(user_indices).squeeze(1)
        movie_bias = self.movie_bias(movie_indices).squeeze(1)
        return self.global_bias + user_bias + movie_bias + mlp_score


@dataclass
class DeepRecommender:
    """Student-friendly wrapper around the PyTorch rating model."""

    num_users: int
    num_movies: int
    embedding_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.1
    batch_size: int = 1024
    epochs: int = 10
    early_stopping_patience: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    min_rating: float = 1.0
    max_rating: float = 5.0
    device: str | None = None
    model: RatingMLP = field(init=False)
    training_history: list[dict[str, float | int]] = field(default_factory=list)
    best_epoch: int = 0
    best_validation_rmse: float = float("inf")
    stopped_early: bool = False

    def __post_init__(self) -> None:
        """Initialize the torch model and device."""

        resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = resolved_device
        self.model = RatingMLP(
            num_users=self.num_users,
            num_movies=self.num_movies,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)

    def fit(self, train_ratings: pd.DataFrame, val_ratings: pd.DataFrame) -> "DeepRecommender":
        """Train the deep recommender and track validation RMSE."""

        if train_ratings.empty:
            raise ValueError("Cannot train the deep recommender with no training ratings.")

        train_loader = self._build_dataloader(train_ratings, shuffle=True)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.MSELoss()
        best_state_dict = None
        self.training_history = []
        self.best_epoch = 0
        self.best_validation_rmse = float("inf")
        self.stopped_early = False
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0

            for user_indices, movie_indices, ratings in train_loader:
                optimizer.zero_grad()
                predictions = self.model(user_indices, movie_indices)
                loss = loss_fn(predictions, ratings)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                batch_count += 1

            validation_rmse = self._compute_rmse(val_ratings)
            mean_train_loss = epoch_loss / max(batch_count, 1)
            self.training_history.append(
                {
                    "epoch": epoch,
                    "train_loss": mean_train_loss,
                    "validation_rmse": validation_rmse,
                }
            )

            if validation_rmse < self.best_validation_rmse - 1e-6:
                self.best_validation_rmse = validation_rmse
                self.best_epoch = epoch
                epochs_without_improvement = 0
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.early_stopping_patience:
                    self.stopped_early = True
                    break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        return self

    def predict(self, ratings: pd.DataFrame) -> pd.Series:
        """Predict ratings for each row of a ratings DataFrame."""

        if ratings.empty:
            return pd.Series(dtype="float64")

        dataloader = self._build_dataloader(ratings, shuffle=False)
        predictions: list[float] = []

        self.model.eval()
        with torch.no_grad():
            for user_indices, movie_indices, _ in dataloader:
                batch_predictions = self.model(user_indices, movie_indices)
                batch_predictions = batch_predictions.clamp(self.min_rating, self.max_rating)
                predictions.extend(batch_predictions.cpu().tolist())

        return pd.Series(predictions, index=ratings.index, dtype="float64")

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save the current model state and configuration."""

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": {
                    "num_users": self.num_users,
                    "num_movies": self.num_movies,
                    "embedding_dim": self.embedding_dim,
                    "hidden_dim": self.hidden_dim,
                    "dropout": self.dropout,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "early_stopping_patience": self.early_stopping_patience,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "min_rating": self.min_rating,
                    "max_rating": self.max_rating,
                    "device": self.device,
                },
                "best_epoch": self.best_epoch,
                "best_validation_rmse": self.best_validation_rmse,
                "stopped_early": self.stopped_early,
                "training_history": self.training_history,
            },
            checkpoint_path,
        )
        return checkpoint_path

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        num_users: int,
        num_movies: int,
    ) -> "DeepRecommender":
        """Load a deep recommender from a saved checkpoint."""

        checkpoint = torch.load(Path(path), map_location="cpu")
        config = checkpoint["config"]
        model = cls(
            num_users=num_users,
            num_movies=num_movies,
            embedding_dim=int(config["embedding_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            dropout=float(config.get("dropout", 0.1)),
            batch_size=int(config["batch_size"]),
            epochs=int(config["epochs"]),
            early_stopping_patience=int(config.get("early_stopping_patience", 2)),
            learning_rate=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
            min_rating=float(config["min_rating"]),
            max_rating=float(config["max_rating"]),
            device="cpu",
        )
        model.model.load_state_dict(checkpoint["model_state_dict"])
        model.best_epoch = int(checkpoint.get("best_epoch", 0))
        model.best_validation_rmse = float(
            checkpoint.get("best_validation_rmse", float("inf"))
        )
        model.stopped_early = bool(checkpoint.get("stopped_early", False))
        model.training_history = list(checkpoint.get("training_history", []))
        model.model.eval()
        return model

    def _build_dataloader(self, ratings: pd.DataFrame, shuffle: bool) -> DataLoader:
        """Create a torch DataLoader from a ratings DataFrame."""

        user_tensor = torch.tensor(
            ratings["user_idx"].to_numpy(),
            dtype=torch.long,
            device=self.device,
        )
        movie_tensor = torch.tensor(
            ratings["movie_idx"].to_numpy(),
            dtype=torch.long,
            device=self.device,
        )
        rating_tensor = torch.tensor(
            ratings["rating"].to_numpy(),
            dtype=torch.float32,
            device=self.device,
        )
        dataset = TensorDataset(user_tensor, movie_tensor, rating_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _compute_rmse(self, ratings: pd.DataFrame) -> float:
        """Compute RMSE on a ratings DataFrame."""

        predictions = self.predict(ratings)
        if predictions.empty:
            return 0.0
        errors = ratings["rating"].astype("float64") - predictions
        return float(sqrt(float((errors**2).mean())))
