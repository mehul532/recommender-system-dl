"""Minimal dataset utilities for MovieLens 1M scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    """Common dataset locations used across the project."""

    data_dir: Path = Path("data/raw/ml-1m")


def load_ratings(config: DatasetConfig | None = None) -> list[dict[str, int | float]]:
    """Return placeholder ratings data."""

    _ = config or DatasetConfig()
    return []


def load_movies(config: DatasetConfig | None = None) -> list[dict[str, str | int]]:
    """Return placeholder movie metadata."""

    _ = config or DatasetConfig()
    return []


def load_users(config: DatasetConfig | None = None) -> list[dict[str, str | int]]:
    """Return placeholder user metadata."""

    _ = config or DatasetConfig()
    return []
