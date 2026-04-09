"""Model interfaces for the recommender scaffold."""

from src.models.baselines import PopularityRecommender, UserItemBiasRecommender
from src.models.recommender import HybridRecommender, Recommendation

__all__ = [
    "HybridRecommender",
    "PopularityRecommender",
    "Recommendation",
    "UserItemBiasRecommender",
]
