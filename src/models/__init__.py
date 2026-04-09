"""Model interfaces for the recommender scaffold."""

from src.models.baselines import PopularityRecommender, UserItemBiasRecommender
from src.models.deep_recommender import DeepRecommender
from src.models.recommender import HybridRecommender, Recommendation

__all__ = [
    "DeepRecommender",
    "HybridRecommender",
    "PopularityRecommender",
    "Recommendation",
    "UserItemBiasRecommender",
]
