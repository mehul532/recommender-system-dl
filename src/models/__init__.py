"""Model interfaces for the recommender scaffold."""

from src.models.baselines import PopularityRecommender, UserItemBiasRecommender
from src.models.deep_recommender import DeepRecommender
from src.models.hybrid_recommender import HybridDeepRecommender
from src.models.recommender import HybridRecommender, Recommendation

__all__ = [
    "DeepRecommender",
    "HybridDeepRecommender",
    "HybridRecommender",
    "PopularityRecommender",
    "Recommendation",
    "UserItemBiasRecommender",
]
