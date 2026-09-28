"""Feature extraction pipelines and mathematical representations for Kaggriculture."""

from .clustering import cluster_opponent_trajectory
from .global_state import extract_global_vector
from .macro_intents import MACRO_INTENT_CLASSES, classify_macro_intent
from .market import extract_market_vector
from .returns import compute_net_worth, compute_return_to_go, compute_reward
from .spatial import extract_opponent_spatial, extract_spatial_tensor
from .storage import TransitionChunkReader, save_transitions_chunk

__all__ = [
    "extract_spatial_tensor",
    "extract_opponent_spatial",
    "extract_global_vector",
    "extract_market_vector",
    "classify_macro_intent",
    "MACRO_INTENT_CLASSES",
    "compute_net_worth",
    "compute_reward",
    "compute_return_to_go",
    "cluster_opponent_trajectory",
    "save_transitions_chunk",
    "TransitionChunkReader",
]
