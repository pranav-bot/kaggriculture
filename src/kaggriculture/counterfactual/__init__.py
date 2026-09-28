"""Counterfactual rollout engine and Rust simulator IPC bridge."""

from .engine import CounterfactualRolloutEngine
from .generator import (
    CounterfactualPipeline,
    CounterfactualRecord,
    CounterfactualUnavailableError,
    create_hold_action,
    detect_large_sells,
    generate_counterfactuals,
    run_counterfactual_pipeline,
)
from .ipc import KaggServeProcess
from .serializer import extract_engine_state, serialize_market_order

__all__ = [
    "KaggServeProcess",
    "extract_engine_state",
    "serialize_market_order",
    "CounterfactualRolloutEngine",
    "generate_counterfactuals",
    "CounterfactualPipeline",
    "run_counterfactual_pipeline",
    "detect_large_sells",
    "create_hold_action",
    "CounterfactualRecord",
    "CounterfactualUnavailableError",
]
