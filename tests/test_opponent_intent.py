import numpy as np

from kaggriculture.meta import MetaController
from kaggriculture.models.opponent_intent import (
    OPPONENT_INTENTS,
    OpponentIntentModel,
    encode_opponent_observation,
)


def _obs(cash=3000.0, plant=False):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    if plant:
        tiles[2][2] = {"kind": "PLANT", "crop": "MELON", "yield_units": 2}
    return {
        "hour": 6,
        "farms": [
            {},
            {"money": cash, "tiles": tiles, "unlocked_quadrants": ["NW"]},
        ],
        "market": {"prices": {"MELON": 100}, "inventory": {"MELON": 4}},
    }


def test_public_encoder_and_cpu_model_are_compact_and_normalized():
    grid, public = encode_opponent_observation(_obs(), 0)
    assert grid.shape == (20, 10, 10)
    assert public.ndim == 1
    result = OpponentIntentModel(use_torch=False).predict_proba([_obs(), _obs(5000, True)])
    assert tuple(result) == OPPONENT_INTENTS
    np.testing.assert_allclose(sum(result.values()), 1.0)
    assert all(0.0 <= value <= 1.0 for value in result.values())


def test_meta_controller_exposes_intent_distribution():
    result = MetaController().get_opponent_intent([_obs()])
    assert set(result) == set(OPPONENT_INTENTS)
    np.testing.assert_allclose(sum(result.values()), 1.0)
