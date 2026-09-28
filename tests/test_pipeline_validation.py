"""Focused validation for the available continuous-training pipeline pieces."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.replay_parser import parse_replay
from kaggriculture.data.dataloader import create_dataloader
from scripts import continuous_train_iql as continuous


def _write_replay(path: Path) -> None:
    document = {
        "id": "synthetic-replay",
        "info": {"seed": 17},
        "configuration": {"seed": 18, "episodeSteps": 2},
        "steps": [
            [
                {
                    "action": {"farmer": ["PASS"]},
                    "observation": {
                        "player": 0,
                        "step": 0,
                        "farms": [
                            {"money": 100, "hands": [], "unlocked_quadrants": []},
                            {"money": 80, "hands": [], "unlocked_quadrants": []},
                        ],
                        "private": {
                            "inventories": [{"WHEAT": 2}],
                            "seeds": {"WHEAT": 3},
                            "shed": {},
                        },
                    },
                    "reward": 0,
                },
                {
                    "action": {"farmer": ["PASS"]},
                    "observation": {
                        "player": 1,
                        "step": 0,
                        "farms": [
                            {"money": 100, "hands": [], "unlocked_quadrants": []},
                            {"money": 80, "hands": [], "unlocked_quadrants": []},
                        ],
                        "private": {"inventories": [{}], "seeds": {}, "shed": {}},
                    },
                    "reward": 0,
                },
            ],
            [
                {
                    "action": {"farmer": ["HARVEST"]},
                    "observation": {
                        "player": 0,
                        "step": 1,
                        "farms": [
                            {"money": 125, "hands": [{"id": 1}], "unlocked_quadrants": [1]},
                            {"money": 80, "hands": [], "unlocked_quadrants": []},
                        ],
                        "private": {
                            "inventories": [{"WHEAT": 5}],
                            "seeds": {"WHEAT": 1},
                            "shed": {"WHEAT": 2},
                        },
                    },
                    "reward": 25,
                    "status": "DONE",
                },
                {
                    "action": {"farmer": ["PASS"]},
                    "observation": {
                        "player": 1,
                        "step": 1,
                        "farms": [
                            {"money": 125, "hands": [{"id": 1}], "unlocked_quadrants": [1]},
                            {"money": 90, "hands": [], "unlocked_quadrants": []},
                        ],
                        "private": {"inventories": [{"WHEAT": 1}], "seeds": {}, "shed": {}},
                    },
                    "reward": 10,
                    "status": "DONE",
                },
            ],
        ],
    }
    path.write_text(json.dumps(document))


def _write_chunk(path: Path, start: int, count: int) -> None:
    np.savez(
        path,
        state_spatial=np.full((count, 23, 10, 10), start, dtype=np.float32),
        opponent_spatial=np.full((count, 20, 10, 10), start, dtype=np.float32),
        market=np.full((count, 35), start, dtype=np.float32),
        action=np.arange(start, start + count, dtype=np.int64),
        reward=np.arange(start, start + count, dtype=np.float32),
        return_to_go=np.arange(start, start + count, dtype=np.float32),
        strategy_cluster=np.arange(start, start + count, dtype=np.int64),
        counterfactual_value=np.arange(start, start + count, dtype=np.float32),
    )


def test_replay_ingestion_preserves_order_and_deltas(tmp_path: Path):
    replay = tmp_path / "replay.json"
    _write_replay(replay)

    record = parse_replay(replay)

    assert record.seed == 17
    assert record.players == 2
    assert [(turn.turn, turn.player) for turn in record.turns] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    player_zero = record.turns[2]
    assert player_zero.cash_delta == pytest.approx(25)
    assert player_zero.inventory_delta["carried"] == {"WHEAT": 3}
    assert player_zero.inventory_delta["seeds"] == {"WHEAT": -2}
    assert record.terminal_rewards == {"0": 25.0, "1": 10.0}


def test_dataloader_batches_across_chunk_boundaries(tmp_path: Path):
    _write_chunk(tmp_path / "chunk_0000.npz", start=0, count=3)
    _write_chunk(tmp_path / "chunk_0001.npz", start=3, count=2)

    loader = create_dataloader(tmp_path, batch_size=4, shuffle=False)
    batches = list(loader)

    assert [batch[0].shape[0] for batch in batches] == [4, 1]
    assert batches[0][3].tolist() == [0, 1, 2, 3]
    assert batches[1][3].tolist() == [4]


def _training_record(target: float, *, prediction_loss: float = 0.0, bankruptcy: bool = False):
    return {
        "spatial": np.zeros((23, 10, 10), dtype=np.float32).tolist(),
        "vec": np.zeros(65, dtype=np.float32).tolist(),
        "return": target,
        "action": 0,
        "prediction_loss": prediction_loss,
        "bankruptcy": bankruptcy,
    }


class _TinyValueModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(65, 1)

    def forward(self, spatial, vec):
        return self.head(vec).squeeze(-1)


def test_live_loss_weights_and_weighted_expectile_are_normalized():
    prediction_loss = torch.tensor([0.0, 1.0, 0.0])
    bankruptcy = torch.tensor([False, False, True])
    assert continuous.importance_weights(prediction_loss, bankruptcy).tolist() == [1.0, 7.0, 11.0]

    pred = torch.zeros(2)
    target = torch.tensor([1.0, -1.0])
    weights = torch.tensor([1.0, 3.0])
    expected = (1.0 * 0.7 + 3.0 * 0.3) / 4.0
    assert continuous.weighted_expectile_loss(pred, target, weights).item() == pytest.approx(expected)


def test_continuous_training_publishes_v2_only_after_holdout_improves(tmp_path: Path, monkeypatch):
    v1 = tmp_path / "v1.pt"
    v2 = tmp_path / "v2.pt"
    replay = tmp_path / "replay.json"
    v1.write_bytes(b"placeholder")
    replay.write_text(json.dumps([_training_record(1.0), _training_record(2.0)]))

    model = _TinyValueModel()
    monkeypatch.setattr(continuous, "load_model", lambda path: (model, {"tau": 0.7}))
    scores = iter([2.0, 1.0])
    monkeypatch.setattr(continuous, "evaluate_mse", lambda model, loader: next(scores))

    result = continuous.train_continuous(
        v1,
        v2,
        [replay],
        epochs=1,
        batch_size=1,
        seed=0,
    )

    assert result["status"] == "improved"
    assert result["saved"] is True
    assert v2.is_file()
    bundle = torch.load(v2, map_location="cpu", weights_only=False)
    assert bundle["config"]["base_checkpoint"] == str(v1)
    assert bundle["config"]["holdout_mse"] == 1.0


def test_continuous_training_keeps_existing_v2_when_holdout_not_better(tmp_path: Path, monkeypatch):
    v1 = tmp_path / "v1.pt"
    v2 = tmp_path / "v2.pt"
    replay = tmp_path / "replay.json"
    v1.write_bytes(b"placeholder")
    v2.write_bytes(b"existing")
    replay.write_text(json.dumps([_training_record(1.0), _training_record(2.0)]))

    model = _TinyValueModel()
    monkeypatch.setattr(continuous, "load_model", lambda path: (model, {"tau": 0.7}))
    scores = iter([1.0, 1.0])
    monkeypatch.setattr(continuous, "evaluate_mse", lambda model, loader: next(scores))

    result = continuous.train_continuous(v1, v2, [replay], epochs=1, batch_size=1, seed=0)

    assert result["status"] == "unchanged"
    assert result["saved"] is False
    assert v2.read_bytes() == b"existing"
