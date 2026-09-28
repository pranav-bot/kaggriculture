import os
import sys

import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.continuous_train_iql import (  # noqa: E402
    ReplayDataset,
    importance_weights,
    train_continuous,
    weighted_expectile_loss,
)
from scripts.train_iql_value import IQLValueNet, VEC_DIM  # noqa: E402


def _record(target=1.0, **extra):
    return {"spatial": torch.zeros(23, 10, 10), "vec": torch.zeros(VEC_DIM),
            "action": 0, "return": target, **extra}


def test_replay_dataset_validates_and_retains_action():
    dataset = ReplayDataset([_record(), {"return": 1.0}, _record(action=4, bankruptcy=True)])
    assert len(dataset) == 2
    assert dataset[1][3].item() == 4
    assert dataset[1][-1].item() is True


def test_bankruptcy_and_difficult_samples_are_heavily_weighted():
    weights = importance_weights(torch.tensor([0.0, 1.0]), torch.tensor([False, True]))
    assert weights[1] > weights[0] * 10
    assert weighted_expectile_loss(torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([1.0])).item() > 0


def test_empty_live_buffers_are_blocked_without_writing_v2(tmp_path):
    v1 = tmp_path / "iql_weights_v1.pt"
    v2 = tmp_path / "iql_weights_v2.pt"
    torch.save({"state_dict": IQLValueNet().state_dict(), "config": {"vec_dim": VEC_DIM, "hidden": 64}}, v1)
    result = train_continuous(str(v1), str(v2), [tmp_path / "missing"])
    assert result["status"] == "blocked"
    assert not v2.exists()
