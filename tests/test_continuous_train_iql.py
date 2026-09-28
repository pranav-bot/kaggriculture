import os
import sys

import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from scripts.continuous_train_iql import (  # noqa: E402
    ReplayDataset,
    create_live_replay_dataloader,
    extract_tuples_from_kaggle_replay,
    importance_weights,
    train_continuous,
    weighted_expectile_loss,
)
from scripts.train_iql_value import IQLValueNet, VEC_DIM, SPATIAL_CHANNELS  # noqa: E402


def _record(target=1.0, **extra):
    return {
        "spatial": torch.zeros(SPATIAL_CHANNELS, 10, 10),
        "vec": torch.zeros(VEC_DIM),
        "action": 0,
        "return": target,
        **extra,
    }


def test_replay_dataset_validates_and_retains_action():
    dataset = ReplayDataset([
        _record(),
        {"return": 1.0},  # Invalid: missing spatial & vec
        _record(action=4, bankruptcy=True, is_loss=True),
    ])
    assert len(dataset) == 2
    assert dataset[1][3].item() == 4
    # Check bankruptcy (index 5) and is_loss (index 6)
    assert dataset[1][5].item() is True
    assert dataset[1][6].item() is True


def test_importance_weights_overestimation_and_loss():
    # Normal state with perfect prediction
    pred_normal = torch.tensor([1.0])
    target_normal = torch.tensor([1.0])
    pred_loss = torch.tensor([0.0])
    bankrupt = torch.tensor([False])
    is_loss = torch.tensor([False])

    w_normal = importance_weights(
        pred_loss,
        bankrupt,
        is_loss=is_loss,
        prediction=pred_normal,
        target=target_normal,
    )

    # Hallucinating loss state: network predicted high return (5.0) but actual was low (0.5)
    pred_hallucination = torch.tensor([5.0])
    target_hallucination = torch.tensor([0.5])
    w_loss_hallucination = importance_weights(
        pred_loss,
        bankrupt,
        is_loss=torch.tensor([True]),
        prediction=pred_hallucination,
        target=target_hallucination,
    )

    # Bankrupt state
    w_bankrupt = importance_weights(
        pred_loss,
        torch.tensor([True]),
        is_loss=torch.tensor([True]),
        prediction=pred_normal,
        target=torch.tensor([-0.1]),
    )

    # Hallucinated loss and bankrupt should have significantly higher weight than normal
    assert w_loss_hallucination.item() > w_normal.item() * 5.0
    assert w_bankrupt.item() > w_normal.item() * 10.0


def test_weighted_expectile_loss():
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([2.0, 1.0])
    weights = torch.tensor([1.0, 2.0])
    loss = weighted_expectile_loss(pred, target, weights, tau=0.7)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_empty_live_buffers_are_blocked_without_writing_v2(tmp_path):
    v1 = tmp_path / "iql_weights_v1.pt"
    v2 = tmp_path / "iql_weights_v2.pt"
    torch.save(
        {"state_dict": IQLValueNet().state_dict(), "config": {"vec_dim": VEC_DIM, "hidden": 64}},
        v1,
    )
    result = train_continuous(str(v1), str(v2), [tmp_path / "missing"])
    assert result["status"] == "blocked"
    assert not v2.exists()


def test_extract_tuples_from_kaggle_replay():
    # Build minimal mock Kaggle replay structure
    mock_replay = {
        "info": {"TeamNames": ["Pranav", "OpponentBot"]},
        "rewards": [500.0, 50000.0],  # Pranav lost and is bankrupt (< 3000 cash)
        "steps": [
            [
                {
                    "observation": {
                        "step": 0,
                        "farms": [{"money": 200, "farmhands": 1, "inventory": {}, "shed": {}}],
                        "market": {"crops": {}, "seeds": {}},
                    }
                },
                {"observation": {}},
            ],
            [
                {
                    "observation": {
                        "step": 1,
                        "farms": [{"money": 250, "farmhands": 1, "inventory": {}, "shed": {}}],
                        "market": {"crops": {}, "seeds": {}},
                    }
                },
                {"observation": {}},
            ],
        ],
    }

    records = extract_tuples_from_kaggle_replay(mock_replay, player_query="Pranav", stride=1)
    if records:  # If feature extractors are available
        assert len(records) == 2
        assert records[0]["is_loss"] is True
        assert records[0]["bankruptcy"] is True
        assert records[0]["terminal_cash"] == 500.0
        assert records[0]["spatial"].shape == (SPATIAL_CHANNELS, 10, 10)
        assert records[0]["vec"].shape == (VEC_DIM,)


def test_create_live_replay_dataloader(tmp_path):
    # Save a small batch of synthetic records in .pt format
    pt_path = tmp_path / "replay_sample.pt"
    records = [_record(target=float(i % 10), action=i % 4) for i in range(10)]
    torch.save(records, pt_path)

    train_loader, holdout_loader, dataset = create_live_replay_dataloader(
        losses_dir=str(tmp_path),
        wins_dir=str(tmp_path / "nonexistent"),
        batch_size=4,
        holdout_fraction=0.3,
    )

    assert len(dataset) == 10
    assert len(train_loader.dataset) == 7
    assert len(holdout_loader.dataset) == 3


def test_train_continuous_validation_gate_emits_v2_on_improvement(tmp_path, monkeypatch):
    v1_path = tmp_path / "iql_weights_v1.pt"
    v2_path = tmp_path / "iql_weights_v2.pt"
    replay_path = tmp_path / "replays.pt"

    # Base model
    model = IQLValueNet(vec_dim=VEC_DIM, hidden=32)
    torch.save({"state_dict": model.state_dict(), "config": {"vec_dim": VEC_DIM, "hidden": 32}}, v1_path)

    # Replay records
    records = [_record(target=0.5, action=1) for _ in range(8)]
    torch.save(records, replay_path)

    # Case 1: Candidate improves -> v2 is emitted
    eval_calls = []
    def mock_eval_improves(m, l):
        eval_calls.append(1)
        return 0.10 if len(eval_calls) == 1 else 0.05

    monkeypatch.setattr("scripts.continuous_train_iql.evaluate_mse", mock_eval_improves)
    res = train_continuous(str(v1_path), str(v2_path), [replay_path], epochs=1, batch_size=4)
    assert res["status"] == "improved"
    assert res["saved"] is True
    assert v2_path.is_file()

    # Case 2: Candidate does NOT improve -> v2 is NOT emitted
    v2_path.unlink()
    eval_calls_no = []
    def mock_eval_no_improvement(m, l):
        eval_calls_no.append(1)
        return 0.05 if len(eval_calls_no) == 1 else 0.10

    monkeypatch.setattr("scripts.continuous_train_iql.evaluate_mse", mock_eval_no_improvement)
    res_no = train_continuous(str(v1_path), str(v2_path), [replay_path], epochs=1, batch_size=4)
    assert res_no["status"] == "unchanged"
    assert res_no["saved"] is False
    assert not v2_path.exists()



