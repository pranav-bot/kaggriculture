"""Tests for the IQL value network (scripts/train_iql_value.py)."""

import os
import sys

import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from scripts.train_iql_value import IQLValueNet, TAU, TARGET_SCALE, VEC_DIM, expectile_loss

ARTIFACT = os.path.join(ROOT, "experiments", "iql_value", "iql_value_net.pt")


def test_expectile_asymmetry_tau_0_7():
    assert TAU == 0.7
    pred = torch.zeros(4)
    target = torch.tensor([1.0, -1.0, 2.0, -2.0])
    loss = expectile_loss(pred, target, tau=0.7)
    # positive diffs weighted 0.7, negative diffs weighted 0.3
    expected = (0.7 * 1 + 0.3 * 1 + 0.7 * 4 + 0.3 * 4) / 4
    assert loss.item() == pytest.approx(expected)


def test_expectile_reduces_to_mse_at_tau_0_5():
    torch.manual_seed(0)
    pred, target = torch.randn(32), torch.randn(32)
    assert expectile_loss(pred, target, tau=0.5).item() == pytest.approx(
        0.5 * ((target - pred) ** 2).mean().item()
    )


def test_value_net_state_only_no_action_input():
    """V_psi(s) takes state tensors only -- no action/policy query possible."""
    import inspect

    sig = inspect.signature(IQLValueNet.forward)
    assert list(sig.parameters.keys()) == ["self", "spatial", "vec"]
    model = IQLValueNet()
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(3, 23, 10, 10), torch.zeros(3, VEC_DIM))
    assert out.shape == (3,)


def test_vec_dim_covers_cash_market_prices():
    assert VEC_DIM == 30 + 35  # global (cash, ...) + market (prices, inventory)


def test_artifact_exists_small_and_loads():
    assert os.path.exists(ARTIFACT), "run scripts/train_iql_value.py first"
    size_mb = os.path.getsize(ARTIFACT) / 1e6
    assert size_mb < 20.0, f"{size_mb:.2f} MB exceeds Kaggle tarball budget"
    bundle = torch.load(ARTIFACT, map_location="cpu", weights_only=False)
    assert bundle["config"]["tau"] == pytest.approx(0.7)
    assert bundle["config"]["target_scale"] == TARGET_SCALE
    model = IQLValueNet()
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    with torch.no_grad():
        v = model(torch.zeros(2, 23, 10, 10), torch.zeros(2, VEC_DIM))
    assert torch.isfinite(v).all()
