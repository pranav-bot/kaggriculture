import torch

from kaggriculture.models.market_forecaster import (
    HISTORY_LENGTH,
    MARKET_FEATURE_NAMES,
    MarketForecaster,
    gaussian_nll_loss,
    get_optimal_liquidation_volume,
)
from kaggriculture.env.items import MARKET_I0, PRICE_FLOOR


def test_market_forecaster_is_causal_and_returns_six_gaussians():
    model = MarketForecaster(hidden_size=16)
    states = torch.randn(2, HISTORY_LENGTH, len(MARKET_FEATURE_NAMES))
    mean, log_variance = model(states)
    assert mean.shape == (2, 6)
    assert log_variance.shape == (2, 6)
    assert torch.isfinite(log_variance).all()

    changed = states.clone()
    changed[:, -1] += 100
    changed_mean, _ = model(changed)
    assert not torch.equal(mean, changed_mean)


def test_gaussian_nll_is_finite_and_minimized_at_mean():
    mean = torch.zeros(2, 6)
    log_variance = torch.zeros(2, 6)
    assert gaussian_nll_loss(mean, log_variance, mean).item() == 0
    assert gaussian_nll_loss(mean, log_variance, torch.ones_like(mean)).item() > 0


def test_liquidation_respects_floor_and_nonlinear_curve():
    assert get_optimal_liquidation_volume(MARKET_I0 + 1_000_000, [1, 1, 1]) == 0
    quantity = get_optimal_liquidation_volume(MARKET_I0, [PRICE_FLOOR])
    assert quantity > 0
    assert quantity <= MARKET_I0
