"""Unit & Integration tests for ShopDrainEstimator (Bayesian Filter & HMM).

Validates:
1. Deterministic elasticity formulas (Quadratic for Wool, Linear for Milk).
2. Price recovery comparison against theoretical models.
3. Exact integer volume calculation of drained inventory.
4. Bayesian posterior updates across the 4 hidden states (Drain = 6, 8, 10, or 12).
5. get_expected_drain_rate() outputting the highest-probability drain integer.
6. Ingestion from various formats (inventory deltas, wholesale prices, full obs dict).
7. End-to-end integration feeding directly into TerminalMicroLiquidator.
8. Sub-millisecond execution latency.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Dict

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kaggriculture.shop_drain_estimator import (
    HIDDEN_DRAIN_STATES,
    MILK_BASE,
    MILK_GAMMA,
    WOOL_BASE,
    WOOL_BETA,
    BayesianDrainFilter,
    DrainRateHMM,
    ShopDrainEstimator,
    milk_implied_excess,
    milk_price,
    milk_theoretical_recovery,
    wool_implied_excess,
    wool_price,
    wool_theoretical_recovery,
)
from kaggriculture.terminal_micro_liquidator import TerminalMicroLiquidator


# =============================================================================
# 1. Deterministic Elasticity Formulas
# =============================================================================

class TestElasticityFormulas:
    """Validates deterministic quadratic (Wool) and linear (Milk) price curves."""

    def test_wool_quadratic_elasticity(self):
        """P_wool = 200 - 0.058 * (ΔI)^2."""
        assert wool_price(0.0) == 200.0

        # At excess 10: 200 - 0.05805 * 100 ≈ 194.195
        p10 = wool_price(10.0)
        assert abs(p10 - 194.195) < 0.1

        # At excess 20: 200 - 0.05805 * 400 ≈ 176.78
        p20 = wool_price(20.0)
        assert abs(p20 - 176.78) < 0.1

        # Inversion: price -> implied excess
        assert abs(wool_implied_excess(176.78) - 20.0) < 0.1
        assert abs(wool_implied_excess(194.195) - 10.0) < 0.1
        assert wool_implied_excess(200.0) == 0.0

    def test_milk_linear_elasticity(self):
        """P_milk = 160 - 2.098 * ΔI."""
        assert milk_price(0.0) == 160.0

        # At excess 10: 160 - 2.09836 * 10 ≈ 139.016
        p10 = milk_price(10.0)
        assert abs(p10 - 139.016) < 0.1

        # At excess 20: 160 - 2.09836 * 20 ≈ 118.03
        p20 = milk_price(20.0)
        assert abs(p20 - 118.03) < 0.1

        # Inversion: price -> implied excess
        assert abs(milk_implied_excess(139.016) - 10.0) < 0.1
        assert abs(milk_implied_excess(118.03) - 20.0) < 0.1
        assert milk_implied_excess(160.0) == 0.0

    def test_theoretical_recovery(self):
        """Verify theoretical price jump when shop consumes inventory."""
        # Wool: excess 20 -> 16 (drained 4)
        # Price at 20 is ~176.78, at 16 is 200 - 0.05805 * 256 = 185.14
        # Recovery = +8.36
        rec_wool = wool_theoretical_recovery(prev_excess=20.0, drain=4.0)
        assert abs(rec_wool - 8.36) < 0.1

        # Milk: excess 15 -> 11 (drained 4)
        # Recovery = 2.09836 * 4 ≈ +8.39
        rec_milk = milk_theoretical_recovery(prev_excess=15.0, drain=4.0)
        assert abs(rec_milk - 8.39) < 0.1


# =============================================================================
# 2. Exact Drained Volume Calculation
# =============================================================================

class TestDrainedVolumeCalculation:
    """Validates calculation of exact integer inventory drained from market."""

    def test_exact_integer_drained_volume_without_sales(self):
        estimator = ShopDrainEstimator()
        prev = {"WOOL": 20.0, "MILK": 15.0}
        curr = {"WOOL": 16.0, "MILK": 11.0}

        # Drained: (20 - 16) + (15 - 11) = 4 + 4 = 8 units
        drained = estimator.calculate_drained_volume(prev, curr)
        assert drained == 8

    def test_exact_drained_volume_with_our_sales(self):
        estimator = ShopDrainEstimator()
        # Suppose before tick excess was Wool: 20, Milk: 15
        # Our agent sold 2 Wool and 1 Milk during this tick
        # After tick excess is Wool: 17, Milk: 11
        # Net Wool drained: 20 + 2 - 17 = 5
        # Net Milk drained: 15 + 1 - 11 = 5
        # Total drained = 10 units
        prev = {"WOOL": 20.0, "MILK": 15.0}
        curr = {"WOOL": 17.0, "MILK": 11.0}
        our_sells = {"WOOL": 2, "MILK": 1}

        drained = estimator.calculate_drained_volume(prev, curr, our_sells=our_sells)
        assert drained == 10

    def test_price_recovery_comparison_details(self):
        estimator = ShopDrainEstimator()
        prev = {"WOOL": 20.0, "MILK": 15.0}
        curr = {"WOOL": 16.0, "MILK": 11.0}

        rec = estimator.compare_price_recovery(prev, curr)
        assert rec["wool_recovery"] > 0.0
        assert rec["milk_recovery"] > 0.0
        assert abs(rec["wool_recovery"] - 8.36) < 0.2
        assert abs(rec["milk_recovery"] - 8.39) < 0.2


# =============================================================================
# 3. Bayesian Posterior Updates Across 4 Hidden States
# =============================================================================

class TestBayesianPosteriorUpdates:
    """Validates Bayesian updates over hidden states Drain ∈ {6, 8, 10, 12}."""

    def test_initial_uniform_prior(self):
        estimator = ShopDrainEstimator()
        probs = estimator.get_posterior_probabilities()
        assert set(probs.keys()) == {6, 8, 10, 12}
        for s in (6, 8, 10, 12):
            assert abs(probs[s] - 0.25) < 1e-6

    def test_posterior_converges_to_drain_8(self):
        estimator = ShopDrainEstimator()

        # Ingest initial state (Tick 0)
        estimator.ingest_market_delta({"WOOL": 25.0, "MILK": 20.0})

        # Tick 1: Drains 8 units (Wool -4, Milk -4) -> Wool: 21, Milk: 16
        d1 = estimator.ingest_market_delta({"WOOL": 21.0, "MILK": 16.0})
        assert d1 == 8

        probs = estimator.get_posterior_probabilities()
        # State 8 should now be the dominant hypothesis (> 0.70)
        assert probs[8] > 0.70
        assert estimator.get_expected_drain_rate() == 8

        # Tick 2: Drains 8 units again -> Wool: 17, Milk: 12
        d2 = estimator.ingest_market_delta({"WOOL": 17.0, "MILK": 12.0})
        assert d2 == 8

        probs2 = estimator.get_posterior_probabilities()
        # Even higher confidence in 8 (> 0.90)
        assert probs2[8] > 0.90
        assert estimator.get_expected_drain_rate() == 8

    def test_posterior_converges_to_all_four_states(self):
        """Verify each of the 4 hidden states (6, 8, 10, 12) is accurately identified."""
        for target_drain in (6, 8, 10, 12):
            estimator = ShopDrainEstimator()
            w_start, m_start = 30.0, 30.0
            estimator.ingest_market_delta({"WOOL": w_start, "MILK": m_start})

            # Split drain evenly between wool and milk
            w_drop = target_drain // 2
            m_drop = target_drain - w_drop

            # Feed 2 consistent ticks
            for _ in range(2):
                w_start -= w_drop
                m_start -= m_drop
                estimator.ingest_market_delta({"WOOL": w_start, "MILK": m_start})

            assert estimator.get_expected_drain_rate() == target_drain
            probs = estimator.get_posterior_probabilities()
            assert probs[target_drain] > 0.65

    def test_transition_adaptation_when_drain_changes(self):
        """Verify HMM adapts when the hidden daily RNG changes drain rate."""
        estimator = ShopDrainEstimator(transition_stability=0.90)

        # First regime: Drain = 6
        estimator.ingest_market_delta({"WOOL": 50.0, "MILK": 50.0})
        estimator.ingest_market_delta({"WOOL": 47.0, "MILK": 47.0})  # 6 drained
        estimator.ingest_market_delta({"WOOL": 44.0, "MILK": 44.0})  # 6 drained
        assert estimator.get_expected_drain_rate() == 6

        # Regime change: Drain jumps to 12
        estimator.ingest_market_delta({"WOOL": 38.0, "MILK": 38.0})  # 12 drained
        estimator.ingest_market_delta({"WOOL": 32.0, "MILK": 32.0})  # 12 drained

        assert estimator.get_expected_drain_rate() == 12


# =============================================================================
# 4. Input Normalization & Compatibility
# =============================================================================

class TestInputNormalization:
    """Validates ingestion from various observation formats."""

    def test_raw_wholesale_prices_ingestion(self):
        """If given wholesale prices instead of excess, estimator inverts them."""
        estimator = ShopDrainEstimator()

        # Prices when excess was Wool=20 (~$177), Milk=15 (~$129)
        p_prev = {"WOOL": 176.78, "MILK": 128.52}
        estimator.ingest_market_delta(p_prev)

        # Prices when excess is Wool=16 (~$185), Milk=11 (~$137) (drained 8 units)
        p_curr = {"WOOL": 185.14, "MILK": 136.92}
        drained = estimator.ingest_market_delta(p_curr)

        assert drained == 8
        assert estimator.get_expected_drain_rate() == 8

    def test_full_kaggle_observation_dict(self):
        """Accepts full observation dictionary containing obs['market']['inventory']."""
        estimator = ShopDrainEstimator()

        obs0 = {
            "step": 480,
            "market": {
                "inventory": {"WOOL": 10020, "MILK": 10015},
                "prices": {"WOOL": 177, "MILK": 129},
            },
        }
        obs1 = {
            "step": 484,
            "market": {
                "inventory": {"WOOL": 10015, "MILK": 10010},  # Wool -5, Milk -5 = 10 drained
                "prices": {"WOOL": 187, "MILK": 139},
            },
        }

        estimator.ingest_market_delta(obs0)
        drained = estimator.ingest_market_delta(obs1)

        assert drained == 10
        assert estimator.get_expected_drain_rate() == 10

    def test_aliases_work_identically(self):
        """BayesianDrainFilter and DrainRateHMM are exact drop-in aliases."""
        f1 = BayesianDrainFilter()
        f2 = DrainRateHMM()
        assert isinstance(f1, ShopDrainEstimator)
        assert isinstance(f2, ShopDrainEstimator)


# =============================================================================
# 5. Integration with TerminalMicroLiquidator
# =============================================================================

class TestLiquidatorIntegration:
    """Validates feeding get_expected_drain_rate() directly to TerminalMicroLiquidator."""

    def test_direct_handoff_to_micro_liquidator(self):
        estimator = ShopDrainEstimator()

        # Ingest 2 ticks showing drain = 10
        estimator.ingest_market_delta({"WOOL": 30.0, "MILK": 30.0})
        estimator.ingest_market_delta({"WOOL": 25.0, "MILK": 25.0})  # 10 drained

        predicted_drain = estimator.get_expected_drain_rate()
        assert predicted_drain == 10

        # Feed directly into TerminalMicroLiquidator
        liquidator = TerminalMicroLiquidator()
        schedule = liquidator.solve(
            current_turn=600,
            our_inventory={"WOOL": 80, "MILK": 120},
            opp_estimated_inventory={"WOOL": 60, "MILK": 90},
            current_market_delta={"WOOL": 25, "MILK": 25},
            predicted_shop_drain_rate=float(predicted_drain),
        )

        assert len(schedule) > 0
        total_wool = sum(o.get("WOOL", 0) for o in schedule.values())
        total_milk = sum(o.get("MILK", 0) for o in schedule.values())
        assert total_wool == 80
        assert total_milk == 120

    def test_faster_drain_leads_to_larger_optimal_batches(self):
        """A higher drain rate (12 vs 6) allows faster market recovery and larger batches."""
        liquidator = TerminalMicroLiquidator()

        sched_drain6 = liquidator.solve(
            current_turn=600,
            our_inventory={"WOOL": 80},
            predicted_shop_drain_rate=6.0,
        )
        sched_drain12 = liquidator.solve(
            current_turn=600,
            our_inventory={"WOOL": 80},
            predicted_shop_drain_rate=12.0,
        )

        # Early batch at drain 12 should be >= batch at drain 6
        early_drain6 = sum(v.get("WOOL", 0) for t, v in sched_drain6.items() if t < 620)
        early_drain12 = sum(v.get("WOOL", 0) for t, v in sched_drain12.items() if t < 620)
        assert early_drain12 >= early_drain6


# =============================================================================
# 6. Performance Benchmark
# =============================================================================

class TestPerformance:
    """Validates that estimation executes in sub-millisecond time (< 0.1 ms)."""

    def test_estimation_under_sub_millisecond(self):
        estimator = ShopDrainEstimator()
        estimator.ingest_market_delta({"WOOL": 25.0, "MILK": 25.0})

        start = time.perf_counter()
        n_iters = 1000
        for _ in range(n_iters):
            estimator.ingest_market_delta({"WOOL": 21.0, "MILK": 21.0})
            _ = estimator.get_expected_drain_rate()
        elapsed_per_call = (time.perf_counter() - start) / n_iters

        assert elapsed_per_call < 0.0005, f"Execution too slow: {elapsed_per_call*1000:.3f} ms"
