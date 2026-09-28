"""Hidden Markov Model & Bayesian Filter for estimating town shop drain rates.

In Kaggriculture, town shops consume wholesale market inventory every 4 hours
(24 simulation steps / 6 ticks per day), naturally recovering prices. The drain
rate varies between 6, 8, 10, or 12 units depending on which specific shops the
hidden daily RNG spawned.

This module implements a Bayesian Filter / Hidden Markov Model that:
1. Ingests the ``current_market_delta`` at the start of every 4-hour tick.
2. Compares price recovery against the deterministic elasticity formulas:
   - Quadratic for Wool:  P_wool = 200 - 0.058 * (ΔI)^2
   - Linear for Milk:     P_milk = 160 - 2.098 * ΔI
3. Calculates the exact integer volume of inventory drained from the market.
4. Updates the Bayesian posterior probabilities over the 4 hidden states:
   Drain ∈ {6, 8, 10, 12}.
5. Exposes ``get_expected_drain_rate()`` returning the MAP drain integer (6, 8, 10, or 12)
   ready to feed directly into the ``TerminalMicroLiquidator`` or DP solver.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

try:
    from kaggriculture.env.items import (
        MARKET_I0,
        MARKET_PARAMS,
        PRICE_FLOOR,
        PRODUCTS_LIST,
        TOWN_SHOP_SELL_INTERVAL,
        TURNS_PER_DAY,
        market_price,
    )
except ImportError:
    MARKET_I0 = 10_000
    PRICE_FLOOR = 1
    PRODUCTS_LIST = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
    TOWN_SHOP_SELL_INTERVAL = 4
    TURNS_PER_DAY = 24
    MARKET_PARAMS = {
        "WOOL": {"base": 200, "I0": 10000, "T": 105, "above_func": "sq", "above_target": 3.20},
        "MILK": {"base": 160, "I0": 10000, "T": 122, "above_func": "linear", "above_target": 1.60},
    }

    def market_price(item: str, inventory: int, params: Any = None) -> int:
        if str(item).upper() == "WOOL":
            delta = max(0, inventory - 10000)
            return max(1, int(round(200 - 0.0580498866 * (delta ** 2))))
        elif str(item).upper() == "MILK":
            delta = max(0, inventory - 10000)
            return max(1, int(round(160 - 2.0983606557 * delta)))
        return 100


# =============================================================================
# Mathematical Constants & Elasticity Formulas
# =============================================================================

HIDDEN_DRAIN_STATES: Tuple[int, ...] = (6, 8, 10, 12)

# Wool: base=200, T=105, above_func='sq', above_target=3.20
# beta = 3.20 * 200 / (105^2) = 640 / 11025 ≈ 0.0580498866...
WOOL_BASE: float = 200.0
WOOL_BETA: float = (3.20 * 200.0) / (105.0 ** 2)  # ≈ 0.0580498866

# Milk: base=160, T=122, above_func='linear', above_target=1.60
# gamma = 1.60 * 160 / 122 = 256 / 122 ≈ 2.0983606557...
MILK_BASE: float = 160.0
MILK_GAMMA: float = (1.60 * 160.0) / 122.0  # ≈ 2.0983606557


def wool_price(excess: float) -> float:
    """Deterministic quadratic wholesale price for Wool given positive excess ΔI."""
    if excess <= 0.0:
        return WOOL_BASE
    raw = WOOL_BASE - WOOL_BETA * (excess ** 2)
    return max(float(PRICE_FLOOR), raw)


def milk_price(excess: float) -> float:
    """Deterministic linear wholesale price for Milk given positive excess ΔI."""
    if excess <= 0.0:
        return MILK_BASE
    raw = MILK_BASE - MILK_GAMMA * excess
    return max(float(PRICE_FLOOR), raw)


def wool_implied_excess(price: float) -> float:
    """Invert quadratic price formula to obtain Wool excess inventory ΔI."""
    if price >= WOOL_BASE:
        return 0.0
    clamped_p = max(float(PRICE_FLOOR), float(price))
    sq_diff = (WOOL_BASE - clamped_p) / WOOL_BETA
    return math.sqrt(max(0.0, sq_diff))


def milk_implied_excess(price: float) -> float:
    """Invert linear price formula to obtain Milk excess inventory ΔI."""
    if price >= MILK_BASE:
        return 0.0
    clamped_p = max(float(PRICE_FLOOR), float(price))
    return max(0.0, (MILK_BASE - clamped_p) / MILK_GAMMA)


def wool_theoretical_recovery(prev_excess: float, drain: float) -> float:
    """Theoretical price recovery for Wool when drain units are consumed."""
    new_excess = max(0.0, prev_excess - drain)
    return wool_price(new_excess) - wool_price(prev_excess)


def milk_theoretical_recovery(prev_excess: float, drain: float) -> float:
    """Theoretical price recovery for Milk when drain units are consumed."""
    new_excess = max(0.0, prev_excess - drain)
    return milk_price(new_excess) - milk_price(prev_excess)


# =============================================================================
# ShopDrainEstimator (Bayesian Filter & HMM)
# =============================================================================

class ShopDrainEstimator:
    """Bayesian Filter and Hidden Markov Model for estimating town shop drain rate.

    Maintains posterior probabilities across hidden states Drain ∈ {6, 8, 10, 12}.
    """

    def __init__(
        self,
        prior: Optional[Mapping[int, float]] = None,
        transition_stability: float = 0.95,
        observation_noise_std: float = 0.85,
    ) -> None:
        """Initializes the Bayesian filter.

        Parameters
        ----------
        prior : dict, optional
            Initial probabilities over {6, 8, 10, 12}. Defaults to uniform {s: 0.25}.
        transition_stability : float
            HMM self-transition probability P(S_t = s | S_{t-1} = s). Default 0.95.
        observation_noise_std : float
            Gaussian likelihood standard deviation for volume rounding / noise. Default 0.85.
        """
        self.states: Tuple[int, ...] = HIDDEN_DRAIN_STATES
        self.transition_stability = float(transition_stability)
        self.obs_noise_std = float(observation_noise_std)

        if prior is not None:
            total = sum(prior.get(s, 0.0) for s in self.states)
            self.posterior: Dict[int, float] = {
                s: float(prior.get(s, 0.0)) / (total if total > 0 else 1.0)
                for s in self.states
            }
        else:
            uniform = 1.0 / len(self.states)
            self.posterior = {s: uniform for s in self.states}

        # History tracking
        self.prev_market_delta: Optional[Dict[str, float]] = None
        self.last_observed_drain: Optional[int] = None
        self.last_price_recovery: Dict[str, float] = {}
        self.history: List[Dict[str, Any]] = []

    def reset(self, prior: Optional[Mapping[int, float]] = None) -> None:
        """Resets the filter state and history."""
        self.__init__(
            prior=prior,
            transition_stability=self.transition_stability,
            observation_noise_std=self.obs_noise_std,
        )

    def _normalize_market_input(
        self,
        market_input: Union[Mapping[str, Any], float, int],
    ) -> Dict[str, float]:
        """Normalizes various market observation formats into {product: excess_delta}."""
        if isinstance(market_input, (int, float)):
            # Single numeric total delta
            return {"TOTAL": float(market_input)}

        if not isinstance(market_input, Mapping):
            return {}

        # Handle full observation dict: obs["market"]
        if "market" in market_input and isinstance(market_input["market"], Mapping):
            m = market_input["market"]
            if "inventory" in m and isinstance(m["inventory"], Mapping):
                # Convert inventory I to excess ΔI = I - I0
                return {
                    str(k).upper(): float(v) - float(MARKET_I0)
                    for k, v in m["inventory"].items()
                }
            if "prices" in m and isinstance(m["prices"], Mapping):
                # Invert prices to implied excess
                res = {}
                for k, v in m["prices"].items():
                    k_up = str(k).upper()
                    if k_up == "WOOL":
                        res["WOOL"] = wool_implied_excess(float(v))
                    elif k_up == "MILK":
                        res["MILK"] = milk_implied_excess(float(v))
                    else:
                        res[k_up] = float(v)
                return res

        # Standard dictionary: e.g. {"wool": 15, "milk": 5}
        normalized: Dict[str, float] = {}
        for k, v in market_input.items():
            if isinstance(v, (int, float)):
                normalized[str(k).upper()] = float(v)

        # If values look like raw wholesale prices (e.g. Wool ~ 180, Milk ~ 140) rather than excess:
        # Detect: if WOOL > 60 or MILK > 40, treat as prices and invert
        if normalized.get("WOOL", 0.0) > 60.0:
            normalized["WOOL_PRICE"] = normalized["WOOL"]
            normalized["WOOL"] = wool_implied_excess(normalized["WOOL"])
        if normalized.get("MILK", 0.0) > 40.0:
            normalized["MILK_PRICE"] = normalized["MILK"]
            normalized["MILK"] = milk_implied_excess(normalized["MILK"])

        return normalized

    def compare_price_recovery(
        self,
        prev_delta: Mapping[str, float],
        curr_delta: Mapping[str, float],
    ) -> Dict[str, float]:
        """Compares price recovery against deterministic elasticity formulas.

        Calculates:
        - Price of Wool before and after tick, and price recovery ΔP_wool.
        - Price of Milk before and after tick, and price recovery ΔP_milk.

        Returns
        -------
        dict
            {"wool_recovery": ΔP, "milk_recovery": ΔP, "wool_price_curr": P, ...}
        """
        prev_wool_excess = float(prev_delta.get("WOOL", 0.0))
        curr_wool_excess = float(curr_delta.get("WOOL", 0.0))
        prev_milk_excess = float(prev_delta.get("MILK", 0.0))
        curr_milk_excess = float(curr_delta.get("MILK", 0.0))

        # Quadratic elasticity for Wool
        p_wool_prev = wool_price(prev_wool_excess)
        p_wool_curr = wool_price(curr_wool_excess)
        wool_rec = p_wool_curr - p_wool_prev

        # Linear elasticity for Milk
        p_milk_prev = milk_price(prev_milk_excess)
        p_milk_curr = milk_price(curr_milk_excess)
        milk_rec = p_milk_curr - p_milk_prev

        return {
            "wool_recovery": wool_rec,
            "milk_recovery": milk_rec,
            "wool_price_prev": p_wool_prev,
            "wool_price_curr": p_wool_curr,
            "milk_price_prev": p_milk_prev,
            "milk_price_curr": p_milk_curr,
            "wool_excess_prev": prev_wool_excess,
            "wool_excess_curr": curr_wool_excess,
            "milk_excess_prev": prev_milk_excess,
            "milk_excess_curr": curr_milk_excess,
        }

    def calculate_drained_volume(
        self,
        prev_delta: Mapping[str, float],
        curr_delta: Mapping[str, float],
        our_sells: Optional[Mapping[str, int]] = None,
    ) -> int:
        """Calculates the exact integer volume of inventory drained during the tick.

        Formula:
        For each commodity p:
            drain_p = max(0, prev_excess_p + our_sells_p - curr_excess_p)
        Total drain = sum(drain_p)
        """
        sells = {str(k).upper(): int(v) for k, v in (our_sells or {}).items()}

        # 1. Total scalar delta case
        if "TOTAL" in prev_delta and "TOTAL" in curr_delta:
            sold_total = sum(sells.values())
            raw_drain = prev_delta["TOTAL"] + sold_total - curr_delta["TOTAL"]
            return max(0, int(round(raw_drain)))

        # 2. Key commodities: Wool and Milk
        wool_drain = 0.0
        if "WOOL" in prev_delta and "WOOL" in curr_delta:
            w_sold = float(sells.get("WOOL", 0))
            wool_drain = max(0.0, prev_delta["WOOL"] + w_sold - curr_delta["WOOL"])

        milk_drain = 0.0
        if "MILK" in prev_delta and "MILK" in curr_delta:
            m_sold = float(sells.get("MILK", 0))
            milk_drain = max(0.0, prev_delta["MILK"] + m_sold - curr_delta["MILK"])

        # 3. Sum over all products present in both observations
        all_keys = set(prev_delta.keys()).intersection(curr_delta.keys())
        total_drain = 0.0

        # Filter out price metadata keys
        prod_keys = [k for k in all_keys if not k.endswith("_PRICE") and k != "TOTAL"]

        if prod_keys:
            for p in prod_keys:
                sold_p = float(sells.get(p, 0))
                d_p = max(0.0, prev_delta[p] + sold_p - curr_delta[p])
                total_drain += d_p
        else:
            total_drain = wool_drain + milk_drain

        # If only wool and milk were tracked, and total drain is partial:
        # Check if wool + milk represent the full drain, or integer round
        return max(0, int(round(total_drain)))

    def update_posterior(self, observed_drain: Union[int, float]) -> Dict[int, float]:
        """Updates Bayesian posterior probabilities given observed drained volume.

        Hidden states: S ∈ {6, 8, 10, 12}.
        """
        v = float(observed_drain)
        prior_state = dict(self.posterior)

        # 1. HMM Transition / Time Update (Prediction Step)
        # P(S_t = s) = (1 - ε)*P(S_{t-1}=s) + (ε / (N-1))*∑_{s'≠s} P(S_{t-1}=s')
        n_states = len(self.states)
        stay_p = self.transition_stability
        switch_p = (1.0 - stay_p) / max(1, n_states - 1)

        predicted_prior: Dict[int, float] = {}
        for s in self.states:
            p_s = stay_p * prior_state[s] + switch_p * sum(
                prior_state[other] for other in self.states if other != s
            )
            predicted_prior[s] = p_s

        # 2. Observation Likelihood (Measurement Step)
        # Discrete / Gaussian kernel: L(v | s) ∝ exp( - (v - s)^2 / (2 * σ^2) )
        sigma_sq = 2.0 * (self.obs_noise_std ** 2)
        unnorm_posterior: Dict[int, float] = {}

        for s in self.states:
            diff = v - float(s)
            likelihood = math.exp(-(diff ** 2) / sigma_sq)
            unnorm_posterior[s] = predicted_prior[s] * likelihood

        total = sum(unnorm_posterior.values())
        if total <= 1e-12:
            # Fallback to closest state if observation is an extreme outlier
            closest_state = min(self.states, key=lambda s: abs(s - v))
            new_posterior = {s: 1.0 if s == closest_state else 0.0 for s in self.states}
        else:
            new_posterior = {s: unnorm_posterior[s] / total for s in self.states}

        self.posterior = new_posterior
        return self.posterior

    def ingest_market_delta(
        self,
        current_market_delta: Union[Mapping[str, Any], float, int],
        our_sells: Optional[Mapping[str, int]] = None,
    ) -> int:
        """Ingests the market delta at the start of a 4-hour tick and updates posterior.

        Parameters
        ----------
        current_market_delta : Mapping or numeric
            Current inventory deviation ΔI = I - I0, wholesale prices, or full market obs.
        our_sells : Mapping, optional
            Units sold by our agent during the preceding 4-hour window.

        Returns
        -------
        int
            The exact integer volume of inventory calculated as drained.
        """
        curr_normalized = self._normalize_market_input(current_market_delta)

        # If this is the very first observation, buffer it and return prior estimate
        if self.prev_market_delta is None:
            self.prev_market_delta = curr_normalized
            # Assume baseline expected drain for turn 0
            best_guess = self.get_expected_drain_rate()
            self.history.append({
                "tick": 0,
                "drained_volume": best_guess,
                "posterior": dict(self.posterior),
                "expected_drain": best_guess,
                "market_delta": curr_normalized,
            })
            return best_guess

        # 1. Compare price recovery against deterministic elasticity formulas
        recovery_info = self.compare_price_recovery(self.prev_market_delta, curr_normalized)
        self.last_price_recovery = recovery_info

        # 2. Calculate exact integer volume of drained inventory
        drained_volume = self.calculate_drained_volume(
            self.prev_market_delta,
            curr_normalized,
            our_sells=our_sells,
        )
        self.last_observed_drain = drained_volume

        # 3. Update Bayesian posterior probabilities
        self.update_posterior(drained_volume)

        # 4. Record history and advance state
        expected_drain = self.get_expected_drain_rate()
        self.history.append({
            "tick": len(self.history),
            "drained_volume": drained_volume,
            "posterior": dict(self.posterior),
            "expected_drain": expected_drain,
            "recovery_info": recovery_info,
            "market_delta": curr_normalized,
        })
        self.prev_market_delta = curr_normalized

        return drained_volume

    # Aliases for convenience
    update = ingest_market_delta
    observe = ingest_market_delta

    def get_expected_drain_rate(self) -> int:
        """Outputs the highest-probability drain integer (6, 8, 10, or 12).

        Returns
        -------
        int
            MAP estimate: argmax_{s ∈ {6, 8, 10, 12}} P(Drain = s).
            Feeds directly into TerminalMicroLiquidator(predicted_shop_drain_rate=...).
        """
        # argmax over posterior
        best_state = max(self.states, key=lambda s: self.posterior.get(s, 0.0))
        return int(best_state)

    def get_expected_drain_value(self) -> float:
        """Returns the expectation E[Drain] = ∑ s * P(Drain = s)."""
        return sum(s * self.posterior.get(s, 0.0) for s in self.states)

    def get_posterior_probabilities(self) -> Dict[int, float]:
        """Returns a copy of the current posterior probabilities."""
        return dict(self.posterior)


# Aliases for compatibility
BayesianDrainFilter = ShopDrainEstimator
DrainRateHMM = ShopDrainEstimator
