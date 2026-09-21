"""Lockstep premium sell quantity duel (research/05 innovation 1)."""

from __future__ import annotations

from typing import Sequence

from kaggriculture.helpers.market_prediction import simulate_sell_slippage
from kaggriculture.helpers.phase_brain import PREMIUM_ITEMS

PRESSURE_SLOPE = 15
PRESSURE_SOFT_START = 90
SHED_FORCE_SELL = 88

_CANDIDATE_Q: Sequence[int] = (0, 2, 4, 8)


def duel_sell_qty(
    item: str,
    have: int,
    market_inv: int,
    opp_hat: int,
    shed_total: int,
) -> int:
    """
    Choose sell quantity maximizing worst-case lockstep revenue vs opponent
    sell volumes {0, opp_hat, 2*opp_hat}. Returns 0 to hold when duel favors wait
    unless shed_total >= SHED_FORCE_SELL.
    """
    have = max(0, int(have))
    if have <= 0:
        return 0

    opp_hat = max(0, int(opp_hat))
    opp_scenarios = (0, opp_hat, 2 * opp_hat)
    candidates = sorted({q for q in _CANDIDATE_Q if q <= have} | {have})

    def pressure_penalty(q: int) -> float:
        return PRESSURE_SLOPE * max(0, shed_total + q - PRESSURE_SOFT_START)

    best_q = 0
    best_score = float("-inf")

    for q in candidates:
        if q == 0:
            worst_rev = 0.0
        else:
            worst_rev = float(
                min(
                    simulate_sell_slippage(
                        item,
                        q,
                        market_inv,
                        opponent_simultaneous_sell=oq,
                    ).total_revenue
                    for oq in opp_scenarios
                )
            )
        score = worst_rev - pressure_penalty(q)
        if score > best_score:
            best_q, best_score = q, score

    if best_q == 0 and shed_total >= SHED_FORCE_SELL:
        return have
    return best_q


def is_premium_product(item: str) -> bool:
    return str(item).upper() in PREMIUM_ITEMS
