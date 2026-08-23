"""
Opponent Modeling & Adversarial Analytics for Kaggriculture.

Provides:
- Adversarial Asset Tracker: Dedicated parser for the opponent's public farm state.
  Identifies their highest-volume crops, projects impending harvest dates, and
  flags strategic market sabotage & front-running windows.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from kaggriculture.actions.actions import Actions, CROPS, ANIMALS
from kaggriculture.env.items import (
    MARKET_I0,
    TURNS_PER_DAY,
    Plants,
    Animals,
    Products,
)


@dataclass
class OpponentCropGroup:
    crop: str
    planted_day: int
    count: int
    optimal_harvest_day: int
    estimated_total_yield: int
    estimated_market_value: float
    days_until_optimal_harvest: int


@dataclass
class SabotageOpportunity:
    opportunity_type: str              # 'FRONT_RUN_HARVEST', 'GLUT_CRASH', 'FEED_STARVATION'
    target_product: str
    target_turn_window: Tuple[int, int] # (start_step, end_step)
    target_day: int
    estimated_opponent_volume: int
    recommended_action: str
    expected_impact: str


@dataclass
class OpponentProfile:
    opponent_money: float
    unlocked_quadrants: List[str]
    total_crops_planted: int
    crop_counts: Dict[str, int]
    animal_counts: Dict[str, int]
    weeds_count: int
    vacant_count: int
    crop_groups: List[OpponentCropGroup]
    sabotage_opportunities: List[SabotageOpportunity]


def analyze_opponent_farm(
    opponent_farm: Dict[str, Any],
    current_day: int = 0,
    current_hour: int = 0,
    market_prices: Optional[Dict[str, int]] = None,
) -> OpponentProfile:
    """
    Parses the opponent's public farm dictionary to model their strategy,
    track crop maturities, and flag high-impact market sabotage windows.
    """
    tiles = opponent_farm.get("tiles", [])
    money = float(opponent_farm.get("money", 0.0))
    unlocked = opponent_farm.get("unlocked_quadrants", ["NW"])
    board_size = len(tiles)
    current_step = current_day * TURNS_PER_DAY + current_hour

    crop_counts = {c: 0 for c in CROPS}
    animal_counts = {a: 0 for a in ANIMALS}
    weeds_count = 0
    vacant_count = 0
    grouped_crops: Dict[Tuple[str, int], int] = {}  # (crop, planted_day) -> count

    for y in range(board_size):
        for x in range(board_size):
            tile = tiles[y][x]
            if tile == "LOCKED":
                continue
            if tile is None:
                vacant_count += 1
                continue
            if isinstance(tile, dict):
                kind = tile.get("kind")
                if kind == "WEED":
                    weeds_count += 1
                elif kind == "PLANT":
                    crop_name = tile.get("crop", "WHEAT")
                    planted_day = tile.get("planted_day", 0)
                    crop_counts[crop_name] = crop_counts.get(crop_name, 0) + 1
                    key = (crop_name, planted_day)
                    grouped_crops[key] = grouped_crops.get(key, 0) + 1
                elif "animal" in tile and tile["animal"]:
                    animal_name = tile["animal"]
                    animal_counts[animal_name] = animal_counts.get(animal_name, 0) + 1

    # Project harvest groups
    crop_groups: List[OpponentCropGroup] = []
    prices = market_prices or {c: CROPS[c].base_market_price for c in CROPS}

    for (crop_name, planted_day), count in grouped_crops.items():
        cfg = CROPS.get(crop_name)
        if not cfg:
            continue

        opt_day = planted_day + (cfg.first_yield_day if cfg.ongoing else cfg.time_to_max_yield)
        days_left = opt_day - current_day
        est_yield_per_tile = cfg.unfertilized_max_yield if not cfg.ongoing else cfg.max_yield
        total_est_yield = count * est_yield_per_tile
        market_val = total_est_yield * prices.get(crop_name, cfg.base_market_price)

        crop_groups.append(OpponentCropGroup(
            crop=crop_name,
            planted_day=planted_day,
            count=count,
            optimal_harvest_day=opt_day,
            estimated_total_yield=total_est_yield,
            estimated_market_value=market_val,
            days_until_optimal_harvest=days_left,
        ))

    # Sort groups by impending harvest day asc, then estimated value desc
    crop_groups.sort(key=lambda g: (g.days_until_optimal_harvest, -g.estimated_market_value))

    # Identify Sabotage & Front-Running Opportunities
    sabotage_opps: List[SabotageOpportunity] = []

    for grp in crop_groups:
        # 1. Front-Running / Glut Crash for Premium Goods (Melon, Strawberry, Tomato)
        if grp.estimated_total_yield >= 10 and grp.days_until_optimal_harvest in (0, 1):
            sab_step = max(0, grp.optimal_harvest_day * TURNS_PER_DAY - 2)
            sabotage_opps.append(SabotageOpportunity(
                opportunity_type="FRONT_RUN_HARVEST",
                target_product=grp.crop,
                target_turn_window=(sab_step - 2, sab_step + 4),
                target_day=grp.optimal_harvest_day,
                estimated_opponent_volume=grp.estimated_total_yield,
                recommended_action=f"SELL {grp.crop} right before Turn {grp.optimal_harvest_day * 24}",
                expected_impact=f"Crashes market price of {grp.crop} before opponent can dump {grp.estimated_total_yield} units.",
            ))

    # 2. Livestock Wheat Starvation Sabotage
    total_opponent_livestock = sum(animal_counts.values())
    if total_opponent_livestock >= 2:
        sabotage_opps.append(SabotageOpportunity(
            opportunity_type="FEED_STARVATION",
            target_product="WHEAT",
            target_turn_window=(current_step, current_step + 12),
            target_day=current_day,
            estimated_opponent_volume=total_opponent_livestock,
            recommended_action="BUY_PRODUCT WHEAT from market to induce wheat scarcity",
            expected_impact=f"Forces opponent to pay scarcity prices or risk {total_opponent_livestock} animals escaping unfed.",
        ))

    return OpponentProfile(
        opponent_money=money,
        unlocked_quadrants=unlocked,
        total_crops_planted=sum(crop_counts.values()),
        crop_counts=crop_counts,
        animal_counts=animal_counts,
        weeds_count=weeds_count,
        vacant_count=vacant_count,
        crop_groups=crop_groups,
        sabotage_opportunities=sabotage_opps,
    )
