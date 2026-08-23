"""
State & Resource Tracking Helpers for Kaggriculture.

Provides:
- Board State Flattener: Ingests 2D grid into grouped queryable structures.
- Fibonacci Cost Calculator: Calculates cumulative hiring budgets with liquidity safety.
- Yield Trajectory Forecaster: Projects harvestable produce on any future turn/day.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from kaggriculture.actions.actions import (
    Actions,
    CropConfig,
    AnimalConfig,
    CROPS,
    ANIMALS,
    Wheat,
    Carrot,
    Tomato,
    Strawberry,
    Melon,
    Goose,
    Cow,
    Sheep,
)
from kaggriculture.env.items import (
    BOARD_SIZE,
    TURNS_PER_DAY,
    FARM_HAND_COST_MULT,
    Plants,
    Animals,
    Structures,
)


# ==============================================================================
# 1. Board State Flattener
# ==============================================================================

@dataclass
class CropTileInfo:
    x: int
    y: int
    tile: Dict[str, Any]
    crop: str
    config: CropConfig
    age_days: int
    yield_units: int
    watered_today: bool
    consecutive_unwatered: int
    is_thirsty: bool
    is_in_danger: bool
    is_harvestable: bool
    is_ripe: bool
    is_decaying: bool
    fertilized_until_day: int


@dataclass
class AnimalTileInfo:
    x: int
    y: int
    tile: Dict[str, Any]
    structure: str
    animal: Optional[str]
    config: Optional[AnimalConfig]
    placed_day: int
    yield_units: int
    fed_today: bool
    consecutive_unfed: int
    cared_today: bool
    fertilizer_available: bool
    pending_care_bonus: int
    is_hungry: bool
    is_in_danger: bool
    needs_care: bool
    is_harvestable: bool


@dataclass
class FlattenedBoard:
    """Grouped, queryable collection of all tiles on a player's farm."""
    # Crops
    all_crops: List[CropTileInfo] = field(default_factory=list)
    thirsty_crops: List[CropTileInfo] = field(default_factory=list)
    harvestable_crops: List[CropTileInfo] = field(default_factory=list)
    ripe_crops: List[CropTileInfo] = field(default_factory=list)
    danger_crops: List[CropTileInfo] = field(default_factory=list)
    decaying_crops: List[CropTileInfo] = field(default_factory=list)

    # Animals & Structures
    all_structures: List[AnimalTileInfo] = field(default_factory=list)
    occupied_animals: List[AnimalTileInfo] = field(default_factory=list)
    hungry_animals: List[AnimalTileInfo] = field(default_factory=list)
    danger_animals: List[AnimalTileInfo] = field(default_factory=list)
    careable_animals: List[AnimalTileInfo] = field(default_factory=list)
    fertilizer_animals: List[AnimalTileInfo] = field(default_factory=list)
    empty_structures: List[Tuple[int, int, str]] = field(default_factory=list)

    # Map terrain
    vacant_unlocked: List[Tuple[int, int]] = field(default_factory=list)
    weeds: List[Tuple[int, int]] = field(default_factory=list)
    locked_tiles: List[Tuple[int, int]] = field(default_factory=list)

    # Summary Tallies
    crop_counts: Dict[str, int] = field(default_factory=dict)
    animal_counts: Dict[str, int] = field(default_factory=dict)
    total_unlocked_tiles: int = 0

    @property
    def total_vacant(self) -> int:
        return len(self.vacant_unlocked)

    @property
    def total_weeds(self) -> int:
        return len(self.weeds)

    @property
    def total_thirsty(self) -> int:
        return len(self.thirsty_crops)

    @property
    def total_harvestable(self) -> int:
        return len(self.harvestable_crops)


def flatten_board_state(
    tiles: List[List[Any]],
    current_day: int = 0,
    current_step: int = 0,
) -> FlattenedBoard:
    """
    Ingests the nested 2D tiles array and flattens it into categorized, queryable lists.
    """
    board = FlattenedBoard()
    board_size = len(tiles)
    
    # Initialize count dictionaries
    for c in CROPS:
        board.crop_counts[c] = 0
    for a in ANIMALS:
        board.animal_counts[a] = 0

    for y in range(board_size):
        for x in range(board_size):
            tile = tiles[y][x]
            pos = (x, y)

            if tile == "LOCKED":
                board.locked_tiles.append(pos)
                continue

            board.total_unlocked_tiles += 1

            if tile is None:
                board.vacant_unlocked.append(pos)
                continue

            if isinstance(tile, dict):
                kind = tile.get("kind")

                # Case 1: WEED
                if kind == "WEED":
                    board.weeds.append(pos)
                    continue

                # Case 2: PLANT
                if kind == "PLANT":
                    crop_name = tile.get("crop", "WHEAT")
                    crop_cfg = CROPS.get(crop_name, Wheat)
                    planted_day = tile.get("planted_day", 0)
                    age_days = current_day - planted_day
                    yield_units = tile.get("yield_units", 0)
                    watered_today = tile.get("watered_today", False)
                    consec_unwatered = tile.get("consecutive_unwatered", 0)
                    fert_until = tile.get("fertilized_until_day", -1)
                    mls = tile.get("max_lifespan_step", -1)

                    is_thirsty = not watered_today
                    is_danger = consec_unwatered >= 1 and not watered_today
                    is_harv = crop_cfg.is_harvestable(planted_day, current_day, yield_units)
                    is_ripe = crop_cfg.is_optimal_harvest_age(planted_day, current_day) and yield_units > 0
                    is_decay = mls >= 0 and current_step >= mls

                    info = CropTileInfo(
                        x=x,
                        y=y,
                        tile=tile,
                        crop=crop_name,
                        config=crop_cfg,
                        age_days=age_days,
                        yield_units=yield_units,
                        watered_today=watered_today,
                        consecutive_unwatered=consec_unwatered,
                        is_thirsty=is_thirsty,
                        is_in_danger=is_danger,
                        is_harvestable=is_harv,
                        is_ripe=is_ripe,
                        is_decaying=is_decay,
                        fertilized_until_day=fert_until,
                    )

                    board.all_crops.append(info)
                    board.crop_counts[crop_name] = board.crop_counts.get(crop_name, 0) + 1

                    if is_thirsty:
                        board.thirsty_crops.append(info)
                    if is_danger:
                        board.danger_crops.append(info)
                    if is_harv:
                        board.harvestable_crops.append(info)
                    if is_ripe:
                        board.ripe_crops.append(info)
                    if is_decay:
                        board.decaying_crops.append(info)
                    continue

                # Case 3: Structure (COOP / PASTURE)
                if kind in (Structures.COOP, Structures.PASTURE):
                    animal_name = tile.get("animal")
                    anim_cfg = ANIMALS.get(animal_name) if animal_name else None
                    placed_day = tile.get("placed_day", 0)
                    yield_units = tile.get("yield_units", 0)
                    fed_today = tile.get("fed_today", False)
                    consec_unfed = tile.get("consecutive_unfed", 0)
                    cared_today = tile.get("cared_today", False)
                    fert_avail = tile.get("fertilizer_available", False)
                    care_bonus = tile.get("pending_care_bonus", 0)

                    is_hungry = (animal_name is not None) and not fed_today
                    is_danger = (animal_name is not None) and consec_unfed >= 1 and not fed_today
                    needs_care = (animal_name is not None) and not cared_today
                    is_harv = yield_units > 0

                    anim_info = AnimalTileInfo(
                        x=x,
                        y=y,
                        tile=tile,
                        structure=kind,
                        animal=animal_name,
                        config=anim_cfg,
                        placed_day=placed_day,
                        yield_units=yield_units,
                        fed_today=fed_today,
                        consecutive_unfed=consec_unfed,
                        cared_today=cared_today,
                        fertilizer_available=fert_avail,
                        pending_care_bonus=care_bonus,
                        is_hungry=is_hungry,
                        is_in_danger=is_danger,
                        needs_care=needs_care,
                        is_harvestable=is_harv,
                    )

                    board.all_structures.append(anim_info)

                    if animal_name is None:
                        board.empty_structures.append((x, y, kind))
                    else:
                        board.occupied_animals.append(anim_info)
                        board.animal_counts[animal_name] = board.animal_counts.get(animal_name, 0) + 1

                        if is_hungry:
                            board.hungry_animals.append(anim_info)
                        if is_danger:
                            board.danger_animals.append(anim_info)
                        if needs_care:
                            board.careable_animals.append(anim_info)
                        if fert_avail:
                            board.fertilizer_animals.append(anim_info)

    return board


# ==============================================================================
# 2. Fibonacci Cost Calculator
# ==============================================================================

def cumulative_hire_cost(
    num_hires: int,
    already_hired_today: int = 0,
    mult: int = FARM_HAND_COST_MULT,
) -> int:
    """
    Calculates the exact total cost of hiring `num_hires` additional farm hands today,
    starting from `already_hired_today`.
    
    Sequence with mult=1: fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5, fib(5)=8, fib(6)=13, ...
    """
    total = 0
    for i in range(num_hires):
        total += Actions.hire_cost(already_hired_today + i, mult=mult)
    return total


def max_affordable_hires(
    available_money: float,
    already_hired_today: int = 0,
    reserve_liquidity: float = 0.0,
    mult: int = FARM_HAND_COST_MULT,
    max_cap: int = 20,
) -> int:
    """
    Calculates the maximum number of farm hands that can be hired today
    without breaching the required liquidity reserve.
    """
    usable_capital = max(0.0, available_money - reserve_liquidity)
    spent = 0
    count = 0
    
    for i in range(max_cap):
        cost = Actions.hire_cost(already_hired_today + i, mult=mult)
        if spent + cost <= usable_capital:
            spent += cost
            count += 1
        else:
            break
            
    return count


# ==============================================================================
# 3. Yield Trajectory Forecaster
# ==============================================================================

@dataclass
class CropProjection:
    pos: Tuple[int, int]
    crop: str
    current_yield: int
    projected_yield: int
    is_ready_on_target: bool
    is_decayed_on_target: bool
    optimal_harvest_day: int


def forecast_crop_yield_trajectory(
    tiles: List[List[Any]],
    target_day: int,
    target_turn: int = 0,
    assume_daily_watering: bool = True,
    current_day: int = 0,
    current_turn: int = 0,
) -> Dict[str, Any]:
    """
    Projects crop production and harvestable quantities on any future day/turn.
    
    Returns:
      - 'by_crop': total harvestable units per crop type on target_day
      - 'ready_tiles_count': number of tiles ready for harvest
      - 'decayed_tiles_count': number of tiles that will have decayed
      - 'projections': list of CropProjection per active plant tile
    """
    by_crop = {c: 0 for c in CROPS}
    ready_tiles = 0
    decayed_tiles = 0
    projections: List[CropProjection] = []
    board_size = len(tiles)
    target_step = target_day * TURNS_PER_DAY + target_turn

    for y in range(board_size):
        for x in range(board_size):
            tile = tiles[y][x]
            if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
                continue

            crop_name = tile.get("crop", "WHEAT")
            crop_cfg = CROPS.get(crop_name)
            if not crop_cfg:
                continue

            planted_day = tile.get("planted_day", 0)
            fert_until = tile.get("fertilized_until_day", -1)
            curr_units = tile.get("yield_units", 0)

            # Build hypothetical watered days set up to target_day
            watered_days = set()
            if tile.get("watered_today", False):
                watered_days.add(current_day)
            if assume_daily_watering:
                for d in range(current_day + 1, target_day + 1):
                    watered_days.add(d)

            # Calculate accumulated units
            proj_units = crop_cfg.accumulated_yield_units(
                planted_day=planted_day,
                current_day=target_day,
                watered_days=watered_days,
                fertilized_until_day=fert_until,
            )

            # Account for max lifespan decay
            decay_step = crop_cfg.decay_start_step(planted_day, turns_per_day=TURNS_PER_DAY)
            remaining_yield, is_weed = crop_cfg.simulate_decay(
                current_yield=proj_units,
                max_lifespan_step=decay_step,
                current_step=target_step,
            )

            is_ready = crop_cfg.is_harvestable(planted_day, target_day, remaining_yield)
            optimal_day = planted_day + crop_cfg.first_yield_day if crop_cfg.ongoing else planted_day + crop_cfg.time_to_max_yield

            if is_ready and not is_weed:
                by_crop[crop_name] += remaining_yield
                ready_tiles += 1
            if is_weed:
                decayed_tiles += 1

            projections.append(CropProjection(
                pos=(x, y),
                crop=crop_name,
                current_yield=curr_units,
                projected_yield=0 if is_weed else remaining_yield,
                is_ready_on_target=is_ready and not is_weed,
                is_decayed_on_target=is_weed,
                optimal_harvest_day=optimal_day,
            ))

    return {
        "by_crop": by_crop,
        "ready_tiles_count": ready_tiles,
        "decayed_tiles_count": decayed_tiles,
        "projections": projections,
    }
