"""
Example 06 — Helper-driven harvest timing.

Combines library helpers (flatten_board_state, check_max_yield_met) with a
thin ActionController wrapper. Prioritizes ripe tiles using explicit yield math
instead of relying solely on controller heuristics.
"""
import os
import sys
from typing import Any, Dict, Optional, Set, Tuple

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import ActionController, Actions, Plants, check_max_yield_met, flatten_board_state


class HelperHarvestController(ActionController):
    def __init__(self) -> None:
        super().__init__(
            target_crop=Plants.MELON,
            auto_water=True,
            auto_harvest=False,  # harvest routing handled below
            auto_fertilize=True,
            auto_dig_weeds=True,
            auto_sell=True,
            auto_expand_land=True,
            auto_hire_hands=True,
            max_hires_per_day=6,
            min_sell_margin=0.4,
        )

    def _ripe_priority_tile(
        self,
        farm: dict,
        unit_pos: Tuple[int, int],
        current_day: int,
    ) -> Optional[Tuple[int, int]]:
        board = flatten_board_state(farm["tiles"], current_day=current_day)
        best: Optional[Tuple[int, int]] = None
        best_score = -1

        for crop in board.ripe_crops:
            tile = crop.tile
            cfg = crop.config
            if cfg.ongoing:
                ready = crop.is_harvestable
            else:
                # Approximate bonus counts from tile state for yield_check helper
                planted_day = tile.get("planted_day", current_day)
                age = current_day - planted_day
                bonus_start = (cfg.time_to_max_yield + 1) // 2
                watered_days = max(0, min(age, cfg.time_to_max_yield) - bonus_start + 1)
                fert_until = tile.get("fertilized_until_day", -1)
                fert_days = sum(
                    1
                    for d in range(planted_day + bonus_start, planted_day + age + 1)
                    if fert_until >= d
                )
                ready = check_max_yield_met({
                    "plant_type": crop.crop,
                    "age_in_days": age,
                    "watered_bonus_days_count": max(0, watered_days - fert_days),
                    "fertilized_bonus_days_count": fert_days,
                })

            if not ready or crop.yield_units <= 0:
                continue

            dist = self.manhattan_distance(unit_pos, (crop.x, crop.y))
            score = crop.yield_units * 100 - dist
            if score > best_score:
                best_score = score
                best = (crop.x, crop.y)

        return best

    def plan_unit_action(
        self,
        unit_idx: int,
        farm: dict,
        private: dict,
        available_seeds: dict,
        obs: dict,
        claimed_tiles: Set[Tuple[int, int]],
    ) -> list:
        current_day = int(obs.get("day", 0))
        units = [farm["farmer"]] + farm.get("hands", [])
        unit = units[unit_idx]
        pos = (unit["x"], unit["y"])

        ripe = self._ripe_priority_tile(farm, pos, current_day)
        if ripe and ripe not in claimed_tiles:
            tx, ty = ripe
            if pos == (tx, ty):
                claimed_tiles.add(ripe)
                return Actions.harvest()
            return self.move_to(pos, ripe)

        return super().plan_unit_action(
            unit_idx, farm, private, available_seeds, obs, claimed_tiles
        )


controller = HelperHarvestController()


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return controller.act(obs)
