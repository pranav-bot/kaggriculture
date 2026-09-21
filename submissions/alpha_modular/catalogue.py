from __future__ import annotations
from dataclasses import dataclass

from .mechanics import CROPS, expected_unfertilized_yield, harvest_age


@dataclass(frozen=True)
class CropPattern:
    crop: str
    harvest_age_days: int
    expected_yield: int
    seed_cost: int
    watering_actions: int
    tile_days: int

    def projected_margin(self, unit_price: int, labor_shadow_price: float = 0.0) -> float:
        return self.expected_yield * unit_price - self.seed_cost - labor_shadow_price * self.watering_actions


def build_crop_catalogue() -> dict[str, CropPattern]:
    result: dict[str, CropPattern] = {}
    for crop, data in CROPS.items():
        if data["ongoing"]:
            age = int(data["first_yield_day"] + (data["max_yield"] - 1) * data["interval"])
        else:
            age = harvest_age(crop)
        result[crop] = CropPattern(
            crop=crop,
            harvest_age_days=age,
            expected_yield=expected_unfertilized_yield(crop),
            seed_cost=int(data["seed"]),
            watering_actions=age + 1,
            tile_days=age + 1,
        )
    return result


CROP_CATALOGUE = build_crop_catalogue()
