"""Typed dataclass models for the Kaggriculture observation schema."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PlantTile:
    """A crop tile on the farm."""
    kind: str = "PLANT"
    crop: str = "WHEAT"
    planted_day: int = 0
    watered_today: bool = False
    consecutive_unwatered: int = 1  # New seed starts at 1 (planting day counts as first missed)
    yield_units: int = 1
    max_lifespan_step: int = -1     # Step at which decay begins; -1 = no decay yet
    fertilized_until_day: int = -1  # Last day fertilizer bonus applies; -1 = none

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlantTile":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class WeedTile:
    kind: str = "WEED"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WeedTile": return cls()
    def to_dict(self) -> Dict[str, Any]: return {"kind": "WEED"}


@dataclass
class AnimalTile:
    """A livestock structure tile (COOP/PASTURE), optionally occupied."""
    kind: str = "COOP"
    animal: Optional[str] = None      # None until PLACEd
    placed_day: int = 0
    yield_units: int = 0
    fed_today: bool = False
    consecutive_unfed: int = 0        # Newly placed animal starts at 0
    cared_today: bool = False
    fertilizer_available: bool = False # Set at end-of-day; cleared by COLLECT_FERTILIZER
    pending_care_bonus: int = 0       # Banked CARE bonus; applied on next yield tick

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnimalTile":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class Observation:
    """Top-level observation passed to each agent at each turn."""
    player: int
    day: int
    hour: int
    step: int  # Global step index (0–719)
    farms: List[Dict[str, Any]]
    market: Dict[str, Any]
    town: Dict[str, Any]
    private: Dict[str, Any]

    @property
    def my_farm(self) -> Dict[str, Any]:
        return self.farms[self.player] if self.player < len(self.farms) else {}

    @property
    def opp_farm(self) -> Dict[str, Any]:
        return self.farms[1 - self.player] if len(self.farms) > 1 else {}

    @property
    def my_cash(self) -> float: return float(self.my_farm.get("money", 0))

    @property
    def opp_cash(self) -> float: return float(self.opp_farm.get("money", 0))

    @property
    def step_index(self) -> int:
        """Return the engine-provided step, including custom day lengths."""
        return self.step

    @classmethod
    def from_dict(cls, obs: Dict[str, Any]) -> "Observation":
        return cls(
            player=int(obs.get("player", 0)),
            day=int(obs.get("day", 0)),
            hour=int(obs.get("hour", 0)),
            step=int(obs.get("step", obs.get("day", 0) * 24 + obs.get("hour", 0))),
            farms=obs.get("farms", []),
            market=obs.get("market", {}),
            town=obs.get("town", {}),
            private=obs.get("private", {}),
        )
