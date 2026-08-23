from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class PlantTile:
    """Represents a crop tile on the farm."""
    kind: str = "PLANT"
    crop: str = "WHEAT"
    planted_day: int = 0
    watered_today: bool = False
    consecutive_unwatered: int = 1
    yield_units: int = 1
    max_lifespan_step: int = -1
    fertilized_until_day: int = -1

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlantTile":
        return cls(
            kind=d.get("kind", "PLANT"),
            crop=d.get("crop", "WHEAT"),
            planted_day=d.get("planted_day", 0),
            watered_today=d.get("watered_today", False),
            consecutive_unwatered=d.get("consecutive_unwatered", 1),
            yield_units=d.get("yield_units", 1),
            max_lifespan_step=d.get("max_lifespan_step", -1),
            fertilized_until_day=d.get("fertilized_until_day", -1),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "crop": self.crop,
            "planted_day": self.planted_day,
            "watered_today": self.watered_today,
            "consecutive_unwatered": self.consecutive_unwatered,
            "yield_units": self.yield_units,
            "max_lifespan_step": self.max_lifespan_step,
            "fertilized_until_day": self.fertilized_until_day,
        }


@dataclass
class WeedTile:
    """Represents a weed tile that must be dug out."""
    kind: str = "WEED"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WeedTile":
        return cls(kind="WEED")

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "WEED"}


@dataclass
class AnimalTile:
    """Represents a livestock structure tile (COOP or PASTURE), optionally occupied."""
    kind: str = "COOP"
    animal: Optional[str] = None
    placed_day: int = 0
    yield_units: int = 0
    fed_today: bool = False
    consecutive_unfed: int = 0
    cared_today: bool = False
    fertilizer_available: bool = False
    pending_care_bonus: int = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnimalTile":
        return cls(
            kind=d.get("kind", "COOP"),
            animal=d.get("animal"),
            placed_day=d.get("placed_day", 0),
            yield_units=d.get("yield_units", 0),
            fed_today=d.get("fed_today", False),
            consecutive_unfed=d.get("consecutive_unfed", 0),
            cared_today=d.get("cared_today", False),
            fertilizer_available=d.get("fertilizer_available", False),
            pending_care_bonus=d.get("pending_care_bonus", 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "animal": self.animal,
            "placed_day": self.placed_day,
            "yield_units": self.yield_units,
            "fed_today": self.fed_today,
            "consecutive_unfed": self.consecutive_unfed,
            "cared_today": self.cared_today,
            "fertilizer_available": self.fertilizer_available,
            "pending_care_bonus": self.pending_care_bonus,
        }


@dataclass
class FarmState:
    """Public farm state visible to both players."""
    money: float
    tiles: List[List[Any]]
    farmer: List[int]
    hands: List[List[int]]
    unlocked_quadrants: List[str]
    hires_today: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FarmState":
        return cls(
            money=float(d.get("money", 0.0)),
            tiles=d.get("tiles", []),
            farmer=list(d.get("farmer", [4, 4])),
            hands=[list(h) for h in d.get("hands", [])],
            unlocked_quadrants=list(d.get("unlocked_quadrants", ["NW"])),
            hires_today=int(d.get("hires_today", 0)),
        )


@dataclass
class MarketState:
    """Public market state showing live prices and shared inventory."""
    inventory: Dict[str, int]
    prices: Dict[str, int]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MarketState":
        return cls(
            inventory=dict(d.get("inventory", {})),
            prices=dict(d.get("prices", {})),
        )


@dataclass
class TownState:
    """Public town state listing all unlocked shop instances."""
    unlocked_shops: List[str]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TownState":
        return cls(
            unlocked_shops=list(d.get("unlocked_shops", [])),
        )


@dataclass
class PrivateState:
    """Private player state hidden from the opponent."""
    shed: Dict[str, int]
    seeds: Dict[str, int]
    inventories: List[Dict[str, int]]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PrivateState":
        return cls(
            shed=dict(d.get("shed", {})),
            seeds=dict(d.get("seeds", {})),
            inventories=[dict(inv) for inv in d.get("inventories", [{}])],
        )


@dataclass
class Observation:
    """
    Complete observation passed to each player at each turn.
    """
    player: int
    day: int
    hour: int
    farms: List[Dict[str, Any]]
    market: Dict[str, Any]
    town: Dict[str, Any]
    private: Dict[str, Any]

    @property
    def my_farm(self) -> Dict[str, Any]:
        return self.farms[self.player] if len(self.farms) > self.player else {}

    @property
    def opp_farm(self) -> Dict[str, Any]:
        opp_id = 1 - self.player
        return self.farms[opp_id] if len(self.farms) > opp_id else {}

    @property
    def my_cash(self) -> float:
        return float(self.my_farm.get("money", 0.0))

    @property
    def opp_cash(self) -> float:
        return float(self.opp_farm.get("money", 0.0))

    @property
    def step_index(self) -> int:
        return self.day * 24 + self.hour

    @classmethod
    def from_dict(cls, obs: Dict[str, Any]) -> "Observation":
        return cls(
            player=int(obs.get("player", 0)),
            day=int(obs.get("day", 0)),
            hour=int(obs.get("hour", 0)),
            farms=obs.get("farms", []),
            market=obs.get("market", {}),
            town=obs.get("town", {}),
            private=obs.get("private", {}),
        )
