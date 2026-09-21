from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

from mechanics import BOARD_SIZE, TURNS_PER_DAY, read


@dataclass(frozen=True)
class Unit:
    index: int
    position: tuple[int, int]
    inventory: Mapping[str, int]


@dataclass(frozen=True)
class GameState:
    observation: Any
    player: int
    step: int
    day: int
    hour: int
    board_size: int
    farm: Any
    opponent_farm: Any
    private: Any
    market: Any
    town: Any

    @classmethod
    def from_observation(cls, obs: Any) -> "GameState":
        player = int(read(obs, "player", 0))
        farms = read(obs, "farms", []) or []
        farm = farms[player] if player < len(farms) else {}
        opponent = farms[1 - player] if len(farms) == 2 else {}
        tiles = read(farm, "tiles", []) or []
        board_size = len(tiles) or BOARD_SIZE
        step = int(read(obs, "step", int(read(obs, "day", 0)) * TURNS_PER_DAY + int(read(obs, "hour", 0))))
        return cls(
            observation=obs,
            player=player,
            step=step,
            day=int(read(obs, "day", step // TURNS_PER_DAY)),
            hour=int(read(obs, "hour", step % TURNS_PER_DAY)),
            board_size=board_size,
            farm=farm,
            opponent_farm=opponent,
            private=read(obs, "private", {}) or {},
            market=read(obs, "market", {}) or {},
            town=read(obs, "town", {}) or {},
        )

    @property
    def money(self) -> float:
        return float(read(self.farm, "money", 0.0))

    @property
    def shed(self) -> Mapping[str, int]:
        return read(self.private, "shed", {}) or {}

    @property
    def seeds(self) -> Mapping[str, int]:
        return read(self.private, "seeds", {}) or {}

    @property
    def unlocked_count(self) -> int:
        return len(read(self.farm, "unlocked_quadrants", ["NW"]) or ["NW"])

    def units(self) -> list[Unit]:
        positions = [read(self.farm, "farmer", [0, 0]), *(read(self.farm, "hands", []) or [])]
        inventories = read(self.private, "inventories", []) or []
        result = []
        for index, position in enumerate(positions):
            inventory = inventories[index] if index < len(inventories) else {}
            result.append(Unit(index=index, position=(int(position[0]), int(position[1])), inventory=inventory or {}))
        return result

    def iter_tiles(self) -> Iterator[tuple[int, int, Any]]:
        for y, row in enumerate(read(self.farm, "tiles", []) or []):
            for x, tile in enumerate(row):
                yield x, y, tile

    def crop_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, _, tile in self.iter_tiles():
            if isinstance(tile, Mapping) and read(tile, "kind") == "PLANT":
                crop = str(read(tile, "crop"))
                counts[crop] = counts.get(crop, 0) + 1
        return counts

    def empty_positions(self) -> list[tuple[int, int]]:
        return [(x, y) for x, y, tile in self.iter_tiles() if tile is None]

    def market_price(self, product: str) -> int:
        return int((read(self.market, "prices", {}) or {}).get(product, 1))

    def market_inventory(self, product: str) -> int:
        return int((read(self.market, "inventory", {}) or {}).get(product, 10_000))
