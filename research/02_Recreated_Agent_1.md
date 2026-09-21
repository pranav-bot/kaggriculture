# 02 — Recreated Agent 1 (Architecture Alpha)

**Family:** Online modular heuristic phase machine  
**Scope:** Exact 1:1 recreation of the recoverable modular champion logic, fully anonymized.  
**Usage:** Drop modules into a package named `policy_stack/` (or paste into a single notebook). Wire `agent(observation)` as the Kaggle entrypoint.

---

## Design Contract

```
decide(obs, config):
  state ← GameState.from_observation(obs)
  fallback ← cash_preserving_sell_all(state)
  try:
    if elapsed > 0.20s: return fallback
    tasks ← materialize_tasks(state, config)
    unit_actions, planned_drop ← assign_actions(state, tasks, config)
    market ← make_market_orders(state, config, planned_drop)
    if elapsed > 0.40s: return fallback
    return validate_joint_action(state, unit_actions, market)
  except:
    return fallback
```

---

## Complete Python Template

### `mechanics.py`

```python
"""Pinned environment mechanics used by planning (no env import required)."""
from __future__ import annotations

import math
from typing import Any, Mapping

EPISODE_STEPS = 720
TURNS_PER_DAY = 24
BOARD_SIZE = 10
SHED_CAPACITY = 100
MAX_MARKET_ORDERS = 10
PRICE_FLOOR = 1
MARKET_I0 = 10_000

CROPS: dict[str, dict[str, Any]] = {
    "WHEAT": {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT": {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO": {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON": {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

ANIMALS: dict[str, dict[str, Any]] = {
    "GOOSE": {"cost": 300, "structure": "COOP", "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW": {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")

MARKET_PARAMS: dict[str, dict[str, Any]] = {
    "WHEAT": {"base": 25, "I0": MARKET_I0, "T": 400, "below_func": "sqrt", "below_target": 0.80, "above_func": "log", "above_target": 0.20},
    "CARROT": {"base": 35, "I0": MARKET_I0, "T": 450, "below_func": "hinge", "below_target": 1.00, "above_func": "sqrt", "above_target": 0.70},
    "TOMATO": {"base": 60, "I0": MARKET_I0, "T": 200, "below_func": "hinge", "below_target": 0.40, "above_func": "sqrt", "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": MARKET_I0, "T": 100, "below_func": "sqrt", "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON": {"base": 250, "I0": MARKET_I0, "T": 300, "below_func": "log", "below_target": 0.20, "above_func": "sq", "above_target": 3.60},
    "EGG": {"base": 50, "I0": MARKET_I0, "T": 332, "below_func": "hinge", "below_target": 0.40, "above_func": "log", "above_target": 0.20},
    "MILK": {"base": 160, "I0": MARKET_I0, "T": 122, "below_func": "sqrt", "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL": {"base": 200, "I0": MARKET_I0, "T": 105, "below_func": "log", "below_target": 0.20, "above_func": "sq", "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": MARKET_I0, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}

SHOPS: dict[str, tuple[str, ...]] = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}

LAND_PRICES = (1_000, 2_000, 4_000)


def read(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def shape(name: str, x: float, anchor: float | None = None) -> float:
    x = max(0.0, x)
    if name == "linear":
        return x
    if name == "sq":
        return x * x
    if name == "sqrt":
        return math.sqrt(x)
    if name == "log":
        return math.log1p(x)
    if name == "hinge":
        if not anchor or anchor <= 0:
            return x
        u = x / anchor
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


def market_price(item: str, inventory: int, params: Mapping[str, Mapping[str, Any]] | None = None) -> int:
    p = (params or MARKET_PARAMS)[item]
    base, initial, anchor = p["base"], p["I0"], p["T"]
    if inventory < initial:
        func, target, sign = p["below_func"], p["below_target"], 1
    else:
        func, target, sign = p["above_func"], p["above_target"], -1
    amplitude = target * base / shape(func, anchor, anchor)
    price = base + sign * amplitude * shape(func, abs(inventory - initial), anchor)
    return max(PRICE_FLOOR, int(round(price)))


def shed_access_tiles(board_size: int = BOARD_SIZE) -> tuple[tuple[int, int], ...]:
    half = board_size // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def is_shed_adjacent(position, board_size: int = BOARD_SIZE) -> bool:
    return tuple(position) in shed_access_tiles(board_size)


def hire_cost(number_already_hired: int) -> int:
    a, b = 1, 1
    for _ in range(number_already_hired):
        a, b = b, a + b
    return a


def harvest_age(crop: str) -> int:
    if crop == "MELON":
        return 10
    return int(CROPS[crop]["max_yield_day"])


def expected_unfertilized_yield(crop: str) -> int:
    if crop == "WHEAT":
        return 4
    if crop == "CARROT":
        return 3
    if crop == "MELON":
        return 6
    return int(CROPS[crop]["max_yield"])


def last_profitable_start_day(crop: str, season_days: int = 30) -> int:
    data = CROPS[crop]
    grow = int(data["first_yield_day"]) if data["ongoing"] else harvest_age(crop)
    return season_days - 1 - grow
```

### `config.py`

```python
from __future__ import annotations
from copy import deepcopy
from typing import Any

BASELINE_CONFIG: dict[str, Any] = {
    "name": "baseline-v1",
    "max_active": 18,
    "early_mix": {"CARROT": 14, "WHEAT": 4},
    "late_mix": {"CARROT": 14, "WHEAT": 4},
    "mix_switch_day": 30,
    "crop_order": ["CARROT", "WHEAT"],
    "target_unlocked": 1,
    "hands_by_unlocked": {1: 5},
    "sell_mode": "immediate",
    "premium_batch": 100,
    "terminal_day": 29,
    "terminal_return_hour": 15,
    "operating_reserve": 100,
    "adaptive_mix": False,
}

CHAMPION_CONFIG: dict[str, Any] = {
    "name": "champion-v1",
    "max_active": 40,
    "early_mix": {"MELON": 14, "CARROT": 14, "WHEAT": 12},
    "late_mix": {"CARROT": 20, "WHEAT": 20},
    "mix_switch_day": 13,
    "crop_order": ["MELON", "CARROT", "WHEAT"],
    "target_unlocked": 2,
    "hands_by_unlocked": {1: 6, 2: 9},
    "sell_mode": "batched",
    "premium_batch": 8,
    "terminal_day": 29,
    "terminal_return_hour": 13,
    "operating_reserve": 100,
    "adaptive_mix": True,
}


def cloned(config: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(config)
```

### `state.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

from .mechanics import BOARD_SIZE, EPISODE_STEPS, TURNS_PER_DAY, read


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
```

### `catalogue.py`

```python
from __future__ import annotations
from dataclasses import asdict, dataclass
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
```

### `production.py`

```python
from __future__ import annotations
from typing import Any
from .catalogue import CROP_CATALOGUE
from .mechanics import SHOPS, last_profitable_start_day, read
from .state import GameState


def desired_mix(state: GameState, config: dict[str, Any]) -> dict[str, int]:
    source = config["early_mix"] if state.day < int(config["mix_switch_day"]) else config["late_mix"]
    mix = {str(crop): int(count) for crop, count in source.items()}
    if not config.get("adaptive_mix") or state.day < int(config["mix_switch_day"]):
        return mix

    shops = read(state.town, "unlocked_shops", []) or []
    carrot_pull = sum(2 if shop == "PET_CAFE" else 1 for shop in shops if "CARROT" in SHOPS.get(shop, ()))
    wheat_pull = sum(1 for shop in shops if "WHEAT" in SHOPS.get(shop, ()))
    shift = min(3, abs(carrot_pull - wheat_pull))
    if carrot_pull > wheat_pull and mix.get("WHEAT", 0) >= shift:
        mix["CARROT"] = mix.get("CARROT", 0) + shift
        mix["WHEAT"] -= shift
    elif wheat_pull > carrot_pull and mix.get("CARROT", 0) >= shift:
        mix["WHEAT"] = mix.get("WHEAT", 0) + shift
        mix["CARROT"] -= shift
    return mix


def seed_deficits(state: GameState, config: dict[str, Any]) -> dict[str, int]:
    active = state.crop_counts()
    targets = desired_mix(state, config)
    deficits: dict[str, int] = {}
    for crop, target in targets.items():
        if state.day > last_profitable_start_day(crop):
            continue
        have = active.get(crop, 0) + int(state.seeds.get(crop, 0))
        if target > have:
            deficits[crop] = target - have
    return deficits


def plant_assignments(state: GameState, config: dict[str, Any]) -> list[tuple[tuple[int, int], str]]:
    active = state.crop_counts()
    targets = desired_mix(state, config)
    room = max(0, int(config["max_active"]) - sum(active.values()))
    if room <= 0:
        return []

    available = {crop: int(state.seeds.get(crop, 0)) for crop in targets}
    deficits = {
        crop: max(0, target - active.get(crop, 0))
        for crop, target in targets.items()
        if state.day <= last_profitable_start_day(crop)
    }
    order = [crop for crop in config["crop_order"] if crop in deficits]
    center = state.board_size // 2 - 1
    empties = sorted(
        state.empty_positions(),
        key=lambda p: (abs(p[0] - center) + abs(p[1] - center), p[1], p[0]),
    )
    assignments: list[tuple[tuple[int, int], str]] = []
    for position in empties:
        choices = [crop for crop in order if deficits.get(crop, 0) > 0 and available.get(crop, 0) > 0]
        if not choices or len(assignments) >= room:
            break
        crop = min(choices, key=lambda c: (-deficits[c] / max(1, targets[c]), order.index(c)))
        assignments.append((position, crop))
        deficits[crop] -= 1
        available[crop] -= 1
    return assignments
```

### `tasks.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping

from .mechanics import ANIMALS, CROPS, expected_unfertilized_yield, harvest_age, read
from .production import plant_assignments
from .state import GameState


@dataclass(frozen=True)
class Task:
    task_id: str
    target: tuple[int, int]
    action: tuple[Any, ...]
    priority: int
    deadline_step: int
    loss_if_omitted: float
    reason: str


def _plant_task(state: GameState, x: int, y: int, tile: Mapping[str, Any]) -> Task | None:
    crop = str(read(tile, "crop"))
    data = CROPS[crop]
    age = state.day - int(read(tile, "planted_day", state.day))
    watered = bool(read(tile, "watered_today", False))
    missed = int(read(tile, "consecutive_unwatered", 0))
    held = int(read(tile, "yield_units", 0))
    terminal = state.day >= 29

    if data["ongoing"]:
        if held > 0 and (terminal or held >= int(data["max_yield"])):
            return Task(f"harvest:{x}:{y}", (x, y), ("HARVEST",), 1 if terminal else 3, state.step + 1, held * 50.0, "collect ongoing yield")
        if not watered and not terminal:
            priority = 0 if missed >= 1 else 3
            return Task(f"water:{x}:{y}", (x, y), ("WATER",), priority, state.step + (1 if missed >= 1 else 20), 200.0, "prevent crop death")
        return None

    target_age = harvest_age(crop)
    target_yield = expected_unfertilized_yield(crop)
    if terminal and held > 0:
        return Task(f"harvest:{x}:{y}", (x, y), ("HARVEST",), 0, state.step + 1, held * state.market_price(crop), "terminal liquidation harvest")
    if age >= target_age:
        can_add_yield = age <= int(data["max_yield_day"]) and held < target_yield
        if not watered and can_add_yield:
            return Task(f"water-peak:{x}:{y}", (x, y), ("WATER",), 1, state.step + 8, state.market_price(crop), "realize peak yield")
        return Task(f"harvest:{x}:{y}", (x, y), ("HARVEST",), 2, state.step + 12, held * state.market_price(crop), "mature harvest")
    if not watered:
        priority = 0 if missed >= 1 else 4
        return Task(
            f"water:{x}:{y}",
            (x, y),
            ("WATER",),
            priority,
            state.step + (1 if missed >= 1 else 20),
            float(CROPS[crop]["seed"] + target_yield * state.market_price(crop)),
            "prevent crop death" if missed >= 1 else "daily yield service",
        )
    return None


def materialize_tasks(state: GameState, config: dict[str, Any]) -> list[Task]:
    tasks: list[Task] = []
    for x, y, tile in state.iter_tiles():
        if tile is None or tile == "LOCKED":
            continue
        if isinstance(tile, Mapping) and read(tile, "kind") == "PLANT":
            task = _plant_task(state, x, y, tile)
            if task:
                tasks.append(task)
        elif isinstance(tile, Mapping) and "animal" in tile:
            animal = str(read(tile, "animal"))
            if not bool(read(tile, "fed_today", False)):
                tasks.append(Task(f"feed:{x}:{y}", (x, y), ("FEED",), 0, state.step + 1, float(ANIMALS[animal]["cost"]), "prevent animal escape"))
            elif int(read(tile, "yield_units", 0)) > 0:
                tasks.append(Task(f"animal-harvest:{x}:{y}", (x, y), ("HARVEST",), 3, state.step + 20, 100.0, "collect animal product"))
        elif isinstance(tile, Mapping) and read(tile, "kind") == "WEED" and state.day < 27:
            tasks.append(Task(f"dig:{x}:{y}", (x, y), ("DIG",), 8, state.step + 48, 10.0, "recover productive tile"))

    for (x, y), crop in plant_assignments(state, config):
        tasks.append(Task(f"plant:{crop}:{x}:{y}", (x, y), ("PLANT", crop), 7, state.step + 20, 50.0, "selected production pattern"))
    return sorted(tasks, key=lambda task: (task.priority, task.deadline_step, task.target[1], task.target[0], task.task_id))
```

### `routing.py`

```python
from __future__ import annotations
from typing import Any
from .mechanics import is_shed_adjacent, shed_access_tiles
from .state import GameState, Unit
from .tasks import Task


def distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _step_toward(source: tuple[int, int], target: tuple[int, int], unit_index: int) -> list[str]:
    dx, dy = target[0] - source[0], target[1] - source[1]
    horizontal_first = unit_index % 2 == 0
    if horizontal_first and dx:
        return ["EAST" if dx > 0 else "WEST"]
    if dy:
        return ["SOUTH" if dy > 0 else "NORTH"]
    if dx:
        return ["EAST" if dx > 0 else "WEST"]
    return ["PASS"]


def _nearest_shed(position: tuple[int, int], board_size: int) -> tuple[int, int]:
    return min(shed_access_tiles(board_size), key=lambda tile: (distance(position, tile), tile[1], tile[0]))


def _return_action(unit: Unit, state: GameState) -> list[Any]:
    if is_shed_adjacent(unit.position, state.board_size):
        return ["DROP"]
    return _step_toward(unit.position, _nearest_shed(unit.position, state.board_size), unit.index)


def assign_actions(state: GameState, tasks: list[Task], config: dict[str, Any]) -> tuple[list[list[Any]], dict[str, int]]:
    units = state.units()
    actions: list[list[Any]] = [["PASS"] for _ in units]
    assigned_units: set[int] = set()
    assigned_tasks: set[str] = set()
    planned_drop: dict[str, int] = {}
    terminal_return = state.day >= int(config["terminal_day"]) and state.hour >= int(config["terminal_return_hour"])

    if terminal_return:
        for unit in units:
            if sum(int(n) for n in unit.inventory.values()) <= 0:
                continue
            actions[unit.index] = _return_action(unit, state)
            assigned_units.add(unit.index)
            if actions[unit.index][0] == "DROP":
                for item, amount in unit.inventory.items():
                    planned_drop[item] = planned_drop.get(item, 0) + int(amount)

    pairs = []
    for task in tasks:
        for unit in units:
            if unit.index in assigned_units:
                continue
            pairs.append((task.priority, distance(unit.position, task.target), task.deadline_step, task.task_id, unit.index, task))
    for _, _, _, _, unit_index, task in sorted(pairs):
        if unit_index in assigned_units or task.task_id in assigned_tasks:
            continue
        unit = units[unit_index]
        actions[unit_index] = list(task.action) if unit.position == task.target else _step_toward(unit.position, task.target, unit.index)
        assigned_units.add(unit_index)
        assigned_tasks.add(task.task_id)

    return actions, planned_drop
```

### `sale_sim.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from .mechanics import PRICE_FLOOR, market_price


@dataclass(frozen=True)
class SaleResult:
    revenue: int
    inventory: int
    units: int
    prices: tuple[int, ...]


def simulate_sale(item: str, inventory: int, quantity: int) -> SaleResult:
    revenue = 0
    prices: list[int] = []
    units = 0
    for _ in range(max(0, int(quantity))):
        price = market_price(item, inventory)
        prices.append(price)
        revenue += price
        units += 1
        if price > PRICE_FLOOR:
            inventory += 1
    return SaleResult(revenue=revenue, inventory=inventory, units=units, prices=tuple(prices))
```

### `market_orders.py`

```python
from __future__ import annotations
from typing import Any
from .mechanics import CROPS, LAND_PRICES, MAX_MARKET_ORDERS, PRODUCTS, hire_cost
from .production import seed_deficits
from .sale_sim import simulate_sale
from .state import GameState

PREMIUM_PRODUCTS = {"STRAWBERRY", "MELON", "MILK", "WOOL"}


def _sale_quantity(item: str, amount: int, state: GameState, config: dict[str, Any]) -> int:
    if amount <= 0:
        return 0
    terminal = state.day >= int(config["terminal_day"])
    if terminal:
        return amount
    mode = config.get("sell_mode", "immediate")
    if mode == "delayed" and state.hour < 18 and sum(int(v) for v in state.shed.values()) < 80:
        return 0
    if mode == "batched" and item in PREMIUM_PRODUCTS:
        return min(amount, int(config.get("premium_batch", 8)))
    return amount


def make_market_orders(state: GameState, config: dict[str, Any], planned_drop: dict[str, int] | None = None) -> list[list[Any]]:
    planned_drop = planned_drop or {}
    orders: list[list[Any]] = []
    projected_cash = float(state.money)

    available = {item: int(state.shed.get(item, 0)) + int(planned_drop.get(item, 0)) for item in PRODUCTS}
    for item in sorted(PRODUCTS, key=lambda p: (-state.market_price(p), p)):
        quantity = _sale_quantity(item, available[item], state, config)
        if quantity <= 0 or len(orders) >= MAX_MARKET_ORDERS:
            continue
        orders.append(["SELL", item, quantity])
        projected_cash += simulate_sale(item, state.market_inventory(item), quantity).revenue

    planned_unlocked = state.unlocked_count
    target_unlocked = int(config.get("target_unlocked", 1))
    if planned_unlocked < target_unlocked and state.day <= 2 and len(orders) < MAX_MARKET_ORDERS:
        price = LAND_PRICES[planned_unlocked - 1]
        if projected_cash >= price + int(config.get("operating_reserve", 0)):
            orders.append(["BUY_LAND"])
            projected_cash -= price
            planned_unlocked += 1

    deficits = seed_deficits(state, config)
    for crop in config.get("crop_order", []):
        quantity = int(deficits.get(crop, 0))
        if quantity <= 0 or len(orders) >= MAX_MARKET_ORDERS:
            continue
        unit_cost = int(CROPS[crop]["seed"])
        reserve = int(config.get("operating_reserve", 0))
        affordable = max(0, int((projected_cash - reserve) // unit_cost))
        quantity = min(quantity, affordable)
        if quantity <= 0:
            continue
        orders.append(["BUY_SEED", crop, quantity])
        projected_cash -= quantity * unit_cost

    hands_by_unlocked = config.get("hands_by_unlocked", {})
    target_hands = int(hands_by_unlocked.get(planned_unlocked, hands_by_unlocked.get(str(planned_unlocked), 0)))
    current_hands = len(state.units()) - 1
    hires_today = int(state.farm.get("hires_today", current_hands)) if isinstance(state.farm, dict) else current_hands
    while current_hands < target_hands and len(orders) < MAX_MARKET_ORDERS:
        cost = hire_cost(hires_today)
        if projected_cash < cost + int(config.get("operating_reserve", 0)):
            break
        orders.append(["HIRE"])
        projected_cash -= cost
        current_hands += 1
        hires_today += 1
    return orders[:MAX_MARKET_ORDERS]
```

### `validator.py`

```python
from __future__ import annotations
from collections import Counter
from typing import Any
from .mechanics import MAX_MARKET_ORDERS, read
from .state import GameState

VALID_UNIT_OPS = {
    "NORTH", "SOUTH", "EAST", "WEST", "PASS", "DROP", "PICKUP", "PLACE",
    "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP",
    "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE",
}
VALID_MARKET_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"}


def validate_joint_action(state: GameState, unit_actions: list[list[Any]], market_orders: list[list[Any]]) -> dict[str, Any]:
    expected = len(read(state.farm, "hands", []) or []) + 1
    actions = [a if isinstance(a, list) and a and a[0] in VALID_UNIT_OPS else ["PASS"] for a in unit_actions[:expected]]
    actions.extend([["PASS"] for _ in range(expected - len(actions))])

    requested = Counter(str(action[1]) for action in actions if len(action) >= 2 and action[0] == "PLANT")
    blocked = {crop for crop, amount in requested.items() if amount > int(state.seeds.get(crop, 0))}
    if blocked:
        actions = [["PASS"] if action[0] == "PLANT" and str(action[1]) in blocked else action for action in actions]

    market = [order for order in market_orders if isinstance(order, list) and order and order[0] in VALID_MARKET_OPS]
    return {"farmer": actions[0], "hands": actions[1:], "market": market[:MAX_MARKET_ORDERS]}


def safe_fallback(state: GameState) -> dict[str, Any]:
    hands = len(read(state.farm, "hands", []) or [])
    sell = [["SELL", item, int(amount)] for item, amount in state.shed.items() if int(amount) > 0][:MAX_MARKET_ORDERS]
    return {"farmer": ["PASS"], "hands": [["PASS"] for _ in range(hands)], "market": sell}
```

### `orchestrator.py` (entrypoint)

```python
from __future__ import annotations
import time
from typing import Any
from .config import CHAMPION_CONFIG, BASELINE_CONFIG
from .market_orders import make_market_orders
from .routing import assign_actions
from .state import GameState
from .tasks import materialize_tasks
from .validator import safe_fallback, validate_joint_action


def decide(observation: Any, config: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    state = GameState.from_observation(observation)
    fallback = safe_fallback(state)
    try:
        tasks = materialize_tasks(state, config)
        if time.monotonic() - started > 0.20:
            return fallback
        unit_actions, planned_drop = assign_actions(state, tasks, config)
        market_orders = make_market_orders(state, config, planned_drop)
        if time.monotonic() - started > 0.40:
            return fallback
        return validate_joint_action(state, unit_actions, market_orders)
    except Exception:
        return fallback


def agent(observation: Any) -> dict[str, Any]:
    return decide(observation, CHAMPION_CONFIG)


def baseline_agent(observation: Any) -> dict[str, Any]:
    return decide(observation, BASELINE_CONFIG)
```

---

## Companion Helpers (Present in Source, Lightly Used Online)

### Opponent classification

```python
from collections import Counter

def classify_visible_farm(farm) -> str:
    crops = Counter()
    animals = 0
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                crops[str(tile.get("crop"))] += 1
            elif tile.get("animal") in ANIMALS:
                animals += 1
    if animals > sum(crops.values()):
        return "livestock-specialist"
    if crops:
        crop, count = crops.most_common(1)[0]
        if count >= max(4, sum(crops.values()) * 2 // 3):
            return f"{crop.lower()}-specialist"
    if len(farm.get("unlocked_quadrants", ["NW"]) or ["NW"]) >= 3:
        return "aggressive-expander"
    return "mixed-or-inactive"
```

### Demand forecast (for extensions)

```python
def expected_town_demand(state, product: str, turns: int) -> int:
    shop_ticks = max(0, turns // 4)
    center_ticks = max(0, turns // 24)
    per_tick = 0
    for shop in state.town.get("unlocked_shops", []) or []:
        products = SHOPS.get(str(shop), ())
        if product in products:
            per_tick += 2 if len(products) == 1 else 1
    return shop_ticks * per_tick + (0 if product == "FERTILIZER" else center_ticks)
```

---

## Port Notes for Your Agent

Highest-value transplants into `ActionController` / `market_planning.py`:

1. `operating_reserve` cash floor on all buys  
2. `planned_drop` same-turn SELL inclusion  
3. `sell_mode="batched"` with `premium_batch=8`  
4. `hands_by_unlocked` hire targets  
5. Overplant PASS validator  
6. `mix_switch_day` + adaptive ±3 staple shift  

See `04_Improvement_Plan_For_My_Agent.md` for injection sites.
