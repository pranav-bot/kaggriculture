# 03 — Recreated Agent 2 (Architecture Beta)

**Family:** Multi-tape shop-pair router + live repairs + terminal physical search  
**Scope:** Exact recreation of the *recoverable* online logic. Opaque 719-turn action tapes are represented as a `ScheduleLibrary` interface — contents are not reconstructed.

---

## System Diagram

```
ScheduleLibrary (13 tapes × 719 steps)
        │
        ▼
ShopPairRouter.act(obs)
  ├─ step 144: plan_id ← SHOP_PLANS[first_two_shops]
  ├─ step 648: plan_id ← FINAL_PLAN (2)
  ├─ deepcopy(tape[step])
  ├─ weed_queue_repair
  ├─ subtract_advanced_sales
  ├─ advance_sales (+1 turn pull-forward)
  ├─ room_guard (hour 23 → keep ≤ 99)
  └─ step 718: liquidate

Optional TerminalOverlay (steps 712–718)
  ├─ shadow_baseline through parent (no live mutate)
  ├─ plan_terminal: propose harvest/deposit suffixes
  ├─ accept only if dominates(baseline) ∧ deposit_gain ∧ sold_delta ≥ 0
  └─ execute accepted schedule; abandon → resume parent
```

---

## Python Template

### `schedule_types.py`

```python
from __future__ import annotations
from typing import Any, Protocol

Action = dict[str, Any]  # {"farmer": [...], "hands": [...], "market": [...]}
Tape = list[Action]      # length 719 (steps 0..718)


class ScheduleLibrary(Protocol):
    def tapes(self) -> list[Tape]:
        """Return exactly 13 complete tapes of length 719."""
```

### `shop_router.py`

```python
"""Stored-route parent policy with public-state routing and narrow observation-based repairs."""
from __future__ import annotations

import copy
from collections import deque
from typing import Any

TURNS_PER_DAY = 24
ROUTE_STEP = 144
FINAL_PLAN_STEP = 648
LAST_STEP = 718
SHED_CAPACITY = 100
MAX_ORDERS = 10
PRODUCTS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)
WEED_BLOCKED_WORK = {"PLANT", "BUILD_COOP", "BUILD_PASTURE"}
ANIMALS = {"GOOSE", "COW", "SHEEP"}

# Keys: first two unlocked shops in observed order → tape index.
# All unlisted pairs keep plan 0. Plan 2 is the forced endgame liquidation tape.
SHOP_PLANS = {
    ("BAKERY", "YARN_STORE"): 3,
    ("BRUNCH_SPOT", "YARN_STORE"): 4,
    ("FARMERS_MARKET", "YARN_STORE"): 5,
    ("ICE_CREAM_SHOP", "YARN_STORE"): 6,
    ("PET_CAFE", "YARN_STORE"): 5,
    ("PIZZA_SHOP", "YARN_STORE"): 7,
    ("SMOOTHIE_SHOP", "YARN_STORE"): 8,
    ("YARN_STORE", "BAKERY"): 9,
    ("YARN_STORE", "BRUNCH_SPOT"): 9,
    ("YARN_STORE", "FARMERS_MARKET"): 1,
    ("YARN_STORE", "ICE_CREAM_SHOP"): 9,
    ("YARN_STORE", "PET_CAFE"): 10,
    ("YARN_STORE", "PIZZA_SHOP"): 6,
    ("YARN_STORE", "SMOOTHIE_SHOP"): 11,
    ("YARN_STORE", "YARN_STORE"): 12,
}


class FarmView:
    def __init__(self, observation: dict[str, Any]):
        farm = observation["farms"][observation["player"]]
        private = observation["private"]
        self.tiles = farm["tiles"]
        self.positions = [farm["farmer"], *farm["hands"]]
        self.inventories = private["inventories"]
        self.shed = {item: max(0, int(qty)) for item, qty in private["shed"].items()}
        self.prices = observation["market"]["prices"]

    def inventory(self, worker: int) -> dict[str, int]:
        return self.inventories[worker] if worker < len(self.inventories) else {}

    def beside_shed(self, position) -> bool:
        center = len(self.tiles) // 2
        return position[0] in (center - 1, center) and position[1] in (center - 1, center)


class DayState:
    def __init__(self):
        self.plan = 0
        self.last_step = -1
        self.day = -1
        self.queues: dict[int, deque] = {}
        self.sale_due_step = -1
        self.advanced_sales: dict[str, int] = {}


def repair_weeds(action: dict, view: FarmView, state: DayState, step: int) -> None:
    day = step // TURNS_PER_DAY
    if day != state.day:
        state.day = day
        state.queues.clear()

    workers = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
    for worker in range(min(len(workers), len(view.positions))):
        queue = state.queues.setdefault(worker, deque())
        queue.append(list(workers[worker]))
        x, y = view.positions[worker]
        tile = view.tiles[y][x]
        blocked = (
            queue[0][0] in WEED_BLOCKED_WORK
            and isinstance(tile, dict)
            and tile.get("kind") == "WEED"
        )
        workers[worker] = ["DIG"] if blocked else queue.popleft()
    action["farmer"], action["hands"] = workers[0], workers[1:]


def projected_shed(action: dict, view: FarmView) -> dict[str, int]:
    stock = {item: view.shed.get(item, 0) for item in PRODUCTS}
    stock.update(view.shed)
    total = sum(stock.values())
    workers = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
    for worker in range(min(len(workers), len(view.positions))):
        if not view.beside_shed(view.positions[worker]):
            continue
        work = workers[worker]
        operation = work[0] if work else "PASS"
        inventory = view.inventory(worker)
        if operation == "PICKUP" and len(work) >= 2 and work[1] in stock:
            quantity = max(0, int(work[2]) if len(work) >= 3 else 1)
            taken = min(stock[work[1]], quantity)
            stock[work[1]] -= taken
            total -= taken
        elif operation == "DROP":
            for item, held in inventory.items():
                added = min(max(0, int(held)), max(0, SHED_CAPACITY - total))
                if added > 0:
                    stock[item] = stock.get(item, 0) + added
                    total += added
        elif operation == "PLACE" and len(work) >= 2 and work[1] not in ANIMALS:
            item = work[1]
            quantity = max(0, int(work[2]) if len(work) >= 3 else 1)
            added = min(quantity, max(0, int(inventory.get(item, 0))), max(0, SHED_CAPACITY - total))
            if added > 0:
                stock[item] = stock.get(item, 0) + added
                total += added
    return stock


def subtract_advanced_sales(action: dict, state: DayState, step: int) -> None:
    if state.sale_due_step == step:
        remaining = dict(state.advanced_sales)
        for order in action["market"]:
            if order and order[0] == "SELL" and len(order) >= 3:
                item = order[1]
                removed = min(max(0, int(order[2])), remaining.get(item, 0))
                if removed > 0:
                    order[2] = int(order[2]) - removed
                    remaining[item] -= removed
    state.advanced_sales = {}
    state.sale_due_step = -1


def advance_sales(action: dict, view: FarmView, state: DayState, tape: list, step: int) -> None:
    next_step = step + 1
    if next_step > LAST_STEP or next_step % 72 == 0 or step % 4 == 0:
        return
    planned: dict[str, int] = {}
    for order in tape[next_step].get("market") or []:
        if order and order[0] == "SELL" and len(order) >= 3 and order[1] in PRODUCTS:
            item = order[1]
            planned[item] = planned.get(item, 0) + max(0, int(order[2]))
    already_selling = {order[1] for order in action["market"] if order and order[0] == "SELL" and len(order) > 1}
    stock = projected_shed(action, view)
    for item in PRODUCTS:
        if item in ("WHEAT", "FERTILIZER") or item in already_selling:
            continue
        quantity = min(stock.get(item, 0), planned.get(item, 0))
        if quantity <= 0 or int(view.prices.get(item, 0)) < 2:
            continue
        if len(action["market"]) >= MAX_ORDERS:
            break
        action["market"].append(["SELL", item, quantity])
        state.advanced_sales[item] = quantity
    if state.advanced_sales:
        state.sale_due_step = next_step


def room_guard(action: dict, view: FarmView, step: int) -> None:
    if step % TURNS_PER_DAY != TURNS_PER_DAY - 1:
        return
    carried = sum(max(0, int(n)) for inv in view.inventories for n in inv.values())
    total = sum(view.shed.values()) + carried
    if total <= 99:
        return
    needed = total - 99
    if step >= FINAL_PLAN_STEP and needed < 8:
        return
    planned_sells: dict[str, int] = {}
    for order in action.get("market", []):
        if order and order[0] == "SELL" and len(order) >= 3:
            item = order[1]
            planned_sells[item] = planned_sells.get(item, 0) + max(0, int(order[2]))
    priority = sorted(PRODUCTS, key=lambda item: -int(view.prices.get(item, 0)))
    for item in priority:
        available = max(0, view.shed.get(item, 0) - planned_sells.get(item, 0))
        quantity = min(needed, available)
        if quantity <= 0:
            continue
        if len(action["market"]) >= MAX_ORDERS:
            break
        action["market"].append(["SELL", item, quantity])
        planned_sells[item] = planned_sells.get(item, 0) + quantity
        needed -= quantity
        if needed <= 0:
            break


def liquidate(view: FarmView) -> dict[str, Any]:
    workers = [
        ["DROP"] if view.beside_shed(pos) and view.inventory(worker) else ["PASS"]
        for worker, pos in enumerate(view.positions)
    ]
    action = {"farmer": workers[0], "hands": workers[1:], "market": []}
    stock = projected_shed(action, view)
    action["market"] = [["SELL", item, stock[item]] for item in PRODUCTS if stock[item] > 0]
    action["market"].sort(key=lambda order: -int(view.prices.get(order[1], 0)) * order[2])
    return action


class ShopPairRouter:
    def __init__(self, tapes: list[list[dict]]):
        if len(tapes) != 13 or any(len(tape) != LAST_STEP + 1 for tape in tapes):
            raise ValueError("Expected 13 complete, 719-turn action tapes")
        self.tapes = tapes
        self.players: dict[int, DayState] = {}

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        step, player = int(observation["step"]), int(observation["player"])
        state = self.players.get(player)
        if state is None or step <= state.last_step:
            state = self.players[player] = DayState()
        state.last_step = step

        if step == ROUTE_STEP:
            shops = observation["town"]["unlocked_shops"]
            state.plan = SHOP_PLANS.get(tuple(shops[:2]), 0)
        if step == FINAL_PLAN_STEP:
            state.plan = 2

        view = FarmView(observation)
        tape = self.tapes[state.plan]
        action = copy.deepcopy(tape[step])
        repair_weeds(action, view, state, step)
        subtract_advanced_sales(action, state, step)
        advance_sales(action, view, state, tape, step)
        room_guard(action, view, step)
        action["market"] = action["market"][:MAX_ORDERS]
        return liquidate(view) if step == LAST_STEP else action
```

### `terminal_search.py` (overlay core)

```python
"""Seven-turn terminal physical search over a frozen baseline schedule.

Accept a candidate only when it jointly dominates the baseline on:
  - zero overflow
  - per-turn pre-market shed prefixes
  - sold quantities
  - per-actor deposited quantities
and produces positive physical deposit gain plus non-negative sold deltas.
"""
from __future__ import annotations

from copy import deepcopy
from time import perf_counter
from typing import Any

START, FINAL = 712, 718
PRODUCTS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)


class Unsupported(ValueError):
    pass


def _shed_access_tiles(board_size: int = 10):
    h = board_size // 2
    return ((h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h))


def _walk(start, end):
    x, y = start
    tx, ty = end
    return (
        [["EAST"]] * max(0, tx - x)
        + [["WEST"]] * max(0, x - tx)
        + [["SOUTH"]] * max(0, ty - y)
        + [["NORTH"]] * max(0, y - ty)
    )


def _return(pos):
    targets = _shed_access_tiles(10)
    target = min(targets, key=lambda xy: (abs(pos[0] - xy[0]) + abs(pos[1] - xy[1]), targets.index(xy)))
    return _walk(pos, target) + [["DROP"]]


def dominates(candidate: dict, baseline: dict) -> bool:
    if candidate["overflow_units"]:
        return False
    for new, old in zip(candidate["rows"], baseline["rows"]):
        if any(new["pre_market_shed"].get(i, 0) < old["pre_market_shed"].get(i, 0) for i in PRODUCTS):
            return False
        if any(new["sold"].get(i, 0) < old["sold"].get(i, 0) for i in PRODUCTS):
            return False
        for a, b in zip(new["deposited_by_actor"], old["deposited_by_actor"]):
            if any(a.get(i, 0) < b.get(i, 0) for i in PRODUCTS):
                return False
    return True


def _value(run: dict, prices: dict) -> float:
    shed = run["private"]["shed"]
    return sum((run["sold"].get(item, 0) + shed.get(item, 0)) * prices[item] for item in PRODUCTS)


# NOTE: simulate(...) must apply unit actions via a parity unit model and return:
#   {rows, sold, overflow_units, private, farm, actions, events, states}
# Wire your local unit_model.apply here — omitted for brevity; see source unit_model.py mechanics.


def plan_terminal(
    obs: dict,
    config: dict,
    baseline_remaining: list[dict],
    simulate,
    *,
    max_simulations: int = 64,
    passes: int = 1,
    proposals_per_actor: int = 4,
) -> dict[str, Any]:
    begun = perf_counter()
    fallback = {"accepted": False, "reason": "", "actions": None, "simulations": 0}
    try:
        if int(obs.get("step", -1)) != START or len(baseline_remaining) != FINAL - START + 1:
            raise Unsupported("planning requires step 712 and exactly seven actions")
        baseline = simulate(obs, config, baseline_remaining, detailed=True)
        prices = {item: max(1.0, float(obs["market"]["prices"].get(item, 1))) for item in PRODUCTS}
        current = deepcopy(baseline_remaining)
        best = baseline
        baseline_value = best_value = _value(baseline, prices)
        changes, simulations = [], 0
        n = len(baseline["private"]["inventories"])

        # Proposal generation: for each actor, build walk→HARVEST/COLLECT→DROP
        # route suffixes that fit the remaining horizon, ranked by value/distance.
        # (Full _proposals body matches research doc 01; truncated here to the accept loop.)

        if best_value <= baseline_value:
            return {**fallback, "reason": "no positive physical delivery gain", "simulations": simulations}

        physical = simulate(obs, config, current)
        if not dominates(physical, baseline):
            raise Unsupported("no zero-overflow dominating continuation")

        return {
            "accepted": True,
            "reason": "joint physical dominance",
            "baseline": deepcopy(baseline_remaining),
            "actions": current,
            "simulations": simulations,
            "changes": changes,
            "planning_ms": (perf_counter() - begun) * 1000,
        }
    except (Unsupported, KeyError, TypeError, ValueError, IndexError) as exc:
        return {**fallback, "reason": str(exc), "planning_ms": (perf_counter() - begun) * 1000}
```

### `terminal_overlay.py`

```python
"""Bounded overlay that activates only at step 712 when parent is warm on final plan."""
from __future__ import annotations

from copy import copy, deepcopy
from typing import Any

START, FINAL = 712, 718
DEFAULT_SETTINGS = {
    "enabled": True,
    "max_simulations": 64,
    "passes": 1,
    "proposals_per_actor": 4,
}


class TerminalOverlay:
    def __init__(self, parent_router, planner_module, settings=None):
        self.settings = dict(DEFAULT_SETTINGS)
        self.settings.update(settings or {})
        self.parent = parent_router
        self.planner = planner_module
        self.plans: dict[int, dict] = {}
        self.last_steps: dict[int, int] = {}

    def act(self, obs: dict[str, Any], config=None) -> dict[str, Any]:
        seat, step = int(obs["player"]), int(obs["step"])
        previous = self.last_steps.get(seat)
        if step == 0 or (previous is not None and step <= previous):
            self.plans.pop(seat, None)
            self.parent.players.pop(seat, None)
        self.last_steps[seat] = step

        plan = self.plans.get(seat)
        if plan and plan.get("accepted") and START <= step <= FINAL:
            # Serve committed schedule; on safety failure resume parent.
            index = step - START
            return plan["actions"][index]

        if not self.settings["enabled"] or step != START:
            return self.parent.act(obs)

        # Shadow parent through 712..718 without mutating live state, then search.
        try:
            baseline, states_before = self._shadow_baseline(obs, config)
        except Exception:
            return self.parent.act(obs)

        actual = self.parent.act(obs)
        if actual != baseline[0]:
            return actual

        plan = self.planner.plan_terminal(
            obs, config, baseline,
            max_simulations=self.settings["max_simulations"],
            passes=self.settings["passes"],
            proposals_per_actor=self.settings["proposals_per_actor"],
        )
        plan["parent_states_before"] = states_before
        if not plan.get("accepted"):
            return actual
        self.plans[seat] = plan
        return plan["actions"][0]

    def _shadow_baseline(self, obs, config):
        # Exact recreation requires: parent warm through step 711 on plan 2,
        # no weed-blocked queued work, supported shed items only.
        # See full shadow_baseline in research extraction for guard clauses.
        raise NotImplementedError("wire parent clone + simulate loop")
```

### Entrypoint

```python
_ROUTER = None


def agent(observation, configuration=None):
    global _ROUTER
    if _ROUTER is None:
        tapes = load_schedule_library()  # your offline corpus loader
        _ROUTER = ShopPairRouter(tapes)
        # Optional: wrap with TerminalOverlay(_ROUTER, terminal_search)
    return _ROUTER.act(observation)
```

---

## Exact Constants Checklist

| Constant | Value |
|----------|-------|
| Route selection step | 144 |
| Final plan force step | 648 |
| Last playable step | 718 |
| Terminal search window | 712–718 (7 turns) |
| Shed soft cap | 99 |
| Max market orders | 10 |
| Sale advance skip | `step % 4 == 0` or `next % 72 == 0` |
| Advanced-sale price floor | 2 |
| Skip advance for | WHEAT, FERTILIZER |
| Number of tapes | 13 |
| Supported search budgets | (64,1,4), (128,1,8), (256,1,16) |

---

## What You Can Port Without Tapes

Even without a schedule corpus, these Beta mechanics stand alone:

1. `projected_shed` before clamping sells  
2. `room_guard` at hour 23 targeting 99  
3. One-turn sale advance relative to a *planned* next sell  
4. Weed DIG insertion when standing on WEED with PLANT/BUILD queued  
5. Terminal 7-turn deposit search over *your own* open-loop plan  

These are the highest-leverage non-tape transplants into your controller.
