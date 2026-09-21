# 07 — Recreated Agent 4 (Architecture Delta)

**Family:** Opaque schedule replay + weed repair + impact-scored sell reordering  
**Scope:** Exact recreation of the *online* scoring/repair layer. Compressed action blobs remain behind `load_schedules()`.

---

## Pipeline

```
regime ← "rebalance" if townCenterSellInterval >= 24 else "legacy"
schedule ← schedules[regime]
action ← deepcopy(schedule[step])
action ← weed_repair(action)          # DIG then replay intended BUILD/PLANT
action ← rank_sell_slots(action)      # reorder SELLs by impact × urgency
action ← align_hands(action, obs)
return action
```

---

## Exact Scoring Math

Let \(I\) = current market inventory, \(q\) = sell quantity, \(P(\cdot)\) = official price curve.

\[
\text{impact} = q \cdot \max(0,\, P(I) - P(I+q))
\]

Daily demand \(D\):

\[
D = \sum_{\text{shops containing item}} \frac{24}{\text{shopInterval}} \cdot w_{\text{shop}}
+ \mathbf{1}_{\text{item} \neq \text{FERT}} \cdot \frac{24}{\text{centerInterval}} \cdot m_{\text{day}}
\]

where \(w_{\text{shop}} = 2\) if shop has a single product else \(1\).

Legacy center multiplier \(m_{\text{day}}\):

| Day | Multiplier |
|-----|------------|
| &lt; 10 | 1 |
| 10–19 | 2 |
| ≥ 20 | 4 |

Rebalance regime: \(m_{\text{day}} = 1\), center default interval 24.

Urgency (rebalance only, when impact &gt; 0):

\[
\text{urgency} = \min\!\left(1,\; \frac{\max(0, I+q-10000)}{10 \cdot \max(0.25, D)}\right)
\]

\[
\text{score} = \text{impact} \cdot (1 + 0.25 \cdot \text{urgency})
\]

SELL slots are stably sorted by `(score, -original_index)` and written back into their original SELL positions (non-SELL orders stay fixed).

---

## Python Template

```python
"""Opaque schedule + impact sell ranking + multi-step weed repair."""
from __future__ import annotations

import copy
import math
from typing import Any

_PRICE_FLOOR = 1
_DEMAND_ALPHA = 0.25
_WEED_REPLAY_STEPS = 8

_MARKET_PARAMS = {
    "WHEAT": (25, 10000, 400, "sqrt", 0.8, "log", 0.2),
    "CARROT": (35, 10000, 450, "log", 0.2, "sqrt", 0.7),
    "TOMATO": (60, 10000, 200, "linear", 0.4, "sqrt", 0.6),
    "STRAWBERRY": (120, 10000, 100, "sqrt", 0.7, "linear", 1.6),
    "MELON": (250, 10000, 300, "log", 0.2, "sq", 3.6),
    "EGG": (50, 10000, 332, "linear", 0.4, "log", 0.2),
    "MILK": (160, 10000, 122, "sqrt", 0.6, "linear", 1.6),
    "WOOL": (200, 10000, 105, "log", 0.2, "sq", 3.2),
    "FERTILIZER": (100, 10000, 200, "linear", 0.4, "linear", 0.4),
}

_SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}

_WEED_STATE: dict[int, dict] = {0: {}, 1: {}}


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _regime(configuration) -> str:
    interval = int(_get(configuration, "townCenterSellInterval", 12) or 12)
    return "rebalance" if interval >= 24 else "legacy"


def _copy_action(action) -> dict:
    action = copy.deepcopy(action or {})
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(order or ["PASS"]) for order in (action.get("hands") or [])],
        "market": [list(order) for order in (action.get("market") or [])],
    }


def _seat(obs) -> int:
    return 1 if int(_get(obs, "player", 0) or 0) == 1 else 0


def _farm(obs, seat):
    farms = list(_get(obs, "farms", []) or [])
    return farms[seat] if seat < len(farms) else {}


def _align_hands(action, obs):
    action = _copy_action(action)
    expected = len(_get(_farm(obs, _seat(obs)), "hands", []) or [])
    hands = list(action.get("hands") or [])
    if len(hands) < expected:
        hands.extend([["PASS"] for _ in range(expected - len(hands))])
    action["hands"] = [list(order or ["PASS"]) for order in hands[:expected]]
    return action


def _tile_at(farm, position):
    try:
        x, y = int(position[0]), int(position[1])
        return (_get(farm, "tiles", []) or [])[y][x]
    except (IndexError, TypeError, ValueError):
        return "LOCKED"


def _trace_actor_action(actions, step, actor):
    trace = actions[min(max(int(step), 0), len(actions) - 1)] or {}
    if actor == "farmer":
        return list(trace.get("farmer") or ["PASS"])
    hands = trace.get("hands", []) or []
    return list(hands[actor] if actor < len(hands) else ["PASS"])


def weed_repair_action(obs, action, actions, step):
    """If BUILD_PASTURE/PLANT lands on WEED: DIG now, replay intended next turn,
    then echo prior-step tape actions for _WEED_REPLAY_STEPS turns."""
    action = _align_hands(action, obs)
    seat = _seat(obs)
    game = _WEED_STATE[seat]
    if step == 0 or step < game.get("last_step", -1):
        game = {"last_step": step, "active": {}}
        _WEED_STATE[seat] = game
    game["last_step"] = step
    farm = _farm(obs, seat)
    positions = [_get(farm, "farmer"), *list(_get(farm, "hands", []) or [])]
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    active = game["active"]

    for actor, transaction in list(active.items()):
        index = 0 if actor == "farmer" else int(actor) + 1
        if index >= len(unit_actions):
            active.pop(actor, None)
            continue
        age = step - transaction["start"]
        if age == 1:
            unit_actions[index] = list(transaction["intended"])
        elif 2 <= age <= 1 + _WEED_REPLAY_STEPS:
            unit_actions[index] = _trace_actor_action(actions, step - 1, actor)
        else:
            active.pop(actor, None)

    for index, (position, intended) in enumerate(zip(positions, unit_actions)):
        actor = "farmer" if index == 0 else index - 1
        if actor in active or not isinstance(intended, list) or not intended:
            continue
        if intended[0] not in ("BUILD_PASTURE", "PLANT"):
            continue
        tile = _tile_at(farm, position)
        if not isinstance(tile, dict) or tile.get("kind") != "WEED":
            continue
        active[actor] = {"start": step, "intended": list(intended)}
        unit_actions[index] = ["DIG"]

    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
    action["hands"] = unit_actions[1:]
    return _align_hands(action, obs)


def _shape(name, value):
    value = max(0.0, float(value))
    if name == "linear":
        return value
    if name == "sq":
        return value * value
    if name == "sqrt":
        return math.sqrt(value)
    if name == "log":
        return math.log1p(value)
    if name == "log10":
        return math.log10(1.0 + value)
    raise ValueError(name)


def market_price(item, inventory):
    base, equilibrium, scale, below_func, below_target, above_func, above_target = _MARKET_PARAMS[item]
    if inventory < equilibrium:
        amplitude = below_target * base / _shape(below_func, scale)
        price = base + amplitude * _shape(below_func, equilibrium - inventory)
    else:
        amplitude = above_target * base / _shape(above_func, scale)
        price = base - amplitude * _shape(above_func, inventory - equilibrium)
    return max(_PRICE_FLOOR, int(round(price)))


def _is_sell(order) -> bool:
    return (
        isinstance(order, (list, tuple))
        and len(order) >= 3
        and order[0] == "SELL"
        and order[1] in _MARKET_PARAMS
    )


def impact_score(obs, order) -> float:
    if not _is_sell(order):
        return float("-inf")
    item = str(order[1])
    try:
        quantity = max(0, int(order[2]))
    except (TypeError, ValueError):
        return 0.0
    market = _get(obs, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    prices = _get(market, "prices", {}) or {}
    current_inventory = int(_get(inventory, item, 10000) or 0)
    current_quote = float(_get(prices, item, market_price(item, current_inventory)) or 0)
    later_quote = float(market_price(item, current_inventory + quantity))
    return float(quantity) * max(0.0, current_quote - later_quote)


def demand_per_day(obs, configuration, item) -> float:
    town = _get(obs, "town", {}) or {}
    shops = list(_get(town, "unlocked_shops", []) or [])
    turns_per_day = int(_get(configuration, "turnsPerDay", 24) or 24)
    shop_interval = max(1, int(_get(configuration, "townShopSellInterval", 4) or 4))
    demand = 0.0
    for shop in shops:
        products = _SHOP_PRODUCTS.get(shop, ())
        if item in products:
            demand += (turns_per_day / shop_interval) * (2 if len(products) == 1 else 1)
    regime = _regime(configuration)
    if item != "FERTILIZER":
        center_default = 24 if regime == "rebalance" else 12
        center_interval = max(1, int(_get(configuration, "townCenterSellInterval", center_default) or center_default))
        day = int(_get(obs, "day", int(_get(obs, "step", 0) or 0) // 24) or 0)
        multiplier = 1 if regime == "rebalance" else (4 if day >= 20 else 2 if day >= 10 else 1)
        demand += (turns_per_day / center_interval) * multiplier
    return demand


def order_score(obs, configuration, order) -> float:
    score = impact_score(obs, order)
    if _regime(configuration) != "rebalance" or score <= 0 or not _is_sell(order):
        return score
    item = str(order[1])
    quantity = max(0, int(order[2]))
    market = _get(obs, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    current_inventory = int(_get(inventory, item, 10000) or 0)
    demand = max(0.25, demand_per_day(obs, configuration, item))
    excess = max(0.0, current_inventory + quantity - 10000)
    urgency = min(1.0, (excess / demand) / 10.0)
    return score * (1.0 + _DEMAND_ALPHA * urgency)


def rank_sell_slots(obs, action, configuration):
    action = _copy_action(action)
    market = list(action.get("market") or [])
    rows = [
        (order_score(obs, configuration, order), -index, list(order))
        for index, order in enumerate(market)
        if _is_sell(order)
    ]
    if len(rows) < 2:
        return action
    rows.sort(reverse=True)
    ranked = iter(row[2] for row in rows)
    action["market"] = [next(ranked) if _is_sell(order) else order for order in market]
    return action


def agent(obs, configuration=None):
    try:
        schedules = load_schedules()  # {"legacy": [...], "rebalance": [...]}
        actions = schedules["rebalance" if _regime(configuration) == "rebalance" else "legacy"]
        step = min(max(0, int(_get(obs, "step", 0) or 0)), len(actions) - 1)
        action = weed_repair_action(obs, _copy_action(actions[step]), actions, step)
        return _align_hands(rank_sell_slots(obs, action, configuration), obs)
    except Exception:
        farm = _farm(obs, _seat(obs))
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])],
            "market": [],
        }
```

---

## Port Into Your Agent (No Schedule Needed)

You already have `simulate_sell_slippage` in `helpers/market_prediction.py`. Wire impact ranking onto whatever market list `plan_market_actions` emits:

```python
def rank_my_sells(obs, orders, configuration=None):
    sells = [o for o in orders if o and o[0] == "SELL"]
    others = [o for o in orders if not (o and o[0] == "SELL")]
    sells.sort(key=lambda o: order_score(obs, configuration or {}, o), reverse=True)
    return (sells + others)[:10]
```

This is Architecture Delta’s highest-ROI transplant.
