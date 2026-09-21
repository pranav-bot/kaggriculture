"""Market timing overlays (research/08): premium delay, collision hold, clone-like gate."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from kaggriculture.env.items import MAX_MARKET_ORDERS_PER_TURN

PREMIUM_ITEMS = frozenset({"MILK", "WOOL", "STRAWBERRY", "FERTILIZER"})

PREMIUM_BASE_PRICE: dict[str, int] = {
    "MILK": 160,
    "WOOL": 200,
    "STRAWBERRY": 120,
    "FERTILIZER": 100,
}

_BUY_OPS = frozenset({"BUY_ANIMAL", "BUY_LAND", "BUY_PRODUCT"})

_premium_pending: dict[str, int] = defaultdict(int)
_premium_age: dict[str, int] = defaultdict(int)

_collision_prev_inv: dict[str, Any] | None = None
_collision_prev_price: dict[str, Any] | None = None
_collision_pending: dict[str, int] = defaultdict(int)
_collision_age: dict[str, int] = defaultdict(int)


def reset_market_overlay_state() -> None:
    """Clear overlay memory (for tests and episode restarts)."""
    _premium_pending.clear()
    _premium_age.clear()
    _collision_pending.clear()
    _collision_age.clear()
    global _collision_prev_inv, _collision_prev_price
    _collision_prev_inv = None
    _collision_prev_price = None


def public_farm_counts(farm: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    animals = {"COW": 0, "SHEEP": 0, "GOOSE": 0}
    crops = {"WHEAT": 0, "CARROT": 0, "TOMATO": 0, "STRAWBERRY": 0, "MELON": 0}
    for row in farm.get("tiles", []) or []:
        for tile in row or []:
            if not isinstance(tile, dict):
                continue
            animal = tile.get("animal")
            if animal in animals:
                animals[str(animal)] += 1
            if tile.get("kind") == "PLANT" and tile.get("crop") in crops:
                crops[str(tile["crop"])] += 1
    return animals, crops


def clone_like(obs: Mapping[str, Any]) -> bool:
    """True when opponent public farm layout closely matches ours (overlay C gate)."""
    farms = obs.get("farms") or []
    player = int(obs.get("player", 0) or 0)
    if len(farms) != 2:
        return False
    me = farms[player] or {}
    opp = farms[1 - player] or {}
    my_animals, my_crops = public_farm_counts(me)
    opp_animals, opp_crops = public_farm_counts(opp)
    hand_gap = abs(len(me.get("hands", []) or []) - len(opp.get("hands", []) or []))
    land_gap = abs(len(me.get("unlocked_quadrants", []) or []) - len(opp.get("unlocked_quadrants", []) or []))
    animal_gap = sum(abs(my_animals[k] - opp_animals[k]) for k in my_animals)
    crop_gap = sum(abs(my_crops[k] - opp_crops[k]) for k in my_crops)
    return hand_gap <= 1 and land_gap == 0 and animal_gap <= 2 and crop_gap <= 5


def _player_farm(obs: Mapping[str, Any]) -> dict[str, Any]:
    player = int(obs.get("player", 0) or 0)
    farms = obs.get("farms") or []
    if player < len(farms) and isinstance(farms[player], dict):
        return farms[player]
    return {}


def _shed_total(obs: Mapping[str, Any]) -> int:
    shed = (obs.get("private") or {}).get("shed") or {}
    return sum(int(v or 0) for v in shed.values() if isinstance(v, (int, float)))


def premium_phase_shift(obs: Mapping[str, Any], base_action: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay A: delay premium SELLs by one turn when safe (cash/shed/day gates)."""
    act = dict(base_action)
    day = int(obs.get("day", 0) or 0)
    money = float(_player_farm(obs).get("money", 0) or 0)
    shed = (obs.get("private") or {}).get("shed") or {}
    shed_total = _shed_total(obs)

    market = list(act.get("market", []) or [])
    out: list[list[Any]] = []
    released: set[str] = set()

    cash_critical = money < 1600 or any(
        isinstance(order, list) and order and order[0] in _BUY_OPS for order in market
    )
    pressure = shed_total >= 82
    endgame = day >= 27
    safe_to_delay = not cash_critical and not pressure and not endgame

    for item in list(_premium_pending):
        if _premium_pending[item] > 0:
            _premium_age[item] += 1

    for order in market:
        is_premium_sell = (
            isinstance(order, list)
            and len(order) >= 3
            and order[0] == "SELL"
            and str(order[1]) in PREMIUM_ITEMS
        )
        if not is_premium_sell:
            out.append(list(order))
            continue
        item = str(order[1])
        if _premium_pending[item] > 0:
            out.append(list(order))
            _premium_pending[item] = 0
            _premium_age[item] = 0
            released.add(item)
            continue
        if safe_to_delay:
            _premium_pending[item] = min(100, max(0, int(order[2])))
            _premium_age[item] = 0
            continue
        out.append(list(order))

    if len(out) < MAX_MARKET_ORDERS_PER_TURN:
        for item in PREMIUM_ITEMS:
            if len(out) >= MAX_MARKET_ORDERS_PER_TURN:
                break
            if item in released or _premium_pending[item] <= 0:
                continue
            if _premium_age[item] < 1 and not endgame:
                continue
            qty = min(int(shed.get(item, 0) or 0), int(_premium_pending[item]))
            if qty > 0:
                out.append(["SELL", item, qty])
            _premium_pending[item] = 0
            _premium_age[item] = 0

    act["market"] = out[:MAX_MARKET_ORDERS_PER_TURN]
    return act


def collision_guard(obs: Mapping[str, Any], base_action: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay B: hold premium SELLs when market inventory rises or price drops sharply."""
    global _collision_prev_inv, _collision_prev_price

    act = dict(base_action)
    day = int(obs.get("day", 0) or 0)
    money = float(_player_farm(obs).get("money", 0) or 0)
    shed = (obs.get("private") or {}).get("shed") or {}
    shed_total = _shed_total(obs)
    market_info = obs.get("market") or {}
    inv = dict(market_info.get("inventory") or {})
    price = dict(market_info.get("prices") or {})

    delta = {item: 0.0 for item in PREMIUM_ITEMS}
    pdelta = {item: 0.0 for item in PREMIUM_ITEMS}
    if _collision_prev_inv is not None and _collision_prev_price is not None:
        for item in PREMIUM_ITEMS:
            delta[item] = float(inv.get(item, 0) or 0) - float(_collision_prev_inv.get(item, 0) or 0)
            pdelta[item] = float(price.get(item, 0) or 0) - float(_collision_prev_price.get(item, 0) or 0)

    market = list(act.get("market", []) or [])
    out: list[list[Any]] = []
    released: set[str] = set()

    for item in list(_collision_pending):
        if _collision_pending[item] > 0:
            _collision_age[item] += 1

    endgame = day >= 27
    pressure = shed_total >= 84
    cash_critical = money < 1500 or any(
        isinstance(order, list) and order and order[0] in _BUY_OPS for order in market
    )

    for order in market:
        is_premium_sell = (
            isinstance(order, list)
            and len(order) >= 3
            and order[0] == "SELL"
            and str(order[1]) in PREMIUM_ITEMS
        )
        if not is_premium_sell:
            out.append(list(order))
            continue
        item = str(order[1])
        if _collision_pending[item] > 0:
            out.append(list(order))
            _collision_pending[item] = 0
            _collision_age[item] = 0
            released.add(item)
            continue
        base = PREMIUM_BASE_PRICE[item]
        ratio = float(price.get(item, base) or base) / base
        saturating = delta[item] >= 3 or pdelta[item] <= -3
        if saturating and ratio < 1.10 and not endgame and not pressure and not cash_critical:
            _collision_pending[item] = min(100, max(0, int(order[2])))
            _collision_age[item] = 0
            continue
        out.append(list(order))

    if len(out) < MAX_MARKET_ORDERS_PER_TURN:
        for item in PREMIUM_ITEMS:
            if len(out) >= MAX_MARKET_ORDERS_PER_TURN or item in released or _collision_pending[item] <= 0:
                continue
            recovering = delta[item] <= -1 or pdelta[item] >= 1
            if not recovering and _collision_age[item] < 2 and not endgame:
                continue
            qty = min(int(shed.get(item, 0) or 0), int(_collision_pending[item]))
            if qty > 0:
                out.append(["SELL", item, qty])
            _collision_pending[item] = 0
            _collision_age[item] = 0

    _collision_prev_inv, _collision_prev_price = inv, price
    act["market"] = out[:MAX_MARKET_ORDERS_PER_TURN]
    return act
