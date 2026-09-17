"""
Example 05 — Manual action construction (no ActionController).

Builds the action dict directly from the observation. Useful when you need
full control over unit routing or want a minimal agent with no library overhead
in a single-file submission.
"""
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import Actions, CROPS


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _direction_towards(src: Tuple[int, int], dst: Tuple[int, int]) -> List[str]:
    sx, sy = src
    tx, ty = dst
    if sx == tx and sy == ty:
        return Actions.pass_action()
    if abs(tx - sx) >= abs(ty - sy):
        return Actions.east() if tx > sx else Actions.west()
    return Actions.south() if ty > sy else Actions.north()


def _find_nearest_tile(
    pos: Tuple[int, int],
    tiles: List[List[Any]],
    predicate,
) -> Optional[Tuple[int, int]]:
    best: Optional[Tuple[int, int]] = None
    best_dist = 10**9
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if tile is None or not predicate(tile):
                continue
            dist = _manhattan(pos, (x, y))
            if dist < best_dist:
                best_dist = dist
                best = (x, y)
    return best


def _plan_farmer(farm: dict, private: dict, day: int) -> List[str]:
    farmer = farm["farmer"]
    pos = (farmer["x"], farmer["y"])
    tiles = farm["tiles"]
    seeds = private.get("seeds", {})

    # 1. Water a thirsty crop on this tile
    here = tiles[pos[1]][pos[0]]
    if here and here.get("kind") == "PLANT" and not here.get("watered_today", False):
        return Actions.water()

    # 2. Harvest ripe wheat on this tile
    if here and here.get("kind") == "PLANT":
        cfg = CROPS[here.get("crop", "WHEAT")]
        if cfg.is_harvestable(here.get("planted_day", day), day, here.get("yield_units", 0)):
            return Actions.harvest()

    # 3. Plant wheat on empty unlocked tile
    if here is None and seeds.get("WHEAT", 0) > 0:
        return Actions.plant_wheat()

    # 4. Move toward nearest thirsty crop
    thirsty = _find_nearest_tile(
        pos,
        tiles,
        lambda t: t.get("kind") == "PLANT" and not t.get("watered_today", False),
    )
    if thirsty:
        return _direction_towards(pos, thirsty)

    # 5. Move toward nearest empty tile to plant
    vacant = _find_nearest_tile(pos, tiles, lambda t: t is None)
    if vacant and seeds.get("WHEAT", 0) > 0:
        return _direction_towards(pos, vacant)

    return Actions.pass_action()


def _plan_market(farm: dict, private: dict, market: dict, day: int) -> List[List[Any]]:
    orders: List[List[Any]] = []
    money = float(farm.get("money", 0))
    seeds = private.get("seeds", {})
    shed = private.get("shed", {})

    if seeds.get("WHEAT", 0) < 5 and money >= 50:
        orders.append(Actions.buy_seed("WHEAT", min(5, int(money // 10))))

    wheat_in_shed = int(shed.get("WHEAT", 0))
    if wheat_in_shed > 0 and float(market.get("prices", {}).get("WHEAT", 0)) >= 20:
        orders.append(Actions.sell("WHEAT", wheat_in_shed))

    if day < 15 and money >= 1000:
        orders.append(Actions.buy_land())

    return orders


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    player = obs["player"]
    farm = obs["farms"][player]
    private = obs.get("private", {})
    market = obs.get("market", {})
    day = int(obs.get("day", 0))

    hands = farm.get("hands", [])
    return {
        "farmer": _plan_farmer(farm, private, day),
        "hands": [Actions.pass_action() for _ in hands],
        "market": _plan_market(farm, private, market, day),
    }
