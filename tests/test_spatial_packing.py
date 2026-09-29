"""Tests for high-density spatial packing (routing layer + KM wiring)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaggriculture.routing import (
    FLEX,
    GRAZING,
    HIGH_YIELD,
    LOCKED,
    ZONE_VIOLATION_PENALTY,
    assign_with_penalties,
    packing_cost,
    rank_pasture_sites,
    reclamation_jobs,
    reclamation_plan,
    zone_map,
    zone_of,
)


def _obs(tiles=None, quads=None, farmer=(4, 4), hands=(), day=10):
    t = tiles if tiles is not None else [[None] * 10 for _ in range(10)]
    return {
        "step": day * 24, "day": day, "hour": 0, "player": 0,
        "farms": [{"money": 3000.0, "tiles": t, "farmer": list(farmer),
                   "hands": [list(h) for h in hands], "hires_today": 0,
                   "unlocked_quadrants": list(quads or ["NW"])}],
        "private": {"shed": {}, "seeds": {}, "inventories": []},
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
    }


def test_zone_map_hardcoded_segregation_and_locking():
    zones = zone_map(["NW", "NE", "SW"])
    assert zone_of((0, 0), zones) == GRAZING      # NW grazes
    assert zone_of((7, 2), zones) == HIGH_YIELD   # NE cash crops
    assert zone_of((2, 7), zones) == HIGH_YIELD   # SW cash crops
    assert zone_of((7, 7), zones) == LOCKED       # SE still locked
    full = zone_map(["NW", "NE", "SW", "SE"])
    assert zone_of((7, 7), full) == FLEX


def test_packing_cost_penalizes_cows_in_cash_zones():
    zones = zone_map(["NW", "NE", "SW", "SE"])
    assert packing_cost((6, 1), (7, 1), "animal", zones) >= ZONE_VIOLATION_PENALTY
    assert packing_cost((6, 1), (7, 1), "crop", zones) < 20
    assert packing_cost((1, 1), (2, 1), "crop", zones) >= ZONE_VIOLATION_PENALTY
    assert packing_cost((1, 1), (2, 1), "animal", zones) < 20
    assert ZONE_VIOLATION_PENALTY > 18  # dominates any travel saving


def test_penalized_jobs_lose_to_farther_clean_jobs():
    zones = zone_map(["NW", "NE", "SW", "SE"])
    # One unit in NE; adjacent HY animal job (cost 1+100) vs far grazing one.
    pairs = assign_with_penalties([(6, 1)],
                                  [((7, 1), ["FEED"], "animal"),
                                   ((1, 1), ["FEED"], "animal")], zones)
    assert pairs == [(0, 1)]  # 5 clean beats 1 penalized
    # Mirror: crop job in grazing NW loses to clean NE job.
    pairs = assign_with_penalties([(1, 2)],
                                  [((1, 1), ["WATER"], "crop"),
                                   ((7, 2), ["WATER"], "crop")], zones)
    assert pairs == [(0, 1)]


def test_pasture_sites_never_in_high_yield():
    zones = zone_map(["NW", "NE", "SW", "SE"])
    empties = [(1, 1), (7, 2), (7, 7), (2, 7)]
    ranked = rank_pasture_sites(empties, zones)
    assert (7, 2) not in ranked and (2, 7) not in ranked
    assert ranked[0] == (1, 1)  # grazing first


def test_reclamation_orders_high_yield_first():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[1][1] = {"kind": "WEED"}            # NW weed
    tiles[2][7] = {"kind": "WEED"}            # NE (high-yield) weed
    tiles[7][2] = {"kind": "PLANT", "crop": "WHEAT", "yield_units": 0,
                   "planted_day": 0}          # SW (high-yield) exhausted
    obs = _obs(tiles, quads=["NW", "NE", "SW"], day=20)
    jobs = reclamation_jobs(obs, 0, day=20)
    kinds = [(t, c[0]) for (t, c, _) in jobs]
    assert kinds[0] == ((7, 2), "DIG")        # high-yield weed first
    assert kinds[1][0] == (2, 7)              # high-yield exhausted second
    assert kinds[2] == ((1, 1), "DIG")        # grazing weed last


def test_reclamation_plan_digs_and_steps():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[4][4] = {"kind": "WEED"}
    tiles[0][1] = {"kind": "WEED"}
    obs = _obs(tiles, farmer=(4, 4), hands=[(0, 0)], day=5)
    plan = reclamation_plan(obs, 0, day=5)
    assert plan["farmer"] == ["DIG"]          # co-located: instant DIG
    assert plan["hands"][0] in (["EAST"], ["SOUTH"])  # steps to nearby weed
    assert len(plan["hands"]) == 1


def test_scratch_wiring_exposes_packing_keys():
    import scratch_grandmaster as sg
    tiles = [[None] * 10 for _ in range(10)]
    tiles[2][7] = {"kind": "WEED"}
    obs = _obs(tiles, quads=["NW", "NE"], day=10)
    ops = sg.mechanical_operations_for_trajectory(obs, (0,) * 3)
    assert "_routing" in ops and "_reclamation" in ops and "_packing_zones" in ops
    assert ops["_packing_zones"]["7,2"] == HIGH_YIELD
    # legacy KM signature untouched
    assert sg.kuhn_munkres_route([(0, 0)], [(1, 1)]) == [(0, 0)]
    ops_off = sg.mechanical_operations_for_trajectory(obs, (0,) * 3, packing=False)
    assert "_reclamation" not in ops_off
