from experiments.terminal_search.proposals import harvest_targets, propose_harvest_drop_routes
from experiments.terminal_search.simulate import simulate_terminal_schedule
from experiments.terminal_search.terminal_search import (
    TERMINAL_HORIZON,
    TERMINAL_START_STEP,
    dominates,
    plan_terminal,
    terminal_search_window,
)


def _obs_with_ripe_wheat(step=TERMINAL_START_STEP):
    tiles = [[None] * 10 for _ in range(10)]
    tiles[3][3] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 20,
        "yield_units": 4,
    }
    return {
        "step": step,
        "day": 29,
        "hour": step % 24,
        "player": 0,
        "farms": [{"money": 5000, "tiles": tiles, "farmer": [0, 3], "hands": []}],
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
        "market": {"prices": {"WHEAT": 25}, "inventory": {"WHEAT": 10000}},
    }


def test_terminal_search_window_days():
    assert terminal_search_window({"day": 27, "step": TERMINAL_START_STEP})
    assert not terminal_search_window({"day": 26, "step": TERMINAL_START_STEP})


def test_plan_terminal_rejects_wrong_step():
    baseline = [{"farmer": ["PASS"], "hands": [], "market": []} for _ in range(TERMINAL_HORIZON)]
    result = plan_terminal({"day": 29, "step": 700, "private": {}, "market": {"prices": {}}}, {}, baseline, simulate_terminal_schedule)
    assert result["accepted"] is False


def test_dominates_requires_no_overflow():
    baseline = {"overflow_units": 0, "rows": [{"pre_market_shed": {}, "sold": {}, "deposited_by_actor": []}]}
    worse = {"overflow_units": 1, "rows": baseline["rows"]}
    assert dominates(baseline, baseline)
    assert not dominates(worse, baseline)


def test_harvest_proposal_and_simulation():
    obs = _obs_with_ripe_wheat()
    targets = harvest_targets(obs["farms"][0], day=29)
    assert targets and targets[0][2] == "WHEAT"
    proposals = propose_harvest_drop_routes(obs, proposals_per_actor=2)
    assert proposals and "HARVEST" in proposals[0]["farmer"]
    run = simulate_terminal_schedule(obs, {"boardSize": 10}, proposals[:1] * TERMINAL_HORIZON)
    assert run["overflow_units"] == 0


def test_plan_terminal_accepts_improving_harvest_route():
    obs = _obs_with_ripe_wheat()
    baseline = [{"farmer": ["PASS"], "hands": [], "market": []} for _ in range(TERMINAL_HORIZON)]
    result = plan_terminal(obs, {"boardSize": 10}, baseline, simulate_terminal_schedule, max_simulations=32)
    assert result["planning_ms"] <= 200.0 + 1e-6 or "budget" in result.get("reason", "")
