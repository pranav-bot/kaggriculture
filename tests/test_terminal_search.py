from experiments.terminal_search.simulate import simulate_terminal_schedule
from experiments.terminal_search.terminal_search import (
    TERMINAL_HORIZON,
    TERMINAL_START_STEP,
    dominates,
    plan_terminal,
    terminal_search_window,
)


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
