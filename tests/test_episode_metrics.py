"""Smoke test for episode metrics hooks (24-step episode)."""

import json

from kaggriculture.actions.controller import ActionController
from kaggriculture.env import Environment
from kaggriculture.env.items import Plants
from kaggriculture.helpers.episode_metrics import (
    EpisodeMetricsRecorder,
    MetricsWrapper,
    count_unfillable_sell_slots,
)


def test_count_unfillable_sell_slots():
    planned = [["SELL", "WHEAT", 10], ["SELL", "MELON", 5]]
    after = [["SELL", "WHEAT", 10]]
    assert count_unfillable_sell_slots(planned, after) == 1


def test_episode_metrics_smoke_24_steps():
    recorder = EpisodeMetricsRecorder()
    controller = ActionController(
        target_crop=Plants.WHEAT,
        auto_hire_hands=True,
        max_hires_per_day=2,
    )
    wrapped = MetricsWrapper(controller, recorder)

    env = Environment(configuration={"episodeSteps": 24})
    final = env.run_env(wrapped, "pass")
    obs = env.get_current_state(final, agent1=True)
    summary = recorder.finalize(obs, steps=24)

    assert summary["steps"] == 24
    assert "decide_ms_p50" in summary
    assert "decide_ms_p95" in summary
    assert "mean_impact_score_of_sells" in summary
    assert "unfillable_sell_slots_burned" in summary
    assert "shed_overflow_units_lost" in summary
    assert "collision_holds" in summary
    assert "collision_releases" in summary
    assert "terminal_carried_goods_step_718" in summary
    assert "hires_by_day" in summary
    assert summary["decide_ms_p50"] >= 0.0
    json.dumps(summary)
