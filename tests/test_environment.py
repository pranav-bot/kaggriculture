import pytest
import pandas as pd
from kaggriculture.env import Environment, Observation, PlantTile, AnimalTile, WeedTile
from kaggriculture.actions.controller import ActionController
from kaggriculture.env.items import Plants


def test_environment_run():
    env = Environment(configuration={"episodeSteps": 48})
    final = env.run_env(ActionController(target_crop=Plants.WHEAT), "random")
    assert final is not None and len(final) == 2

    df = env.extract_time_series_metrics()
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    assert "agent_0_cash" in df.columns and "wheat_price" in df.columns


def test_state_helpers():
    env = Environment(configuration={"episodeSteps": 24})
    final = env.run_env(ActionController(target_crop=Plants.MELON), "pass")
    obs = env.get_current_state(final, agent1=True)

    assert "money" in env.get_farm(obs, 0)
    assert "money" in env.get_opponent_farm(obs, 0)
    assert "shed" in env.get_private(obs, 0)
    assert env.get_shed_item_count(obs, 0) + env.get_shed_free_space(obs, 0) == 100
    assert len(env.get_quadrant_tiles(obs, 0, "NW")) == 25
    assert "prices" in env.get_market(obs)
    assert "unlocked_shops" in env.get_town(obs)


def test_observation_model():
    obs = Observation.from_dict({
        "player": 0, "day": 2, "hour": 5, "step": 53,
        "farms": [
            {"money": 3500.0, "tiles": [[None]*10]*10, "farmer": [4,4],
             "hands": [[5,4]], "unlocked_quadrants": ["NW","NE"], "hires_today": 1},
            {"money": 2800.0, "tiles": [[None]*10]*10, "farmer": [4,4],
             "hands": [], "unlocked_quadrants": ["NW"], "hires_today": 0},
        ],
        "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}},
        "town": {"unlocked_shops": ["BAKERY"]},
        "private": {"shed": {"WHEAT": 10}, "seeds": {"WHEAT": 5}, "inventories": [{"WHEAT": 2}, {}]},
    })
    assert obs.player == 0 and obs.step == 53
    assert obs.step_index == 53  # day*24 + hour
    assert obs.my_cash == 3500.0 and obs.opp_cash == 2800.0

    custom_day_length_obs = Observation.from_dict(
        {"day": 1, "hour": 2, "step": 26}
    )
    assert custom_day_length_obs.step_index == 26


def test_tile_models():
    pt = PlantTile.from_dict({"kind": "PLANT", "crop": "WHEAT", "planted_day": 1,
                               "watered_today": True, "consecutive_unwatered": 0,
                               "yield_units": 1, "max_lifespan_step": 144, "fertilized_until_day": -1})
    assert pt.crop == "WHEAT" and pt.watered_today is True
    assert pt.to_dict()["max_lifespan_step"] == 144

    at = AnimalTile.from_dict({"kind": "COOP", "animal": "GOOSE", "yield_units": 2, "fed_today": True})
    assert at.animal == "GOOSE" and at.yield_units == 2

    assert WeedTile().to_dict() == {"kind": "WEED"}
