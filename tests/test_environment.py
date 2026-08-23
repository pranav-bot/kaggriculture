import pytest
import pandas as pd
from kaggriculture.env import (
    Environment,
    Observation,
    PlantTile,
    AnimalTile,
    WeedTile,
    FarmState,
    MarketState,
    TownState,
    PrivateState,
)
from kaggriculture.actions.controller import ActionController
from kaggriculture.env.items import Plants


def test_environment_run_with_controller():
    env = Environment(configuration={"episodeSteps": 48})
    controller_wheat = ActionController(target_crop=Plants.WHEAT)

    final_step = env.run_env(controller_wheat, "random")
    assert final_step is not None
    assert len(final_step) == 2

    # Verify time-series metrics extraction
    df = env.extract_time_series_metrics()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "agent_0_cash" in df.columns
    assert "wheat_price" in df.columns
    assert "melon_price" in df.columns


def test_environment_state_helpers():
    env = Environment(configuration={"episodeSteps": 24})
    controller = ActionController(target_crop=Plants.MELON)
    final_step = env.run_env(controller, "pass")

    obs = env.get_current_state(final_step, agent1=True)
    farm = env.get_farm(obs, player=0)
    assert "money" in farm
    assert "tiles" in farm

    opp_farm = env.get_opponent_farm(obs, my_player_id=0)
    assert "money" in opp_farm
    assert "tiles" in opp_farm

    private = env.get_private(obs, player=0)
    assert "shed" in private
    assert "seeds" in private

    item_count = env.get_shed_item_count(obs, player=0)
    free_space = env.get_shed_free_space(obs, player=0, capacity=100)
    assert item_count >= 0
    assert free_space <= 100
    assert item_count + free_space == 100

    nw_tiles = env.get_quadrant_tiles(obs, player=0, quadrant="NW")
    assert len(nw_tiles) == 25

    market = env.get_market(obs)
    assert "prices" in market
    assert "inventory" in market

    town = env.get_town(obs)
    assert "unlocked_shops" in town


def test_typed_observation_models():
    sample_obs = {
        "player": 0,
        "day": 2,
        "hour": 5,
        "farms": [
            {
                "money": 3500.0,
                "tiles": [[None] * 10 for _ in range(10)],
                "farmer": [4, 4],
                "hands": [[5, 4]],
                "unlocked_quadrants": ["NW", "NE"],
                "hires_today": 1,
            },
            {
                "money": 2800.0,
                "tiles": [[None] * 10 for _ in range(10)],
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            },
        ],
        "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}},
        "town": {"unlocked_shops": ["BAKERY"]},
        "private": {
            "shed": {"WHEAT": 10},
            "seeds": {"WHEAT": 5},
            "inventories": [{"WHEAT": 2}, {}],
        },
    }

    obs = Observation.from_dict(sample_obs)
    assert obs.player == 0
    assert obs.day == 2
    assert obs.hour == 5
    assert obs.step_index == 2 * 24 + 5  # step 53
    assert obs.my_cash == 3500.0
    assert obs.opp_cash == 2800.0
    assert obs.my_farm["unlocked_quadrants"] == ["NW", "NE"]

    # PlantTile
    plant_d = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 1,
        "watered_today": True,
        "consecutive_unwatered": 0,
        "yield_units": 1,
        "max_lifespan_step": 144,
        "fertilized_until_day": -1,
    }
    pt = PlantTile.from_dict(plant_d)
    assert pt.crop == "WHEAT"
    assert pt.watered_today is True
    assert pt.to_dict() == plant_d

    # AnimalTile
    animal_d = {
        "kind": "COOP",
        "animal": "GOOSE",
        "placed_day": 0,
        "yield_units": 2,
        "fed_today": True,
        "consecutive_unfed": 0,
        "cared_today": True,
        "fertilizer_available": True,
        "pending_care_bonus": 1,
    }
    at = AnimalTile.from_dict(animal_d)
    assert at.animal == "GOOSE"
    assert at.yield_units == 2
    assert at.to_dict() == animal_d

    # WeedTile
    wt = WeedTile.from_dict({"kind": "WEED"})
    assert wt.kind == "WEED"
    assert wt.to_dict() == {"kind": "WEED"}
