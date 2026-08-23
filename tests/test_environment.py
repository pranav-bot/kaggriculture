import pytest
import pandas as pd
from kaggriculture.env import Environment
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
    assert len(nw_tiles) == 25  # 5x5 = 25 tiles

    market = env.get_market(obs)
    assert "prices" in market
    assert "inventory" in market

    town = env.get_town(obs)
    assert "unlocked_shops" in town
