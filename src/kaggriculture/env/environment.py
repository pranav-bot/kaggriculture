import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from kaggle_environments import make
import pandas as pd

from kaggriculture.env.items import get_quadrant_bounds


class Environment:
    """
    Environment wrapper for Kaggriculture that provides seamless execution,
    state inspection, analytical metrics extraction, and replay rendering.
    """

    def __init__(self, configuration: Optional[Dict[str, Any]] = None, debug: bool = True):
        self.configuration = configuration or {}
        self.debug = debug
        self.env = make("kaggriculture", configuration=self.configuration, debug=self.debug)

    def reset(self):
        """Resets the environment."""
        return self.env.reset()

    def run_env(self, agent1: Any, agent2: Any):
        """
        Executes a full match between agent1 and agent2.
        Supports python callable agents, ActionController instances, or built-in string agent names ('random', 'pass', 'starter').
        """
        resolved_agent1 = self._resolve_agent(agent1)
        resolved_agent2 = self._resolve_agent(agent2)

        self.env.run([resolved_agent1, resolved_agent2])
        final = self.env.steps[-1]
        return final

    def _resolve_agent(self, agent: Any):
        if hasattr(agent, "act") and callable(getattr(agent, "act")):
            return lambda obs: agent.act(obs)
        return agent

    def render_match(self, mode: str = "ipython", width: int = 800, height: int = 600):
        """
        Renders the visual replay.
        Use mode='ipython' for notebooks, mode='html' for standalone HTML.
        """
        return self.env.render(mode=mode, width=width, height=height)

    def save_replay(self, filepath: str) -> None:
        """Saves the match replay to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.env.toJSON(), f)

    # --------------------------------------------------------------------------
    # State Inspection Helpers
    # --------------------------------------------------------------------------

    def get_current_state(self, obs: Any, agent1: bool = True) -> Any:
        """Gets observation for agent 1 or agent 2."""
        idx = 0 if agent1 else 1
        if isinstance(obs, list) and len(obs) > idx:
            return obs[idx].observation
        return obs

    def get_farm(self, obs: Any, player: Union[int, bool] = 0) -> Dict[str, Any]:
        """Returns the farm dictionary for the specified player (0 or 1, or True/False)."""
        player_id = 0 if player is True else (1 if player is False else int(player))
        observation = obs if isinstance(obs, dict) else self.get_current_state(obs, player_id == 0)
        farms = observation.get("farms", [])
        if len(farms) > player_id:
            return farms[player_id]
        return {}

    def get_opponent_farm(self, obs: Any, my_player_id: int = 0) -> Dict[str, Any]:
        """Returns the public farm state of the opponent."""
        opp_id = 1 - my_player_id
        return self.get_farm(obs, opp_id)

    def get_private(self, obs: Any, player: Union[int, bool] = 0) -> Dict[str, Any]:
        """Returns private state (shed, seeds, inventories) for this player."""
        player_id = 0 if player is True else (1 if player is False else int(player))
        observation = obs if isinstance(obs, dict) else self.get_current_state(obs, player_id == 0)
        return observation.get("private", {})

    def get_shed_item_count(self, obs: Any, player: Union[int, bool] = 0) -> int:
        """Returns total non-seed items stored in the player's shed."""
        private = self.get_private(obs, player)
        shed = private.get("shed", {})
        return sum(shed.values())

    def get_shed_free_space(self, obs: Any, player: Union[int, bool] = 0, capacity: int = 100) -> int:
        """Returns remaining capacity in the player's shed."""
        return max(0, capacity - self.get_shed_item_count(obs, player))

    def get_market(self, obs: Any) -> Dict[str, Any]:
        """Returns market state (prices and inventory)."""
        observation = obs if isinstance(obs, dict) else self.get_current_state(obs, True)
        return observation.get("market", {})

    def get_town(self, obs: Any) -> Dict[str, Any]:
        """Returns town state (unlocked shops)."""
        observation = obs if isinstance(obs, dict) else self.get_current_state(obs, True)
        return observation.get("town", {})

    def get_plants(self, obs: Any, player: Union[int, bool] = 0) -> List[Tuple[int, int, dict]]:
        """Returns a list of (x, y, tile_dict) for all plants on the player's farm."""
        farm = self.get_farm(obs, player)
        tiles = farm.get("tiles", [])
        plants = []
        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    plants.append((x, y, tile))
        return plants

    def get_animals(self, obs: Any, player: Union[int, bool] = 0) -> List[Tuple[int, int, dict]]:
        """Returns a list of (x, y, tile_dict) for all animal structures with animals on the farm."""
        farm = self.get_farm(obs, player)
        tiles = farm.get("tiles", [])
        animals = []
        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                if isinstance(tile, dict) and "animal" in tile:
                    animals.append((x, y, tile))
        return animals

    def get_weeds(self, obs: Any, player: Union[int, bool] = 0) -> List[Tuple[int, int]]:
        """Returns a list of (x, y) coordinates for all weeds on the farm."""
        farm = self.get_farm(obs, player)
        tiles = farm.get("tiles", [])
        weeds = []
        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                if isinstance(tile, dict) and tile.get("kind") == "WEED":
                    weeds.append((x, y))
        return weeds

    def get_empty_tiles(self, obs: Any, player: Union[int, bool] = 0) -> List[Tuple[int, int]]:
        """Returns a list of (x, y) coordinates for all empty unlocked tiles on the farm."""
        farm = self.get_farm(obs, player)
        tiles = farm.get("tiles", [])
        empty = []
        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                if tile is None:
                    empty.append((x, y))
        return empty

    def get_quadrant_tiles(self, obs: Any, player: Union[int, bool] = 0, quadrant: str = "NW", board_size: int = 10) -> List[Tuple[int, int, Any]]:
        """Returns list of (x, y, tile_content) belonging to the specified quadrant."""
        farm = self.get_farm(obs, player)
        tiles = farm.get("tiles", [])
        xmin, xmax, ymin, ymax = get_quadrant_bounds(quadrant, board_size)
        res = []
        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                res.append((x, y, tiles[y][x]))
        return res

    # --------------------------------------------------------------------------
    # Metrics & Observability
    # --------------------------------------------------------------------------

    def extract_time_series_metrics(self) -> pd.DataFrame:
        """
        Parses the entire match history to create a comprehensive time-series DataFrame.
        Provides observability into market prices, player bank balances, and shed inventories.
        """
        history = []
        for step_idx, step_data in enumerate(self.env.steps):
            obs0 = step_data[0].observation
            obs1 = step_data[1].observation if len(step_data) > 1 else {}

            day = obs0.get("day", 0)
            hour = obs0.get("hour", 0)

            farms = obs0.get("farms", [])
            p1_cash = farms[0].get("money", 0) if len(farms) > 0 else 0
            p2_cash = farms[1].get("money", 0) if len(farms) > 1 else 0

            market = obs0.get("market", {}) or {}
            prices = market.get("prices", {}) or {}
            inventory = market.get("inventory", {}) or {}

            p1_private = obs0.get("private", {}) or {}
            p1_shed = p1_private.get("shed", {}) or {}

            p2_private = obs1.get("private", {}) or {} if isinstance(obs1, dict) else {}
            p2_shed = p2_private.get("shed", {}) or {}

            record = {
                "step": step_idx,
                "day": day,
                "hour": hour,
                "agent_0_cash": p1_cash,
                "agent_1_cash": p2_cash,
                "wheat_price": prices.get("WHEAT", 0),
                "carrot_price": prices.get("CARROT", 0),
                "tomato_price": prices.get("TOMATO", 0),
                "strawberry_price": prices.get("STRAWBERRY", 0),
                "melon_price": prices.get("MELON", 0),
                "egg_price": prices.get("EGG", 0),
                "milk_price": prices.get("MILK", 0),
                "wool_price": prices.get("WOOL", 0),
                "fertilizer_price": prices.get("FERTILIZER", 0),
                "wheat_supply": inventory.get("WHEAT", 0),
                "melon_supply": inventory.get("MELON", 0),
                "agent_0_shed_wheat": p1_shed.get("WHEAT", 0),
                "agent_0_shed_melon": p1_shed.get("MELON", 0),
                "agent_1_shed_wheat": p2_shed.get("WHEAT", 0),
                "agent_1_shed_melon": p2_shed.get("MELON", 0),
            }
            history.append(record)

        return pd.DataFrame(history)