from typing import Dict, List, Optional, Set, Tuple, Any, Union

from kaggriculture.actions.actions import (
    Actions,
    CropConfig,
    AnimalConfig,
    CROPS,
    ANIMALS,
    get_crop,
    get_animal,
)
from kaggriculture.env.items import Plants, Animals, Products, Structures, Quadrants


class ActionController:
    """
    Intelligent high-level action controller and tactical coordinator for Kaggriculture.
    Coordinates farmer movements, hired hands, field maintenance, and market trading.
    """

    def __init__(
        self,
        target_crop: Union[str, Plants] = Plants.WHEAT,
        auto_water: bool = True,
        auto_harvest: bool = True,
        auto_fertilize: bool = False,
        auto_sell: bool = True,
        auto_expand_land: bool = True,
        auto_hire_hands: bool = False,
        max_hires_per_day: int = 1,
        min_sell_margin: float = 0.8,
        board_size: int = 10,
    ):
        self.target_crop = str(target_crop).upper()
        self.auto_water = auto_water
        self.auto_harvest = auto_harvest
        self.auto_fertilize = auto_fertilize
        self.auto_sell = auto_sell
        self.auto_expand_land = auto_expand_land
        self.auto_hire_hands = auto_hire_hands
        self.max_hires_per_day = max_hires_per_day
        self.min_sell_margin = min_sell_margin
        self.board_size = board_size
        self._assigned_targets: Set[Tuple[int, int]] = set()

    # --------------------------------------------------------------------------
    # Grid & Navigation Utilities
    # --------------------------------------------------------------------------

    @staticmethod
    def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @staticmethod
    def get_shed_adjacent_tiles(board_size: int = 10) -> List[Tuple[int, int]]:
        half = board_size // 2
        return [
            (half - 1, half - 1),
            (half, half - 1),
            (half - 1, half),
            (half, half),
        ]

    @staticmethod
    def is_shed_adjacent(pos: Tuple[int, int], board_size: int = 10) -> bool:
        return tuple(pos) in set(ActionController.get_shed_adjacent_tiles(board_size))

    @staticmethod
    def nearest_shed_tile(pos: Tuple[int, int], board_size: int = 10) -> Tuple[int, int]:
        tiles = ActionController.get_shed_adjacent_tiles(board_size)
        return min(tiles, key=lambda t: ActionController.manhattan_distance(pos, t))

    @staticmethod
    def direction_towards(src: Tuple[int, int], target: Tuple[int, int]) -> str:
        """Determines the single best cardinal move direction from src to target."""
        sx, sy = src
        tx, ty = target
        dx = tx - sx
        dy = ty - sy

        if dx == 0 and dy == 0:
            return Actions.PASS

        # Prioritize axis with largest distance
        if abs(dx) >= abs(dy):
            return Actions.EAST if dx > 0 else Actions.WEST
        else:
            return Actions.SOUTH if dy > 0 else Actions.NORTH

    def move_to(self, src: Tuple[int, int], target: Tuple[int, int]) -> List[str]:
        """Returns the movement action to advance from src towards target."""
        direction = self.direction_towards(src, target)
        return [direction]

    # --------------------------------------------------------------------------
    # Target Finding & Task Planning
    # --------------------------------------------------------------------------

    def find_best_tile_for_unit(
        self,
        unit_pos: Tuple[int, int],
        farm: dict,
        private: dict,
        current_day: int,
        current_step: int,
        has_seeds: bool,
        claimed_tiles: Set[Tuple[int, int]],
    ) -> Optional[Tuple[int, int, str]]:
        """
        Finds the highest priority tile and task for a given farmer/hand unit.
        Prioritization:
          1. Critical watering (in danger of becoming weed)
          2. Optimal harvesting (reached max yield / ready)
          3. Regular watering (needs water today)
          4. Emergency animal feeding / care
          5. Planting empty tiles
          6. Clearing weeds
        """
        ux, uy = unit_pos
        board_size = len(farm["tiles"])
        candidates = []

        for y in range(board_size):
            for x in range(board_size):
                pos = (x, y)
                if pos in claimed_tiles and pos != unit_pos:
                    continue

                tile = farm["tiles"][y][x]
                if tile == "LOCKED":
                    continue

                dist = self.manhattan_distance(unit_pos, pos)

                # Case 1: Plant Tile
                if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    crop_name = tile["crop"]
                    crop_cfg = CROPS.get(crop_name)
                    if not crop_cfg:
                        continue

                    # 1a. Critical water danger
                    if crop_cfg.in_danger_of_weed(tile):
                        candidates.append((0, dist, x, y, "water"))
                        continue

                    # 1b. Ripe / Optimal Harvest
                    if crop_cfg.is_optimal_harvest_age(tile["planted_day"], current_day, tile.get("fertilized_until_day", -1) >= current_day) and tile.get("yield_units", 0) > 0:
                        candidates.append((1, dist, x, y, "harvest"))
                        continue

                    # 1c. Regular watering
                    if crop_cfg.needs_watering(tile):
                        candidates.append((2, dist, x, y, "water"))
                        continue

                    # 1d. Non-optimal but harvestable
                    if crop_cfg.is_harvestable(tile["planted_day"], current_day, tile.get("yield_units", 0)):
                        candidates.append((3, dist, x, y, "harvest"))
                        continue

                # Case 2: Animal Tile
                elif isinstance(tile, dict) and "animal" in tile:
                    animal_name = tile["animal"]
                    animal_cfg = ANIMALS.get(animal_name)
                    if not animal_cfg:
                        continue

                    # Emergency feed
                    if animal_cfg.in_danger_of_escape(tile):
                        candidates.append((0, dist, x, y, "feed"))
                        continue

                    # Harvest produce
                    if animal_cfg.is_harvestable(tile):
                        candidates.append((1, dist, x, y, "harvest"))
                        continue

                    # Feed & Care
                    if animal_cfg.needs_feed(tile):
                        candidates.append((2, dist, x, y, "feed"))
                        continue
                    if animal_cfg.needs_care(tile):
                        candidates.append((3, dist, x, y, "care"))
                        continue
                    if animal_cfg.has_fertilizer(tile):
                        candidates.append((4, dist, x, y, "collect_fertilizer"))
                        continue

                # Case 3: Weed Tile
                elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                    candidates.append((5, dist, x, y, "dig"))

                # Case 4: Empty Tile for Planting
                elif tile is None and has_seeds:
                    candidates.append((4, dist, x, y, "plant"))

        if not candidates:
            return None

        # Sort by: priority asc, distance asc
        candidates.sort(key=lambda item: (item[0], item[1]))
        _, _, best_x, best_y, action_type = candidates[0]
        return (best_x, best_y, action_type)

    # --------------------------------------------------------------------------
    # Unit Action Generation
    # --------------------------------------------------------------------------

    def plan_unit_action(
        self,
        unit_idx: int,
        farm: dict,
        private: dict,
        obs: dict,
        claimed_tiles: Set[Tuple[int, int]],
    ) -> List[Any]:
        """Plans a turn action for the given farmer or hand unit."""
        if unit_idx == 0:
            pos = farm["farmer"]
        else:
            hand_idx = unit_idx - 1
            if hand_idx >= len(farm["hands"]):
                return Actions.pass_action()
            pos = farm["hands"][hand_idx]

        ux, uy = pos[0], pos[1]
        tile = farm["tiles"][uy][ux]
        current_day = obs.get("day", 0)
        current_step = obs.get("step", 0)
        seeds = private.get("seeds", {})
        has_seeds = seeds.get(self.target_crop, 0) > 0 or any(v > 0 for v in seeds.values())

        # If on current tile and actionable task exists
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            crop_cfg = CROPS.get(tile["crop"])
            if crop_cfg:
                # Harvest if optimal or harvestable
                if crop_cfg.is_optimal_harvest_age(tile["planted_day"], current_day) and tile.get("yield_units", 0) > 0:
                    claimed_tiles.add((ux, uy))
                    return Actions.harvest()
                # Water if needed
                if crop_cfg.needs_watering(tile):
                    claimed_tiles.add((ux, uy))
                    return Actions.water()
                # Harvest if ongoing and has units
                if crop_cfg.is_harvestable(tile["planted_day"], current_day, tile.get("yield_units", 0)):
                    claimed_tiles.add((ux, uy))
                    return Actions.harvest()

        elif isinstance(tile, dict) and "animal" in tile:
            animal_cfg = ANIMALS.get(tile["animal"])
            if animal_cfg:
                if animal_cfg.is_harvestable(tile):
                    return Actions.harvest()
                inv = private["inventories"][unit_idx] if unit_idx < len(private["inventories"]) else {}
                if animal_cfg.needs_feed(tile) and inv.get("WHEAT", 0) > 0:
                    return Actions.feed()
                if animal_cfg.needs_care(tile):
                    return Actions.care()
                if animal_cfg.has_fertilizer(tile):
                    return Actions.collect_fertilizer()

        elif isinstance(tile, dict) and tile.get("kind") == "WEED":
            return Actions.dig()

        elif tile is None:
            # Check if we have seeds to plant
            chosen_crop = self.target_crop if seeds.get(self.target_crop, 0) > 0 else next((c for c, v in seeds.items() if v > 0), None)
            if chosen_crop:
                claimed_tiles.add((ux, uy))
                return Actions.plant(chosen_crop)

        # Find best tile to navigate towards
        best_target = self.find_best_tile_for_unit(
            (ux, uy),
            farm,
            private,
            current_day,
            current_step,
            has_seeds,
            claimed_tiles,
        )

        if best_target:
            tx, ty, action_type = best_target
            claimed_tiles.add((tx, ty))
            if (ux, uy) == (tx, ty):
                if action_type == "water":
                    return Actions.water()
                elif action_type == "harvest":
                    return Actions.harvest()
                elif action_type == "plant":
                    chosen_crop = self.target_crop if seeds.get(self.target_crop, 0) > 0 else next((c for c, v in seeds.items() if v > 0), self.target_crop)
                    return Actions.plant(chosen_crop)
                elif action_type == "feed":
                    return Actions.feed()
                elif action_type == "care":
                    return Actions.care()
                elif action_type == "collect_fertilizer":
                    return Actions.collect_fertilizer()
                elif action_type == "dig":
                    return Actions.dig()
            else:
                return self.move_to((ux, uy), (tx, ty))

        return Actions.pass_action()

    # --------------------------------------------------------------------------
    # Market Trading Strategy
    # --------------------------------------------------------------------------

    def plan_market_actions(
        self,
        farm: dict,
        private: dict,
        market: dict,
        current_day: int,
    ) -> List[List[Any]]:
        """Generates market orders (BUY_SEED, SELL, BUY_LAND, HIRE, etc.)."""
        orders: List[List[Any]] = []
        money = farm.get("money", 0)
        unlocked = farm.get("unlocked_quadrants", ["NW"])
        seeds = private.get("seeds", {})
        shed = private.get("shed", {})
        prices = market.get("prices", {})

        # 1. Land Expansion
        if self.auto_expand_land and len(unlocked) < 4:
            next_costs = {1: 1000, 2: 2000, 3: 4000}
            cost = next_costs.get(len(unlocked), 999999)
            if money >= cost:
                orders.append(Actions.buy_land())
                money -= cost

        # 2. Farm Hand Hiring
        if self.auto_hire_hands and farm.get("hires_today", 0) < self.max_hires_per_day:
            hire_cost = 1  # 1st hire costs 1
            if money >= hire_cost:
                orders.append(Actions.hire())
                money -= hire_cost

        # 3. Sell Shed Produce
        if self.auto_sell:
            for item, count in shed.items():
                if count > 0:
                    current_price = prices.get(item, 1)
                    # Look up base price
                    base_price = 25
                    if item in CROPS:
                        base_price = CROPS[item].base_market_price
                    elif item in ANIMALS:
                        base_price = ANIMALS[item].base_market_price

                    if current_price >= base_price * self.min_sell_margin:
                        orders.append(Actions.sell(item, count))

        # 4. Seed Purchasing
        target_crop_cfg = CROPS.get(self.target_crop, CROPS["WHEAT"])
        seed_count = seeds.get(self.target_crop, 0)
        if seed_count < 5 and money >= target_crop_cfg.seed_cost:
            qty = min(5 - seed_count, int(money // target_crop_cfg.seed_cost))
            if qty > 0:
                orders.append(Actions.buy_seed(self.target_crop, qty))
                money -= qty * target_crop_cfg.seed_cost

        return orders[:10]

    # --------------------------------------------------------------------------
    # Master Step / Turn Action Builder
    # --------------------------------------------------------------------------

    def act(self, obs: dict) -> Dict[str, Any]:
        """
        Executes one step of decision making for the given observation,
        returning the complete action dictionary for the agent.
        """
        player = obs["player"]
        me = obs["farms"][player]
        private = obs.get("private", {})
        market = obs.get("market", {})
        current_day = obs.get("day", 0)

        claimed_tiles: Set[Tuple[int, int]] = set()

        # 1. Farmer Action
        farmer_act = self.plan_unit_action(0, me, private, obs, claimed_tiles)

        # 2. Hands Actions
        hands_act = []
        num_hands = len(me.get("hands", []))
        for h_idx in range(num_hands):
            h_act = self.plan_unit_action(h_idx + 1, me, private, obs, claimed_tiles)
            hands_act.append(h_act)

        # 3. Market Actions
        market_act = self.plan_market_actions(me, private, market, current_day)

        return {
            "farmer": farmer_act,
            "hands": hands_act,
            "market": market_act,
        }


# Alias for backward compatibility with potential typos in existing files
ActionContoller = ActionController