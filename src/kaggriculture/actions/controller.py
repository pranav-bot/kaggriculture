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
    Coordinates farmer movements, hired hands, field maintenance, livestock, and market trading.
    """

    def __init__(
        self,
        target_crop: Union[str, Plants] = Plants.WHEAT,
        target_animal: Optional[Union[str, Animals]] = None,
        auto_water: bool = True,
        auto_harvest: bool = True,
        auto_fertilize: bool = False,
        auto_feed_animals: bool = True,
        auto_care_animals: bool = True,
        auto_collect_fertilizer: bool = True,
        auto_dig_weeds: bool = True,
        auto_sell: bool = True,
        auto_expand_land: bool = True,
        auto_hire_hands: bool = False,
        max_hires_per_day: int = 1,
        min_sell_margin: float = 0.8,
        board_size: int = 10,
    ):
        self.target_crop = str(target_crop).upper()
        self.target_animal = str(target_animal).upper() if target_animal else None
        self.auto_water = auto_water
        self.auto_harvest = auto_harvest
        self.auto_fertilize = auto_fertilize
        self.auto_feed_animals = auto_feed_animals
        self.auto_care_animals = auto_care_animals
        self.auto_collect_fertilizer = auto_collect_fertilizer
        self.auto_dig_weeds = auto_dig_weeds
        self.auto_sell = auto_sell
        self.auto_expand_land = auto_expand_land
        self.auto_hire_hands = auto_hire_hands
        self.max_hires_per_day = max_hires_per_day
        self.min_sell_margin = min_sell_margin
        self.board_size = board_size

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
        return Actions.is_shed_adjacent(pos, board_size)

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
        unit_inv: dict,
        farm: dict,
        private: dict,
        available_seeds: Dict[str, int],
        current_day: int,
        current_step: int,
        claimed_tiles: Set[Tuple[int, int]],
    ) -> Optional[Tuple[int, int, str]]:
        """
        Finds the highest priority tile and task for a given farmer/hand unit.
        Prioritization:
          0. Emergency watering (in danger of becoming weed) / Emergency feeding (danger of escape)
          1. Optimal harvesting (reached max yield / ready)
          2. Regular watering (needs water today)
          3. Livestock care, feeding, fertilizer collection
          4. Planting empty tiles (respecting available_seeds quota)
          5. Clearing weeds
        """
        ux, uy = unit_pos
        board_size = len(farm["tiles"])
        candidates = []
        has_seeds = available_seeds.get(self.target_crop, 0) > 0 or any(v > 0 for v in available_seeds.values())

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

                    # 1a. Critical water danger (missed yesterday, not watered today)
                    if crop_cfg.in_danger_of_weed(tile):
                        candidates.append((0, dist, x, y, "water"))
                        continue

                    # 1b. Ripe / Optimal Harvest
                    if crop_cfg.is_optimal_harvest_age(tile["planted_day"], current_day) and tile.get("yield_units", 0) > 0:
                        candidates.append((1, dist, x, y, "harvest"))
                        continue

                    # 1c. Regular watering
                    if self.auto_water and crop_cfg.needs_watering(tile):
                        candidates.append((2, dist, x, y, "water"))
                        continue

                    # 1d. Fertilizing (if unit carries fertilizer)
                    if self.auto_fertilize and unit_inv.get("FERTILIZER", 0) > 0 and tile.get("fertilized_until_day", -1) < current_day:
                        candidates.append((2, dist, x, y, "fertilize"))
                        continue

                    # 1e. Non-optimal but harvestable
                    if self.auto_harvest and crop_cfg.is_harvestable(tile["planted_day"], current_day, tile.get("yield_units", 0)):
                        candidates.append((3, dist, x, y, "harvest"))
                        continue

                # Case 2: Animal Tile
                elif isinstance(tile, dict) and "animal" in tile:
                    animal_name = tile["animal"]
                    animal_cfg = ANIMALS.get(animal_name)
                    if not animal_cfg:
                        continue

                    # Emergency feed (missed yesterday, not fed today)
                    if animal_cfg.in_danger_of_escape(tile):
                        candidates.append((0, dist, x, y, "feed"))
                        continue

                    # Harvest produce (eggs, milk, wool)
                    if self.auto_harvest and animal_cfg.is_harvestable(tile):
                        candidates.append((1, dist, x, y, "harvest"))
                        continue

                    # Feed & Care
                    if self.auto_feed_animals and animal_cfg.needs_feed(tile):
                        candidates.append((2, dist, x, y, "feed"))
                        continue
                    if self.auto_care_animals and animal_cfg.needs_care(tile):
                        candidates.append((3, dist, x, y, "care"))
                        continue
                    if self.auto_collect_fertilizer and animal_cfg.has_fertilizer(tile):
                        candidates.append((4, dist, x, y, "collect_fertilizer"))
                        continue

                # Case 3: Empty structure waiting for animal placement
                elif isinstance(tile, dict) and tile.get("kind") in (Structures.COOP, Structures.PASTURE) and "animal" not in tile:
                    for anim in ("GOOSE", "COW", "SHEEP"):
                        if unit_inv.get(anim, 0) > 0 and ANIMALS[anim].structure == tile.get("kind"):
                            candidates.append((2, dist, x, y, f"place_{anim}"))
                            break

                # Case 4: Weed Tile
                elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                    if self.auto_dig_weeds:
                        candidates.append((5, dist, x, y, "dig"))

                # Case 5: Empty Tile for Planting
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
        available_seeds: Dict[str, int],
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
        inventories = private.get("inventories", [])
        inv = inventories[unit_idx] if unit_idx < len(inventories) else {}

        # 1. Action on current tile if applicable
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
                # Fertilize if carrying fertilizer and not active
                if self.auto_fertilize and inv.get("FERTILIZER", 0) > 0 and tile.get("fertilized_until_day", -1) < current_day:
                    claimed_tiles.add((ux, uy))
                    return Actions.fertilize()
                # Harvest if harvestable
                if crop_cfg.is_harvestable(tile["planted_day"], current_day, tile.get("yield_units", 0)):
                    claimed_tiles.add((ux, uy))
                    return Actions.harvest()

        elif isinstance(tile, dict) and "animal" in tile:
            animal_cfg = ANIMALS.get(tile["animal"])
            if animal_cfg:
                if animal_cfg.is_harvestable(tile):
                    claimed_tiles.add((ux, uy))
                    return Actions.harvest()
                if animal_cfg.needs_feed(tile) and inv.get("WHEAT", 0) > 0:
                    claimed_tiles.add((ux, uy))
                    return Actions.feed()
                if animal_cfg.needs_care(tile):
                    claimed_tiles.add((ux, uy))
                    return Actions.care()
                if animal_cfg.has_fertilizer(tile):
                    claimed_tiles.add((ux, uy))
                    return Actions.collect_fertilizer()

        elif isinstance(tile, dict) and tile.get("kind") in (Structures.COOP, Structures.PASTURE) and "animal" not in tile:
            for anim in ("GOOSE", "COW", "SHEEP"):
                if inv.get(anim, 0) > 0 and ANIMALS[anim].structure == tile.get("kind"):
                    return Actions.place(anim, 1)

        elif isinstance(tile, dict) and tile.get("kind") == "WEED":
            claimed_tiles.add((ux, uy))
            return Actions.dig()

        elif tile is None:
            # Check if we have seeds to plant
            chosen_crop = self.target_crop if available_seeds.get(self.target_crop, 0) > 0 else next((c for c, v in available_seeds.items() if v > 0), None)
            if chosen_crop and available_seeds.get(chosen_crop, 0) > 0:
                available_seeds[chosen_crop] -= 1
                claimed_tiles.add((ux, uy))
                return Actions.plant(chosen_crop)

        # 2. Find best target tile to navigate towards
        best_target = self.find_best_tile_for_unit(
            (ux, uy),
            inv,
            farm,
            private,
            available_seeds,
            current_day,
            current_step,
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
                elif action_type == "fertilize":
                    return Actions.fertilize()
                elif action_type == "plant":
                    chosen_crop = self.target_crop if available_seeds.get(self.target_crop, 0) > 0 else next((c for c, v in available_seeds.items() if v > 0), None)
                    if chosen_crop and available_seeds.get(chosen_crop, 0) > 0:
                        available_seeds[chosen_crop] -= 1
                        return Actions.plant(chosen_crop)
                elif action_type == "feed":
                    return Actions.feed()
                elif action_type == "care":
                    return Actions.care()
                elif action_type == "collect_fertilizer":
                    return Actions.collect_fertilizer()
                elif action_type.startswith("place_"):
                    anim_name = action_type.replace("place_", "")
                    return Actions.place(anim_name, 1)
                elif action_type == "dig":
                    return Actions.dig()
            else:
                return self.move_to((ux, uy), (tx, ty))

        # 3. Shed drop if unit carries harvested goods/fertilizer and is near shed
        if sum(inv.values()) > 0 and Actions.is_shed_adjacent((ux, uy), self.board_size):
            return Actions.drop()

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
                if count > 0 and item not in ANIMALS:
                    current_price = prices.get(item, 1)
                    base_price = 25
                    if item in CROPS:
                        base_price = CROPS[item].base_market_price
                    elif item in Products:
                        if item == "EGG": base_price = 50
                        elif item == "MILK": base_price = 160
                        elif item == "WOOL": base_price = 200
                        elif item == "FERTILIZER": base_price = 100

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

        # 5. Animal Purchasing (if configured)
        if self.target_animal and self.target_animal in ANIMALS:
            anim_cfg = ANIMALS[self.target_animal]
            if shed.get(self.target_animal, 0) == 0 and money >= anim_cfg.cost:
                orders.append(Actions.buy_animal(self.target_animal, 1))
                money -= anim_cfg.cost

        return orders[:10]

    # --------------------------------------------------------------------------
    # Master Step / Turn Action Builder
    # --------------------------------------------------------------------------

    def act(self, obs: dict) -> Dict[str, Any]:
        """
        Executes one step of decision making for the given observation,
        returning the complete action dictionary for the agent.
        Ensures multi-unit seed safety to avoid simultaneous over-planting penalties.
        """
        player = obs["player"]
        me = obs["farms"][player]
        private = obs.get("private", {})
        market = obs.get("market", {})
        current_day = obs.get("day", 0)

        # Virtual seed tracker for this turn to prevent simultaneous over-planting
        available_seeds = dict(private.get("seeds", {}))
        claimed_tiles: Set[Tuple[int, int]] = set()

        # 1. Farmer Action
        farmer_act = self.plan_unit_action(0, me, private, available_seeds, obs, claimed_tiles)

        # 2. Hands Actions
        hands_act = []
        num_hands = len(me.get("hands", []))
        for h_idx in range(num_hands):
            h_act = self.plan_unit_action(h_idx + 1, me, private, available_seeds, obs, claimed_tiles)
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