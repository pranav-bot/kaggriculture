"""Compound Expansion Engine submission for Kaggriculture.

The planner is intentionally integer and bounded: each observation is reduced to
small priority queues, then a three-day score decides between land, labor,
production, and livestock.  This keeps the agent inexpensive enough for the
720-turn episode limit while retaining rolling-horizon behavior.
"""

from dataclasses import dataclass
import heapq
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from kaggriculture import Actions, ActionController, ANIMALS, CROPS, MARKET_PARAMS, SHOPS, calculate_town_daily_consumption, flatten_board_state, simulate_sell_slippage  # noqa: E402


LIQUIDITY_BUFFER = 500.0
INVESTMENT_CUTOFF_DAY = 26
LIQUIDATION_START_DAY = 28
SEASON_END_DAY = 30
SHOP_UNLOCK_INTERVAL_DAYS = 3
SHOP_SAMPLES = 8


def _farm_from_obs(obs: Mapping[str, Any]) -> Mapping[str, Any]:
    farms = obs.get("farms", [])
    player = int(obs.get("player", 0))
    return farms[player] if player < len(farms) else {}


def parse_board_state(obs: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten the farm grid into compact, directly actionable priority lists."""
    farm = _farm_from_obs(obs)
    board = flatten_board_state(
        farm.get("tiles", []),
        current_day=int(obs.get("day", 0)),
        current_step=int(obs.get("step", 0)),
    )
    premium = [
        {"pos": (c.x, c.y), "item": c.crop, "quantity": c.yield_units}
        for c in board.harvestable_crops
        if c.config.base_market_price > 100
    ]
    return {
        "board": board,
        "thirsty_crops": [(c.x, c.y) for c in board.thirsty_crops],
        "danger_crops": [(c.x, c.y) for c in board.danger_crops],
        "harvestable_crops": [(c.x, c.y) for c in board.harvestable_crops],
        "harvestable_premium_goods": premium,
        "available_fertilizer": sum(
            a.tile.get("fertilizer_available", False) for a in board.fertilizer_animals
        ),
        "hungry_animals": [(a.x, a.y) for a in board.hungry_animals],
        "careable_animals": [(a.x, a.y) for a in board.careable_animals],
        "weeds": list(board.weeds),
        "vacant_tiles": list(board.vacant_unlocked),
        "empty_structures": list(board.empty_structures),
        "crop_counts": dict(board.crop_counts),
        "animal_counts": dict(board.animal_counts),
    }


def calculate_labor_budget(current_money: float, hires_today: int) -> int:
    """Return a Fibonacci-scaled hiring count after preserving safe liquidity."""
    reserve = max(LIQUIDITY_BUFFER, float(current_money) * 0.20)
    usable = max(0.0, float(current_money) - reserve)
    spent = 0
    hires = 0
    for offset in range(8):
        cost = Actions.hire_cost(int(hires_today) + offset)
        if spent + cost > usable:
            break
        spent += cost
        hires += 1
    return hires


def _distance(a: Sequence[int], b: Sequence[int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def dispatch_workers(
    tasks: Iterable[Mapping[str, Any]],
    farmer: Sequence[int],
    hands: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    """Assign queued spatial tasks to nearest hands; reserve the farmer for shed work."""
    queue: List[Tuple[float, int, Mapping[str, Any]]] = []
    for index, task in enumerate(tasks):
        position = task.get("pos", (0, 0))
        priority = float(task.get("priority", 100))
        heapq.heappush(queue, (priority, index, task))

    assignments: Dict[str, Any] = {"farmer": None, "hands": [None] * len(hands)}
    used: set[int] = set()
    while queue:
        _, _, task = heapq.heappop(queue)
        if task.get("worker") == "farmer":
            current = assignments["farmer"]
            if current is None or float(task.get("priority", 100)) < float(current.get("priority", 100)):
                assignments["farmer"] = task
            continue
        candidates = [
            (idx, _distance(pos, task.get("pos", (0, 0))))
            for idx, pos in enumerate(hands)
            if idx not in used
        ]
        if candidates:
            idx, _ = min(candidates, key=lambda item: (item[1], item[0]))
            assignments["hands"][idx] = task
            used.add(idx)
        elif assignments["farmer"] is None:
            assignments["farmer"] = task
    return assignments


def optimize_sales_queue(
    inventory: Mapping[str, int],
    market_prices: Mapping[str, Any],
    town_shops: Sequence[str],
) -> List[List[Any]]:
    """Create small sequential SELL orders, protecting value on curved markets."""
    prices = market_prices.get("prices", market_prices)
    demand = calculate_town_daily_consumption(list(town_shops))
    scored: List[Tuple[float, str, int]] = []
    for item, raw_count in inventory.items():
        count = int(raw_count or 0)
        if count <= 0 or item.upper() in ANIMALS:
            continue
        item = item.upper()
        price = float(prices.get(item, MARKET_PARAMS.get(item, {}).get("base", 0)))
        demand_weight = 1.0 + min(1.0, demand.get(item, 0) / 20.0)
        scored.append((price * demand_weight, item, count))

    orders: List[List[Any]] = []
    for _, item, count in sorted(scored, reverse=True):
        # Premium stacks are split more finely; common goods use efficient groups.
        batch = 1 if MARKET_PARAMS.get(item, {}).get("base", 0) >= 100 else 4
        remaining = count
        while remaining > 0 and len(orders) < Actions.MAX_MARKET_ORDERS_PER_TURN:
            quantity = min(batch, remaining)
            orders.append(Actions.sell(item, quantity))
            remaining -= quantity
    return orders


def forecast_shop_demand(
    day: int,
    unlocked_shops: Sequence[str],
    horizon_days: int = 30,
) -> Dict[str, Any]:
    """Forecast expected commodity demand under uniform three-day shop unlocks."""
    products = tuple(MARKET_PARAMS)
    expected = {item: 0.0 for item in products}
    transition = {shop: {next_shop: 0.0 for next_shop in SHOPS} for shop in SHOPS}
    for shop in unlocked_shops:
        for item, quantity in calculate_town_daily_consumption([shop]).items():
            expected[item] = expected.get(item, 0.0) + quantity

    remaining_unlocks = max(0, min(SHOP_SAMPLES - len(unlocked_shops),
                                   (horizon_days - int(day)) // SHOP_UNLOCK_INTERVAL_DAYS))
    probability = 1.0 / max(1, len(SHOPS))
    for shop in SHOPS:
        for next_shop in SHOPS:
            transition[shop][next_shop] = probability
    for _ in range(remaining_unlocks):
        for shop, products_for_shop in SHOPS.items():
            for item in products_for_shop:
                expected[item] = expected.get(item, 0.0) + (
                    6.0 * (2.0 if len(products_for_shop) == 1 else 1.0)
                    * probability
                )
    return {"expected_demand": expected, "transition_matrix": transition}


def evaluate_production_enpv(
    day: int,
    market_prices: Mapping[str, Any],
    unlocked_shops: Sequence[str],
    horizon_turns: int = 720,
    fertilizer_price: float = 100.0,
) -> Dict[str, float]:
    """Score crops and livestock over the remaining 720-turn horizon."""
    prices = market_prices.get("prices", market_prices)
    horizon_days = max(0.0, (horizon_turns - int(day) * 24) / 24.0)
    demand = forecast_shop_demand(day, unlocked_shops, 30)["expected_demand"]
    scores: Dict[str, float] = {}
    for name, config in CROPS.items():
        days = min(horizon_days, float(config.time_to_max_yield))
        if days < config.time_to_first_yield:
            scores[name] = -float(config.seed_cost)
            continue
        price = float(prices.get(name, config.base_market_price))
        demand_factor = 1.0 + min(0.5, demand.get(name, 0.0) / 500.0)
        cycles = max(1.0, horizon_days / max(1, config.time_to_max_yield))
        yield_per_cycle = config.max_yield if config.ongoing else config.unfertilized_max_yield
        scores[name] = (
            cycles * yield_per_cycle * price * demand_factor
            - cycles * config.seed_cost
            - cycles * config.time_to_max_yield * config.action_cost
        )
    for name, config in ANIMALS.items():
        price = float(prices.get(config.product, config.base_market_price))
        production_days = max(0.0, horizon_days - config.first_yield_day)
        units = production_days / max(1, config.interval)
        feed_cost = production_days * float(prices.get("WHEAT", 25))
        scores[name] = units * price - feed_cost - config.cost
    return scores


def evaluate_expansion_roi(
    day: int, money: float, unlocked_quadrants: Sequence[str]
) -> Dict[str, Any]:
    """Compare the next land purchase with a three-day cow/sheep investment."""
    cost = Actions.land_cost(list(unlocked_quadrants))
    if cost is None:
        return {
            "should_expand": False,
            "next_quadrant": None,
            "cost": None,
            "expansion_roi": 0.0,
            "livestock_roi": 0.0,
        }
    days = max(0, min(3, INVESTMENT_CUTOFF_DAY - int(day)))
    prices = {"MILK": 160, "WOOL": 200}
    livestock_roi = max(
        (max(0, days - ANIMALS[a].first_yield_day) / max(1, ANIMALS[a].interval))
        * prices[ANIMALS[a].product]
        - ANIMALS[a].cost
        for a in ("COW", "SHEEP")
    )
    # A newly opened quadrant adds 25 planting slots. Use the best fast crop.
    best_crop = max(
        CROPS.values(),
        key=lambda c: c.base_market_price * c.yield_per_tile_per_day - c.seed_cost / 3
    )
    expansion_profit = 25 * (
        best_crop.base_market_price * best_crop.yield_per_tile_per_day * days / 3
        - best_crop.seed_cost
    )
    expansion_roi = (expansion_profit - cost) / max(1, cost)
    livestock_roi = livestock_roi / 500.0
    return {
        "should_expand": (
            int(day) <= INVESTMENT_CUTOFF_DAY
            and float(money) >= cost + LIQUIDITY_BUFFER
            and expansion_roi > livestock_roi
        ),
        "next_quadrant": Actions.next_quadrant(list(unlocked_quadrants)),
        "cost": cost,
        "expansion_roi": expansion_roi,
        "livestock_roi": livestock_roi,
    }


@dataclass
class _Task:
    priority: int
    pos: Tuple[int, int]
    action: str
    worker: str | None = None


class CompoundExpansionController(ActionController):
    """State-tracking controller with bounded rolling-horizon investment choices."""

    def __init__(self) -> None:
        super().__init__(
            target_crop="WHEAT",
            auto_water=True,
            auto_harvest=True,
            auto_fertilize=True,
            auto_feed_animals=True,
            auto_care_animals=True,
            auto_collect_fertilizer=True,
            auto_dig_weeds=True,
            auto_sell=True,
            auto_expand_land=False,
            auto_hire_hands=False,
            max_hires_per_day=3,
            min_sell_margin=0.0,
        )

    @staticmethod
    def _direction(src: Sequence[int], target: Sequence[int]) -> List[str]:
        if tuple(src) == tuple(target):
            return Actions.pass_action()
        dx, dy = int(target[0]) - int(src[0]), int(target[1]) - int(src[1])
        return Actions.move("EAST" if abs(dx) >= abs(dy) and dx > 0 else
                            "WEST" if abs(dx) >= abs(dy) else
                            "SOUTH" if dy > 0 else "NORTH")

    def _assigned_action(
        self, assignment: Mapping[str, Any] | None, position: Sequence[int]
    ) -> List[Any] | None:
        if not assignment:
            return None
        target = assignment.get("pos", position)
        if tuple(position) != tuple(target):
            return self._direction(position, target)
        action = assignment.get("action")
        if isinstance(action, str) and action.startswith("plant:"):
            return Actions.plant(action.split(":", 1)[1])
        if isinstance(action, str) and action.startswith("place:"):
            return Actions.place(action.split(":", 1)[1], 1)
        return {
            "water": Actions.water,
            "harvest": Actions.harvest,
            "fertilize": Actions.fertilize,
            "feed": Actions.feed,
            "care": Actions.care,
            "collect_fertilizer": Actions.collect_fertilizer,
            "dig": Actions.dig,
            "build_coop": Actions.build_coop,
            "build_pasture": Actions.build_pasture,
        }.get(action, Actions.pass_action)()

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        day = int(obs.get("day", 0))
        step = int(obs.get("step", day * 24))
        farm = _farm_from_obs(obs)
        private = obs.get("private", {})
        state = parse_board_state(obs)
        money = float(farm.get("money", 0))
        shed = private.get("shed", {})
        hands = farm.get("hands", [])
        farmer = farm.get("farmer", [4, 4])
        market = obs.get("market", {})
        shops = obs.get("town", {}).get("unlocked_shops", [])
        scores = evaluate_production_enpv(day, market, shops)
        crop_scores = {name: score for name, score in scores.items() if name in CROPS}
        self.target_crop = max(crop_scores, key=crop_scores.get)

        tasks: List[Mapping[str, Any]] = []
        for pos in state["danger_crops"]:
            tasks.append({"priority": 0, "pos": pos, "action": "water"})
        for pos in state["hungry_animals"]:
            tasks.append({"priority": 1, "pos": pos, "action": "feed"})
        for pos in state["harvestable_crops"]:
            tasks.append({"priority": 2, "pos": pos, "action": "harvest"})
        for pos in state["thirsty_crops"]:
            tasks.append({"priority": 3, "pos": pos, "action": "water"})
        for pos in state["careable_animals"]:
            tasks.append({"priority": 4, "pos": pos, "action": "care"})
        for pos in state["weeds"]:
            tasks.append({"priority": 5, "pos": pos, "action": "dig"})

        for x, y, kind in state["empty_structures"]:
            for animal in ("SHEEP", "COW", "GOOSE"):
                if (
                    kind == ANIMALS[animal].structure
                    and int(shed.get(animal, 0)) > 0
                ):
                    tasks.append({
                        "priority": 1,
                        "pos": (x, y),
                        "action": f"place:{animal}",
                        "worker": "farmer",
                    })
                    break

        fertilizer = sum(
            int(inv.get("FERTILIZER", 0))
            for inv in private.get("inventories", [])
            if isinstance(inv, dict)
        ) + int(shed.get("FERTILIZER", 0))
        prices = market.get("prices", {})
        fertilizer_value = max(
            0.0,
            (CROPS["MELON"].fertilized_max_yield - CROPS["MELON"].unfertilized_max_yield)
            * float(prices.get("MELON", CROPS["MELON"].base_market_price))
            - float(prices.get("FERTILIZER", 100)),
        )
        if fertilizer and fertilizer_value > 0:
            for crop in state["board"].all_crops:
                if crop.crop in ("MELON", "STRAWBERRY") and crop.fertilized_until_day < day:
                    tasks.append({
                        "priority": 2,
                        "pos": (crop.x, crop.y),
                        "action": "fertilize",
                    })

        seeds = dict(private.get("seeds", {}))
        for pos in state["vacant_tiles"]:
            if seeds.get(self.target_crop, 0) <= 0:
                break
            tasks.append({
                "priority": 6,
                "pos": pos,
                "action": f"plant:{self.target_crop}",
            })
            seeds[self.target_crop] -= 1

        if day <= INVESTMENT_CUTOFF_DAY and money >= 900:
            animal = max(
                ("COW", "SHEEP"),
                key=lambda name: scores.get(name, -float("inf")),
            )
            self.target_animal = animal
            compatible = [
                (x, y)
                for x, y, kind in state["empty_structures"]
                if kind == ANIMALS[animal].structure
            ]
            occupied = state["animal_counts"].get(animal, 0)
            if not compatible and not any(
                kind == ANIMALS[animal].structure for _, _, kind in state["empty_structures"]
            ):
                if state["vacant_tiles"]:
                    tasks.append({
                        "priority": 6,
                        "pos": state["vacant_tiles"][0],
                        "action": "build_pasture",
                        "worker": "farmer",
                    })

        assignments = dispatch_workers(tasks, farmer, hands)
        farmer_assignment = assignments["farmer"]
        farmer_inventory = private.get("inventories", [{}])[0] if private.get("inventories") else {}
        if sum(farmer_inventory.values()) > 0:
            shed_tile = min(Actions.shed_access_tiles(), key=lambda p: _distance(farmer, p))
            if tuple(farmer) == tuple(shed_tile):
                farmer_action = Actions.drop()
            else:
                farmer_action = self._direction(farmer, shed_tile)
        else:
            farmer_action = self._assigned_action(farmer_assignment, farmer)
            if farmer_action is None:
                farmer_action = super().plan_unit_action(
                    0, farm, private, dict(private.get("seeds", {})), obs, set()
                )

        hand_actions = []
        for idx, pos in enumerate(hands):
            assigned = self._assigned_action(assignments["hands"][idx], pos)
            hand_actions.append(assigned or super().plan_unit_action(
                idx + 1, farm, private, dict(private.get("seeds", {})), obs, set()
            ))

        # Fixed-size rolling investment policy; all investment is disabled after day 26.
        market_orders: List[List[Any]] = []
        sell_window = step % 4 == 1
        if day >= LIQUIDATION_START_DAY:
            market_orders = optimize_sales_queue(
                shed, obs.get("market", {}), obs.get("town", {}).get("unlocked_shops", [])
            ) if sell_window else []
        elif day <= INVESTMENT_CUTOFF_DAY:
            roi = evaluate_expansion_roi(day, money, farm.get("unlocked_quadrants", ["NW"]))
            spendable = money - LIQUIDITY_BUFFER
            if roi["should_expand"] and spendable >= float(roi["cost"]):
                market_orders.append(Actions.buy_land())
                spendable -= float(roi["cost"])

            phase_cap = 3 if day <= 6 else 5 if day <= 18 else 6
            phase_buffer = 500 if day <= 6 else 1000 if day <= 18 else 2000
            labor_money = max(0.0, money - max(0, phase_buffer - LIQUIDITY_BUFFER))
            hires = min(
                calculate_labor_budget(labor_money, int(farm.get("hires_today", 0))),
                max(0, phase_cap - int(farm.get("hires_today", 0))),
            )
            market_orders.extend(Actions.hire() for _ in range(hires))
            # Keep a seed runway, switching to melons once enough land/capital exists.
            seed_count = int(private.get("seeds", {}).get(self.target_crop, 0))
            if seed_count < 5:
                cfg = CROPS[self.target_crop]
                quantity = min(5 - seed_count, int(max(0, spendable) // cfg.seed_cost))
                if quantity:
                    market_orders.append(Actions.buy_seed(self.target_crop, quantity))
            if (
                day >= 10
                and self.target_animal
                and state["animal_counts"].get(self.target_animal, 0) == 0
                and int(shed.get(self.target_animal, 0)) == 0
                and money >= ANIMALS[self.target_animal].cost + LIQUIDITY_BUFFER
            ):
                market_orders.append(Actions.buy_animal(self.target_animal, 1))
        else:
            # Days 27 is a transition day: maintain crops, but do not buy assets.
            market_orders = optimize_sales_queue(
                shed, obs.get("market", {}), obs.get("town", {}).get("unlocked_shops", [])
            ) if sell_window else []

        if day < LIQUIDATION_START_DAY and sell_window:
            market_orders.extend(
                optimize_sales_queue(
                    shed, obs.get("market", {}), obs.get("town", {}).get("unlocked_shops", [])
                )
            )

        return {"farmer": farmer_action, "hands": hand_actions,
                "market": market_orders[:Actions.MAX_MARKET_ORDERS_PER_TURN]}


controller = CompoundExpansionController()


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Kaggle entrypoint."""
    return controller.act(obs)
