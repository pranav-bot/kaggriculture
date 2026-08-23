"""
Solver Integration & Action Discretization Helpers for Kaggriculture.

Provides:
- Constraint Formulator: Translates game rules and capacity limits into continuous mathematical constraints for numerical solvers (IPOPT, Scipy optimize, Linear Programming).
- Action Discretizer: Maps continuous target vectors into ordered, legal discrete game commands.
"""
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from kaggriculture.actions.actions import Actions, CROPS, ANIMALS
from kaggriculture.env.items import (
    SHED_CAPACITY,
    TURNS_PER_DAY,
    Plants,
    Animals,
    Products,
)


# ==============================================================================
# 1. Constraint Formulator
# ==============================================================================

@dataclass
class OptimizationProblem:
    """
    Mathematical programming formulation representing production constraints over a horizon.
    Ready for standard solvers: Scipy linprog, IPOPT, CVXPY, or Pyomo.
    """
    variable_names: List[str]
    c_objective: List[float]                  # Maximize: -c_objective for linprog minimization
    A_ub: List[List[float]]                   # Linear inequality matrix A_ub * x <= b_ub
    b_ub: List[float]                         # Inequality RHS bounds
    bounds: List[Tuple[float, Optional[float]]] # Variable lower/upper bounds
    constraint_labels: List[str]

    def to_scipy_args(self) -> Dict[str, Any]:
        """Formats the problem dictionary for scipy.optimize.linprog."""
        return {
            "c": [-val for val in self.c_objective],  # negate for maximization
            "A_ub": self.A_ub,
            "b_ub": self.b_ub,
            "bounds": self.bounds,
            "method": "highs",
        }


def formulate_production_constraints(
    available_money: float,
    unlocked_tiles_count: int = 25,
    current_shed_occupancy: int = 0,
    shed_capacity: int = SHED_CAPACITY,
    horizon_days: int = 4,
    reserve_cash: float = 100.0,
    market_prices: Optional[Dict[str, int]] = None,
    hands_hired_per_day: int = 0,
) -> OptimizationProblem:
    """
    Formulates a rolling-horizon production allocation problem:
    
    Decision variables:
      x0: Wheat plants
      x1: Carrot plants
      x2: Tomato plants
      x3: Strawberry plants
      x4: Melon plants
      x5: Goose livestock
      x6: Cow livestock
      x7: Sheep livestock
    """
    crops_order = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
    animals_order = ["GOOSE", "COW", "SHEEP"]
    var_names = crops_order + animals_order
    num_vars = len(var_names)

    prices = market_prices or {
        "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
        "EGG": 50, "MILK": 160, "WOOL": 200,
    }

    # 1. Objective: Expected Net Profit over horizon
    c_obj = []
    for var in crops_order:
        cfg = CROPS[var]
        p = prices.get(var, cfg.base_market_price)
        # Expected yield over horizon
        effective_days = min(horizon_days, cfg.time_to_max_yield)
        expected_yield = cfg.yield_per_tile_per_day * effective_days
        net_profit = expected_yield * p - cfg.seed_cost
        c_obj.append(net_profit)

    for var in animals_order:
        cfg = ANIMALS[var]
        p = prices.get(cfg.product, cfg.base_market_price)
        effective_days = max(0, horizon_days - cfg.first_yield_day)
        expected_yield = (effective_days / cfg.interval) if cfg.interval > 0 else 0
        feed_cost = effective_days * prices.get("WHEAT", 25)
        net_profit = expected_yield * p - cfg.cost - feed_cost
        c_obj.append(net_profit)

    A_ub = []
    b_ub = []
    labels = []

    # 2. Land Capacity Constraint: sum(tiles) <= unlocked_tiles
    A_land = [1.0] * num_vars
    A_ub.append(A_land)
    b_ub.append(float(unlocked_tiles_count))
    labels.append("Land Capacity (tiles)")

    # 3. Capital / Liquidity Budget Constraint: sum(setup_costs) <= money - reserve
    A_budget = []
    for var in crops_order:
        A_budget.append(float(CROPS[var].seed_cost))
    for var in animals_order:
        A_budget.append(float(ANIMALS[var].cost))
    
    usable_money = max(0.0, available_money - reserve_cash)
    A_ub.append(A_budget)
    b_ub.append(usable_money)
    labels.append("Capital Budget ($)")

    # 4. Shed Storage Constraint (for harvestable produce + animals stored)
    A_shed = []
    for var in crops_order:
        cfg = CROPS[var]
        A_shed.append(float(cfg.unfertilized_max_yield))
    for var in animals_order:
        cfg = ANIMALS[var]
        A_shed.append(float(cfg.max_held))

    remaining_shed = max(0.0, float(shed_capacity - current_shed_occupancy))
    A_ub.append(A_shed)
    b_ub.append(remaining_shed)
    labels.append("Shed Capacity Limit (items)")

    # 5. Action Economy Constraint: total actions <= (1 + hands) * 24 * horizon_days
    # Each crop requires ~1 plant + (days * 1 water) + 1 harvest
    # Each animal requires ~1 build + 1 place + (days * 1 feed) + 1 harvest
    A_actions = []
    for var in crops_order:
        A_actions.append(float(2 + horizon_days))
    for var in animals_order:
        A_actions.append(float(2 + horizon_days))

    total_action_budget = float((1 + hands_hired_per_day) * TURNS_PER_DAY * horizon_days)
    A_ub.append(A_actions)
    b_ub.append(total_action_budget)
    labels.append("Action Economy Budget (turns)")

    # 6. Variable Bounds: [0, unlocked_tiles]
    bounds = [(0.0, float(unlocked_tiles_count)) for _ in range(num_vars)]

    return OptimizationProblem(
        variable_names=var_names,
        c_objective=c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        constraint_labels=labels,
    )


# ==============================================================================
# 2. Action Discretizer
# ==============================================================================

@dataclass
class DiscretePlan:
    seed_purchases: Dict[str, int] = field(default_factory=dict)
    animal_purchases: Dict[str, int] = field(default_factory=dict)
    plant_targets: Dict[str, int] = field(default_factory=dict)
    sell_orders: Dict[str, int] = field(default_factory=dict)
    market_action_queue: List[List[Any]] = field(default_factory=list)


def discretize_continuous_solution(
    solution_targets: Dict[str, float],
    current_seeds: Dict[str, int],
    current_shed: Dict[str, int],
    available_money: float,
    vacant_tiles: int,
    rounding_mode: str = "stochastic",
) -> DiscretePlan:
    """
    Converts continuous solver targets (e.g. {'WHEAT': 14.7, 'MELON': 3.2, 'SELL_WHEAT': 20.0})
    into an ordered, legal set of discrete integer game commands.
    """
    plan = DiscretePlan()
    money_left = available_money
    planted_count = 0

    # 1. Discretize planting/seed targets
    for crop_name, float_val in solution_targets.items():
        crop_upper = crop_name.upper()
        if crop_upper not in CROPS:
            continue

        if rounding_mode == "stochastic":
            int_target = math.floor(float_val) + (1 if (float_val % 1.0) > 0.5 else 0)
        else:
            int_target = int(round(float_val))

        int_target = max(0, min(vacant_tiles - planted_count, int_target))
        if int_target <= 0:
            continue

        plan.plant_targets[crop_upper] = int_target
        planted_count += int_target

        # Determine seed deficit
        have_seeds = current_seeds.get(crop_upper, 0)
        deficit = max(0, int_target - have_seeds)
        if deficit > 0:
            cost = deficit * CROPS[crop_upper].seed_cost
            if money_left >= cost:
                plan.seed_purchases[crop_upper] = deficit
                plan.market_action_queue.append(Actions.buy_seed(crop_upper, deficit))
                money_left -= cost

    # 2. Discretize animal targets
    for anim_name, float_val in solution_targets.items():
        anim_upper = anim_name.upper()
        if anim_upper not in ANIMALS:
            continue

        int_target = int(round(float_val))
        if int_target <= 0:
            continue

        have_in_shed = current_shed.get(anim_upper, 0)
        deficit = max(0, int_target - have_in_shed)
        if deficit > 0:
            cost = deficit * ANIMALS[anim_upper].cost
            if money_left >= cost:
                plan.animal_purchases[anim_upper] = deficit
                plan.market_action_queue.append(Actions.buy_animal(anim_upper, deficit))
                money_left -= cost

    # 3. Discretize sell targets (e.g. 'SELL_WHEAT': 25.0)
    for key, float_val in solution_targets.items():
        if key.startswith("SELL_"):
            item = key.replace("SELL_", "").upper()
            qty = min(current_shed.get(item, 0), int(round(float_val)))
            if qty > 0:
                plan.sell_orders[item] = qty
                plan.market_action_queue.append(Actions.sell(item, qty))

    return plan
