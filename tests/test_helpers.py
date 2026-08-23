import pytest
from kaggriculture.helpers import (
    flatten_board_state,
    FlattenedBoard,
    cumulative_hire_cost,
    max_affordable_hires,
    forecast_crop_yield_trajectory,
    predict_upcoming_consumption_ticks,
    find_next_consumption_window,
    estimate_market_drain_over_horizon,
    simulate_sell_slippage,
    simulate_buy_slippage,
    formulate_production_constraints,
    discretize_continuous_solution,
    analyze_opponent_farm,
)


# ==============================================================================
# 1. State & Resource Tracking Tests
# ==============================================================================

def test_board_state_flattener():
    tiles = [[None] * 10 for _ in range(10)]
    # Place some locked tiles (SE quadrant)
    for y in range(5, 10):
        for x in range(5, 10):
            tiles[y][x] = "LOCKED"

    # Place a thirsty plant
    tiles[1][1] = {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
        "watered_today": False, "consecutive_unwatered": 1,
        "yield_units": 1, "max_lifespan_step": -1, "fertilized_until_day": -1,
    }

    # Place a ripe harvestable plant
    tiles[1][2] = {
        "kind": "PLANT", "crop": "MELON", "planted_day": 0,
        "watered_today": True, "consecutive_unwatered": 0,
        "yield_units": 6, "max_lifespan_step": 264, "fertilized_until_day": -1,
    }

    # Place an occupied animal in danger
    tiles[2][2] = {
        "kind": "COOP", "animal": "GOOSE", "placed_day": 0,
        "yield_units": 2, "fed_today": False, "consecutive_unfed": 1,
        "cared_today": False, "fertilizer_available": True, "pending_care_bonus": 1,
    }

    # Place an empty pasture
    tiles[3][3] = {"kind": "PASTURE", "animal": None}

    # Place a weed
    tiles[0][0] = {"kind": "WEED"}

    fb: FlattenedBoard = flatten_board_state(tiles, current_day=10, current_step=240)

    # Check terrain
    assert len(fb.locked_tiles) == 25
    assert fb.total_weeds == 1
    assert fb.total_vacant == 75 - 4 - 1  # 75 unlocked minus 4 occupied minus 1 weed = 70

    # Check crops
    assert len(fb.all_crops) == 2
    assert len(fb.thirsty_crops) == 1
    assert fb.thirsty_crops[0].crop == "WHEAT"
    assert len(fb.danger_crops) == 1
    assert fb.danger_crops[0].crop == "WHEAT"
    # Both Wheat (age 10 >= 4) and Melon (age 10 >= 10) are ripe
    assert len(fb.ripe_crops) == 2
    assert len(fb.harvestable_crops) == 2

    # Check animals
    assert len(fb.occupied_animals) == 1
    assert len(fb.hungry_animals) == 1
    assert len(fb.danger_animals) == 1
    assert len(fb.fertilizer_animals) == 1
    assert len(fb.empty_structures) == 1
    assert fb.empty_structures[0] == (3, 3, "PASTURE")


def test_fibonacci_cost_calculator():
    # Sequence: 1, 1, 2, 3, 5, 8, 13, 21
    assert cumulative_hire_cost(num_hires=1, already_hired_today=0) == 1
    assert cumulative_hire_cost(num_hires=3, already_hired_today=0) == 1 + 1 + 2  # 4
    assert cumulative_hire_cost(num_hires=2, already_hired_today=2) == 2 + 3      # 5
    assert cumulative_hire_cost(num_hires=4, already_hired_today=0) == 1 + 1 + 2 + 3  # 7

    # Max affordable hires given capital
    # Costs: 1, 1, 2, 3, 5 (cumul: 1, 2, 4, 7, 12)
    assert max_affordable_hires(available_money=10, reserve_liquidity=0) == 4  # 1+1+2+3=7 <= 10
    assert max_affordable_hires(available_money=10, reserve_liquidity=5) == 3  # 1+1+2=4 <= 5
    assert max_affordable_hires(available_money=0) == 0


def test_yield_trajectory_forecaster():
    tiles = [[None] * 10 for _ in range(10)]
    # Plant wheat at day 0
    tiles[0][0] = {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
        "watered_today": True, "consecutive_unwatered": 0,
        "yield_units": 1, "max_lifespan_step": -1, "fertilized_until_day": -1,
    }
    # Plant melon at day 0
    tiles[0][1] = {
        "kind": "PLANT", "crop": "MELON", "planted_day": 0,
        "watered_today": True, "consecutive_unwatered": 0,
        "yield_units": 1, "max_lifespan_step": -1, "fertilized_until_day": -1,
    }

    # Forecast at day 4 (Wheat is ready with 4 units; Melon is immature age 4)
    res_day4 = forecast_crop_yield_trajectory(tiles, target_day=4, current_day=0)
    assert res_day4["by_crop"]["WHEAT"] == 4
    assert res_day4["by_crop"]["MELON"] == 0  # Immature (first yield day 10)
    assert res_day4["ready_tiles_count"] == 1

    # Forecast at day 10 (Melon reaches peak yield 6)
    res_day10 = forecast_crop_yield_trajectory(tiles, target_day=10, current_day=0)
    assert res_day10["by_crop"]["MELON"] == 6


# ==============================================================================
# 2. Market Prediction & Concurrency Tests
# ==============================================================================

def test_demand_cycle_predictor():
    shops = ["BAKERY", "YARN_STORE"]
    # Bakery: EGG 1, WHEAT 1
    # Yarn Store: WOOL 2 (single-product 2x)
    # Town center: 1 of each non-fert every 24 steps

    ticks = predict_upcoming_consumption_ticks(shops, current_step=0, lookahead_turns=24)
    assert len(ticks) == 6  # 24 / 4 = 6 shop ticks (steps 4, 8, 12, 16, 20, 24)

    # Step 4: shop tick
    t4 = next(t for t in ticks if t.global_step == 4)
    assert t4.consumed_products["WHEAT"] == 1
    assert t4.consumed_products["EGG"] == 1
    assert t4.consumed_products["WOOL"] == 2
    assert t4.is_town_center_tick is False

    # Step 24: combined shop + town center tick
    t24 = next(t for t in ticks if t.global_step == 24)
    assert t24.is_shop_tick is True
    assert t24.is_town_center_tick is True
    assert t24.consumed_products["WHEAT"] == 1 + 1  # 1 shop + 1 town center
    assert t24.consumed_products["WOOL"] == 2 + 1   # 2 shop + 1 town center

    # Next consumption window
    offset, drained = find_next_consumption_window(shops, current_step=1, product="WOOL")
    assert offset == 3  # step 1 -> step 4 (3 turns)
    assert drained == 2

    # Total drain over 24 turns
    total_drain = estimate_market_drain_over_horizon(shops, start_step=0, num_turns=24)
    assert total_drain["WOOL"] == 2 * 6 + 1  # 13
    assert total_drain["WHEAT"] == 1 * 6 + 1 # 7


def test_slippage_simulator():
    # Selling 400 Wheat from equilibrium (I0 = 10000)
    # Starting price is $25; final price after 400 sold is $20
    res = simulate_sell_slippage("WHEAT", quantity=400, current_market_inv=10000)
    assert res.starting_price == 25
    assert res.ending_price == 20
    assert 20 <= res.average_price <= 25
    assert res.final_market_inventory == 10400
    assert len(res.unit_prices) == 400

    # Simultaneous concurrent opponent selling 400 as well
    res_opp = simulate_sell_slippage("WHEAT", quantity=400, current_market_inv=10000, opponent_simultaneous_sell=400)
    assert res_opp.final_market_inventory == 10800
    assert res_opp.ending_price == 19  # P(I0 + 2T) = 19
    assert res_opp.total_revenue < res.total_revenue  # Lower due to competition slippage

    # Buying 100 Wheat
    res_buy = simulate_buy_slippage("WHEAT", quantity=100, current_market_inv=10000)
    assert res_buy.final_market_inventory == 9900
    assert res_buy.total_revenue > 0  # Total purchase cost


# ==============================================================================
# 3. Solver Integration & Action Discretizer Tests
# ==============================================================================

def test_constraint_formulator():
    prob = formulate_production_constraints(
        available_money=3000.0,
        unlocked_tiles_count=25,
        current_shed_occupancy=20,
        horizon_days=4,
        reserve_cash=200.0,
    )

    assert len(prob.variable_names) == 8  # 5 crops + 3 animals
    assert len(prob.A_ub) == 4            # land, capital, shed, action economy
    assert prob.b_ub[0] == 25.0           # 25 tiles
    assert prob.b_ub[1] == 2800.0         # $3000 - $200 reserve
    assert prob.b_ub[2] == 80.0           # 100 cap - 20 current

    scipy_args = prob.to_scipy_args()
    assert "c" in scipy_args
    assert "A_ub" in scipy_args
    assert "b_ub" in scipy_args
    assert "bounds" in scipy_args


def test_action_discretizer():
    continuous_targets = {
        "WHEAT": 14.6,
        "MELON": 4.1,
        "GOOSE": 1.0,
        "SELL_WHEAT": 20.0,
    }
    plan = discretize_continuous_solution(
        solution_targets=continuous_targets,
        current_seeds={"WHEAT": 5, "MELON": 0},
        current_shed={"WHEAT": 25, "GOOSE": 0},
        available_money=2000.0,
        vacant_tiles=20,
    )

    assert plan.plant_targets["WHEAT"] == 15
    assert plan.plant_targets["MELON"] == 4
    assert plan.seed_purchases["WHEAT"] == 10  # 15 target - 5 existing
    assert plan.seed_purchases["MELON"] == 4   # 4 target - 0 existing
    assert plan.animal_purchases["GOOSE"] == 1
    assert plan.sell_orders["WHEAT"] == 20

    # Check market action queue
    orders = plan.market_action_queue
    assert ["BUY_SEED", "WHEAT", 10] in orders
    assert ["BUY_SEED", "MELON", 4] in orders
    assert ["BUY_ANIMAL", "GOOSE", 1] in orders
    assert ["SELL", "WHEAT", 20] in orders


# ==============================================================================
# 4. Opponent Modeling Tests
# ==============================================================================

def test_adversarial_asset_tracker():
    opp_tiles = [[None] * 10 for _ in range(10)]
    # Opponent planted 15 Melons on Day 0 (10 on row 0, 5 on row 2)
    for i in range(10):
        opp_tiles[0][i] = {
            "kind": "PLANT", "crop": "MELON", "planted_day": 0,
            "watered_today": True, "consecutive_unwatered": 0, "yield_units": 1,
        }
    for i in range(5):
        opp_tiles[2][i] = {
            "kind": "PLANT", "crop": "MELON", "planted_day": 0,
            "watered_today": True, "consecutive_unwatered": 0, "yield_units": 1,
        }
    # Opponent has 2 Geese
    opp_tiles[1][0] = {"kind": "COOP", "animal": "GOOSE"}
    opp_tiles[1][1] = {"kind": "COOP", "animal": "GOOSE"}

    opp_farm = {
        "money": 1200.0,
        "tiles": opp_tiles,
        "unlocked_quadrants": ["NW"],
    }

    # Analyze on Day 9 (1 day before opponent's Melon peak yield on Day 10)
    profile = analyze_opponent_farm(opp_farm, current_day=9, current_hour=0)

    assert profile.opponent_money == 1200.0
    assert profile.crop_counts["MELON"] == 15
    assert profile.animal_counts["GOOSE"] == 2
    assert len(profile.crop_groups) == 1

    grp = profile.crop_groups[0]
    assert grp.crop == "MELON"
    assert grp.optimal_harvest_day == 10
    assert grp.days_until_optimal_harvest == 1
    assert grp.estimated_total_yield == 15 * 6  # 90 melons

    # Verify Sabotage Opportunities flagged
    assert len(profile.sabotage_opportunities) >= 2
    front_run = next(s for s in profile.sabotage_opportunities if s.opportunity_type == "FRONT_RUN_HARVEST")
    assert front_run.target_product == "MELON"
    assert "SELL MELON" in front_run.recommended_action

    starvation = next(s for s in profile.sabotage_opportunities if s.opportunity_type == "FEED_STARVATION")
    assert starvation.target_product == "WHEAT"
