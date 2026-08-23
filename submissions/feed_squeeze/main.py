"""
Feed Squeeze – Adversarial Market Manipulation Agent for Kaggriculture.

Implements four interlocking strategies:
  1. Feed Squeeze (Resource Starvation) – Buy wheat to starve opponent livestock.
  2. Just-in-Time Market Flooding  – Pre-dump crops to crash prices before opponent harvest.
  3. Concurrent Trade Execution    – Micro-batch orders to maximize opponent slippage.
  4. Counter-Hedge (Self-Sustaining) – Plant own wheat for feed + sell excess into inflated market.

Architecture:
  - Subclasses ActionController for automated farming (watering, harvesting, navigation).
  - Completely overrides plan_market_actions() with adversarial market intelligence.
  - Uses analyze_opponent_farm() for real-time opponent crop/livestock tracking.
  - Uses simulate_sell_slippage() / simulate_buy_slippage() for price impact modeling.
"""
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure local bundled packages (e.g. kaggriculture) are importable
if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import (
    Actions,
    ActionController,
    Plants,
    Animals,
    Products,
    CROPS,
    ANIMALS,
    MARKET_I0,
    MARKET_PARAMS,
    TURNS_PER_DAY,
    market_price,
    analyze_opponent_farm,
    simulate_sell_slippage,
    simulate_buy_slippage,
)


# =============================================================================
# Strategy Constants
# =============================================================================

# Feed Squeeze thresholds
MIN_OPPONENT_LIVESTOCK_FOR_SQUEEZE = 2   # Only attack if opponent has ≥2 animals
WHEAT_BUY_BUDGET_RATIO = 0.25           # Spend up to 25% of cash on wheat buying
MAX_WHEAT_BUY_PER_TURN = 8             # Cap wheat purchases per turn (leave slots for other orders)
WHEAT_PRICE_SQUEEZE_THRESHOLD = 40      # Start squeezing when wheat price exceeds this

# Just-in-Time Market Flooding
FRONT_RUN_DAYS_AHEAD = 1                # Dump product 0–1 days before opponent harvest
MIN_OPPONENT_HARVEST_VOLUME = 6         # Only front-run if opponent has ≥6 units maturing
MIN_DUMP_UNITS = 3                      # Need at least 3 units in shed to dump

# Counter-Hedge
WHEAT_SELF_SUFFICIENCY_TARGET = 8       # Keep ≥8 wheat seeds in inventory for own planting
WHEAT_SELL_PRICE_THRESHOLD = 35         # Sell excess wheat when price > 35 (base is 25)
RESERVE_WHEAT_IN_SHED = 5              # Always keep ≥5 wheat in shed for animal feed

# Phase gating (30-day game)
EARLY_GAME_END = 6                      # Days 0–5: build economy
MIDGAME_END = 20                        # Days 6–19: aggressive sabotage
# Days 20–29: liquidation phase


class FeedSqueezeController(ActionController):
    """
    Adversarial agent that weaponizes the market against opponents.

    Farm strategy: Self-sustaining wheat loop with opportunistic melon/carrot side crops.
    Market strategy: Feed starvation + harvest front-running + slippage optimization.
    """

    def __init__(self):
        super().__init__(
            target_crop=Plants.WHEAT,
            target_animal=None,               # No animals — we're the predator, not the prey
            auto_water=True,
            auto_harvest=True,
            auto_fertilize=False,             # Save fertilizer budget for market warfare
            auto_feed_animals=False,          # We don't keep animals
            auto_care_animals=False,
            auto_collect_fertilizer=False,
            auto_dig_weeds=True,
            auto_sell=False,                  # CRITICAL: Disable auto-sell — we control market timing
            auto_expand_land=True,
            auto_hire_hands=True,
            max_hires_per_day=1,
            min_sell_margin=0.8,
            board_size=10,
        )
        # Sabotage state tracking
        self._last_opponent_profile = None
        self._squeeze_active = False
        self._front_run_targets: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Core: Override act() to inject adversarial intelligence
    # -------------------------------------------------------------------------

    def act(self, obs: dict) -> Dict[str, Any]:
        """Master entry point with adversarial strategy layer."""
        player = obs["player"]
        me = obs["farms"][player]
        private = obs.get("private", {})
        market = obs.get("market", {})
        current_day = obs.get("day", 0)
        current_hour = obs.get("hour", 0)

        # --- Phase 1: Opponent Intelligence ---
        opp_idx = 1 - player
        opp_farm = obs["farms"][opp_idx] if len(obs["farms"]) > 1 else {}
        prices = market.get("prices", {})
        inventory = market.get("inventory", {})

        self._last_opponent_profile = analyze_opponent_farm(
            opponent_farm=opp_farm,
            current_day=current_day,
            current_hour=current_hour,
            market_prices=prices,
        )

        # --- Phase 2: Adapt crop target based on game phase ---
        self._adapt_crop_strategy(current_day, prices)

        # --- Phase 3: Standard farming automation (inherited from ActionController) ---
        available_seeds = dict(private.get("seeds", {}))
        claimed_tiles: Set[Tuple[int, int]] = set()

        farmer_act = self.plan_unit_action(0, me, private, available_seeds, obs, claimed_tiles)

        hands_act = []
        for h_idx in range(len(me.get("hands", []))):
            h_act = self.plan_unit_action(h_idx + 1, me, private, available_seeds, obs, claimed_tiles)
            hands_act.append(h_act)

        # --- Phase 4: Adversarial Market Actions (the weapon) ---
        market_act = self._plan_adversarial_market(
            farm=me,
            private=private,
            market=market,
            current_day=current_day,
            current_hour=current_hour,
        )

        return {
            "farmer": farmer_act,
            "hands": hands_act,
            "market": market_act,
        }

    # -------------------------------------------------------------------------
    # Strategy Adaptation
    # -------------------------------------------------------------------------

    def _adapt_crop_strategy(self, current_day: int, prices: Dict[str, int]) -> None:
        """
        Counter-Hedge: Plant wheat primarily for self-sustaining feed + market inflation profit.
        In early game, mix in melons for a big payday. Late game, fast-turnaround crops only.
        """
        days_remaining = 30 - current_day

        if current_day < EARLY_GAME_END:
            # Early: Wheat + some Melons if time permits
            if days_remaining >= 12:
                # Alternate: mostly wheat with some melon plantings
                self.target_crop = Plants.WHEAT
            else:
                self.target_crop = Plants.WHEAT
        elif current_day < MIDGAME_END:
            # Mid-game: Pure wheat for feed loop + market inflation profit
            self.target_crop = Plants.WHEAT
        else:
            # Late game: Only fast crops (wheat 4-day, carrot 3-day)
            if days_remaining >= 4:
                # Sell wheat into inflated market
                wheat_price = prices.get("WHEAT", 25)
                carrot_price = prices.get("CARROT", 35)
                if wheat_price > 35:
                    self.target_crop = Plants.WHEAT
                elif carrot_price > 45:
                    self.target_crop = Plants.CARROT
                else:
                    self.target_crop = Plants.WHEAT
            else:
                self.target_crop = Plants.WHEAT

    # -------------------------------------------------------------------------
    # Adversarial Market Engine
    # -------------------------------------------------------------------------

    def _plan_adversarial_market(
        self,
        farm: dict,
        private: dict,
        market: dict,
        current_day: int,
        current_hour: int,
    ) -> List[List[Any]]:
        """
        The core sabotage engine. Generates market orders in priority order:

        Priority 1: Land expansion (if affordable)
        Priority 2: Hire hands (1/day for farming throughput)
        Priority 3: FEED SQUEEZE — buy wheat to starve opponent livestock
        Priority 4: FRONT-RUN HARVEST — preemptive dump to crash opponent's crop prices
        Priority 5: COUNTER-HEDGE SELL — sell excess wheat into inflated market
        Priority 6: Buy seeds for own farming
        Priority 7: Opportunistic sells of any other harvested products

        All orders are micro-batched for slippage control.
        """
        orders: List[List[Any]] = []
        money = float(farm.get("money", 0))
        unlocked = farm.get("unlocked_quadrants", ["NW"])
        seeds = private.get("seeds", {})
        shed = private.get("shed", {})
        prices = market.get("prices", {})
        inventory = market.get("inventory", {})
        opp = self._last_opponent_profile

        # =====================================================================
        # Priority 1: Land Expansion
        # =====================================================================
        if len(unlocked) < 4:
            cost = Actions.land_cost(unlocked)
            if cost is not None and money >= cost * 1.2:  # Keep 20% buffer
                orders.append(Actions.buy_land())
                money -= cost

        # =====================================================================
        # Priority 2: Hire Hands (1 per day for throughput)
        # =====================================================================
        hires_today = int(farm.get("hires_today", 0))
        if hires_today < 1 and current_day < 25:
            hire_cost = Actions.hire_cost(hires_today)
            if money >= hire_cost + 200:  # Keep buffer for market ops
                orders.append(Actions.hire())
                money -= hire_cost

        # =====================================================================
        # Priority 3: FEED SQUEEZE — Wheat Starvation Attack
        # =====================================================================
        squeeze_orders = self._execute_feed_squeeze(
            opp=opp,
            money=money,
            shed=shed,
            prices=prices,
            inventory=inventory,
            current_day=current_day,
        )
        for order in squeeze_orders:
            orders.append(order)
            # Track approximate spend (wheat BUY_PRODUCT)
            if order[0] == "BUY_PRODUCT" and order[1] == "WHEAT":
                qty = order[2]
                buy_price = prices.get("WHEAT", 25)
                money -= buy_price * qty

        # =====================================================================
        # Priority 4: FRONT-RUN HARVEST — Preemptive Price Crash
        # =====================================================================
        frontrun_orders = self._execute_front_run_harvest(
            opp=opp,
            shed=shed,
            prices=prices,
            inventory=inventory,
            current_day=current_day,
        )
        for order in frontrun_orders:
            orders.append(order)

        # =====================================================================
        # Priority 5: COUNTER-HEDGE SELL — Profit from inflated wheat market
        # =====================================================================
        hedge_orders = self._execute_counter_hedge_sell(
            shed=shed,
            prices=prices,
            inventory=inventory,
            current_day=current_day,
        )
        for order in hedge_orders:
            orders.append(order)

        # =====================================================================
        # Priority 6: Buy Seeds
        # =====================================================================
        seed_count = seeds.get(self.target_crop, 0)
        if seed_count < 5:
            crop_cfg = CROPS.get(self.target_crop, CROPS["WHEAT"])
            qty = min(5 - seed_count, int(money // crop_cfg.seed_cost))
            if qty > 0 and len(orders) < Actions.MAX_MARKET_ORDERS_PER_TURN:
                orders.append(Actions.buy_seed(self.target_crop, qty))
                money -= qty * crop_cfg.seed_cost

        # =====================================================================
        # Priority 7: Sell Non-Wheat Products at Fair Prices
        # =====================================================================
        non_wheat_orders = self._sell_non_wheat_products(shed, prices, current_day)
        for order in non_wheat_orders:
            if len(orders) < Actions.MAX_MARKET_ORDERS_PER_TURN:
                orders.append(order)

        # Cap at engine limit
        return orders[:Actions.MAX_MARKET_ORDERS_PER_TURN]

    # -------------------------------------------------------------------------
    # Strategy 1: Feed Squeeze (Wheat Starvation)
    # -------------------------------------------------------------------------

    def _execute_feed_squeeze(
        self,
        opp: Optional[Any],
        money: float,
        shed: dict,
        prices: dict,
        inventory: dict,
        current_day: int,
    ) -> List[List[Any]]:
        """
        Target the Dependency: Every animal requires daily wheat.
        Corner the Market: Buy wheat aggressively to drain inventory and inflate price.
        Force Escapes: If opponent can't afford inflated wheat, animals hit consecutive_unfed=2.

        Micro-batching: Split large buys into individual orders for slippage control.
        The market processes orders unit-by-unit, so buying 1 at a time lets us
        see price escalation before committing more.
        """
        if opp is None:
            return []

        total_opp_livestock = sum(opp.animal_counts.values())
        if total_opp_livestock < MIN_OPPONENT_LIVESTOCK_FOR_SQUEEZE:
            self._squeeze_active = False
            return []

        # Only squeeze mid-game when opponent is committed to livestock
        if current_day < EARLY_GAME_END:
            return []

        self._squeeze_active = True
        orders: List[List[Any]] = []

        # Budget: up to WHEAT_BUY_BUDGET_RATIO of current cash
        wheat_budget = money * WHEAT_BUY_BUDGET_RATIO
        wheat_price = prices.get("WHEAT", 25)
        wheat_inv = inventory.get("WHEAT", MARKET_I0)

        # Don't buy if price is already sky-high (diminishing returns)
        if wheat_price > 80:
            return []

        # Model slippage: how many units can we buy within budget?
        slippage = simulate_buy_slippage(
            item="WHEAT",
            quantity=MAX_WHEAT_BUY_PER_TURN,
            current_market_inv=wheat_inv,
        )

        # Find max affordable quantity within budget
        affordable_qty = 0
        cumulative_cost = 0
        for unit_price in slippage.unit_prices:
            if cumulative_cost + unit_price <= wheat_budget:
                cumulative_cost += unit_price
                affordable_qty += 1
            else:
                break

        if affordable_qty <= 0:
            return []

        # --- Concurrent Trade Execution: Micro-batch the buys ---
        # Split into batches of 2 to control slippage and maximize price impact
        # The market engine processes all orders for a turn concurrently,
        # so each BUY_PRODUCT order is one "slot" in the 10-order queue.
        remaining = affordable_qty
        while remaining > 0 and len(orders) < MAX_WHEAT_BUY_PER_TURN:
            batch = min(2, remaining)
            orders.append(Actions.buy_product("WHEAT", batch))
            remaining -= batch

        return orders

    # -------------------------------------------------------------------------
    # Strategy 2: Just-in-Time Market Flooding
    # -------------------------------------------------------------------------

    def _execute_front_run_harvest(
        self,
        opp: Optional[Any],
        shed: dict,
        prices: dict,
        inventory: dict,
        current_day: int,
    ) -> List[List[Any]]:
        """
        Lifecycle Tracking: Parse opponent's crop groups to find imminent harvests.
        Preemptive Dump: Sell our hoarded supply of the same crop RIGHT BEFORE they harvest.
        Quadratic Crashing: Premium crops use aggressive price shapes (sq for MELON/WOOL).
            A preemptive sell crashes the price to $1 floor, neutralizing their harvest.

        The key insight: MELON uses above_func="sq" with above_target=3.60 —
        dumping even 10 melons into a balanced market drops the price from $250 to pennies.
        """
        if opp is None:
            return []

        orders: List[List[Any]] = []

        for group in opp.crop_groups:
            # Only target imminent harvests of premium crops
            if group.days_until_optimal_harvest > FRONT_RUN_DAYS_AHEAD:
                continue
            if group.estimated_total_yield < MIN_OPPONENT_HARVEST_VOLUME:
                continue

            crop_name = group.crop
            our_stock = shed.get(crop_name, 0)

            if our_stock < MIN_DUMP_UNITS:
                continue

            # Model the crash: simulate selling our stock
            crop_inv = inventory.get(crop_name, MARKET_I0)
            slippage = simulate_sell_slippage(
                item=crop_name,
                quantity=our_stock,
                current_market_inv=crop_inv,
            )

            # Only dump if we crash the price significantly (>40% drop)
            price_before = prices.get(crop_name, MARKET_PARAMS.get(crop_name, {}).get("base", 100))
            if slippage.ending_price < price_before * 0.6:
                # Micro-batch the dump: sell in batches of 3 for slippage precision
                remaining = our_stock
                while remaining > 0 and len(orders) < 4:  # Max 4 dump orders per turn
                    batch = min(3, remaining)
                    orders.append(Actions.sell(crop_name, batch))
                    remaining -= batch

        return orders

    # -------------------------------------------------------------------------
    # Strategy 3: Counter-Hedge (Self-Sustaining Wheat + Profit from Chaos)
    # -------------------------------------------------------------------------

    def _execute_counter_hedge_sell(
        self,
        shed: dict,
        prices: dict,
        inventory: dict,
        current_day: int,
    ) -> List[List[Any]]:
        """
        Self-Sustaining Loop: We grow our own wheat — no market dependency.
        Capitalize on Chaos: When wheat price is inflated (because WE drained the market),
            sell our EXCESS wheat into the inflated market for massive profit.
        This transfers the opponent's potential wealth directly into our bank.
        """
        orders: List[List[Any]] = []

        wheat_in_shed = shed.get("WHEAT", 0)
        wheat_price = prices.get("WHEAT", 25)

        # Keep a reserve for potential future animal purchases (defensive flexibility)
        sellable_wheat = wheat_in_shed - RESERVE_WHEAT_IN_SHED

        if sellable_wheat <= 0:
            return []

        # Only sell wheat when the price is inflated above threshold
        # (either by our squeeze or by natural scarcity)
        if wheat_price >= WHEAT_SELL_PRICE_THRESHOLD:
            # Model slippage to optimize sell quantity
            wheat_inv = inventory.get("WHEAT", MARKET_I0)
            slippage = simulate_sell_slippage(
                item="WHEAT",
                quantity=sellable_wheat,
                current_market_inv=wheat_inv,
            )

            # Only sell if average price stays profitable
            if slippage.average_price >= WHEAT_SELL_PRICE_THRESHOLD * 0.9:
                # Micro-batch sells for slippage control
                remaining = sellable_wheat
                while remaining > 0 and len(orders) < 3:
                    batch = min(4, remaining)
                    orders.append(Actions.sell("WHEAT", batch))
                    remaining -= batch

        return orders

    # -------------------------------------------------------------------------
    # Sell Non-Wheat Products
    # -------------------------------------------------------------------------

    def _sell_non_wheat_products(
        self,
        shed: dict,
        prices: dict,
        current_day: int,
    ) -> List[List[Any]]:
        """Sell non-wheat harvested products at fair market price."""
        orders: List[List[Any]] = []

        for item, count in shed.items():
            if count <= 0:
                continue
            if item == "WHEAT":
                continue  # Handled by counter-hedge
            if item in ANIMALS:
                continue  # Don't sell animals from shed

            base_price = MARKET_PARAMS.get(item, {}).get("base", 0)
            current_price = prices.get(item, 0)

            # Sell if price is at least 60% of base (more aggressive than default 80%)
            if base_price and current_price >= base_price * 0.6:
                # Late game: dump everything
                if current_day >= 25:
                    orders.append(Actions.sell(item, count))
                else:
                    orders.append(Actions.sell(item, count))

        return orders


# =============================================================================
# Kaggle Entrypoint
# =============================================================================

squeeze_controller = FeedSqueezeController()


def agent(obs):
    """Kaggle entrypoint for Feed Squeeze agent."""
    return squeeze_controller.act(obs)
