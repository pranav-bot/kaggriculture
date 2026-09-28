"""Opening Book Controller for Kaggriculture.

Implements the deterministic Rank 1 opening book (Days 0-15) extracted from elite
competitors (e.g. Boey, Kaggledew Valley), overriding step-level Beam Search with
consensus macro-intents, and cleanly releasing control to the Option-Critic
MacroOptionManager on Day 16.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Tactical build targets for Days 0-15 based on majority consensus
OPENING_BOOK_SCHEDULE: Dict[int, Dict[str, Any]] = {
    0: {
        "macro_intent": "DAY0_TURBO_HERD",
        "description": "Launch Day 0 compounding engine: 3 Cows + 2 Sheep + 10 Wheat seeds + 8 Melon seeds.",
        "buffer_fertilizer": False,
        "sell_fertilizer": True,
        "target_cows": 3,
        "target_sheep": 2,
        "expand": None,
    },
    1: {
        "macro_intent": "FEED_CARE_COMPOUND",
        "description": "Feed and care for herd; zero animal buys; buffer Fertilizer.",
        "buffer_fertilizer": True,
        "sell_fertilizer": False,
        "expand": None,
    },
    2: {
        "macro_intent": "FEED_CARE_COMPOUND",
        "description": "Harvest initial Milk and Wool; buffer Fertilizer; compound cash.",
        "buffer_fertilizer": True,
        "sell_fertilizer": False,
        "expand": None,
    },
    3: {
        "macro_intent": "SCALE_HERD_COWS",
        "description": "Scale herd to 7 Cows + 2 Sheep as compound cash flows in.",
        "buffer_fertilizer": True,
        "sell_fertilizer": False,
        "target_cows": 7,
        "expand": None,
    },
    4: {
        "macro_intent": "BUFFER_FERTILIZER",
        "description": "Buffer 100% of Fertilizer in shed; preserve cash for Day 5/6 NE expansion.",
        "buffer_fertilizer": True,
        "sell_fertilizer": False,
        "expand": None,
    },
    5: {
        "macro_intent": "LIQUIDATE_FOR_EXPANSION",
        "description": "Sell buffered Fertilizer and excess Wheat; reach $1,000+ cash threshold for NE expansion.",
        "buffer_fertilizer": False,
        "sell_fertilizer": True,
        "target_cash": 1000,
        "expand": None,
    },
    6: {
        "macro_intent": "EXPAND_NE_QUADRANT",
        "description": "Unlock NE Quadrant ($1,000) at Hour 2; construct pastures; plant initial Strawberries.",
        "buffer_fertilizer": False,
        "sell_fertilizer": True,
        "expand": "NE",
        "expand_hour": 2,
        "target_strawberries": 12,
    },
    7: {
        "macro_intent": "SCALE_STRAWBERRIES_NE",
        "description": "Scale Strawberry plantation on NE quadrant (16 plots); care for herd.",
        "buffer_fertilizer": False,
        "sell_fertilizer": True,
        "target_strawberries": 16,
        "expand": None,
    },
    8: {
        "macro_intent": "ACCUMULATE_SW_RESERVE",
        "description": "High-frequency commodity selling; accumulate $2,000 liquid cash reserve for SW expansion.",
        "buffer_fertilizer": False,
        "sell_fertilizer": True,
        "target_cash": 2000,
        "expand": None,
    },
    9: {
        "macro_intent": "EXPAND_SW_QUADRANT",
        "description": "Unlock SW Quadrant ($2,000) at Hour 2; construct additional pastures; expand Strawberry plots.",
        "buffer_fertilizer": False,
        "sell_fertilizer": True,
        "expand": "SW",
        "expand_hour": 2,
        "target_strawberries": 24,
    },
    10: {
        "macro_intent": "TRI_QUADRANT_SCALING",
        "description": "Operate across all 3 unlocked quadrants (NW, NE, SW); scale herd to 15+; harvest high-margin Strawberries.",
        "buffer_fertilizer": False,
        "sell_fertilizer": True,
        "target_herd": 15,
        "target_strawberries": 32,
        "expand": None,
    },
    11: {
        "macro_intent": "TRI_QUADRANT_SCALING",
        "description": "Maintain tri-quadrant production: watering, weeding, and high-frequency market sales.",
        "buffer_fertilizer": False,
        "sell_fertilizer": True,
        "target_herd": 15,
        "target_strawberries": 32,
        "expand": None,
    },
    12: {
        "macro_intent": "TRI_QUADRANT_SCALING",
        "description": "Midday market evaluation; scale Strawberry plots to 36; steady livestock care.",
        "buffer_fertilizer": False,
        "sell_fertilizer": True,
        "target_herd": 16,
        "target_strawberries": 36,
        "expand": None,
    },
    13: {
        "macro_intent": "PRE_HANDOFF_CONSOLIDATION",
        "description": "Consolidate feed loop; replant Wheat; prepare livestock for midgame scaling.",
        "buffer_fertilizer": False,
        "sell_fertilizer": True,
        "target_herd": 16,
        "target_strawberries": 40,
        "expand": None,
    },
    14: {
        "macro_intent": "PRE_HANDOFF_CONSOLIDATION",
        "description": "High liquidity maintenance; weed clearance; maximize strawberry collections.",
        "buffer_fertilizer": False,
        "sell_fertilizer": True,
        "target_herd": 18,
        "target_strawberries": 40,
        "expand": None,
    },
    15: {
        "macro_intent": "PRE_HANDOFF_CONSOLIDATION",
        "description": "Final opening book phase; verify herd health before handing off to Option-Critic Beam Search on Day 16.",
        "buffer_fertilizer": False,
        "sell_fertilizer": True,
        "target_herd": 18,
        "target_strawberries": 44,
        "expand": None,
    },
}


class OpeningBookController:
    """Runtime controller that executes the deterministic Rank 1 Opening Book
    for Days 0-15 and cleanly hands off control to the MacroOptionManager on Day 16.

    Attributes:
        HANDOFF_DAY: The day when control is permanently released to MacroOptionManager (16).
        active: True when operating under Opening Book (days 0-15), False on day 16+.
    """

    HANDOFF_DAY: int = 16

    def __init__(
        self,
        option_manager: Any = None,
        liquidation_controller: Any = None,
    ) -> None:
        """Initialize the OpeningBookController.

        Args:
            option_manager: Optional MacroOptionManager instance to hand off to.
                If None, lazily imports from scratch_grandmaster.
            liquidation_controller: Optional LiquidationController instance for Days 20-29.
        """
        self._option_manager = option_manager
        self._liquidation_controller = liquidation_controller
        self.active: bool = True
        self._last_day: int = -1
        self._current_intent: Optional[str] = None
        self._intent_description: Optional[str] = None

    @property
    def option_manager(self) -> Any:
        if self._option_manager is None:
            from scratch_grandmaster import MacroOptionManager
            self._option_manager = MacroOptionManager()
        return self._option_manager

    @property
    def liquidation_controller(self) -> Any:
        if self._liquidation_controller is None:
            from kaggriculture.liquidation_solver import LiquidationController
            self._liquidation_controller = LiquidationController()
        return self._liquidation_controller

    def get_day_directive(self, day: int) -> Dict[str, Any]:
        """Get the deterministic build directive for the given in-game day."""
        return OPENING_BOOK_SCHEDULE.get(day, {
            "macro_intent": "MAINTAIN_HERD",
            "description": f"Standard maintenance for Day {day}",
            "buffer_fertilizer": False,
            "sell_fertilizer": True,
            "expand": None,
        })

    def is_active(self, obs: Mapping[str, Any]) -> bool:
        """Return True if the opening book should govern decisions on this turn."""
        day = int(obs.get("day", 0))
        return day < self.HANDOFF_DAY

    def act(
        self,
        obs: Dict[str, Any],
        simulator: Any = None,
        IQL_Value_Net: Any = None,
        action_to_simulator: Any = None,
    ) -> Dict[str, Any]:
        """Execute one turn: either follow the Opening Book (Days 0-15)
        or hand off to MacroOptionManager (Day 16+).

        Args:
            obs: Observation dictionary.
            simulator: Optional simulator for MacroOptionManager after handoff.
            IQL_Value_Net: Optional IQL value network for MacroOptionManager after handoff.
            action_to_simulator: Optional simulator action mapper.

        Returns:
            Action dictionary with farmer, hands, and market commands.
        """
        from scratch_grandmaster import agent as base_agent

        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))

        # =====================================================================
        # PHASE 1: OPENING BOOK OVERRIDE (Days 0 - 15)
        # =====================================================================
        if day < self.HANDOFF_DAY:
            self.active = True
            directive = self.get_day_directive(day)
            self._current_intent = directive["macro_intent"]
            self._intent_description = directive["description"]

            # Compute standard mechanical operations (Kuhn-Munkres bipartite matching)
            ops = base_agent(obs)
            market_orders = ops.get("market") or []

            # Apply Opening Book Market Overrides
            farm = obs.get("farms", [{}, {}])[int(obs.get("player", 0))]
            money = float(farm.get("money", 0.0))
            quadrants = list(farm.get("unlocked_quadrants") or ["NW"])

            # 1. Fertilizer Buffering: In Days 1-4, suppress fertilizer sales
            if directive.get("buffer_fertilizer", False):
                market_orders = [
                    o for o in market_orders
                    if not (len(o) >= 2 and str(o[0]).upper() == "SELL" and str(o[1]).upper() == "FERTILIZER")
                ]

            # 2. Quadrant Expansion overrides at specific hours
            target_expand = directive.get("expand")
            expand_hour = directive.get("expand_hour", 2)

            if target_expand == "NE" and "NE" not in quadrants:
                if money >= 1000 and hour >= expand_hour:
                    has_expand = any(str(o[0]).upper() in ("BUY_LAND", "EXPAND") for o in market_orders)
                    if not has_expand and len(market_orders) < 10:
                        market_orders.insert(0, ["BUY_LAND", "NE"])

            elif target_expand == "SW" and "SW" not in quadrants:
                if money >= 2000 and hour >= expand_hour:
                    has_expand = any(str(o[0]).upper() in ("BUY_LAND", "EXPAND") for o in market_orders)
                    if not has_expand and len(market_orders) < 10:
                        market_orders.insert(0, ["BUY_LAND", "SW"])

            ops["market"] = market_orders
            ops["_opening_book_active"] = True
            ops["_opening_day"] = day
            ops["_opening_macro_intent"] = self._current_intent
            ops["_opening_description"] = self._intent_description
            ops["_handoff_ready"] = (day == 15 and hour >= 20)
            return ops

        # =====================================================================
        # PHASE 2: CLEAN HANDOFF TO MACRO-OPTION-MANAGER (Days 16 - 19)
        # =====================================================================
        self.active = False
        mgr = self.option_manager

        ops = mgr.act(
            obs,
            simulator=simulator,
            IQL_Value_Net=IQL_Value_Net,
            action_to_simulator=action_to_simulator,
        )

        ops["_opening_book_active"] = False
        ops["_handed_off_to_option_critic"] = True

        # =====================================================================
        # PHASE 3: RETROGRADE DP / MILP LIQUIDATION OVERRIDE (Days 20 - 29)
        # =====================================================================
        if day >= 20:
            ops = self.liquidation_controller.filter_mechanical_actions(ops, obs)
            ops["_liquidation_active"] = True

        return ops

    def stats(self) -> Dict[str, Any]:
        """Return diagnostic state."""
        return {
            "active": self.active,
            "handoff_day": self.HANDOFF_DAY,
            "current_intent": self._current_intent,
            "description": self._intent_description,
            "option_manager_stats": self.option_manager.stats() if self._option_manager else None,
        }
