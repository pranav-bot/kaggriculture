"""Tests for the MidgameMarketManager O(1) guardrail."""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kaggriculture.env.items import MARKET_I0, market_price
from kaggriculture.market import (
    MidgameMarketManager,
    calculate_safe_sale_volume,
    is_midgame,
    micro_batch_order,
)


def test_wool_safe_volume_matches_engine_boundary():
    n = calculate_safe_sale_volume(200, MARKET_I0, "WOOL", 150)
    assert n == 30
    assert market_price("WOOL", MARKET_I0 + n - 1) >= 150  # last unit clears
    assert market_price("WOOL", MARKET_I0 + n) < 150  # next unit would not


def test_milk_safe_volume_matches_engine_boundary():
    n = calculate_safe_sale_volume(160, MARKET_I0, "MILK", 80)
    assert n == 39
    assert market_price("MILK", MARKET_I0 + n - 1) >= 80
    assert market_price("MILK", MARKET_I0 + n) < 80


def test_79_wool_dump_would_hit_floor_but_guard_clamps():
    assert market_price("WOOL", MARKET_I0 + 79) == 1  # the incident
    mgr = MidgameMarketManager()
    d = mgr.guard_sell_intent("WOOL", 79, MARKET_I0, turn_number=300)
    assert d["action"] == "MICRO_BATCH_SELL"
    assert d["safe_qty"] == 30 and d["deferred_qty"] == 49
    assert d["order"] == ["SELL", "WOOL", 30]
    assert d["base_intent"] == "DUMP_WOOL"


def test_safe_boundary_holds_across_book_states():
    for item, margin in (("WOOL", 150), ("MILK", 80), ("WOOL", 100), ("MILK", 120)):
        for inv in (9900, 9990, MARKET_I0, 10050, 10200):
            n = calculate_safe_sale_volume(market_price(item, inv), inv, item, margin)
            if n > 0:
                assert market_price(item, inv + n - 1) >= margin, (item, inv, n)
            nxt = market_price(item, inv + n)
            if nxt > 1:  # ignore floor truncation far above the book
                assert nxt < margin or n == 0, (item, inv, n)


def test_small_requests_pass_through_unclamped():
    mgr = MidgameMarketManager()
    d = mgr.guard_sell_intent("MILK", 10, MARKET_I0, turn_number=100)
    assert d["clamped"] is False and d["qty"] == 10
    assert d["action"] == "DUMP_MILK"


def test_zero_when_book_already_below_margin():
    assert calculate_safe_sale_volume(55, MARKET_I0 + 50, "WOOL", 150) == 0
    mgr = MidgameMarketManager()
    d = mgr.guard_sell_intent("WOOL", 5, MARKET_I0 + 50, turn_number=100)
    assert d["safe_qty"] == 0 and d["deferred_qty"] == 5


def test_handoff_disables_manager_from_day_25():
    assert is_midgame(0) and is_midgame(599)
    assert not is_midgame(600) and not is_midgame(719)
    mgr = MidgameMarketManager()
    d = mgr.guard_sell_intent("WOOL", 79, MARKET_I0, turn_number=600)
    assert d["action"] == "YIELD_TO_TERMINAL"
    assert mgr.yields == 1 and mgr.intercepts == 0


def test_executes_under_0_1ms():
    mgr = MidgameMarketManager()
    mgr.guard_sell_intent("WOOL", 79, MARKET_I0, 300)  # warmup
    t0 = time.perf_counter()
    k = 2000
    for i in range(k):
        mgr.guard_sell_intent("WOOL", 79, MARKET_I0 + (i % 50), 300)
    avg_ms = (time.perf_counter() - t0) / k * 1000
    assert avg_ms < 0.1, f"{avg_ms:.4f}ms per guard call"


def test_micro_batch_order_format():
    assert micro_batch_order("milk", 12) == ["SELL", "MILK", 12]
    assert micro_batch_order("WOOL", -3) == ["SELL", "WOOL", 0]
