import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).parents[1] / "submissions" / "catalog_mill" / "main.py"
_SPEC = importlib.util.spec_from_file_location("catalog_mill_test", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_sell_qty = _MODULE._sell_qty


def test_catalog_terminal_day_liquidates_all_sellable_stock():
    assert _sell_qty("MILK", 18, 160, 40, False, 29, False, False) == 18
    assert _sell_qty("FERTILIZER", 18, 1, 40, False, 29, False, False) == 18


def test_catalog_low_fertilizer_quote_is_rate_limited_under_pressure():
    assert _sell_qty("FERTILIZER", 10, 19, 84, False, 15, False, False) == 0
    assert _sell_qty("FERTILIZER", 10, 19, 85, False, 15, False, False) == 1
