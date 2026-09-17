# Kaggriculture Agent Examples

Step-by-step examples for building competition-ready agents, from a 30-line starter to helper-driven harvest logic. Each example under `agents/` is a complete `main.py` you can test locally and copy into `submissions/`.

## Prerequisites

```bash
uv sync
source .venv/bin/activate
```

All examples assume you run commands from the **repository root**.

---

## The submission contract

Every Kaggle agent must expose a single top-level function:

```python
def agent(obs: dict) -> dict:
    return {
        "farmer": ["WATER"],           # list of action tokens for the farmer
        "hands": [["PASS"], ["NORTH"]], # one action list per hired hand
        "market": [["BUY_SEED", "WHEAT", 5]],  # up to 10 market orders
    }
```

The observation `obs` dict includes:

| Key | Contents |
|-----|----------|
| `player` | `0` or `1` — your player index |
| `day` | Current day (0–29) |
| `step` | Current turn within the episode |
| `farms` | Public farm state for both players (tiles, units, money) |
| `private` | Your shed, seeds, and per-unit inventories |
| `market` | Prices and market inventories |
| `town` | Unlocked shops and town-center demand |

### Boilerplate for local development

When developing outside a Kaggle bundle, add this path shim so `kaggriculture` imports resolve (the build script strips it from submissions):

```python
import os, sys
if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())
```

On Kaggle, `main.py` sits next to the bundled `kaggriculture/` package and the shim is harmless.

---

## Example progression

| # | Directory | Pattern | When to use |
|---|-----------|---------|-------------|
| 01 | `agents/01_basic_wheat_loop` | `ActionController` defaults | First submission, learning the API |
| 02 | `agents/02_melon_throughput` | Controller flags (hire, fertilize) | Higher throughput without custom code |
| 03 | `agents/03_adaptive_crop` | Subclass + override `act()` | Dynamic crop choice from market data |
| 04 | `agents/04_custom_market_planning` | Override `plan_market_actions()` | Phased buy/sell/hire/expand strategy |
| 05 | `agents/05_manual_actions` | Raw `Actions` builders | Full control, minimal dependencies |
| 06 | `agents/06_helper_driven_harvest` | `flatten_board_state` + `check_max_yield_met` | Precise harvest timing with helpers |

---

## 01 — Basic wheat loop

**File:** `agents/01_basic_wheat_loop/main.py`

The smallest viable agent. `ActionController` handles navigation, watering, harvesting, selling, weed removal, and land expansion automatically.

```python
from kaggriculture import ActionController, Plants

controller = ActionController(
    target_crop=Plants.WHEAT,
    auto_water=True,
    auto_harvest=True,
    auto_sell=True,
    auto_expand_land=True,
    auto_dig_weeds=True,
    min_sell_margin=0.8,
)

def agent(obs):
    return controller.act(obs)
```

**Key `ActionController` flags:**

| Flag | Effect |
|------|--------|
| `target_crop` | Seed type to plant on empty tiles |
| `auto_water` / `auto_harvest` | Field maintenance |
| `auto_sell` | Sell shed goods when price ≥ `min_sell_margin × base_price` |
| `auto_expand_land` | Buy land when affordable |
| `auto_hire_hands` | Hire farm hands each day (with `max_hires_per_day`) |
| `auto_fertilize` | Apply fertilizer carried by units |

**Test:**

```bash
python scripts/test_submission.py -a1 examples/agents/01_basic_wheat_loop/main.py -a2 random -s 24
```

**Submit:**

```bash
cp -r examples/agents/01_basic_wheat_loop submissions/my_wheat_agent
python scripts/build_submission.py -a my_wheat_agent
```

---

## 02 — Melon throughput

**File:** `agents/02_melon_throughput/main.py`

Same pattern as 01, but tuned for melon economics: fertilizer, four daily hires, faster sell liquidation, and a lower margin threshold.

```bash
python scripts/test_submission.py -a1 examples/agents/02_melon_throughput/main.py -a2 shop_opportunist
```

Melon reaches peak yield at day 10 with up to 6 units per tile — higher upside than wheat, but needs more watering and shed management across the season.

---

## 03 — Adaptive crop selection

**File:** `agents/03_adaptive_crop/main.py`

Subclass `ActionController` and override `act()` to update `self.target_crop` before delegating to `super().act(obs)`.

The scoring loop:

1. Skip crops that cannot mature before day 30.
2. Read live `market.prices` and town shop demand via `Actions.calculate_town_daily_consumption()`.
3. Rank crops by `price × yield_per_tile_per_day × demand_bonus − seed_cost`.

```python
class AdaptiveCropController(ActionController):
    def act(self, obs):
        self.target_crop = self._pick_best_crop(obs)
        return super().act(obs)
```

This is the same architectural pattern used by `submissions/market_velocity` and `submissions/shop_opportunist`.

---

## 04 — Custom market planning

**File:** `agents/04_custom_market_planning/main.py`

Override `plan_market_actions()` to replace the default seed/sell/hire logic while keeping the controller's unit routing (water, harvest, plant, navigate).

The example uses three phases:

| Phase | Days | Behavior |
|-------|------|----------|
| Bootstrap | 0–9 | Buy land + bulk melon seeds |
| Scale | 10–24 | Hire up to 6 hands per day |
| Liquidate | all | Sell melon when price ≥ 200 or shed ≥ 80 units |

Disable conflicting auto-flags when overriding:

```python
super().__init__(auto_sell=False, auto_expand_land=False, auto_hire_hands=False)
```

Market orders are capped at `Actions.MAX_MARKET_ORDERS_PER_TURN` (10).

---

## 05 — Manual actions

**File:** `agents/05_manual_actions/main.py`

Skip `ActionController` entirely. Build the return dict yourself:

```python
def agent(obs):
    farm = obs["farms"][obs["player"]]
    return {
        "farmer": _plan_farmer(farm, private, day),
        "hands": [Actions.pass_action() for _ in farm.get("hands", [])],
        "market": _plan_market(farm, private, market, day),
    }
```

Use `Actions.water()`, `Actions.harvest()`, `Actions.plant_wheat()`, `Actions.buy_seed()`, `Actions.sell()`, etc. Navigation uses cardinal direction lists: `Actions.north()`, `Actions.east()`, …

Best for:

- Minimal single-file submissions (pair with `build_submission.py -f single`)
- Custom multi-unit task assignment
- Learning exactly what the engine expects each turn

---

## 06 — Helper-driven harvest

**File:** `agents/06_helper_driven_harvest/main.py`

Combine `ActionController` with library helpers for smarter decisions:

- **`flatten_board_state(tiles, current_day)`** — groups tiles into `ripe_crops`, `thirsty_crops`, `harvestable_crops`, etc.
- **`check_max_yield_met(plant_state)`** — boolean peak-yield check using bonus-window math

Override `plan_unit_action()` to route units toward helper-identified ripe tiles first, then fall back to default controller behavior:

```python
board = flatten_board_state(farm["tiles"], current_day=current_day)
for crop in board.ripe_crops:
    if check_max_yield_met({...}):
        # navigate to (crop.x, crop.y) and HARVEST
```

See also `submissions/feed_squeeze` for advanced helper usage (`analyze_opponent_farm`, `simulate_sell_slippage`).

---

## Utility scripts in `examples/`

### `run_match.py`

Short 72-turn demo of `Environment`, crop configs, and `ActionController` vs random:

```bash
python examples/run_match.py
```

### `run_feed_squeeze.py`

Runs the adversarial `feed_squeeze` submission against a wheat defender and random:

```bash
python examples/run_feed_squeeze.py
```

---

## From example to competition submission

```bash
# 1. Copy an example as your starting point
cp -r examples/agents/03_adaptive_crop submissions/my_agent

# 2. Edit submissions/my_agent/main.py

# 3. Smoke test (one day)
python scripts/test_submission.py -a1 my_agent -a2 random -s 24

# 4. Full-season benchmark
python scripts/test_submission.py -a1 my_agent -a2 shop_opportunist

# 5. Round-robin against all templates
python standoff/run_standoff.py -a my_agent -o standoff/my_agent_results.json

# 6. Package and validate
python scripts/build_submission.py -a my_agent

# 7. Submit
kaggle competitions submit kaggriculture -f build/submission.tar.gz -m "my_agent v1"
```

---

## Choosing an approach

```
Need fastest path to a working bot?
  └─ 01_basic_wheat_loop

Need more cash without custom logic?
  └─ 02_melon_throughput (flags only)

Market prices / shop demand matter?
  └─ 03_adaptive_crop (override act)

Phased investment (early expand, late liquidate)?
  └─ 04_custom_market_planning (override plan_market_actions)

Want minimal code / full control?
  └─ 05_manual_actions

Harvest timing is critical?
  └─ 06_helper_driven_harvest

Opponent-aware market warfare?
  └─ submissions/feed_squeeze (see examples/run_feed_squeeze.py)
```

---

## Further reading

- `README.md` — scripts, standoff, and workflow
- `strategy_learnings.md` — benchmark results and strategy notes
- `submissions/` — production agent templates ranked by performance
- `src/kaggriculture/actions/controller.py` — full `ActionController` API
- `src/kaggriculture/helpers/` — board flattening, market prediction, opponent analysis
