# Creating Kaggriculture Agents

A complete guide to building, testing, and submitting agents in this workspace — from your first wheat loop to market-aware and adversarial strategies.

**Who this is for:** Anyone new to the repo or the Kaggle competition who wants to understand how the game works, what an agent must do each turn, and how to grow a simple bot into a competitive submission.

**Companion resources:**

| Resource | Purpose |
|----------|---------|
| [`examples/agents/`](../examples/agents/) | Runnable `main.py` files for each skill level |
| [`examples/README.md`](../examples/README.md) | Quick index of example agents |
| [`submissions/`](../submissions/) | Production templates ranked by benchmark performance |
| [`strategy_learnings.md`](../strategy_learnings.md) | What worked and what failed in local testing |

---

## Table of contents

1. [How the game works](#1-how-the-game-works)
2. [What your agent does each turn](#2-what-your-agent-does-each-turn)
3. [Workspace layout](#3-workspace-layout)
4. [Your first agent (Level 1)](#4-your-first-agent-level-1)
5. [Tuning the controller (Level 2)](#5-tuning-the-controller-level-2)
6. [Dynamic strategy (Level 3)](#6-dynamic-strategy-level-3)
7. [Custom market logic (Level 4)](#7-custom-market-logic-level-4)
8. [Manual actions (Level 5)](#8-manual-actions-level-5)
9. [Library helpers (Level 6)](#9-library-helpers-level-6)
10. [Advanced / adversarial agents (Level 7)](#10-advanced--adversarial-agents-level-7)
11. [Crop and animal reference](#11-crop-and-animal-reference)
12. [Observation schema](#12-observation-schema)
13. [Action reference](#13-action-reference)
14. [Development workflow](#14-development-workflow)
15. [Common pitfalls](#15-common-pitfalls)

---

## 1. How the game works

Kaggriculture is a **two-player farming economy game** played over a **30-day season** (720 turns). Each player manages a 10×10 farm, hires workers, plants crops, raises livestock, and sells produce on a shared market. The player with the **most cash at the end of day 29** wins.

### Time structure

| Unit | Value |
|------|-------|
| Turns per day | 24 |
| Days per season | 30 |
| Total turns | 720 |
| Starting cash | $3,000 |

Each turn your agent is called once. You issue actions for your **farmer**, every **hired hand**, and the **market** simultaneously.

### The farm board

```
  NW quadrant (unlocked)  |  NE quadrant (locked until bought)
  ------------------------+------------------------
  SW quadrant (locked)    |  SE quadrant (locked)
```

- You start with only the **NW** quadrant (25 tiles) unlocked.
- Land expansions cost **$1,000 → $2,000 → $4,000** (NE, SW, SE order).
- The **shed** sits at the center; units drop harvested goods on the four tiles adjacent to it.
- A tile can be: empty (`None`), a **plant**, a **weed**, a **coop/pasture** (with or without an animal), or `"LOCKED"`.

### Units

| Unit | Role |
|------|------|
| **Farmer** | Starts on the board; carries inventory; performs tile actions |
| **Farm hands** | Hired daily; spawn near the shed; reset each day (not permanent payroll) |
| **Market orders** | Up to 10 per turn: buy seeds, sell goods, hire, expand land, buy animals |

Each unit gets **one action per turn**: move one step (N/S/E/W), pass, or perform a tile action (water, harvest, plant, etc.).

### Crops (simplified)

**One-time crops** (wheat, carrot, melon) grow for several days, accumulate yield during a **bonus window**, then should be harvested. If you miss watering for two consecutive days, the tile becomes a **weed**.

**Ongoing crops** (tomato, strawberry) produce on a schedule (daily or every other day) for a fixed number of harvests. They also need watering every two days to survive.

**Yield bonus window** (one-time crops):

- Base yield = 1
- Bonus window starts at `ceil(time_to_max_yield / 2)` days of age
- Each watered day in the window: **+1** yield
- Each watered **and fertilized** day: **+2** yield
- Capped at the crop's absolute max (wheat/carrot need fertilizer to reach their top cap)

### Animals

Geese (coops), cows and sheep (pastures) produce eggs, milk, and wool on schedules. They must be **fed** (wheat) and **cared for** regularly or they escape. Animals can also produce **fertilizer** for crops.

### Market and town

- **Market prices** move with supply and demand (shared between both players).
- Selling large quantities **crashes** the price; buying large quantities **raises** it.
- **Town shops** unlock every 3 days and consume products on a schedule, creating demand spikes.
- Timing sales around shop drains and opponent harvests is a major competitive lever.

### Competition time limits

On Kaggle, each turn has a **1 second** soft budget with a **60 second** overage bank per episode. The `standoff/` runner enforces this locally. Keep `act()` fast — prefer pre-computed strategy over heavy optimization per turn unless you have headroom.

---

## 2. What your agent does each turn

### The contract

Every submission must define one function:

```python
def agent(obs: dict) -> dict:
    ...
```

**Input:** `obs` — full game state visible to your player (see [Observation schema](#12-observation-schema)).

**Output:** an action dictionary with three keys:

```python
{
    "farmer":  ["WATER"],                    # list of action tokens
    "hands":   [["NORTH"], ["PASS"]],         # one list per hired hand
    "market":  [["BUY_SEED", "WHEAT", 5]],   # up to 10 orders
}
```

### Turn execution order (conceptual)

1. Your agent receives `obs`.
2. The engine applies farmer action, then each hand action, then market orders.
3. End-of-day housekeeping runs every 24 turns (watering resets, shop consumption, etc.).
4. The next turn begins with an updated `obs`.

### Minimal mental model

```
obs ──► your logic ──► { farmer, hands, market } ──► engine
```

You do **not** control the opponent. You **do** see their public farm tiles (crops, animals, structures) but not their shed, seeds, or unit inventories.

### Using the library vs writing everything yourself

| Approach | Lines of code | Best for |
|----------|---------------|----------|
| `ActionController` with defaults | ~25 | First working bot |
| `ActionController` + flag tuning | ~30 | Higher throughput |
| Subclass `ActionController` | ~60–120 | Adaptive strategy |
| Manual `Actions` builders | ~80+ | Full control |
| Helpers + custom routing | ~100+ | Precision timing, opponent modeling |

The rest of this guide walks through each level in order.

---

## 3. Workspace layout

```
kaggriculture/
├── src/kaggriculture/          # Shared library (bundled into submissions)
│   ├── actions/                # Actions, ActionController, crop/animal configs
│   ├── env/                    # Environment wrapper, constants, tile models
│   └── helpers/                # Board flattening, market prediction, opponent analysis
├── submissions/<name>/main.py  # Your competition agents (one folder per agent)
├── examples/agents/            # Tutorial agents (copy these to submissions/)
├── scripts/
│   ├── test_submission.py      # Run head-to-head matches
│   └── build_submission.py     # Package + validate for Kaggle
├── standoff/run_standoff.py    # Round-robin benchmark vs all templates
└── docs/creating-agents.md     # This file
```

### Setup

```bash
uv sync
source .venv/bin/activate
```

### Where agents live

- **Develop** in `submissions/my_agent/main.py` (or copy from `examples/agents/`).
- **Test** with `scripts/test_submission.py`.
- **Package** with `scripts/build_submission.py` → `build/submission.tar.gz`.
- **Submit** with the Kaggle CLI.

Every `main.py` needs a Kaggle entrypoint:

```python
def agent(obs):
    return controller.act(obs)
```

### Local import boilerplate

When running outside a Kaggle bundle, add this at the top of `main.py` so imports work:

```python
import os, sys
if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())
```

The build script copies the `kaggriculture` package next to `main.py` and strips this shim automatically.

---

## 4. Your first agent (Level 1)

**Goal:** A working submission in under 30 lines.

**Example:** [`examples/agents/01_basic_wheat_loop/main.py`](../examples/agents/01_basic_wheat_loop/main.py)

```python
import os, sys
if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

### What `ActionController.act()` does

Each turn it builds the return dict by:

1. **`plan_unit_action`** for the farmer — navigate, water, harvest, plant, dig weeds
2. **`plan_unit_action`** for each hired hand — same logic, with tile-claiming to avoid collisions
3. **`plan_market_actions`** — buy seeds, sell shed goods, hire hands, buy land

You configure behavior through constructor flags; no custom code required.

### Create your own

```bash
cp -r examples/agents/01_basic_wheat_loop submissions/my_first_agent
# edit submissions/my_first_agent/main.py if needed
python scripts/test_submission.py -a1 my_first_agent -a2 random -s 24
```

A 24-turn smoke test runs one game-day. You should see final cash printed with no errors.

---

## 5. Tuning the controller (Level 2)

**Goal:** Increase throughput by flipping flags — still no subclassing.

**Example:** [`examples/agents/02_melon_throughput/main.py`](../examples/agents/02_melon_throughput/main.py)

### Key `ActionController` parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `target_crop` | `WHEAT` | Seed type planted on empty tiles |
| `target_animal` | `None` | Livestock to buy and place |
| `auto_water` | `True` | Route units to thirsty crops |
| `auto_harvest` | `True` | Harvest when ripe |
| `auto_fertilize` | `False` | Apply carried fertilizer to crops |
| `auto_feed_animals` | `True` | Feed livestock with carried wheat |
| `auto_care_animals` | `True` | Care for livestock |
| `auto_collect_fertilizer` | `True` | Collect animal fertilizer |
| `auto_dig_weeds` | `True` | Clear weed tiles |
| `auto_sell` | `True` | Sell shed goods via market orders |
| `auto_expand_land` | `True` | Buy land when affordable |
| `auto_hire_hands` | `False` | Hire farm hands each day |
| `max_hires_per_day` | `1` | Cap on daily hires |
| `min_sell_margin` | `0.8` | Sell when `price ≥ margin × base_price` |

### Melon throughput recipe

Melon has the highest peak value per tile ($250 base, up to 6 units) but needs 10 days to mature:

```python
controller = ActionController(
    target_crop=Plants.MELON,
    auto_water=True,
    auto_harvest=True,
    auto_fertilize=True,       # reach max yield faster
    auto_hire_hands=True,
    max_hires_per_day=4,       # more tiles serviced per day
    auto_sell=True,
    min_sell_margin=0.5,       # sell sooner to free shed space
    auto_expand_land=True,
)
```

### Unit task priority (built into controller)

When routing a unit, the controller prioritizes:

1. **Emergency** — crops about to become weeds; animals about to escape
2. **Ripe harvest** — optimal-age crops with yield ready
3. **Water / fertilize** — regular maintenance
4. **Plant** — empty tiles when seeds available
5. **Weeds** — dig if enabled

Understanding this priority helps you decide which flags to enable.

### When Level 2 is enough

Flag tuning works well when one crop strategy dominates your plan (melon flood, wheat loop). It stops being enough when **market prices**, **shop demand**, or **season phase** should change your behavior — move to Level 3.

---

## 6. Dynamic strategy (Level 3)

**Goal:** Change `target_crop` (or other settings) each turn based on `obs`.

**Example:** [`examples/agents/03_adaptive_crop/main.py`](../examples/agents/03_adaptive_crop/main.py)

### Pattern: subclass and override `act()`

```python
from kaggriculture import ActionController, Actions, CROPS, Plants

class AdaptiveCropController(ActionController):
    def __init__(self):
        super().__init__(
            target_crop=Plants.WHEAT,
            auto_water=True,
            auto_harvest=True,
            auto_fertilize=True,
            auto_hire_hands=True,
            max_hires_per_day=2,
            auto_sell=True,
            min_sell_margin=0.6,
        )

    def _pick_best_crop(self, obs):
        day = obs["day"]
        prices = obs["market"]["prices"]
        shops = obs["town"]["unlocked_shops"]
        demand = Actions.calculate_town_daily_consumption(shops)
        days_left = 30 - day

        scores = {}
        for name, crop in CROPS.items():
            if crop.time_to_first_yield > days_left:
                continue  # won't mature in time
            price = prices.get(name, crop.base_market_price)
            demand_bonus = 1.0 + 0.05 * demand.get(name, 0)
            scores[name] = price * crop.yield_per_tile_per_day * demand_bonus - crop.seed_cost

        return max(scores, key=scores.get, default=Plants.WHEAT)

    def act(self, obs):
        self.target_crop = self._pick_best_crop(obs)
        return super().act(obs)   # delegate field work + default market logic

controller = AdaptiveCropController()

def agent(obs):
    return controller.act(obs)
```

### What to read from `obs` for decisions

| Signal | Location | Use |
|--------|----------|-----|
| Current day | `obs["day"]` | Phase gating, maturity checks |
| Market prices | `obs["market"]["prices"]` | Crop scoring, sell timing |
| Your cash | `obs["farms"][obs["player"]]["money"]` | Affordability |
| Unlocked shops | `obs["town"]["unlocked_shops"]` | Demand forecasting |
| Shed contents | `obs["private"]["shed"]` | Liquidation pressure |
| Opponent tiles | `obs["farms"][1 - obs["player"]]["tiles"]` | Public board intel |

### Season phase gating

Strong agents often disable expensive actions late in the season:

```python
def act(self, obs):
    day = obs["day"]
    if day >= 26:
        self.auto_expand_land = False
        self.auto_hire_hands = False
    self.target_crop = self._pick_best_crop(obs)
    return super().act(obs)
```

### Production templates using this pattern

- `submissions/market_velocity` — adaptive crop scoring + aggressive hiring
- `submissions/shop_opportunist` — shop demand + livestock opportunism

---

## 7. Custom market logic (Level 4)

**Goal:** Control buying, selling, hiring, and expansion while keeping the controller's field routing.

**Example:** [`examples/agents/04_custom_market_planning/main.py`](../examples/agents/04_custom_market_planning/main.py)

### Pattern: override `plan_market_actions()`

```python
class PhasedMarketController(ActionController):
    def __init__(self):
        super().__init__(
            target_crop=Plants.MELON,
            auto_sell=False,          # we sell manually
            auto_expand_land=False,   # we expand manually
            auto_hire_hands=False,    # we hire manually
            ...
        )

    def plan_market_actions(self, farm, private, market, current_day):
        orders = []
        money = farm["money"]
        shed = private["shed"]

        if current_day < 10:
            orders.append(Actions.buy_land())          # if affordable
            orders.append(Actions.buy_seed("MELON", 10))

        elif current_day < 25:
            orders.append(Actions.hire())              # repeat while affordable

        if shed.get("MELON", 0) >= 80:
            orders.append(Actions.sell("MELON", 20))

        return orders[: Actions.MAX_MARKET_ORDERS_PER_TURN]
```

### Disable conflicting auto-flags

If you override `plan_market_actions()`, turn off the matching `auto_*` flags in `__init__`. Otherwise both your custom orders and the default logic may fight each other.

### Market order format

```python
Actions.buy_seed("MELON", 5)      # → ["BUY_SEED", "MELON", 5]
Actions.sell("WHEAT", 10)         # → ["SELL", "WHEAT", 10]
Actions.hire()                    # → ["HIRE"]
Actions.buy_land()                # → ["BUY_LAND"]
Actions.buy_animal("SHEEP", 1)    # → ["BUY_ANIMAL", "SHEEP", 1]
Actions.buy_product("WHEAT", 5)   # → ["BUY_PRODUCT", "WHEAT", 5]
```

Maximum **10 orders per turn** (`Actions.MAX_MARKET_ORDERS_PER_TURN`).

### Hire cost

Farm hand costs scale daily (Fibonacci-style). Use helpers for budgeting:

```python
from kaggriculture import cumulative_hire_cost, max_affordable_hires

cost_for_three = cumulative_hire_cost(3)
affordable = max_affordable_hires(current_cash, already_hired_today=2)
```

---

## 8. Manual actions (Level 5)

**Goal:** Build the action dict yourself without `ActionController`.

**Example:** [`examples/agents/05_manual_actions/main.py`](../examples/agents/05_manual_actions/main.py)

### When to go manual

- You want a **minimal single-file** submission (`build_submission.py -f single`)
- You need **custom multi-unit task assignment** the controller doesn't support
- You're learning exactly what the engine expects

### Structure

```python
from kaggriculture import Actions, CROPS

def agent(obs):
    player = obs["player"]
    farm = obs["farms"][player]
    private = obs["private"]
    day = obs["day"]

    return {
        "farmer": _plan_farmer(farm, private, day),
        "hands": [Actions.pass_action() for _ in farm.get("hands", [])],
        "market": _plan_market(farm, private, obs["market"], day),
    }
```

### Navigation

Units move one tile per turn on a cardinal direction:

```python
Actions.north()   # → ["NORTH"]
Actions.east()    # → ["EAST"]
Actions.pass_action()  # → ["PASS"]
```

To approach a target tile, compare positions and pick the axis with the larger delta (same logic as `ActionController.direction_towards`).

### Tile actions (when standing on the tile)

```python
Actions.water()
Actions.harvest()
Actions.plant_wheat()          # or Actions.plant("MELON")
Actions.fertilize()
Actions.dig()                  # remove weed
Actions.feed()                 # feed animal
Actions.drop()                 # drop inventory to shed (must be adjacent)
```

### Iterating the board

```python
tiles = farm["tiles"]  # tiles[y][x]
for y, row in enumerate(tiles):
    for x, tile in enumerate(row):
        if tile is None:
            ...  # empty, can plant
        elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
            age = day - tile["planted_day"]
            ...
```

---

## 9. Library helpers (Level 6)

**Goal:** Use pre-built analysis tools for smarter harvest timing, forecasting, and opponent modeling.

**Example:** [`examples/agents/06_helper_driven_harvest/main.py`](../examples/agents/06_helper_driven_harvest/main.py)

### Board flattening

Turns the raw 2D tile grid into queryable groups:

```python
from kaggriculture import flatten_board_state

board = flatten_board_state(farm["tiles"], current_day=day)

board.ripe_crops          # optimal-age crops
board.harvestable_crops   # any crop with yield > 0
board.thirsty_crops       # need water today
board.danger_crops        # about to become weeds
board.vacant_unlocked     # empty plantable tiles
board.total_vacant        # count
```

Each crop entry has `.x`, `.y`, `.crop`, `.yield_units`, `.config`, `.is_ripe`, etc.

### Peak yield detection

```python
from kaggriculture import check_max_yield_met

ready = check_max_yield_met({
    "plant_type": "WHEAT",
    "age_in_days": 4,
    "watered_bonus_days_count": 3,
    "fertilized_bonus_days_count": 0,
})
```

Use this to harvest one-time crops at exactly their peak instead of relying on controller heuristics.

### Yield forecasting

```python
from kaggriculture import forecast_crop_yield_trajectory

projections = forecast_crop_yield_trajectory(
    farm["tiles"], current_day=day, horizon_days=5,
)
# List of CropProjection with projected_yield per tile
```

### Market prediction

```python
from kaggriculture import (
    predict_upcoming_consumption_ticks,
    find_next_consumption_window,
    simulate_sell_slippage,
)

ticks = predict_upcoming_consumption_ticks(unlocked_shops, current_step=step)
slippage = simulate_sell_slippage("MELON", quantity=10, market_inventory=inv)
```

### Opponent analysis

```python
from kaggriculture import analyze_opponent_farm

profile = analyze_opponent_farm(
    opponent_tiles, current_day=day, market_prices=prices,
)
# OpponentProfile with crop groups, sabotage opportunities, etc.
```

### Pattern: override `plan_unit_action()`

Keep the controller for market logic but replace unit routing:

```python
class HelperHarvestController(ActionController):
    def plan_unit_action(self, unit_idx, farm, private, available_seeds, obs, claimed_tiles):
        # use flatten_board_state + check_max_yield_met to find best harvest
        ripe = self._find_ripe_tile(...)
        if ripe:
            return navigate_or_harvest(ripe)
        return super().plan_unit_action(...)  # fallback
```

---

## 10. Advanced / adversarial agents (Level 7)

**Goal:** Weaponize market mechanics against the opponent.

**Reference:** [`submissions/feed_squeeze/main.py`](../submissions/feed_squeeze/main.py)

### Techniques

| Strategy | Idea |
|----------|------|
| **Feed squeeze** | Buy wheat to starve opponent livestock |
| **Harvest front-running** | Sell into the market just before opponent's bulk harvest crashes price |
| **Slippage modeling** | Use `simulate_sell_slippage` / `simulate_buy_slippage` to size orders |
| **Opponent profiling** | `analyze_opponent_farm` to detect maturing crop groups |

### Key architectural choices

```python
class FeedSqueezeController(ActionController):
    def __init__(self):
        super().__init__(
            auto_sell=False,   # manual sell timing is critical
            ...
        )

    def plan_market_actions(self, farm, private, market, current_day):
        # completely custom — squeeze, front-run, hedge
        ...

    def act(self, obs):
        self._update_opponent_profile(obs)
        return super().act(obs)
```

### Run locally

```bash
python examples/run_feed_squeeze.py
python standoff/run_standoff.py -a feed_squeeze
```

---

## 11. Crop and animal reference

### Crops

| Crop | Type | Seed | Base price | Matures (days) | Max yield | Notes |
|------|------|------|------------|----------------|-----------|-------|
| WHEAT | One-time | $10 | $25 | 4 | 6 (4 unfert.) | Fast cash loop |
| CARROT | One-time | $20 | $35 | 3 | 4 (3 unfert.) | Quick turnaround |
| MELON | One-time | $80 | $250 | 10 | 6 | Highest peak value |
| TOMATO | Ongoing | $50 | $60 | 8+ | 4 harvests | Daily after day 8 |
| STRAWBERRY | Ongoing | $100 | $120 | 10+ | 4 harvests | Every other day |

Access configs in code:

```python
from kaggriculture import CROPS, Wheat, Melon

Wheat.time_to_max_yield      # 4
Melon.base_market_price      # 250
CROPS["TOMATO"].ongoing      # True
```

### Animals

| Animal | Cost | Product | Structure | Feed |
|--------|------|---------|-----------|------|
| GOOSE | $300 | EGG | COOP | Wheat |
| COW | $400 | MILK | PASTURE | Wheat |
| SHEEP | $500 | WOOL | PASTURE | Wheat |

### Economic constraints to remember

- **Shed capacity:** 100 non-seed items — harvest continuously or overflow
- **Hands reset daily** — hiring is a throughput decision, not permanent investment
- **Shops unlock every 3 days** — demand spikes follow a predictable schedule
- **Land expansion:** $1k / $2k / $4k for NE / SW / SE

---

## 12. Observation schema

Top-level keys in `obs`:

```python
{
    "player": 0,                    # your index (0 or 1)
    "day": 5,                       # 0–29
    "hour": 12,                     # 0–23 within the day
    "step": 132,                    # global turn 0–719
    "farms": [farm_p0, farm_p1],    # public state for both players
    "private": { ... },             # YOUR shed, seeds, inventories only
    "market": { ... },
    "town": { ... },
}
```

### `farm` dict (`obs["farms"][obs["player"]]`)

```python
{
    "money": 4500.0,
    "farmer": [4, 5],                # [x, y]
    "hands": [[3, 4], [5, 4]],      # hand positions
    "hires_today": 2,
    "unlocked_quadrants": ["NW", "NE"],
    "tiles": [                      # tiles[y][x]
        [None, {"kind": "PLANT", "crop": "WHEAT", ...}, ...],
        ...
    ],
}
```

### Plant tile fields

| Field | Meaning |
|-------|---------|
| `kind` | `"PLANT"` |
| `crop` | `"WHEAT"`, `"MELON"`, etc. |
| `planted_day` | Day the seed was planted |
| `watered_today` | Watered this turn |
| `consecutive_unwatered` | Days without water (≥1 + not watered today → weed risk) |
| `yield_units` | Current harvestable units on tile |
| `fertilized_until_day` | Last day fertilizer bonus applies (-1 = none) |

### `private` dict

```python
{
    "shed": {"WHEAT": 12, "MELON": 3},       # central storage
    "seeds": {"WHEAT": 5, "MELON": 10},      # plantable seeds
    "inventories": [                          # per-unit carried goods
        {"WHEAT": 2},                         # index 0 = farmer
        {},                                   # index 1 = first hand
    ],
}
```

### `market` dict

```python
{
    "prices": {"WHEAT": 28, "MELON": 245, ...},
    "inventories": {"WHEAT": 150, ...},       # market stock (affects price)
}
```

### `town` dict

```python
{
    "unlocked_shops": ["BAKERY", "PIZZA_SHOP"],
}
```

### Typed wrapper (optional)

```python
from kaggriculture import Observation

obs_typed = Observation.from_dict(obs)
obs_typed.my_cash
obs_typed.my_farm
obs_typed.opp_farm
```

---

## 13. Action reference

### Movement

| Action | Tokens |
|--------|--------|
| North / South / East / West | `["NORTH"]` etc. |
| Pass | `["PASS"]` |

### Field

| Action | Tokens |
|--------|--------|
| Water | `["WATER"]` |
| Harvest | `["HARVEST"]` |
| Plant | `["PLANT", "WHEAT"]` |
| Fertilize | `["FERTILIZE"]` |
| Dig weed | `["DIG"]` |
| Feed animal | `["FEED"]` |
| Care animal | `["CARE"]` |
| Collect fertilizer | `["COLLECT_FERTILIZER"]` |
| Drop to shed | `["DROP"]` |
| Pickup from shed | `["PICKUP", "WHEAT", 3]` |
| Place animal | `["PLACE", "SHEEP", 1]` |

### Market

| Action | Tokens |
|--------|--------|
| Buy seed | `["BUY_SEED", "MELON", 5]` |
| Sell | `["SELL", "MELON", 10]` |
| Hire hand | `["HIRE"]` |
| Buy land | `["BUY_LAND"]` |
| Buy animal | `["BUY_ANIMAL", "GOOSE", 1]` |
| Buy product | `["BUY_PRODUCT", "WHEAT", 5]` |

Use `Actions.water()`, `Actions.buy_seed(Plants.MELON, 5)`, etc. — they return the correctly formatted lists.

---

## 14. Development workflow

### Step-by-step

```bash
# 1. Start from an example
cp -r examples/agents/03_adaptive_crop submissions/my_agent

# 2. Edit the agent
#    submissions/my_agent/main.py

# 3. Smoke test (1 day = 24 turns)
python scripts/test_submission.py -a1 my_agent -a2 random -s 24

# 4. Full-season test vs a strong opponent
python scripts/test_submission.py -a1 my_agent -a2 shop_opportunist

# 5. Optional: HTML replay
python scripts/test_submission.py -a1 my_agent -a2 melon_rusher --render
# → opens match_replay.html

# 6. Round-robin benchmark (slow, ~40 matches)
python standoff/run_standoff.py -a my_agent -o standoff/my_agent_results.json

# 7. Package and validate
python scripts/build_submission.py -a my_agent

# 8. Submit to Kaggle
kaggle competitions submit kaggriculture -f build/submission.tar.gz -m "my_agent v1"
```

### Skill progression checklist

- [ ] Level 1: `01_basic_wheat_loop` runs without errors for 24 turns
- [ ] Level 2: Melon agent beats random over 720 turns
- [ ] Level 3: Adaptive crop switches based on prices
- [ ] Level 4: Custom market phases (expand early, liquidate late)
- [ ] Level 5: Manual agent waters and harvests without controller
- [ ] Level 6: Helper-driven harvest uses `flatten_board_state`
- [ ] Level 7: Standoff win rate > 50% against field

### Modifying an existing submission

1. Open `submissions/<name>/main.py`.
2. Identify the pattern (flags only? subclass? manual?).
3. Make one change at a time.
4. Re-run `test_submission.py` with `-s 24` after each change.
5. Run a full 720-turn test before standoff.

---

## 15. Common pitfalls

### Plants become weeds

If `consecutive_unwatered >= 1` and the crop is not watered today, it becomes a weed at end-of-day. The controller handles this with priority-0 emergency watering — if you go manual, check `danger_crops` every turn.

### Shed overflow

Shed holds **100** non-seed items. A full-board melon harvest can exceed this. Enable `auto_sell` or sell aggressively when `sum(shed.values())` approaches 80+.

### Planting without seeds

Market seed orders and planting happen in the same turn. The controller tracks a virtual `available_seeds` dict during `act()` so multiple units don't over-commit seeds.

### Late-season planting

Don't plant crops that won't mature before day 30:

```python
days_left = 30 - obs["day"]
if crop.time_to_max_yield > days_left:
    skip
```

### Slow `act()` functions

Heavy optimization per turn can exhaust the 60-second overage bank over 720 turns. Profile with standoff timing metrics (`max_ms`, `overage_seconds` in results JSON).

### Forgetting `agent()` export

Kaggle loads `main.py` and calls `agent(obs)`. A class instance alone is not enough — you need the top-level function.

### Wrong player index

Always index your farm with `obs["player"]`, not hardcoded `0`:

```python
farm = obs["farms"][obs["player"]]
```

Player 1 agents that read `farms[0]` will control the wrong board.

---

## Quick reference: which pattern should I use?

```
Just need something that runs?
  → Level 1: ActionController + agent()

Want more cash, same architecture?
  → Level 2: tune flags (crop, hires, sell margin)

Prices and shops matter?
  → Level 3: subclass act(), update target_crop each turn

Phased investment strategy?
  → Level 4: override plan_market_actions()

Need full control or minimal code?
  → Level 5: manual Actions

Harvest timing is critical?
  → Level 6: flatten_board_state + check_max_yield_met

Want to attack the opponent's economy?
  → Level 7: see feed_squeeze, opponent helpers
```

Start at Level 1, validate with `test_submission.py`, and move up only when you hit a ceiling. The examples in [`examples/agents/`](../examples/agents/) map 1:1 to each level.
