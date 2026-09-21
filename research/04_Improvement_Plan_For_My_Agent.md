# 04 — Improvement Plan For My Agent

Step-by-step technical roadmap to upgrade the existing Kaggriculture agent using methods learned from anonymous top agents. **No code has been applied** — these are planned injections only.

---

## 0. Current State Snapshot

**Framework strengths you already have:**

| Module | Capability |
|--------|------------|
| `ActionController` | Priority tile search, multi-unit claiming, shed adjacency nav |
| `market_planning.plan_market_actions` | Land → hire → margin sell → 5-seed top-up → animal buy |
| `helpers/market_prediction` | Exact demand ticks + sell slippage (+ lockstep opponent) |
| `helpers/opponent` | Crop groups, sabotage windows, opponent profile |
| `helpers/yield_check` / `tracking` / `solver` | Harvest readiness, board eval, discretization |

**Best verified submissions (from your learnings):**

- `market_velocity` — strongest new (~36k solo, 18/20 standoff) but below ~80k target  
- `shop_opportunist` — incumbent adaptive shop policy, hard to beat locally  
- Gap vs elites: capacity/slot economy, hire-by-unlocked, premium batching, projected DROP sells, collision timing

**Dominant failure modes in your audits:**

1. Conservative seed caps (5) and incomplete land fill  
2. Labor routing walks from daily shed spawn (naive one-task)  
3. Sell-all dumps without impact ranking  
4. Livestock branches that trap cash in shed animals  
5. No day-close 99-capacity guard; no terminal forced DROP

---

## 1. Target Architecture (After Upgrade)

Keep `ActionController` as the body. Add a **PolicyBrain** layer that sets phase knobs each turn, then run a **MarketRepair** post-pass.

```
obs
 → PolicyBrain.update(obs)          # phase, mix, hire target, sell mode
 → ActionController.act(obs)        # units (enhanced tasking)
 → MarketRepair.postprocess(obs, action)
      projected_drop_sells
      room_guard_99
      clamp_fillable
      impact_rank
      collision_or_premium_delay
      operating_reserve on buys
 → validate overplant PASS
 → return action
```

Base the first strong candidate on `market_velocity` + shop scoring from `shop_opportunist`, not on livestock-first branches.

---

## 2. Phase Roadmap

### Phase P0 — Safety & Slot Economy (1–2 days, highest ROI)

**Goal:** Stop silent score leaks (overflow, unfillable sells, cash insolvency).

#### P0.1 Operating reserve on all buys

**Inject into** `src/kaggriculture/actions/market_planning.py`

```python
OPERATING_RESERVE = 100

def plan_market_actions(..., operating_reserve: int = OPERATING_RESERVE):
    money = float(farm.get("money", 0))
    # ... after computing each buy cost:
    # if money - cost < operating_reserve: skip
```

Also thread `operating_reserve` through `ActionController.plan_market_actions`.

#### P0.2 Projected same-turn DROP into SELL availability

**Inject into** `ActionController.act` after unit actions are known:

```python
planned_drop = {}
for idx, unit_act in enumerate([farmer_act, *hands_act]):
    inv = inventories[idx] if idx < len(inventories) else {}
    pos = me["farmer"] if idx == 0 else me["hands"][idx - 1]
    if unit_act and unit_act[0] == "DROP" and Actions.is_shed_adjacent(tuple(pos)):
        for item, n in inv.items():
            planned_drop[item] = planned_drop.get(item, 0) + int(n)

# Pass planned_drop into plan_market_actions so SELL qty uses shed[item] + planned_drop[item]
```

#### P0.3 Day-close room_guard targeting 99

**New helper** `helpers/capacity_guard.py` (planned file):

```python
def room_guard_99(obs, market_orders, unit_actions) -> list:
    step = int(obs.get("step", 0))
    if step % 24 != 23:
        return market_orders
    # compute shed + carried + produced - consumed - planned_sells
    # if needed > 0: append/boost SELLs by price desc until fillable
    # never exceed 10 orders
    return market_orders[:10]
```

#### P0.4 Clamp unfillable SELLs

```python
def clamp_sells(projected_shed: dict, orders: list) -> list:
    avail = dict(projected_shed)
    kept = []
    for o in orders:
        if o and o[0] == "SELL":
            have = avail.get(o[1], 0)
            n = min(int(o[2]), have)
            if n <= 0:
                continue
            avail[o[1]] = have - n
            kept.append(["SELL", o[1], n])
        else:
            kept.append(o)
    return kept
```

#### P0.5 Overplant PASS validator

At end of `ActionController.act`:

```python
from collections import Counter
requested = Counter(a[1] for a in [farmer_act, *hands_act] if a and a[0] == "PLANT" and len(a) > 1)
seeds = private.get("seeds", {})
blocked = {c for c, n in requested.items() if n > int(seeds.get(c, 0))}
if blocked:
    def fix(a):
        return ["PASS"] if a and a[0] == "PLANT" and a[1] in blocked else a
    farmer_act, hands_act = fix(farmer_act), [fix(a) for a in hands_act]
```

**Acceptance:** solo cash variance down; shed overflow events → 0 in telemetry; no illegal plant turns.

---

### Phase P1 — Alpha Production / Labor / Sell Policy (3–5 days)

**Goal:** Match Architecture Alpha’s economic skeleton inside your controller.

#### P1.1 Configured crop mix + mix_switch_day

Replace single `target_crop` with mix quotas (champion defaults):

```python
EARLY_MIX = {"MELON": 14, "CARROT": 14, "WHEAT": 12}
LATE_MIX = {"CARROT": 20, "WHEAT": 20}
MIX_SWITCH_DAY = 13
MAX_ACTIVE = 40
```

**Inject into** `find_best_tile_for_unit` plant branch: choose crop that maximizes `deficit[c] / target[c]` among seeded crops, empties near shed center first.

#### P1.2 Adaptive ±3 staple shift from shops

Reuse Delta/Alpha shop pull:

```python
def adaptive_shift(mix, unlocked_shops):
    # PET_CAFE counts 2 toward carrot; any wheat shop +1
    # shift = min(3, abs(carrot_pull - wheat_pull)); move WHEAT↔CARROT
    return mix
```

#### P1.3 Hire-by-unlocked (not flat max_hires_per_day)

```python
HANDS_BY_UNLOCKED = {1: 6, 2: 9}
# In plan_market_actions:
target = HANDS_BY_UNLOCKED.get(len(unlocked), 0)
while current_hands < target and money - hire_cost >= operating_reserve:
    orders.append(Actions.hire())
```

Disable hire/land after day 26 (keep your `market_velocity` cutoff).

#### P1.4 Batched premium sells

```python
PREMIUM = {"STRAWBERRY", "MELON", "MILK", "WOOL"}
PREMIUM_BATCH = 8

def sale_qty(item, amount, day, terminal_day=29):
    if day >= terminal_day:
        return amount
    if item in PREMIUM:
        return min(amount, PREMIUM_BATCH)
    return amount
```

Replace `min_sell_margin` dump-all with: sell staples fully when `price >= base * margin`; sell premiums in batches; always sell if shed_total ≥ 82.

#### P1.5 Terminal return + DROP

In `plan_unit_action`, if `day >= 29 and hour >= 13` and unit carries goods: move to shed / DROP (mirror Alpha). Force harvest any held yield at day ≥ 29.

**Acceptance:** beat `market_velocity` solo mean; win rate vs `shop_opportunist` ≥ 50% over 20 swapped matches.

---

### Phase P2 — Microstructure Overlays (2–3 days)

#### P2.1 Impact ranking using your slippage helper

```python
from kaggriculture.helpers.market_prediction import simulate_sell_slippage

def impact_rank(obs, orders):
    inv = obs["market"]["inventory"]
    def score(o):
        if not (o and o[0] == "SELL"):
            return float("-inf")
        r = simulate_sell_slippage(o[1], int(o[2]), inv.get(o[1], 10000))
        return r.quantity * max(0, r.starting_price - r.ending_price)
    sells = sorted([o for o in orders if o and o[0] == "SELL"], key=score, reverse=True)
    rest = [o for o in orders if not (o and o[0] == "SELL")]
    return (sells + rest)[:10]
```

Optional: add Architecture Delta urgency term under rebalance regime.

#### P2.2 Collision guard (clone-gated)

Wire Overlay B/C from `08_Overlay_Patterns.md` after impact ranking. Start with clone_like gate to avoid over-holding vs dissimilar opponents.

#### P2.3 Post-drain preference

Using `predict_upcoming_consumption_ticks`: if next shop tick is within 1–2 turns and shed pressure &lt; 82, delay non-premium sells by 1 turn (your microbatch insight, with capacity safety).

**Acceptance:** higher average revenue per sold unit; fewer self-crashes when clone-matched.

---

### Phase P3 — Task Quality (routing)

#### P3.1 Dollar `loss_if_omitted` in candidate sort

Change `find_best_tile_for_unit` sort key from `(priority, dist)` to `(priority, -loss_estimate, dist)` where loss ≈ seed + expected_yield * price for missed water, animal cost for missed feed.

#### P3.2 Axis-alternating movement

```python
def direction_towards(self, src, target, unit_index=0):
    dx, dy = target[0] - src[0], target[1] - src[1]
    horizontal_first = unit_index % 2 == 0
    if horizontal_first and dx:
        return Actions.EAST if dx > 0 else Actions.WEST
    if dy:
        return Actions.SOUTH if dy > 0 else Actions.NORTH
    ...
```

#### P3.3 Idle-hand water rescue (Overlay D)

Only after P1 hire targets are stable — rescues amplify watering capacity of 6–9 hand openings.

---

### Phase P4 — Endgame Local Search (stretch)

Port Beta terminal planner concepts onto your own open-loop plan for steps 712–718:

1. Snapshot planned 7-turn unit schedule (even if greedy)  
2. Propose harvest→DROP suffixes for actors with nearby yield  
3. Accept only if no overflow and deposited value rises  

Skip full tape corpus until online agent plateaus above ~60k mean.

---

## 3. Submission Candidate Sequence

| Candidate | Base | Adds | Benchmark |
|-----------|------|------|-----------|
| `alpha_velocity_p0` | market_velocity | P0 guards | solo + 20-match |
| `alpha_velocity_p1` | p0 | mix/hire/batch/terminal | solo + 20-match |
| `alpha_shop_hybrid` | p1 | shop_opportunist scoring + impact rank | standoff vs shop_opportunist |
| `alpha_collision` | hybrid | clone collision guard | clone-heavy panel |
| `alpha_terminal` | collision | 712–718 search | final panel |

Keep latency &lt; 30 ms mean; hard abort to sell-fallback if &gt; 200 ms.

---

## 4. Concrete Injection Map (Files → Changes)

| File | Planned change | Phase |
|------|----------------|-------|
| `actions/market_planning.py` | reserve, hire-by-unlocked, batch sells, bulk seeds to mix deficit, planned_drop | P0–P1 |
| `actions/controller.py` | terminal DROP, movement desync, loss-aware sort, postprocess hook, overplant PASS | P0–P3 |
| `helpers/market_prediction.py` | expose `impact_score` wrapper around slippage | P2 |
| `helpers/opponent.py` | add `clone_like()` + feed collision gate | P2 |
| **new** `helpers/capacity_guard.py` | room_guard_99, clamp_sells, projected_shed | P0 |
| **new** `helpers/phase_brain.py` | mix switch, investment cutoff, sell mode | P1 |
| `submissions/<new>/main.py` | thin entry wiring PolicyBrain + controller | each phase |

Do **not** rewrite `ActionController` from scratch; patch knobs and postprocess.

---

## 5. Telemetry To Add (Before Claiming Wins)

Log per episode:

- shed overflow units lost  
- unfillable SELL slots burned  
- hire count by day  
- mean impact score of sells  
- collision holds / releases  
- terminal carried goods at step 718  
- latency p50/p95  

Kill any change that raises overflow or latency overage.

---

## 6. Explicit Non-Goals (This Roadmap)

- Building a 13×719 tape corpus (defer)  
- Native `.so` policies  
- Full livestock mill as primary (bootstraps failed in your audits)  
- Modifying files in this research phase (planning only)

---

## 7. First Coding Session Checklist (When You Approve Implementation)

1. Create `helpers/capacity_guard.py` with projected_shed / clamp / room_guard_99  
2. Patch `market_planning.py` with reserve + planned_drop + batch premiums  
3. Patch `controller.act` postprocess + overplant PASS  
4. New submission `alpha_velocity_p0` cloning `market_velocity` entry  
5. Run solo 720 + 20-match standoff; record telemetry  
6. Only then proceed to P1 mix/hire

Say the word when you want these applied to the repo.
