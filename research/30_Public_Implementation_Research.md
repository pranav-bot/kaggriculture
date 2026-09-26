# Kaggriculture: Public Implementation Research
### Compiled for trigger design — opponent response, predictive returns, shop prediction

**Date:** 2026-09-26
**Sources:** 101 public Kagaggle notebooks downloaded via the Kaggle kernels API, 46 unique agent
sources extracted (10.3 MB), 66 markdown write-ups, the official engine `AGENTS.md`, and the
`Beiciccc/Kaggriculture` 35-submission experiment ledger.
**Corpus location (not committed):** `/private/var/folders/.../opencode/kag/`
→ `nb/` (notebooks) · `agents/` (46 deduped agent sources) · `md/` (write-ups) · `extracted/` (raw)

---

## 0. TL;DR — the architecture everything converges on

```
┌─ ROUTER ────────────────────────────────────────────────────────────┐
│ pick one of N recorded 719-step action TAPES from public state      │
│ (almost always: the ordered first TWO town shops, latched at t=144)  │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌─ CHASSIS ───────────────────────────────────────────────────────────┐
│ replay tape[step]; align hands to len(farm.hands); repair weeds      │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌─ REACTIVE OVERLAYS (~10 to ~50, each a thin `agent` wrapper) ───────┐
│ read public obs → edit ONE kind of decision → pass to next layer     │
│ every layer ends with `agent = globals().pop('agent')`               │
└────────────────────────────────────────────────────────────────────┘
```

**This is the single most important finding.** Nobody in the public meta is running
PPO/DQN/MCTS end-to-end. The winning pattern is *offline tactical memory + public-state
branch + tiny closed-loop market control*. The RL-looking layer is not the policy — it is
the **overlay stack**, and the marginal wins all come from market-timing overlays.

**The meta has converged.** Rayk's ledger: *"A single field hash appeared in 144 episodes
from 40 teams. Most ranks 3-20 were 99-100% identical on field actions."* Mamarin's audit:
top players are 100% byte-identical across games. The competitive edge is *who sells first,
into a shared book that falls off a cliff*.

**Three facts that reframe everything** (all verified against the engine — see §1.5):
1. **Ladder rating is W/L/T only. Coin margin is worth literally nothing.** C94 beat C90/91/92
   **6-0 with a mean margin of $27**.
2. **Crossing `I0 = 10,000` on a premium item is a ONE-WAY DOOR.** `if price > 1:
   inv[item] += 1` — a $1 sale adds no supply, so the only recovery is town consumption at
   1 unit / 4 turns. One replay variant crossed I0 on MILK, ratcheted to $1, and never
   recovered. Losing the slot race can cost the season, not a few dollars.
3. **`step % 4` residue is irrelevant to price — proven causally.** Moving a sell block to
   hour 4 (*same* residue) cost exactly the same as hours 1/2/3/6. A shop tick is 1 unit
   against a 10,000-unit equilibrium. Most "sell before the town tick" heuristics in the
   public meta are optimising a 0.01% effect.
4. **A full shed blocks `BUY_ANIMAL` and `BUY_PRODUCT` outright.** Shed headroom is a
   *purchase gate*, not storage hygiene. ~66% of requested SELL orders never execute because
   the shed binds, not the market.

---

## 1. Hard facts that constrain everything

### 1.1 Scoring — this changes your objective function

| | |
|---|---|
| Terminal reward | `farms[player]["money"]` at step ≥ 718 |
| **Ladder rating** | **Win / Loss / Tie only. Coin margin is worth literally nothing.** |

Rayk: *"C94 beat C90/C91/C92 **6-0**, despite a mean margin of only 27. That is exactly the
distinction this notebook targets: maximize the chance of winning the matchup, not the size
of an already-won bank."*

And the corollary trap: *"Increasing margins against agents it already beat could not
reliably move a sub-3000 rating."*

**Consequence for trigger design:** any trigger must be gated on
`P(win | trigger) > P(win | no trigger)`, not on `E[margin | trigger]`. Optimizers that
maximize mean cash will drift toward high-variance strategies that lose more often.

### 1.2 The price cliff (computed from the authoritative engine params)

`_R37_MARKET_PARAMS`, read from `obs['market']['params']` at runtime. `price(I) = base ±
amp·shape(|I − I0|)`, `I0 = 10000`:

| product | base | T | @+50 | @+100 | @+200 | **units to $1 floor** | glut shape |
|---|---:|---:|---:|---:|---:|---:|---|
| STRAWBERRY | 120 | 100 | 24 | **1** | 1 | **62** | linear (steep) |
| WOOL | 200 | 105 | 55 | **1** | 1 | **59** | sq |
| MILK | 160 | 122 | 55 | **1** | 1 | **76** | linear |
| MELON | 250 | 300 | 225 | 150 | **1** | **158** | sq |
| TOMATO | 60 | 200 | 42 | 35 | 24 | 529 | sqrt |
| CARROT | 35 | 450 | 27 | 23 | 19 | 842 | sqrt |
| FERTILIZER | 100 | 200 | 90 | 80 | 60 | 493 | linear |
| EGG | 50 | 332 | 43 | 42 | 41 | never | log |
| WHEAT | 25 | 400 | 22 | 21 | 21 | never | log |

**One field's worth of surplus (≈60–160 units) zeroes out every premium book.** That is the
entire game. WHEAT and EGG are ballast — they never crash.

### 1.3 Town demand — and the two products with no demand at all

Shop consumes 1 unit of each product it wants every 4 turns (**6/day**; **12/day** if it is a
single-product shop). Town Center takes 1 of each non-fertilizer product every 24 turns
(**1/day**, flat all season).

| product | per shop/day | + centre | 8 shops all-on | days to absorb 100u surplus |
|---|---:|---:|---:|---:|
| WHEAT | 30 | 1 | 248 | 0.40 |
| STRAWBERRY | 24 | 1 | 200 | 0.50 |
| CARROT / MILK | 18 | 1 | 152 | 0.66 |
| TOMATO / EGG / WOOL | 12 | 1 | 104 | 0.96 |
| **MELON** | **0** | 1 | **8** | **12.50** |
| **FERTILIZER** | **0** | **0** | **0** | **never** |

Two decisive consequences:
- **MELON has essentially no sink.** It is a day-10 capital spike, not a cash flow. Never
  hold melon; dump it.
- **FERTILIZER has no sink at all.** Any price > $1, sell. Every surviving animal makes 1
  fertilizer/day, so this is the early-game engine.

### 1.4 Market microstructure — the thing nobody tells you

**Turn order** (verified in `kaggriculture.py` `interpreter`):
```
_process_market(...)     # both players' slot-k orders, per-unit lockstep
_town_consume(...)      # shops every 4 turns, centre every 24  -> then _refresh_prices
_decay_plants(...)
_end_of_day(...)        # hands wiped, all carried inventory dropped into the shed
```
Because the town consumes **after** the market clears, a sell on a residue-0 step is priced
against the **pre-consumption** inventory. See §1.6 — this turns out to kill the most
commonly-assumed timing rule in the competition.

Both players' `market` lists are **settled together, slot by slot, one unit at a time at the
same quote**, up to 10 orders per turn per player.

```
slot 1:  you SELL MILK 6   ↔  rival SELL WHEAT 30  → your 6 milk clear at the pre-glut price
slot 2:  you SELL WHEAT 20  ↔  rival SELL MILK 12  → the rival's milk clears into YOUR glut
```

**Your list is an order book. Position decides who eats whose glut.** And: *"Every sale is
also denial. When I removed 103 strawberries from a route, I lost $5.5k of revenue **and the
rival gained $5.5k**."*

### 1.5 ⭐⭐ The causal picture, corrected by engine-instrumented replay
This section **overturns** the naive reading of §1.2-1.4. It was established by replaying the
real tapes through the actual engine with `_commit_unit`/`_do_hire`/`_do_buy_land`
monkeypatched to record which orders *actually* commit, then intervening causally
(relocating/deleting tape blocks and re-replaying). All three claims below were independently
re-verified against `kaggle_environments/envs/kaggriculture/kaggriculture.py` and against
`src/kaggriculture/env/items.py` in your own repo.

#### (a) ⭐ `step % 4` residue is IRRELEVANT to price. Stop optimising it.
Per-item mean realised price is flat across all four residues (WHEAT $47.2 / $46.9 / $47.0 /
$47.4 — a 1% spread). **Causal proof:** tape_0's hour-0 SELL block was relocated to hour 4 —
*also* residue 0, so the town-consumption phase is byte-identical — and it cost **exactly the
same −$21,933** as relocating to hours 1, 2, 3 and 6.

The mechanism is mechanical: `_town_consume` runs *after* `_process_market`, so a residue-0
sell is priced against pre-consumption inventory; and over a 4-step window the inventory
barely moves anyway (a shop tick is **1 unit** against an `I0` of **10,000**). A tick is
0.01% of equilibrium.

> **Every "sell just before the town tick" heuristic in the public meta is optimising a
> 1-unit effect against a 10,000-unit denominator.** The `preempt_phase_shift` overlays,
> the `_v17_town_demand_at` gate, and the `step % 4 == 1` sell batches found across the corpus
> are all chasing noise.

#### (b) ⭐ The hour-0 SELL sweep is a SHED FLUSH, not a price trade
| variant | final money | revenue | steps with shed ≥ 90 |
|---|---:|---:|---:|
| baseline (hour 0) | **$161,553** | $201,694 | 1 |
| → hour 3 | $139,620 (−$21,933) | $179,761 | 6 |
| → hour 4 (same residue!) | $139,620 (−$21,933) | $179,761 | 7 |
| → hour 8+ | $92,553 (−$69,000) | — | 11 |
| deleted entirely | $52,553 (−$109,000) | $91,934 | — |

**Residue is irrelevant; hour is not** — and only because of shed pressure. Past ~hour 6 the
shed is full when the day's harvest lands and **overflow is silently discarded**.

#### (c) ⭐⭐ A FULL SHED BLOCKS `BUY_ANIMAL` AND `BUY_PRODUCT` ENTIRELY
This is the strongest reason to flush the shed, and it is not mentioned anywhere in the public
write-ups. From `_commit_unit`:
```python
if op == "BUY_PRODUCT":
    if sum(private["shed"].values()) >= shed_capacity: return False
if op == "BUY_ANIMAL":
    if sum(private["shed"].values()) >= shed_capacity: return False
# note: BUY_SEED has NO such check -- seeds live in their own slot
```
**A shed at 99 units means you cannot buy another cow.** Shed headroom is therefore not a
storage optimisation — it is a *purchase gate*. This inverts the priority: the hour-0 flush is
what makes the morning's `BUY_ANIMAL` affordable, which is worth far more than any price
timing.

#### (d) ⭐⭐ The market ratchet — crossing `I0` is a ONE-WAY DOOR
```python
if op == "SELL":
    private["shed"][item] -= 1
    farm["money"] += price
    # Sales at $1 do not increase market supply.
    if price > 1:
        market["inventory"][item] += 1
```
Once inventory crosses `I0 = 10,000` the price collapses toward $1. Because **a $1 sale adds
no supply**, further selling cannot make it worse — but the only route back is town
consumption, which at 1 unit per 4 turns against a 10,000 overshoot takes **~11 hours of every
shop consuming that product simultaneously**. In the hour-8 variant MILK crossed `I0`, the
price ratcheted to **$1**, and never recovered. **The tape loses the season on that one step.**

Demonstrated wool divergence across tapes on identical mechanics:
| tape | wool realised | final bank |
|---|---:|---:|
| tape_0 (sells more, earlier) | **$102/unit** | — |
| tape_3 / tape_8 (sells less) | **$225/unit** | $51,504 |

**Selling less is worth more than selling more**, because the marginal unit sells into a book
you already crashed. Every "liquidate harder" overlay in the corpus is exposed to this.

#### (e) Only ~34% of requested SELL orders ever execute
Requested SELL orders spray median ~7 units, max 77 — far beyond the 100-unit shed. **The
shed, not the market, is the binding constraint ~66% of the time.** Executed units concentrate
even harder than written ones: 46% land at hour 0 (vs 15% of written orders).

#### (f) What survives from the naive reading
- **Slot ordering against the rival still matters enormously** — but for the *ratchet* reason
  (d), not the tick reason (a). Losing the slot race pushes inventory over `I0` and the loss
  is permanent. This is *why* `_v44y_lockstep` best-response ordering and the `r36_debts`
  reservation are the highest-value market layers in the corpus.
- **Keeping premium inventory below `I0` all season is the dominant objective.** That is what
  `r36_debts` does: it pulls sales forward precisely so cumulative supply never accumulates
  past equilibrium.
- **METERED batches still matter** (§1.2's price cliff), for the same ratchet reason: the cliff
  is at ~60-160 units of *cumulative* surplus, and a batch is safe only if cumulative supply
  stays under it.

### 1.6 Other engine facts that repeatedly decide games
- `SELL` only sees the **shed**. `HARVEST` puts goods in a *unit's* hands. You must `DROP`
  while shed-adjacent (tiles `(4,4) (5,4) (4,5) (5,5)`) or the sale is unfunded.
- Shed cap = **100** non-seed items; overflow **destroyed** at end of day. Hands reset to the
  shed at midnight.
- Hired hands are a **daily lease**: reset at hour 0, cost `fib(n)` for the n-th hire *today*.
  8 hands ≈ $54/day, but #12 and #13 cost **$144 + $233 = $377/day**. SL2 (one fewer hand on
  non-harvest days) was `+15/−0`.
- `CARE` is most of an animal's output: a cared cow gives **3 milk** not 1, sheep **4 wool**,
  goose **2 eggs**. Unfed on its production day loses the whole bank.
- Step **718 executes; index 719 does not.** (Rayk; delayed his terminal controller 712→717.)
- The engine version matters enormously: on the stale 1.29.3 build a mirror match of the #1
  public agent ends with **$2 per farm**; on 1.32.7 it banks ~$100k.

---

## 2. THEME 1 — Dynamically adjust to opponent actions

### 2.0 The ranked ladder (0 → 8), one line each

| Lvl | Technique | Where | Key idea |
|---|---|---|---|
| 0 | **No opponent model** | `lucifer19`, `kaitofukami_25-27`, `salemali7` | Get self-inflicted market impact exactly right *first*. `lucifer19` says so in code: *"It predicts no opponent action and uses no identity."* |
| 1 | **Opponent-blind route router on a coarse clock** | `thomastschinkel` 74.5 / 93.8 / 95.5 | Rival → 21 public numbers → embedded decision tree → swap the whole 719-step tape every 144 (or 72) turns |
| 2 | **Population-histogram clone gate + add-and-repay shift** | `boatlee_v14`, `romanrozen` | 13-number signature, threshold 6, sell **2×** tomorrow's premium qty *today*, book a debt |
| 3 | **Multi-checkpoint AND-latch + single-commodity relay** | `boatlee_v16-rc2` | Threshold 8, sampled at 3 checkpoints, AND-ed permanent; 3-turn FERTILIZER relay anchored to a known `%24==17` dump turn |
| 4 | **⭐ Calibrated mirror probability + exposure slot-hoisting** | `kaitofukami_159-160` | The **only logistic regression** in the corpus; 3 nested hystereses; counter = slot index, quantities untouched |
| 5 | **Behavioural clone detection (no board reads) + horizon escalation** | `thomastschinkel_2945`, `prvsiyan`, `arsgorynich` | Cash-trajectory probes; earned irreversible escalation `8 → 24` |
| 6 | **⭐ Recovered rival sale stream → fitted reservation horizon** | `thomastschinkel_2945` + 5 shared copies | Turn "is the rival selling?" into "**by how many turns do they beat me?**" → reserve that many |
| 7 | **Recorded-opponent libraries** | `thomastschinkel_2945` | `CT_TABLE` (fingerprint→counter-plan) and `_V92_P` (soft-DTW NN over 64 shop-pair cells) |
| 8 | **⭐⭐ Maximin best-response ordering** | `prvsiyan` `_v44y_*`, `_cxd_reorder` | The opponent is **inside the objective function**, not inside a gate |

### 2.1 The ladder in detail

#### Level 0 — price threshold + opponent animal count (what most heuristics do, incl. your `agent_final.py`)
```python
milk_oversupply = (milk_price < 145 or opp_cows >= 3)
wool_oversupply = (wool_price < 180 or opp_sheep >= 2)
```
Two numbers, one turn late by construction.

Three of the nine highest-profile public files have **no opponent model at all**, and that is
not a gap — `lucifer19/night-harvest` states its own scope:
> "It predicts no opponent action and uses no identity. […] The score is the revenue lost
> versus hypothetically executing every unit at the current quote."

```python
execution_revenue = current_quote
for offset in range(1, quantity):
    execution_revenue += float(_market_price(item, current_inventory + offset))
flat_revenue = float(quantity) * current_quote
return max(0.0, flat_revenue - execution_revenue)
```
It then permutes *existing* SELL orders among *existing* slots and never creates, deletes or
resizes an order. **This is the base layer the entire 2945-farm inherits** — get it right
before modelling anyone. `kaitofukami_25-27` adapts only to
`configuration["townCenterSellInterval"]` — public, non-adversarial, identical for both seats.

#### Level 1 — opponent-blind route router on a coarse clock
`thomastschinkel_93.8:_features(obs)` reduces the rival to **21 public numbers**: money,
8 population counts, **9 standing `yield_units`**, weeds, empty tiles, quadrants. `yield_units`
is a public read on the rival's *un-sold* inventory. Zero-padded to 100 dims, walked through an
embedded binary tree `_choose(block, x)` (`block = turn // 144`), re-decided every 144 turns.

`95.5` adds two novel tree primitives:
```python
elif feature==-2:
    node=yes if previous==int(cut) else no    # tests the RUNNING route
else:
    return previous if route<0 else route     # leaf may DECLINE to switch
```
`feature == -2` is a **state-dependent branch** — "am I currently on route *k*?" — which is how
a decision tree encodes hysteresis. A leaf with `route < 0` **declines to switch**, so route
changes are opt-in per block.

`74.5` has the cleanest irreversibility guard in the corpus:
```python
def _switch_ok(self, target, turn):
    """A switch is legal only onto a tail identical to the current one so far."""
    for t in range(turn):
        if self.R[self.cur][t] != self.R[target][t]: return False
    return True
```
The candidate must be **byte-identical to the current route for every past turn** — you can
never fork into a path that would have decided differently earlier.

#### Level 1 — weighted L1 clone distance over a public farm signature
`tetsutani/adaptive-farming-strategy-for-kaggriculture` → `tetsu_main.py:223-254`
```python
def _public_signature(farm):          # 11 counters, first-match-wins crop→animal→kind
    return (len(farm["hands"]), len(farm["unlocked_quadrants"]),
            tuple(counts[k] for k in sorted(counts)))

def _clone_distance(obs):             # 0 == pixel-identical farms
    left, right = _public_signature(farms[0]), _public_signature(farms[1])
    return (abs(left[0]-right[0]) + 3*abs(left[1]-right[1])
            + sum(abs(a-b) for a,b in zip(left[2], right[2])))
```
`_PREEMPT_MAX_CLONE_DISTANCE = 6`. Quadrant count weighted ×3 (land spread is the strongest
family signal).

#### Level 2 — tile-overlap similarity
`prvsiyan` → `_r37_similarity(obs)`: fraction of **non-empty** tiles with matching
`(crop, animal)`, requiring identical `unlocked_quadrants` and ≥8 non-empty tiles.
Threshold `>= .90` / `>= .95`.

#### Level 3 — positional clone detection over a 6-sample history
`prvsiyan` → `_race_clone` (line 5794): 6-sample rolling history of `_race_positions_equal`
(farmer + hand positions byte-identical), **≥4 of 6 true AND `_r37_similarity >= .95`**.
Plus a **mirror gate**: `|rival.money − own.money| < 0.5` at step 1 ⇒ the rival ran our exact
opening.

#### Level 4 — behavioural family fingerprinting (latching, one-way)
`tetsutani` → two *mirror-image* detectors, keyed on two features only:
```python
# R5 family = wool/sheep specialist
if sheep >= 4 and cows <= 3:            → latch True   (earliest step 24)
# MD family = milk/fertilizer-heavy cow specialist
if (quadrants >= 2 and cows >= 4 and sheep <= 2) or cows >= 9:
                                            → latch True   (earliest step 160)
```
Each is a **one-way latch**: once true, never re-checked. Reset only on `step == 0` or a
step regression. Reaction differs per family — R5 gets 50% pre-sell at `step+3` with a town
gate; MD gets **200%** pre-sell at `step+1` with no gate.

#### Level 5 — ⭐ RIVAL SALE RECOVERY (the single most valuable idea in the meta)
The opponent's sales are **publicly inferable** every turn, exactly:
```
rival_sold[p] = inv'[p] − inv[p] + town_draw[p] − own_sold[p]
```
valid above the $1 floor. Implementations: `prvsiyan` `_V9_RACE` (3853), `_or2_draw`
(4753), `_v9_town_draw` (3794); `tetsutani` `_V17_R5_MARKETS` / `_V17_MD_MARKETS` (a
*recorded* rival tape indexed by step).

`town_draw` is fully determined and must be modelled exactly:
```python
# every 4th turn each shop consumes 1 of each product (2 if single-product)
# every 24th turn the town centre takes 1 of each
```
Once you can subtract it, the residual is *pure opponent intent*. This turns the shared
market from noise into a signal, and it is the precondition for everything in Level 6-7.

#### Level 6 — opponent SALE-STREAM LIBRARY (shop-conditioned nearest-neighbour)
`prvsiyan` → `_V92_P_BLOB` (553 KB) + `_V92_P_INDEX` (64 shop-pair slices) + `_v92_p_pair()`.
Structure: **2,398 recorded rival sale streams, 438,301 `(tick, item)` events**, indexed by
the **ordered first-two-shop pair** (8×8 = 64 slots). Ticks are varint-delta encoded
(`255` escape → u16 absolute). Runtime: nearest-neighbour match of the rival's *recovered*
sale ticks (±1 turn, 240-turn window) against the streams for **our** first-two-shops; if the
winning stream sells ≥4 units of MILK/WOOL/STRAWBERRY in the next 2 turns, sell our own
planned lots now.
`thomastschinkel`'s v9/4 rebuilt this as **PREDICT** against **451k sale events from 1,000
recent top-25 games**.

#### Level 7 — exact two-player best-response ordering
See §3.3. `_v44y_lockstep` literally simulates the opponent's order list against yours.

### 2.1bis Levels 4-8 — the advanced end, in detail

#### Level 4 — ⭐ calibrated mirror probability (the only ML in the corpus)
`kaitofukami_159-160` is the only agent that converts "am I mirrored?" into a **probability**.
Distance uses ~100 rich features with hand-set weights — farmer position ×2, hand count ×3,
hand positions ×1, quadrant symdiff ×4, and a per-cell 7-field tile token
(`kind, crop, animal, growth, yield_units, fertilizer_available, cared_today`).

```python
def _mirror_probability(distance, money_gap, board_streak, step):
    values = (math.log1p(max(0.0, distance)), float(distance == 0),
              math.log1p(max(0.0, money_gap)),  float(money_gap <= 5.0),
              min(max(0,board_streak),96)/96.0, min(max(0,step),718)/718.0,
              float(step >= 480))
    logit = _MODEL_INTERCEPT + sum(c*((v-m)/s) for v,m,s,c in
              zip(values, _MODEL_MEAN, _MODEL_SCALE, _MODEL_COEFFICIENTS))
    return 1.0/(1.0+math.exp(-min(35.0, max(-35.0, logit))))
# _MODEL_INTERCEPT = 1.2457…,  _MODEL_THRESHOLD = 0.8065…
```
**The features are all shape-of-gap statistics, not raw counts** — that is why the weights are
learnable at all. Frozen standardisation (`_MODEL_MEAN`, `_MODEL_SCALE`), fit offline.

**Three nested hystereses:**
1. `board_streak` — only *consecutive* `distance <= 2` turns feed the classifier.
2. `latched` — one-way; needs **8 consecutive diverging turns** to clear.
3. `mode` (`"mirror"`/`"open"`) — decided **once** at `step >= 48`, never revised. Even a
   successful un-latch cannot change the strategy *class*.

The counter is **slot hoisting with quantities preserved**:
```python
score = (1.0 + exposure.get(item,0.0)) * _GLUT_WEIGHT.get(item,1.0) \
        * max(1.0, price) * math.log1p(quantity)
# _GLUT_WEIGHT: STRAWBERRY 2.0, MELON 3.6, MILK 2.0, WOOL 3.2, EGG 1.5, TOMATO 1.3, rest 1.0
remainder = [o for o in market if not (o[0]=="SELL" and o[1] in targeted)]
action["market"] = (sells + remainder)[:10]
```
Interpretation: **sell hardest into the products the rival is most glutted with**, because that
is where their own sale will crash the price. Animals contribute `1.0 + yield_units` — the
animal will yield again, so the rival's *future* supply is counted too.

#### Level 5 — behavioural clone detection with no board reads
Three detectors here read **cash, not tiles** — strictly more information than the board:
```python
# EXP288 mirror gate: |rival.money - own.money| < 0.5 at step 1  =>  horizon 8 -> 24
```
```python
# _r44 cash probe: execute an all-SELL action, measure OUR cash delta, check the
# rival's delta matches within max(5.0, 5%) of a >= $100 probe, gated on .90 similarity
if st['money'] is not None and st['probe']>=100 and _r37_similarity(obs)>=.90:
    own=money[0]-st['money'][0]; rival=money[1]-st['money'][1]
    if own>0 and rival>0 and abs(own-rival)<=max(5.0,.05*st['probe']): st['matched']=True
```
This detects *replay-of-us*, not merely *similar-to-us*. Window `[336, 648)` — days 14-27.

**`_race_lost` — the subtlest temporal discriminator in the corpus:**
```python
# every member of this lineage sells at the scheduled turn, one turn early (sale lead) or up to
# four turns early (the base reservation): only a sale further ahead of the plan is a race
if any(planned(t) for t in range(step-1, min(719, step+5))):  continue
if any(planned(t) for t in range(step+5, min(719, step+24))): return True
```
Rival sold something we hold, tape has no sale in `[step-1, step+5)` but *does* in
`[step+5, step+24)` → permanent escalation. The `[step-1, step+5)` exclusion is calibrated to
*this lineage's own* sale-lead and base-reservation widths, so it only fires on sales that
genuinely jumped the queue.

#### Level 6 — ⭐ recovered rival sale stream → fitted reservation horizon
The identity is written down exactly once (`thomastschinkel_2945:3738`) and implemented 5×.
**Two subtleties make it exact:**
```python
# own_sold is SHED-CLAMPED, not the raw order quantity:
n = min(max(0,int(o[2])), max(0, stock.get(o[1],0) - own.get(o[1],0)))
# an un-fillable SELL never entered the market, so it must NOT be subtracted
# from the rival's inferred sale
```
```python
if prev["prices"].get(item, 0) <= 3:  continue   # conservative stand-in for the $1 floor
```
(OR2 uses the exact `> 1`; below ~$3 the curve is too flat for attribution to be anything but
noise.)

**The novel step is converting a rival sale into a timing parameter:**
```python
planned = _v9_planned_sells(tape, item, t)     # scan +-30 turns of our own tape
after  = [s for s in planned if s >= t]
before = [s for s in planned if s < t]
if not after or (before and t - before[-1] < V9_RACE_GAP): continue
st["lead"] = max(st["lead"], after[0] - t)     # how many turns AHEAD did the rival quote?
horizon = clamp(lead + 12, 40, 48)             # -> reservation depth
```
**Its tuning record is in the file, and shows a real optimum with a real cost:**
> 6/4 → 40/12 wins 73-7; 44/12 beats 40 head-to-head but drops the shallow baseline to
> **65-15**; **constant 48 collapses to 14-26** — *"because a horizon past the rival's next lot
> gives up town-demand recovery for nothing."*

That is the most useful calibration data point in the corpus for a race horizon.

#### Level 7 — recorded-opponent libraries
**`CT_TABLE`** — a fingerprint-keyed counter-plan library:
```python
CT_TABLE = {
  (979.0, 9989): ((7,0,("BUY_PRODUCT","WHEAT",5)), (7,1,("SELL","WHEAT",5)),
                  (8,0,("BUY_PRODUCT","WHEAT",5)), (8,1,("SELL","WHEAT",5))),
  (33.0, 9990):  ((3,0,("BUY_PRODUCT","WHEAT",20)),(3,1,("SELL","WHEAT",20))),
}
# key = (round(rival.money,3), int(market.inventory.WHEAT)), computed once at step == 2,
# comment-asserted UNIQUE over 4,604 recorded games
```
The action is **slot co-location** (`market.insert(slot, o)`) — "trade alongside their own early
wheat orders in the same market slots." `_V93_ROUTE_BY_RIVAL = {(229.0, 9989): 128}` is the
same idea as a route swap instead.

**`_V92_P`** — 64 cells (the 8×8 first-two-shops grid) of recorded rival sale streams, matched
by **soft-DTW nearest neighbour with ±1-turn tolerance**:
```python
score = m - 0.5*f - 0.5*miss        # hit, false-positive, unexplained-observation
# 240-turn window, best-1, LEAVE-ONE-OUT BY EPISODE PARITY
```
Action is a **quorum**: "the matched opponent is predicted to dump ≥4 units within 2 turns" →
pull our own next-48-turn planned sales forward. `_V92_Q` is the deployed form: inverted index
`index[(tau,i)] -> [candidates]` with incremental online scoring (hit `+1.0`, false positive
`−1.0`, near-miss `+0.5`), closed only after the `±1` neighbourhood is fully observed.

**`_hd2_ev` uses the similarity verdict to change a *purchase*, not just a sale:**
```python
if similar:   # _r37_similarity(obs) >= 0.9
    existing[1-seat][d] += u      # credit the rival with this supply
# else: treat the rival's future supply as ZERO — a deliberate optimistic-for-us bias
```

**OR2 goes furthest on inference** — it recovers the rival's *harvest* from tile-yield diffs
(ongoing crops `TOMATO`/`STRAWBERRY` excluded because their `yield_units` resets on pick
rather than the tile disappearing) and combines with the sales identity to build an
**inferred rival shed** entirely from public data.

#### Level 8 — ⭐⭐ maximin best-response ordering
See §3.3 for `_v44y_lockstep` / `_v44y_reorder`. The opponent's order book is **our own order
list** (`opp = [list(o) for o in orders]`) — licensed entirely by the clone gate, since a
`.95`-identical board means the rival is running our tape.

The `_cxd_reorder` cell generalises to a *list* of rival models and switches max → **min**:
```python
_CXD_BUDGET = 800
margins = [_v44y_factor_margin(m, inv0, stock, params) for m in models]
def margin(cand): return min(f(cand) for f in margins)      # MAXIMIN
```
**The only maximin / regret-minimisation construction in the corpus**, with an explicit
800-evaluation cap. This is the natural extension if you have more than one credible rival
model.

### 2.1b Five different hysteresis implementations, increasing sophistication
1. **The debt ledger itself** — `state["due"]` non-empty blocks a second shift (`boatlee_v14`).
2. **3-checkpoint AND-latch** — three chances to veto, no oscillation (`boatlee_v16-rc2`).
3. **One-way `latched` + 8-turn divergence counter + frozen `mode`** (`kaitofukami_159`).
4. **Self-referential tree branches** (`feature == -2`) + decline-to-switch leaves (`route < 0`)
   (`thomastschinkel_95.5`).
5. **Trajectory-equivalence guard** — irreversible forks only (`thomastschinkel_74.5`).

Every per-episode dict is reset by the same idiom: `if step == 0 or step < state["last_step"]`.

### 2.1c Guardrails are monotone — the corpus almost never scales a counter *up* on positive evidence
Where it does, it is latched and irreversible (`CT_TABLE` at step 2, `_V93_ROUTE_BY_RIVAL` at
step 144, `_RACE_HORIZON_MIRROR` at step 1, `_R44 matched`, kaitofukami `latched`). The one
deliberate exception is the lead-driven horizon — and its own tuning record shows a real cost
past the peak. **Design implication: prefer irreversible escalation earned by evidence of
having been beaten, over proportional response to a similarity score.**

### 2.2 The two concrete reactions that actually flip games

**(a) Clone pre-emption** — the debt-tracked pull-forward. `tetsutani:_preempt_shift` (304-345):
```
if 120 <= step < 680
   and no repayment outstanding
   and _clone_distance(obs) <= 6            # they are running OUR tape
   and tape[step+1] sells >= 4 units of a premium item
   and market has a free slot
   and projected shed can cover it:
     sell it NOW, record due={item: qty} at due_step=step+1
```
`_repay_shift` (267-291) then trims exactly that quantity off tomorrow's SELL of the same
item. **Total season volume is conserved; only the timing moves.** If the repay step is
missed the debt is simply forgiven — the shift is opportunistic, never a debt.

**(b) Horizon inference from observed races** — Rayk's C68, the best version of this idea:
> "The controller observes premium-product market inventory changes, removes its own sale
> and deterministic town drain, **fits the opponent's extra batches to horizons 1-6**, and
> races one turn ahead. It defaults to horizon 4 until enough evidence arrives. [...] In
> diagnostics it correctly identified horizons 2, 3, 4 and 5. The classifier is gated by
> public farm similarity."

That is **online horizon estimation from the recovered rival-sale residual**, gated by clone
similarity. Rayk also swept fixed horizons and found the lesson the hard way: horizon 25 won
the aggregate gate, then **lost 0-6 to horizon 2** on fresh seeds. *"Aggregate wins against
weak historical agents had hidden the strategically important parent regression."*

### 2.3 Failure modes to design against
- **Over-holding.** Your `regime_counter` lost **0/20** to `shop_opportunist`. Diagnosis:
  classifier + delays over-hold inventory. In a book where surplus clears in <1 day,
  holding is strictly dominated.
- **Hard-coding a specific rival.** `prvsiyan`'s `CT_TABLE` maps exact
  `(rival_money, market_WHEAT)` pairs to hard-coded wheat trades at specific market slots.
  Rayk's ledger records this pattern as *"opponent-specific hard-coding, not generalisation."*
  Prefer behavioural features over identity.
- **Rival sale stream prediction with too little evidence** — `V92 PREDICT2` is shipped
  **inert** because its stream library wasn't included in the distribution. Verify your
  payload actually decodes.
- **Clone-confidence gating suppressed useful moves** without improving robustness (Rayk
  C94 negative result). Gate *scope*, don't gate *whether to adapt at all*.

---

## 3. THEME 2 — Best returns from predictive future returns

Ranked by sophistication, with the exact objective function for each.

### 3.1 Price model (foundation — and there is a live correctness bug in half the corpus)

All top agents reimplement it and patch from `obs['market']['params']`:
```python
def _shape(f, x, T=None):  # linear | sq | sqrt | log | log10 | hinge
    x = max(0.0, float(x))
    if f == "hinge":
        if not T or T <= 0: return x
        u = x / float(T)
        return u + 8.0 * max(0.0, u - 1.0) ** 2      # linear to the knee T, then quadratic blowup
    ...
```

#### ⭐⭐ Verified bug: two rival parameter tables, and the older one cannot express the engine
There are **two** tables in the corpus and **they are not equivalent**:

| | CARROT below | TOMATO below | EGG below |
|---|---|---|---|
| **`_R37_MARKET_PARAMS`** (prvsiyan, thomastschinkel_2945, arsgorynich, lynnsakurai, ahmedberatozer, aurax7) | `hinge` | `hinge` | `hinge` |
| **`_MARKET_PARAMS`** (boatlee v14 + v16-rc2, romanrozen, lucifer19, kaitofukami_25-27) | `log` | `linear` | `linear` |

The older table's `_shape` **raises `ValueError` on `"hinge"`** — it physically cannot represent
the engine's shape. I verified against **`src/kaggriculture/env/items.py` in your own repo**,
which is an exact engine clone: it uses `hinge` for CARROT/TOMATO/EGG with
`shape_func(..., T)` implementing `u + HINGE_GAIN·max(0, u−1)²`. **The `_R37` table is correct;
the older one is wrong.** (And note your own `agent_final.py` / `scratch_grandmaster.py` contain
**no price model at all** — so this bug is one you can avoid rather than one you have.)

Measured divergence, short (scarce) side only:

| item | inv | `_R37` (correct) | `_MARKET_PARAMS` (old) | ratio |
|---|---:|---:|---:|---:|
| TOMATO | 9100 | **2520** | 168 | **15.0×** |
| CARROT | 9100 | **385** | 43 | **9.0×** |
| EGG | 9100 | **573** | 104 | **5.5×** |
| TOMATO | 9600 | 300 | 108 | 2.8× |
| CARROT | 9600 | 66 | 42 | 1.6× |

**I verified the glutted side is bit-identical for all 9 items across d = 0…2999.** So the
divergence is confined to the *scarce* side of exactly the three items that use `hinge`.

**Consequence:** the boatlee / romanrozen / lucifer19 / kaitofukami lineage systematically
**undervalues a scarce carrot, tomato or egg**, which biases them toward never holding one —
in a game where a scarce tomato is worth 15× what their model thinks. `romanrozen` calls itself
"ECONOMIST"; this is the single biggest correctness gap in the public corpus.

Both tables are hardcoded and then patched from the observation:
```python
def _v44y_params(obs):
    params = {k: dict(v) for k, v in _R37_MARKET_PARAMS.items()}
    for k, patch in (obs['market'].get('params') or {}).items():
        if k in params and isinstance(patch, dict): params[k].update(patch)
    return params
```
**The older table's lineage never reads `obs['market']['params']` at all** — fully hardcoded.
Another reason to take the `_R37` version.

#### The premium collapse (both tables agree — this is where the money is)
```
MILK       floor at I0+76      STRAWBERRY I0+62      WOOL I0+59      MELON I0+158
TOMATO     I0+529             CARROT   I0+842        FERTILIZER I0+493
WHEAT / EGG: no floor within +4000
```
Revenue of one fixed 60-unit lot as the market moves (correct table):
```
CARROT      -600:  5992   -300:  3362   +0:  1747  +100:  1313  +300:   843  +600:   362
STRAWBERRY  -600: 19237   -300: 15486   +0:  3801  +100:    60  +300:    60  +600:    60
MILK        -600: 22054   -300: 18172   +0:  5886  +100:    60  +300:    60  +600:    60
WOOL        -600: 15267   -300: 14883   +0:  7929  +100:    60  +300:    60  +600:    60
```
**A 16–22× spread on an identical basket.** That is the entire economic basis of "sell early,
sell first", and it is why a wrong *scarce*-side model costs you nothing here while a wrong
*glutted*-side model would cost everything.


### 3.2 Self-inflicted impact scoring (cheap, always worth it)
`tetsutani:_impact_score` (558) — the price damage *this* order does to itself:
```python
return qty * max(0.0, current_quote - _market_price(item, current_inventory + qty))
```
Convex in qty, which is the right ordering signal. `C71 Giovanni Impact` sorted premium SELLs
by this and won a round robin on untouched seeds.

Urgency weighting (`tetsutani:_order_score`, 591):
```python
excess  = max(0, current_inventory + qty - 10000)
urgency = min(1.0, (excess / daily_demand) / 10.0)
score  *= (1 + 0.25 * urgency)          # max +25% — a nudge, not a strategy change
```

### 3.3 ⭐⭐⭐ Exact two-player best-response SELL ordering
**`prvsiyan` `_v44y_lockstep` (6048) + `_v44y_reorder` (6125).** This is the most powerful
per-turn algorithm in the public corpus. It is cheap because prices are additive per item.

**Step 1 — replay the engine's lockstep for a candidate ordering:**
```python
def _v44y_lockstep(orders_me, orders_opp, inv0, stock_me, stock_opp, params):
    """Replay the engine's per-slot / per-unit lockstep for SELL and BUY_PRODUCT.
       Returns (revenue_me, revenue_opp)."""
    inv = dict(inv0); stock = [dict(stock_me), dict(stock_opp)]; rev = [0.0, 0.0]
    queues = [list(orders_me), list(orders_opp)]
    for i in range(max(len(queues[0]), len(queues[1]))):     # slot index, both sides
        rem = [None, None]
        for p in (0, 1):                                       # arm slot i for each side
            if i < len(queues[p]):
                o = queues[p][i]
                if o and len(o) >= 3 and o[0] in ('SELL','BUY_PRODUCT') and o[1] in params:
                    n = int(o[2] or 0)
                    if n > 0: rem[p] = [o[0], o[1], n]
        while True:                                             # one unit at a time
            quoted = [None, None]
            for p in (0, 1):
                r = rem[p]
                if r is None or r[2] <= 0: continue
                if r[0] == 'SELL':
                    quoted[p] = ('SELL', r[1], _v44y_price(r[1], inv[r[1]], params))
                elif r[1] in ('WHEAT','FERTILIZER'):            # only these are buyable
                    quoted[p] = ('BUY_PRODUCT', r[1], _v44y_price(r[1], inv[r[1]]-1, params))
                else:
                    rem[p] = None
            if quoted[0] is None and quoted[1] is None: break
            for p in (0, 1):
                op, item, price = quoted[p] if quoted[p] else (None,)*3
                if op == 'SELL':
                    if stock[p].get(item,0) <= 0: rem[p] = None; continue
                    stock[p][item] -= 1; rev[p] += price
                    if price > 1: inv[item] += 1                  # $1 sales don't move the book
                else:
                    stock[p][item] = stock[p].get(item,0)+1; rev[p] -= price; inv[item] -= 1
                rem[p][2] -= 1
    return rev[0], rev[1]
```
Note the two subtleties that make it *exact*: **only SELL/BUY_PRODUCT arms a slot**, and
**a unit sold at the $1 floor does not increase market inventory** (`if price > 1`).

**Step 2 — objective, with per-item memoisation** (`_v44y_factor_margin`, 6093):
```python
margin(cand) = Σ_item  [ lockstep(mine_item, theirs_item).mine − .theirs ]
```
Because there is no shared cash or shed constraint in the model, per-item revenues add, so
you solve each product independently and cache on `(item, schedule)`. That is what makes
brute-force affordable.

**Step 3 — search only inside contiguous SELL blocks of size 2..6:**
```python
blocks = contiguous runs of SELL orders with 2 <= len <= 6
for block in blocks:
    for perm in itertools.permutations(range(n)):
        if dup of an earlier perm: continue
        v = margin(candidate)
        if v > best + 0.5: accept
if best <= base + 0.5: return action unchanged        # can only move a sale if it pays
```
Constraints respected: purchases, round-trips and deliberate empty slots (`[]` holes) are
**kept in place**; the original ordering is always in the candidate set and scores 0.
`shiiin9`'s Layer D generalises this to *all* orderings of the free slots, capped at **800
orderings/turn**, and reports **89 losses → 0 wins lost over 280 games**.

**Gate:** `step >= 216 and _v44y_clone_gate(obs)` — clone gate is `horizon > 0` from the race
detector, or `_r37_similarity >= .95` after a 240-turn equality streak in the last 4 samples.

**Cost:** ~3 ms/turn average for the whole v9/4 stack, stdlib only. Comfortably inside the
1 s budget.

### 3.4 Future-sell reservation + the debt ledger
`prvsiyan` → `Chassis.future_sells(route, item, step)` (446): a **backward suffix sum** over
the tape of planned SELLs. Every reservation layer queries this table.

The debt ledger is the mechanism behind every pull-forward:
```
sell X q units at step t        → book r36_debts[t+1][X] += q
at step t+1, walk the market list and TRIM q off the SELL of X (drop the order if fully repaid)
```
This is the single most transferable trigger in the competition. It appears independently in
`tetsutani` (`_SHIFT_STATE`/`_repay_shift`), `prvsiyan` (`r36_debts`), Rayk (c15/c45/C95), and
`thomastschinkel` v9 (ADV layer). The winning parameterisation across all of them:
**advance 1 turn for premium goods, 2 turns for WHEAT/FERTILIZER, cap the batch at 10-12
units, and only from a sale that is already planned.**

### 3.5 Rival-exposure ranking (ORDERPRI2 / `_or2_exposure`)
Estimate the rival's **unsold stock** from public tile harvests + recovered sales, cap it
(`_OR2_CAP = 30`), then move the product most exposed to a rival batch to the front of the
list; when all 10 slots are full, swap out the weakest order and refund its booked debts.
`shiiin9` re-measured the gate: `_OR2_SLOT_MARGIN` 20 → **8** (the gain needed before a sale
moves into an earlier slot) was worth 12-24 wins on its own.

### 3.6 Species EV over a projected price path (`_hd2_ev`)
`prvsiyan:5495` (HD2 HERD2) — full expected-value comparison of GOOSE vs COW vs SHEEP over a
30-day projected price path, including **forward shop unlocks**:
```python
unlocks_left = max(0, 8 - len(shops))
opened       = min(unlocks_left, max(0, (d//3) - (day//3)))   # one shop every 3 days
```
plus `_hd2_future_demand_per_day` (5281) for the expected value of one more uniformly-drawn
shop. Then rewrite `BUY_ANIMAL`, `BUILD_COOP`→`BUILD_PASTURE`, `PICKUP`/`PLACE`, and sell the
extra product. `arsgorynich/herd-safe-v3` is the "risk-aware feed" variant of the same idea.

### 3.7 Lookahead / search actually used
- **Beam search over fertiliser placement** — `_r51_input_forecast` (2159): simulates the
  *entire remaining tape* (movements, HIRE spawns, midnight respawns) to find each crop's last
  watering/HARVEST, then 8-deep beam over which tiles to fertilise. Days 12-28, hours 1-3.
- **7-turn terminal shadow planner** — step 712: clone the parent agent's private state, run
  it 7 steps through the real `_apply_unit_action`, accept only if steps 712-717 emit exactly
  9 SELLs covering all 9 products at ≥100 each, then `plan_terminal(max_simulations=64,
  passes=1, proposals_per_actor=4)` under a 200 ms budget.
- **Reed-Solomon coupon-collector framing** — `jaxa623` ("2780 beyond 48-0: 128/128 worlds with
  95 CIs") treats the 64 shop-pair worlds as a coupon-collector problem for sample sizing.

### 3.8 What is conspicuously ABSENT from the entire public corpus
No PPO / DQN / policy gradient / MCTS / PUCT / CFR / Nash-solve anywhere. No bandits beyond
ε-greedy. **No Monte Carlo and no stochasticity at all** — grep for `random` in the agent
bodies returns nothing used in decision logic. Every "simulation" in the corpus is a
**deterministic forward replay**; the engine has no RNG in the market.

This is a deliberate finding, not an oversight. With a 1 s/turn budget and a deterministic
719-step engine, an explicit price/order simulator **dominates** a learned value function,
because the value function would have to learn the price curve you can just evaluate. The top
agents' sophistication is entirely in **state estimation and adversarial modelling**, not in
sampling.

Your repo's own `research/01 §3` reached the same conclusion from the other direction.

### 3.9 Measured costs — everything fits, with 2 orders of magnitude of headroom
Benchmarked on the corpus (Apple Silicon, CPython 3.9; Kaggle CPU typically 2-3× slower, so
double for a conservative budget). Target <50 ms/turn so a 30-40 layer stack has room.

| Tier | What | Cost |
|---|---|---|
| **0 — foundation** | `_r37_market_price` / `future_sells` suffix table / `_v9_town_draw` | 0.7 µs / **0.4 ms once per route**, 0.03 µs per query / 2 µs |
| **1 — per-turn scoring** | `_or2_exposure`, `_r37_quote_priority`, `_order_score`, `_ca_yield_path` | 9–86 µs each |
| **2 — single-decision EV** | `_hd2_ev` (two-path 30-day projection) | 48–93 µs/option, **~0.2 ms for 3 options** |
| | `_r51_input_path` (beam, depth 8 × width 8) | **3.3–17 µs**/call, ≤6 calls ⇒ <0.1 ms |
| | `_cxtb_expected_revenue` (calibrated 8-day) | 65 µs, once at step 432 |
| **3 — best-response ordering** | `_v44y_reorder` (≤744 evals) | **0.33–5.2 ms** |
| | `_cxd_reorder` (800-eval cap, maximin) | **1.8–4.8 ms** |
| | `_r132_mirror_reorder` (pair-swap ascent) | 4.6–19.5 ms, but fires 1 turn in 72 |
| **4 — the debt ledger** | `_r36_reserve` + `r36_debts` (48-step tape walk × 9 items) | **<0.2 ms** |
| **5 — endgame** | `plan_terminal` (≤64 sims, 7 steps, real engine fns) | **6.1 ms** shipped cfg, 15.7 ms max, once at step 712 |

**The per-item factorisation in `_v44y_factor_margin` is why the search is affordable** — it
turns `O(permutations × 10 items)` into `O(permutations × 1 item)`, and any item whose schedule
is unchanged across a permutation is free.

**Everything in the corpus is affordable.** A complete stack — price model + suffix table +
per-turn exposure scoring + maximin reorder + debt ledger + terminal planner — is **~6.5 ms
per turn amortised, or 0.65% of the 1 s budget.**

### 3.10 The three highest-leverage ideas, ranked
1. **The debt ledger (`r36_debts`) is the enabling trick.** It is what makes it safe to run an
   aggressive 40-turn pull-forward *every single turn*: a sale moved from `t+40` to `t` is not
   duplicated, because a debt is booked at `t+40` and `pop`ped (one-shot) when that step
   arrives, **with the order slot preserved at qty 0 so the whole downstream market race is
   unchanged**. Deleting the order instead would shift every later slot. Costs ~50 µs. This is
   the most reusable idea in the corpus.
2. **The town-drain law is what makes the rival observable.** The rival's shed is private, but
   `market['inventory']` is public and town consumption is a *deterministic* function of
   `(step, unlocked_shops)`. So the recovery is exact, not an estimate. This is what turns
   "guess what the rival will do" into "read what the rival did".
3. **The lockstep resolver + maximin permutation search is the most sophisticated object here**,
   and its real insight is subtle: against a clone, the rival's order list is *exactly your own
   parent list*, so the search space collapses from "joint policy" to "my own permutation" —
   `≤6! = 720` per block, factorisable per item. The `min` over rival models in `_cxd_reorder`
   is the natural next step and costs nothing today because `_CXD_MODELS` is still empty.

### 3.11 The terminal planner's three safety predicates
`plan_terminal` is a first-improvement coordinate ascent over actors with three independent
gates that must **all** hold:
| gate | predicate |
|---|---|
| value | strictly more stock value at **frozen current prices** |
| dominance | zero overflow ∧ shed/sold/per-actor-deposits ≥ baseline at **every** step |
| per-item | no *other* worker's `(xy, op, actor) → items` set is shrunk |

plus the final acceptance: `worker_change ∧ deposited_gain ∧ (any Δ>0) ∧ (all Δ≥0)` — a real
action change, a real physical delivery increase, and **no product sells fewer units than
baseline**. Note it scores with `(sold + residual_shed) · frozen_price` and deliberately does
**not** model the market: over 7 steps the rival can barely interfere, so it only decides what
to physically deliver and liquidate. That's the correct scope.


---

## 4. THEME 3 — Predicting the shop

### 4.1 The mechanics you are predicting
A shop unlocks **every 3 days**, drawn uniformly **with replacement** (duplicates possible),
**capped at 8 instances**. `obs["town"]["unlocked_shops"]` is append-only, so shop *i* is known
at roughly `day 3i` — i.e. turn `72i`. Eight shop types:

| shop | products | demand weight |
|---|---|---|
| YARN_STORE | WOOL | **12/day** (single-product ⇒ 2×) |
| PET_CAFE | CARROT | **12/day** |
| BAKERY | EGG, WHEAT | 6/day each |
| PIZZA_SHOP | MILK, TOMATO, WHEAT | 6/day each |
| BRUNCH_SPOT | EGG, WHEAT, STRAWBERRY | 6/day each |
| ICE_CREAM_SHOP | STRAWBERRY, MILK, WHEAT | 6/day each |
| SMOOTHIE_SHOP | STRAWBERRY, MILK | 6/day each |
| FARMERS_MARKET | WHEAT, CARROT, TOMATO, STRAWBERRY | 6/day each |

Plus Town Center: 1 of each non-fertilizer product, every 24 turns, all season.

### 4.2 Level 0 — periodic demand, no prediction at all
`tetsutani:_v17_town_demand_at` (448). Pure modular arithmetic, so it works at *any future step*:
```python
def _v17_town_demand_at(obs, item, step):
    demand = 1 if item != "FERTILIZER" and step % 24 == 0 else 0   # town centre
    if step % 4 != 0: return demand                                  # shops buy on a 4-turn cadence
    for shop in obs["town"]["unlocked_shops"]:
        products = _SHOP_PRODUCTS.get(shop, ())
        if item in products:
            demand += 2 if len(products) == 1 else 1                  # arity IS the weight
    return demand
```
This is a **complete and exact** model of *periodic* consumption. It does **not** forecast
future unlocks — and that is fine, because by the time you care, the list is already known.

### 4.3 Level 1 — branch the route on the observed shop list (the dominant pattern)
Nearly every top agent forks its plan on the **ordered first two shops**, latched at
**step 144** (day 6).

`tetsutani:_kawa_route_label` — 5 routes, a strict cascade on the *position* of YARN_STORE:
```python
if shops[:1] == ["YARN_STORE"]:            return "6c12s_4q_first_yarn"
if "YARN_STORE" in shops[:2]:              return "6c12s_4q_second_yarn"
if "YARN_STORE" in shops[:3]:              return "6c8s_3q"
if {"PIZZA_SHOP","ICE_CREAM_SHOP","SMOOTHIE_SHOP"} & set(shops[:3]):
                                               return "10c4s_3q"
return "8c6s_3q"
```
Because `unlocked_shops` is append-only, this decision is effectively frozen after the third
unlock.

`prvsiyan` has **41 routes** in its tape payload (`routes` = 41 × 719 indices into a pool of
3,982 shared action dicts) plus an explicit **8×8 = 64-entry shop-pair → route-id map**
(`shops` key of `_R108_DATA`). `yhay81`'s Six-Day Fieldbook does the same at 4 block
boundaries with **8 routes / 16 binary tests / 20 leaves / 30 possible full-season paths**.

**Why two shops and not three** (destbreso, measured): land/herd/construction commit in week
one, so a shop revealed on day 3 or 6 can still steer them; day 9+ meets a committed farm.
The strongest public router priced its second branch at **+0.136 rating points/game** over
his own 24,000-game census. Each further branch multiplies the space by 8, and validation
episodes per world collapse. **Whether a third branch pays is an open question, not a
settled no.**

### 4.4 Level 2 — probabilistic expected value over the remaining draws
`_hd2_ev` / `_hd2_future_demand_per_day` (`prvsiyan:5281-5368`) is the only place in the corpus
that does this properly: enumerate the remaining `unlocks_left = max(0, 8 - len(shops))`
draws, weight each by its demand contribution, and take the expected value of one more
uniformly-drawn shop. `shiiin9` pushed the same idea into a **priced gate**: instead of
counting shops, project TOMATO inventory day by day (town demand + the rival's public tomato
tiles with their plant days + our own 20/day) and price every unit with the engine's own
curve; commit when the projection clears **$9,000**. Forcing the investment regardless costs
**34 wins in 119 games**.

#### 4.4a ⭐⭐ THE HEADLINE GAP: the future-draw model exists and is switched off

`_hd2_future_demand_per_day` is **textbook-correct**:
```python
def _hd2_future_demand_per_day(item):
    """Expected extra daily demand of one more uniformly drawn shop instance."""
    return sum(6.0 * (2 if len(p) == 1 else 1)
               for p in _HD2_SHOP_TYPES.values() if item in p) / len(_HD2_SHOP_TYPES)
```
It feeds a 26-day forward market-inventory simulation with an exactly correct unlock-count
schedule, and the EV includes the **price-swing externality on the rival's book**:
```python
unlocks_left = max(0, 8 - len(shops))
base_demand  = _hd2_daily_shop_demand(item, shops) + 1.0        # +1 = town centre
extra_demand = _hd2_future_demand_per_day(item) * _HD2_FUTURE   # <-- THE PARAMETER
...
for d in range(day + 1, 30):
    opened = min(unlocks_left, max(0, (d // 3) - (day // 3)))   # 1 shop / 3 days, max 8
    inv -= base_demand + extra_demand * opened
    inv += ours_existing.get(d, 0.0) + rival_existing.get(d, 0.0)
    prices[d] = _r37_market_price(item, int(round(inv)), params)
...
swing = sum((new_prices[d] - base_prices[d]) * (ours_existing[d] - rival_existing[d]) for d in base_prices)
return revenue + swing - spec["cost"] * k
```

**But `_HD2_FUTURE = 0.0` in all four published agents that ship this code** (verified by grep
across `prvsiyan`, `thomastschinkel_2945`, `arsgorynich`, `evgendvorkin`). The expectation term
is multiplied straight out, so the model degenerates to "current shops only."

Setting it to 1.0 gives these `extra_demand` values (units/day per opened future shop):

| product | E[extra/day] | driven by |
|---|---:|---|
| WHEAT | **3.75** | BAKERY, BRUNCH_SPOT, FARMERS_MARKET, ICE_CREAM_SHOP, PIZZA_SHOP (all 1×) |
| STRAWBERRY | **3.00** | BRUNCH_SPOT, FARMERS_MARKET, ICE_CREAM_SHOP, SMOOTHIE_SHOP (all 1×) |
| CARROT / MILK | 2.25 | PET_CAFE(2×)+FARMERS_MARKET / PIZZA+ICE_CREAM+SMOOTHIE |
| TOMATO / EGG / WOOL | 1.50 | FARMERS+PIZZA / BAKERY+BRUNCH / YARN_STORE(2×) |
| **MELON / FERTILIZER** | **0.00** | no shop demands either |

The strategic read is sharp and immediately actionable: **WHEAT and STRAWBERRY have the
highest expected future demand; MELON and FERTILIZER have literally zero.** That is an
argument for a wheat/strawberry-weighted forward plan and against ever holding melon into the
back half — obtainable from a one-line constant every published agent set to 0.

Whether `1.0` actually wins is empirical, and four agents shipping `0.0` is evidence against.
But those four are **the same code lineage**, so that is one decision replicated, not four
independent confirmations — and it was never A/B tested against a W/L/T objective on fresh
seeds. That is precisely the experiment worth running.

#### 4.4b The complete ordered-pair route table (highest resolution that exists)
`_R108_SHOP_ROUTES` (`prvsiyan:953`) — **64 cells = the full 8×8 ordered first-two-shop grid**,
mapping to **28 distinct route ids** (101, 103–128) out of 41 tapes. Provenance comment:
*"EXP239 native schedules: Yusuke Hayashi (yhay81), Shop Router 0913."* Selected rows:

| pair | route | pair | route |
|---|---:|---|---:|
| (BAKERY, BAKERY) | 101 | (PIZZA_SHOP, BAKERY) | 120 |
| (BAKERY, BRUNCH_SPOT) | 103 | (PIZZA_SHOP, BRUNCH_SPOT) | 105 |
| (BAKERY, FARMERS_MARKET) | 104 | (PIZZA_SHOP, FARMERS_MARKET) | 121 |
| (BAKERY, PET_CAFE) | 106 | (PIZZA_SHOP, PET_CAFE) | 105 |
| (PET_CAFE, PET_CAFE) | 118 | (PET_CAFE, YARN_STORE) | 119 |
| (FARMERS_MARKET, YARN_STORE) | 114 | (YARN_STORE, PIZZA_SHOP) | 128 |
| (ICE_CREAM_SHOP, YARN_STORE) | 115 | (YARN_STORE, YARN_STORE) | 125 |

**Route 105 is the junk-draw fallback — 19 of 64 cells.** Even the best router buckets
farm-heavy and ice-cream-heavy prefixes together.

Caveat: `_V92_TABLE` (`prvsiyan:962`) overwrites all 15 YARN pairs to route 9, so the shipped
router only really uses R108 for the 49 non-YARN cells. The router also carries a hard-coded
rival-fingerprint override keyed on `(round(rival.money,3), market.inventory.WHEAT)` captured
at step 2 — clone pre-emption, not shop prediction.

#### 4.4c The other routers, in full

**`yhay81 shop-router-0909` — 13 tapes, 15 ordered-pair keys, latched at step 144:**
```python
SHOP_PLANS = {
    ("BAKERY","YARN_STORE"): 3,        ("BRUNCH_SPOT","YARN_STORE"): 4,
    ("FARMERS_MARKET","YARN_STORE"): 5, ("ICE_CREAM_SHOP","YARN_STORE"): 6,
    ("PET_CAFE","YARN_STORE"): 5,      ("PIZZA_SHOP","YARN_STORE"): 7,
    ("SMOOTHIE_SHOP","YARN_STORE"): 8,
    ("YARN_STORE","BAKERY"): 9,        ("YARN_STORE","BRUNCH_SPOT"): 9,
    ("YARN_STORE","FARMERS_MARKET"): 1,("YARN_STORE","ICE_CREAM_SHOP"): 9,
    ("YARN_STORE","PET_CAFE"): 10,     ("YARN_STORE","PIZZA_SHOP"): 6,
    ("YARN_STORE","SMOOTHIE_SHOP"): 11,("YARN_STORE","YARN_STORE"): 12,
}
if step == 144:  state.plan = SHOP_PLANS.get(tuple(shops[:2]), 0)
if step == 648:  state.plan = 2      # hard override; "all routes share the ending from 648"
```
**Every one of the 15 keys contains `YARN_STORE`** — this router is structurally a YARN
detector; all 49 non-yarn cells collapse to plan 0. Tapes were stitched from 1,326 public
histories / 1,313 unique action sequences, selected by ~21M candidate-game comparisons.

**`yhay81 three-day-shop-router` — 2 routes, latched at step 360, shop is only one of two
predicates.** The router is in C++:
```cpp
int select_route(const kag::State& state, int seat) {
    if (state.n_shops >= 1 && state.shops[0] == kag::SHOP_BAKERY &&
        state.market.inventory[kag::FERTILIZER] <= 10232.5) return 1;
    if (state.n_shops >= 1 && state.shops[0] == kag::SHOP_PET_CAFE &&
        plant_tiles(state.farms[1 - seat]) <= 64.5) return 1;
    return 0;
}
```
The continuous public co-predicates (fertilizer inventory, **rival plant-tile count**) are
tuned to make the two shop predicates rare and non-overlapping — deliberate gating, not a
model. This is the only agent that forwards all 8 shop instances to native code.

**`yhay81 fieldbook` — 6 routes, and the ONLY agent that uses a third shop:**
```python
if   first  == "YARN_STORE":                       return "YARN", 3
elif second == "YARN_STORE":
    return ("LAND_RECOVERY", 16) if cache["land_gap"] < 0 else ("YARN", 3)
else:                                              return "BALANCED", 0
# stage 2 at step 216 adds shops[2] + cash_gap + wheat_price against tuned thresholds
#   severe_cash_gap = -199.0, very_severe_cash_gap = -721.5, wheat_cheap = 29.5, low_cas = 242.5
```
Also the only file that reads `townShopUnlockInterval` / `townShopSellInterval` from
configuration — and it uses them to **suppress** a sell-lead near an unlock boundary, not to
predict the shop.

**`thomastschinkel 93.8% Public State Router` — the only order-invariant, count-aware,
periodically re-deciding router:**
```python
x = [shops.count(s) for s in _SHOPS]      # 9-dim bag-of-shops -> per-product demand vector
route = _choose(block, x)                 # per-block binary tree
if step % 144 == 0: re-decide             # so it DOES see shops 3, 4, 5
```

**`avioon` — the only continuously-reactive shop consumer**, and the crudest featurization:
```python
has_yarn_store     = any('YARN' in s for s in unlocked_shops)
has_dairy_shop     = any('SMOOTHIE' in s or 'PIZZA' in s or 'ICE_CREAM' in s for s in unlocked_shops)
has_berry_shop     = any('SMOOTHIE' in s or 'BRUNCH' in s or 'ICE_CREAM' in s or 'FARMERS' in s ...)
has_pet_cafe       = any('PET_CAFE' in s for s in unlocked_shops)
has_farmers_market = any('FARMERS' in s for s in unlocked_shops)
```
Never latched, re-evaluated every turn — but it only moves **sell timing and thresholds**
(`step % 4 == 1` batches, price thresholds 70-80), never herd species or land.

#### 4.4d ⭐ How shop prediction actually changes actions in practice
Across 14 files the shop→action link is almost always `_v233_eligible`, a **count threshold**:
```python
if obs['town']['unlocked_shops'].count('YARN_STORE') < 2:  return False
```
plus `WOOL >= 220`, `WHEAT <= 45`, ≥3 quadrants, free pasture sites, no sheep in transit. When
it passes, `_v219_request` injects at day 12, hour ≤1:
```python
[['BUY_LAND'], ['BUY_ANIMAL','SHEEP',6], ['BUY_PRODUCT','WHEAT',6], ['HIRE'], ['HIRE']]
```
**That single injected block — land buy + herd species + feed + two hires — is what a shop
prediction actually buys you in this competition.** Nothing more exotic is deployed anywhere.
`boatlee`'s `_v16_yarn_route` is the only variant that also rewrites species at the *tape*
level (`BUY_ANIMAL COW` → `SHEEP` at step 192) and gates WOOL liquidation cadence (batches of
16, pressure at 78 units, min gap 6 turns).

#### 4.4e The concrete gap, stated plainly
Every top agent latches its shop-driven decision at **step 144 / 161 / 216 / 289 / 360 / 648**
— i.e. right after the 2nd or 3rd shop appears — then commits for the rest of the game.
**Shops 3 through 8 are essentially never used to change the plan**, with exactly two
exceptions: yhay81 fieldbook's `shops[2]` at day 9, and thomastschinkel's re-decision at every
144-turn boundary.

Meanwhile a correct expectation-of-the-future-draw function sits unused in four of the
strongest agents, and `_v92_p_pair` already does Bayesian stream-matching against the observed
pair — the natural next step is to condition that library on `unlocks_left` further shop types
and let it pick the route, rather than hard-switching to `plan 2` at step 648. **Nothing in
the corpus does that today.**

### 4.5 Level 3 — ⭐ shop-conditioned rival behaviour prediction
`_v92_p_pair(shops)` indexes a library of **2,398 recorded rival sale streams (438,301
events)** by the ordered first-two-shop pair. This is the only true "predict the shop *and*
the opponent" structure in the meta: *the shop draw determines which behavioural stream the
opponent will follow.* Same first-two-shops ⇒ same rival rhythm, which is exactly what makes
`rival_sold` a usable leading indicator rather than noise.

### 4.6 Level 4 — minimax over colliding public states
`kaitofukami` v58 is the most intellectually honest treatment. Four *different* public agents
share a **bit-identical public signature at step 72** in the ICE regime; no classifier can
separate identical inputs. Two continuations each overfit one side:
| continuation | vs Pico | vs xiongrui |
|---|---:|---:|
| v57 backbone | −462 | +1,429 |
| direct recovery | +3,721 | −4,781 |

The fix is a **worst-case objective, not a better classifier**:
> "I screened state-compatible suffixes over every colliding family, selected the best
> worst-regime route, then changed only one market event."

> "The router does **not** pretend it can identify an unobservable suffix. It picks one action
> that remains profitable across every known continuation compatible with the observation."

**This is the right frame for all three of your themes** when public state aliases.

### 4.7 Trap to avoid: the router vs. reactivity confusion
`destbreso`'s x-ray classifies agents by comparing each episode to its own mode, twice:
- **GLOBALLY** and **WITHIN-WORLD** (grouped by the shop pair actually drawn).
- *"A shop router forks its plan at turns 72 and 144 with the draw, so it reads as wildly
  adaptive globally while each world's games are near-identical. A genuinely live policy
  diverges inside a world too. **The gap between the two readings is the diagnosis.**"*

So if you evaluate a shop router on globally-diverse seeds you will systematically
overestimate its reactivity — and, worse, you will not notice which of your own triggers are
actually firing.

---

## 5. Cross-cutting engineering rules (from the whole corpus)

1. **Kaggle runs the *last callable*, not `agent`.** The loader takes
   `[v for v in namespace.values() if callable(v)][-1]`. Redefining `agent` later does not
   move it, because a dict keeps the key's original insertion position. Every serious agent
   ends each layer with `agent = globals().pop('agent')`. **`prvsiyan` hits this live**: C17
   scored **3000-3000** on validation and lost its first public match 3000-58567 because the
   raw loader picked a two-argument helper instead of the policy entrypoint.
2. **Wrap the whole turn in `try/except` and return a legal no-op.** An unhandled exception on
   *any* turn flips the match to `ERROR` — effectively an instant loss. One line, and it
   matters more than most of the economic logic.
3. **Always `_align_hands`** — pad with `["PASS"]` / truncate to exactly
   `len(farm["hands"])`. The schema is positional: `hands[i]` is bound to the i-th hand
   currently on the farm. Divergence cascades.
4. **Reserve empty `[]` market slots as index-preserving holes.** Because orders settle by
   index, a hole is how you reorder without shifting anyone else's slot.
5. **Layer order matters: repay → rank → append.** Anything appended after the single
   re-ranking pass lands in the lowest-priority slots and is never re-scored.
6. **Latch state per seat, reset on `step == 0 or step < last_step`.** Every state machine in
   the corpus does this.
7. **Commit a diverted worker only at the last legal moment** (`_v17_feed_guard:825`):
   `if distance + 1 < remaining_actions: continue`, then require
   `distance + 1 == remaining_actions`. Never steal a worker mid-plan.
8. **Budget against a simulated future**: `shed − existing_sells − pickup_reserve`, with the
   shed cap 100 and the market cap 10 as hard invariants.

---

## 6. Evaluation protocol (this determines whether any of the above works for you)

The single most repeated lesson across every serious notebook:

1. **Both seats, always.** *"Shared-market games are not symmetric. A one-seat test can reverse
   the apparent winner."* (26 of 30 seeds gave identical results in both seats — the other 4
   did not, and those are exactly the games that decide the rating.)
2. **The unmodified parent and the incumbent are explicit VETO opponents.** A loss to either
   vetoes the change *even if aggregate wins look strong*. This is how Rayk caught horizon 25.
3. **Bradley-Terry on head-to-head results, not coin magnitude.** 918 games, 18 agents, 3 seeds,
   both seats, 6 games per pairing.
4. **Common random numbers**: force identical shop draws across all opponents
   (thomastschinkel: *"Each seed's eight shop unlocks were drawn at random and forced
   identically for every opponent"*).
5. **Filter replay panels**: keep only games where the frozen opponent retains ≥95% of its
   recorded score. *"A recorded opponent can't react. If your agent starves it [...] the
   replay keeps playing its recorded moves into nothing and your 'win' is an artifact."*
   Across five promotions one agent's replay panel rose 589→648 wins/1000 while the live score
   went 2956.6 → 2944.7.
6. **Freeze all parameters, then run one untouched final seed block.**
7. **Report games flipped (+a / −b), not aggregate wins.** Any change with more flips against
   than for is rejected.
8. **Check your engine version.** 1.32.7 for the current rules.
9. **Keep the losses.** *"Losses were inspected instead of removed as 'bad demonstrations'."*

### Published negative results — do not re-run these
| idea | result |
|---|---|
| Fourth quadrant (SE, $4,000) | **0-10** to the incumbent, three implementation families, 450 games |
| Fixed horizon 25 | won aggregate gate, then **0-6** on fresh seeds |
| Copy a full rival market tape | large generalisation failure |
| "Predict two turns ahead" | fell to 2-2 |
| Hold premium goods back for a better price | **−$7.5k to −$34k/game** |
| From-scratch demand-driven planner (v10) | live 990-1,578 vs 2,945 for the tape |
| Adaptive planner takes over after day 12/18 | 6-24 and 7-23 vs 24-6 for the tape |
| Tomato overlay on route wheat tiles (day 13) | 0 gained / 85 lost — **labour-bound, not land-bound** |
| 20-tile tomato patch on new land | 0 / 56 |
| Buy 14-19 wheat and resell the surplus | strong exploit, brittle vs one public family |
| Extra hands to fertilize young wheat | $144+/day per hand > the extra wheat earns |
| Sheep expansion with only ONE yarn store | −$11k to −$17k for us vs −$2k to −$3k for them |
| Skip day-11 strawberries when no berry shop | −$1.5k/game (the rival sells them instead) |
| Fixed 10 cows forever | mirror banks collapse toward ~$40k |
| Clone-confidence gate on adaptation | suppressed useful moves, no robustness gain |

**The unsolved frontier (thomastschinkel §6):** v9/4 beats every public notebook 519-21 but
went **0-36** to seven top-10 ladder teams. It leads until day 10 and loses everything after
day 11 — to **tomatoes**. Those farms buy 9 seeds from day ~12, hold 10 tiles on day 20, sell
71 tomatoes at $114. v9/4 buys 1.6, first on day 18, sells 7 at $316. Every tomato program
failed because *"the route tape is the constraint. Its workers are busy from dawn to dusk, so
a tomato program needs a different labour plan, not just a different crop choice."*

---

## 7. Concrete trigger designs this research supports

Each is cheap, bounded, exception-safe, and has a stated objective function consistent with
W/L/T scoring.

### T0 — ⭐ Shed-headroom gate (do this first; nothing else works without it)
Verified in `_commit_unit`: **a shed at capacity 100 blocks `BUY_ANIMAL` and `BUY_PRODUCT`
outright.** Seeds are exempt (own slot). Shed headroom is a *purchase gate*, not storage
hygiene — and ~66% of requested SELL orders never execute because the shed binds, not the
market.

```
every turn:
    shed  = sum(private.shed.values())
    head  = 100 - shed
    carried = sum(unit inventories)
    projected = projected_shed(action)              # incl. this turn's DROP/PLACE
    need_animal_soon = any(unit is carrying COW/SHEEP/GOOSE to PLACE) or tape plans BUY_ANIMAL
    if projected > 70 or (projected > 55 and need_animal_soon):
        sell down to ~40, priority: WOOL, MILK, EGG, MELON, STRAWBERRY, TOMATO, CARROT,
                                   FERTILIZER, WHEAT   (WHEAT last: it is feed reserve)
        and issue the SELLs in the FIRST market slots, not appended at the end
hard invariants: market[:10]; never sell the 2-day feed reserve
```

The hour-0 sweep in every top tape exists for exactly this reason — relocating it costs
$22k-109k (§1.5b). It is not a price trade.

### T0b — ⭐⭐ The I0 ratchet guard (the single highest-value invariant)
`if price > 1: inv[item] += 1` — **a $1 sale adds no supply**, so once inventory crosses
`I0 = 10,000` the only recovery is town consumption, which is far too slow. Crossing I0 is a
**one-way door**; the hour-8 replay variant lost the season on that single step.

```
every turn, for each premium item in (STRAWBERRY, MILK, WOOL, MELON):
    excess = market.inventory[item] - I0            # I0 = 10000
    if excess > -MARGIN:                             # MARGIN ~ 250 for MILK, ~200 STRAWBERRY
        # we are within MARGIN of the cliff. Sell FIRST, sell SMALL, and take the low slot.
        force a SELL of at least min(stock, ceil(-excess * 1.5) + buffer) into slot 0..1
        and DO NOT add supply-bearing orders behind anything that would delay it
    # converse: if the projected post-sale inventory would exceed I0 + cliff_margin, ABORT the sale
    #   (selling into a glut is how you cross I0 in the first place)
cliff = {STRAWBERRY: 62, MILK: 76, WOOL: 59, MELON: 158}    # units to $1 floor
```
This is why `_v44y_lockstep` best-response ordering (§3.3) and the `r36_debts` pull-forward
(§3.4) are the two most valuable market layers in the corpus: **losing the slot race does not
cost you a few dollars, it can cost you the season.**

### T1 — Rival sale recovery → adaptive race horizon  *(Theme 1)*
```
every turn:
  inv, inv'  = market.inventory[t-1], market.inventory[t]
  draw       = town_demand(prev_shops, t-1)         # exact, from §4.2
  own_sold   = Σ qty of my SELLs that ACTUALLY cleared (shed-clamped -- an un-fillable
               SELL never entered the market and must not be subtracted)
  residual[p]= inv'[p] - inv[p] + draw[p] - own_sold[p]     # == what the rival sold
  skip if prev_prices[p] <= 3                       # below ~$3 the curve is too flat to attribute
  fit residual peaks to horizons 1..6; horizon = argmax count, default 4
  if similarity(me, rival) >= .90 and residual is non-trivial:
      pull my planned premium SELLs within `horizon` turns forward, debt-tracked
gate: 144 <= step < 700, shed headroom > 0 (see T0), price[p] > base[p]
cost: O(1) + O(horizon)
```

### T2 — Lockstep best-response reordering  *(Theme 2)*
```
if step >= 216 and clone_gate:
    for each contiguous SELL block of size 2..6:
        for perm in permutations(block):
            v = Σ_item [lockstep(mine, theirs).mine - .theirs]   # memoised per item
            keep best
    apply only if best > base + 0.5
respect: keep purchases / round-trips / [] holes in place; market[:10]
cost: ~0.2-1 ms with per-item memoisation
```

### T3 — Shop-pair router with a priced gate  *(Theme 3)*
```
at step 144:  route = TABLE[tuple(obs.town.unlocked_shops[:2])]     # 64 entries
hysteresis:   latch; reset only on step 0 or step regression
additionally, price the decision instead of counting shops:
    projected_inventory[p] += town_demand(shops, day) - rival_public_tiles[p] - my_supply[p]
    commit if Σ projected units × engine_price(p, projected_inventory[p]) > THRESHOLD
```

### T3b — ⭐ Turn on the future-draw expectation (the cheapest open experiment)
This is the single highest-value untested knob found in the corpus. It requires **no new
machinery** — `_hd2_ev` already exists, is correct, and is multiplied by zero.

```
HD2_FUTURE = 0.0   (shipped)   →  1.0   (candidate)

protocol (per §6, non-negotiable):
  1. both seats, 3 seeds, 6 games/pairing vs the incumbent AND the unmodified parent
  2. parent + incumbent are VETO opponents — one loss each vetoes
  3. common random numbers: force identical shop draws for every opponent
  4. report games flipped (+a / −b), not aggregate wins
  5. sweep 0.0 / 0.5 / 1.0 — and separately per-product, because the sign almost
     certainly differs by product:
        WHEAT 3.75, STRAWBERRY 3.00, CARROT/MILK 2.25, TOMATO/EGG/WOOL 1.50,
        MELON 0.00, FERTILIZER 0.00
  6. freeze, then one untouched final seed block

predicted effect: biases the forward plan toward wheat/strawberry and away from holding
melon or fertilizer into the back half. MELON and FERTILIZER are unaffected *by
construction* (their expectation is exactly 0), so any measured change is attributable to
the other seven — which makes this a clean experiment rather than a confounded one.
```

### T4 — Minimax continuation under public-state aliasing  *(all three)*
```
when two or more known opponent families share your exact public signature at a checkpoint:
    evaluate every compatible continuation against every colliding family
    pick argmax over candidates of MIN over families of margin
    change the SMALLEST number of market events that flips the worst case positive
(Kaito's v58 changed exactly one: a 15-unit SELL from step 689 to 690.)
```

### T5 — Terminal 7-step shadow search  *(Theme 2, endgame)*
```
at step 712: clone private state; run the real _apply_unit_action forward 7 steps
             accept only if steps 712..717 emit 9 SELLs covering all 9 products at >=100
             then plan_terminal(max_sims=64, proposals_per_actor=4, budget=200ms)
at step 716: top-up sells from live shed
at step 718: full re-listing of the entire shed (the env clamps over-sells)
             NOTE: step 718 executes, index 719 does not
```

---

## 8. What to do with YOUR repo specifically

Your `agent_final.py` ("Sovereign Apex") reads exactly **two** opponent numbers
(`opp_cows >= 3`, `opp_sheep >= 2`). Meanwhile `helpers/opponent.py` (rival sale
estimation, profile + sabotage), `helpers/regime_counter.py` (4-regime classifier),
`helpers/sell_duel.py` (minimax sell quantity) and `helpers/sell_mpc.py` are all built,
unit-tested — and the agents that shipped with them went **0/20**. The diagnosis in your own
`strategy_learnings.md` is right: *over-holding inventory*.

The corpus says the fix is not more classification, it is **closing the loop on the rival's
recovered sales** and acting on it in the next 1-4 turns. Concretely:

| your file | corpus counterpart | action |
|---|---|---|
| `helpers/opponent.py` `estimate_imminent_sell_volume` | `_v9_town_draw` + `rival_sold` residual | upgrade to the exact per-turn identity (§2.1 L5); drop the Manhattan-radius heuristic |
| `helpers/regime_counter.py` | `_race_clone` + horizon inference | replace the 4-regime classifier + delays with clone-similarity gating + debt-tracked pull-forward |
| `helpers/sell_mpc.py` | `_v44y_reorder` | MPC on the demand calendar **regressed** ($31,046 vs $33,470). The corpus version that works is lockstep reordering, which is a different objective |
| `helpers/market_prediction.py` `simulate_sell_slippage` | `_v44y_lockstep` | you have the primitives; add the two-player arm and the permutation search |
| `helpers/solver.py` (LP, unwired) | `_hd2_ev` | the LP is the right shape but the corpus wins with explicit simulation + EV over a projected price path |
| `experiments/terminal_search/` | T5 above | already the right idea; wire it and move the start 712 → 717 (Rayk's verified correction) |
| `agent_final.py:331-344` | — | the 2-number heuristic is the thing to delete, not extend |

Two blockers to fix regardless of triggers:
- `agent_final._hire_cost = 20 + 10k` contradicts the engine's Fibonacci (correct in
  `scratch_apex_engine.py`). Rayk's SL2 shows the *marginal* cost of hands #12/#13 is
  $377/day — the single cheapest win in the list is getting this function right.
- `run_agent_analysis.py` reports exactly $3,000 / 0.0 ms p95 for every `ActionController`
  agent. That is a telemetry-wiring bug in the panel harness, and it means **your current
  measurements cannot see your own agent**. Fix it before trusting any number.

And per `research/18`'s own verdict: *"apex_mill $28,778, 0/6 wins vs velocity_mill $58,033,
6/6. Therefore the previous Apex design claims are not accepted as evidence of progress
until reproduced by the current simulator in head-to-head matches."* Solo-vs-`random`
numbers (~$30-60k) systematically overstate shared-market performance (~$8-17k).

### 8.1 One place you are positioned *better* than half the corpus
**Your `src/kaggriculture/env/items.py` is an exact engine clone and already has the CORRECT
`hinge` price shape** (`u + HINGE_GAIN·max(0, u−1)²`, used for CARROT/TOMATO/EGG — verified).
A large share of the public agents — `boatlee` v14 + v16-rc2, `romanrozen`, `lucifer19`,
`kaitofukami_25-27` — use a table that **cannot express `hinge` at all** and therefore
undervalue a scarce carrot/tomato/egg by **5–15×** (see §3.1). Your `agent_final.py` has no
price model at all, so this is a bug you can sidestep rather than one you inherited.

**Use `shape_func` + `MARKET_PARAMS` from your own `items.py` as the single source of truth,
and patch from `obs['market']['params']`.** The older public lineage never reads the
observation at all — another reason to take the `_R37` version.

### 8.2 Highest-leverage concrete changes, in order
1. **Add the price model** from your own `items.py` (`agent_final.py` has none) plus the
   `future_sells` suffix table — 0.4 ms once per route, free thereafter. These are Tier 0;
   every layer above them is built on them.
2. **Replace `helpers/opponent.py:estimate_imminent_sell_volume`** with the exact identity
   `rival_sold = inv' − inv + town_draw − own_sold`, `own_sold` **shed-clamped**, `town_draw`
   from the exact law in §1.3. Then feed the measured lead into a reservation horizon (§3.4).
3. **Implement the debt ledger** (`r36_debts`) — ~50 µs, and it is what makes an aggressive
   40-turn pull-forward safe to run *every* turn.
4. **Port `_v44y_lockstep` + `_v44y_reorder`** from `prvsiyan` (Apache-2.0, attribution
   required). You already have `helpers/market_prediction.py:simulate_sell_slippage`; what's
   missing is the **second player's arm** and the permutation search. ≤5.2 ms/turn.
5. **Move `experiments/terminal_search/` start from 712 → 717** (Rayk's verified correction:
   step 718 executes, index 719 does not).
6. **Sweep `_HD2_FUTURE` 0.0 → 1.0** (T3b). Cheapest open experiment in the corpus.
