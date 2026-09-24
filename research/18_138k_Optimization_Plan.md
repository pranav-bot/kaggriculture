# 138K optimization plan and experiment log

**Date:** 2026-09-24  
**Objective:** Reach at least `$138,000` terminal cash reliably in full
720-turn head-to-head episodes against strong local agents.

## Current quantitative baseline

The exact market simulator was used for the existing SciPy analyses:

- `scripts/quant_full_product.py --quick`
- `scripts/quant_land_labor.py`

The joint milk + fertilizer model estimates `$161,211` for a 23-cow
production schedule when there is no opponent sell pressure. This is an upper
bound, not an agent result: it excludes movement, structure placement,
capital timing, shared-market competition, and action-order failures.

The land model finds that NE expansion on day 6 has the highest modeled
marginal payoff (`+$22,700` for six additional cows), with the payoff
declining as expansion is delayed. This confirms that expansion must be
funded early rather than treated as an endgame option.

The current code-level Apex architecture does not yet meet the target. A
three-seed, seat-swapped 720-turn match against `velocity_mill` produced:

| Agent | Mean cash | Wins |
|---|---:|---:|
| `apex_mill` | `$28,778` | 0/6 |
| `velocity_mill` | `$58,033` | 6/6 |

Therefore the previous Apex design claims are not accepted as evidence of
progress until reproduced by the current simulator in head-to-head matches.

## Target policy architecture

Build the next candidate as a new core decision module, not as another
calendar-only patch to `care_mill`:

1. **Opening compound:** reserve enough cash for feed and daily labor, then
   acquire the first demand-backed animals and plant the early melon catalyst.
2. **Cash-triggered expansion:** unlock NE near day 6 when the expansion plus
   animal purchase still leaves two days of feed and the next labor pulse.
3. **Production rotation:** use early wheat for feed, harvest it at maturity,
   and replace selected tiles with strawberry; retain enough wheat capacity
   for the herd rather than buying all feed indefinitely.
4. **Mixed herd:** score cows, sheep, and geese from current shop drain,
   production rate, base price, feed cost, and market curve. Permit a pivot
   while committed capital remains below a fixed pivot budget.
5. **Continuous cash conversion:** reserve critical order slots for feed,
   hires, animals, and land; append merged sell orders afterward.
6. **Low-idle routing:** assign every worker a useful action or staging move;
   require PASS to remain below 5% of unit turns.
7. **Terminal convergence:** from day 27 stop discretionary expansion; from
   day 29 sell all sellable stock and route every carrier to the shed.

## Control variables to optimize

The first search should vary only parameters with clear economic meaning:

| Parameter | Initial search range |
|---|---|
| Day-3 herd target | 4–7 |
| Day-7 herd target | 9–13 |
| Day-14 herd target | 16–23 |
| NE expansion day | 5–8 |
| Feed reserve | 1.5–3.0 days |
| Labor target | 5–8 hands |
| Sell pressure threshold | 82–92 shed units |
| Premium sell start | day 10–18 |
| Pivot advantage | 1.25–1.50× |

Use common seeds and both seats for every comparison. Do not optimize against
the passive opponent only.

## Experiment sequence

### E1 — Production ramp

Compare calendar targets, cash-triggered targets, and the hybrid target
`6 / 12 / 18 / 23`. Record cash at days 7, 14, 21, 28; animals; feed buys;
harvested units; and terminal cash.

**Acceptance:** at least 15% mean cash improvement over `velocity_mill` with
no increase in feed starvation or invalid actions.

### E2 — Land timing

Hold the production policy fixed and test NE on days 5, 6, 7, and 8. Buy SW
only after the melon windfall or when the modeled marginal production return
exceeds the land cost plus feed reserve.

**Acceptance:** expansion must improve mean head-to-head cash, not merely solo
cash against `random`.

### E3 — Market queue

Compare sell-first, two-slot reservation, and merged sells with two critical
slots. Measure order count, sell revenue, quote drawdown, feed starvation, and
terminal stock.

**Acceptance:** zero critical-order starvation and terminal sellable stock
below five units.

### E4 — Worker utilization

Add fallback tasks and terminal routes incrementally. Track PASS, MOVE,
HARVEST, FEED, CARE, WATER, and fertilizer collection shares.

**Acceptance:** PASS at or below 5%, blocked movement below 2%, and no
reduction in harvested production.

### E5 — Opponent pressure

Run the best candidate against `velocity_mill`, `alpha_modular`, `surge_mill`,
and the strongest available local submissions with both seats. Tune only
one parameter at a time after the first 20-seed panel.

**Acceptance:** mean terminal cash at least `$138,000`, median at least
`$130,000`, and at least 60% seat-swapped wins.

## Required telemetry

Each candidate benchmark should emit or persist:

- cash at days 3, 7, 14, 21, 26, and 29;
- herd size by species and active structure count;
- seeds planted by product;
- feed bought versus wheat harvested;
- land unlock days;
- hire count and first hire hour per day;
- market slots, merged sell quantities, and critical-order omissions;
- market quote at each sell;
- unit action percentages and PASS rate;
- shed and carried goods at steps 718 and 719.

## Immediate next implementation

The first implementation should fork the current `velocity_mill` decision
module into an `elite_cashflow` module and add only:

1. melon + wheat opening purchases;
2. day-6 NE expansion with a cash/feed guard;
3. strawberry rotation after wheat harvest;
4. a 23-animal demand/capacity target;
5. continuous merged sells with two reserved critical slots;
6. day-29 all-in liquidation and shed return routing.

Do not claim the `$138K` target until the resulting candidate passes the
20-seed, seat-swapped benchmark. The `$161K` SciPy result remains a useful
economic ceiling and parameter guide, not a validation result.

## Iteration log

### Iteration 1 — land plus melon opening

Hypothesis: an 8-melon opening plus NE land around day 6 would reproduce the
elite capital catalyst while preserving velocity feed/hire ordering.

Result: `iteration1_land_velocity` versus `velocity_mill`, 5 seeds and both
seats, all 720 turns:

| Candidate mean | Baseline mean | Candidate wins |
|---:|---:|---:|
| `$12,642` | `$57,455` | 0/10 |

Several candidate episodes ended below `$1,000`. The variant was discarded:
the melon opening displaced the wheat/feed-safe opening and did not generate a
reliable capital windfall in the current action planner.

### Iteration 2 — solvent land-only expansion

Hypothesis: retain the proven velocity opening and unlock NE only when cash is
at least `$4,200` and wheat covers the herd plus eight units. This isolates the
land effect and avoids the failed melon/feed coupling.

Result: 4 wins, 4 losses, and 2 draws over 10 seat-swapped matches. Both
policies averaged `$39,422`, so land-only expansion was neutral and did not
approach the target.

### Iteration 3 — gated four-melon catalyst

Hypothesis: buy only four melon seeds and forbid melon planting until all eight
wheat plots are established. This preserves feed while testing whether a
smaller day-10 liquidity injection improves expansion and herd scaling.

Result: candidate mean `$27,838` versus `$38,420` for `velocity_mill`, with
0/10 wins. The smaller catalyst still reduced cash flow and was discarded.

### Iteration 4 — surge baseline with fertilizer floor

Hypothesis: use the strongest existing local policy (`surge_mill`) and stop
selling fertilizer below `$40` while the shed is below 90 units. This tests
whether protecting the fertilizer quote improves compounding without changing
the proven herd, labor, routing, or order-budget logic. Terminal pressure
still forces liquidation from day 28.

Result: candidate mean `$39,150` versus `$39,091` for `velocity_mill`, with
5/10 wins. The change was effectively neutral and remains far below the
benchmark. Existing `demand_mill` expansion logic was also rejected in a
3-seed probe (`$3,918` versus `$81,644` for velocity).

### Iteration 5 — solvent second-quadrant expansion

Hypothesis: add one guarded `BUY_LAND` order after hiring and before animal
orders, but only from day 6 with at least `$5,500` cash and eight units of
wheat headroom. This gives the herd room to exceed the 18-animal opening cap
without allowing land to consume survival cash.

Result: identical to Iteration 4 (`$39,150` candidate mean versus `$39,091`
for `velocity_mill`, 5/10 wins). The guard did not activate in these matches,
so the policy remains capped by its opening production architecture. The
`$100,000+` consistency benchmark was not achieved.

### Iteration 6 — compound implementation: feed-safe opening with guarded expansion

**Exact hypothesis:** Preserve `velocity_mill`'s wheat-first opening and
proportional feed ordering, target the aggressive elite ramp (6 animals by day
3, 12 by day 7, 18 by day 14, and 23 after expansion), and buy NE on or after
day 6 only after feed and hiring orders, with at least `$4,200` cash, a
two-day wheat reserve, and the next-hire reserve remaining after the `$1,000`
land purchase. Add only two melon seeds when the planner reports at least ten
dedicated empty tiles and `$1,200` cash beyond operating reserves; plant those
seeds only after all eight wheat plots are established, using independent wheat
and melon planting counters so the liquidity crop cannot displace feed.

Implementation: created `submissions/compound_implementation/main.py` as a
new core policy variant from `velocity_mill`. The market planner reserves feed
and labor before expansion, caps the post-expansion herd at 23, and the worker
planner tracks wheat and melon quotas independently while harvesting both crops.

Result: full 720-turn seat-swapped benchmark against `velocity_mill`, 5 common
seeds (10 episodes):

| Candidate mean | Baseline mean | Candidate wins |
|---:|---:|---:|
| `$46,501` | `$41,475` | 5/10 |

The candidate improved mean cash by `$5,026` (+12.1%) and tied the baseline in
wins, but remains below the `$100,000+` termination benchmark. The policy is
compiled successfully; further optimization is required.

The verified per-episode consistency comparison was:

| Metric | `compound_implementation` | `velocity_mill` |
|---|---:|---:|
| Median cash | `$40,875` | `$45,418` |
| Population standard deviation | `$19,273` | `$11,067` |
| Minimum cash | `$22,403` | `$24,793` |
| Maximum cash | `$73,030` | `$56,145` |
| Episodes at or above `$50,000` | 4/10 | 5/10 |

The compound policy therefore improved mean and maximum cash, but was less
consistent: its median was lower, standard deviation higher, and it reached
`$50,000` in one fewer episode.

### Iteration 7 — bounded cash conversion and terminal merge

**Hypothesis:** Keep `compound_implementation`'s critical order sequence
(wheat/feed, hires, land, animals, then sells), but improve reliable cash
conversion using the quantitative thresholds already identified in this log:
begin normal premium selling on day 14, accelerate at 75 shed units, use a
hard pressure threshold of 85, and enter merged terminal liquidation on day
27. Each product is emitted as one sell order, ranked by quote/base value
during production and by stranded terminal value during liquidation. This
should reduce dead inventory without speculative crop changes or feed
starvation.

Implementation: created `submissions/cash_conversion_mill/main.py` from
`compound_implementation`. The candidate changes only sell timing, pressure
thresholds, product ranking, and terminal quantities; critical market
ordering and independent wheat/melon planting accounting remain unchanged.

Canonical validation: both comparisons used common seeds `0–4`, both player
seats, and exactly `720` turns per episode (`10` episodes per comparison).
The candidate source and the compound opponent both passed Python compilation
before the benchmark.

Against `velocity_mill`, `cash_conversion_mill` won `8/10` with no draws:

| Metric | `cash_conversion_mill` | `velocity_mill` |
|---|---:|---:|
| Mean cash | `$47,963.5` | `$39,957.1` |
| Median cash | `$43,938` | `$43,882.5` |
| Minimum cash | `$23,428` | `$23,372` |
| Maximum cash | `$73,249` | `$51,978` |
| Episodes over `$50,000` | 5/10 | 5/10 |
| Episodes over `$100,000` | 0/10 | 0/10 |

Candidate episode cash by seed/seat was
`$51,125, $36,751, $73,248, $73,249, $23,428, $23,485,
$70,535, $57,715, $35,743, $34,356`; baseline cash was
`$51,978, $37,641, $50,267, $50,266, $23,397, $23,372,
$50,124, $50,740, $33,051, $28,735`.

Against `compound_implementation`, `cash_conversion_mill` won `9/10` with no
draws:

| Metric | `cash_conversion_mill` | `compound_implementation` |
|---|---:|---:|
| Mean cash | `$46,004.1` | `$43,657.0` |
| Median cash | `$39,599` | `$33,898.5` |
| Minimum cash | `$22,588` | `$21,099` |
| Maximum cash | `$66,886` | `$66,600` |
| Episodes over `$50,000` | 4/10 | 4/10 |
| Episodes over `$100,000` | 0/10 | 0/10 |

Candidate episode cash by seed/seat was
`$39,937, $39,261, $66,885, $66,886, $22,590, $22,588,
$64,806, $64,428, $35,752, $36,908`; compound cash was
`$33,433, $34,364, $66,600, $66,599, $21,099, $21,101,
$64,388, $64,717, $32,719, $31,550`.

The candidate improves both opponent means and wins, but remains well below
the `$100,000+` termination benchmark and the `$138,000` objective. This
iteration is therefore recorded as an improvement over the tested baselines,
not as a stopping result; further optimization is required.

### Iteration 10 — solvent incumbent-cow opening

**Hypothesis:** The `cash_conversion_mill` demand gate can collapse before any
shop is unlocked: its animal focus defaults to a demand score, and its market
planner consequently buys no animal during the opening. Permit only a small
two-cow incumbent opening before unlocked-shop demand, while retaining the
existing cash, operating-reserve, next-hire, pasture, and wheat/feed checks.
Keep the incumbent locked until at least four owned cows are placed and
productive; secondary species remain demand-gated after that point.

Implementation: forked `submissions/cash_conversion_mill/main.py` to
`submissions/opening_herd_mill/main.py`. The change is limited to incumbent
cow status/focus and the animal purchase gate; feed, labor, land, sell, crop,
and terminal policies are unchanged.

### Iteration 8 — early herd trigger and controlled second species

**Hypothesis:** The delayed day-3 purchase gate and permanent single-species
lock leave capital idle during the highest-value compounding window. Permit
the incumbent species to be purchased before day 3 whenever the transaction
still leaves the operating reserve, the next hire cost, and the wheat floor.
After day 3, admit at most four animals of one second species only when its
demand-weighted production score is at least `1.50x` the incumbent score.
This preserves feed and labor reserves while allowing a material shop signal
to overcome the incumbent lock.

Implementation: created `submissions/early_herd_design/main.py` from
`cash_conversion_mill`. The candidate changes only herd timing and species
selection: the early target is 2/3/4 animals on days 0/1/2, purchases are
cash-triggered after reserve accounting, and the second-species cap is four.
Critical order sequencing, wheat-first buying, land guards, sell pressure, and
terminal liquidation remain unchanged.

### Crop-rotation design — surplus wheat to strawberry

**Hypothesis:** Once wheat has actually been harvested, replacing only those
now-empty wheat tiles that are surplus to a hard two-day live-animal feed
reserve with strawberry will add high-value ongoing production without
reducing feed safety. The rotation must not claim arbitrary empty land, must
not count a seed purchase as plantable until the next observation, and must
keep wheat and strawberry seed/planted counters independent.

Implementation: created `submissions/surplus_wheat_strawberry/main.py` as a
candidate based on `compound_implementation`. It records wheat coordinates
only when a real `HARVEST` action occurs, intersects that set with currently
empty tiles, and plants strawberry only on those recorded tiles. Market
logic reserves `max(4, live_animals * 2 + 2)` wheat units before wheat sells
or rotation. Strawberry seed purchases are separate from wheat/melon
accounting; the unit planner uses seeds from the current observation only, so
newly purchased seeds cannot be planted in the same turn.

Validation status before benchmarking: both candidate sources and both
baselines compiled successfully with:

```text
.venv/bin/python -m py_compile \
  submissions/early_herd_design/main.py \
  submissions/surplus_wheat_strawberry/main.py \
  submissions/velocity_mill/main.py \
  submissions/cash_conversion_mill/main.py
```

### Iteration 9 — canonical validation of the next-policy candidates

Both candidates were run with the canonical
`scripts/h2h_bench.py` harness, seeds `0–4`, both player seats, and exactly
`720` turns per episode: 10 episodes per matchup. Threshold counts below use
strictly greater than (`>`) the stated cash value.

#### `early_herd_design` versus `velocity_mill`

| Metric | `early_herd_design` | `velocity_mill` |
|---|---:|---:|
| Mean cash | `$24,447.8` | `$52,753.6` |
| Median cash | `$23,456.5` | `$46,854.0` |
| Wins | 5/10 | 5/10 |
| Minimum cash | `$551` | `$16,389` |
| Maximum cash | `$72,740` | `$98,407` |
| Episodes over `$50,000` | 2/10 | 5/10 |
| Episodes over `$100,000` | 0/10 | 0/10 |
| Draws | 0/10 | 0/10 |

Candidate episode cash by seed/seat:
`$52,432, $72,740, $575, $599, $23,428, $23,485, $569, $551,
$35,743, $34,356`.

#### `early_herd_design` versus `cash_conversion_mill`

| Metric | `early_herd_design` | `cash_conversion_mill` |
|---|---:|---:|
| Mean cash | `$23,474.1` | `$33,944.1` |
| Median cash | `$21,851.5` | `$29,351.5` |
| Wins | 2/10 | 8/10 |
| Minimum cash | `$532` | `$11,222` |
| Maximum cash | `$58,864` | `$65,486` |
| Episodes over `$50,000` | 2/10 | 3/10 |
| Episodes over `$100,000` | 0/10 | 0/10 |
| Draws | 0/10 | 0/10 |

Candidate episode cash by seed/seat:
`$37,361, $36,844, $575, $592, $21,859, $21,844, $58,864, $55,733,
$532, $537`.

#### `surplus_wheat_strawberry` versus `velocity_mill`

| Metric | `surplus_wheat_strawberry` | `velocity_mill` |
|---|---:|---:|
| Mean cash | `$42,427.0` | `$47,922.6` |
| Median cash | `$39,017.0` | `$48,640.0` |
| Wins | 0/10 | 10/10 |
| Minimum cash | `$26,471` | `$28,221` |
| Maximum cash | `$64,953` | `$69,058` |
| Episodes over `$50,000` | 4/10 | 5/10 |
| Episodes over `$100,000` | 0/10 | 0/10 |
| Draws | 0/10 | 0/10 |

Candidate episode cash by seed/seat:
`$64,953, $64,931, $27,248, $42,271, $35,763, $35,763, $50,248, $50,077,
$26,545, $26,471`.

#### `surplus_wheat_strawberry` versus `cash_conversion_mill`

| Metric | `surplus_wheat_strawberry` | `cash_conversion_mill` |
|---|---:|---:|
| Mean cash | `$34,194.1` | `$36,860.8` |
| Median cash | `$33,071.5` | `$35,490.0` |
| Wins | 1/10 | 9/10 |
| Minimum cash | `$21,357` | `$23,308` |
| Maximum cash | `$52,412` | `$52,710` |
| Episodes over `$50,000` | 2/10 | 3/10 |
| Episodes over `$100,000` | 0/10 | 0/10 |
| Draws | 0/10 | 0/10 |

Candidate episode cash by seed/seat:
`$22,610, $22,613, $25,864, $42,775, $40,279, $40,283, $21,357, $21,359,
$52,412, $52,389`.

Neither candidate meets the `$100,000+` consistency benchmark or the
`$138,000` objective. `early_herd_design` is rejected for this validation
pass because it is substantially below both baselines. The strawberry
rotation is also rejected as a next policy: it loses all 10 seat-swapped
episodes against `velocity_mill` and wins only 1/10 against
`cash_conversion_mill`, while reducing mean and median cash in both panels.

## Safe production-candidate design (not implemented)

The next candidate should change exactly one control variable: lower the
normal cash-conversion shed threshold from **85 to 75 units** in the existing
`cash_conversion_mill` sell planner. Keep the herd, crop mix, feed reserve,
land guards, order priority, worker routing, and terminal liquidation
unchanged. At 75 units, sell the existing small production lots; retain the
current 85-unit emergency quantity and day-27 terminal dump. This is a
conservative pressure-relief change, not a mixed-herd or crop rewrite.

Rationale from the failure traces: weak agents retained 45 wheat, 49 wheat,
73 milk, or 54 fertilizer at the terminal step, while elite traces ended at
zero to 17 units. The tested cash-conversion policy improved to 8/10 wins
against `velocity_mill`, but still averaged only `$47,964` and had zero
episodes above `$100,000`; its remaining measurable loss is inventory that
was produced but not converted. Earlier herd, melon, and strawberry changes
were rejected for liquidity/feed or shared-market regressions. Lowering one
existing pressure threshold should recycle inventory earlier without changing
the production system or consuming critical order slots.

Validation gate for a future implementation: compare common seeds in both
seats for 720 turns against `velocity_mill` and `compound_implementation`;
accept only if mean and median cash improve, terminal sellable stock falls,
and feed-starvation, invalid actions, overflow, and critical-order omissions
do not increase. Reject on any safety regression or if the improvement is
seed-specific.

### Iteration 10 — pressure-75 cash conversion

**Hypothesis:** Lowering only the normal shed-pressure selling threshold from
85 to 75 units will convert small production lots earlier, reducing stranded
inventory and improving terminal cash without changing feed, hiring, animal,
land, worker, or terminal-liquidation behavior.

**Implementation:** Forked `cash_conversion_mill` to
`submissions/pressure75_mill/main.py` and changed only
`SELL_HARD_PRESSURE` from `85` to `75`. The emergency sell quantity,
all ordering priorities, and terminal thresholds remain unchanged.

**Validation status:** Candidate compiled successfully. Full 720-turn
seat-swapped benchmark remains the next evaluation gate; no benchmark result
is claimed by this implementation entry.

**Canonical validation result:** Both comparisons used common seeds `0–4`,
both player seats, and exactly `720` turns per episode: 10 episodes per
matchup. Threshold counts below use strictly greater than (`>`) the stated
cash value. The candidate compiled successfully before both runs.

#### `pressure75_mill` versus `velocity_mill`

| Metric | `pressure75_mill` | `velocity_mill` |
|---|---:|---:|
| Mean cash | `$47,965.2` | `$39,955.4` |
| Median cash | `$43,946.5` | `$43,874.0` |
| Wins | 8/10 | 2/10 |
| Minimum cash | `$23,428` | `$23,372` |
| Maximum cash | `$73,249` | `$51,978` |
| Episodes over `$50,000` | 5/10 | 5/10 |
| Episodes over `$100,000` | 0/10 | 0/10 |
| Draws | 0/10 | 0/10 |

Candidate episode cash by seed/seat:
`$51,125, $36,768, $73,248, $73,249, $23,428, $23,485, $70,535,
$57,715, $35,743, $34,356`.

Velocity episode cash by seed/seat:
`$51,978, $37,624, $50,267, $50,266, $23,397, $23,372, $50,124,
$50,740, $33,051, $28,735`.

#### `pressure75_mill` versus `cash_conversion_mill`

| Metric | `pressure75_mill` | `cash_conversion_mill` |
|---|---:|---:|
| Mean cash | `$44,998.7` | `$44,889.8` |
| Median cash | `$37,355.5` | `$36,811.0` |
| Wins | 6/10 | 4/10 |
| Minimum cash | `$21,844` | `$21,844` |
| Maximum cash | `$66,854` | `$66,854` |
| Episodes over `$50,000` | 4/10 | 4/10 |
| Episodes over `$100,000` | 0/10 | 0/10 |
| Draws | 0/10 | 0/10 |

Candidate episode cash by seed/seat:
`$37,361, $37,350, $66,853, $66,854, $21,859, $21,844, $64,763,
$64,406, $34,369, $34,328`.

Cash-conversion episode cash by seed/seat:
`$36,844, $36,778, $66,854, $66,853, $21,844, $21,859, $64,406,
$64,763, $34,328, $34,369`.

The threshold reduction improves mean, median, and wins against both tested
baselines, but does not produce any episode above `$100,000` and remains far
below the `$138,000` objective. It is therefore a measurable but insufficient
improvement; the `$100,000+` termination benchmark is not met.

### Iteration 11 — opening-herd validation

**Candidate:** `submissions/opening_herd_mill/main.py`

**Safety review:** The candidate compiles successfully with
`.venv/bin/python -m py_compile submissions/opening_herd_mill/main.py`.
Its market-order priority places the wheat-floor calculation and conditional
`BUY_PRODUCT WHEAT` order before hiring, expansion, animal purchases, seed
orders, and sells. Animal purchases additionally require operating reserve,
the next-hire reserve, sufficient wheat for the live herd plus the purchase,
free structure capacity, and per-turn quantity limits. The required invariant
("prefer `wheat_floor` + buy when applicable") is therefore preserved.

**Canonical validation:** Both comparisons used common seeds `0–4`, both
player seats, and exactly `720` turns per episode: 10 episodes per matchup.
Threshold counts use strictly greater than (`>`) the stated cash value.

#### `opening_herd_mill` versus `velocity_mill`

| Metric | `opening_herd_mill` | `velocity_mill` |
|---|---:|---:|
| Mean cash | `$29,594.2` | `$54,309.6` |
| Median cash | `$28,920.5` | `$55,798.5` |
| Wins | 4/10 | 6/10 |
| Minimum cash | `$8,134` | `$23,372` |
| Maximum cash | `$49,383` | `$82,577` |
| Episodes over `$50,000` | 0/10 | 6/10 |
| Episodes over `$100,000` | 0/10 | 0/10 |
| Draws | 0/10 | 0/10 |

Candidate episode cash by seed/seat:
`$49,374, $49,383, $8,134, $8,134, $23,428, $23,485, $42,301,
$21,604, $35,743, $34,356`.

Velocity episode cash by seed/seat:
`$82,577, $82,577, $78,895, $78,895, $23,397, $23,372, $55,114,
$56,483, $33,051, $28,735`.

#### `opening_herd_mill` versus `cash_conversion_mill`

| Metric | `opening_herd_mill` | `cash_conversion_mill` |
|---|---:|---:|
| Mean cash | `$20,364.6` | `$54,228.9` |
| Median cash | `$21,851.5` | `$60,825.5` |
| Wins | 2/10 | 8/10 |
| Minimum cash | `$421` | `$21,844` |
| Maximum cash | `$39,384` | `$77,067` |
| Episodes over `$50,000` | 0/10 | 6/10 |
| Episodes over `$100,000` | 0/10 | 0/10 |
| Draws | 0/10 | 0/10 |

Candidate episode cash by seed/seat:
`$37,361, $36,844, $9,741, $9,741, $21,859, $21,844, $39,384,
$23,536, $2,915, $421`.

Cash-conversion episode cash by seed/seat:
`$36,844, $37,361, $75,713, $75,713, $21,844, $21,859, $67,020,
$54,631, $77,067, $74,237`.

The opening-herd candidate passes compilation and the wheat-floor safety
invariant, but fails the performance gate against both baselines: it trails
on mean and median cash, wins only 4/10 versus `velocity_mill` and 2/10
versus `cash_conversion_mill`, and produces no episode above `$50,000` or
`$100,000`. It is not accepted as the next optimization baseline.
