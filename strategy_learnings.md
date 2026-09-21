# Kaggriculture Strategy Learnings

## Baseline: Compound Expansion

- The full 720-turn standoff completed within the timeout budget: peak observed
  latency was below 30 ms and no overage was charged.
- It lost all eight swapped-side matches against the existing submissions.
- Its terminal cash was generally below 2,000, while melon and shop-focused
  agents reached roughly 18,000–25,000 in the same local configuration.
- Main failure mode: conservative investment and incomplete production loops.
  It hired temporary hands but did not consistently fill newly unlocked land,
  and it could leave livestock in the shed rather than producing sellable goods.

## Economic Findings

- The engine resets hired hands every day. Hiring is therefore a daily throughput
  decision, not a permanent payroll investment.
- Melons are one-time crops with six units at peak yield and a 250 base price.
  A full-board melon plan has substantially more upside than wheat or a small
  livestock allocation when watering capacity is available.
- The shed holds only 100 non-seed items. Harvesting and selling must be kept
  continuous; delaying all sales until the final days risks overflow.
- Town shops unlock every three days with replacement and consume every four
  turns. Post-drain selling is useful, but production throughput and shed
  capacity dominate the local score.
- The actual installed engine should be treated as authoritative when it differs
  from convenience-model constants in `src/kaggriculture`.

## Candidate Queue

### `melon_conveyor`

Aggressive baseline using the existing controller:

- Melon-only production
- Up to eight daily hands
- Automatic expansion and immediate sales
- No livestock diversion

Benchmark status: pending full 720-turn standoff.

### `melon_pipeline`

The first `melon_flood` run exposed a bootstrap failure: purchasing land plus 23
seeds left too little cash to fund the daily hands before the first harvest.
`melon_pipeline` starts with 20 melons, hires eight hands daily, delays expansion
until the first harvest, then buys land and seeds in bulk from realized cash.
Benchmark status: pending full 720-turn standoff.

### `sheep_mill`

The economics audit showed wool has a stronger long-run conversion than melons:
one sheep converts roughly one $25 wheat/day into a $200 wool unit every three
days. This candidate bootstraps with 20 melons, then builds pastures, buys sheep,
buys feed wheat, and explicitly schedules pickup/place/feed/care/harvest tasks.
Benchmark status: pending full 720-turn standoff.

### `melon_surge`

Explicit nearest-task scheduling with eight daily hands performed worse than the
generic controller in the first full-season runs. It planted only a small number
of tiles, showing that a naive one-task-per-turn router spends too much time
walking from the daily shed spawn. The next candidate keeps the generic routing
but removes its five-seed purchase bottleneck.

### `melon_flood`
Generic controller with eight daily hands, bulk melon seed purchases sized to
vacant land, immediate sales, and no livestock diversion. Benchmark status:
pending full 720-turn standoff.

### `market_velocity`
Adaptive crop-only candidate using live price/shop demand scoring, eight daily
hands, automatic fertilizer and selling, and a Day 26 investment cutoff. The
candidate intentionally avoids livestock because the sheep iteration converted
cash into shed animals faster than it could establish a feed/output cycle.
The first full 720-turn solo run reached 36,665 cash. In the 20-match standoff
it won 18 of 20 matches, losing only to `shop_opportunist`; peak measured
latency was 10.6 ms with zero overage usage. This is the strongest new
candidate, but it remains below the 80,000 target.

### Library Phase P0 (capacity + slot economy)

Implemented in `ActionController` per research/04 P0:

- `helpers/capacity_guard.py`: `projected_shed` / `projected_shed_from_action`,
  `clamp_sells`, `room_guard_99` (target occupancy 99 at hour 23),
  `planned_drop_inventory`, `dead_stock_sells` (not wired in controller yet).
- `market_planning.py`: `OPERATING_RESERVE=100` on `BUY_LAND`, `HIRE`,
  `BUY_SEED`, `BUY_ANIMAL`; SELL availability uses `shed + planned_drop`.
- `act()` pipeline: unit actions → overplant PASS cap → `planned_drop` → market
  plan → project shed → clamp sells → room guard on day-close turns.

Submission `alpha_velocity_p0` is `market_velocity` policy on the enhanced
controller (no extra strategy logic).

**720-turn benchmark (2026-09-22, solo vs `random`, one seed):**

| Agent | Terminal cash (player 0) |
|-------|-------------------------|
| `alpha_velocity_p0` | $33,317 |
| `market_velocity` | $34,947 |

Head-to-head on the same episode: `alpha_velocity_p0` $20,392 vs
`market_velocity` $19,585 (seat 0 vs 1). Policies are nearly identical; small
gaps are seat/opponent-interaction noise. P0 plumbing is validated by
`tests/test_controller.py` and `tests/test_capacity_guard.py` (18 tests).
Next ROI is P1 mix/hire/sell policy (Alpha production skeleton), not more P0
guards.

## Latest Benchmark

`market_velocity` is currently the best verified new strategy. Its main
remaining gap is adaptive execution against the stronger `shop_opportunist`
policy; further work should focus on reproducing that policy's demand response
without sacrificing the crop-only throughput and eight-hand labor budget.
Five additional full-season samples ranged from 25,800 to 34,668, confirming
that its observed performance is seed-sensitive and still far below 80,000.

### `shop_velocity`
Combines the shop-demand and animal-switching logic from `shop_opportunist`
with eight daily hands, lower sell friction, fertilizer collection, and a
Day-26 expansion cutoff. Benchmarking is in progress.

### `market_velocity_plus`
Removes livestock switching from `market_velocity`, reduces labor to four hands,
and uses a lower sell margin to preserve cash flow. This tests whether the
36,665 result came from crop throughput or from overspending on daily labor.
Its full 720-turn benchmark reached 21,862, so reduced labor and removing
livestock adaptation both hurt materially; it is rejected.

### `shop_compound`
Preserves `shop_opportunist`'s conservative two-hands-per-day policy and
adaptive crop production, while replacing the five-seed replenishment cap with
bulk purchases sized to vacant land. This isolates whether the incumbent's
advantage comes from liquidity control or from the seed cap.
Its full 720-turn benchmark reached 18,623, below the incumbent, so the
five-seed cap is not the primary limitation; the incumbent's conservative
sequencing is more important.

## Quant/Game-Theory Candidate Batch

The next candidate batch tests three independent signals:

- `quant_momentum`: demand-weighted price/margin scoring with moderate labor.
- `game_theory_supply`: opponent crop concentration as a scarcity signal.
- `liquidity_guard`: explicit land/labor cutoffs and a late fast-crop transition.

All candidates require full 720-turn benchmarks before acceptance.

### Batch results

Full 720-turn solo results: `quant_momentum` 30,762,
`game_theory_supply` 28,065, and `liquidity_guard` 24,002. The game-theory
candidate won most of its standoffs but still lost to `shop_opportunist` and
`shop_compound`; opponent supply is useful as a secondary signal, not as the
primary policy. `equilibrium_harvest` combines that signal with the incumbent's
liquidity settings and is the next candidate to benchmark.
Its full 720-turn benchmark reached 26,186, an improvement over several
variants but not a breakthrough toward 80,000.

## Production Bottleneck Audit

Terminal inspection showed that weak melon agents often ended with dozens of
weeded/expired tiles and little or no shed inventory. This means the dominant
constraint is recurring replant throughput and seed cash flow, not only initial
land acquisition. The next batch tests fast recurring carrots and a
wheat-to-carrot rotation against the adaptive incumbent.

### `microbatch_opportunist`
Tests the market microstructure hypothesis directly: retain the incumbent's
adaptive crop score and conservative two-hands policy, but sell only four units
on post-drain turns instead of dumping an entire shed stack. This is intended
to reduce quadratic price decay while avoiding shed overflow.
Its full 720-turn solo benchmark reached 26,767. The 40-match standoff lost
against the strongest adaptive agents, so four-unit execution lots alone do not
explain the incumbent's advantage.

## Incumbent Audit

Repeated full-season samples of `shop_opportunist` ranged from 23,081 to
32,578. Its advantage is reproducible in head-to-head standoffs, but it does
not approach 80,000 in the local authoritative engine. Increasing labor,
bulk-buying seeds, and enabling animal switching independently reduced results.

### `shop_velocity` result
The full 720-turn benchmark reached 21,647, below both `market_velocity` and
`shop_opportunist`. Eight hires combined with opportunistic animal switching
overcommits cash and is rejected.
