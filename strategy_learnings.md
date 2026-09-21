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

#### P0 metrics

`scripts/compare_agents.py` + `docs/metrics-playbook.md` (2026-09-22): 10 seeds,
solo vs `random`, 720 steps, CSV `standoff/p0_metrics_compare.csv`.

| Agent | Mean cash | Median cash |
|-------|-----------|-------------|
| `market_velocity` | $29,958 | $30,982 |
| `alpha_velocity_p0` | $33,473 | $31,629 |

`alpha_velocity_p0` higher terminal cash on 6/10 seeds; mean Δ ≈ +$3.5k for
P0 vs MV (high variance on individual seeds). Telemetry: `overflow` and
`slots_burned` were 0 for both on this panel; `decide_ms_p95` &lt; 1 ms for
both. Treat as “P0 does not hurt safety on solo random”; cash edge is seed-noisy
— confirm with more seeds or head-to-head before claiming a win over MV.

### Library Phase P1 + `alpha_velocity_p1`

`helpers/phase_brain.py` adds champion mix (`EARLY_MIX` / `LATE_MIX`,
`MIX_SWITCH_DAY=13`), `MAX_ACTIVE=40`, shop `adaptive_shift` ±3,
`HANDS_BY_UNLOCKED` {1:6, 2:9}, `PREMIUM_BATCH=8`, terminal return
(day≥29, hour≥13) and day-29 forced harvest in `ActionController` when
`enable_terminal_return` / `enable_alpha_planting` are set.

`market_planning.py` `alpha_p1` mode: bulk seed buys to mix deficits,
hire-by-unlocked, batched premium sells, full sells when `shed_total≥82`,
day≥26 land/hire cutoff (same as `market_velocity`).

**720-turn benchmark (2026-09-22, one random seed per solo run):**

| Agent | Solo vs random |
|-------|----------------|
| `alpha_velocity_p1` | $32,596 |
| `market_velocity` | $28,798 |
| `shop_opportunist` | $23,968 |

**Mini standoff (4 matches: P1 vs MV and vs shop, both seats):** P1 won
0/4 on this seed (shared-market seat interaction; solo cash lead does not
translate to head-to-head yet). Tuning mix planting throughput and sell
timing under opponent pressure is the next iteration.

### Innovation 2 — `demand_mpc` (sell MPC)

`helpers/sell_mpc.py`: `plan_sell_horizon()` uses
`predict_upcoming_consumption_ticks` + `simulate_sell_slippage` to choose
sell-now vs wait (next drain within 2 turns when `shed_total < 85`).
`submissions/demand_mpc/main.py` runs P1 buys/hire/seed but replaces alpha_p1
SELLs with MPC orders.

**Solo vs `random`, 5 seeds (2026-09-22), `episode_metrics`:**

| Agent | Mean cash | Mean est. revenue / sold unit |
|-------|-----------|-------------------------------|
| `alpha_velocity_p1` | $33,470 | $61.50 |
| `demand_mpc` | $31,046 | $58.73 |

MPC improved sell timing on some ticks but reduced volume sold; net cash
regressed on this panel. Keep helper for calendar-aware overlays; tune
wait threshold or blend with batched premium sells before submission.

### Phase P2 + `alpha_shop_hybrid`

`ActionController` P2 path (when `enable_market_microstructure=True`):
after P0 guards → `rank_sell_slots` → `collision_guard` only if
`opponent.clone_like(obs)`.

`alpha_shop_hybrid`: P0+P1 (`alpha_p1` market policy), hybrid MV+shop crop
scores for planting, `auto_feed_animals=False`.

**20×720 vs `shop_opportunist` (10 rounds, both seats):** 0W/20L, mean cash
hybrid ~15.5k vs shop ~34.2k (`standoff/alpha_shop_hybrid_vs_shop.json`).
Collision gating and sell ranking are wired; hybrid policy still loses badly
to the incumbent in shared-market play.

### Innovation 4 — `regime_counter`

`helpers/regime_counter.py` classifies `OpponentProfile` into
`melon_rush` / `yarn_sheep` / `aggressive_expander` / `mixed_inactive` and
maps to `RegimeKnobs` (mix weights, plant priority, sell-delay flags,
`collision_aggressive`, expander hire pacing). `submissions/regime_counter/main.py`
extends the shop hybrid stack and overrides planting mix, sell filtering,
collision gating, and hire boost.

**20×720 vs `shop_opportunist` (10 rounds, both seats):** 0W/20L, mean cash
regime_counter ~11.2k vs shop ~34.1k (`standoff/regime_counter_vs_shop.json`).
Regime counters did not close the gap vs shop; classifier + delays may be
over-holding inventory against an active shop planner.

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
