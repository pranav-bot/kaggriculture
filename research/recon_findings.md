# Replay and Simulator Recon Findings

**Phase:** 1 findings and Phase 2 implementation contract  
**Status:** Phase 1 approved; Phase 2 engine implemented in `scripts/`

This document consolidates the completed replay-schema and simulator-hook
reconnaissance. It is an implementation contract for a later Phase 2; no
parser, surrogate, runner, or metrics module is implemented here.

## 1. Replay JSON contract

Each replay is a JSON object with these relevant top-level keys:

```text
configuration, description, id, info, module_version, name, rewards,
schema_version, specification, statuses, steps, title, version
```

`info` contains:

```text
Agents: [{Name, ThumbnailUrl}, ...]
EpisodeId
LiveVideoPath
TeamNames
seed
```

`steps` is a 720-element list for the default season. Each element is a
player-indexed list, normally `[player_0_frame, player_1_frame]`. A frame has:

```text
action, info, observation, reward, status
```

The action object has exactly these operational fields:

```text
farmer: [operation, ...arguments]
hands: [[operation, ...arguments], ...]
market: [[order, item, quantity], ...]
```

The observation object has:

```text
player, step, day, hour, farms, private, market, town,
remainingOverageTime
```

`farms` is the shared, player-indexed public array. Each farm contains:

```text
farmer: [x, y]
hands: [[x, y], ...]
hires_today
money
tiles: tiles[y][x]
unlocked_quadrants
```

Tile values are `null` (empty unlocked), `"LOCKED"`, or a structure/plant
object. Plant objects use `kind`, `crop`, `planted_day`, `watered_today`,
`consecutive_unwatered`, `yield_units`, `max_lifespan_step`, and
`fertilized_until_day`. Animal/structure objects use `kind`, `animal`,
`placed_day`, `yield_units`, `consecutive_unfed`, `fed_today`, `cared_today`,
`fertilizer_available`, and `pending_care_bonus` as applicable.

`private` is for the observing player only:

```text
inventories: [{item: quantity}, ...]  # main farmer, then hands
seeds: {crop: quantity}
shed: {product_or_animal: quantity}
```

`market` contains `inventory` and `prices`, keyed by product. It may also
contain resolved `params` when market overrides are configured. `town` contains
`unlocked_shops`, a list in which duplicate shop names are valid. The public
farm state does not expose the opponent's private shed or carried inventories.

## 2. Turn and player indexing

- `steps[t]` is the recorded frame list for turn index `t`; player `p` is
  `steps[t][p]`.
- `observation.player` identifies which player's private observation is being
  shown (0 or 1).
- `observation.farms[p]` is always the public farm for player `p`, independent
  of which player's observation is being viewed.
- `private.inventories[0]` is the main farmer; `inventories[i + 1]` is hand
  `i`.
- `tiles[y][x]` uses row `y`, column `x`; movement is `NORTH=(0,-1)`,
  `SOUTH=(0,1)`, `EAST=(1,0)`, `WEST=(-1,0)`.
- `day` and `hour` are zero-based; defaults are 24 turns per day and 720 turns
  per season. `step` is the framework turn counter.

The simulator records the post-interpreter state at the next replay index.
When reproducing an opponent action against a live environment, preserve the
runner's observed offset: the existing replay evaluator selects the recorded
opponent action at `opp_acts[obs.step + 1]`, with a PASS action after the
recorded sequence. A Phase 2 runner must test this alignment explicitly rather
than assuming `steps[obs.step]` is the next action.

## 3. Seed handling

The replay's authoritative reproducibility field is `info.seed`. The original
environment configuration may also contain `seed`; the Kaggle interpreter
resolves it at initialization, removes it from agent-visible configuration,
and stores the resolved value in `env.info["seed"]`. Agents therefore read
normal observations, not the seed. A replay runner should pass `info.seed` into
`make("kaggriculture", configuration={..., "seed": seed})` and retain the
same episode configuration (especially `episodeSteps=720`).

End-of-day stochastic refreshes use the resolved seed combined with the day,
so changing the seed or episode configuration changes weeds/shop evolution and
invalidates replay comparisons.

## 4. Reset/step hooks

Kaggriculture is a Kaggle interpreter environment, not a Gym class with
public `reset()` and `step()` methods in the simulator module.

- **Reset/initialization hook:** `interpreter(state, env)` detects that the
  observation has no farms and calls `_initialize(state, env)`. Initialization
  creates farms, private inventories, market, town, day/hour zero, and assigns
  each state slot its `observation.player`.
- **Step hook:** subsequent calls to `interpreter(state, env)` consume each
  state's `action`, apply unit actions, process market orders, apply town
  consumption, decay plants, perform end-of-day refreshes, advance
  `day/hour`, and update shared public observations.
- If `env.done` is already true, the interpreter returns state unchanged.
- The final status is set to `DONE` near the configured episode boundary and
  reward is terminal farm money.

The action schema supports unit operations `NORTH`, `SOUTH`, `EAST`, `WEST`,
`PASS`, `PICKUP`, `PLANT`, `WATER`, `HARVEST`, `FERTILIZE`, `BUILD_COOP`,
`BUILD_PASTURE`, `DIG`, `PLACE`, `FEED`, `COLLECT_FERTILIZER`, and `CARE`.
Market operations are `BUY_SEED`, `BUY_PRODUCT`, `BUY_ANIMAL`, `SELL`, `HIRE`,
and `BUY_LAND`.

## 5. Market validation and invalid-action behavior

At most `maxMarketOrdersPerTurn` orders are considered (default 10); extras
are silently dropped. Orders are processed in per-order position and
per-unit lockstep between players, so both players can affect the quote before
the next unit. Malformed order shapes, unknown operations/items, non-positive
quantities, unaffordable purchases, empty-shed sells, full sheds, unavailable
land, and unaffordable hires are rejected without raising an agent-facing
exception.

Unit actions are also silent no-ops when illegal (bad shape, unavailable
position, locked tile, missing item, wrong structure/crop state, etc.).
`PLANT` has one special atomic validation: if the combined farmer/hand plant
requests for a crop exceed private seed count, **all** plant requests for that
crop on that turn are replaced with `PASS`.

Successful sells decrement the private shed and increase farm cash. A sale at
price greater than 1 increases market inventory; a sale at price 1 does not.
Purchases consume cash and, where applicable, shed capacity. Seeds are held in
`private.seeds`, not the shed. Market orders beyond the first ten must not be
used for critical feed, land, hire, or animal purchases.

## 6. Terminal cash semantics

The terminal score/reward is the player's farm `money` after the final
interpreter transition. Cash is not the value of remaining shed goods,
carried goods, seeds, buildings, animals, or land. Unsold shed inventory and
goods still carried by workers therefore do not contribute unless liquidated
before termination. End-of-day shed overflow can be discarded at the configured
capacity, and terminal liquidation must account for workers still carrying
goods. Both player rewards are their respective terminal `farms[p].money`.

## 7. Proposed Phase 2 three-tier fallback hierarchy

The later evaluator should resolve each opponent turn in this order:

1. **Tier 1 — Exact replay action.** Use the recorded action for the selected
   player/turn when the replay has a matching seed, player index, and valid
   action frame. Preserve the action object without normalizing away empty
   `hands` or `market` lists.
2. **Tier 2 — Deterministic surrogate policy.** If a frame is missing,
   misaligned, or unusable, infer a legal action from the current observation
   using a deterministic policy: survival feed/water first, then harvest/care,
   critical market orders, sell pressure, useful movement, and PASS only when
   no legal fallback exists. This tier keeps an evaluation episode running
   while remaining reproducible.
3. **Tier 3 — Safe no-op baseline.** If parsing, observation shape, or surrogate
   inference fails, emit
   `{"farmer":["PASS"],"hands":[],"market":[]}`. Record the fallback reason and
   turn so metrics distinguish replay coverage from surrogate/no-op coverage.

The hierarchy should be explicit in Phase 2 metrics: counts and percentages
for exact replay, surrogate, and safe-no-op turns, plus invalid-action and
terminal-cash outcomes. Do not silently mix tiers.

## Phase boundary

Phase 1 was approved for continuation. The Phase 2 implementation now lives
in `scripts/replay_parser.py`, `scripts/surrogate_agent.py`,
`scripts/eval_runner.py`, and `scripts/eval_metrics.py`. Their detailed
assumptions, fallback counters, validation results, and remaining limitations
are documented in `research/replay_parser.md`,
`research/surrogate_agent.md`, and `research/phase2_eval_engine.md`.
