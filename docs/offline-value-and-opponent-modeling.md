# Offline Value, Opponent Modeling & PSRO Meta-Control

Three implemented workstreams on top of the `kaggriculture` feature library
(spatial tensor, global/market vectors, macro intents). All are covered by
`tests/test_iql_value.py` (5), `tests/test_hoarding_bocd.py` (17) and
`tests/test_meta_controller.py` (12); full suite: 151 passed.

## 1. IQL Value Network (`scripts/train_iql_value.py`)

Offline state-value function V(s) for the 720-turn economy, trained with
Implicit Q-Learning expectile regression (τ=0.7) on the `kaggriculture-il`
episode store. The loss regresses V(s) onto in-sample undiscounted terminal
cash deltas only — no policy is ever queried on out-of-distribution actions.

- **Inputs**: (23,10,10) farm-grid tensor (`features/spatial.py`) + 65-dim
  continuous vector (30-dim global + 35-dim market).
- **Target**: `return_to_go = (terminal_cash − cash_t) / 100000`.
- **Model**: `IQLValueNet` — 3-layer spatial CNN + vector MLP + fused head
  (79,105 params, ~0.32 MB fp32).
- **Run**: `.venv/bin/python scripts/train_iql_value.py --num-episodes 150
  --stride 10 --epochs 10` → `experiments/iql_value/iql_value_net.pt`
  (0.33 MB, under the 20 MB Kaggle tarball budget). Reference run: train
  expectile 0.067→0.010, val 0.0247→0.0144 (best epoch 8), val MAE ≈ 12.7k
  cash. Bundle includes config (τ, target scale, dims) for inference as
  `cash_now + V(s)·1e5`.

## 2. BOCD Hoarding Detection (`scripts/quant_full_product.py:613+`)

Protects against simultaneous-dump price collapses (engine-exact: +100 shared
surplus drives both Milk, linear, and Wool, quadratic, to the $1 floor).
Fertilizer is the tripwire: nothing consumes it (town drain = 0), so its
market-inventory deltas equal executed player sells.

- `parse_opponent_state` — public-state accounting per turn:
  `opponent_fertilizer_sold`, `opponent_cash_reserves` (exact on clean fills;
  partial-fill noise absorbed downstream).
- `BayesianChangePointDetector` — Gaussian Adams–MacKay BOCD; decision
  statistic is posterior mass on short run lengths (recent-regime-shift
  probability). Steady-series peak 0.076 vs shift peaks 0.87–0.95 (20 seeds).
- `HoardingMonitor` — 24-turn window aggregation; latches
  `OPPONENT_HOARDING_DETECTED` when P > 0.85 **and** post-change mean drops
  (cessation signature; upward surges ignored). Flag persists until `clear()`.
- Override — `holding_penalty_for_inventory` ($320/Milk, $400/Wool per unit),
  `adjust_beam_score`, `apply_hoarding_override_to_policy` (start_day=0,
  floor=$1, unbounded cap → proven full liquidation in `simulate_product`).
- Beam hook — `step_level_beam_search(..., holding_penalty_fn=None)` in
  `scratch_grandmaster.py:949`; default off, subtracts the penalty from leaf
  scores when supplied.
- **Run**: `--hoarding-demo` (synthetic selling→hoarding fires at P=0.872,
  54.3→0.0) and `--hoarding-replay <episode.gz> --replay-seat 0` (720-turn
  validation, no false latch).

## 3. PSRO MetaController (`src/kaggriculture/meta/__init__.py`)

Hierarchical top level above Beam Search. Holds three MacroIntent priors
(24-dim, over `MACRO_INTENT_CLASSES`): Policy_A Milk Dominance, Policy_B
Strawberry/Insulated-Crop Defense, Policy_C Deceptive Poisoned Well.

- `update(obs, seat, bocd_status)` — one meta-tick: perceive → infer → gate.
- `infer_latent_strategy` — Melon-rush tripwire: opponent SW empties > 5
  **and** cash hoard > $3000 → `MELON_RUSH`; else hoarding/pastures ≥ 8 →
  `MILK_FLOODER`, plants ≥ 15 → `STRAWBERRY_CONTINGENCY`, rich+idle →
  `CONSERVATIVE_SAVER`, else `UNKNOWN`.
- `gate` (`RESPONSE_MATRIX`) — instant best-response switch: flooder→B,
  melon-rush→C, strawberry/saver→A. Policy_B entry auto-queues DIG orders
  for **unoccupied** pastures + weeds only (`plan_transition`);
  `pop_transition_commands` renders greedy nearest-unit mechanical actions
  (DIG on arrival), draining the queue while Beam Search runs concurrently.
- `profile_for_beam` — packages policy id, intent prior, top-k action space,
  live transition queue, and the Milk/Wool penalty hook (active iff hoarding
  latched) for the search layer.
- Expansion edge detection — the controller diffs the opponent's public
  unlocked-quadrant set across ticks; growth raises
  `OPPONENT_EXPANSION_DETECTED` (latent `EXPANSION_RUSH` → Policy_B) on the
  exact observable turn, with a 72-turn dwell that holds the response while
  the irreversible event is digested (no flap-back on heuristic noise).

## 5. Ghost-Opponent Test (`scripts/ghost_boey_test.py`)

Exploitability check without self-play: Boey's Rank-1 replay
(`replays/other_agents/rank1/112542379.json`) is fed turn-by-turn through
BOCD + MetaController from the opponent's seat view, asserting the expansion
flag fires on the log-derived ground-truth turn and Policy_A → Policy_B
engages within 24 turns and holds. Verified: NE flag at turn 146 (Day 6),
0-turn lag. Resolved during verification: forensics docs said NE lands
"Day 5" (corrected to Day 6, step-anchored); the $170,961 gauntlet-table
figure is Boey-tape-vs-candidate cash, while the file logs $130,115.

## 4. PSRO League (`scripts/psro_league.py`)

Fictitious-Play master loop for the non-transitive economy. No single policy
is robust, so the league maintains an empirical game, solves its Nash
mixture, trains a Best Response against that mixture, and repeats.

- **Init**: baselines copied from `submissions/{melon_rusher, care_mill,
  agent_final}` into `experiments/psro_league/members/`.
- **Payoffs**: round-robin `kagg tournament` over all pairs x 100 seeds
  (`--seeds`) x both seats; `summary["matrix"]` mean scores (win=1, draw=0.5)
  form M. Resumable via the tournament output dir.
- **Meta-solve**: `fictitious_play` on the antisymmetrized zero-sum game
  `(M − Mᵀ)/2` → Nash mix + exploitability (RPS test game recovers uniform
  ±0.05, exploitability < 0.05).
- **Best response**: grid search over `ActionController` hyperparameters
  (crop, sell margin, hiring) maximizing expected score vs the Nash mixture;
  winner materialized as `members/br_iter{k}/main.py` + `meta.json`.
- **Output**: Nash-weighted ensemble dir (`main.py` commits per-episode to a
  Nash-sampled member; `nash.json` records weights) — buildable with
  `scripts/build_submission.py --agent <ensemble>/main.py --format tar`.
- **Run**: `.venv/bin/python scripts/psro_league.py --iters 3`
  (smoke: `--seeds 8 --eval-seeds 4 --iters 1`); `--final-only` re-emits the
  ensemble from the last solved Nash.

## Ladder-Ghost Pipeline (`scripts/ladder_ghost.py`)

Turns a parsed ladder loss into a trainable league opponent. `loss_analysis.json`
(build order: `EXPAND_*`, `HIRE`, `BUY_SEED/ANIMAL`, `PLANT`, `SELL/DUMP` with
day/count) is validated and day-sorted, then rendered into a deterministic
`submissions/ladder_ghost_<EPISODE_ID>/main.py`: one-shot market orders via a
blind per-episode ledger, seed provisioning, shed liquidation, and field work
through our vendored Kuhn-Munkres routing layer (scipy, greedy fallback).
The ghost is evaluated with the grandmaster (`agent_final`) plus league context
through the PSRO meta-solver (round-robin matrix → Fictitious Play Nash). If the
grandmaster's win rate vs the ghost drops below 0.5, Optuna HPO (default 12
trials) searches Beam Search weights (3 intent-prior scales, holding-penalty
multiplier, 2 sell margins, top-k intents) on a tunable MetaController hybrid,
and ships the winner as `submissions/ladder_counter_<ID>/`. Verified live:
Boey ghost injected, grandmaster holds 1.000 (no HPO); forced HPO run completed
3 real trials with counter materialization.

## 4. Counterfactual Market Data Generator (`kaggriculture.counterfactual.generator`)

Retroactively evaluates alternative market decisions in historical replay episodes to generate `counterfactual_value` regression targets for market sequence forecasting.

- **Market-Critical Trigger**: Identifies whenever a historical player submitted a `SELL` market order for `MILK` or `WOOL` with quantity strictly $> 10$ (`detect_large_sells`).
- **Counterfactual Action Injection**: Constructs a synthetic `HOLD` action by withholding/suppressing the large sell order while preserving physical worker moves, field jobs, and non-target market orders (`create_hold_action`).
- **Rust Simulator Integration**: Initializes the Rust environment (`kaggsim.env.Env(state=...)`) at the exact historical timestep $t$ via `LOADSTATE`.
- **Heuristic Rollout & Evaluation**: Steps the counterfactual action pair, then executes a fast heuristic rollout policy (defaulting to the apex grandmaster agent from `scratch_grandmaster.py`) across all remaining steps to terminal step 719.

## 5. Causal Market Forecaster (`kaggriculture.models`)

`MarketForecaster` is a lightweight causal GRU for predicting market
inventory deltas from counterfactual and replay trajectories.

- **Input window**: exactly 48 hourly rows.
- **Features**: current price and inventory, observed volume and demand,
  opponent cash reserves, opponent-visible cows, cyclical hour features,
  price/inventory deltas, and one-hot active-town-shop indicators.
- **Output**: six future means and log variances, covering the next 1–6
  turns (4–24 hours).
- **Training loss**: `gaussian_nll_loss(mean, log_variance, target)` for a
  diagonal Gaussian predictive distribution.
- **Windowing**: `MarketWindowDataset` returns leak-free historical windows
  and the following six-step target.
- **Liquidation integration**:
  `get_optimal_liquidation_volume(current_inventory, predicted_price_curve)`
  evaluates the canonical nonlinear market curve unit by unit and stops at
  the reservation value or the `$1` floor. Use `max_volume` to impose an
  operational cap.

Example:

```python
from kaggriculture.models import MarketForecaster

forecaster = MarketForecaster()
mean, log_variance = forecaster(window_48h)
volume = forecaster.get_optimal_liquidation_volume(
    inventory, mean.detach(), item="WOOL"
)
```

## 6. Submission Packaging and Gauntlet

`scripts/build_submission.py` now packages the agent entrypoint, sibling
modules, the shared library, selected `.pt` weights, and a static Linux
`kagg` binary. It rejects host-native binaries, enforces the 20 MiB per-weight
limit, and excludes `__pycache__`, `.pyc`, and `.pyo` files.

```bash
python scripts/build_submission.py \
  --agent two_team_grandmaster \
  --weights experiments/iql_value/iql_value_net.pt \
  --rust-binary /path/to/x86_64-unknown-linux-musl/release/kagg
```

`scripts/test_submission.py` provides:

- `profile_agent_calls`, which requires 720 real observations and asserts
  every call is at most 50 ms with at most 5 seconds cumulative compute.
- `run_ab_evaluation`, which configures 100 seeds with both seat orders and
  derives paired McNemar statistics from the Rust tournament result JSONL.

The available development binary may be macOS arm64; a compatible static
Linux/musl binary is required for the final Kaggle tarball.
- **Value Target Calculation**: Computes `counterfactual_value = synthetic_terminal_cash - historical_terminal_cash`.
- **Multiprocessing Parallelization**: `CounterfactualPipeline` distributes replay episodes across CPU worker processes without pipe contention, streaming JSONL records to disk.
- **CLI Runner**: `scripts/generate_counterfactual_dataset.py` with `--episodes-dir`, `--output`, `--workers`, `--max-episodes`.

## 5. Strategic Stuttering: Option-Critic Meta-Controller (`MacroOptionManager`)

Located in `scratch_grandmaster.py`, this meta-controller radically compresses the 720-step search horizon into ~60 macro-decisions to operate well within Kaggle's 1.0s/turn soft limit and 60s overage bank.

- **Temporal Anchors**: Restricts full Step-Level Beam Search to exactly two triggers per day:
  - **Hour 0 (Dawn)**: hire farmhands, set day plan.
  - **Hour 12 (Midday)**: re-evaluate market conditions, adjust herd/crop ratio.
- **Option Lock-In**: Locks in a `MacroIntent` (labels, trajectory indices, leaf value) for the subsequent 11 hours.
- **Mechanical Bypass**: For the intervening 11 hours, neural-network beam search is completely bypassed. The locked intent is routed directly through the Kuhn-Munkres bipartite matching layer (`kuhn_munkres_route`) with zero inference latency.
- **Emergency Interrupt**: `BayesianMarketPredictor` monitors opponent market inventory deltas. If `OPPONENT_HOARDING_DETECTED` fires mid-cycle, the active option is instantly terminated and an out-of-cycle search is executed.

## 6. "Poisoned Well" Market Trap Execution (`PoisonedWellTrap`)

Adversarial game-theory execution layer in `scratch_grandmaster.py` exploiting asymmetric price degradation mechanics:
- **Game Mechanics**: Milk prices drop *linearly* with inventory surplus, while Wool drops *quadratically*. Town shops consume inventory every 4 hours, naturally recovering prices.
- **Phase 1 (`DECEPTIVE_MODE`)**: Activates when private shed holds $>20$ units of Milk or Wool. Forcefully plants 2–3 Wheat plots in publicly visible tiles (NW quadrant) to signal livestock expansion, while strictly suppressing sells of the buffered commodity.
- **Phase 2 (`RECOVERY_WATCH_MODE`)**: Triggered when a large positive market inventory delta ($\ge 15$ units) confirms the opponent panic-dumped into the market and crashed the price. Tracks 4-hour town shop consumption ticks (`PIZZA_SHOP`, `YARN_STORE`, etc.) while continuing to withhold inventory.
- **Phase 3 (`SELL_EXECUTE`)**: Once town shop consumption recovers wholesale price above `base_price * 0.85` (or after 4 consumption ticks), dumps our buffered stockpile at the recovered premium.
- **Unified Controller**: `GrandStrategyController` chains `MacroOptionManager.act()` with `PoisonedWellTrap.update()` as a single `act(obs)` call per turn.

## 7. Test Suite Status & Verification

All 56 end-to-end and boundary test cases are passing:
- `test_r1_eda_schema.py`: 6 passed
- `test_r2_features.py`: 9 passed
- `test_r3_counterfactual.py`: 6 passed
- `test_r4_dataset_loader.py`: 6 passed
- `test_subtask6_counterfactual_generator.py`: 4 passed
- `test_r1_boundaries.py`: 7 passed
- `test_r2_boundaries.py`: 6 passed
- `test_r3_boundaries.py`: 6 passed
- `test_r4_boundaries.py`: 6 passed
- **Total**: `56 passed, 1 warning in 3.31s`
