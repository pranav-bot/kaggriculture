# Kaggriculture

Agent development toolkit for the [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) Kaggle competition. Agents live in `submissions/<name>/main.py` and use the shared `kaggriculture` library in `src/kaggriculture/`.

## Setup

Requires Python 3.12+.

```bash
# Install dependencies (uses uv; pip works too)
uv sync

# Or activate the virtual environment directly
source .venv/bin/activate
```

You also need the Kaggle CLI configured if you plan to submit:

```bash
pip install kaggle
kaggle competitions download -c kaggriculture   # confirms credentials work
```

Built artifacts are written to `build/` (gitignored).

**New to agent development?** Read **[`docs/creating-agents.md`](docs/creating-agents.md)** — full guide from game mechanics to advanced agents. Runnable tutorials live in [`examples/agents/`](examples/agents/).

---

## High-Performance Simulation Engine (`kaggriculture-simulation`)

The project is integrated with **`kaggriculture-simulation`**, a byte-identical Rust port of the Kaggle engine operating at **293 episodes/sec** (~210,000 steps/sec) natively, and **~0.30s per 720-step match** for Python agents.

See **[`docs/kaggriculture_simulation_guide.md`](docs/kaggriculture_simulation_guide.md)** for architecture, benchmarks, and API usage.

### Causal market forecasting

`kaggriculture.models.MarketForecaster` is a lightweight causal GRU for
counterfactual market data. It consumes exactly 48 hourly feature rows:
market price and inventory, observed volume and demand, opponent cash
reserves, opponent-visible cows, cyclical hour features, price/inventory
changes, and one-hot active-town-shop indicators. The model returns six
future inventory-delta means and log variances (covering the next 1–6 turns);
`gaussian_nll_loss` trains the diagonal Gaussian output.

```python
from kaggriculture.models import MarketForecaster, get_optimal_liquidation_volume

model = MarketForecaster()
mean, log_variance = model(window_48h)
safe_volume = get_optimal_liquidation_volume(
    current_inventory=inventory,
    predicted_price_curve=mean.detach(),
    item="WOOL",
)
```

The liquidation helper evaluates the repository's nonlinear market curve one
unit at a time, stops at the `$1` floor or when the marginal quote is no
longer above the forecast reservation value, and accepts an optional
`max_volume` cap. Forecaster tests require the project's PyTorch and pytest
dependencies; without them, source compilation can still be checked with
`python -m py_compile`.

---

## Scripts Guide & Research Infrastructure

The `scripts/` directory provides an end-to-end quantitative research, replay parsing, optimization, and evaluation infrastructure. All evaluation tools automatically leverage the high-performance Rust simulator by default (with seamless `--official` fallback).

See **[`research/21_Scripts_Reference_and_User_Guide.md`](research/21_Scripts_Reference_and_User_Guide.md)** for the complete 27+ tool reference and **[`research/20_Replay_Forensics_and_Gauntlet_Findings.md`](research/20_Replay_Forensics_and_Gauntlet_Findings.md)** for forensic findings from elite ladder matches.

Key tools in workflow order:

1. **`scripts/eval_against_replays.py`** — Gauntlet benchmark: test agents against 20 top-tier real-world ladder replays in ~6–8s (Boey, Clement Ling, DECEM, Kaggledew Valley, BenPalmer59, etc.)
2. **`scripts/h2h_bench.py`** — Fast head-to-head multi-seed benchmark between two agents (~0.2s/match)
3. **`scripts/trace_replay_match.py`** — Forensic simulation tracer comparing live actions against historical opponent steps turn-by-turn (~0.5s)
4. **`scripts/eval_cash.py`** — Evaluates terminal cash across 14 benchmark seeds (1, 3, 5, 7, 10, 15, 20) in ~4s
5. **`standoff/run_standoff.py`** — Fast round-robin tournament across all 62 agents in `submissions/` (~23s total)
6. **`scripts/pipeline_offline_rl.py`** — Unified Offline RL pipeline runner: EDA schema validation, feature transformation (spatial tensors, macro-intents, net worth returns), chunked caching, and PyTorch DataLoader benchmarking (>132k trans/sec)
7. **`scripts/generate_counterfactual_dataset.py`** — Multiprocessing counterfactual data generator: forks replay states in Rust `kaggsim.env.Env`, injects HOLD market actions, executes fast heuristic rollouts, and streams tuples to `.jsonl`
8. **`scripts/quant_full_product.py`** — SciPy-based multi-product pricing and liquidation optimizer; also hosts the BOCD opponent-hoarding detector (`--hoarding-demo`, `--hoarding-replay`)
9. **`scripts/train_iql_value.py`** — IQL expectile value network (τ=0.7) → `experiments/iql_value/iql_value_net.pt`
10. **`scratch_grandmaster.py`** — Apex division-of-labor grandmaster with `MacroOptionManager` (Hour 0/12 Option-Critic gating), 11-hour Kuhn-Munkres bypass, and `PoisonedWellTrap` (deceptive Wheat signaling, 4h shop recovery tracking)
11. **`src/kaggriculture/meta/`** — PSRO MetaController switching MacroIntent profiles above Beam Search
12. **`scripts/psro_league.py`** — PSRO Fictitious-Play league: payoff matrix via `kagg tournament`, Nash solve, best-response training, Nash ensemble
13. **`scripts/find_best_agent.py`** — Master evaluation: 500-seed round-robin over all submissions + elite tape gauntlet, Elo/McNemar/latency Markdown leaderboard
13. **`scripts/opening_book_generator.py`** — Replay consensus parser & runtime `OpeningBookController` for deterministic Days 0–15 expansion books
14. **`scripts/run_elo_tournament.py`** — Master evaluation script & local Elo ladder engine wrapping `kagg tournament` with McNemar A/B test CIs and 5.0ms/turn latency profiling gate
15. **`scripts/solve_liquidation.py`** — Retrograde DP & MILP terminal liquidation solver calculating Days 20–29 daily quotas and 6-tick shop synchronizations
16. **`scripts/test_submission.py`** — Instant local match simulator (<1ms/turn) and Kaggle validator
17. **`scripts/build_submission.py`** — Package and validate a Kaggle-ready submission (single-file or tar.gz)

### `scripts/test_submission.py`

Runs a head-to-head match in `kaggle_environments` and prints final cash balances, rewards, and per-turn timing.

#### Usage

```bash
python scripts/test_submission.py [options]
```

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--agent1`, `-a1` | `shop_opportunist` | First player |
| `--agent2`, `-a2` | `random` | Second player (opponent) |
| `--steps`, `-s` | `720` | Number of turns (720 = full 30-day season) |
| `--render`, `-r` | off | Save an HTML replay to `match_replay.html` |

#### Resolving agents

Each `--agent` value is resolved in this order:

1. **Filesystem path** — any existing file (e.g. `submissions/melon_rusher/main.py` or `build/submission.tar.gz`)
2. **Template name** — looks up `submissions/<name>/main.py` (e.g. `melon_rusher`)
3. **Built artifact** — looks up `build/<name>` (e.g. `build/main.py`)
4. **Built-in opponents** — `random`, `pass`, or `starter` (Kaggle environment defaults)

#### Examples

```bash
# Full season vs the random baseline (default)
python scripts/test_submission.py

# Compare two local templates
python scripts/test_submission.py -a1 market_velocity -a2 shop_opportunist

# Quick smoke test (one day)
python scripts/test_submission.py -a1 wheat_loop -a2 random -s 24

# Test a built tar.gz before uploading
python scripts/test_submission.py -a1 build/submission.tar.gz -a2 random

# Save a visual replay
python scripts/test_submission.py -a1 shop_opportunist -a2 melon_rusher --render
```

#### Output

On success the script prints:

- Total wall-clock time and average milliseconds per turn
- Final cash and reward for each player
- Win/loss/tie verdict

With `--render`, open `match_replay.html` in a browser to step through the match visually.

---

### `scripts/build_submission.py`

Packages an agent plus the `kaggriculture` library into a Kaggle-compliant artifact, optionally validates it, and prints (or runs) the `kaggle competitions submit` command.

#### Usage

```bash
python scripts/build_submission.py [options]
```

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--agent`, `-a` | `shop_opportunist` | Template name or path to `main.py` |
| `--format`, `-f` | `tar` | `tar` (recommended) or `single` |
| `--no-validate` | off | Skip the dry-run match against `random` |
| `--weights PATH` | auto-discovered | Include a PyTorch `.pt` file; repeatable, each file must be under 20 MiB |
| `--rust-binary PATH` | auto-discovered | Static Linux `kagg` executable to include in the archive |
| `--message`, `-m` | auto-generated | Submission message for the Kaggle CLI |
| `--submit`, `-s` | off | Submit directly via `kaggle competitions submit` |

#### Submission formats

**`tar` (default, recommended)**

Creates `build/submission.tar.gz` containing:

```
submission.tar.gz
├── main.py          # your agent entrypoint
├── *.py             # sibling modules from the same submission folder (if any)
├── kaggriculture/   # full library copy
├── *.pt              # selected IQL/value weights under 20 MiB each
└── kagg              # static Linux Rust simulator binary
```

Kaggle extracts the archive with `main.py` at the root. This is the preferred format because it keeps agent code separate from the library and mirrors the multi-file layout used during development.

The builder rejects host-native binaries and requires an ELF Linux executable
that is statically linked (or built with musl). The macOS development binary
under `kaggriculture-simulation/src-rust/target/release/kagg` cannot be
packaged for Kaggle; build an `x86_64-unknown-linux-musl` release binary and
pass it explicitly:

```bash
python scripts/build_submission.py \
  -a two_team_grandmaster \
  --weights experiments/iql_value/iql_value_net.pt \
  --rust-binary /path/to/x86_64-unknown-linux-musl/release/kagg
```

Archives exclude `__pycache__`, `.pyc`, and `.pyo` files. Weight discovery
includes explicitly supplied files, weights next to the selected agent, and
`.pt` files under `experiments/`; use `--weights` when the intended artifact
must be unambiguous.

**`single`**

Creates `build/main.py` — one self-contained file that inlines these library modules in dependency order:

- `env/items.py`, `env/models.py`
- `actions/actions.py`, `actions/market_planning.py`, `actions/controller.py`
- `helpers/tracking.py`, `helpers/market_prediction.py`, `helpers/solver.py`, `helpers/opponent.py`
- your agent's `main.py`

Local `kaggriculture` imports and `sys.path` hacks are stripped automatically. Use this format only if the competition requires a single file.

#### Validation

Unless `--no-validate` is passed, the script runs a 720-turn match (`agent` vs `random`) through `kaggle_environments.make("kaggriculture")`. Validation must pass before submission instructions are printed. Common failure causes:

- Syntax errors in the bundled code
- Missing or non-callable `agent(obs)` function in `main.py`
- Runtime exceptions during a turn
- Missing static Linux `kagg` binary when building the tar format

### Latency profiling and A/B evaluation

`scripts/test_submission.py` exposes two strict helpers for deployment
validation:

```python
from scripts.test_submission import profile_agent_calls, run_ab_evaluation

profile = profile_agent_calls(agent, observations)
ab = run_ab_evaluation("candidate/main.py", "sovereign_apex/main.py")
```

`profile_agent_calls` requires exactly 720 observations by default and raises
`AssertionError` if any `agent(obs)` call exceeds 50 ms or if cumulative agent
compute exceeds 5 seconds. It returns total, maximum, median, and p95 timing
statistics. Feed it the real observation stream from a full match rather than
synthetic observations.

`run_ab_evaluation` requires exactly 100 seeds, configures the Rust `kagg
tournament` runner with both seat orders, and, when tournament result files are
available, adds paired McNemar statistics from the completed result rows.
Install/build the Rust toolkit and make its Python package importable before
using this integration. If the tournament package is unavailable, provide a
real `fallback_runner`; the helper will not fabricate match results.

#### Examples

```bash
# Build and validate the default agent
python scripts/build_submission.py

# Package a specific template as tar.gz
python scripts/build_submission.py -a market_velocity

# Build from a custom main.py path
python scripts/build_submission.py -a submissions/melon_rusher/main.py

# Produce a single-file submission, skip validation
python scripts/build_submission.py -a wheat_loop -f single --no-validate

# Build, validate, and submit in one step
python scripts/build_submission.py -a shop_opportunist -m "shop opportunist v2" --submit
```

#### After building

On success the script prints the exact Kaggle CLI command:

```bash
kaggle competitions submit kaggriculture -f "build/submission.tar.gz" -m "Kaggriculture shop_opportunist (tar)"
```

---

## Available agent templates

Each directory under `submissions/` is a named template you can pass to either script:

| Template | Description (from naming) |
|----------|---------------------------|
| **`two_team_grandmaster`** | **Two-Team Division of Labor: Segregated Livestock (Units 0-2) & Field Teams (Units 3-7), Day 0-7 Fertilizer Liquidation, Multi-Quadrant Scaling** |
| **`sovereign_apex`** | **Demand-Coupled Herd Scaling & Dual-Species Pivot with Strawberry Diversification** |
| **`apex_engine`** | **Day 0 Melon Kickstart, Day 5-6 NE Unlock, Day 10 Melon Cash Influx funding SW Expansion** |
| **`straw_empire`** | **Strawberry Cash-Crop Specialization with early fertilizer boost** |
| **`care_mill`** | **Cared-Cow production loop with wheat feed management** |
| **`velocity_mill`** | **Adaptive town shop demand velocity controller** |
| `carrot_compound` | Carrot-focused compound strategy |
| `compound_expansion` | Conservative expansion baseline |
| `delta_ranked_velocity` | Market-velocity controller with impact-ranked sells |
| `equilibrium_harvest` | Harvest-timing equilibrium play |
| `feed_squeeze` | Livestock feed pressure strategy |
| `game_theory_supply` | Supply/demand game-theory approach |
| `liquidity_guard` | Cash-reserve defensive play |
| `market_velocity` | Adaptive crop scoring with market demand |
| `market_velocity_plus` | Extended market-velocity variant |
| `melon_conveyor` | Aggressive melon-only baseline |
| `melon_flood` | Bulk melon seeding with generic controller |
| `melon_pipeline` | Delayed expansion melon bootstrap |
| `melon_rusher` | Fast melon rush |
| `melon_surge` | Explicit nearest-task melon scheduling |
| `microbatch_opportunist` | Small-batch opportunistic selling |
| `quant_momentum` | Momentum-based market timing |
| `sheep_mill` | Sheep/wool production loop |
| `shop_compound` | Shop-aware compound strategy |
| `shop_opportunist` | Adaptive shop demand exploitation (default) |
| `shop_velocity` | Shop-timing velocity strategy |
| `wheat_carrot_rotation` | Wheat/carrot rotation |
| `wheat_loop` | Simple wheat production loop |

See `strategy_learnings.md` for benchmark notes and design rationale.

---

## Typical workflow

```bash
# 1. Edit or create an agent
#    submissions/my_agent/main.py must expose: def agent(obs) -> action

# 2. Quick local test
python scripts/test_submission.py -a1 my_agent -a2 random -s 24

# 3. Full-season benchmark against a strong opponent
python scripts/test_submission.py -a1 my_agent -a2 shop_opportunist

# 4. Package and validate
python scripts/build_submission.py -a my_agent

# 5. Submit (or add --submit to step 4)
kaggle competitions submit kaggriculture -f build/submission.tar.gz -m "my_agent v1"
```

### Agent contract

Every `submissions/<name>/main.py` must define a top-level callable:

```python
def agent(obs) -> dict:
    """Return an action dict for the current observation."""
    ...
```

During development the agent imports from `kaggriculture` normally. The build script copies (or inlines) the library so the submission is self-contained on Kaggle.

---

## Standoff (`standoff/run_standoff.py`)

The standoff runner is the primary **round-robin benchmark** for comparing a candidate agent against every other template in `submissions/`. Use it after local smoke tests (`test_submission.py`) and before packaging for Kaggle.

### What it does

1. Discovers all agents under `submissions/*/main.py`.
2. Runs your selected agent against every other template for a **full 720-turn season** (30 days).
3. By default, **swaps starting positions** — each pairing is played twice so neither side has a map-order advantage.
4. Wraps each agent in a `TimedAgent` that measures per-turn latency and enforces Kaggle-style time limits locally.
5. Writes structured JSON results to `standoff/results.json` (or a custom path).

With 21 templates, a full standoff runs **40 matches** (20 opponents × 2 sides) and takes several minutes.

### Time budget model

The standoff mirrors competition constraints:

| Rule | Value |
|------|-------|
| Per-turn soft limit | 1.0 s |
| Episode overage bank | 60.0 s total |
| Season length | 720 turns |

Each turn that exceeds 1 s consumes from the 60 s overage bank. If cumulative overage exceeds 60 s, the agent raises `TimeoutError` and the match is recorded as `failed`. Timing metrics (mean ms, max ms, overage seconds) are saved per side for every match.

### Usage

```bash
python standoff/run_standoff.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--agent`, `-a` | `compound_expansion` | Agent template name or path to `main.py` |
| `--output`, `--mode` | `summary` | `summary` (human-readable) or `detailed` (full JSON to stdout) |
| `--results`, `-o` | `standoff/results.json` | Where to write the JSON report |
| `--no-swap` | off | Play each opponent once (selected agent always player 1) |

### Examples

```bash
# Quick summary against all opponents (default output)
python standoff/run_standoff.py --agent market_velocity

# Save results under a custom filename
python standoff/run_standoff.py -a shop_opportunist -o standoff/shop_opportunist_results.json

# Dump full per-match JSON to the terminal
python standoff/run_standoff.py -a melon_flood --output detailed

# Single-sided matches only (faster, less fair)
python standoff/run_standoff.py -a wheat_loop --no-swap
```

### Reading the output

**Summary mode** prints one line per opponent:

```
Standoff: market_velocity vs 20 agents (40 full 720-turn matches)
market_velocity vs shop_opportunist: 1W/1L/0T, cash 36665-82000, max 10.6ms, overage 0.000s
...
Results saved to standoff/results.json
```

Each line shows wins/losses/ties, mean final cash for both sides, peak turn latency, and max overage consumed.

**JSON results** (`standoff/results.json`) contain:

- `selected_agent` — the agent under test
- `configuration` — episode steps, timeout, overage bank, whether sides were swapped
- `summary` — per-opponent aggregates (wins, losses, mean cash, timing peaks)
- `matches` — full records for every individual game (cash, winner, status, per-side timing)

Past benchmark runs are archived in `standoff/*_results.json` for comparison across strategy iterations. See `strategy_learnings.md` for interpreted findings.

### When to use standoff vs `test_submission.py`

| Tool | Best for |
|------|----------|
| `test_submission.py` | Fast head-to-head checks, HTML replays, testing built tarballs |
| `standoff/run_standoff.py` | Ranking a candidate against the full field, latency profiling, regression tracking |

---

## Master Elo Tournament & Profiling Gate (`scripts/run_elo_tournament.py`)

Executes high-throughput, parallel round-robin tournaments across all submissions in `submissions/` using native Rust `kagg tournament`. Computes global Elo ratings, paired McNemar A/B test confidence intervals, and enforces a strict 5.0ms/turn latency profiling gate.

```bash
# Evaluate key submissions across 100 fixed seeds (with 8 parallel workers)
python scripts/run_elo_tournament.py --agents agent_final care_mill shop_opportunist melon_rusher --seeds 100 --workers 8

# Run across all submissions with custom 5ms profiling gate and updated Elo storage
python scripts/run_elo_tournament.py --all --seeds 50 --max-ms 5.0 --output-dir tournaments/full_league
```

Key features:
- **Round-Robin Fair Scheduling**: Plays both seat assignments ($0 \text{ vs } 1$ and $1 \text{ vs } 0$) for every submission pair across identical world seeds.
- **McNemar A/B Testing**: Exact two-sided binomial p-values and 95% cash differential confidence intervals against the top-ranked agent.
- **Elo Rating Updates**: Iterative logistic rating updates ($K=32$) persisted to `data/elo_leaderboard.json`.
- **Latency Profiling Gate**: Parses per-turn execution time; highlights any submission exceeding 5.0ms/turn in bold red as a Latency Violation to prevent consuming Kaggle's 60s overage bank.

---

## Retrograde DP & MILP Liquidation Solver (`scripts/solve_liquidation.py`)

Solves the multi-product terminal liquidation problem for Days 20 through 29 using Mixed-Integer Linear Programming (`scipy.optimize.milp`), continuous non-linear optimization (`scipy.optimize.minimize`), a terminal-first SciPy retrograde scheduler (`retrograde` backend), and discrete Bellman backward induction (`RetrogradeDPSolver`).

```bash
# Run benchmark demonstration comparing optimal solver schedule vs Day 29 dump:
python scripts/solve_liquidation.py --demo

# Evaluate on a historical replay file:
python scripts/solve_liquidation.py --replay replays/other_agents/rank1/112521191.json.gz
```

Key features:
- **Town Shop Drain Synchronization**: Aligns sales with the 6 daily consumption ticks (hours 0, 4, 8, 12, 16, 20), preventing quadratic (Wool) and linear (Milk) price collapses to the $1 floor.
- **Seed Maturation Enforcement**: Strictly prohibits planting Strawberry/Melon on Days 20+, Tomato on Days 22+, and all crops on Days 28+ whose maturation cycle extends past Day 30.
- **Shed Capacity Bound**: Guarantees total stored inventory remains $\le 100$ units across all 10 days.
- **Daily LiquidationIntent**: Overrides standard Beam Search on Days 20–29 with exact per-product quotas and terminal Day 29 asset flushing.

---

## Safe Fallback Controller & `@impenetrable_agent` (`src/kaggriculture/safety.py`)

Bulletproof wrapper that prevents competition forfeiture from unhandled runtime exceptions, tensor shape mismatches, FFI crashes, and Kaggle timeout overage bank exhaustion.

```python
from kaggriculture.safety import impenetrable_agent, SafeFallbackController

@impenetrable_agent
def agent(obs: dict, config: dict = None) -> dict:
    # Primary neural network / Beam Search policy
    ...
```

Key features:
- **Zero-Dependency `<1ms` SafeFallbackController**: Pure-Python implementation requiring zero external libraries (no numpy, scipy, torch). Executes in $\approx 0.28\text{ms}$ per turn.
- **Care Mill Strategy**: Feeds existing cows, administers daily care, harvests milk, collects fertilizer, and routes goods to the shed. Sells 100% of stored fertilizer if Day $< 8$; holds fertilizer thereafter, and flushes all residual inventory on Day 29.
- **Exception Shield**: Catches all `Exception` classes (including `KeyError`, `IndexError`, FFI failures) and seamlessly returns valid fallback moves.
- **Watchdog & Circuit Breaker**: Uses `time.perf_counter()` to monitor cumulative overage consumption ($>1.0\text{s}$ soft turn limit) and remaining bank. If remaining overage drops below $5.0\text{s}$, it permanently trips the circuit breaker, disabling heavy Beam Search/neural inference for the remainder of the episode.
- **Autopsy Logging**: Emits structured diagnostics and full stack traces to `sys.stderr` for rapid post-match debugging.

---

## Live Submission Telemetry & Fallback Monitor (`scripts/monitor_live_submissions.py`)

Automated live telemetry monitor that queries the Kaggle CLI to inspect live match execution logs for fallback triggers, RAM breaches, and timing overages.

```bash
# Monitor the latest 10 matches of our active submission:
python scripts/monitor_live_submissions.py --max-episodes 10

# Assert submission.tar.gz size strictly < 100 MiB before upload:
python scripts/monitor_live_submissions.py --check-size build/submission.tar.gz

# Inspect a specific submission:
python scripts/monitor_live_submissions.py --submission-id 56647370
```

Key features:
- **Pre-Flight Archive Validation**: Asserts `submission.tar.gz` is strictly $< 100\text{ MiB}$ (104,857,600 bytes) to prevent silent Kaggle Docker build aborts.
- **Automated Submission & Episode Resolution**: Queries Kaggle CLI to automatically discover the latest successful submission ID and its recent completed matches.
- **Seat Resolution**: Leverages Kaggle's 403 Forbidden permission model to deterministically identify whether our agent played as seat 0 or seat 1 without downloading multi-megabyte replay files.
- **Telemetry Regex Scanner**: Scans live `sys.stderr` logs for `[IMPENETRABLE_AGENT]`, `SafeFallbackController`, `MemoryError`, `numpy ArrayMemoryError` (6.5 GiB RAM threshold), and overage bank depletion.
- **Diagnostic Report**: Emits a formatted terminal summary (`Analyzed X recent episodes. Fallback triggered in Y episodes. Average overage consumed: Z seconds`) with detailed exception autopsies.

---

## Replay Safety Fallback Auditor (`scripts/audit_replay_fallbacks.py`)

Forensic auditor that inspects offline `.json` or `.json.gz` match replays to verify whether the agent executed its active policy or fell back into `SafeFallbackController` or an all-PASS stagnation loop.

```bash
python scripts/audit_replay_fallbacks.py --dir replays/my_agents/psro_leauge_pick
```

---

## Other tooling

**`tests/`** — unit tests for the library (`pytest`).

**[`docs/creating-agents.md`](docs/creating-agents.md)** — complete agent development guide (mechanics → submission).

**[`examples/`](examples/)** — runnable tutorial agents and local match runners.

**`kaggriculture.helpers.check_max_yield_met`** — utility to determine whether a crop tile has reached peak yield and is ready to harvest (see `src/kaggriculture/helpers/yield_check.py`).
