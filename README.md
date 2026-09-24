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

## Scripts Guide & Research Infrastructure

The `scripts/` directory provides an end-to-end quantitative research, replay parsing, optimization, and evaluation infrastructure. See **[`research/21_Scripts_Reference_and_User_Guide.md`](research/21_Scripts_Reference_and_User_Guide.md)** for the complete 27+ tool reference and **[`research/20_Replay_Forensics_and_Gauntlet_Findings.md`](research/20_Replay_Forensics_and_Gauntlet_Findings.md)** for forensic findings from elite ladder matches.

Key tools in workflow order:

1. **`scripts/eval_against_replays.py`** — Gauntlet benchmark: test agents against 20 top-tier real-world ladder replays (Boey, Clement Ling, DECEM, Kaggledew Valley, BenPalmer59, etc.)
2. **`scripts/trace_replay_match.py`** — Forensic simulation tracer comparing live actions against historical opponent steps turn-by-turn
3. **`scripts/eval_cash.py`** — Evaluates terminal cash across 14 benchmark seeds (1, 3, 5, 7, 10, 15, 20)
4. **`scripts/quant_full_product.py`** — SciPy-based multi-product pricing and liquidation optimizer
5. **`scripts/test_submission.py`** — Run matches locally, measure latency (<2ms), and profile agent performance
6. **`scripts/build_submission.py`** — Package and validate a Kaggle-ready submission (single-file or tar.gz)

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
| `--message`, `-m` | auto-generated | Submission message for the Kaggle CLI |
| `--submit`, `-s` | off | Submit directly via `kaggle competitions submit` |

#### Submission formats

**`tar` (default, recommended)**

Creates `build/submission.tar.gz` containing:

```
submission.tar.gz
├── main.py          # your agent entrypoint
├── *.py             # sibling modules from the same submission folder (if any)
└── kaggriculture/   # full library copy
```

Kaggle extracts the archive with `main.py` at the root. This is the preferred format because it keeps agent code separate from the library and mirrors the multi-file layout used during development.

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

## Other tooling

**`tests/`** — unit tests for the library (`pytest`).

**[`docs/creating-agents.md`](docs/creating-agents.md)** — complete agent development guide (mechanics → submission).

**[`examples/`](examples/)** — runnable tutorial agents and local match runners.

**`kaggriculture.helpers.check_max_yield_met`** — utility to determine whether a crop tile has reached peak yield and is ready to harvest (see `src/kaggriculture/helpers/yield_check.py`).
