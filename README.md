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

---

## Scripts Guide

Two helper scripts live in `scripts/`. Use them in this order while developing an agent:

1. **`test_submission.py`** — run matches locally and compare agents
2. **`build_submission.py`** — package and validate a Kaggle-ready submission

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
├── main.py          # your agent
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

## Related tooling

**`standoff/run_standoff.py`** — round-robin benchmark runner. Plays one agent against every other template for a full 720-turn season (both starting positions unless `--no-swap`), tracks per-turn latency against a 1 s turn budget with a 60 s overage bank, and writes results to `standoff/results.json`.

```bash
python standoff/run_standoff.py --agent market_velocity
python standoff/run_standoff.py -a shop_opportunist --output detailed
```

**`tests/`** — unit tests for the library (`pytest`).

**`src/kaggriculture/examples/`** — example match runners for interactive development.
