# Agent Examples

Runnable tutorial agents that map to the skill levels in **[`docs/creating-agents.md`](../docs/creating-agents.md)** — the full guide covering game mechanics, observation schema, and how to build from basic to advanced agents.

## Quick start

```bash
uv sync && source .venv/bin/activate

# Smoke test the starter agent (1 game-day)
python scripts/test_submission.py -a1 examples/agents/01_basic_wheat_loop -a2 random -s 24

# Copy an example into submissions/ and iterate
cp -r examples/agents/01_basic_wheat_loop submissions/my_agent
```

## Example agents

| Directory | Level | What it demonstrates |
|-----------|-------|---------------------|
| [`agents/01_basic_wheat_loop/`](agents/01_basic_wheat_loop/main.py) | 1 | Minimal `ActionController` + `agent(obs)` |
| [`agents/02_melon_throughput/`](agents/02_melon_throughput/main.py) | 2 | Controller flags: hire, fertilize, sell margin |
| [`agents/03_adaptive_crop/`](agents/03_adaptive_crop/main.py) | 3 | Subclass `act()` — dynamic crop from market/shop data |
| [`agents/04_custom_market_planning/`](agents/04_custom_market_planning/main.py) | 4 | Override `plan_market_actions()` — phased buy/sell/hire |
| [`agents/05_manual_actions/`](agents/05_manual_actions/main.py) | 5 | Raw `Actions` builders, no controller |
| [`agents/06_helper_driven_harvest/`](agents/06_helper_driven_harvest/main.py) | 6 | `flatten_board_state` + `check_max_yield_met` |

## Utility scripts

| Script | Purpose |
|--------|---------|
| [`run_match.py`](run_match.py) | Short `Environment` demo (72 turns) |
| [`run_feed_squeeze.py`](run_feed_squeeze.py) | Run the adversarial `feed_squeeze` submission locally |

## Full documentation

Read **[`docs/creating-agents.md`](../docs/creating-agents.md)** for:

- How the game works (turns, board, crops, market, town)
- The agent contract (`obs` in → action dict out)
- Step-by-step progression from Level 1 to Level 7
- Observation and action reference tables
- Development workflow (test → standoff → build → submit)
- Common pitfalls
