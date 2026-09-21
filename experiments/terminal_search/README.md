# Terminal search (research prototype)

Seven-turn **deposit search** skeleton aligned with Architecture Beta’s
`terminal_planner` (research/03) and innovation **B1** (research/05): sparse
search during days **27–29**, anchored at step **712** for a **7-turn** horizon
(712–718).

## Scope

- **Not** wired into any submission `main.py`.
- Uses `yield_check` for ripe-plant detection and a **minimal** unit-step model
  (`unit_step.py`) for walk / HARVEST / DROP.
- `plan_terminal()` enforces **`MAX_SEARCH_MS = 200`** per planning call and
  accepts schedules only when they **dominate** the baseline on shed prefixes,
  sold quantities, and per-actor deposits with **zero overflow**.

## Modules

| File | Role |
|------|------|
| `terminal_search.py` | `dominates`, `plan_terminal`, route helpers |
| `proposals.py` | Harvest → shed DROP route candidates |
| `simulate.py` | Replay a 7-action schedule into a dominance-compatible trace |
| `unit_step.py` | One-turn farmer/hand command application |
| `state.py` | Mutable farm/private snapshot |

## Usage (local)

```python
from experiments.terminal_search import plan_terminal, terminal_search_window
from experiments.terminal_search.simulate import simulate_terminal_schedule

if terminal_search_window(obs):
    result = plan_terminal(obs, config, baseline_7_turns, simulate_terminal_schedule)
```

Future work: PUCT / macro actions (`WATER_ROUTE`, `HARVEST_CLUSTER`), full env
parity, and earlier activation inside day 27 before step 712.
