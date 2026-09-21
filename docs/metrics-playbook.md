# Episode metrics playbook

Telemetry lives in `kaggriculture.helpers.episode_metrics` and is recorded when
you wrap an `ActionController` with `MetricsWrapper` or set
`controller.metrics_recorder`. Plain callables without `act()` still run, but
overflow / slot / impact columns stay empty.

## Single episode (JSON)

```bash
.venv/bin/python scripts/analyze_episode.py --agent market_velocity --steps 720
.venv/bin/python scripts/analyze_episode.py --agent alpha_velocity_p0 --opponent random
```

Key fields:

| Field | Meaning |
|-------|---------|
| `final_cash` | Player 0 money at episode end |
| `shed_overflow_units_lost` | Estimated shed units lost to capacity pressure |
| `unfillable_sell_slots_burned` | SELL slots dropped by clamp / room guard |
| `mean_impact_score_of_sells` | Average sell `impact_score` for the turn’s final market |
| `mean_revenue_per_sold_unit` | Slippage-model revenue per unit across final SELL orders |
| `decide_ms_p50` / `decide_ms_p95` | `ActionController.act()` latency per step |
| `collision_holds` / `collision_releases` | Premium sell overlay diffs (microstructure on) |
| `hires_by_day` | HIRE orders per day |

## Compare two agents (CSV)

```bash
.venv/bin/python scripts/compare_agents.py market_velocity alpha_velocity_p0 \
  --seeds 20 --output standoff/p0_metrics_compare.csv
```

Each seed runs **two solo** episodes (same seed, vs `random` by default). CSV
columns: `seed`, `agent`, `terminal_cash`, `overflow`, `slots_burned`,
`mean_impact`, `decide_ms_p50`, `decide_ms_p95`.

A short summary (mean/median cash, win count on cash) prints to stderr.

## How to interpret diffs

1. **Cash** — Primary outcome. Use at least 10–20 seeds; single-seed solo
   numbers are noisy (weed spawns, shop unlock order).
2. **Overflow & slots_burned** — Safety metrics from research/04 §5. A
   candidate that wins cash but raises overflow or burned slots is suspect until
   you explain the tradeoff (e.g. deliberate endgame dump).
3. **mean_impact** — Higher is not automatically better; it reflects sell
   sizing and item mix. Use it to see whether a policy is dumping low-impact
   staples vs batched premiums.
4. **Latency** — Keep `decide_ms_p95` well under 30 ms mean target; investigate
   if p95 approaches 200 ms (terminal search / heavy routing).

## When to accept a candidate

Accept a library or submission change when **all** of the following hold on your
benchmark panel:

- Mean (or median) terminal cash **does not regress** vs the baseline you are
  replacing, on ≥10 seeds with the same opponent.
- `overflow` and `slots_burned` are **not worse** than baseline on average
  (or you document a known one-step regression with a follow-up fix).
- Latency p95 stays within budget for 720-step runs.
- Head-to-head standoff vs the relevant rival (e.g. `shop_opportunist`) is run
  separately; metrics CSV alone is solo-only.

Reject or rework if cash gains come only from 1–2 lucky seeds, or if safety
metrics worsen without a matching cash win.

## Related commands

- Full season bench: `scripts/test_submission.py`
- Multi-agent panel: `standoff/run_standoff.py`
- Tests: `pytest tests/test_episode_metrics.py`
