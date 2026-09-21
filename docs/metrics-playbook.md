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

## Panel analysis

Run many submissions in one pass with stderr progress, a ranked leaderboard on
stdout, and artifacts under `--output-dir`:

```bash
# Quick 5-seed smoke (all submissions)
.venv/bin/python scripts/run_agent_analysis.py \
  --agents all --seeds 5 --steps 720 --output-dir standoff/smoke

# Candidate stack vs shop baseline (solo + head-to-head)
.venv/bin/python scripts/run_agent_analysis.py \
  --agents market_velocity,alpha_velocity_p0,alpha_velocity_p1,alpha_shop_hybrid,demand_mpc,regime_counter,shop_opportunist \
  --seeds 30 \
  --baseline shop_opportunist \
  --head-to-head shop_opportunist \
  --output-dir standoff/panel_main
```

**Artifacts:** `panel_summary.json`, `panel_leaderboard.csv`, and
`<agent>_seeds.csv` per agent.

**Leaderboard table (stdout):** sorted by `mean_cash`. With `--baseline`, the
baseline row is marked `*` and `win_vs_baseline` is the fraction of seeds where
that agent beat the baseline on solo cash (paired by seed).

**Head-to-head table:** when `--head-to-head` is set, each candidate (except the
named opponent) plays two full 720-turn matches vs that opponent (seat 0 and seat
1). Use this for shared-market behavior; solo random panels can rank higher than
head-to-head vs `shop_opportunist` because shop timing exploits interaction.

**How to read:**

1. Trust **mean/median cash** on ≥20 seeds before promoting a candidate.
2. Compare **overflow_mean** and **slots_mean** to the baseline row — regressions
   need an explanation.
3. If solo wins but **H2H wins** stay at 0, prioritize opponent-aware sells/plant
   mix (regime counter, duel MPC) rather than more solo tuning.

## Related commands

- Pairwise CSV: `scripts/compare_agents.py`
- Full season bench: `scripts/test_submission.py`
- Multi-agent standoff (all pairs): `standoff/run_standoff.py`
- Tests: `pytest tests/test_episode_metrics.py tests/test_run_agent_analysis.py`
