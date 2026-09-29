# Master Integration Gauntlet — Test Infrastructure Specification (`TEST_INFRA.md`)

## 1. Test Philosophy

The Master Integration Gauntlet evaluates, packages, and benchmarks the autonomous quantitative research and policy optimization agent under live Kaggle competition conditions. The end-to-end (E2E) test harness is architected under strict **opaque-box, requirement-driven, and mathematically rigorous** principles:

1. **Opaque-Box Contract Verification**: Tests validate observable behaviors, output action dictionaries, system logs, execution latencies, and quantitative acceptance criteria against authoritative requirements (`ORIGINAL_REQUEST.md`, `PROJECT.md`), rather than coupling to internal implementation nuances.
2. **Progressive Testability & Milestone Decoupling**: Features are implemented across sequential milestones (M1: Candidate Packaging & Watchdog, M2: Replay Scraping & Ghost Fleet, M3: Master Tournament, M4: Diagnostic Autopsy & Optuna HPO). Tests are structured to execute cleanly from day one:
   - Features already implemented or verified via reference submissions execute and pass immediately.
   - Downstream milestone artifacts not yet written to disk are detected and reported via structured `pytest.skip` notifications rather than fatal crashes, ensuring continuous CI readiness.
3. **Authoritative Oracles & References**: Expected behaviors and values are derived directly from authoritative specifications:
   - *Kaggle Competition Rules*: Max 1.0 second per single turn, 60.0 seconds cumulative overage bank per 720-step episode, action dictionary format (`farmer`, `hands`, `market`), submission file size (<100 MiB), and memory limit (<6.5 GiB).
   - *Native Rust Simulation Engine (`kagg tournament` / `kagg-sim`)*: Deterministic ground truth for step progression (steps 0..719), dual time limit enforcement (`"time_limits": {"act_s": 1.0, "overage_s": 60.0}`), forfeiture mechanics, and structured summary/result outputs (`results.jsonl`, `summary.json`).
   - *Quantitative Benchmark Targets*: Aggregate win rate $\ge 60.0\%$, average terminal cash balance $\ge \$130,000$, and candidate mean turn latency $< 5.0\text{ ms}$.
4. **Self-Contained Isolation**: Every test case sets up its own state, utilizes isolated temporary directories (`tmp_path`), and produces zero side effects or disk pollution.
5. **Adversarial & Boundary Rigor**: Tests explicitly probe extreme timing conditions (turn duration over 1.0s, overage bank depletion, safety threshold breaches), episode boundary steps (Step 0 cold start, Step 71 pre-unlock, Step 718 penultimate, Step 719 terminal liquidation), memory leaks over 100 turns, and corrupted tournament output rows.

---

## 2. Feature Inventory Coverage Matrix

The table below maps all 11 architectural features defined in `PROJECT.md § Feature Inventory` across their requirement IDs (R1–R5), milestones, E2E test tiers, verification methods, and authoritative test implementations.

| # | Feature | Req ID | Milestone | Tier | Verification Method | Primary Test File / Function |
|---|---------|--------|-----------|------|---------------------|------------------------------|
| 1 | Candidate Packaging | R1 | M1 | Tier 1 | AST parsing, file existence, size check (<100 MiB), entrypoint `agent(obs, config=None)` at bottom | `tests/test_gauntlet_e2e.py::test_candidate_packaging_compliance` |
| 2 | Packaging Bug Fixes | R1 | M1 | Tier 1 | AST callable ordering verification (agent is last top-level function) | `tests/test_gauntlet_e2e.py::test_candidate_packaging_compliance` |
| 3 | Execution Watchdog | R3 | M1 | Tier 1 & 2 | `@impenetrable_agent` wrapping, default parameters (1.0s soft limit, 60.0s overage, 5.0s circuit breaker), drawdown logic | `tests/test_gauntlet_e2e.py::test_watchdog_compliance_attributes`<br>`tests/test_gauntlet_e2e.py::test_watchdog_overage_drawdown` |
| 4 | Top 10 Replay Scraping | R2 | M2 | Tier 1 | Check downloaded replays in `replays/live_top10/` (20 JSON files) | `tests/test_gauntlet_e2e.py::test_ghost_fleet_compliance` |
| 5 | Ghost Opponent Generation | R2 | M2 | Tier 1 | Validate `submissions/ladder_ghost_*` directories and verify `main.py` entrypoint | `tests/test_gauntlet_e2e.py::test_ghost_fleet_compliance` |
| 6 | Ghost Zero-Crash Execution | R2 | M2 | Tier 1 & 3 | Verify ghost callable returns valid action dictionary with zero exceptions over test steps | `tests/test_gauntlet_e2e.py::test_ghost_fleet_compliance`<br>`tests/test_gauntlet_e2e.py::test_candidate_vs_ghost_match_execution` |
| 7 | Master Tournament Orchestration | R4 | M3 | Tier 1 & 3 | Validate `tournaments/gauntlet_1000/config.json` schema, time limits, Python executable, and runner invocation | `tests/test_gauntlet_e2e.py::test_tournament_config_compliance`<br>`tests/test_gauntlet_e2e.py::test_candidate_vs_ghost_match_execution` |
| 8 | Tournament Performance Gate | R5 | M3 | Tier 4 | Evaluate `results.jsonl` and `summary.json`: Win Rate $\ge 60.0\%$, Terminal Cash $\ge \$130,000$, Mean Latency $< 5.0\text{ ms}$ | `tests/test_gauntlet_e2e.py::test_tournament_results_parser_and_assertions_passing`<br>`tests/test_gauntlet_e2e.py::test_tournament_results_parser_and_assertions_failing`<br>`tests/test_gauntlet_e2e.py::test_live_tournament_acceptance_gate` |
| 9 | Diagnostic Autopsy Pipeline | R5 | M4 | Tier 4 | Verify `scripts/autopsy_ladder_losses.py:analyze_loss_match` computes crossover turn, net worth deficit, and catalysts | `tests/test_gauntlet_e2e.py::test_autopsy_pipeline_loss_validation` |
| 10 | Automated Optuna HPO Loop | R5 | M4 | Tier 4 | Verify `scripts/ladder_ghost.py:sample_beam_weights` (7 weights) and `run_hpo` executes trials and saves `counter_best` | `tests/test_gauntlet_e2e.py::test_optuna_hpo_loop_validation` |
| 11 | E2E Test Suite | Dual Track | E2E | Tiers 1–4 | Full automated pytest execution covering all 4 tiers with zero unhandled regressions | `tests/test_gauntlet_e2e.py` |

---

## 3. 4-Tier Test Architecture & Methodology

The acceptance suite is organized into four complementary tiers, progressing from structural feature validation to system-level tournament gate assertions:

```
tests/test_gauntlet_e2e.py
├── Tier 1: Feature Coverage
│   ├── test_candidate_packaging_compliance
│   ├── test_watchdog_compliance_attributes
│   ├── test_ghost_fleet_compliance
│   └── test_tournament_config_compliance
├── Tier 2: Boundary & Corner Cases
│   ├── test_watchdog_overage_drawdown
│   ├── test_watchdog_circuit_breaker_and_timeout_forfeiture
│   ├── test_boundary_step_verification
│   └── test_memory_and_latency_assertion
├── Tier 3: Cross-Feature Interactions
│   └── test_candidate_vs_ghost_match_execution
└── Tier 4: Master Tournament Acceptance Gate & Fallback Loop
    ├── test_tournament_results_parser_and_assertions_passing
    ├── test_tournament_results_parser_and_assertions_failing
    ├── test_live_tournament_acceptance_gate
    ├── test_autopsy_pipeline_loss_validation
    └── test_optuna_hpo_loop_validation
```

### Tier 1: Feature Coverage (Packaging, Watchdog, Fleet, Config)
- **Candidate Packaging**: Asserts that candidate file `submissions/hybrid_grandmaster_v2/main.py` exists, is self-contained (<100 MiB, valid Python AST, no missing module dependencies), and defines `agent(obs, config=None)` as the final top-level callable in the file.
- **Watchdog Compliance**: Asserts `@impenetrable_agent` enforces `DEFAULT_SOFT_LIMIT_S = 1.0`, `DEFAULT_OVERAGE_BANK_S = 60.0`, and `DEFAULT_SAFETY_THRESHOLD_S = 5.0`, wrapping functions with full diagnostic statistics (`.stats()`).
- **Ghost Fleet Compliance**: Validates each ghost opponent in `submissions/ladder_ghost_*` exposes a valid `agent(obs)` entrypoint returning the standard action schema (`farmer`, `hands`, `market`). Asserts that all 20 ghost submissions exist upon M2 completion.
- **Tournament Configuration**: Validates `tournaments/gauntlet_1000/config.json` against the native Rust schema (`"time_limits": {"act_s": 1.0, "overage_s": 60.0}`, `"schedule": "gauntlet"`, `"on_error": "forfeit"`, `"python": {"exe": ...}`).

### Tier 2: Boundary & Corner Cases (Timing, Limits, Steps, Memory)
- **Overage Drawdown Mechanics**: Verifies that any turn exceeding 1.0s deducts the excess duration from the 60.0s overage bank, while turns $\le 1.0\text{s}$ leave the overage bank untouched.
- **Circuit Breaker & Timeout Forfeiture**: Verifies that when remaining overage drops below 5.0s, the circuit breaker permanently trips, routing all subsequent turns through `SafeFallbackController` (<1ms execution) to prevent match timeout forfeiture. Verifies that simulator timeout rules trigger `forfeit: true`.
- **Boundary Turn Execution**: Evaluates agent behavior on critical edge steps: Step 0 (cold start initialization), Step 71 (pre-unlock day boundary), Step 718 (penultimate turn), and Step 719 (terminal season liquidation). Ensures zero division errors, proper inventory dumping, and valid action formats.
- **Memory Stability & Latency Profile**: Profiles agent execution across 100 consecutive turns. Asserts mean turn latency $< 5.0\text{ ms}$, max turn latency $< 50.0\text{ ms}$, and verifies flat memory allocation via `tracemalloc` (zero memory leaks).

### Tier 3: Cross-Feature Interactions (Candidate vs Ghost Match)
- **Head-to-Head Simulation**: Pits candidate agent against representative ghost opponent (`submissions/ladder_ghost_112542379`) across a complete 720-step season using `FastSimulation` / native Rust engine.
- **Contract Verification**: Asserts simulation runs to completion (`steps == 719`, `done == True`), zero unhandled exceptions occur, `forfeit` is `False`, and all generated action dictionaries conform strictly to the game engine schema.

### Tier 4: Master Tournament Acceptance Gate & Diagnostic Fallback
- **Acceptance Gate Parser**: Implements authoritative evaluator `evaluate_tournament_acceptance(summary_json, results_jsonl)`:
  - **Win Rate Assertion**: Candidate win rate $\ge 60.0\%$.
  - **Terminal Cash Assertion**: Average candidate terminal cash $\ge \$130,000$.
  - **Latency Profile Assertion**: Candidate mean turn latency $< 5.0\text{ ms}$, max turn latency $< 1000.0\text{ ms}$, and zero simulation errors.
- **Adversarial / Failure Testing**: Verifies that `evaluate_tournament_acceptance` raises explicit, informative `AssertionError` exceptions when any threshold is breached (low win rate, low cash, latency violation, error/forfeiture).
- **Live Tournament Gate**: When `tournaments/gauntlet_1000/summary.json` and `results.jsonl` are present on disk, executes the full acceptance gate against live tournament data.
- **Loss Autopsy Pipeline**: Verifies `scripts/autopsy_ladder_losses.py` extracts loss match metadata, detects permanent crossover turns, and identifies opponent macro catalysts.
- **Optuna HPO Tuning Loop**: Verifies `scripts/ladder_ghost.py:sample_beam_weights` spans all 7 Beam Search hyperparameters and `run_hpo` executes multi-trial optimization to generate tuned counter-strategies.

---

## 4. Pytest Runner Invocation & Verification Commands

All tests are executable via the project virtual environment `.venv/bin/pytest`:

```bash
# Execute full Master Integration Gauntlet E2E Acceptance Suite
.venv/bin/pytest tests/test_gauntlet_e2e.py -v

# Execute specific tiers
.venv/bin/pytest tests/test_gauntlet_e2e.py -k "tier1" -v
.venv/bin/pytest tests/test_gauntlet_e2e.py -k "tier2" -v
.venv/bin/pytest tests/test_gauntlet_e2e.py -k "tier3" -v
.venv/bin/pytest tests/test_gauntlet_e2e.py -k "tier4" -v

# Run with duration profiling to verify latency
.venv/bin/pytest tests/test_gauntlet_e2e.py --durations=10 -v
```

---

## 5. Coverage & Reliability Thresholds

| Dimension | Threshold Specification | Enforcement Mechanism |
|-----------|-------------------------|-----------------------|
| Candidate Packaging | File size $< 100\text{ MiB}$, self-contained AST, entrypoint `agent` last | `validate_candidate_packaging()` in Tier 1 |
| Watchdog Timing Limits | Soft limit $1.0\text{ s}$, overage bank $60.0\text{ s}$, safety trip $5.0\text{ s}$ | `ImpenetrableAgentWrapper` assertions in Tier 1 & 2 |
| Ghost Fleet Size | Exactly 20 Top 10 ghost opponent directories upon M2 completion | `submissions/ladder_ghost_*` inspection in Tier 1 |
| Tournament Configuration | Schedule `"gauntlet"`, dual time limits, valid Python executable | `validate_gauntlet_tournament_config()` in Tier 1 |
| Turn Latency | Candidate mean latency $< 5.0\text{ ms}$, max turn $< 50.0\text{ ms}$ (isolated) / $< 1000.0\text{ ms}$ (in-game) | Latency assertion in Tier 2 & Tier 4 |
| Memory Stability | Memory allocation delta $< 500\text{ KB}$ across 100 turns | `tracemalloc` assertion in Tier 2 |
| Tournament Win Rate Gate | Candidate overall win rate $\ge 60.0\%$ | `evaluate_tournament_acceptance()` in Tier 4 |
| Terminal Cash Gate | Candidate average cash $\ge \$130,000$ across all tournament matches | `evaluate_tournament_acceptance()` in Tier 4 |
| Autopsy & HPO Fallback | Loss autopsy extracts crossover turn; Optuna tunes 7 beam weights | Tier 4 verification fixtures |
| Flakiness Tolerance | Zero flake tolerance (100% deterministic test execution) | Fixed seeds (`seed=42`) across test cases |
