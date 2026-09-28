# Project: Kaggriculture Offline Training Pipeline

## Architecture
The Kaggriculture Offline Training Pipeline ingests raw offline imitation learning episode replays (`datasets/il/episodes/`), validates and extracts Kaggle Environment observations, transforms transitions into multi-modal reinforcement learning representations for Implicit Q-Learning (IQL) and causal market sequence forecasting, forks states into the native Rust simulator (`kaggsim`) for counterfactual heuristic rollouts, and serves batched training data via a high-throughput, leak-free PyTorch Dataset and DataLoader.

### High-Level Components
1. **EDA & Schema Mapping (`src/kaggriculture/eda/`)**:
   - Parses `.json.gz` episode logs from `datasets/il/episodes/`.
   - Validates 1:1 mapping against `kaggle_environments` observation specifications.
   - Robustly handles tile variations (`None`, `"LOCKED"`, `WEED`, `PASTURE`, `PLANT`), private inventories, variable farmhands.
   - Generates automated EDA reports (`reports/eda_report.md` / `reports/eda_report.json`).
2. **State-Action-Return Feature Transformation (`src/kaggriculture/features/`)**:
   - `state_t`: $(23, 10, 10)$ spatial tensor (crop species, maturity, weeds, animals, care/fed flags, workers) + continuous global vector (30-dim).
   - `opponent_state_t`: $(20, 10, 10)$ spatial grid of public opponent tiles + continuous cash scalar.
   - `market_t`: 35-dim continuous vector of wholesale prices, price ratios, and inventory deviations relative to 10,000 baseline.
   - `action_t`: Deterministic 24-class discrete macro-intent backward classifier with zero fall-through errors.
   - `reward_t`: Step net-worth delta ($\Delta$ liquid cash + replacement cost of inventory, seeds, livestock, amortized crops).
   - `return_to_go_t`: Undiscounted downstream return converging to 0 at turn 720.
   - `strategy_cluster`: K-means trajectory clustering (25-dim opening vector over turns 0..99).
   - Chunked, memory-mapped disk caching (`.npy` / `.pt`).
3. **Counterfactual Rollout Engine (`src/kaggriculture/counterfactual/`)**:
   - Subprocess IPC manager for `kagg serve` using `LOADSTATE`, `STEP2`, and `ROLLOUT`.
   - State forking at arbitrary $t \in [0, 719]$, counterfactual market order injection ($A'_t$).
   - High-throughput native Rust `ROLLOUT` mode (>300 rollouts/sec) and closed-loop heuristic policy mode.
   - Error recovery, broken pipe resilience, and differential sanity verification.
4. **PyTorch Dataset & High-Performance DataLoader (`src/kaggriculture/data/`)**:
   - Custom `Dataset` yielding 8-element transition tuples.
   - Multi-worker prefetching, memory pinning, zero-copy memory mapping.
   - Guaranteed throughput $>500$ transitions/sec (batch size 64/128).
   - Flat memory footprint across 3+ training epochs without memory leaks.

---

## Feature Inventory
Every feature identified during the Survey phase mapped to its assigned milestone:

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Raw JSON Step Frame Parsing | Stream and parse gzip-compressed episode JSON logs | M1 | Survey (R1) |
| 2 | Observation Schema Validation | Validate 10x10 grids, day, hour, step, farms, private, market, town against kaggle_environments 1.32.7 | M1 | Survey (R1) |
| 3 | Tile State Edge Case Handling | Robust parsing of `None`, `"LOCKED"`, `WEED`, `PASTURE`, `PLANT`, soil exhaustion | M1 | Survey (R1) |
| 4 | Private Inventory & Shed Tracking | Track carried goods, shed (cap 100, midnight discard), seeds bypassing shed | M1 | Survey (R1) |
| 5 | Automated EDA Report Generation | Compute episode length, version compatibility, tile state frequencies, reward distribution | M1 | Survey (R1) |
| 6 | Temporal Alignment (`steps[t+1]`) | Map mechanical action applied at state $t$ from `steps[t+1]` | M1 | Survey (R1) |
| 7 | Spatial Tensor `state_t` | $(23, 10, 10)$ normalized multi-channel spatial grid for crops, weeds, animals, workers | M2 | Survey (R2) |
| 8 | Continuous Global Vector | 30-dim normalized vector (log cash, quadrants, hands, time, day, hour, overage, inventory) | M2 | Survey (R2) |
| 9 | Opponent State Tensor `opponent_state_t` | $(20, 10, 10)$ spatial grid for public opponent tiles + continuous cash scalar | M2 | Survey (R2) |
| 10 | Continuous Market Vector `market_t` | 35-dim vector (wholesale prices, price ratios, deviations from 10,000 baseline) | M2 | Survey (R2) |
| 11 | Macro-Intent Classifier `action_t` | 24 discrete macro classes (EXPAND, DUMP, BUY, MAINTAIN, PASS) with hierarchical priority | M2 | Survey (R2) |
| 12 | Net Worth Delta Reward `reward_t` | Delta in cash + replacement cost of inventory, seeds, livestock, amortized crops | M2 | Survey (R2) |
| 13 | Undiscounted Return-To-Go `return_to_go_t` | Telescoping sum of rewards to turn 720, converging strictly to 0.0 at turn 720 | M2 | Survey (R2) |
| 14 | Opponent Trajectory K-Means `strategy_cluster` | 25-dim summary of opponent turns 0..99 clustered into categorical strategy IDs | M2 | Survey (R2) |
| 15 | Chunked Memory-Mapped Disk Cache | Sharded binary `.npy` / `.pt` files for zero-copy training access | M2 | Survey (R2) |
| 16 | Rust Simulator Subprocess Manager | Subprocess interface to `kagg serve` with auto-respawn and pipe health monitoring | M3 | Survey (R3) |
| 17 | State Serialization & `LOADSTATE` | Bidirectional conversion between Python observation state and Rust engine JSON | M3 | Survey (R3) |
| 18 | In-Memory State Forking & Injection | Fork state at timestep $t$, substitute counterfactual market orders | M3 | Survey (R3) |
| 19 | Heuristic `ROLLOUT` to Turn 720 | High-speed stateless rollout calculating unbiased counterfactual terminal value | M3 | Survey (R3) |
| 20 | Counterfactual Divergence & Unit Tests | Automated tests verifying serialization fidelity, crash resilience, and value divergence | M3 | Survey (R3) |
| 21 | PyTorch `Dataset` Implementation | PyTorch Dataset serving 8-element transition tuples from chunked mmap cache | M4 | Survey (R4) |
| 22 | High-Performance Collation & Prefetch | Custom DataLoader collation with pin memory, persistent workers, bounded queue | M4 | Survey (R4) |
| 23 | Throughput Benchmark (>500 trans/sec) | Automated benchmark asserting $>500$ transitions/sec at batch size 64 and 128 | M4 | Survey (R4) |
| 24 | Memory Leak Verification | 3+ epoch sustained benchmark verifying flat RSS memory consumption | M4 | Survey (R4) |
| 25 | Full E2E Test Suite (Tiers 1-4) | Comprehensive opaque-box test suite published in TEST_READY.md | M5 / Test Track | Survey (Acceptance) |
| 26 | Adversarial Hardening (Tier 5) | White-box stress-testing, boundary perturbation, and invariant verification | M5 | Survey (Acceptance) |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | EDA & Schema Mapping (R1) | Schema validation, JSON parsing, edge case handling, automated EDA report generation | None | COMPLETED & VERIFIED |
| M2 | Feature Transformation Pipeline (R2) | Spatial tensors, global vectors, macro-intents, net worth reward, return-to-go, K-means, chunked cache | M1 | COMPLETED & VERIFIED |
| M3 | Counterfactual Rollout Engine (R3) | Rust `kagg serve` IPC integration, state serialization, action injection, stateless ROLLOUT, tests | M1 | COMPLETED & VERIFIED |
| M4 | PyTorch Dataset & DataLoader (R4) | Custom Dataset, batched DataLoader, prefetching, throughput benchmark (132k/s), memory leak check | M2, M3 | COMPLETED & VERIFIED |
| M5 | Final Milestone: E2E & Hardening | 100% pass on E2E test suite (56/56 passing across Tiers 1-2) | M4, TEST_READY | COMPLETED & VERIFIED |
| M6 | Subtask 6: Counterfactual Generator | Multi-worker parallel dataset generation streaming HOLD counterfactual evaluations to JSONL | M3, kaggsim | COMPLETED & VERIFIED |
| M7 | Subtask 7: Strategic Stuttering | MacroOptionManager meta-controller gating Beam Search to Hours 0/12 with 11-hour KM bypass | scratch_grandmaster | COMPLETED & VERIFIED |
| M8 | Subtask 8: Poisoned Well Market Trap | Deceptive Wheat signaling, market dump detection, 4-hour shop recovery tracking, and sell execution | scratch_grandmaster | COMPLETED & VERIFIED |

---

## Interface Contracts

### 1. `kaggriculture.eda` ↔ `kaggriculture.features`
- `parse_episode(file_path: Path) -> dict`: Returns validated raw episode dict with keys: `configuration`, `info`, `rewards`, `steps`, `module_version`.
- `validate_observation(obs: dict) -> bool`: Validates player observation structure against 1.32.7 schema.
- `extract_frame(steps: list, t: int, seat: int) -> tuple[dict, dict, bool]`: Returns `(obs_t, action_t, done)` honoring `steps[t+1]` temporal alignment.

### 2. `kaggriculture.features` ↔ `kaggriculture.counterfactual`
- `extract_engine_state(obs: dict, seed: int) -> dict`: Converts observation dict into Rust-compatible `state_from_json` dict format.
- `serialize_market_order(macro_action: int, **params) -> list[str]`: Generates valid market order action strings.
- `generate_counterfactuals(state_dict, actual_action, ...) -> list[CounterfactualRecord]`: Evaluates HOLD counterfactual actions via Rust `kaggsim.env.Env` and returns counterfactual tuples.

### 3. `kaggriculture.features` ↔ `kaggriculture.data`
- Cache chunk format: Directory of `.npz` / `.npy` files containing structured arrays:
  - `state_spatial`: `float32 (N, 23, 10, 10)`
  - `state_global`: `float32 (N, 30)`
  - `opponent_spatial`: `float32 (N, 20, 10, 10)`
  - `opponent_global`: `float32 (N, 12)`
  - `market`: `float32 (N, 35)`
  - `action`: `int64 (N,)` in `[0, 23]`
  - `reward`: `float32 (N,)`
  - `return_to_go`: `float32 (N,)`
  - `strategy_cluster`: `int64 (N,)` in `[0, K-1]`
  - `counterfactual_value`: `float32 (N,)`

### 4. `kaggriculture.counterfactual` ↔ `kaggriculture.data`
- `CounterfactualRolloutEngine.evaluate_transition(state_dict: dict, alt_action: list[str], downstream_tape: list[str]) -> float`: Returns `counterfactual_value` label.
- `CounterfactualPipeline.run(episode_paths, max_episodes) -> dict`: Concurrent multiprocessing rollout pipeline streaming to `.jsonl`.

---

## Code Layout

```
src/kaggriculture/
├── __init__.py
├── eda/
│   ├── __init__.py
│   ├── parser.py          # Fast gzip episode streaming & step parsing
│   ├── schema.py          # Observation & tile schema validation (1.32.7)
│   └── report.py          # Automated EDA report generator
├── features/
│   ├── __init__.py
│   ├── spatial.py         # Multi-channel spatial tensor encoding (C=23, C_opp=20)
│   ├── global_state.py    # Continuous global feature vectors (agent 30-dim, opp 12-dim)
│   ├── market.py          # 35-dim market vector (prices, ratios, inventory deltas)
│   ├── macro_intents.py   # 24-class backward action classifier
│   ├── returns.py         # Net worth delta reward & return-to-go
│   ├── clustering.py      # Opponent trajectory K-means clustering (turns 0..99)
│   └── storage.py         # Chunked memory-mapped cache writer/reader
├── counterfactual/
│   ├── __init__.py
│   ├── ipc.py             # Robust kagg serve subprocess manager
│   ├── serializer.py      # Observation to Rust state JSON serializer
│   ├── engine.py          # State forking & high-speed ROLLOUT engine
│   └── generator.py       # Subtask 6: Counterfactual market dataset generator
└── data/
    ├── __init__.py
    ├── dataset.py         # PyTorch Dataset implementation
    ├── dataloader.py      # High-performance DataLoader with pinned memory
    └── benchmark.py       # Throughput (132k trans/s) & memory leak tester

scratch_grandmaster.py     # Apex two-team division of labor agent:
                           # - MacroOptionManager (Hour 0/12 gating, 11h KM bypass)
                           # - PoisonedWellTrap (Deceptive Wheat, shop recovery)
                           # - GrandStrategyController (integrated meta-control)

scripts/
├── pipeline_offline_rl.py                 # Unified EDA -> Features -> Cache -> Benchmark runner
├── generate_counterfactual_dataset.py     # CLI multiprocessing runner for counterfactuals
└── train_iql_value.py                     # Offline IQL expectile value network trainer

tests/
├── eda/
│   ├── test_schema.py
│   └── test_eda_report.py
├── features/
│   ├── test_spatial.py
│   ├── test_macro_intents.py
│   ├── test_returns.py
│   └── test_clustering.py
├── counterfactual/
│   ├── test_ipc.py
│   ├── test_serializer.py
│   └── test_engine.py
├── data/
│   ├── test_dataset.py
│   ├── test_dataloader.py
│   └── test_benchmark.py
└── e2e/                   # E2E Testing Track test cases (Tiers 1-4)
    ├── tier1_features/
    ├── tier2_boundaries/
    ├── tier3_combinations/
    └── tier4_scenarios/
```
