# Kaggriculture Offline Training Pipeline — Test Infrastructure Specification (`TEST_INFRA.md`)

## 1. Test Philosophy

The Kaggriculture offline reinforcement learning (RL) and counterfactual rollout pipeline is built for high-stakes competition performance (Top 10 Kaggriculture agent). As such, the test harness is engineered under strict **opaque-box, contract-driven, and mathematically rigorous** principles:

1. **Opaque-Box Verification**: Tests validate observable behaviors, output data structures, mathematical invariants, and error boundaries against authoritative requirements (`ORIGINAL_REQUEST.md`, `PROJECT.md`), rather than coupling to internal implementation nuances.
2. **Progressive Testability**: Features are implemented across sequential milestones (M1 through M4). Tests are structured to run cleanly from day one, employing graceful skip notifications (`pytest.skip`) when target implementation modules are pending in worker milestones. Once a worker implements a component, tests activate automatically without modifications.
3. **Authoritative Oracle Derivation**: Expected behaviors and values are derived from two primary oracles:
   - *The Official Kaggle Environment Specification (`kaggle-environments 1.32.7`)*: For step frames, observation dictionaries, market mechanics, and inventory limits.
   - *The Native Rust Simulator (`kaggserve` / `kaggsim`)*: As the ground-truth deterministic reference for game dynamics, state loading (`LOADSTATE`), and counterfactual forward rollouts (`ROLLOUT`).
4. **Self-Contained Independence**: Every test case sets up its own state, utilizes isolated fixtures or temporary directories, generates synthetic or sampled data, and guarantees no order-dependent side effects or disk pollution.
5. **Adversarial & Boundary Rigor**: Extreme values (bankruptcy, 100,000-unit market inventory oversupply, division-by-zero risks, null/locked tiles, shed capacity overflow, subprocess pipe breaks) are explicitly tested to ensure zero unhandled exceptions.

---

## 2. Feature Inventory Coverage Matrix

The table below maps all 26 architectural features identified in `PROJECT.md` to their corresponding requirement ID (R1–R4), E2E test tier, verification method, and authoritative test file.

| # | Feature | Req ID | Milestone | Tier | Verification Method | Primary Test File |
|---|---------|--------|-----------|------|---------------------|-------------------|
| 1 | Raw JSON Step Frame Parsing | R1 | M1 | Tier 1 & 2 | Gzip streaming, 720-step replay parsing, corruption handling | `tests/e2e/tier1_features/test_r1_eda_schema.py`<br>`tests/e2e/tier2_boundaries/test_r1_boundaries.py` |
| 2 | Observation Schema Validation | R1 | M1 | Tier 1 & 2 | 1.32.7 schema verification, missing key assertions | `tests/e2e/tier1_features/test_r1_eda_schema.py`<br>`tests/e2e/tier2_boundaries/test_r1_boundaries.py` |
| 3 | Tile State Edge Case Handling | R1 | M1 | Tier 1 & 2 | Parsing `None`, `"LOCKED"`, `WEED`, `PASTURE`, `PLANT` | `tests/e2e/tier1_features/test_r1_eda_schema.py`<br>`tests/e2e/tier2_boundaries/test_r1_boundaries.py` |
| 4 | Private Inventory & Shed Tracking | R1 | M1 | Tier 1 & 2 | Carried goods, shed cap 100, midnight discard rule | `tests/e2e/tier1_features/test_r1_eda_schema.py`<br>`tests/e2e/tier2_boundaries/test_r1_boundaries.py` |
| 5 | Automated EDA Report Generation | R1 | M1 | Tier 1 | Metrics summary (length, frequencies, reward distribution) | `tests/e2e/tier1_features/test_r1_eda_schema.py` |
| 6 | Temporal Alignment (`steps[t+1]`) | R1 | M1 | Tier 1 | Action applied at $t$ mapped from $t+1$ observations | `tests/e2e/tier1_features/test_r1_eda_schema.py` |
| 7 | Spatial Tensor `state_t` | R2 | M2 | Tier 1 & 2 | Shape `(23, 10, 10)`, normalized channel bounds $[0, 1]$ | `tests/e2e/tier1_features/test_r2_features.py`<br>`tests/e2e/tier2_boundaries/test_r2_boundaries.py` |
| 8 | Continuous Global Vector | R2 | M2 | Tier 1 & 2 | 30-dim normalized agent vector (cash, time, hands) | `tests/e2e/tier1_features/test_r2_features.py`<br>`tests/e2e/tier2_boundaries/test_r2_boundaries.py` |
| 9 | Opponent State Tensor `opponent_state_t` | R2 | M2 | Tier 1 | Shape `(20, 10, 10)` public grid + cash scalar | `tests/e2e/tier1_features/test_r2_features.py` |
| 10 | Continuous Market Vector `market_t` | R2 | M2 | Tier 1 & 2 | 35-dim wholesale prices, ratios, deviations from 10k baseline | `tests/e2e/tier1_features/test_r2_features.py`<br>`tests/e2e/tier2_boundaries/test_r2_boundaries.py` |
| 11 | Macro-Intent Classifier `action_t` | R2 | M2 | Tier 1 & 2 | 24-class backward mapping with zero fall-through errors | `tests/e2e/tier1_features/test_r2_features.py`<br>`tests/e2e/tier2_boundaries/test_r2_boundaries.py` |
| 12 | Net Worth Delta Reward `reward_t` | R2 | M2 | Tier 1 & 2 | $\Delta$ Cash + physical asset replacement cost | `tests/e2e/tier1_features/test_r2_features.py`<br>`tests/e2e/tier2_boundaries/test_r2_boundaries.py` |
| 13 | Undiscounted Return-To-Go `return_to_go_t` | R2 | M2 | Tier 1 & 2 | Telescoping downstream sum converging to 0.0 at turn 720 | `tests/e2e/tier1_features/test_r2_features.py`<br>`tests/e2e/tier2_boundaries/test_r2_boundaries.py` |
| 14 | Opponent Trajectory K-Means `strategy_cluster` | R2 | M2 | Tier 1 & 2 | Opening turns 0..99 trajectory clustering into cluster ID | `tests/e2e/tier1_features/test_r2_features.py`<br>`tests/e2e/tier2_boundaries/test_r2_boundaries.py` |
| 15 | Chunked Memory-Mapped Disk Cache | R2 | M2 | Tier 1 & 2 | Sharded binary `.npy` / `.npz` / `.pt` files | `tests/e2e/tier1_features/test_r2_features.py`<br>`tests/e2e/tier2_boundaries/test_r2_boundaries.py` |
| 16 | Rust Simulator Subprocess Manager | R3 | M3 | Tier 1 & 2 | Subprocess IPC to `kagg serve` with health checks & respawn | `tests/e2e/tier1_features/test_r3_counterfactual.py`<br>`tests/e2e/tier2_boundaries/test_r3_boundaries.py` |
| 17 | State Serialization & `LOADSTATE` | R3 | M3 | Tier 1 & 2 | Observation dict to Rust engine JSON bidirectional check | `tests/e2e/tier1_features/test_r3_counterfactual.py`<br>`tests/e2e/tier2_boundaries/test_r3_boundaries.py` |
| 18 | In-Memory State Forking & Injection | R3 | M3 | Tier 1 & 2 | Fork state at step $t \in [0, 719]$, substitute market order | `tests/e2e/tier1_features/test_r3_counterfactual.py`<br>`tests/e2e/tier2_boundaries/test_r3_boundaries.py` |
| 19 | Heuristic `ROLLOUT` to Turn 720 | R3 | M3 | Tier 1 & 2 | Stateless rollout to terminal step calculating counterfactual value | `tests/e2e/tier1_features/test_r3_counterfactual.py`<br>`tests/e2e/tier2_boundaries/test_r3_boundaries.py` |
| 20 | Counterfactual Divergence & Unit Tests | R3 | M3 | Tier 1 & 2 | Value difference between factual and counterfactual orders | `tests/e2e/tier1_features/test_r3_counterfactual.py`<br>`tests/e2e/tier2_boundaries/test_r3_boundaries.py` |
| 21 | PyTorch `Dataset` Implementation | R4 | M4 | Tier 1 & 2 | 8-element transition tuple indexing from disk cache | `tests/e2e/tier1_features/test_r4_dataset_loader.py`<br>`tests/e2e/tier2_boundaries/test_r4_boundaries.py` |
| 22 | High-Performance Collation & Prefetch | R4 | M4 | Tier 1 & 2 | Multi-worker batch collation, pinned memory, prefetching | `tests/e2e/tier1_features/test_r4_dataset_loader.py`<br>`tests/e2e/tier2_boundaries/test_r4_boundaries.py` |
| 23 | Throughput Benchmark (>500 trans/sec) | R4 | M4 | Tier 1 & 4 | DataLoader throughput benchmark asserting $>500$ trans/sec | `tests/e2e/tier1_features/test_r4_dataset_loader.py`<br>`tests/e2e/tier4_scenarios/test_benchmark_pipeline.py` |
| 24 | Memory Leak Verification | R4 | M4 | Tier 1 & 4 | Flat RSS memory consumption over 3+ training epochs | `tests/e2e/tier1_features/test_r4_dataset_loader.py`<br>`tests/e2e/tier4_scenarios/test_benchmark_pipeline.py` |
| 25 | Full E2E Test Suite (Tiers 1-4) | Acceptance | M5 | Tiers 1-4 | End-to-end integration and system-level suites | `tests/e2e/` |
| 26 | Adversarial Hardening (Tier 5) | Acceptance | M5 | Tier 5 | Stress perturbation, memory pressure, invariant validation | `tests/e2e/tier2_boundaries/` & Tier 5 suites |

---

## 3. Test Architecture & Directory Layout

### Directory Hierarchy
```
tests/e2e/
├── conftest.py                       # Shared fixtures, oracles, synthetic data builders
├── tier1_features/                   # Happy path feature tests (>=5 test cases per feature)
│   ├── test_r1_eda_schema.py         # R1: EDA, parsing, schema validation, tile extraction
│   ├── test_r2_features.py           # R2: Spatial tensors, macro-intents, rewards, clustering
│   ├── test_r3_counterfactual.py     # R3: Rust simulator IPC, LOADSTATE, ROLLOUT, divergence
│   └── test_r4_dataset_loader.py     # R4: PyTorch Dataset, DataLoader, prefetching, throughput
├── tier2_boundaries/                 # Edge cases & stress boundaries (>=5 test cases per feature)
│   ├── test_r1_boundaries.py         # R1: Null tiles, locked quadrants, shed overflow, corrupt gzip
│   ├── test_r2_boundaries.py         # R2: Bankruptcy, division by zero, step 0/719, unmapped moves
│   ├── test_r3_boundaries.py         # R3: Terminal step forks, malformed states, broken pipes, SIGKILL
│   └── test_r4_boundaries.py         # R4: 1-sample dataset, uneven chunks, worker=0, non-CUDA fallback
├── tier3_combinations/               # Pairwise module interactions (pipeline integration)
└── tier4_scenarios/                  # Full offline RL loop & multi-epoch throughput benchmarks
```

### Shared Fixtures in `tests/e2e/conftest.py`
- `sample_episode_path`: Locates an actual gzip replay from `datasets/il/episodes/00/` or synthesizes a valid fallback replay.
- `sample_episode_data`: Decodes and parses sample replay frames into a Python dict.
- `sample_raw_observation`: Provides a canonical step-0 player observation conforming to 1.32.7 schema.
- `synthetic_edge_case_observations`: Supplies an inventory of boundary states (`all_null_tiles`, `all_locked`, `shed_overflow`, `extreme_market`, `bankrupt_farm`).
- `kagg_binary_path`: Locates compiled `kagg` binary or skips simulator-dependent tests if unavailable.
- `mock_rust_engine_state`: Provides valid JSON state payloads matching the Rust engine's `LOADSTATE` expectations.
- `temp_cache_dir`: Isolated temporary directory with automatic teardown for testing chunked dataset caches.

### Pytest Runner Commands
```bash
# Execute entire E2E test harness
uv run pytest tests/e2e/ -v

# Execute Tier 1 Feature Coverage tests only
uv run pytest tests/e2e/tier1_features/ -v

# Execute Tier 2 Boundary & Stress tests only
uv run pytest tests/e2e/tier2_boundaries/ -v

# Filter by requirement tag
uv run pytest tests/e2e/ -k "r1" -v
uv run pytest tests/e2e/ -k "r2" -v
uv run pytest tests/e2e/ -k "r3" -v
uv run pytest tests/e2e/ -k "r4" -v
```

---

## 4. Real-World Application Scenarios (Tier 4)

Tier 4 tests model end-to-end user workflows:
1. **Full Replay Ingestion to DataLoader Stream**:
   - Ingest raw `.json.gz` episode logs from disk.
   - Run EDA schema validation and extract aligned transitions `(obs_t, action_{t+1})`.
   - Compute spatial `state_t (23, 10, 10)`, global `market_t (35)`, and macro `action_t`.
   - Calculate step net-worth `reward_t` and telescoping `return_to_go_t`.
   - Fork selected states at $t=100, 200, 300$, query `kagg serve` for counterfactual rollout value.
   - Save chunked memory-mapped arrays and load via `KaggricultureDataset` and `DataLoader`.
   - Validate batch shape, tensor type, and gradient-ready float32 tensors.
2. **Throughput Benchmark ($>500$ transitions/sec)**:
   - Stream batches of size 64 and 128 through multi-worker DataLoader.
   - Record transition processing rate across 5,000+ transitions.
   - Assert throughput strictly exceeds 500 transitions/sec.
3. **Multi-Epoch Memory Leak Verification**:
   - Run continuous DataLoader iteration across 3 full epochs.
   - Monitor Resident Set Size (RSS) memory consumption via `psutil`.
   - Assert memory growth between Epoch 1 and Epoch 3 remains flat ($< 5\%$ variance).

---

## 5. Coverage & Reliability Thresholds

| Dimension | Threshold Specification | Enforcement Mechanism |
|-----------|-------------------------|-----------------------|
| Feature Coverage | $\ge 5$ test cases per requirement feature (R1–R4) | Static test count assertion & coverage report |
| Boundary Coverage | $\ge 5$ boundary/corner test cases per feature (R1–R4) | Boundary test suite structure |
| Suite Pass Rate | 100% pass on implemented modules; 0 unhandled fatal crashes | Pytest exit code 0 |
| Graceful Skip Handling | Clean `SKIPPED` status for planned modules (M1–M4 in progress) | `pytest.skip` guarded import checks |
| Execution Time | Full Tier 1 + Tier 2 execution $< 30$ seconds (CPU) | Pytest `--durations` tracking |
| DataLoader Throughput | $> 500$ transitions / second at batch size 64 / 128 | Benchmark assertion in R4 suite |
| Memory Stability | $< 5\%$ RSS growth across 3 training epochs | `psutil` memory tracking fixture |
| Flakiness Tolerance | Zero flake tolerance (100% deterministic test results) | Fixed RNG seeds across all tests |
