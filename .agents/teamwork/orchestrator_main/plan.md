# Master Execution Plan: Kaggriculture Policy Optimization

## Objective
Optimize the Kaggriculture policy to satisfy all user requirements:
- R1: Complete Land Utilization & Multi-Quadrant Scaling (NE and SW quadrants)
- R2: Core Strawberry Cash-Crop Integration (6-8 plots, watering priority, high-margin cashflow)
- R3: Opponent-Aware Market Defense & Adaptive Herd Capping (oversupply detection, dynamic liquidation, commodity pivoting)
- R4: Replay-Beating Iterative Evaluation Loop (beat ladder opponents in replays/ across benchmark seeds, saving final verified agent to agent_final.py and submissions/agent_final/main.py with $100k+ cash)

---

## Phases and Milestones

### Phase 0: Survey & Architecture Discovery (Current)
- Dispatch 3 parallel Explorers:
  - **Explorer 1 (Agent Architecture & Current Policy)**: Analyze `agent_final.py`, `agent.py`, `main.py`, quadrant expansion logic, strawberry planting gates, pasture allocation, worker scheduling.
  - **Explorer 2 (Environment Mechanics, Replays, & Opponent Forensics)**: Analyze `replays/my_agents/` and `replays/other_agents/`, `scripts/eval_against_replays.py`, market wholesale pricing formulas, oversupply triggers, cashflow dynamics.
  - **Explorer 3 (Testing Harness, Evaluation Seeds, & Verification Rules)**: Analyze benchmark evaluation scripts, seeds (1, 3, 5, 7, 10, 15, 20), `test_submission.py`, performance constraints (<2ms/turn, self-contained single-file).
- Synthesize findings into `PROJECT.md` with full Feature Inventory, Milestones, and Interface Contracts.

### Phase 1: Dual-Track Execution Setup
- **Track 1: E2E Testing Track**:
  - Independent requirement-driven test harness and test suites across Tiers 1-4 (Feature Coverage, Boundary Cases, Pairwise Combinations, Real-World Replay Benchmarks).
  - Publishes `TEST_INFRA.md` and `TEST_READY.md`.
- **Track 2: Implementation Track**:
  - Sequential/parallel milestones executed via Sub-Orchestrators or standard iteration loop (Explorer -> Worker -> Reviewer -> Challenger -> Auditor -> Gate).

### Phase 2: Implementation Milestones
- **Milestone 1 (R1 - Land Utilization & Multi-Quadrant Scaling)**:
  - 100% usable tile utilization on NE quadrant.
  - Automatic expansion to SW quadrant ($2,000) when cash > $3,000 on Days 10–18.
  - Multi-quadrant pathfinding and worker dispatch optimization without shed congestion.
- **Milestone 2 (R2 - Core Strawberry Cash-Crop Integration)**:
  - 6-8 Strawberry plots planted starting Day 3–6 independently of livestock shop unlock.
  - Strict watering priority for thirsty/endangered plants (`consecutive_unwatered >= 1`) before fertilizer collection.
  - Capture $200–$280 market quotes for non-collapsing cashflow.
- **Milestone 3 (R3 - Opponent-Aware Market Defense & Adaptive Herd Capping)**:
  - Real-time wholesale price tracking for Milk and Wool relative to base ($160 Milk, $200 Wool).
  - Dynamic oversupply detection; freeze herd expansion when opponent floods commodity.
  - Market-clearing dynamic liquidation instead of terminal price hoarding.
  - Commodity pivoting (Milk flooded -> Sheep; Wool flooded -> Cows + Strawberries).

### Phase 3: Final Integration & Replay-Beating Loop (R4)
- **Phase 3A: E2E Replay Validation**:
  - Benchmark evaluation across seeds (1, 3, 5, 7, 10, 15, 20) and replay matches against `BenPalmer59`, `Clement Ling`, and Rank 1/2/3 elite agents.
  - Iterate until 100% replay win rate and >$100,000 average cash balance.
- **Phase 3B: Adversarial Coverage Hardening (Tier 5)**:
  - White-box challenger generates edge-case seeds and adversarial opponent behaviors to verify robustness.

### Phase 4: Final Packaging & Forensic Audit Gate
- Export final winning self-contained agent to `agent_final.py` and `submissions/agent_final/main.py`.
- Run `test_submission.py` (<2ms turn limit, standalone verification).
- Independent Forensic Audit verification (zero hardcoding, genuine logic).
- Sentinel report & completion claim.
