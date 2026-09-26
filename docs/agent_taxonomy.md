# Agent Architecture & Strategy Taxonomy

This document defines the formal naming convention, algorithmic designs, operational profiles, and benchmark records of the named agents in this repository.

---

## 1. Why Named Agents Replace `agent_final`

Previously, experimental logic was repeatedly overwritten in a single generic file (`agent_final.py`), which led to:
- **Obscured Lineage:** Impossible to determine which commit or strategy achieved a given benchmark or suffered a regression.
- **Silent Pathologies:** Changes that worked against a passive offline opponent broke when facing live ladder competitors (e.g., the "zero-crop" bug where `agent_final: Sovereign Apex k+` scored 508 on the ladder).
- **Difficult Comparative Profiling:** Tools like `h2h_bench.py` and `standoff` require explicit, uniquely named opponents to evaluate paired win rates.

All agents in `submissions/` now adhere to strict, self-describing functional names representing their core economic and algorithmic hypotheses.

---

## 2. Core Agent Taxonomy & Architectural Catalog

### 1. `two_team_grandmaster`
- **Location:** `submissions/two_team_grandmaster/main.py` (also available as `agent_two_team_grandmaster.py`)
- **Family:** Two-Team Segregated Multi-Agent Planner
- **Core Strategy:**
  1. **Two-Team Division of Labor:**
     - **Livestock Team (Units 0–2: Main Farmer + Hands 0–1):** 100% dedicated to animal care (Feed, Care, Milk/Wool harvest, Fertilizer collection, and open pasture delivery). Never distracted by crop fields.
     - **Field & Expansion Team (Units 3–7: Hands 2–6):** 100% dedicated to crop cultivation and expansion (Strawberry planting across NE and SW quadrants, danger watering, crop harvesting, weed digging, and pasture building). Never distracted by livestock.
  2. **Day 0–7 100% Fertilizer Liquidation Rule:** Liquidates 100% of early fertilizer ($500/day for 5 animals), generating $3,500 in non-dilutive liquidity to self-fund the Northeast land expansion ($1,000) on Day 4–5 before milk unlocks.
  3. **High-Frequency Market Clearing:** Evaluates sales *first* (Priority 0) before assessing hiring or expansion, completely eliminating cash starvation at dawn.
  4. **$60 Fibonacci Wage Reserve:** Always holds $\ge \$60$ cash at sunset, ensuring dawn hand hiring is never rejected by the environment.
- **Benchmark Record:**
  - 20-Replay Gauntlet: **6/20 Won (30%)**, average cash **$61,820** against top ladder opponents.
  - Decisive wins: BenPalmer59 (**+$54,818 margin**), M & M & P & Q (**+$33,010 and +$27,307 margins**), Dohwan Kwak (**+$30,549 margin**).

---

### 2. `sovereign_apex`
- **Location:** `submissions/sovereign_apex/main.py` (also available as `agent_sovereign_apex.py`)
- **Family:** Demand-Coupled Sizing & Dual-Species Pivot
- **Core Strategy:**
  - Dynamic Cow and Sheep herd sizing strictly bounded by town shop consumption rates.
  - Dual-species pasture sharing (pivoting between Cow and Sheep based on relative market demand).
  - Strawberry contingency cultivation.
  - 100% terminal liquidation on Days 27–29 down to zero inventory.
- **Forensic Diagnosis (Ladder Submission 508):**
  - Offline performance against passive `pass` bot: **$104,614 mean cash**.
  - Ladder performance: **$34k–$36k** due to rigid Strawberry prerequisites (0 crops planted in live competition) and lack of Southwest land expansion.

---

### 3. `apex_engine`
- **Location:** `submissions/apex_engine/main.py` (also available as `agent_apex_engine.py`)
- **Family:** Compounding Cash-Crop Catalyst
- **Core Strategy:**
  - **Day 0–1 Melon Kickstart:** Plants 6 Melons and 8 Wheat on Day 0.
  - **Day 11 Cash Influx:** 6 Melons mature $\to$ $9,000+ pure cash injection on Day 11.
  - Immediately funds Southwest quadrant unlock ($2,000) and expands herd to 15+ animals and 30+ Strawberries.
  - Opponent-aware milk market defense (caps cows at 4 if opponent owns cows to prevent milk price collapse).

---

### 4. `straw_empire`
- **Location:** `submissions/straw_empire/main.py` (also available as `agent_straw_empire.py`)
- **Family:** High-Margin Cash-Crop Monoculture
- **Core Strategy:**
  - Maximizes Strawberry plot density across NW and NE quadrants.
  - Routine fertilizer application to ensure 2× harvest yields across all 4 production cycles.
  - Exploits the fact that wholesale Strawberry prices climb to $180–$280 without suffering quadratic market penalties.

---

### 5. `care_mill`
- **Location:** `submissions/care_mill/main.py`
- **Family:** Livestock Production Engine (Baseline)
- **Core Strategy:**
  - Automated Cow caring and feeding loop (+1 milk unit per harvest).
  - Proactive Wheat restocking from town or crop plots.
  - Fertilizer collection and late-game wholesale liquidation.

---

### 6. `velocity_mill`
- **Location:** `submissions/velocity_mill/main.py`
- **Family:** Market Demand Velocity Controller
- **Core Strategy:**
  - Real-time scoring of town shops to identify products experiencing the fastest consumption drain.
  - Dynamically weights crop planting based on shop composition.

---

## 3. Benchmark Comparison Matrix

Results measured over full 720-turn matches on the fast Rust simulation engine:

| Agent Template | Primary Mechanism | Land Expansion | Crop Strategy | Gauntlet vs Replays |
|---|---|---|---|---|
| **`two_team_grandmaster`** | Segregated Teams (Livestock vs Field) | NW $\to$ NE $\to$ SW (72 tiles) | Strawberry (Fertilized) + Wheat + Melon | **Highest Win Rate (Wins vs BenPalmer59, M&M, Dohwan)** |
| **`sovereign_apex`** | Demand-Coupled Sizing | NW $\to$ NE (48 tiles) | Strawberry Contingency | Solid offline ($104k), constrained on ladder |
| **`apex_engine`** | Day 11 Melon Cash Injection | NW $\to$ NE $\to$ SW (72 tiles) | Melons (Early) $\to$ Strawberry (Late) | Strong mid-game liquidity bootstrap |
| **`straw_empire`** | Strawberry Yield Doubling | NW $\to$ NE (48 tiles) | 100% Strawberry Focus | High revenue per seed, sensitive to labor |
| **`care_mill`** | Cared-Cow Care & Feed Loop | NW Only (24 tiles) | Wheat feed only | Strong single-quadrant baseline ($50k-$75k) |
| **`velocity_mill`** | Shop Demand Velocity Scoring | NW Only (24 tiles) | Multi-crop opportunistic | Agile adaptation to shop unlocks |

---

## 4. How to Test and Compare Named Agents

```bash
# 1. Compare two named agents head-to-head across 5 seeds (10 games):
python scripts/h2h_bench.py two_team_grandmaster care_mill --seeds 5

# 2. Test a named agent against the 20-replay gauntlet:
python scripts/eval_against_replays.py two_team_grandmaster --quick

# 3. Benchmark terminal cash across 7 standard seeds:
python scripts/eval_cash.py two_team_grandmaster pass 1 3 5 7 10 15 20

# 4. Run round-robin tournament for a named agent against all 62 opponents:
python standoff/run_standoff.py -a two_team_grandmaster --no-swap

# 5. Build and validate a Kaggle-ready submission for a named agent:
python scripts/build_submission.py -a two_team_grandmaster
```
