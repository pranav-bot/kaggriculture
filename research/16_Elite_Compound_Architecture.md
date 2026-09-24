# 16 — The Elite Compound Architecture (Rank 1–3 Forensic Blueprint)

**Date:** 2026-09-24  
**Author:** Quantitative Research Agent  
**Subject:** Empirical Reverse-Engineering of 3000–3100 ELO Agents in Kaggle Kaggriculture

---

## 1. Executive Summary

Previous agents (`care_mill`, `rl_fert_mill`, `velocity_mill`) operated under a "Single-Quadrant Cow Monopoly" hypothesis:
- Confine farm to 1 quadrant (NW, 25 tiles).
- Plant 8 wheat plots strictly for feed.
- Ramp herd to 14–18 cows.
- Never buy land, never plant commercial cash crops.
- Ceiling observed: **~$65,000 – $86,000**.

Forensic inspection of Rank 1, Rank 2, and Rank 3 match replays (`replays/other_agents/rank{1,2,3}/*.json`) revealed that **all three top agents execute an identical, sophisticated compound-expansion macroeconomic strategy**:
- **Day 0–1:** 10 Wheat + 10 Melon + 5 Animals (mixed Cow + Sheep).
- **Day 4–5:** Harvested Wheat is replanted with **Strawberry** (ongoing $120 crop).
- **Day 6–7:** **`BUY_LAND NE` ($1,000)** unlocks Quadrant 2; expand herd to 11–13 animals + plant 20 more Strawberries.
- **Day 9–11:** **Melon Harvest Windfall** (60 units @ $250+ base = **+$15,000 cash injection**).
- **Day 10–12:** **`BUY_LAND SW` ($2,000)** & **`BUY_LAND SE` ($4,000)** unlock Quadrants 3 & 4.
- **Day 14–30:** 20–24 animals (Cows + Sheep) supported by 30+ homegrown Wheat plots and 25–35 ongoing Strawberry plants, generating continuous daily multi-thousand-dollar revenues.
- **Terminal Cash:** **$105,000 – $135,476**.

---

## 2. Quantitative Comparison Table

| Metric | Single-Quadrant Heuristics (`care_mill` / `rl_fert_mill`) | `velocity_mill` (P0 Baseline) | Elite Architecture (Rank 1–3) |
|---|---|---|---|
| **Land Unlocked** | 1 Quadrant (25 tiles) | 1 Quadrant (25 tiles) | **3 to 4 Quadrants (75–100 tiles)** |
| **Day 2 Crops** | 8 Wheat, 0 Cash Crops | 8 Wheat, 0 Cash Crops | **10 Wheat + 10 Melon** |
| **Day 5 Crops** | 8 Wheat | 8 Wheat | **10 Strawberry + 10 Melon** |
| **Day 8 Crops** | 0–8 Wheat | 0–8 Wheat | **35–42 (Strawberry + Melon + Wheat)** |
| **Day 12 Crops** | 0–8 Wheat | 0–8 Wheat | **52–67 (Strawberry + Wheat)** |
| **Day 7 Herd Size**| 4.5 animals | 6–8 animals | **11–13 animals (Cow + Sheep mix)** |
| **Day 14 Herd Size**| 9.3 animals | 14–18 animals | **21–23 animals** |
| **Daily Hand Hiring**| Ad-hoc / Turn-based | Budget-checked hourly | **Batched at Dawn (H0–H1): 6 to 9 hands daily** |
| **Day 11 Cash** | ~$500 – $1,500 | ~$1,000 – $2,500 | **$11,744 – $12,850 (Melon Windfall)** |
| **Day 14 Cash** | ~$2,000 – $5,000 | ~$3,000 – $8,000 | **$18,324 – $23,015** |
| **Terminal Cash** | $43,000 – $67,000 | $57,000 – $86,000 | **$105,000 – $135,476** |

---

## 3. The 5 Engine Mechanics Unlocked by Top Agents

### Mechanic A: Hands are Daily Leases, Not Permanent Capital
* In Kaggriculture, `hands` reset to `0` at every Day boundary (Hour 0).
* `hire_cost(hires_today)` follows the Fibonacci sequence ($1, $1, $2, $3, $5, $8, $13, $21$).
* **Elite Execution:** At Hour 1 of every day, top agents emit `['HIRE'] * 6` to `['HIRE'] * 8`. For just **$20 to $54 total per day**, they acquire an 8-worker labor force for the entire 23 hours of the day!
* Hiring mid-day (Hour 15) pays the same $54 for only 9 hours of work. Dawn hiring maximizes hours-per-dollar by 2.5×.

### Mechanic B: The Melon Capital Catalyst
* Melon seed costs $80, matures on Day 10, yields **6 units per tile**.
* Base price is **$250** with log scarcity.
* 10 melon tiles yield 60 melons.
* Liquidating 60 melons around Day 10–12 injects **$14,000 – $17,000 of pure liquidity** right when land expansion costs ($1k + $2k + $4k) and animal costs ($400 × 12 = $4.8k) are highest.
* Without the Melon catalyst, farms starve for capital on Days 7–10 and cannot afford land expansion.

### Mechanic C: Wheat-to-Strawberry Crop Rotation
* Wheat planted on Day 0 matures on Day 3–4 (yielding 4 units unfertilized, 6 fertilized = 40–60 wheat).
* This provides 40–60 feed units, covering the 5 opening animals for over a week for free.
* Once harvested on Day 4, the vacant tiles are immediately replanted with **Strawberry**.
* Strawberry has `YieldType.ONGOING`: first harvest at age 10, then **every 2 days indefinitely**, base price **$120** with square-root scarcity.
* 10–25 strawberry tiles yield 10–25 fruit every 2 days = **$1,200 – $3,500 passive daily income**.

### Mechanic D: Land Expansion Super-Linear Compounding
* Quadrant 1 (NW) has only 25 tiles. Minus 4 shed tiles, 4 wheat plots, 2 paths = only 15–18 tiles for animals/crops.
* By unlocking NE ($1,000, 25 tiles), total farm size doubles to 50 tiles.
* By unlocking SW ($2,000, 25 tiles), farm reaches 75 tiles.
* By unlocking SE ($4,000, 25 tiles), farm reaches 100 tiles.
* This allows simultaneous operation of:
  - 24 animals (12 Cows + 12 Sheep)
  - 30 Wheat plots (completely eliminating feed purchase costs)
  - 30 Strawberry plots (ongoing high-margin cash stream)

### Mechanic E: Cow + Sheep Multi-Species Hedge
* Milk has base $160 ($\sqrt{x}$ scarcity); Wool has base $200 ($\log$ scarcity).
* By running both Cows and Sheep:
  1. The agent draws from two separate shop demand sinks (`PIZZA_SHOP` / `ICE_CREAM_SHOP` for Milk, `YARN_STORE` for Wool).
  2. Spreading supply across two books prevents crashing either market below base price.
  3. Wool yields every 3 days (4 units), Milk yields every 2 days (3 units) — staggering harvest labor.

---

## 4. Architectural Implementation Plan: `apex_mill`

To reach the 3000+ ELO tier, we will construct **`apex_mill`** integrating:
1. **Dawn Labor Pulse:** Batch hire 6–8 hands at Hour 0/1 daily.
2. **Phase 1 (Days 0–3):** Plant 10 Wheat + 10 Melon. Purchase 2 Cows + 3 Sheep.
3. **Phase 2 (Days 4–6):** Harvest Wheat $\to$ Replant with Strawberry. Save capital for land.
4. **Phase 3 (Days 7–9):** Buy NE Quadrant ($1,000). Expand herd to 10–12 animals + plant Strawberry/Wheat.
5. **Phase 4 (Days 10–13):** Harvest & Liquidate Melons. Buy SW Quadrant ($2,000). Expand herd to 18–22 animals.
6. **Phase 5 (Days 14–27):** Steady-state industrial operation: 20+ animals, 30 Wheat, 25+ Strawberry. Continuous batched selling above base price.
7. **Phase 6 (Days 28–29):** Terminal liquidation of all shed inventory + worker shed convergence.
