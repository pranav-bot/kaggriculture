# 17 — Apex Mill: Multi-Quadrant Compound Expansion

**Date:** 2026-09-24  
**Author:** Quantitative Research Agent  
**Subject:** Discovery of Day-Labor Reset Dynamics, Multi-Quadrant Land Expansion, and Creation of `apex_mill`

---

## 1. Executive Summary & New Performance Peak

Building upon forensic replay analysis of 3000–3100 ELO agents and Scipy pre-analysis, we designed **`apex_mill`** ([`submissions/apex_mill/main.py`](file:///Users/pranav/dev/kaggriculture/submissions/apex_mill/main.py)).

### Benchmark Results Across 7 Episode Seeds

| Agent | Mean Cash | Seed 1 | Seed 3 | Seed 5 | Seed 7 | Seed 10 | Seed 15 | Seed 20 |
|---|---|---|---|---|---|---|---|---|
| **`apex_mill`** | **$67,375** | **$42,586** | **$89,632** | $40,268 | **$86,540** | **$83,561** | $84,410 | **$44,630** |
| `velocity_mill` | $57,109 | $25,422 | $67,999 | **$55,653** | $69,822 | $59,529 | **$86,855** | $34,486 |
| `rl_fert_mill` | $56,639 | $36,354 | $73,774 | $53,462 | $62,966 | $61,762 | $69,309 | $38,848 |
| `care_mill` | $49,344 | $27,169 | $67,163 | $43,207 | $54,719 | $58,436 | $64,061 | $30,654 |

**Key Takeaways:**
- `apex_mill` beats `velocity_mill` by **+$10,266 (+18.0%)** on average.
- `apex_mill` beats `rl_fert_mill` by **+$10,736 (+19.0%)** on average.
- `apex_mill` achieved **$83,000 – $89,632 on 4 out of 7 seeds**, setting new all-time records for Seeds 1, 3, 7, 10, and 20.

---

## 2. Core Game Mechanics Discovery: Daily Labor Reset

Previous research assumed farm hands were permanent assets with high cumulative capital costs.
Forensic extraction of `replays/other_agents/rank1/*.json` revealed the true engine mechanics:
1. **Hands are Daily Leases:** All farm hands expire at Hour 0 of every day (`hands` count resets to 0; only the farmer persists).
2. **Fibonacci Cost Function:** `hire_cost(hires_today)` resets to $1 at every midnight boundary:
   - Hand 1: $1
   - Hand 2: $1
   - Hand 3: $2
   - Hand 4: $3
   - Hand 5: $5
   - Hand 6: $8
   - Hand 7: $13
   - Hand 8: $21
   - **Total cost for an 8-worker labor force:** **$54 per day**.
3. **The Dawn Labor Pulse:**
   - Top agents emit `['HIRE'] * 6` to `['HIRE'] * 8` at Hour 0 or Hour 1 of every day.
   - For a nominal $20–$54 daily fee, they unlock 144–192 worker actions every day.
   - Ad-hoc hiring at Hour 15 pays the exact same $54 but only receives 9 hours of work (a 62% waste of labor capital).

---

## 3. The Multi-Quadrant Expansion Model

### Mathematical Payoff of Unlocking Quadrant 2 (NE, $1,000)
From Scipy analysis (`scripts/quant_land_labor.py`):
- Quadrant 1 (NW) saturates at 18 pastures / crops due to shed and path constraints.
- Unlocking NE on Day 6–8 provides 25 fresh tiles.
- 6 additional cows placed on Day 8 yield 144 extra milk units by Day 30.
- At an average sale quote of $175:
  $$\text{Gross Revenue} = 144 \times \$175 = \$25,200$$
  $$\text{Total Cost} = \$1,000 \text{ (land)} + \$2,400 \text{ (cows)} + \$2,250 \text{ (feed)} = \$5,650$$
  $$\text{Net Payoff} = +\$19,550$$

---

## 4. `apex_mill` Architecture

### A. Progressive Capital Allocation
1. **Day 0:** Deploy starting $3,000 into:
   - 8 Melon seeds ($640) + 8 Wheat seeds ($80)
   - 2 Cows ($800) + 2 Sheep ($1,000)
   - 6 Wheat feed ($150)
   - 4 Dawn hands ($7)
2. **Day 1–4:** Plant 8 Wheat + 8 Melon. Build 4 pastures. Place livestock.
3. **Day 4:** Wheat harvests, providing 40–50 free feed units. Vacant plots replanted with Strawberry.
4. **Day 6–7:** Land expansion to **NE Quadrant** ($1,000). Total farm size reaches 50 tiles.
5. **Day 10–11:** **Melon Harvest Windfall:** 48–60 Melons harvested and sold across Days 10–12 for **$12,000–$15,000**.
6. **Day 10–12:** Land expansion to **SW Quadrant** ($2,000). Total farm size reaches 75 tiles.
7. **Day 12–26:** Scale herd to 20–24 animals (Cows + Sheep). Expand Strawberry plantations. Continuous batched selling above base prices.
8. **Day 27–29:** Terminal liquidation of all shed stocks (fertilizer, milk, wool, melons) down to zero.

---

## 5. Next Planned Iterations

1. **Head-to-Head Duel Benchmark:** Test `apex_mill` directly against `velocity_mill` in shared-market matches (`scripts/h2h_bench.py`).
2. **Adversarial Collision Guard:** Implement Overlay B from research docs to prevent price undercutting wars when competing against duplicate apex bots.
3. **Feed Crop Self-Sufficiency in SW Quadrant:** Plant 20 continuous wheat plots in Quadrant 3 to achieve 100% feed independence.
