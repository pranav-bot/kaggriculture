# 15 — Velocity Mill & Head-to-Head Breakthrough

**Date:** 2026-09-24  
**Focus:** Forensic replay analysis, scipy multi-product optimization, implementation of `velocity_mill`, and direct Head-to-Head validation vs top baselines.

---

## 1. Executive Summary & Breakthrough Results

By analyzing match replays from top-tier agents (rated 3000–3100) against current implementations (rated 400–600), we identified five structural flaws holding back previous agents:
1. **Fixed Turn-Based Schedules:** Herd expansion was gated by rigid day numbers rather than available capital.
2. **Market Slot Starvation:** Emitting SELL orders consumed the 10-slot limit, silently dropping critical feed (`BUY_PRODUCT WHEAT`) and worker hires.
3. **Dead Feed Inventory:** Buffering 16–36 wheat late into the season burned up to $900 in unspent cash.
4. **Labor PASS Idleness:** Workers passed on 14–40% of turns due to lack of a deterministic fallback task chain.
5. **Terminal Inventory Waste:** Dozens of unliquidated goods at Step 719 were lost for $0.

To solve this, we created **`velocity_mill`** ([`submissions/velocity_mill/main.py`](file:///Users/pranav/dev/kaggriculture/submissions/velocity_mill/main.py)).

### Competitive Head-to-Head Showdown (Both Seats, 10 Games)

In direct head-to-head matches against the previous best agent (`rl_fert_mill`), on identical board seeds across Player 1 and Player 2 seats:

| Seed | Seat | `velocity_mill` Cash | `rl_fert_mill` Cash | Winner | Cash Delta |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | P1 | $47,589 | $59,264 | `rl_fert_mill` | -$11,675 |
| 1 | P2 | $39,874 | $29,440 | **`velocity_mill`** | **+$10,434** |
| 2 | P1 | $28,188 | $19,600 | **`velocity_mill`** | **+$8,588** |
| 2 | P2 | $28,255 | $19,699 | **`velocity_mill`** | **+$8,556** |
| 3 | P1 | $40,095 | $22,917 | **`velocity_mill`** | **+$17,178** (+75%) |
| 3 | P2 | $40,092 | $22,918 | **`velocity_mill`** | **+$17,174** (+75%) |
| 4 | P1 | $45,376 | $42,309 | **`velocity_mill`** | **+$3,067** |
| 4 | P2 | $44,871 | $39,489 | **`velocity_mill`** | **+$5,382** |
| 5 | P1 | $38,620 | $20,243 | **`velocity_mill`** | **+$18,377** (+90%) |
| 5 | P2 | $38,613 | $20,254 | **`velocity_mill`** | **+$18,359** (+90%) |

**Head-to-Head Outcome:**
- **Record:** `velocity_mill` won **9 out of 10 matches (90% Win Rate)**
- **Mean Cash:** `velocity_mill` **$39,157** vs `rl_fert_mill` **$29,613**
- **Net Margin:** **+$9,544 average advantage (+32.2% more capital)**

---

## 2. Forensic Replay Analysis: 500-Rated vs 3100-Rated

From forensic extraction across `replays/my_agents/` vs `replays/other_agents/`:

| Dimension | Our Agents (400–600) | Elite Replays (3000–3100) | Strategic Causal Mechanism |
|---|---|---|---|
| **Day 3 Animals** | 0.7 | **5.9** | Elites pre-build and purchase animals on Day 3 tick 0 |
| **Day 7 Animals** | 4.5 | **11.5** | Reinvestment compounding: profits buy the next animal immediately |
| **Day 14 Animals** | 9.3 | **18.7** | Maxing out quadrant 1 (18 animals) a full week earlier |
| **PASS Action %** | 14.0% – 39.9% | **2.4% – 3.3%** | Elite workers never stand idle; fallback chain keeps units productive |
| **Market Orders** | 378 – 592 | **1,434 – 8,190** | Frequent, continuous micro-sales instead of huge late dumps |
| **Terminal Stock** | 45 – 73 units | **0 – 17 units** | Aggressive liquidation schedule + worker return to shed before turn 720 |

---

## 3. Scipy Optimization & Price Impact Across All Goods

Using [`scripts/quant_full_product.py`](file:///Users/pranav/dev/kaggriculture/scripts/quant_full_product.py), we parameterized the official market dynamics:
$$\text{Price}(I) = \text{Base} + \text{sign}(I_0 - I) \cdot \text{amp} \cdot f(|I - I_0|)$$

### Price Sensitivity at Inventory Perturbations

| Product | Base | Scarcity Curve | $T$ | $+10$ Deficit | $+50$ Deficit | $+100$ Deficit | $-50$ Glut | $-100$ Glut |
|---|---|---|---|---|---|---|---|---|
| **MILK** | $160 | $\sqrt{x}$ | 122 | **+$27** | **+$61** | **+$87** | **-$105** | **-$159** |
| **WOOL** | $200 | $\log(1+x)$ | 105 | +$21$ | +$34$ | +$40$ | **-$145** | **-$199** |
| **FERTILIZER**| $100 | $\text{linear}$ | 200 | +$2$ | +$10$ | +$20$ | -$10$ | -$20$ |
| **STRAWBERRY**| $120 | $\sqrt{x}$ | 100 | +$27$ | +$59$ | +$84$ | -$96$ | -$119$ |
| **MELON** | $250 | $\log(1+x)$ | 300 | +$21$ | +$34$ | +$40$ | -$25$ | -$100$ |
| **EGG** | $50 | $\text{hinge}$ | 332 | +$1$ | +$3$ | +$6$ | -$7$ | -$8$ |

### Optimization Takeaways:
1. **Milk Dominance:** Square-root scarcity gives sustained high premiums without plateauing like log curves.
2. **Wool Fragility:** Quadratic glut penalty makes wool catastrophic in head-to-head matches if both agents produce it (price crashes immediately to $1).
3. **Fertilizer Liquidation:** Linear response with no town drain means holding fertilizer yields zero benefit. Scipy differential evolution proves dumping fertilizer at any price above $1 maximizes compounding cash flow.
4. **Theoretical Joint Ceiling:** Differential evolution of simultaneous cow scaling + milk/fertilizer liquidation yields a theoretical peak of **$161,211** cash.

---

## 4. `velocity_mill` Architectural Blueprint

Implemented in [`submissions/velocity_mill/main.py`](file:///Users/pranav/dev/kaggriculture/submissions/velocity_mill/main.py):

### A. Market Order Priority Queuing
```python
# Priority 1: Feed (guarantees survival)
wheat_floor = 8 if live == 0 else live + 4
# Priority 2: Hiring labor (up to 8 hands)
# Priority 3: Purchasing animals (when affordable + operating reserve)
# Priority 4: Seeds (day 0 only)
# Priority 5: Liquidating shed products (fills remaining slots up to 10)
```
*Guarantees zero slot starvation for feed and expansion.*

### B. Lean Feed Management
- Replaced the old rule `max(live * 2, 8)` with `live + 4`.
- Prevents buying surplus grain on days 20–29. At game end, wheat inventory dropped from **36 units** down to near zero.

### C. 15-Level Worker Task Fallback Chain
Eliminates PASS by routing idle hands to productive tasks or proximity staging:
1. Act on animal underfoot (feed, care, harvest, collect fertilizer)
2. Feed hungry animals across the pasture
3. Fetch feed from shed
4. Place carried animals
5. Care uncared animals
6. Harvest mature animals
7. Harvest ripe wheat crops
8. Collect fertilizer
9. Water crops (consecutive unwatered prioritized)
10. Pickup animals from shed
11. Build pastures / coops
12. Plant wheat
13. Clear weeds (`DIG`)
14. Drop carried goods at central shed
15. Reposition towards shed staging area (never stand idle)

### D. Dynamic Sell Pacing & Endgame Dump
- **Early Stage (Days 0–13):** Hold inventory to let town deficit establish high prices.
- **Mid Stage (Days 14–21):** Continuous micro-sales of 2–3 units whenever price $\ge$ base ($160).
- **Late Stage (Days 22–26):** Accelerated pace of 3–4 units.
- **Pressure Relief:** If shed stock $\ge 75$, sell 4; if $\ge 85$, sell 8.
- **Endgame Dump (Days 27–29):** Dump up to 10 units per turn, floor $1, forcing workers to return to the shed on Day 29 Hour 13+.

---

## 5. Multi-Seed Solo Benchmarks

Benchmarking over 7 diverse episode seeds vs passive opponent:

| Agent | Mean Cash | Best Seed Cash | Worst Seed Cash | Win Rate vs Baseline |
|---|---|---|---|---|
| **`velocity_mill`** | **$57,109** | **$86,855** | $25,422 | **+15.7% over `care_mill`** |
| `rl_fert_mill` | $56,639 | $73,774 | $36,354 | Reference |
| `care_mill` | $49,344 | $67,163 | $27,169 | Baseline |

*Note: On favorable milk seeds (e.g. Seed 15), `velocity_mill` achieves **$86,855**, setting a new all-time solo record in the repository.*

---

## 6. Implementation Inventory & Repository Artifacts

- **New Champion Agent:** [`submissions/velocity_mill/main.py`](file:///Users/pranav/dev/kaggriculture/submissions/velocity_mill/main.py)
- **RL-Infused Variant:** [`submissions/surge_mill/main.py`](file:///Users/pranav/dev/kaggriculture/submissions/surge_mill/main.py)
- **Scipy Multi-Product Optimizer:** [`scripts/quant_full_product.py`](file:///Users/pranav/dev/kaggriculture/scripts/quant_full_product.py)
- **Multi-Product RL Trainer:** [`scripts/train_multi_q.py`](file:///Users/pranav/dev/kaggriculture/scripts/train_multi_q.py)
- **Head-to-Head Benchmarker:** [`scripts/h2h_bench.py`](file:///Users/pranav/dev/kaggriculture/scripts/h2h_bench.py)
- **Quant Summary Document:** [`research/14_Quant_Optimization.md`](file:///Users/pranav/dev/kaggriculture/research/14_Quant_Optimization.md)

---

## 7. Next Milestones to Reach 3000+ Rating

1. **Second Quadrant Expansion (`BUY_LAND NE`):** On high-demand milk boards, unlock NE quadrant around Day 9–11 to scale the herd from 18 to 24 cows.
2. **Adversarial Collision Guard:** Detect when an opponent is dumping into the milk book and briefly pause sales for 1–2 turns until the shop drain restores the price.
3. **Multi-Tape Shop Classifier:** If Day 0–3 reveals 0 cow shops and multiple crop/yarn shops, immediately route to specialized sheep or melon lines without wasting pasture capital.
