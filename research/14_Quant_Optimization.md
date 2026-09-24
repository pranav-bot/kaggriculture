# Research Progress: Quantitative Optimization & Agent Development

**Updated:** 2026-09-24
**Phase:** Active experimentation — scipy optimization, RL training, new agents

---

## 1. Full Product Price Sensitivity Analysis

Every product's price response to inventory changes (from `quant_full_product.py`):

| Product | Base | Scarcity | T | @+10 | @+50 | @+100 | @+200 | @-10 | @-50 | @-100 | @-200 |
|---------|------|----------|---|------|------|-------|-------|------|------|-------|-------|
| WHEAT | 25 | sqrt | 400 | +3 | +7 | +10 | +14 | -2 | -3 | -4 | -4 |
| CARROT | 35 | hinge | 450 | +1 | +4 | +8 | +16 | -4 | -8 | -12 | -16 |
| TOMATO | 60 | hinge | 200 | +1 | +6 | +12 | +24 | -8 | -18 | -25 | -36 |
| STRAWBERRY | 120 | sqrt | 100 | +27 | +59 | +84 | +119 | -19 | -96 | -119 | -119 |
| MELON | 250 | log | 300 | +21 | +34 | +40 | +46 | -1 | -25 | -100 | -249 |
| EGG | 50 | hinge | 332 | +1 | +3 | +6 | +12 | -4 | -7 | -8 | -9 |
| MILK | 160 | sqrt | 122 | +27 | +61 | +87 | +123 | -21 | -105 | -159 | -159 |
| WOOL | 200 | log | 105 | +21 | +34 | +40 | +45 | -6 | -145 | -199 | -199 |
| FERTILIZER | 100 | linear | 200 | +2 | +10 | +20 | +40 | -2 | -10 | -20 | -40 |

### Key Insights:
- **Milk and Strawberry** have sqrt scarcity → best premium from deficit
- **Wool and Melon** have log scarcity → premium tops out fast, but **glut is devastating** (sq penalty)
- **Eggs** barely move — hinge at T=332 means 18 geese can't break the threshold
- **Fertilizer** is perfectly linear — no penalty for dumping, no reward for holding
- **Implication:** Milk is unambiguously the best animal product. Wool only viable on yarn-store-only boards

---

## 2. Scipy Product Optimization Results

### Animal Product Net Cash (18 animals, drain=12)

| Animal | Product | Ramp | Net Cash | Start Day | Cap | Floor |
|--------|---------|------|----------|-----------|-----|-------|
| COW | MILK | cash | **$85,979** | 20 | 2 | $80 |
| COW | MILK | fast | $85,943 | 20 | 2 | $80 |
| COW | MILK | slow | $85,266 | 20 | 2 | $80 |
| SHEEP | WOOL | slow | **$83,151** | 16 | 1 | $239 |
| GOOSE | EGG | fast | $28,542 | 8 | 2 | $25 |

### Joint Milk + Fertilizer Optimization (no opponent)

**Optimal:** 23 cows, milk start day 17, cap 1/day, floor $166, fert floor $69, cap 19/day
**Total cash: $161,211** (theoretical ceiling with this production model)

### Crop Products (8-tile farm, drain=6)

| Crop | Net Cash | Notes |
|------|----------|-------|
| MELON | $13,462 | Best crop, but dwarfed by animals |
| STRAWBERRY | $6,383 | Decent but risky glut |
| TOMATO | $2,113 | Marginal |
| WHEAT | $1,104 | Feed crop only |
| CARROT | $978 | Not worth it |

### Fertilizer (18 cows, no drain)

**Floor doesn't matter** — all fertilizer sells at the same total because the no-drain linear
curve means early selling at high prices = late selling at low prices with same sum.
Scipy confirms floor≈$100, cap=8 gives $10,266 total regardless.

---

## 3. Replay Gap Analysis (vs Elite 3000+ agents)

| Metric | Our Agents (400-600) | Elite (3000+) | Gap |
|--------|---------------------|---------------|-----|
| Terminal cash | $32,887 | $104,902 | **3.2×** |
| Animals day 3 | 0.7 | 5.9 | **8.4×** |
| Animals day 7 | 4.5 | 11.5 | **2.6×** |
| Animals day 14 | 9.3 | 18.7 | **2.0×** |
| PASS rate | 14-40% | 2.4-3.3% | **6-13×** |
| Market slots/turn | 0.81-0.92 | 1.29-4.39 | **1.6-5×** |
| SELL orders total | 378-592 | 1,434-8,190 | **3-14×** |
| Terminal inventory | 45-73 units | 0-17 units | |

### Root causes (priority ordered):
1. **Fixed herd ramp** — agents wait until day 4-5 to buy first animals
2. **Market slot starvation** — sells consume all 10 slots, starving feed/hire
3. **Excessive PASS** — workers have no fallback task chain
4. **Terminal inventory** — unsold goods at step 720 worth $0
5. **No cash-flow reinvestment** — selling product doesn't trigger buying more animals

---

## 4. New Agents Created

### `velocity_mill` (2026-09-24)
Cash-flow driven herd controller implementing all P0 improvements:
- Aggressive ramp: target 6 by day 3, 10 by day 5, 14 by day 7
- Slot-budgeted market: feed+hire first, sells last
- Operating reserve: $100 min cash
- Terminal liquidation: dump everything days 28-29
- PASS elimination: 15 priority-levels in task fallback chain
- Fertilizer: sell everything above $1

**Status:** Testing in progress vs rl_fert_mill and care_mill

---

## 5. Scripts & Tools Created

| Script | Purpose |
|--------|---------|
| `scripts/quant_full_product.py` | Scipy optimizer for ALL products |
| `scripts/train_multi_q.py` | Multi-product RL training framework |
| `submissions/velocity_mill/main.py` | Cash-flow controller agent |

---

## 6. Next Experiments Planned

1. **Head-to-head velocity_mill vs rl_fert_mill** on 10 seeds
2. **Herd ramp grid search** in the engine (not toy model)
3. **Worker PASS rate measurement** for velocity_mill
4. **Multi-product RL training** for milk + fert sell timing
5. **Opponent-aware sell policy** — detect clone/mirror and adjust
6. **Land expansion timing** — when does a 2nd quadrant pay for itself?
7. **Build `surge_mill`** — velocity_mill + RL-trained sell policies
8. **Build `duel_mill`** — collision-aware market overlay

---

## 7. Game Mechanics Reference

### Production yields per animal
- **COW:** First harvest 6 units on day `placed+8`, then 3 every 2 days. Max held: 6
- **SHEEP:** First harvest 6 units on day `placed+6`, then 4 every 3 days. Max held: 6
- **GOOSE:** First harvest 4 units on day `placed+4`, then 2 every day. Max held: 4

### Market pricing formula
```
price(item, inventory) = base + sign × amp × f(|inventory - I0|)
  where I0 = 10,000 and f is the shape function
  Below I0: price rises (scarcity premium)
  Above I0: price falls (glut penalty)
```

### Shop consumption
- Single-product shops (YARN_STORE, PET_CAFE): 2× consumption rate
- Multi-product shops: 1× per product
- Shops consume every 4 turns (6 ticks/day)
- Town center: 1 unit per product per day

### Key costs
- COW: $400, SHEEP: $500, GOOSE: $300
- Wheat seed: $10, Carrot: $20, Tomato: $50, Strawberry: $100, Melon: $80
- Land: NE $1000, SW $2000, SE $4000
- Worker hire: escalating cost via `Actions.hire_cost(n)`
