# 01 — Methods and Metrics

Deep architectural analysis of high-performing Kaggriculture agents (anonymous source data), plus a synthesized catalog of metrics and state-switching triggers.

---

## 1. Meta-Architecture Landscape

Top agents cluster into **four decision paradigms**. Understanding which paradigm you are in determines which upgrades transfer cleanly.

### 1.1 Architecture Alpha — Online Modular Planner

**Pipeline (every turn):**

```
observation
  → GameState adapter
  → materialize_tasks (priority + deadline + loss_if_omitted)
  → assign_actions (sorted unit↔task pairs, Manhattan step)
  → make_market_orders (SELL → BUY_LAND → BUY_SEED → HIRE)
  → validate_joint_action (seed-overplant PASS, clip market slots)
  → (latency budget) else safe_fallback (PASS + dump shed)
```

**Strengths:** Auditable, portable, no opaque tapes, easy to ablate.  
**Weaknesses:** Labor routing is greedy 1-step; no multi-day choreography; livestock lightly handled; shop exploitation is coarse.

**Core algorithms:**

| Module | Algorithm |
|--------|-----------|
| Tasks | Priority ladder 0…8 with explicit `loss_if_omitted` dollars |
| Routing | All (task, unit) pairs sorted by `(priority, dist, deadline, id)`; exclusive assignment |
| Movement | Axis-alternating step (`unit_index % 2`) to desync collisions |
| Production | Config mix early/late + bounded adaptive ±3 tiles from shop pull |
| Market | Price-sorted sells; premium batching; cash projection with operating reserve |
| Safety | Wall-clock cuts at ~0.20s / ~0.40s; exception → sell-all fallback |

### 1.2 Architecture Beta — Multi-Tape Shop Router + Terminal Overlay

**Pipeline:**

```
step < 144:  follow plan-0 opening tape
step == 144: select plan from first-two unlocked shops (lookup table → plans 0..12)
step == 648: force final-plan index (liquidation choreography)
every step:  weed-queue repair → subtract advanced sales → advance next-turn sells
             → day-close room_guard → clip to 10 orders
step 712:    optional terminal search (shadow baseline, propose harvest/deposit routes)
step 718:    liquidate (DROP if beside shed + sell projected shed)
```

**Strengths:** Elite physical choreography baked into tapes; shop-contingent midgame; endgame local search recovers stranded yield.  
**Weaknesses:** Requires offline corpus of full-episode action tapes; fragile under regime shifts; hard to merge with online crop choice.

**Recoverable online algorithms (independent of tape contents):**

1. **Weed repair queue** — if scheduled `PLANT`/`BUILD_*` lands on `WEED`, emit `DIG` and shift the blocked command into a per-worker deque that clears at dawn.
2. **Sale advance** — pull tomorrow’s planned SELLs forward by one turn when price ≥ 2, skipping WHEAT/FERTILIZER, shop ticks (`step % 4 == 0`), and land-unlock edges (`step % 72 == 0`).
3. **Room guard** — at `hour == 23`, if `shed + carried > 99`, sell cheapest-priority excess (price-desc) until ≤ 99.
4. **Terminal planner** — at step 712, simulate 7 remaining turns; propose walk→HARVEST/COLLECT→DROP suffixes per actor; accept only if candidate **dominates** baseline (no overflow, ≥ sold, ≥ deposited per actor) and stock value rises.

### 1.3 Architecture Gamma — Conserved Route + Feature Switches

**Pipeline:**

```
decode route family (main + prefix-compatible tails)
at decision turns: if feature ≥ threshold and prefix matches so far → switch tail
replay route[step]
repairs:
  - noop-on-weed → DIG
  - project same-turn DROP/PLACE into shed
  - hour 23 room_guard targeting capacity-1 (99)
  - clamp SELLs to projected fillable qty
  - liquidate dead stock (qty with zero remaining route sells)
```

**Decision features observed in source data:**

| Step | Feature | Threshold | Meaning |
|------|---------|-----------|---------|
| 226 | count(YARN_STORE) | ≥ 1 | Yarn-aligned livestock continuation |
| 360 | market price CARROT | ≥ 42 | Carrot-premium tail |
| 433 | market inventory MILK | ≥ 10067 | Milk-glut avoidance / alternate dump plan |

**Strengths:** Cheap runtime; value-aware capacity; never wastes market slots on unfillable sells.  
**Weaknesses:** Still tape-bound; feature thresholds are seed-sensitive.

### 1.4 Architecture Delta — Opaque Schedule + Impact Sell Ranking

**Pipeline:**

```
select schedule by config regime (town-center interval ≥ 24 → "rebalance" else "legacy")
replay schedule[step]
weed repair with multi-step DIG then replay intended BUILD/PLANT
reorder existing SELL slots by impact_score × demand_urgency
align hand count to live observation
```

**Sell score (exact mechanics):**

\[
\text{impact}(item,q) = q \cdot \max(0,\; P(I) - P(I+q))
\]

Under rebalance regime only:

\[
\text{urgency} = \min\!\left(1,\; \frac{\max(0, I+q-10000)}{10 \cdot D_{\text{day}}}\right)
\]

\[
\text{score} = \text{impact} \cdot (1 + 0.25 \cdot \text{urgency})
\]

Where \(D_{\text{day}}\) is expected daily town drain for that product (shops + town center, with day-based center multipliers in legacy regime).

**Strengths:** Improves any fixed schedule’s market without rewriting farm work.  
**Weaknesses:** Cannot invent missing SELLs; only reorders present slots.

### 1.5 Overlay Family (Thin Wrappers)

These do not replace a base policy; they rewrite only market or idle labor:

| Overlay | Trigger | Action |
|---------|---------|--------|
| Premium phase shift | Safe delay (cash≥~1600, shed&lt;82, day&lt;27) | Hold MILK/WOOL/STRAWBERRY/FERTILIZER sell 1 turn |
| Collision guard | Δinv≥3 or Δprice≤−3 and price/base&lt;1.10 | Suppress premium SELL into cooldown |
| Clone-aware guard | Opponent farm “looks like” ours | Only then apply collision logic |
| Idle-hand water rescue | Day 6–17, hour 16–21, hand PASS, plant missed water | Round-trip WATER ≤7 cmds, restore tile |

---

## 2. Shared Engine Constraints That Drive Design

All strong agents encode these as first-class constraints:

| Constraint | Implication |
|------------|-------------|
| 10 market orders / turn | Slot economy: never emit unfillable SELL; prioritize high-impact products |
| Shed capacity 100 | Day-close room_guard; continuous sell; target 99 reserve in elite designs |
| Same-turn DROP then market | Must project deposits before clamping sells |
| Hire cost Fibonacci, hands reset daily | Hire is a **throughput budget**, not permanent capital |
| Shop unlock every 3 days; shop consume every 4 turns | Sale timing relative to `step % 4` and unlock ticks |
| Quadratic-ish slippage curves | Batch premiums; impact-score ranking; collision cooldowns |
| Plant all-or-nothing per crop | Validator must PASS overplant rather than invalidate whole crop set |
| Episode 720 steps / 30 days | Terminal day 29 forced harvest/return; stop planting past last profitable start |

---

## 3. Search Methods Inventory

| Method | Used by | Search space | Accept criterion |
|--------|---------|--------------|------------------|
| Configured mix + deficits | Alpha | Crop counts vs targets | Fill empties by deficit ratio |
| Greedy bipartite assign | Alpha | Units × tasks | Exclusive min (priority, dist) |
| Bounded adaptive mix ±3 | Alpha | Carrot↔wheat tiles | Shop pull imbalance |
| Tape lookup | Beta/Gamma/Delta | Discrete plan IDs | Shop pair / feature threshold |
| Prefix-compatible switch | Gamma | Tail set | History equality up to switch turn |
| Sale advance 1-turn | Beta | Next tape market | Price≥2, capacity, slot free |
| Local terminal beam | Beta | Actor route suffixes, ≤64–256 sims | Dominance + value gain |
| Impact reorder | Delta | Permutation of existing SELLs | Higher impact×urgency first |
| Overlay delay / release | Overlays | Pending premium qty | Cash/shed/saturation gates |

**Not observed in top anonymous sources (opportunity for innovation):** full MCTS, RL policies, explicit lockstep opponent sale sims in the online loop (your helpers already have lockstep simulators unused by elites).

---

## 4. Heuristic Priority Ladders

### 4.1 Alpha Field Priorities (lower = more urgent)

| Priority | Condition | Action |
|----------|-----------|--------|
| 0 | `consecutive_unwatered ≥ 1` or animal unfed | WATER / FEED |
| 0–1 | Terminal day with yield held | HARVEST / liquidate |
| 1 | Peak-yield water window | WATER then HARVEST |
| 2–3 | Mature harvest / ongoing max yield | HARVEST |
| 3–4 | Routine water | WATER |
| 7 | Plant to fill mix | PLANT |
| 8 | Dig weed (day &lt; 27) | DIG |

### 4.2 Your Existing Controller Ladder (for comparison)

| Priority | Condition |
|----------|-----------|
| 0 | Weed-danger water / escape-danger feed |
| 1 | Optimal harvest / animal harvest |
| 2 | Water / feed / fertilize / place animal |
| 3 | Soft harvest / care |
| 4 | Plant / collect fertilizer |
| 5 | Dig weeds |

**Gap:** Alpha attaches **dollar `loss_if_omitted`** and **deadlines**; your controller uses distance only as secondary key. Elites with tapes skip online ladders entirely and repair no-ops.

---

## 5. Synthesized Metrics Catalog

Use these as features for phase machines, overlays, and logging.

### 5.1 Capital & Liquidity

| Metric | Definition | Typical thresholds |
|--------|------------|--------------------|
| `operating_reserve` | Cash floor kept after buys | 100 (Alpha) |
| `cash_critical` | Force sells / forbid delay | money &lt; 1300–1600 **or** BUY_LAND/ANIMAL/PRODUCT present |
| `hire_budget` | Target hands by unlocked count | {1→6, 2→9} (Alpha champion) |
| `land_target` | Unlock count by day 2 | 2 (Alpha) |
| `investment_cutoff_day` | Stop land/hire | 26 (your market_velocity) / 27 overlays |

### 5.2 Inventory & Capacity

| Metric | Definition | Typical thresholds |
|--------|------------|--------------------|
| `shed_total` | Σ shed units | pressure ≥ 82 / 84 / 88 |
| `shed_room` | 100 − shed_total | room_guard if ≤ 1 |
| `projected_shed` | shed + same-turn DROP/PLACE − PICKUP | clamp sells to this |
| `carried_total` | Σ unit inventories | included in day-close pressure |
| `premium_batch` | Max premium sell qty/turn | 8 (Alpha batched mode) |
| `capacity_target` | Soft max fill | 99 (Gamma/Beta) |

### 5.3 Market Microstructure

| Metric | Definition | Typical thresholds |
|--------|------------|--------------------|
| `price_ratio` | price / base | delay only if &lt; 1.10 under saturation |
| `inv_delta` | I_t − I_{t−1} | ≥ 3 → collision |
| `price_delta` | P_t − P_{t−1} | ≤ −3 → collision |
| `impact_score` | q·(P(I)−P(I+q)) | sort key for SELL slots |
| `demand_day` | shop+center drain / day | urgency denominator |
| `excess_vs_eq` | max(0, I+q−10000) | urgency numerator |
| `post_drain` | step % 4 == 0 just passed | good sell window |
| `sell_mode` | immediate / batched / delayed | Alpha config |

### 5.4 Production & Season

| Metric | Definition | Typical thresholds |
|--------|------------|--------------------|
| `mix_switch_day` | early→late crop mix | 13 (Alpha) |
| `last_profitable_start` | season−1−grow_days | stop seeds after |
| `terminal_day` / `terminal_hour` | forced return+DROP | day≥29, hour≥13 |
| `adaptive_shift` | |carrot_pull−wheat_pull| capped | ≤ 3 tiles |
| `max_active_plants` | Cap concurrent plants | 40 (Alpha) |
| `missed_water` | consecutive_unwatered | ≥1 → priority 0 |

### 5.5 Opponent / Public Signals

| Metric | Definition | Typical thresholds |
|--------|------------|--------------------|
| `opponent_class` | livestock / crop-specialist / expander / mixed | Alpha classifier |
| `product_pressure` | opp tiles holding product | secondary scarcity |
| `clone_like` | hand_gap≤1, land_gap=0, animal_gap≤2, crop_gap≤5 | enable clone guard |
| `shop_pair` | first two unlocked shops | tape plan key at step 144 |

### 5.6 Search / Latency Budgets

| Metric | Definition | Typical thresholds |
|--------|------------|--------------------|
| `decide_ms_soft` | Abort to fallback | 200 ms |
| `decide_ms_hard` | Abort after routing | 400 ms |
| `terminal_sims` | Max sim evaluations | 64 / 128 / 256 |
| `proposals_per_actor` | Terminal route candidates | 4 / 8 / 16 |
| `idle_rescue_cmds` | Max WATER round-trip | 7 |
| `rescues_per_day` | Cap idle rescues | 2 |

---

## 6. State-Switching Trigger Cheat Sheet

Copy-pasteable transition table for a phase machine.

```
OPENING (day 0–2)
  IF unlocked < land_target AND cash >= land_price + reserve → BUY_LAND
  IF hands < hands_by_unlocked[unlocked] AND cash ok → HIRE
  Plant early_mix (e.g. MELON/CARROT/WHEAT)

EXPANSION (day 3–12)
  Maintain early_mix up to max_active
  Sell: batched premiums OR impact-ranked
  Prefer post-shop-tick sells

ADAPTIVE_MID (day >= mix_switch_day)
  Switch to late_mix (staples)
  IF adaptive: shift ≤3 tiles toward shop-favored staple
  IF yarn shop AND wool price high → livestock branch (optional)

PRESSURE
  IF shed_total >= 82–88 → disable sell delays; room_guard
  IF cash_critical → force sells; cancel premium holds

COLLISION
  IF inv_delta>=3 OR price_delta<=-3 AND price_ratio<1.10
     AND NOT endgame AND NOT pressure → hold premium 1–N turns

TERMINAL (day >= 27)
  Stop expansion/hire
  day >= 29 OR (day==29 AND hour>=13): return+DROP all carried
  Sell all remaining (ignore batching)

STEP_ROUTES (tape agents only)
  step 144: shop_pair → plan_id
  step 648: final liquidation tape
  step 712–718: terminal search overlay if enabled
```

---

## 7. What Actually Separates Top Agents From Midboard

1. **Slot economy** — treating the 10 market orders as scarce capital, not a dump chute.  
2. **Same-turn physics** — projecting DROP before SELL clamp.  
3. **Capacity hysteresis** — defending 99/100, not reacting after overflow loss.  
4. **Shop-contingent midgame** — production or tape choice keyed off first shops, not average prices alone.  
5. **Endgame physical search** — recovering harvestable yield stranded by open-loop schedules.  
6. **Latency-safe fallbacks** — never throw; degrade to legal PASS/SELL.  
7. **Hire as daily throughput** — 6–9 hands once land unlocks, funded after first cash events.  
8. **Premium batching / collision awareness** — avoid self- and cross-agent price crashes.

Your framework already has demand ticks, slippage sims, and opponent profiles — elite code often **simpler** but **stricter** about capacity, slots, and phase cuts. The improvement plan (`04`) maps these into your controller without requiring opaque tapes first.

---

## 8. Recommended Port Priority (Preview)

| Priority | Method | Source family | Effort |
|----------|--------|---------------|--------|
| P0 | Operating reserve + projected DROP sells + plant validator | Alpha | Low |
| P0 | Day-close room_guard to 99 | Beta/Gamma | Low |
| P1 | Mix switch + premium batching + hire-by-unlocked | Alpha | Medium |
| P1 | Impact sell ranking using your slippage helper | Delta | Medium |
| P2 | Collision / premium delay overlay | Overlays | Low |
| P2 | Task deadlines + loss_if_omitted in routing | Alpha | Medium |
| P3 | Terminal 7-turn deposit search | Beta | High |
| P3 | Offline tape corpus (only if online plateaus) | Beta/Gamma/Delta | Very high |

Details and injection snippets: `04_Improvement_Plan_For_My_Agent.md`.
