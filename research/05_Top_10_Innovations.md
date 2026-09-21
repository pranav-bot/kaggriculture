# 05 — Top-10 Innovations

Novel strategies **beyond** the provided anonymous source corpus, aimed at pushing toward a top-10 leaderboard rank. Each idea is designed to compose with Architecture Alpha (online planner) rather than requiring opaque tapes first.

---

## Innovation 1 — Adversarial Lockstep Sale Duel

**Gap in corpus:** Elites react to inventory *deltas* after the fact. Your `simulate_sell_slippage(..., opponent_simultaneous_sell=q)` already models lockstep pricing but is unused in decisioning.

**Idea:** Before emitting premium SELLs, solve a 1-step Stackelberg game:

1. Infer opponent *imminent* sell volume from public farm: sum `yield_units` on tiles at optimal harvest age within Manhattan radius of their shed, plus animals with held product.  
2. For candidate own qty \(q \in \{0, 2, 4, 8, \text{all}\}\), simulate lockstep revenue vs opponent qty \(\hat{q} \in \{0, \hat{q}_{\text{map}}, 2\hat{q}_{\text{map}}\}\).  
3. Choose \(q^*\) maximizing worst-case (or expected) net revenue; if all positive \(q\) lose to dumping, hold 1 turn (collision overlay) unless shed pressure ≥ 88.

**Why top-10:** Same-turn sale races dominate late-game MELON/MILK/WOOL books; reactive Δinv guards are one turn late.

**Sketch:**

```python
def duel_sell_qty(item, have, market_inv, opp_hat, prices):
    best_q, best_score = 0, -1e18
    for q in [0, 2, 4, 8, have]:
        worst = min(
            simulate_sell_slippage(item, q, market_inv, opponent_simultaneous_sell=oq).total_revenue
            for oq in (0, opp_hat, 2 * opp_hat)
        )
        # opportunity cost of holding: shed pressure penalty
        score = worst - 15 * max(0, shed_total + q - 90)  # fictional pressure term
        if score > best_score:
            best_q, best_score = q, score
    return best_q
```

---

## Innovation 2 — Demand-Calendar MPC for Sell/Hold

**Gap:** Sources use coarse post-drain heuristics (`step % 4`) or fixed premium delay. You already have exact `predict_upcoming_consumption_ticks`.

**Idea:** Finite-horizon model-predictive control over the next \(H=12\) turns:

- State: shed vector, market inventories, unlocked shops  
- Controls: sell qty per product per turn (≤ 10 total slots/turn)  
- Dynamics: own sells + predicted town drain + estimated opponent drip  
- Objective: Σ discounted revenue − overflow_penalty − slot_waste  

Solve greedily: each turn, enumerate top-K products by impact, decide sell-now vs wait-until-next-tick if `turns_until_event ≤ 2` and `shed_total < 85`.

**Upgrade over microbatch_opportunist:** decisions are product-specific and calendar-aware, not a fixed “sell 4 after drain.”

---

## Innovation 3 — Multi-Agent Task Auction with Time-Expanded Graph

**Gap:** Alpha uses 1-step bipartite assignment; your controller claims nearest priority tile. Both waste labor on long walks after dawn shed spawn.

**Idea:** Build a time-expanded task graph for the remainder of the day:

- Nodes: (tile, hour) for WATER/HARVEST/FEED deadlines  
- Agents: farmer + hands with current positions  
- Cost: Manhattan + opportunity cost of missing higher `loss_if_omitted`  
- Solve: successive shortest paths or Hungarian on a capped candidate set (≤ 40 tasks × ≤ 10 agents)

Warm-start each dawn: assign full watering routes that snake by rows to minimize empty walking — the failure mode you observed in `melon_surge`.

**Optional RL:** learn a scorer for task values with imitation from Alpha materialize_tasks labels; keep assignment combinatorial (safe under latency).

---

## Innovation 4 — Predictive Opponent Regime Classifier → Counter-Mix

**Gap:** Alpha’s classifier (`livestock-specialist`, `{crop}-specialist`, `aggressive-expander`) is descriptive only. Your `opponent.py` already forecasts harvest windows and sabotage opportunities.

**Idea:** Map opponent regime → **counter production + sale timing**:

| Opponent regime | Counter |
|-----------------|---------|
| Melon rush | Avoid early MELON dump collision; plant CARROT/WHEAT staples; sell melons only post-drain or after their harvest day |
| Yarn/sheep | Pre-buy WOOL demand; delay own WOOL if clone_like; starve shared WHEAT feed prices carefully (ethical/rules-legal market buy of WHEAT only if profitable) |
| Aggressive expander | Match land unlock timing; don’t under-hire when they hit 2–3 quadrants |
| Mixed inactive | Full Alpha champion mix; aggressive impact sells |

Couple with Innovation 1: regime-conditioned \(\hat{q}\) priors for the duel.

**Sabotage (legal):** Front-run their projected harvest day with small premium sells *before* their deposit, then hold during their dump (your `SabotageOpportunity` already names `FRONT_RUN_HARVEST`).

---

## Innovation 5 — Fertilizer / Ongoing-Crop Option Value

**Gap:** Corpus underuses ongoing crops (TOMATO/STRAWBERRY) and fertilizer timing relative to shop calendars. Overlays only *delay* FERTILIZER sells.

**Idea:** Treat fertilizer and strawberry tiles as **real options**:

1. Estimate remaining season value of applying fertilizer to MELON/STRAWBERRY vs selling FERTILIZER now (use price curve + your yield helpers).  
2. Only sell fertilizer when `sell_value > option_value * 1.15` or shed pressure forces it.  
3. Plant a bounded STRAWBERRY strip (4–6 tiles) only when BRUNCH/SMOOTHIE/ICE_CREAM shops unlock and day ≤ last_profitable_start; otherwise refuse (prevents late sunk cost).

This is a structured answer to “shop opportunist wins on adaptive sequencing” without copying livestock traps.

---

## Bonus Stretch (If P0–P2 Already Landed)

### B1 — Sparse MCTS for Terminal Week Only

Run PUCT over unit-macro actions (WATER_ROUTE, HARVEST_CLUSTER, RETURN_DROP) for days 27–29 only, with rollout = your greedy controller. Cap 200 sims / turn. Dominates Beta’s fixed 712 window by starting earlier when yield is stranded.

### B2 — Opening Self-Play Fingerprints

Offline: evolve day-0..5 opening parameter vectors (land timing, seed basket, hire curve) via local Elo against your submission panel. Deploy the winning opening as a *config*, not a tape — keeps online Alpha intact.

### B3 — Market-Making via BUY_PRODUCT

Rarely used by elites. When price ≪ base * 0.5 and town demand for that product next day is high, buy small lots to resell post-drain. Strict reserve + impact checks required; treat as experimental.

---

## Suggested Order To Try For Top-10

| Order | Innovation | Depends on | Expected lift |
|-------|------------|------------|---------------|
| 1 | #2 Demand-calendar MPC | P0 capacity guards | Medium–High |
| 2 | #1 Lockstep duel | slippage helper | High in mirror matches |
| 3 | #4 Regime counter-mix | opponent.py | Medium |
| 4 | #3 Time-expanded assignment | P1 hire targets | High throughput |
| 5 | #5 Fertilizer option | shop unlocks | Medium niche |
| 6 | B1 Terminal MCTS | stable midgame | Endgame polish |

**Do not** jump to tapes or native binaries until Innovations 1–2 plus Improvement Plan P0–P1 are beating `shop_opportunist` in local standoffs. Online agents with superior microstructure often close more of the gap to ~80k than opaque schedules that you cannot maintain.
