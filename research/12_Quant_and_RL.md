# Quant liquidation and a learned sell policy

The stage grid fixed the opening: eight wheat plots, a slow herd ramp, no
selling under the base price. This note is about the milk sell path after
that opening. Labor in every agent below is the `care_mill` script. Only
the milk order changes.

`care_mill` is still the agent to beat on a shared book that never breaks.
The learned policy beats it when the quote stays rich, and on the seed
where the quote crashes through the base.

## What the curve says

Milk below equilibrium is `160 + 8.691 * sqrt(deficit)`. Above it,
`160 - 2.098 * surplus`. Deficit is how far inventory sits under 10,000.
Shops remove milk every four turns. A sale on our turn walks the quote
down before the next shop tick.

Two facts from dawn logs (seed 3, `care_mill` vs `pass`):

- The quote is already about $246 by day 14, the first real milk, and it
  stays between $246 and $256 for the rest of a monopoly. Selling the
  surplus does not crash a solo book. The town puts the deficit back.
- Cash starts moving that day. Holding through day 17 donates the peak
  to anyone who is willing to sell.

Against another `care_mill` the paths split:

- Seed 3. The quote peaks near $277 on day 17. The early seller has about
  $10k by day 18 and the holder has almost none. The quote then glides to
  $150. The early seller keeps the lead.
- Seed 5. The early seller leads until day 24, when the quote prints $156,
  under the base. Both farms then refuse milk. The early seller's cash
  falls (wheat and upkeep, nothing to sell). The holder, who sold less on
  the way down, finishes ahead.

## Scipy, before the agents

`scripts/quant_milk_opt.py` prices the official curve. Cows arrive on the
slow ramp and yield 6, then 3 every two days. The shed holds 100.

A grid over four stage fractions (share of the shed sold that day), then
Nelder-Mead from the best cell:

| drain/day | cows | hold | best fractions (days 0–10, 11–16, 17–22, 23–29) |
|-----------|------|------|--------------------------------------------------|
| 8 | 12 | 41,513 | 0.00, 0.09, 0.09, 0.27 |
| 8 | 18 | 37,408 | 0.00, 0.00, 0.00, 0.36 |
| 18 | 18 | 71,273 | 0.00, 0.00, 0.21, 1.00 |
| 30 | 18 | 88,606 | 0.00, 0.00, 0.21, 1.00 |

The search wants almost no sales before day 17, then a dump. That is a
solo shed model. It does not see an opponent, and selling a fraction of a
large shed is a bigger order than the town replaces in one day.

A second fit uses differential evolution on a price band: above `hi` sell
one daily cap, between `lo` and `hi` sell another, stop under the base.
An optional opponent sells a fixed number of units from day 14.

On a deep town (24 milk/day, 14 cows) the search returns the same rule
with no opponent and with an opponent selling 12/day: threshold about
$210, cap about 15 units/day. Cash in that toy model is about $110k solo
and $100k with the opponent. When the opponent sells the whole drain, the
threshold falls toward $187 and the cap shrinks.

## What the engine did with those fits

All of these are replays against `pass` or against `care_mill` on a fixed
seed. Shops match inside a pair. They do not match across pairs, because
each farm's empty tiles change the shop draw. Terminal cash:

| agent | seed 3 vs pass | seed 5 vs pass | seed 7 vs pass | seed 3 vs care (both seats) | seed 5 vs care |
|-------|----------------|----------------|----------------|-----------------------------|----------------|
| care_mill | 67,163 | 43,207 | 54,719 | — | — |
| quant_mill (hold until day 17, then a shed fraction) | 69,507 | 44,603 | 55,659 | care by ~13–17k | quant by ~8–17k |
| trigger_mill (same path, plus sell 3/turn after an $8 dawn drop) | 71,493 | 46,681 | 57,475 | care by ~2–4k | split |
| band_mill (15/day once the quote is ≥ $210) | 65,907 | 40,441 | 53,104 | care by ~14k | care, narrowly |
| floor_mill (keep 8 above $240, keep 20 above $210, else hold) | 68,383 | 44,978 | 56,673 | care by ~9k | floor by ~2–11k |

`band_mill` is the direct encoding of the evolution result, and it lost.
Fifteen units a day is below what a cared herd produces, so milk sat in
the shed at $250–$270. The toy model punished selling faster than that,
because it applies the whole daily order to the deficit with no shop ticks
in between. In the engine the shops nibble every four turns, and a
monopoly quote stayed flat while `care_mill` sold its surplus.

`quant_mill` and `trigger_mill` win the monopoly replays (best solo here
is trigger on seed 3 at $71,493, about $4k over care). They lose the seed
3 head-to-head, because they are still holding while the other farm sells
the $260–$277 window. `floor_mill` wins seed 5 and loses seed 3 harder,
because a hard stop under $210 leaves dozens of unsold units once the
other farm has already walked the quote through that line.

## The learned policy

`scripts/train_sell_q.py` runs paired episodes. Greedy play goes first.
An epsilon-greedy twin (0.45) plays the same seed and opponent. The cash
gap, in thousands of dollars, is credited to each state-action the twin
actually used. Eight pairs: seeds 1, 3, 5, 8, each against `pass` and
against `care_mill`. Mean gap of the noisy twin was +$1.6k. That number
is not the policy; the frozen greedy table is.

State, once per day: stage (before day 14, through day 26, after), price
band ($160, $210, $250), and whether dawn-to-dawn milk fell by more than
$8. Actions are inventory floors of 40, 20, and 8. Anything above the
floor can be sold, a few units a turn, and never under the base. From
day 28 the floor is zero whenever the quote is still at least $160.

Greedy values are the hand prior plus the learned advantage. The table
that results:

- Before day 14, quote $210–$250: floor 8. First milk leaves quickly.
- Quote at least $250 and the dawn price is not falling: floor 20.
- Quote at least $250 and the dawn price is falling: floor 8.
- Quote $210–$250, even if the dawn price fell: floor 20. The prior said
  floor 8 here. The episodes overturned it. A mild dip is not a reason
  to empty the shed.
- Quote $160–$210 and the dawn price is flat: floor 40, which is a hold
  unless the shed is already past that.
- Quote $160–$210 and the dawn price is falling: floor 20. Keep selling
  into a book the other farm is spending. This is the seed 3 correction.
  `floor_mill` held in this band and gave away the head-to-head.

Frozen into `submissions/rl_sell_mill/q_values.json`.

Replays of that greedy policy:

| match | result |
|-------|--------|
| seed 2 vs pass (not in the training seeds) | 55,869 vs care 55,852 |
| seed 3 vs pass | 70,936 vs care 67,163 |
| seed 5 vs pass | 46,416 vs care 43,207 |
| seed 3 vs care, both seats | care ahead by about $2.6k (40.6k vs 43.2k). Both still hold ~50 milk because the quote finished at $143 |
| seed 5 vs care, both seats | learned policy ahead by about $6k |
| seed 3 vs trigger, both seats | split, within $2k |
| seed 5 vs trigger | trigger takes one seat by about $6k, the other seat is even |
| seed 2 and seed 7 vs care | yarn towns, both herds empty, cash within $200 |

## Reading

The monopoly games in this batch top out in the low $70ks, not $138k.
The sell rule is worth a few thousand once the herd exists. It does not
buy the second wave of cows any earlier. That lag is still the gap.

Use `rl_sell_mill` when the comparison is a quiet opponent: it beat
`care_mill` on every monopoly milk seed in this note. Use `care_mill`
when the other farm is also shipping milk into a book that stays above
the base for most of the month. The learned table already moves toward
care's floor of 20 in that band. It does not yet sell the shared peak
fast enough to take seed 3.

Do not ship `band_mill`. The 15-unit cap is what the simulator liked and
what the engine left unsold at $260.
