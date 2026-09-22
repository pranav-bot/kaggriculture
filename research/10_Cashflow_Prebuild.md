# Cashflow, prebuild, and the next $40k

`care_mill` samples sit around a $72k mean, with single games from about
$53k to $106k. 138k still needs the same cared herd placed earlier, or a
second good that does not steal feed and care. This note covers the agents
built after that diagnosis.

The built-in random opponent is unseeded. Solo figures are samples. A
head-to-head between two of our agents on a fixed episode seed is
deterministic, because shop draws follow the weed stream and both agents
are pure functions of the observation.

## `cashflow_mill`

Idea: milk sitting in the shed is cash that could have bought the next cow
eight days earlier. While the herd is short, sell 2–3 units whenever the
price is at least base. Hold only after the herd is paid for. Also skip the
wheat crop and build 10 pastures on days 0–2.

Four solo samples: $48,334, $39,755, $39,335, $74,408. Mean $50,458.
Milk finished at $224–$278, under the $290-plus endings `care_mill` reaches
when it holds.

Head-to-head against `care_mill`, both seats, seeds 0–3:

| Seed | Seat | cashflow | care_mill |
| --- | --- | --- | --- |
| 0 | 0 | 48,919 | 42,558 |
| 0 | 1 | 48,877 | 42,567 |
| 1 | 0 | 42,562 | 51,094 |
| 1 | 1 | 40,047 | 37,954 |
| 2 | 0 | 44,324 | 44,338 |
| 2 | 1 | 39,152 | 43,616 |
| 3 | 0 | 17,712 | 19,050 |
| 3 | 1 | 17,285 | 19,311 |

`cashflow_mill` had the higher cash in 3 of 8 seats. Selling to fund the
next animal gives the premium away. The shared market in these matches also
pulls both agents down: two herds supplying the same good cannot both hold
a $300 price. That is a different problem from the solo 138k target.

## `prebuild_mill`

Same sell rule as `care_mill` (hold above base, never sell below it). The
only structural change is the opening: no wheat crop, 10 pastures built
before day 3, a larger bought wheat stock, and a faster herd ramp (6 / 12 /
18). The bet is that animals placed on day 3 instead of day 5–6 pick up one
extra production cycle, including the 6-unit first harvest from a full care
wait.

Four solo samples: $20,074 (6 cows), $20,161 (11 sheep), $64,872 (11 sheep),
$11,348 (12 geese). Mean $29,114, against $51,548 for `care_mill` on four
samples drawn the same way. Head-to-head, `prebuild_mill` was ahead in 3 of
8 seats, and the losses were large ($11k vs $16k, $4k vs $10k).

The wheat crop was doing two jobs: feed, and keeping cash free for animals.
Thirty-six bought wheat costs $900. By day 4 the agent was often under $100
with a half-size herd and could not buy the next sack. `care_mill`'s eight
wheat plots supply that feed without the cash. Prebuilding alone, with no
crop, is worse.

## `bridge_mill`

Keep a short feed crop (4 wheat plots, not 8) and build 8 pastures on days
0–2. Same hold-above-base sell rule. Herd ramp stays 4 / 8 / 14 / 18 so day
3 does not spend the wheat budget. If cash is under $150 and wheat will not
cover the herd, sell two units of product so feed can be bought. That is a
survival trigger, not the cashflow dump. Geese still wait, now until day 21,
so a late milk shop can still win the season.

Four solo samples: $74,491 (18 cows), $25,679 (11 sheep), $49,835 (10 cows),
$12,424 (14 cows, placed late). Mean $40,607. `care_mill` on four samples
drawn the same way averaged $50,082. The solo gap is one bad episode, not
a broken herd: animals were fed and cared.

Head-to-head on seeds 40–43, both seats, is deterministic:

| Seed | Seat | bridge | care_mill |
| --- | --- | --- | --- |
| 40 | 0 | 74,421 | 67,874 |
| 40 | 1 | 74,641 | 67,824 |
| 41 | 0 | 24,583 | 18,135 |
| 41 | 1 | 24,583 | 18,135 |
| 42 | 0 | 17,336 | 23,176 |
| 42 | 1 | 15,258 | 14,684 |
| 43 | 0 | 37,400 | 39,057 |
| 43 | 1 | 37,244 | 37,879 |

`bridge_mill` had the higher cash in 5 of 8 seats. The wins are the games
where both agents are on the same good and the earlier pastures matter
(seed 40, about +$7k). It does not raise the solo ceiling. The best
`bridge_mill` sample here is $75k. The best `care_mill` sample on record
is still $105,972.

## Where 138k still is

None of these three agents beat `care_mill`'s best solo games. Two negative
results are solid:

- Selling product to buy the next animal lowers the price more than the
  earlier cow is worth. `cashflow_mill` lost the head-to-head 5–3 and its
  milk finished in the $220s.
- Deleting the wheat crop to free labor strands the herd around $100 of
  cash. `prebuild_mill` lost the head-to-head 5–3 and averaged about $29k.

The useful change is smaller: 4 wheat plots plus pastures already standing
on day 3. That wins a head-to-head against `care_mill` more often than not,
by roughly the value of one extra production cycle, not by the $40k still
missing. Eighteen cows still cannot all be placed before day 8, because
the second wave is paid for with fertilizer and the first milk. That wave's
first yield is eight days after it is placed, so it misses the start of the
season. That delay, not the sell rule, is the remaining gap to 138k.
