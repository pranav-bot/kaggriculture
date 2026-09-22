# Stage experiments

`care_mill` is still the highest solo agent (one sample at $105,972, an
eight-sample mean near $72k). `bridge_mill` wins a head-to-head more often
than not, by about one production cycle, and does not raise that ceiling.
The three agents after `care_mill` each changed the opening, the sell rule,
and the herd ramp together, so a loss could not be pinned on one stage.

This file records the questions first, then the measurements.

## What each stage can change

| Days | Scarce resource | Decision that moves cash |
| --- | --- | --- |
| 0–2 | Labor is free. Cash is not. | How many wheat plots versus how many pastures. Wheat plots become feed without a market bill. Pastures let day 3 place an animal in one walk. |
| 3–10 | No milk yet. Fertilizer is the only new cash. | How many animals to buy before the wheat stock and the $3,000 run out. A cow placed here yields on day 11 and can open at 6 milk if it was cared every day. |
| 11–18 | First yields. | Whether to sell into a price that is already above base, or hold and buy the second wave with fertilizer only. Selling funds cows whose first yield is eight days later. Holding keeps the price up for the milk already coming. |
| 19–29 | Herd is fixed. | Sell while price stays at or above base. Selling below base walks milk about $2 per extra unit and wool off a cliff. |

A cow cared every day from placement produces 6 on the first cycle and 3
every two days after that. Moving the whole herd earlier by one cycle is
about `18 × 3 × $250 = $13,500` if the price holds. That is the size of
effect worth detecting. It is not, by itself, the $40k still missing from
138k.

## Controls

The probe is `submissions/stage_probe`. It always raises cows, so species
choice does not mix with the stage effects. The opponent is `pass`, which
does not draw from system entropy. With a fixed episode seed the shop
stream still depends on how many empty tiles roll a weed, so two wheat
settings on the same seed are not the same shops. Sell settings that place
the same animals on the same days should see the same shops.

Factors:

- Wheat plots: 0, 4, 8
- Sell: `hold` (care_mill rule) or `trickle` (one unit per turn while price ≥ base)
- Ramp: `slow` (4 / 8 / 14 / 18) or `fast` (6 / 12 / 18)

Seeds 1 and 5. Terminal cash, ending herd, ending milk price.

Hypothesis, before the runs: wheat 0 goes broke the way `prebuild_mill`
did. Trickle lowers the milk price more than the earlier cows repay.
Fast ramp only helps when the wheat plots exist to feed it. The best cell
is expected to be wheat 4 or 8, hold, and whichever ramp still has the
herd fed on day 10.

## Results: wheat × sell × ramp

Opponent `pass`, cows forced to 18, seeds 1 and 5. A repeated cell matched
exactly ($64,443 both times), so this table is a replay, not noise.

| Mean cash | Wheat | Sell | Ramp | Seed cells (cash, cows, end milk) |
| --- | --- | --- | --- | --- |
| 66,011 | 8 | hold | slow | 64,443 / 67,579, milk 154 and 158 |
| 63,846 | 8 | trickle | slow | 61,636 / 66,056, milk 203 and 200 |
| 54,372 | 0 | hold | fast | ~50–58k, milk ~155 |
| 50,469 | 4 | hold | fast | 87,357 milk 242, and 13,581 milk 154 |
| 15,566 | 4 | hold | slow | 31,113, and 19 with the herd dead |

Every surviving cell fed the whole herd (`unfed` 0). Wheat 0 never matched
wheat 8. Wheat 4 is the fragile one: the fast ramp's best single game
($87,357) and the slow ramp's collapse ($19) are the same feed setting.

Hold beat trickle on cash. Trickle left milk near $200 because one sale a
turn does not clear the shed; the extra units overflow and are discarded.
Hold sells the burst, walks the price back to base, and banks more.

Fast ramp is not a free cycle. On seed 1 with 4 wheat it was worth about
$20k over the slow cell. On seed 5 the same cell lost about $50k. Slow ramp
with 8 wheat is the stable point, and it is the `care_mill` opening.

Ending milk near $155 means 18 cows met or passed the drain on these two
shop streams. Cash near $66k is what selling the whole deficit down to base
is worth here.

A first herd-cap sweep was not usable: fewer cows left more empty tiles,
the weed rolls changed, and the shops changed with them.

## Results: herd cap with the same shops

Wheat 8, hold, slow ramp, 18 pastures built on the same schedule. Opponent
`pass`. Seeds 3, 5, and 7 drew the identical shop list for every cap.
Seed 4 did not, and is left out.

| Seed | 8 cows | 12 cows | 18 cows | Drain formula |
| --- | --- | --- | --- | --- |
| 3 | 68,241 milk 329 | 79,910 milk 318 | **94,953 milk 289** | 81,608 (16 cows, milk 308) |
| 5 | 56,038 milk 250 | **63,683 milk 227** | 56,530 milk 154 | 46,918 (12 cows, milk 266) |
| 7 | 57,191 milk 242 | **65,050 milk 216** | 56,384 milk 156 | 54,554 (8 cows, milk 249) |

When the town can absorb 18 cows, milk stays near $290 and 18 wins by
about $15k over 12. When it cannot, 18 walks the price to base and ties
the 8-cow herd, and 12 wins by about $8k. The drain formula, which sizes
the herd from the shops open today, placed animals later than the fixed
ramp and made less money at the same final count (seed 5, both ended at
12 cows, $47k versus $64k).

The schedule that won is still the slow ramp. The cap on top of it:

- matched cows ≥ 16 (shop drain / 1.5): allow 18
- matched ≥ 8: stop at 12
- otherwise: stop at 8

`sized_mill` is that rule plus `care_mill`'s species choice, wheat bridge,
and hold-above-base selling.

## `sized_mill` against other agents

Against `pass` the live cap still bought 18 cows on seeds 3 and 5, because
later shops pushed the tier up. Cash was $65,272, $76,255, and $66,449,
with milk still above $260. That is the deep-town cell, not the thin-town
cell the cap was meant to catch.

Four solo samples against the unseeded random opponent averaged $61,955
for `sized_mill` and $49,679 for `care_mill`. One of the `care_mill`
samples was an 18-goose game at $17,366, which is most of the gap.

Head-to-head on seeds 3, 5, and 7, both seats:

| Seed | Seat | sized_mill | care_mill |
| --- | --- | --- | --- |
| 3 | 0 | 52,173 | 57,333 |
| 3 | 1 | 68,248 | 61,524 |
| 5 | 0 | 53,963 | 56,507 |
| 5 | 1 | 41,565 | 44,111 |
| 7 | 0 | 34,426 | 38,054 |
| 7 | 1 | 34,668 | 35,009 |

`sized_mill` was ahead in 1 of 6 seats. Once both farms supply the same
good, the shared price falls and the larger herd books more cash before
that happens. The cap that wins against an empty town loses that match.
`care_mill` stays the agent to beat. `sized_mill` is the measured
alternative for a thin town with no opposing herd, and the live tier does
not lock that thin-town cap in time.
