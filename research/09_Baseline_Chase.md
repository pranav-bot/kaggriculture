# Baseline chase: toward ~138k terminal cash

Goal: terminal cash around 138,000 against a random opponent, with notes on
head-to-head play. Reward is end-of-game cash. These notes are the running
log. Numbers below are from local 720-step episodes.

## What actually pays

138k is volume of a premium good sold while the town is still short of it.
Crop dumps and staple loops top out near 30–37k in this repo. The engine
starts every product at inventory 10,000, which is the base price. Town
center removes one of each product except fertilizer every 24 turns. Each
unlocked shop instance removes its products every 4 turns (single-product
shops remove 2). Shops unlock every 3 days, with replacement, up to 8
instances. Selling moves inventory back up and walks the price down.

Milk scarcity is a square-root curve (base 160, T 122, below-target 0.60):
about `160 + 8.7 * sqrt(deficit)`. A few hundred units of unmet drain is
enough to hold milk near $280–350. Oversupply is linear, about $2.1 off the
price per extra unit, so dumping past equilibrium floors the book.

Wool scarcity barely lifts the price (log curve, below-target 0.20). Wool
gluts use a steep square curve, so overselling wool destroys it. Eggs use a
hinge: flat until the deficit passes T 332, then a sharp spike. Fertilizer
has no town sink. Selling it only adds supply and walks the price down from
$100. It is a funding trick, not a product line.

## Care is the production multiplier

Animals escape when `consecutive_unfed` reaches 2. A missed day is
survivable. Two missed days delete the animal and leave the structure.

On a fed production day the yield is `1 + pending_care_bonus`, capped by
`max_held`. The bonus is cleared, then if the animal was both fed and cared
that day the bonus starts again at 1. Care on a non-production day
accumulates.

| Animal | First yield | Interval | Steady yield if cared every day | Cap |
| --- | --- | --- | --- | --- |
| Cow | day 8 | 2 | 3 milk every 2 days (1.5/day) | 6 |
| Sheep | day 6 | 3 | 4 wool every 3 days | 6 |
| Goose | day 4 | 1 | 2 eggs every day (bonus cannot stack past 1) | 4 |

The first production can be larger because every cared day before the first
yield stacks, up to the cap. A cow cared for the whole 8-day wait opens at 6
milk, then 3 every two days. Harvest whenever `yield_units > 0`. Leaving the
first 6 sitting on the animal wastes every later bonus against the cap.

Hands are wiped every dawn and must be rehired. Eight hires cost 54 (the
Fibonacci sequence). Unit actions run before the market, so a hire, a seed,
or an animal bought this turn is not usable until the next turn. Ten market
orders per turn: eight hires will crowd out wheat and sells if they are
placed first.

Only the northwest 5×5 starts unlocked. One shed tile, (4,4), is in that
quadrant. The other three shed tiles start locked, but pickup and drop still
work on them. Twelve pastures fit in the opening quadrant. Buying land is
optional, not a requirement for a 12-animal herd.

## Experiments so far

### `demand_mill`

Shop-matched herd, then several feed and sell patches.

| Version | Final cash | What happened |
| --- | --- | --- |
| First | 100 | Workers picked up sheep before wheat and never fed. Herd escaped. Nine empty pastures. |
| Feed-first, wheat before animals | 40,327 | Herd lived (8 sheep, 2 cows, 6 geese) but one hungry animal blocked placement, and wool was undersold while its price was still rising. |
| Limited wheat fetchers | 29,991 | Herd lived. Wool was sold every turn, price fell to the mid-50s, then the agent refused to sell under 0.45× base and hoarded. |
| Demand-sized herd, no fertilizer, interval sells | 613 | Shops wanted milk. The agent kept sheep, hoarded wool, and never built the cow line. |

### `shop_supply`

| Version | Final cash | What happened |
| --- | --- | --- |
| Strawberries on day 3 plus a mixed herd | 4,710 | Berries were never watered. Animals escaped, then a late goose trickle. |
| Animals before a wheat stockpile | 50, then 20 | Six animals, about eight wheat, money spent to the floor, escape by day 8. |
| 40 wheat before any animal, then a 6-wheat gate | 19,516 | Herd survived (2 cows, 1 sheep) but the gate never saw 24 wheat at once because feeders ate the stock before the buy check. No further animals. |
| Same-turn wheat order counts as stock, one species not yet | 34,108 | Herd lived (5 cows, 4 geese, 4 sheep) and milk rose from 175 to 324, which means milk was scarce. Too few cows, placed too late, and labor was split across three species. |

Those two agents topped out near 40k. `care_mill` is the one that moved
past that band. The current policy and its numbers are at the bottom.

## Why those runs cannot reach the baseline

A cow placed on day 14 produces for about four cycles. Five of them, without
a full care stack, are a few thousand dollars even at a $300 milk price.
The arithmetic that lands near 138k is roughly 12 cows placed by day 3–6,
cared every day, harvested every production, selling a bit under the milk
drain so the price stays above $280:

- first harvest 6, then 3 every two days
- about 27–36 milk per early cow
- 12 × 30 × $320 ≈ $115k, plus a second good if labor remains

The binding constraints are cash and wheat, not the price formula. Twelve
cows cost $4,800 and eat a wheat a day each. Starting cash is $3,000. The
opening has to buy a feed bridge (grown wheat plus a small purchased stock)
and only then animals. Fertilizer can be sold for a few days to buy the
rest of the herd, then stopped, because it has no sink and the price fades.

## `care_mill` stages

`submissions/care_mill/main.py` is a stage policy, not a search.

1. Days 0–2: hire 8, grow 8 wheat as a feed bridge, buy a small wheat stock.
2. From day 3, score only shop drain (the town-center +1 does not count).
   Weight milk 1.7, wool 0.85, eggs 1.05, times cared output per day.
   Geese are not started before day 15 unless milk and wool are already open.
   The choice can change until more than 4 animals are owned.
3. Herd size ramps 4 → 8 → 14 → 18, and is also capped near
   `shop_drain / cared_output_per_day`. One or two animals per turn, and
   only when wheat covers the new mouth.
4. A worker standing on an animal feeds, cares, then harvests before it
   walks away. Harvest any yield immediately so the care bonus is not
   capped away.
5. Hold milk, wool, and eggs while the shed has room and the price is above
   base. Never sell them below base. Fertilizer is sold only to finish
   paying for the herd.

Buying a second quadrant was tried so a milk-heavy board could hold 24 cows.
It did not raise cash. The extra animals arrived late, and one seed bought
land for sheep and finished at $33k with 21 sheep. The opener stays in the
northwest.

## Iteration log

| Agent | Seed | Opponent | Cash | Herd at day 29 | Note |
| --- | --- | --- | --- | --- | --- |
| shop_supply (mixed, wheat gate relaxed) | unseeded | random | 34,108 | 5 cow, 4 goose, 4 sheep | Alive, undersupplied milk |
| demand_mill (best patch) | unseeded | random | 40,327 | 8 sheep, 2 cow, 6 goose | Alive, wool price still high |
| care_mill, 12 cows, sell early | unseeded | random | 54,449 | 12 cow, fed and cared | Price never got much above $250 |
| care_mill, hold milk until the shed is tight | unseeded | random | 54,942 | 6 cow | Wheat gate froze the herd |
| care_mill, hold + 12 cows | 0, but shops depend on weed rolls | random | 79,170 | 12 cow, unfed 0 | Milk ended at $320. Still short of the drain |
| care_mill, 18 animals, same policy | 0–7 | random | see table | 18, all cared | Mean about $36k. Milk seeds $68–98k. Wrong-species seeds under $16k |

Shop draws share the end-of-day random stream with weed rolls. Two agents on
the same seed do not see the same shops. A full board rolls fewer weeds and
shifts every later shop. Solo seed numbers are a distribution, not a paired
test.

### care_mill seeds 0–7 (18 animals, lock on day 3)

| Seed | Cash | Herd | End prices (milk / wool / egg) | Why |
| --- | --- | --- | --- | --- |
| 0 | 98,011 | 18 cow | 272 / 242 / 54 | Milk shops. Best run so far |
| 1 | 87,615 | 18 cow | 252 / 229 / 62 | Milk shops |
| 6 | 68,382 | 18 cow | 93 / 250 / 54 | Cows, then milk was oversupplied |
| 3 | 13,032 | 17 goose | 255 / 240 / 40 | Bakery opened first. Eggs sold under base |
| 5 | 10,331 | 18 sheep | 306 / 55 / 63 | No wool shop at the decision. Wool crashed |
| 7 | 11,046 | 18 sheep | 240 / 1 / 59 | Same crash, price floored |
| 2 | 1,615 | 18 sheep | 364 / 1 / 59 | Farmers market has no wool. 18 sheep dumped into town-only drain |
| 4 | 1,573 | 18 sheep | 311 / 1 / 66 | Same |

`shop_opportunist` on seeds 0–3, same solo setup, finished at $23,393,
$25,185, $24,398, and $25,638 with an empty herd. It is stable and far under
the cow runs, and it does not fall to $1,600.

Equal town-only demand was scored with base price, so sheep (base 200) beat
cows (base 160) when no animal shop was open. Wool's scarcity curve is a log
and its glut curve is a steep square, so that herd floors the price.

### Current `care_mill` (shop drain, price floor, herd near the drain)

The built-in random opponent constructs `random.Random()` with no seed, and
both farms share the weed stream that also draws the next shop. A configured
episode seed does not replay. These are independent samples.

Eight samples: mean $72,023, worst $53,234, best $96,734. One earlier sample
on the same code reached $105,972. Herds survived. Prices no longer finish
at $1.

| Sample | Cash | Herd | End milk / wool / egg |
| --- | --- | --- | --- |
| A | 96,734 | 18 cow | 312 / 247 / 54 |
| B | 74,082 | 14 cow | 267 / 229 / 66 |
| C | 64,024 | 11 sheep | 301 / 238 / 68 |
| D | 53,234 | 14 cow | 250 / 229 / 130 |
| E | 65,493 | 11 sheep | 315 / 238 / 56 |
| F | 63,509 | 14 cow | 243 / 249 / 66 |
| G | 76,332 | 18 sheep | 275 / 251 / 60 |
| H | 82,778 | 18 cow | 284 / 248 / 56 |

Head to head, earlier build of the same herd (both seats, 720 turns):

| Seed | care_mill seat | care_mill | shop_opportunist |
| --- | --- | --- | --- |
| 0 | first | 63,661 | 22,675 |
| 0 | second | 48,994 | 19,722 |
| 1 | first | 61,111 | 22,688 |
| 1 | second | 57,848 | 22,875 |
| 2 | first | 61,900 | 24,269 |
| 2 | second | 60,934 | 24,095 |
| 6 | first | 35,130 | 23,790 |
| 6 | second | 54,572 | 24,904 |

`care_mill` won all eight seats. `shop_opportunist` stays near $20–25k
because it never builds a cared herd. Those seats are also single noisy
episodes, not replays. `market_velocity`, loaded the same way, returned
PASS and finished on $3,000, so it was not a real opponent in this harness.

## Distance to 138k

The best episode observed is $105,972. The best of the eight-sample batch is
$96,734, with 18 cows and milk still at $312, so that book was not exhausted. Eighteen cows is the opening quadrant. The cows bought
after day 10 only get about five production cycles (first yield is 8 days
after placement). That back half produces ~18 milk each. The front half,
placed around day 5 and cared the whole wait, produces ~30. Blended, that is
the mid-$90ks when the average sale is near $230 and the last sales are near
$300.

Another $40k on that seed needs those same cows earlier, which starting cash
does not allow: 18 cows are $7,200 before wheat. Fertilizer sales fund the
second wave, and that wave is structurally late. A second quadrant was the
attempt to add a third wave. It lowered the eight-seed mean from about $61k
to about $57k, so it is not in the agent.

What is still unused on some seeds: a second scarce good. Seed 2 held 11
sheep for a yarn store while milk finished at $350. A side herd would have to
be fed and cared with the same nine workers. That split is what killed the
earlier mixed herds, so it is not in this policy.
