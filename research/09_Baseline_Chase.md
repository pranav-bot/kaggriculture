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

Best solo cash in this chase: about 40k (`demand_mill`) and 34k
(`shop_supply`). Both are the same band as `market_velocity` /
`shop_opportunist`. Neither is a 138k herd.

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

## `care_mill` (in progress)

Stage policy in `submissions/care_mill/main.py`:

1. Day 0–2: hire up to 8, plant 8 wheat and water them every day, buy a
   small wheat stock. No animals, no land.
2. From day 3: lock one species. Score is shop drain per day times base
   price times cared output per day (cow 1.5, sheep 1.33, goose 2).
3. Buy one animal per turn only when wheat on hand covers four days for the
   enlarged herd. Cap 4 animals until the wheat crop is harvestable (day 5),
   then 8, then 12.
4. Workers feed before they do anything else. Care before harvest. Harvest
   any animal yield immediately. Collect and sell fertilizer only while the
   herd is still short or cash is under $1,500 and the fertilizer price is
   at least $60.
5. Sell milk, wool, and eggs while price is at least 85% of base, a few
   units per turn, faster when the shed is holding a burst.

Results are appended below as episodes finish.

## Iteration log

| Agent | Seed | Opponent | Cash | Herd at day 29 | Note |
| --- | --- | --- | --- | --- | --- |
| shop_supply (mixed, wheat gate relaxed) | default | random | 34,108 | 5 cow, 4 goose, 4 sheep | Alive, undersupplied milk |
| demand_mill (best patch) | default | random | 40,327 | 8 sheep, 2 cow, 6 goose | Alive, wool price still high |
| care_mill | — | — | — | — | Not run yet |
