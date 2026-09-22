# Sell timing for every product

Milk was fit in `research/12_Quant_and_RL.md`. This note asks the same
question for the other eight goods, then only codes the rules that beat a
cared-cow herd. Labor stays the `care_mill` script unless a line below
says otherwise.

## Questions, answered before the agents

1. For each good, what does fifty extra units of deficit or glut do to
   the quote? Goods with a log scarcity curve and a square glut (wool,
   melon) cannot be stockpiled and dumped.
2. Fertilizer has no town sink. Does holding it ever raise the price?
3. On eight tiles, which enterprise beats a cared cow once the sell path
   is a grid, not a guess?
4. Only the winner of that table, plus the fertilizer rule, is shipped.

## Method

`scripts/quant_catalog.py` calls the engine `market_price`. Production is
a cared schedule (cows 6 then 3 every two days, sheep 6 then 4 every
three, geese 2 a day, one-shot crops on their harvest day). The shed
holds 100. Drain and sales are split across six ticks so the quote can
recover inside a day. A grid searches the first sell day and the daily
cap. Nothing is sold under the base price, except fertilizer, which has
no scarcity to protect. Differential evolution then fits fertilizer's
floor and cap.

## Price impact at ±50 inventory

| item | base | scarcity | T | Δ at +50 deficit | Δ at −50 glut |
|------|------|----------|---|------------------|---------------|
| WHEAT | 25 | sqrt | 400 | +7 | −3 |
| CARROT | 35 | hinge | 450 | +4 | −8 |
| TOMATO | 60 | hinge | 200 | +6 | −18 |
| STRAWBERRY | 120 | sqrt | 100 | +59 | −96 |
| MELON | 250 | log | 300 | +34 | −25 |
| EGG | 50 | hinge | 332 | +3 | −7 |
| MILK | 160 | sqrt | 122 | +61 | −105 |
| WOOL | 200 | log | 105 | +34 | −145 |
| FERTILIZER | 100 | linear | 200 | +10 | −10 |

Wool's glut is a square. Fifty extra units take $145 off a $200 quote.
Eggs barely move until the hinge at T=332, which an 18-goose herd cannot
cross. Strawberries look like milk on the scarcity side and like a crash
on the glut side. Fertilizer is a straight line with no town drain.

## Eight-tile enterprises, net of animal or seed cost

Best grid path. Drain is shop milk/wool/eggs a day. Crops use the drains
in the table.

| good | drain | start day | daily cap | cash | net after cost |
|------|-------|-----------|-----------|------|----------------|
| MILK | 24 | 16 | 16 | 76,535 | **73,335** |
| MILK | 12 | 16 | 16 | 60,669 | 57,469 |
| WOOL | 24 | 16 | 16 | 56,481 | 52,481 |
| WOOL | 6 | 16 | 16 | 41,155 | 37,155 |
| EGG | 24 | 24 | 40 | 28,086 | 25,686 |
| EGG | 1 | 27 | 16 | 1,573 | **−827** |
| MELON | 1 | 27 | 16 | 9,880 | 9,240 |
| STRAWBERRY | 18 | 27 | 16 | 9,767 | 8,967 |
| TOMATO | 6 | 27 | 16 | 2,501 | 2,101 |
| CARROT | 24 | 27 | 8 | 3,679 | 3,519 |
| WHEAT | 12 | 27 | 16 | 1,376 | 1,296 |

A cared cow herd is the only eight-tile line in the same band as the
$138k games. Wool is second and still $20k behind on a deep drain, and
its glut is lethal. Geese lose money unless a bakery hinge actually
fires. Melon, strawberry, tomato, carrot, and wheat-as-cash are funding
tricks or leftovers, not product lines. Wheat stays a feed crop.

On a thin drain (1/day) every animal wants to hold until day 27. That is
the monopoly-hold we already use for milk.

## Fertilizer has no sink

Twelve units a day from day 8, drain 0:

| floor | start | cap | cash | units sold |
|-------|-------|-----|------|------------|
| $1 | 0 | 16 | 19,457 | 264 |
| $40 | 0 | 16 | 19,457 | 264 |
| $60 | 0 | 16 | 16,200 | 203 |
| $80 | 0 | 8 | 9,250 | 103 |
| $100 | 0 | 1 | 300 | 3 |

Scipy's differential evolution returned floor ≈ $18, cap ≈ 14, cash
$19,457 — the same as selling to $1. Holding for $60 or $80 leaves units
unsold. `care_mill` stopped collecting once the herd was paid, refused
sales under $60, and finished games with 47–87 fertilizer in the shed.

Using fertilizer on wheat is worse than selling it. One extra wheat is
about $25. One fertilizer sale is $40–$100.

## Agents

`fert_mill` is `care_mill` with three changes: collect fertilizer every
day, sell it first in the market queue, stop only under $20.

`catalog_mill` adds the milk inventory floor from `rl_sell_mill` and the
catalog rules for wool (hold through day 27), eggs (hold under $80), and
the side crops.

`rl_fert_mill` is `fert_mill` with a once-a-day fertilizer floor in
{$1, $20, $40, $60}. The prior is $20, or $1 once the quote is already
under $20. `scripts/train_fert_q.py` trains it.

## Engine results

Vs `pass` on fixed seeds. Shop lists can differ when herd size changes
the empty-tile weed stream.

| agent | seed 3 | seed 5 | seed 7 |
|-------|--------|--------|--------|
| care_mill | 67,163 (14 cows, 47 fert left) | 43,207 (18 cows, 18 fert) | 54,719 (10 cows, 53 fert) |
| fert_mill | **78,075** (18 cows, 0 fert, milk $278) | 50,049 (18 cows, 0 fert) | 60,301 (10 cows, 0 fert) |
| catalog_mill | 75,210 (18 cows, 11 milk left) | 49,910 | 60,423 |

Head-to-head, both seats, same shops:

| match | result |
|-------|--------|
| fert vs care, seed 3 | fert 55,237 / 55,236 vs care 45,444 / 45,447 |
| fert vs care, seed 5 | fert 46,799 / 49,796 vs care 21,048 / 17,006. Care left 79–82 fertilizer |
| fert vs care, seed 7 yarn | fert 38,317 vs care 27,470. Eleven sheep, wool held (quote $198) |
| catalog vs care | same direction, slightly less than fert on milk seats, slightly more on the yarn seat |
| rl_fert vs fert, seed 3 | **75,064 / 75,070 vs 58,610 / 58,604**. Learned floor $1 emptied the shed; fert_mill stopped at $20 and left 70 units |
| rl_fert vs fert, seed 5 | rl ahead on both seats, smaller gap |
| rl_fert vs care, seed 2 (holdout) | 61,835 vs 55,852 vs pass; yarn H2H 35–38k vs care 26–28k |

Training (ten paired episodes, ε=0.4) had mean explore gap +$100. Solo
gaps were zero: the $20 greedy path was already the catalog path. The
table that moved cash was late-game, quote $40–$60: prefer floor $1 over
$20. Frozen into `submissions/rl_fert_mill/q_values.json`.

## Reading

The $138k gap was not a better melon or goose line. It was unused
fertilizer and the cows that fertilizer can buy. On seed 3, selling it
funded four extra cows and left milk at $278. That is the first time an
agent in this repo grew to 18 cows on that seed without walking milk
back to base.

`catalog_mill`'s milk hold left 11 milk unsold at $281 and cost about
$3k versus `fert_mill` on the same monopoly. Keep care's milk dump for
quiet towns. Keep the learned fertilizer floor for a shared book.

Do not start geese, strawberry, melon, tomato, or carrot as a second
product. Wool only when there is a yarn store and no milk shop, and even
then hold until the quote is at least $200.

`rl_fert_mill` is the agent to beat. `care_mill` is the labor baseline,
not the cash baseline any more.

The remaining gap to $138k is still the second wave's eight-day milk lag.
Fertilizer only bought that wave a few days earlier. Pasture prebuild on
days 0–2, without cutting the eight wheat plots, is the next experiment
that still has a number attached to it.

## Early pastures on top of fertilizer

`prefert_mill` is `rl_fert_mill` with ten pastures on days 0–2, wheat
planted first, and a build budget so nine hands do not overshoot. Empty
pastures are not `None`, so they change the weed stream and the shops.

Vs `pass` the towns are not the same, and that lottery dominated: seed 5
drew three yarn stores and finished at $66,459 on 17 sheep; seed 7 drew
yarn early and finished at $32,693 while `rl_fert_mill` on its own shop
stream made $60,301 on cows.

Head-to-head the shops match. Prefert was ahead on every seat in this
batch (seed 3 about +$2.8k, seed 5 one seat even and one +$5k, seed 7
about +$5k). The extra stalls help when both farms buy the same day.

Do not replace `rl_fert_mill` with this for a quiet opponent. Use it when
the other farm is also milking.

