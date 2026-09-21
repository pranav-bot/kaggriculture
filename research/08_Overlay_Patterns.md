# 08 — Overlay Patterns

Thin wrappers that modify only market timing or idle labor. Exact recreation of recoverable overlay logic from anonymous elite wrappers.

---

## Overlay A — Premium Sale Phase Shift (+1 Turn)

**Intent:** Delay premium SELLs by one turn when liquid and not capacity-pressured, releasing on the base’s next natural sell or into a free slot.

```python
from collections import defaultdict

PREMIUM = {"MILK", "WOOL", "STRAWBERRY", "FERTILIZER"}
_pending = defaultdict(int)
_age = defaultdict(int)


def premium_phase_shift(obs, base_action):
    global _pending, _age
    act = dict(base_action)
    day = int(obs.get("day", 0) or 0)
    player = int(obs.get("player", 0) or 0)
    farm = (obs.get("farms") or [None, None])[player] or {}
    money = float(farm.get("money", 0) or 0)
    shed = (obs.get("private") or {}).get("shed") or {}
    shed_total = sum(int(v or 0) for v in shed.values() if isinstance(v, (int, float)))

    market = list(act.get("market", []) or [])
    out, released = [], set()

    cash_critical = money < 1600 or any(
        isinstance(o, list) and o and o[0] in {"BUY_ANIMAL", "BUY_LAND", "BUY_PRODUCT"}
        for o in market
    )
    pressure = shed_total >= 82
    endgame = day >= 27
    safe_to_delay = not cash_critical and not pressure and not endgame

    for item in list(_pending):
        if _pending[item] > 0:
            _age[item] += 1

    for order in market:
        is_sell = (
            isinstance(order, list) and len(order) >= 3
            and order[0] == "SELL" and str(order[1]) in PREMIUM
        )
        if not is_sell:
            out.append(order)
            continue
        item = str(order[1])
        if _pending[item] > 0:
            out.append(order)
            _pending[item] = _age[item] = 0
            released.add(item)
            continue
        if safe_to_delay:
            _pending[item] = min(100, max(0, int(order[2])))
            _age[item] = 0
            continue
        out.append(order)

    if len(out) < 10:
        for item in PREMIUM:
            if len(out) >= 10:
                break
            if item in released or _pending[item] <= 0:
                continue
            if _age[item] < 1 and not endgame:
                continue
            qty = min(int(shed.get(item, 0) or 0), int(_pending[item]))
            if qty > 0:
                out.append(["SELL", item, qty])
            _pending[item] = _age[item] = 0

    act["market"] = out[:10]
    return act
```

**Exact thresholds:** cash &lt; 1600 critical; shed ≥ 82 pressure; day ≥ 27 endgame; never evict a base non-pending order.

---

## Overlay B — Market Collision Guard

**Intent:** If shared inventory just rose or price just fell, another seller is saturating the book — hold premium sells briefly.

```python
from collections import defaultdict

PREMIUM = {"MILK", "WOOL", "STRAWBERRY", "FERTILIZER"}
BASE_PRICE = {"MILK": 160, "WOOL": 200, "STRAWBERRY": 120, "FERTILIZER": 100}
_prev_inv = _prev_price = None
_pending = defaultdict(int)
_age = defaultdict(int)


def collision_guard(obs, base_action):
    global _prev_inv, _prev_price, _pending, _age
    act = dict(base_action)
    day = int(obs.get("day", 0) or 0)
    player = int(obs.get("player", 0) or 0)
    farm = (obs.get("farms") or [None, None])[player] or {}
    money = float(farm.get("money", 0) or 0)
    shed = (obs.get("private") or {}).get("shed") or {}
    shed_total = sum(int(v or 0) for v in shed.values() if isinstance(v, (int, float)))
    inv = dict((obs.get("market") or {}).get("inventory") or {})
    price = dict((obs.get("market") or {}).get("prices") or {})

    delta = {x: 0.0 for x in PREMIUM}
    pdelta = {x: 0.0 for x in PREMIUM}
    if _prev_inv is not None:
        for item in PREMIUM:
            delta[item] = float(inv.get(item, 0) or 0) - float(_prev_inv.get(item, 0) or 0)
            pdelta[item] = float(price.get(item, 0) or 0) - float(_prev_price.get(item, 0) or 0)

    market = list(act.get("market", []) or [])
    out, released = [], set()
    for item in list(_pending):
        if _pending[item] > 0:
            _age[item] += 1

    endgame = day >= 27
    pressure = shed_total >= 84
    cash_critical = money < 1500 or any(
        isinstance(o, list) and o and o[0] in {"BUY_ANIMAL", "BUY_LAND", "BUY_PRODUCT"}
        for o in market
    )

    for order in market:
        is_sell = (
            isinstance(order, list) and len(order) >= 3
            and order[0] == "SELL" and str(order[1]) in PREMIUM
        )
        if not is_sell:
            out.append(order)
            continue
        item = str(order[1])
        if _pending[item] > 0:
            out.append(order)
            _pending[item] = _age[item] = 0
            released.add(item)
            continue
        ratio = float(price.get(item, BASE_PRICE[item]) or BASE_PRICE[item]) / BASE_PRICE[item]
        saturating = delta[item] >= 3 or pdelta[item] <= -3
        if saturating and ratio < 1.10 and not endgame and not pressure and not cash_critical:
            _pending[item] = min(100, max(0, int(order[2])))
            _age[item] = 0
            continue
        out.append(order)

    # Release only into draining/recovering book with a free slot
    if len(out) < 10:
        for item in PREMIUM:
            if len(out) >= 10 or item in released or _pending[item] <= 0:
                continue
            recovering = delta[item] <= -1 or pdelta[item] >= 1
            if not recovering and _age[item] < 2 and not endgame:
                continue
            qty = min(int(shed.get(item, 0) or 0), int(_pending[item]))
            if qty > 0:
                out.append(["SELL", item, qty])
            _pending[item] = _age[item] = 0

    _prev_inv, _prev_price = inv, price
    act["market"] = out[:10]
    return act
```

**Exact thresholds:** Δinv ≥ 3 or Δprice ≤ −3; price/base &lt; 1.10; cash &lt; 1500; shed ≥ 84; day ≥ 27.

---

## Overlay C — Clone-Like Detector Gate

Only enable collision logic when the opponent’s public farm “looks like” yours (common when both seats run similar openings).

```python
def public_counts(farm):
    animals = {"COW": 0, "SHEEP": 0, "GOOSE": 0}
    crops = {"WHEAT": 0, "CARROT": 0, "TOMATO": 0, "STRAWBERRY": 0, "MELON": 0}
    for row in farm.get("tiles", []) or []:
        for t in row or []:
            if not isinstance(t, dict):
                continue
            if t.get("animal") in animals:
                animals[t["animal"]] += 1
            if t.get("kind") == "PLANT" and t.get("crop") in crops:
                crops[t["crop"]] += 1
    return animals, crops


def clone_like(obs) -> bool:
    farms = obs.get("farms") or []
    player = int(obs.get("player", 0) or 0)
    if len(farms) != 2:
        return False
    me, opp = farms[player], farms[1 - player]
    ma, mc = public_counts(me)
    oa, oc = public_counts(opp)
    hand_gap = abs(len(me.get("hands", []) or []) - len(opp.get("hands", []) or []))
    land_gap = abs(len(me.get("unlocked_quadrants", []) or []) - len(opp.get("unlocked_quadrants", []) or []))
    animal_gap = sum(abs(ma[k] - oa[k]) for k in ma)
    crop_gap = sum(abs(mc[k] - oc[k]) for k in mc)
    return hand_gap <= 1 and land_gap == 0 and animal_gap <= 2 and crop_gap <= 5
```

Variant thresholds seen: shed pressure 88; cash critical 1300 when gated by clone_like.

---

## Overlay D — Idle-Hand Water Rescue

**Intent:** Replace qualifying hand `PASS` windows with a short round-trip WATER, then restore the hand to its exact starting tile.

**Activation gates (all required):**

| Gate | Value |
|------|-------|
| Step range | `[6*24, 18*24)` i.e. days 6..17 |
| Hour | 16..21 |
| Actor | hired hand (not farmer) |
| Base action | PASS |
| Native schedule | PASS for entire round-trip window |
| Carry | empty inventory |
| Target | PLANT, not watered today, `consecutive_unwatered >= 1` |
| Route length | ≤ 7 commands |
| Cap | ≤ 2 rescues / day |
| Return | exact starting tile |

```python
_MOVES = {"EAST": (1, 0), "WEST": (-1, 0), "NORTH": (0, -1), "SOUTH": (0, 1)}
_OPPOSITE = {"EAST": "WEST", "WEST": "EAST", "NORTH": "SOUTH", "SOUTH": "NORTH"}


def path_to(src, dst):
    x, y = src
    tx, ty = dst
    cmds = []
    while x != tx:
        cmds.append(["EAST" if tx > x else "WEST"])
        x += 1 if tx > x else -1
    while y != ty:
        cmds.append(["SOUTH" if ty > y else "NORTH"])
        y += 1 if ty > y else -1
    return cmds


def build_water_rescue(start, target_xy, max_commands=7):
    go = path_to(start, target_xy)
    work = go + [["WATER"]]
    back = [_OPPOSITE[c[0]] for c in reversed(go)]
    back = [[d] for d in back]
    route = work + back
    if len(route) > max_commands:
        return None
    return route  # execute one command per turn; restore exact start
```

Full stateful executor tracks `task` cursor per hand and aborts if base stops PASSING mid-route.

---

## Composition Order (Recommended)

```
base_action = primary_policy(obs)
base_action = idle_water_rescue(obs, base_action)   # labor only
base_action = room_guard_99(obs, base_action)       # capacity
base_action = clamp_sells(obs, base_action)         # slot economy
base_action = impact_rank_sells(obs, base_action)   # microstructure
base_action = collision_guard(obs, base_action)     # adversarial timing
# OR premium_phase_shift — do not stack both delay overlays blindly
```

Never stack premium-delay and collision-hold without a shared `_pending` ledger — both mutate the same sell queue.
