# 06 — Recreated Agent 3 (Architecture Gamma)

**Family:** Conserved-route replay with prefix-compatible feature switches and capacity-aware market repair  
**Scope:** Exact recreation of online repair logic. Route blobs stay behind `load_routes()`.

---

## Decision Flow

```
routes ← decode(main + prefix-compatible tails)
for each decision checkpoint (turn, feature, threshold, target):
  if step == turn AND feature(obs) >= threshold AND prefix_identical(cur, target, turn):
    cur ← target
base ← routes[cur][step]
repairs:
  1. noop-on-WEED → DIG
  2. project same-turn DROP/PLACE into shed
  3. hour==23 room_guard → target capacity-1 (99)
  4. clamp SELLs to fillable projected qty
  5. dead_stock SELLs = projected − future_route_sells
return {farmer, hands, market[:10]}
```

---

## Exact Decision Table

| Turn | Feature | Threshold | Intent |
|------|---------|-----------|--------|
| 226 | `count(YARN_STORE)` | ≥ 1 | Yarn-aligned livestock continuation |
| 360 | `market.prices["CARROT"]` | ≥ 42 | Carrot-premium tail |
| 433 | `market.inventory["MILK"]` | ≥ 10067 | Milk-glut alternate dump plan |

A switch is legal **only** if the candidate tail equals the current route on every step `< turn` (prefix compatibility).

---

## Python Template

```python
"""Conserved route replay with value-aware one-slot capacity reserve."""
from __future__ import annotations

from typing import Any, Callable

TURNS = 720
BOARD = 10
MAX_ORDERS = 10
SHED_CAP = 100
PRODUCTS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)
ANIMALS = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
PASS = {"farmer": ["PASS"], "hands": [], "market": []}

# Opaque route IDs — replace with your corpus hashes / names.
MAIN = "route_main"
YARN = "route_yarn"
YARN_CARROT = "route_yarn_carrot"
MILK_GLUT = "route_milk_glut"

DECISIONS = (
    (226, "shop_YARN_STORE", 1, YARN),
    (360, "px_CARROT", 42, YARN_CARROT),
    (433, "inv_MILK", 10067, MILK_GLUT),
)


def _shed_adjacent(x: int, y: int, board: int = BOARD) -> bool:
    h = board // 2
    return (x, y) in ((h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h))


def _feature(obs: dict, name: str) -> int:
    if name == "shop_YARN_STORE":
        return (obs.get("town", {}).get("unlocked_shops") or []).count("YARN_STORE")
    if name == "px_CARROT":
        return int(obs["market"]["prices"].get("CARROT", 0))
    if name == "inv_MILK":
        return int(obs["market"]["inventory"].get("MILK", 0))
    return 0


def _noop(act, tile, inv, seeds, x, y, board=BOARD) -> bool:
    """True when the engine will ignore this action (free turn for DIG reuse)."""
    if not act:
        return True
    op = act[0]
    if op in MOVES:
        dx, dy = MOVES[op]
        return not (0 <= x + dx < board and 0 <= y + dy < board)
    if op == "PASS":
        return True
    if op == "DROP":
        return (not _shed_adjacent(x, y, board)) or (not inv)
    if op == "PICKUP":
        return not _shed_adjacent(x, y, board)
    if op == "PLACE":
        item = act[1] if len(act) > 1 else None
        if item in ANIMALS and isinstance(tile, dict) and tile.get("kind") == ANIMALS[item] and tile.get("animal") is None:
            return inv.get(item, 0) <= 0
        if _shed_adjacent(x, y, board):
            return inv.get(item, 0) <= 0
        return True
    if tile == "LOCKED":
        return True
    isd = isinstance(tile, dict)
    kind = tile.get("kind") if isd else None
    animal = isd and tile.get("animal") is not None
    if op == "PLANT":
        return tile is not None or seeds.get(act[1] if len(act) > 1 else None, 0) <= 0
    if op == "WATER":
        return kind != "PLANT" or bool(tile.get("watered_today"))
    if op == "HARVEST":
        return (not isd) or tile.get("yield_units", 0) <= 0
    if op == "FERTILIZE":
        return kind != "PLANT" or inv.get("FERTILIZER", 0) <= 0
    if op == "DIG":
        return tile is None or animal
    if op in ("BUILD_COOP", "BUILD_PASTURE"):
        return tile is not None
    if op == "FEED":
        return (not animal) or bool(tile.get("fed_today")) or inv.get("WHEAT", 0) <= 0
    if op == "COLLECT_FERTILIZER":
        return (not animal) or (not tile.get("fertilizer_available"))
    if op == "CARE":
        return (not animal) or bool(tile.get("cared_today"))
    return True


class ConservedRouteAgent:
    def __init__(self, routes: dict[str, list[dict]]):
        self.R = routes
        self.cur = MAIN
        self._fs = None
        self._fs_for = None

    def future_sells(self, item: str, step: int) -> int:
        if self._fs_for != self.cur:
            r = self.R[self.cur]
            fs = {p: [0] * (len(r) + 1) for p in PRODUCTS}
            for t in range(len(r) - 1, -1, -1):
                add: dict[str, int] = {}
                for o in (r[t].get("market") or []):
                    if o and o[0] == "SELL" and o[1] in fs:
                        add[o[1]] = add.get(o[1], 0) + int(o[2])
                for p in fs:
                    fs[p][t] = fs[p][t + 1] + add.get(p, 0)
            self._fs = fs
            self._fs_for = self.cur
        a = self._fs.get(item)
        return a[step] if a and step < len(a) else 0

    def _switch_ok(self, target: str, turn: int) -> bool:
        a, b = self.R[self.cur], self.R[target]
        if a is b:
            return False
        for t in range(turn):
            if a[t] != b[t]:
                return False
        return True

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        s = obs.get("step")
        step = int(s) if s is not None else int(obs.get("day", 0)) * 24 + int(obs.get("hour", 0))
        me = int(obs.get("player", 0))
        farm = obs["farms"][me]
        priv = obs["private"]
        tiles = farm["tiles"]
        seeds = priv.get("seeds") or {}
        invs = priv.get("inventories") or []
        shed = dict(priv.get("shed") or {})
        prices = obs["market"]["prices"]
        day = int(obs.get("day", step // 24))
        board = len(tiles) or BOARD

        for turn, feat, thr, target in DECISIONS:
            if turn == step and target != self.cur and self._switch_ok(target, turn):
                if _feature(obs, feat) >= thr:
                    self.cur = target

        route = self.R[self.cur]
        base = route[step] if step < len(route) else PASS
        acts = [list(base.get("farmer") or ["PASS"])] + [list(h) for h in (base.get("hands") or [])]
        market = [list(o) for o in (base.get("market") or [])]
        positions = [tuple(farm["farmer"])] + [tuple(p) for p in farm["hands"]]

        # --- weed_dig: wasted turn on weed becomes DIG ---
        for i in range(min(len(acts), len(positions))):
            x, y = positions[i]
            if not (0 <= x < board and 0 <= y < board):
                continue
            tile = tiles[y][x]
            inv = invs[i] if i < len(invs) else {}
            if isinstance(tile, dict) and tile.get("kind") == "WEED" and _noop(acts[i], tile, inv, seeds, x, y, board):
                acts[i] = ["DIG"]

        # --- projected shed after same-turn DROP/PLACE ---
        proj = dict(shed)
        room = SHED_CAP - sum(proj.values())
        for i in range(min(len(acts), len(positions))):
            if room <= 0:
                break
            x, y = positions[i]
            inv = invs[i] if i < len(invs) else {}
            if not inv or not _shed_adjacent(x, y, board):
                continue
            a = acts[i]
            if a and a[0] == "DROP":
                for it, n in inv.items():
                    take = min(n, room)
                    if take > 0:
                        proj[it] = proj.get(it, 0) + take
                        room -= take
            elif a and a[0] == "PLACE" and len(a) > 1 and a[1] not in ANIMALS:
                it = a[1]
                take = min(int(a[2]) if len(a) > 2 else 1, inv.get(it, 0), room)
                if take > 0:
                    proj[it] = proj.get(it, 0) + take
                    room -= take

        # --- room_guard at day close ---
        if step % 24 == 23:
            carried = sum(max(0, int(n)) for inv in invs for n in inv.values())
            produced = consumed = 0
            for i in range(min(len(acts), len(positions))):
                x, y = positions[i]
                if not (0 <= x < board and 0 <= y < board):
                    continue
                tile = tiles[y][x]
                a = acts[i]
                if not a:
                    continue
                op = a[0]
                if op == "HARVEST" and isinstance(tile, dict):
                    produced += max(0, int(tile.get("yield_units", 0)))
                elif op == "COLLECT_FERTILIZER" and isinstance(tile, dict) and tile.get("fertilizer_available"):
                    produced += 1
                elif op in ("FEED", "FERTILIZE"):
                    consumed += 1
                elif op == "PLACE" and len(a) > 1 and a[1] in ANIMALS:
                    consumed += 1
            planned_sells: dict[str, int] = {}
            planned_buys = 0
            for o in market:
                if not o:
                    continue
                if o[0] == "SELL":
                    planned_sells[o[1]] = planned_sells.get(o[1], 0) + max(0, int(o[2]))
                elif o[0] in ("BUY_PRODUCT", "BUY_ANIMAL"):
                    planned_buys += max(0, int(o[2]))
            shed_total = sum(max(0, int(n)) for n in shed.values())
            actual_existing_sells = sum(
                min(max(0, int(shed.get(it, 0))), n) for it, n in planned_sells.items()
            )
            needed = shed_total + carried + produced - consumed + planned_buys - actual_existing_sells - (SHED_CAP - 1)
            if needed > 0:
                priority = sorted(
                    PRODUCTS,
                    key=lambda it: (self.future_sells(it, step + 1) > 0, -prices.get(it, 0), it),
                )
                for it in priority:
                    already = planned_sells.get(it, 0)
                    available = max(0, int(shed.get(it, 0)) - already)
                    qty = min(needed, available)
                    if qty <= 0:
                        continue
                    slot = next((j for j, o in enumerate(market) if o and o[0] == "SELL" and o[1] == it), -1)
                    if slot >= 0:
                        market[slot][2] = max(0, int(market[slot][2])) + qty
                    elif len(market) < MAX_ORDERS:
                        market.append(["SELL", it, qty])
                    else:
                        continue
                    planned_sells[it] = already + qty
                    needed -= qty
                    if needed <= 0:
                        break

        # --- clamp_sells: never burn a slot on unfillable SELL ---
        avail = dict(proj)
        kept = []
        for o in market:
            if o and o[0] == "SELL":
                have = avail.get(o[1], 0)
                if have <= 0:
                    continue
                n = min(int(o[2]), have)
                if n <= 0:
                    continue
                avail[o[1]] = have - n
                kept.append(["SELL", o[1], n])
            else:
                kept.append(o)
        market = kept

        # --- dead_stock: sell what the rest of the route never sells ---
        planned: dict[str, int] = {}
        for o in market:
            if o and o[0] == "SELL":
                planned[o[1]] = planned.get(o[1], 0) + int(o[2])
        extra = []
        for it in PRODUCTS:
            have = proj.get(it, 0) - planned.get(it, 0)
            if have <= 0:
                continue
            surplus = have if day >= 29 else have - self.future_sells(it, step + 1)
            if surplus > 0 and prices.get(it, 0) > 1:
                extra.append(["SELL", it, surplus])
        extra.sort(key=lambda o: -prices.get(o[1], 0) * int(o[2]))

        return {
            "farmer": acts[0],
            "hands": acts[1:],
            "market": (market + extra)[:MAX_ORDERS],
        }


_AGENT = None


def agent(obs: dict[str, Any]) -> dict[str, Any]:
    global _AGENT
    try:
        s = obs.get("step")
        step = int(s) if s is not None else int(obs.get("day", 0)) * 24 + int(obs.get("hour", 0))
        if _AGENT is None or step == 0:
            _AGENT = ConservedRouteAgent(load_routes())  # provide your corpus
        return _AGENT.act(obs)
    except Exception:
        try:
            hands = obs["farms"][int(obs.get("player", 0))].get("hands") or []
        except Exception:
            hands = []
        return {"farmer": ["PASS"], "hands": [["PASS"] for _ in hands], "market": []}
```

---

## Standalone Portables (No Corpus Required)

Implement these three functions against *your* planned market for the rest of the day:

1. **`clamp_sells(projected_shed, market)`** — drop/zero unfillable SELLs  
2. **`room_guard_99(...)`** — day-close capacity defense preferring items with no future planned sale  
3. **`dead_stock_sell(...)`** — sell surplus with zero remaining planned demand  

These three alone typically recover silent score loss from slot waste and shed overflow.
