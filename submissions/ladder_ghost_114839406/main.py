"""Ghost of ladder episode 114839406 (Victor @ Tufa Labs) -- DO NOT EDIT BY HAND.

Deterministic replay of the extracted macro-sequence below. One-shot market
orders fire once via a blind ledger; continuous directives (PLANT/DUMP) run
while active. Field work uses our Kuhn-Munkres routing layer (scipy if
present, greedy fallback -- same as scratch_grandmaster).
Schedule: D0:BUY_ANIMAL COWx2; D0:BUY_ANIMAL SHEEPx3; D0:PLANT WHEATx8; D0:HIRE x5; D6:EXPAND_NE; D6:HIRE x8; D6:PLANT STRAWBERRYx15; D8:EXPAND_SW; D25:DUMP MILK; D25:DUMP WOOL; D25:DUMP FERTILIZER; D27:DUMP STRAWBERRY
"""
import os
import sys

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

SCHEDULE = [{'day': 0, 'action': 'BUY_ANIMAL', 'animal': 'COW', 'count': 2}, {'day': 0, 'action': 'BUY_ANIMAL', 'animal': 'SHEEP', 'count': 3}, {'day': 0, 'action': 'PLANT', 'crop': 'WHEAT', 'count': 8}, {'day': 0, 'action': 'HIRE', 'count': 5}, {'day': 6, 'action': 'EXPAND_NE'}, {'day': 6, 'action': 'HIRE', 'count': 8}, {'day': 6, 'action': 'PLANT', 'crop': 'STRAWBERRY', 'count': 15}, {'day': 8, 'action': 'EXPAND_SW'}, {'day': 25, 'action': 'DUMP', 'product': 'MILK'}, {'day': 25, 'action': 'DUMP', 'product': 'WOOL'}, {'day': 25, 'action': 'DUMP', 'product': 'FERTILIZER'}, {'day': 27, 'action': 'DUMP', 'product': 'STRAWBERRY'}]
MAX_ORDERS = 10

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except ImportError:
    _lsa = None

_fired = set()  # blind one-shot ledger: (day, index)


def kuhn_munkres_route(unit_positions, targets):
    """Our routing layer: min-Manhattan assignment (vendored)."""
    if not unit_positions or not targets:
        return []
    costs = [[abs(int(u[0]) - int(t[0])) + abs(int(u[1]) - int(t[1]))
              for t in targets] for u in unit_positions]
    if _lsa is None:  # greedy fallback
        pairs, used = [], set()
        for row in range(len(costs)):
            best, bj = None, -1
            for col in range(len(targets)):
                if col not in used and (best is None or costs[row][col] < best):
                    best, bj = costs[row][col], col
            if bj >= 0:
                used.add(bj)
                pairs.append((row, bj))
        return pairs
    rows, cols = _lsa(costs)
    return list(zip(rows.tolist(), cols.tolist()))


def _step(src, dst):
    dx, dy = int(dst[0]) - int(src[0]), int(dst[1]) - int(src[1])
    if abs(dx) >= abs(dy) and dx:
        return ["EAST" if dx > 0 else "WEST"]
    if dy:
        return ["SOUTH" if dy > 0 else "NORTH"]
    if dx:
        return ["EAST" if dx > 0 else "WEST"]
    return ["PASS"]


def _tile(farm, r, c):
    try:
        return farm["tiles"][r][c]
    except (IndexError, TypeError, KeyError):
        return "LOCKED"


def agent(obs):
    """Kaggle entrypoint: blind macro-sequence execution."""
    try:
        return _act(obs)
    except Exception:
        n_hands = len(((obs.get("farms") or [{}])[int(obs.get("player", 0))]).get("hands", []))
        return {"farmer": ["PASS"], "hands": [["PASS"]] * n_hands, "market": []}


def _act(obs):
    if int(obs.get("step", 0) or 0) == 0:
        _fired.clear()  # new episode: blind ledger resets
    seat = int(obs.get("player", 0))
    farm = (obs.get("farms") or [{}])[seat] if seat < len(obs.get("farms") or []) else {}
    day = int(obs.get("day", 0) or 0)
    money = float(farm.get("money", 0.0) or 0.0)
    private = obs.get("private") or {}
    shed = dict(private.get("shed") or {})
    seeds = dict((str(k).upper(), int(v or 0)) for k, v in (private.get("seeds") or {}).items())
    farmer = list(farm.get("farmer", [4, 4]))
    hands = [list(p) for p in (farm.get("hands") or [])]
    market, used = [], [0]

    def order(o):
        if used[0] < MAX_ORDERS:
            market.append(o)
            used[0] += 1

    # --- one-shot directives (blind ledger) ---
    for i, d in enumerate(SCHEDULE):
        if d["day"] > day or (d["day"], i) in _fired:
            continue
        a = d["action"]
        if a in ("EXPAND_NE", "EXPAND_SW", "EXPAND_SE"):
            order(["BUY_LAND"])
            _fired.add((d["day"], i))
        elif a == "HIRE":
            for _ in range(int(d.get("count", 1))):
                order(["HIRE"])
            _fired.add((d["day"], i))
        elif a == "BUY_SEED":
            order(["BUY_SEED", d["crop"], int(d.get("count", 4))])
            _fired.add((d["day"], i))
        elif a == "BUY_ANIMAL":
            order(["BUY_ANIMAL", d["animal"], int(d.get("count", 1))])
            _fired.add((d["day"], i))

    # --- seed provisioning for active PLANT directives ---
    for d in SCHEDULE:
        if d["day"] <= day and d["action"] == "PLANT":
            have = seeds.get(d["crop"], 0)
            need = max(0, int(d.get("count", 8)) - have)
            if need > 0 and money > 500:
                order(["BUY_SEED", d["crop"], min(need, 8)])

    # --- continuous DUMP directives: liquidate shed stock ---
    for d in SCHEDULE:
        if d["day"] <= day and d["action"] in ("SELL", "DUMP"):
            qty = int(shed.get(d["product"], 0) or 0)
            if qty > 0:
                order(["SELL", d["product"], min(qty, 12)])

    # --- field work via Kuhn-Munkres routing ---
    tiles = farm.get("tiles") or []
    empties, weeds, thirsty, ripe, animals = [], [], [], [], []
    for r in range(10):
        for c in range(10):
            cell = _tile(farm, r, c)
            if cell is None:
                empties.append((c, r))
            elif isinstance(cell, dict) and cell.get("kind") == "WEED":
                weeds.append((c, r))
            elif isinstance(cell, dict) and cell.get("kind") == "PLANT":
                if not cell.get("watered_today"):
                    thirsty.append((c, r))
                if int(cell.get("yield_units", 0) or 0) > 0:
                    ripe.append((c, r))
            elif isinstance(cell, dict) and cell.get("kind") == "PASTURE":
                if cell.get("animal"):
                    animals.append((c, r))
    plant_crops = [d["crop"] for d in SCHEDULE
                   if d["day"] <= day and d["action"] == "PLANT"
                   and seeds.get(d["crop"], 0) > 0 and empties]
    units = [("farmer", tuple(farmer))] + [(f"h{i}", tuple(p)) for i, p in enumerate(hands)]
    jobs = ([(t, ["DIG"]) for t in weeds]
            + [(t, ["WATER"]) for t in thirsty]
            + [(t, ["HARVEST"]) for t in ripe]
            + [(t, ["FEED"]) for t in animals]
            + [(t, ["CARE"]) for t in animals])
    if plant_crops:
        jobs += [(t, ["PLANT", plant_crops[0]]) for t in empties]
    frontier = jobs[:len(units) * 2]
    acts = {"farmer": ["PASS"], "hands": [["PASS"]] * len(hands)}
    if frontier and units:
        pairs = kuhn_munkres_route([u[1] for u in units], [j[0] for j in frontier])
        for ui, ji in pairs:
            name, pos = units[ui]
            tgt, cmd = frontier[ji]
            go = cmd if tuple(pos) == tuple(tgt) else _step(pos, tgt)
            if name == "farmer":
                acts["farmer"] = go
            else:
                acts["hands"][int(name[1:])] = go
    acts["market"] = market
    return acts
