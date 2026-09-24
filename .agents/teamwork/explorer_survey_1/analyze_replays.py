import json
from collections import Counter

for path in [
    'replays/my_agents/agent_final: Sovereign Apex k+/112619304.json',
    'replays/my_agents/agent_final: Sovereign Apex k+/112618133.json'
]:
    with open(path) as f:
        d = json.load(f)
    print(f"\n========================================================")
    print(f"*** Analysis for {path} ***")
    steps = d.get('steps', [])
    p0_market_actions = Counter()
    p1_market_actions = Counter()
    
    days_to_check = [5, 10, 15, 20, 25, 29]
    for s_idx, s in enumerate(steps):
        day = s[0]['observation']['day']
        hour = s[0]['observation']['hour']
        for p_idx, counter_m in [(0, p0_market_actions), (1, p1_market_actions)]:
            act = s[p_idx].get('action') or {}
            for m in act.get('market', []):
                key = m[0] + (('_' + m[1]) if len(m) > 1 else '')
                counter_m[key] += (m[2] if len(m) > 2 else 1)
        
        if hour == 23 and day in days_to_check:
            days_to_check.remove(day)
            print(f"\n--- Day {day} End Snapshot ---")
            for p_idx in [0, 1]:
                farm = s[p_idx]['observation']['farms'][p_idx]
                tiles = farm['tiles']
                t_counts = Counter()
                for r in tiles:
                    for t in r:
                        if t == 'LOCKED':
                            t_counts['LOCKED'] += 1
                        elif t is None:
                            t_counts['EMPTY'] += 1
                        elif isinstance(t, dict):
                            kind = t.get('kind')
                            if kind == 'PLANT':
                                t_counts['PLANT_' + str(t.get('crop'))] += 1
                            elif kind == 'PASTURE':
                                t_counts['PASTURE_' + str(t.get('animal', 'empty'))] += 1
                            else:
                                t_counts[kind] += 1
                shed = s[p_idx]['observation']['private']['shed']
                money = farm['money']
                quads = farm['unlocked_quadrants']
                print(f"P{p_idx} (Cash ${money:,.0f}, Quads {quads}): tiles={dict(t_counts)}, shed={shed}")

    print("\nP0 Market Totals:", dict(p0_market_actions))
    print("P1 Market Totals:", dict(p1_market_actions))
