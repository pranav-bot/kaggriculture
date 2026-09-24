def interpreter(state, env):
    num_agents = len(state)
    obs0 = state[0].observation

    if not hasattr(obs0, "farms") or not obs0.farms:
        _initialize(state, env)
        return state

    if env.done:
        return state

    cfg = env.configuration
    turns_per_day = max(1, int(get(cfg, "turnsPerDay", 24)))
    board_size = int(get(cfg, "boardSize", 10))
    shed_capacity = int(get(cfg, "shedCapacity", 100))

    step = get(obs0, "step", 0)
    day = step // turns_per_day

    for i, s in enumerate(state):
        action = s.action if isinstance(s.action, dict) else {}
        farmer_action = action.get("farmer", ["PASS"]) if isinstance(action, dict) else ["PASS"]
        hands_actions = action.get("hands", []) if isinstance(action, dict) else []
        if not isinstance(hands_actions, list):
            hands_actions = []

        # Atomic PLANT validation: if total PLANT requests for a crop this turn
        # exceed available seeds, drop ALL PLANT requests for that crop.
        unit_actions = [farmer_action, *hands_actions]
        plant_demand = {}
        for a in unit_actions:
            if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT":
                plant_demand[a[1]] = plant_demand.get(a[1], 0) + 1
        seeds = s.observation.private.get("seeds", {}) if hasattr(s.observation.private, "get") else {}
        blocked = {crop for crop, n in plant_demand.items() if n > seeds.get(crop, 0)}

        def _allowed(a):
            if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT" and a[1] in blocked:
                return ["PASS"]
            return a

        _apply_unit_action(obs0.farms[i], s.observation.private, 0, _allowed(farmer_action),
                           board_size, day, turns_per_day, shed_capacity)
        for h_idx, hand_action in enumerate(hands_actions):
            _apply_unit_action(obs0.farms[i], s.observation.private, h_idx + 1,
                               _allowed(hand_action), board_size, day, turns_per_day, shed_capacity)

    _process_market(state, env)
    _town_consume(env, state, step)
    for farm in obs0.farms:
        _decay_plants(farm, step)
    if (step + 1) % turns_per_day == 0:
        _end_of_day(state, env, day)

    next_step = step + 1
    obs0.day = next_step // turns_per_day
    obs0.hour = next_step % turns_per_day
    for i in range(1, num_agents):
        state[i].observation.farms = obs0.farms
        state[i].observation.market = obs0.market
        state[i].observation.town = obs0.town
        state[i].observation.day = obs0.day
        state[i].observation.hour = obs0.hour

    # `step` here is the previous step counter; framework records the post-interpreter
    # state at the next index. -2 fires DONE on the final recorded step.
    if step >= cfg.episodeSteps - 2:
        for s in state:
            s.status = "DONE"
            s.reward = float(obs0.farms[s.observation.player]["money"])

    return state
