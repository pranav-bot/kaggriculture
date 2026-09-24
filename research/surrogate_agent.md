# Phase 2 surrogate agent

`scripts/surrogate_agent.py` implements the replay fallback contract from
`research/recon_findings.md` without changing the simulator or tests.

## Interface and tiers

`SurrogateAgent.step(observation)` returns the normal Kaggriculture action
mapping (`farmer`, `hands`, and `market`). A replay source can be a JSON file,
the decoded replay object, a turn-indexed mapping, or a list. Replay lookup
uses `observation.step + 1` by default because replay frames describe the
post-interpreter state. A frame can contain either an action directly or an
`action` member.

1. **Recorded intent:** structurally valid recorded actions are retained.
   Quantities are scaled to current private shed/seeds/cash and plant targets
   are changed to an available seed when the recorded target is impossible.
   Such scaling is counted as divergence.
2. **Behavioral archetype:** malformed/missing frames produce deterministic
   survival actions at each worker's current tile: water, harvest, dig, feed,
   care, then sell available shed stock. This is intentionally conservative
   and bounded; it recalculates every turn and cannot wait in an unbounded
   loop.
3. **Safe no-op:** malformed observations or unexpected policy failures return
   `{"farmer": ["PASS"], "hands": [], "market": []}`.

The agent exposes `metrics()`, `exact_replay_turns`, `surrogate_turns`,
`safe_noop_turns`, `divergence_count`, `fallback_count`, and
`fallback_reasons`. Counters are per agent instance and deterministic.

## Assumptions and edge cases

- The observation is the local player's dictionary and `farms[player]` is the
  corresponding public farm.
- Missing or non-numeric quantities are treated as zero.
- The local market price is used as a conservative affordability estimate;
  unknown purchase prices default to 20, so a purchase may still be rejected
  by the simulator rather than crashing the replay.
- The agent validates action shape and operation names, not full tile legality;
  Kaggriculture's interpreter intentionally treats illegal actions as no-ops.
- A seed mismatch (when both the configured replay seed and observation seed
  are present) forces surrogate mode. Most local observations omit the seed.
- The fallback sells at most ten orders and never invents inventory.

This implementation is a focused replay surrogate, not a replacement for the
strong production policies in `submissions/`.
