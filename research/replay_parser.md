# Replay parser (Phase 2)

`scripts/replay_parser.py` is the compact replay extraction layer. It scans
both `replay/` and `replays/` (when present), parses one JSON file at a time,
and emits one typed `ReplayRecord`. The CLI defaults to one-line summaries so
large farm tiles and full observations are never dumped accidentally:

```bash
.venv/bin/python scripts/replay_parser.py --summary --limit 1
.venv/bin/python scripts/replay_parser.py replays/my_agents/care_mill/111752837.json --summary
# Add --full only when a caller needs every compact turn record.
.venv/bin/python scripts/replay_parser.py replays/my_agents/care_mill/111752837.json --full --turn-stop 4
```

## Extracted contract

`ReplayRecord.seed` prefers `info.seed` and falls back to
`configuration.seed`. Each frame becomes a `TurnRecord` containing player and
turn indexes, action (`farmer`, `hands`, `market`), a focused observation,
cash delta, inventory deltas, frame reward, and status. Inventory deltas are
reported independently for carried inventories, seeds, and shed contents.
Terminal rewards come from final/DONE frame rewards, with top-level `rewards`
as a fallback.

Observations retain time, observing player, cash, hand count, hires,
unlocked quadrants, private inventory sections, market, and town. Farm tiles
are intentionally omitted: they are the dominant payload and can be read from
the source replay when a later analysis specifically needs them.

## Assumptions and follow-ups

* `steps[turn]` is a list indexed by player, as documented in
  `research/recon_findings.md`; malformed frames are skipped and recorded in
  `errors`.
* `private` belongs to the observing player. The parser computes deltas per
  frame/player and does not infer hidden opponent inventory.
* A missing `info.seed` is not treated as zero; `None` remains distinguishable
  from an explicit seed.
* The standard library decoder loads one source JSON at a time. This bounds
  retained parsed state to one replay and immediately reduces it to compact
  records. If replay files grow substantially, an optional incremental JSON
  backend (for example `ijson`) should be added rather than changing the
  record schema.
* Some historical replays may omit rewards or terminal statuses; the parser
  falls back to top-level `rewards` and reports no terminal reward when neither
  source exists.
