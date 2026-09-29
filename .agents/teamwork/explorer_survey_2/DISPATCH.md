# DISPATCH: Survey Explorer 2 — Top 10 Replay Scraping & Ghost Fleet Pipeline

## Mission
Investigate the replay scraping mechanisms (Kaggle CLI), existing replay assets, `scripts/ladder_ghost.py`, and the ghost conversion / autopsy pipeline.

## Authoritative Request
Read `/Users/pranav/dev/kaggriculture/.agents/teamwork/ORIGINAL_REQUEST.md` (specifically `## Follow-up — 2026-09-28T19:24:01Z`).

## Scope & Focus
1. Inspect Kaggle CLI / replay fetching:
   - What replays currently exist in `replays/` or `replays/live_top10/`?
   - How does the system query the Kaggle CLI (`kaggle competitions leaderboard kaggriculture` / `episodes`) or existing scrape scripts to download 20 Top 10 ladder match replays?
   - Check what scripts exist for scraping or downloading episodes (e.g. in `scripts/`).
2. Inspect `scripts/ladder_ghost.py` and ghost generation:
   - How does `scripts/ladder_ghost.py` parse a match replay JSON?
   - How does it convert an episode replay into an executable static "Ghost Opponent" submission (`submissions/ladder_ghost_<EPISODE_ID>/main.py`)?
   - How does a ghost opponent replay its recorded actions turn by turn during an episode?
   - What happens if the seed or opponent diverges? How does the ghost handle unexpected state or illegal actions?
3. Inspect autopsy pipelines:
   - What autopsy tools or analysis scripts exist in the repository (e.g. `scripts/autopsy.py` or similar)?
   - How are replays analyzed for strategy, cash flow, and errors?

## Constraints
- Read-only exploration. DO NOT modify any source code files.
- Document exact file locations, script arguments, schemas of replays, and ghost submission structure.

## Deliverables
- Write `handoff.md` in your working directory (`/Users/pranav/dev/kaggriculture/.agents/teamwork/explorer_survey_2/handoff.md`).
- Follow the Handoff Protocol: Observation, Logic Chain, Caveats, Conclusion, Verification Method.


## 2026-09-28T19:26:41Z
Received dispatch from orchestrator_gauntlet (bfd4f1ea-f2d6-42bf-84a1-98a5722c2573):
Investigate:
1. Kaggle CLI replay downloading (`kaggle competitions leaderboard kaggriculture` / `episodes`) and existing replays in `replays/` and `replays/live_top10/`. Check scripts available for scraping or replay management.
2. `scripts/ladder_ghost.py` and the ghost generation pipeline: how it converts replays into executable `submissions/ladder_ghost_<EPISODE_ID>/main.py`, action replay mechanics, legality handling, and file layout.
3. Existing autopsy scripts and diagnostic tools.
