# DISPATCH: Survey Explorer 3 — Master Tournament, kagg CLI, & Optuna HPO

## Mission
Investigate the tournament runner (`kagg tournament`), seed generation, benchmark evaluation, diagnostic autopsy, and the Optuna HPO tuning loop.

## Authoritative Request
Read `/Users/pranav/dev/kaggriculture/.agents/teamwork/ORIGINAL_REQUEST.md` (specifically `## Follow-up — 2026-09-28T19:24:01Z`).

## Scope & Focus
1. Inspect Rust `kagg tournament` / simulation engine:
   - Check the `kagg` CLI tool (where is it built or located? Is `kagg` in PATH, or in `target/release/kagg`, or Python wrapper?).
   - What flags and arguments does `kagg tournament` accept? How does it configure matches, seeds (1,000 seeds), agents, concurrency, output format?
   - How does `kagg tournament` output results (JSON, CSV, stdout)?
2. Inspect Tournament Setup:
   - How to pit `hybrid_grandmaster_v2` against 20 Top 10 ghost opponents across 1,000 fixed seeds.
   - What are the benchmark assertions: overall win rate ≥ 60.0%, average terminal cash ≥ $130,000.
   - How to parse metrics: per-agent win rates, average terminal cash, score differentials, and turn execution latency.
3. Inspect Diagnostic Autopsy & Automated Optuna Tuning:
   - What scripts exist for Optuna HPO or weight tuning (e.g., in `scripts/`, `scripts/ladder_ghost.py`, or tuning modules)?
   - What are the Beam Search heuristic weights mentioned in R5 (intent-prior scales, holding penalties, sell margins)? Where are they located in the code?
   - How does the fallback diagnostic autopsy identify loss catalysts and launch Optuna re-tuning if benchmark criteria are not met?

## Constraints
- Read-only exploration. DO NOT modify any source code files.
- Document exact CLI commands, file paths, parameters, schemas, and metrics calculation.

## Deliverables
- Write `handoff.md` in your working directory (`/Users/pranav/dev/kaggriculture/.agents/teamwork/explorer_survey_3/handoff.md`).
- Follow the Handoff Protocol: Observation, Logic Chain, Caveats, Conclusion, Verification Method.


## 2026-09-28T19:26:41Z
Received dispatch from bfd4f1ea-f2d6-42bf-84a1-98a5722c2573 (orchestrator_gauntlet):
You are Survey Explorer 3 for the Kaggle Kaggriculture Master Integration Gauntlet project.
Your assigned working directory is: /Users/pranav/dev/kaggriculture/.agents/teamwork/explorer_survey_3
Your parent is: bfd4f1ea-f2d6-42bf-84a1-98a5722c2573 (orchestrator_gauntlet)

Please immediately read:
1. /Users/pranav/dev/kaggriculture/.agents/teamwork/ORIGINAL_REQUEST.md (specifically ## Follow-up — 2026-09-28T19:24:01Z)
2. /Users/pranav/dev/kaggriculture/.agents/teamwork/explorer_survey_3/DISPATCH.md

Investigate:
1. The Rust `kagg tournament` command, simulator binary/tooling, CLI options, seed support (1,000 fixed seeds), concurrency, and output reporting.
2. Tournament setup for pitting `hybrid_grandmaster_v2` against 20 Top 10 ghost opponents across 1,000 seeds, and metric evaluation (win rate ≥ 60.0%, cash ≥ $130,000, latency profile).
3. Diagnostic autopsy mechanisms and automated Optuna HPO loop (Beam Search weights: intent-prior scales, holding penalties, sell margins).

Write your findings and recommendations in `/Users/pranav/dev/kaggriculture/.agents/teamwork/explorer_survey_3/handoff.md`.
Maintain progress in `/Users/pranav/dev/kaggriculture/.agents/teamwork/explorer_survey_3/progress.md` and briefing in `BRIEFING.md`.
When finished, notify your parent via send_message with your handoff path.
