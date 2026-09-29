# DISPATCH: Survey Explorer 1 — Candidate Packaging & Watchdog

## Mission
Investigate the source candidate `scratch_grandmaster.py`, existing submission scripts (`scripts/test_submission.py`, existing `submissions/` structure), and the execution watchdog requirements.

## Authoritative Request
Read `/Users/pranav/dev/kaggriculture/.agents/teamwork/ORIGINAL_REQUEST.md` (specifically `## Follow-up — 2026-09-28T19:24:01Z`).

## Scope & Focus
1. Inspect `scratch_grandmaster.py`:
   - What components does it contain (IQL Value Net, Kuhn-Munkres routing layer, Claude Endgame DP solvers, PSRO Meta-Controller)?
   - What external dependencies does it import? Are there weights/models or external files needed?
   - How can it be packaged into a single self-contained `submissions/hybrid_grandmaster_v2/main.py` that satisfies <100 MiB, <6.5 GiB memory, and Kaggle environment constraints?
2. Inspect `scripts/test_submission.py` and submission validation standards:
   - What checks does `test_submission.py` execute?
   - What interfaces must `main.py` expose (`agent(obs, config)` or similar)?
3. Inspect Watchdog requirements:
   - `time.perf_counter()` watchdog enforcing:
     - Max 1.0 second per turn.
     - Max 60.0 seconds cumulative overage bank per 720-step episode.
   - How should the watchdog be integrated into `main.py` or wrapped around the agent function?
   - How does latency tracking behave across turns?

## Constraints
- Read-only exploration. DO NOT modify any source code files.
- Document all findings, exact line references, interface signatures, and packaging risks.

## Deliverables
- Write `handoff.md` in your working directory (`/Users/pranav/dev/kaggriculture/.agents/teamwork/explorer_survey_1/handoff.md`).
- Follow the Handoff Protocol: Observation, Logic Chain, Caveats, Conclusion, Verification Method.


## 2026-09-28T19:26:41Z
Received dispatch from parent `bfd4f1ea-f2d6-42bf-84a1-98a5722c2573` (orchestrator_gauntlet):
Investigate:
1. `scratch_grandmaster.py` and its architecture (IQL Value Net, Kuhn-Munkres routing, Claude Endgame DP solvers, PSRO Meta-Controller). Check all dependencies, weights, and modules.
2. `scripts/test_submission.py` and submission packaging rules (<100 MiB, <6.5 GiB memory, self-contained).
3. The execution watchdog requirements: `time.perf_counter()` watchdog enforcing max 1.0s/turn and max 60.0s cumulative overage bank per 720-step episode.
Write findings and recommendations in `/Users/pranav/dev/kaggriculture/.agents/teamwork/explorer_survey_1/handoff.md`.
Maintain progress in `progress.md` and briefing in `BRIEFING.md`.
Notify parent via send_message with handoff path when finished.


## 2026-09-28T19:34:52Z
Received status check from parent `bfd4f1ea-f2d6-42bf-84a1-98a5722c2573`:
Explorers 2 and 3 completed handoff reports. Conclude investigation of `scratch_grandmaster.py`, `scripts/test_submission.py`, and execution watchdog, write handoff report to `handoff.md`, and notify parent.
