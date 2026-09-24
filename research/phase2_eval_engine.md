# Phase 2 Evaluation Engine

`scripts/eval_runner.py` evaluates a candidate on replay seeds using the local
Kaggle simulator. The recorded opponent action is selected at `obs.step + 1`,
matching the replay convention in `research/recon_findings.md`. If that frame
is missing or malformed, a deterministic surrogate sells one unit of the
highest-priced item in the visible shed; if observation parsing fails, the
runner emits a safe PASS action.

The runner performs a second run on the same seed with the simple surrogate,
then writes `eval_results.md` (or `--output PATH`). `scripts/eval_metrics.py`
reports:

* Seed-Matched Delta: candidate cash minus surrogate-run cash.
* Counterfactual Margin: candidate cash minus the replay-run opponent cash.
* Divergence Rate: surrogate/no-op fallback turns divided by replay turns.
* Exact, surrogate, no-op, malformed counts and terminal farm cash.

## Assumptions and limitations

Terminal cash is farm money only. Replay actions are not normalized, preserving
empty `hands` and `market` lists. The simulator episode length follows the
replay step count. A replay's `info.seed` is required. The surrogate is
deliberately conservative, not an attempt to reproduce an elite policy.
Unexpected simulator exceptions fail only the affected replay and are surfaced
in the Markdown table. Missing Kaggle dependencies produce an explicit error
instead of fabricated metrics. The engine does not modify simulators, tests, or
the virtual environment.
