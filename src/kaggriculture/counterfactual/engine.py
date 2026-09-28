"""Counterfactual rollout engine for injecting alternative actions and evaluating outcomes."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence
from kaggsim.serve import Serve


class CounterfactualRolloutEngine:
    """Forks simulation states at arbitrary steps and executes counterfactual rollouts."""

    def __init__(self, binary_path: Path | str | None = None):
        self.binary_path = str(binary_path) if binary_path is not None else None
        self._serve: Serve | None = None

    def _get_serve(self) -> Serve:
        if self._serve is None:
            self._serve = Serve(self.binary_path)
        return self._serve

    def fork_state(self, state_dict: Mapping[str, Any], step: int | None = None) -> dict[str, Any]:
        """Create an isolated, independent deep copy of an engine state."""
        forked = copy.deepcopy(dict(state_dict))
        if step is not None:
            forked["step"] = int(step)
            forked["day"] = int(step) // 24
            forked["hour"] = int(step) % 24
        return forked

    def evaluate_transition(
        self,
        state_dict: Mapping[str, Any],
        alt_action: Sequence[str] | str,
        downstream_tape: Sequence[str] | None = None,
        seat: int = 0,
        horizon: int = 10,
    ) -> float:
        """Fork state, inject alternative counterfactual market order, and rollout.

        Args:
            state_dict: Full engine state dictionary at timestep t.
            alt_action: Alternative action string or list of order strings to inject at t.
            downstream_tape: Optional recorded subsequent tape actions to replay.
            seat: Player seat index (0 or 1).
            horizon: Number of simulation steps to rollout.

        Returns:
            Terminal cash value resulting from the counterfactual branch.
        """
        srv = self._get_serve()
        forked = self.fork_state(state_dict)

        # Format alternative action as a valid tape action line
        if isinstance(alt_action, (list, tuple)):
            alt_str = ";".join(str(a) for a in alt_action)
        else:
            alt_str = str(alt_action)

        if "\t" not in alt_str:
            first_line = f"PASS\t\t{alt_str}"
        else:
            first_line = alt_str

        lines0 = [first_line]
        if downstream_tape:
            for line in downstream_tape:
                if "\t" not in line:
                    lines0.append(f"PASS\t\t{line}")
                else:
                    lines0.append(line)

        current_step = int(forked.get("step", 0))
        remaining = 719 - current_step
        actual_horizon = min(max(0, remaining), max(len(lines0), horizon))

        if actual_horizon <= 0:
            farms = forked.get("farms") or []
            farm = farms[seat] if seat < len(farms) else {}
            return float(farm.get("money", 0.0) or 0.0)

        while len(lines0) < actual_horizon:
            lines0.append("")

        lines1 = [""] * len(lines0)
        final_obs = srv.rollout(forked, horizon=actual_horizon, lines0=lines0, lines1=lines1)
        farms = final_obs.get("farms") or []
        farm = farms[seat] if seat < len(farms) else {}
        return float(farm.get("money", 0.0) or 0.0)

    def close(self):
        if self._serve is not None:
            self._serve.close()
            self._serve = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
