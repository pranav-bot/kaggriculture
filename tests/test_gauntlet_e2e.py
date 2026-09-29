"""Master Integration Gauntlet E2E Acceptance Test Suite.

Authoritative Test Suite covering:
- Tier 1: Feature Coverage (candidate packaging, watchdog, ghost fleet, tournament config).
- Tier 2: Boundary & Corner Cases (time limit drawdown, circuit breaker, forfeiture, boundary turns, memory/latency).
- Tier 3: Cross-Feature Interactions (candidate vs ghost match execution).
- Tier 4: Master Tournament Acceptance Gate (win rate >= 60.0%, terminal cash >= $130,000, latency profile, loss autopsy, Optuna HPO).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

# Ensure root, src, and scripts directories are on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"

for p in (ROOT, SRC, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from kaggriculture.safety import (
    DEFAULT_OVERAGE_BANK_S,
    DEFAULT_SAFETY_THRESHOLD_S,
    DEFAULT_SOFT_LIMIT_S,
    ImpenetrableAgentWrapper,
    impenetrable_agent,
)
from scripts.autopsy_ladder_losses import analyze_loss_match
from scripts.ladder_ghost import sample_beam_weights, run_hpo
from scripts.run_elo_tournament import find_kagg_binary
from scripts.sim_engine import FastSimulation, is_kagg_available



# =============================================================================
# Acceptance Evaluators & Validation Helpers
# =============================================================================

def validate_candidate_packaging(path: Path) -> Dict[str, Any]:
    """Validate candidate submission packaging compliance per PROJECT.md § Interface Contracts.

    Requirements:
    1. File exists and is < 100 MiB.
    2. Valid standalone Python code (parseable via ast.parse).
    3. Exposes `agent(obs, config=None)` as the last top-level callable in the file.
    """
    assert path.is_file(), f"Candidate submission file not found: {path}"
    file_size_mb = path.stat().st_size / (1024 * 1024)
    assert file_size_mb < 100.0, f"Candidate size {file_size_mb:.2f} MiB exceeds 100 MiB limit"

    content = path.read_text(encoding="utf-8")
    tree = ast.parse(content, filename=str(path))

    # Identify all top-level function definitions
    top_level_funcs = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert len(top_level_funcs) > 0, "No top-level functions defined in submission"

    func_names = [f.name for f in top_level_funcs]
    assert "agent" in func_names, "Callable entry point 'agent' not defined at module level"

    last_func = top_level_funcs[-1]
    assert last_func.name == "agent", (
        f"Callable entry point 'agent' must be the last top-level callable in the file; "
        f"found '{last_func.name}' at line {last_func.lineno}"
    )

    return {
        "path": str(path),
        "size_mb": file_size_mb,
        "total_top_level_funcs": len(top_level_funcs),
        "last_callable": last_func.name,
    }


def validate_ghost_submission(ghost_dir: Path) -> Dict[str, Any]:
    """Validate a single ghost submission directory.

    Requirements:
    1. Contains `main.py`.
    2. Exposes a callable `agent(obs)`.
    3. When invoked with a standard observation, returns valid action dictionary.
    """
    main_py = ghost_dir / "main.py"
    assert main_py.is_file(), f"Ghost submission main.py not found in {ghost_dir}"

    module_name = f"ghost_mod_{ghost_dir.name}"
    spec = importlib.util.spec_from_file_location(module_name, main_py)
    assert spec is not None and spec.loader is not None, f"Could not load spec for {main_py}"

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    assert hasattr(mod, "agent") and callable(mod.agent), (
        f"Ghost module {ghost_dir.name} does not expose callable 'agent'"
    )

    # Test initial turn invocation
    sample_obs = create_mock_obs(step=0, day=0, hour=0)
    action = mod.agent(sample_obs)

    assert isinstance(action, dict), f"Ghost returned non-dict action: {type(action)}"
    for key in ("farmer", "hands", "market"):
        assert key in action, f"Ghost action missing required key '{key}'"

    return {
        "name": ghost_dir.name,
        "main_py": str(main_py),
        "callable": "agent",
    }


def validate_gauntlet_tournament_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate tournament configuration compliance for gauntlet_1000.

    Requirements per PROJECT.md:
    1. Schedule must be 'gauntlet'.
    2. Dual time limits: act_s = 1.0, overage_s = 60.0.
    3. Valid Python executable pointing to project environment.
    4. Candidate specified with valid name and path.
    5. Panel specified containing opponent specifications.
    6. on_error set to 'forfeit'.
    """
    assert cfg.get("schedule") == "gauntlet", (
        f"Expected schedule 'gauntlet', got '{cfg.get('schedule')}'"
    )

    time_limits = cfg.get("time_limits", {})
    assert float(time_limits.get("act_s", 0.0)) == 1.0, (
        f"act_s must be 1.0, got {time_limits.get('act_s')}"
    )
    assert float(time_limits.get("overage_s", 0.0)) == 60.0, (
        f"overage_s must be 60.0, got {time_limits.get('overage_s')}"
    )

    py_cfg = cfg.get("python", {})
    py_exe = py_cfg.get("exe")
    assert py_exe is not None, "Missing 'python.exe' in tournament configuration"
    assert Path(py_exe).is_file() or py_exe in ("python", "python3"), (
        f"Python executable '{py_exe}' does not exist on disk"
    )

    assert "candidate" in cfg and isinstance(cfg["candidate"], dict), "Missing 'candidate' spec"
    assert "name" in cfg["candidate"] and "path" in cfg["candidate"], "Incomplete candidate spec"

    assert "panel" in cfg and isinstance(cfg["panel"], list), "Missing 'panel' list"
    assert len(cfg["panel"]) > 0, "Panel list must not be empty"

    assert cfg.get("on_error") == "forfeit", (
        f"Expected on_error 'forfeit', got '{cfg.get('on_error')}'"
    )

    return {
        "schedule": cfg["schedule"],
        "act_s": time_limits["act_s"],
        "overage_s": time_limits["overage_s"],
        "panel_count": len(cfg["panel"]),
    }


def evaluate_tournament_acceptance(
    summary: Dict[str, Any],
    results_rows: List[Dict[str, Any]],
    *,
    min_win_rate: float = 0.60,
    min_terminal_cash: float = 130000.0,
    max_mean_latency_ms: float = 5.0,
) -> Dict[str, Any]:
    """Parse and evaluate Master Tournament Acceptance Gate criteria.

    Asserts:
    1. Overall candidate win rate >= min_win_rate (60.0%).
    2. Average candidate terminal cash balance >= min_terminal_cash ($130,000).
    3. Candidate mean turn latency < max_mean_latency_ms (5.0ms) and max latency < 1000.0ms.
    4. Zero candidate simulation errors or forfeitures.
    """
    assert len(results_rows) > 0, "Empty tournament results rows"

    # Identify candidate agent name
    candidate_name = None
    if "focus" in summary and isinstance(summary["focus"], dict):
        candidate_name = summary["focus"].get("agent")

    if not candidate_name:
        for row in results_rows:
            roles = row.get("roles", [])
            agents = row.get("agents", [])
            if "candidate" in roles:
                candidate_name = agents[roles.index("candidate")]
                break

    assert candidate_name is not None, "Could not determine candidate agent from tournament data"

    # 1. Win Rate Assertion
    win_rate: Optional[float] = None
    if "focus" in summary and "overall" in summary["focus"]:
        win_rate = float(summary["focus"]["overall"].get("score", 0.0))
    elif "standings" in summary:
        for entry in summary["standings"]:
            if entry.get("agent") == candidate_name:
                win_rate = float(entry.get("score", 0.0))
                break

    if win_rate is None:
        # Compute from results rows
        cand_scores = []
        for row in results_rows:
            if candidate_name in row.get("agents", []):
                idx = row["agents"].index(candidate_name)
                cand_scores.append(float(row["scores"][idx]))
        win_rate = sum(cand_scores) / max(1, len(cand_scores))

    assert win_rate >= min_win_rate, (
        f"Candidate win rate {win_rate * 100:.2f}% breached acceptance threshold {min_win_rate * 100:.1f}%"
    )

    # 2. Terminal Cash Assertion
    candidate_cash_balances: List[float] = []
    cand_errors = 0
    cand_latencies: List[float] = []

    for row in results_rows:
        agents = row.get("agents", [])
        if candidate_name in agents:
            idx = agents.index(candidate_name)
            banks = row.get("banks", [])
            if len(banks) > idx:
                candidate_cash_balances.append(float(banks[idx]))

            if row.get("forfeit", False) or row.get("error") is not None:
                roles = row.get("roles", [])
                if len(roles) > idx and roles[idx] == "candidate":
                    cand_errors += 1

            act_means = row.get("act_ms_mean", [])
            if len(act_means) > idx and act_means[idx] is not None:
                cand_latencies.append(float(act_means[idx]))

    assert len(candidate_cash_balances) > 0, (
        f"No terminal cash records found for candidate '{candidate_name}'"
    )
    avg_terminal_cash = sum(candidate_cash_balances) / len(candidate_cash_balances)
    assert avg_terminal_cash >= min_terminal_cash, (
        f"Candidate average terminal cash ${avg_terminal_cash:,.2f} breached acceptance "
        f"threshold ${min_terminal_cash:,.2f}"
    )

    # 3. Latency Profile Assertion
    cand_mean_latency: Optional[float] = None
    if "standings" in summary:
        for entry in summary["standings"]:
            if entry.get("agent") == candidate_name:
                cand_mean_latency = float(entry.get("act_ms_mean", 0.0))
                break

    if cand_mean_latency is None and cand_latencies:
        cand_mean_latency = sum(cand_latencies) / len(cand_latencies)

    cand_mean_latency = cand_mean_latency or 0.0
    assert cand_mean_latency < max_mean_latency_ms, (
        f"Candidate mean turn latency {cand_mean_latency:.3f}ms breached limit {max_mean_latency_ms:.1f}ms"
    )

    # 4. Zero Error Assertion
    assert cand_errors == 0, f"Candidate suffered {cand_errors} unhandled error/forfeiture matches"

    return {
        "candidate": candidate_name,
        "total_games": len(candidate_cash_balances),
        "win_rate": win_rate,
        "avg_terminal_cash": avg_terminal_cash,
        "mean_latency_ms": cand_mean_latency,
        "errors": cand_errors,
        "gate_passed": True,
    }


def create_mock_obs(
    step: int = 0,
    day: int = 0,
    hour: int = 0,
    money: float = 5000.0,
    shed: Optional[Dict[str, int]] = None,
    seeds: Optional[Dict[str, int]] = None,
    remaining_overage: Optional[float] = 60.0,
) -> Dict[str, Any]:
    """Build a standard observation conforming to Kaggle Kaggriculture schema."""
    tiles = [[None for _ in range(10)] for _ in range(10)]
    return {
        "step": step,
        "day": day,
        "hour": hour,
        "player": 0,
        "remainingOverageTime": remaining_overage,
        "farms": [
            {
                "money": money,
                "tiles": tiles,
                "farmer": [4, 4],
                "hands": [[4, 5]],
                "unlocked_quadrants": ["NW"],
            },
            {
                "money": money,
                "tiles": [[None for _ in range(10)] for _ in range(10)],
                "farmer": [5, 4],
                "hands": [[5, 5]],
                "unlocked_quadrants": ["NW"],
            },
        ],
        "private": {"shed": shed or {}, "seeds": seeds or {}},
        "market": {"inventory": {}, "prices": {"MILK": 160.0, "WOOL": 200.0, "STRAWBERRY": 120.0}},
        "town": {"unlocked_shops": ["BAKERY", "ICE_CREAM_SHOP"]},
    }


# =============================================================================
# Tier 1: Feature Coverage Tests
# =============================================================================

def test_tier1_candidate_packaging_compliance():
    """Tier 1: Verify candidate packaging compliance.

    Verifies packaging validator on existing reference agent, and verifies
    `submissions/hybrid_grandmaster_v2/main.py` if present (or skips if M1 in progress).
    """
    reference_path = ROOT / "submissions" / "agent_final" / "main.py"
    if reference_path.is_file():
        ref_meta = validate_candidate_packaging(reference_path)
        assert ref_meta["last_callable"] == "agent"
        assert ref_meta["size_mb"] < 100.0

    target_candidate = ROOT / "submissions" / "hybrid_grandmaster_v2" / "main.py"
    if not target_candidate.is_file():
        pytest.skip(
            f"Milestone 1 in progress: Candidate submission {target_candidate} not yet generated"
        )

    cand_meta = validate_candidate_packaging(target_candidate)
    assert cand_meta["last_callable"] == "agent"
    assert cand_meta["size_mb"] < 100.0


def test_tier1_watchdog_compliance_attributes():
    """Tier 1: Verify @impenetrable_agent watchdog wrapper attributes and configuration."""
    assert DEFAULT_SOFT_LIMIT_S == 1.0, f"Expected soft limit 1.0s, got {DEFAULT_SOFT_LIMIT_S}"
    assert DEFAULT_OVERAGE_BANK_S == 60.0, f"Expected overage bank 60.0s, got {DEFAULT_OVERAGE_BANK_S}"
    assert DEFAULT_SAFETY_THRESHOLD_S == 5.0, f"Expected threshold 5.0s, got {DEFAULT_SAFETY_THRESHOLD_S}"

    @impenetrable_agent
    def sample_agent(obs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"farmer": ["PASS"], "hands": [["PASS"]], "market": []}

    assert isinstance(sample_agent, ImpenetrableAgentWrapper)
    assert sample_agent.soft_limit_s == 1.0
    assert sample_agent.overage_bank_s == 60.0
    assert sample_agent.safety_threshold_s == 5.0
    assert sample_agent.remaining_overage == 60.0

    stats = sample_agent.stats()
    assert stats["remaining_overage_s"] == 60.0
    assert stats["circuit_broken"] is False
    assert stats["fallback_trigger_count"] == 0

    # Verify execution with observation
    obs = create_mock_obs()
    action = sample_agent(obs)
    assert action == {"farmer": ["PASS"], "hands": [["PASS"]], "market": []}


def test_tier1_ghost_fleet_compliance():
    """Tier 1: Verify ghost opponent directories and executable agents.

    Asserts that existing ghost submissions expose a valid callable agent.
    If full 20-fleet exists (M2 complete), asserts count == 20; otherwise tests existing.
    """
    submissions_dir = ROOT / "submissions"
    ghost_dirs = sorted(submissions_dir.glob("ladder_ghost_*"))

    assert len(ghost_dirs) > 0, "No ghost submissions found in submissions/"

    # Validate each currently existing ghost
    for g_dir in ghost_dirs:
        meta = validate_ghost_submission(g_dir)
        assert meta["callable"] == "agent"

    if len(ghost_dirs) < 20:
        pytest.skip(
            f"Milestone 2 in progress: Found {len(ghost_dirs)} ghost submissions (awaiting 20)"
        )
    else:
        assert len(ghost_dirs) >= 20, f"Expected at least 20 ghost submissions, found {len(ghost_dirs)}"


def test_tier1_tournament_config_compliance():
    """Tier 1: Verify tournament configuration structure for gauntlet_1000."""
    ref_cfg = {
        "name": "gauntlet_1000",
        "schedule": "gauntlet",
        "seats": "both",
        "candidate": {
            "name": "hybrid_grandmaster_v2",
            "type": "python",
            "path": "submissions/hybrid_grandmaster_v2/main.py",
        },
        "panel": [
            {
                "name": f"ladder_ghost_{i}",
                "type": "python",
                "path": f"submissions/ladder_ghost_{i}/main.py",
            }
            for i in range(20)
        ],
        "worlds": {"strategy": "range", "start": 0, "count": 50},
        "workers": 8,
        "on_error": "forfeit",
        "time_limits": {"act_s": 1.0, "overage_s": 60.0},
        "output": {"dir": "tournaments/gauntlet_1000", "resume": False},
        "python": {
            "exe": sys.executable,
            "path": [str(ROOT / "src"), str(ROOT)],
        },
    }

    validated = validate_gauntlet_tournament_config(ref_cfg)
    assert validated["schedule"] == "gauntlet"
    assert validated["act_s"] == 1.0
    assert validated["overage_s"] == 60.0
    assert validated["panel_count"] == 20

    # Check on-disk gauntlet configuration if Milestone 3 has generated it
    target_config = ROOT / "tournaments" / "gauntlet_1000" / "config.json"
    if target_config.is_file():
        disk_cfg = json.loads(target_config.read_text(encoding="utf-8"))
        validate_gauntlet_tournament_config(disk_cfg)


# =============================================================================
# Tier 2: Boundary & Corner Cases Tests
# =============================================================================

def test_tier2_watchdog_overage_drawdown():
    """Tier 2: Verify that turn duration exceeding 1.0s draws down overage bank."""
    # Create wrapper with soft_limit=1.0s, overage_bank=60.0s
    def slow_agent(obs: Dict[str, Any]) -> Dict[str, Any]:
        return {"farmer": ["PASS"], "hands": [], "market": []}

    wrapped = impenetrable_agent(
        slow_agent,
        soft_limit_s=1.0,
        overage_bank_s=60.0,
        safety_threshold_s=5.0,
    )

    # Turn 1: Simulating duration = 1.25s (excess = 0.25s)
    # Using patch on time.perf_counter to ensure exact deterministic math
    with patch("time.perf_counter", side_effect=[100.0, 101.25]):
        action = wrapped(create_mock_obs(step=0))
        assert action["farmer"] == ["PASS"]

    assert abs(wrapped.cumulative_overage_used - 0.25) < 1e-4
    assert abs(wrapped.remaining_overage - 59.75) < 1e-4
    assert wrapped.circuit_broken is False

    # Turn 2: Simulating fast turn = 0.05s (duration <= 1.0s -> no overage consumed)
    with patch("time.perf_counter", side_effect=[102.0, 102.05]):
        wrapped(create_mock_obs(step=1))

    assert abs(wrapped.cumulative_overage_used - 0.25) < 1e-4
    assert abs(wrapped.remaining_overage - 59.75) < 1e-4


def test_tier2_watchdog_circuit_breaker_and_timeout_forfeiture():
    """Tier 2: Verify circuit breaker trips below 5.0s and prevents timeout forfeiture."""
    primary_called = False

    def sluggish_agent(obs: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal primary_called
        primary_called = True
        return {"farmer": ["PASS"], "hands": [], "market": []}

    wrapped = impenetrable_agent(
        sluggish_agent,
        soft_limit_s=1.0,
        overage_bank_s=60.0,
        safety_threshold_s=5.0,
    )

    # Simulate catastrophic overage draw: turn takes 56.5s -> overage consumed 55.5s
    # Remaining overage = 60.0 - 55.5 = 4.5s (< 5.0s threshold)
    with patch("time.perf_counter", side_effect=[10.0, 66.5]):
        wrapped(create_mock_obs(step=10))

    assert wrapped.circuit_broken is True
    assert wrapped.remaining_overage < 5.0

    # Next turn: primary agent MUST NOT be called; fallback executes directly
    primary_called = False
    with patch("time.perf_counter", side_effect=[70.0, 70.0001]):
        action = wrapped(create_mock_obs(step=11))

    assert primary_called is False, "Primary agent was called despite active circuit breaker"
    assert isinstance(action, dict)
    assert "farmer" in action


def test_tier2_boundary_step_verification():
    """Tier 2: Verify execution at boundary steps: Step 0, Step 71, Step 718, and Step 719."""
    reference_path = ROOT / "submissions" / "agent_final" / "main.py"
    if not reference_path.is_file():
        pytest.skip(f"Reference submission {reference_path} not found")

    spec = importlib.util.spec_from_file_location("ref_agent_boundary", reference_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Step 0 (Day 0, Hour 0): Cold start
    obs0 = create_mock_obs(step=0, day=0, hour=0, money=5000.0)
    act0 = mod.agent(obs0)
    assert isinstance(act0, dict)
    assert "farmer" in act0 and "market" in act0

    # Step 71 (Day 2, Hour 23): End of pre-unlock phase
    obs71 = create_mock_obs(step=71, day=2, hour=23, money=12000.0)
    act71 = mod.agent(obs71)
    assert isinstance(act71, dict)

    # Step 718 (Day 29, Hour 22): Penultimate turn
    obs718 = create_mock_obs(step=718, day=29, hour=22, money=95000.0)
    act718 = mod.agent(obs718)
    assert isinstance(act718, dict)

    # Step 719 (Day 29, Hour 23): Terminal turn of 720-step season
    obs719 = create_mock_obs(
        step=719,
        day=29,
        hour=23,
        money=98000.0,
        shed={"MILK": 40, "WOOL": 30, "FERTILIZER": 20, "STRAWBERRY": 50},
    )
    act719 = mod.agent(obs719)
    assert isinstance(act719, dict)
    assert "market" in act719


def test_tier2_memory_and_latency_assertion():
    """Tier 2: Verify turn latency < 50ms and zero unbounded memory leak over 100 turns."""
    reference_path = ROOT / "submissions" / "agent_final" / "main.py"
    if not reference_path.is_file():
        pytest.skip(f"Reference submission {reference_path} not found")

    spec = importlib.util.spec_from_file_location("ref_agent_perf", reference_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    durations: List[float] = []

    # Warm-up 10 turns
    for s in range(10):
        mod.agent(create_mock_obs(step=s, day=s // 24, hour=s % 24))

    # Profile 100 turns with tracemalloc
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    for s in range(10, 110):
        t0 = time.perf_counter()
        mod.agent(create_mock_obs(step=s, day=s // 24, hour=s % 24))
        durations.append(time.perf_counter() - t0)

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    max_turn_ms = max(durations) * 1000.0
    mean_turn_ms = (sum(durations) / len(durations)) * 1000.0

    assert mean_turn_ms < 5.0, f"Mean turn latency {mean_turn_ms:.3f}ms exceeds 5.0ms threshold"
    assert max_turn_ms < 50.0, f"Max turn latency {max_turn_ms:.3f}ms exceeds 50.0ms limit"

    # Memory growth assertion: delta across 100 turns must be bounded (< 500 KB)
    stats_diff = snapshot_after.compare_to(snapshot_before, "lineno")
    total_delta_bytes = sum(stat.size_diff for stat in stats_diff if stat.size_diff > 0)
    delta_kb = total_delta_bytes / 1024.0

    assert delta_kb < 500.0, f"Detected memory leak: {delta_kb:.1f} KB allocated across 100 turns"


# =============================================================================
# Tier 3: Cross-Feature Interactions Tests
# =============================================================================

def test_tier3_candidate_vs_ghost_match_execution():
    """Tier 3: Execute full 720-turn match between candidate and ghost opponent."""
    if not is_kagg_available():
        pytest.skip("Native Rust kagg engine not available for match simulation")

    kagg_bin = find_kagg_binary()
    assert Path(kagg_bin).is_file(), f"kagg executable not found at {kagg_bin}"

    # Select candidate (target hybrid_grandmaster_v2 if ready, else agent_final reference)
    candidate_path = ROOT / "submissions" / "hybrid_grandmaster_v2" / "main.py"
    if not candidate_path.is_file():
        candidate_path = ROOT / "submissions" / "agent_final" / "main.py"

    assert candidate_path.is_file(), f"No candidate submission found at {candidate_path}"

    ghost_path = ROOT / "submissions" / "ladder_ghost_112542379" / "main.py"
    assert ghost_path.is_file(), f"Ghost submission not found at {ghost_path}"

    with FastSimulation() as sim:
        p1_money, p2_money, final_st = sim.run_match(
            str(candidate_path),
            str(ghost_path),
            seed=42,
        )

    # Assert clean completion to final turn
    assert int(final_st.get("step", 0)) == 719, (
        f"Match terminated prematurely at step {final_st.get('step')}"
    )
    assert final_st.get("forfeit") is None, f"Match forfeited: {final_st.get('forfeit')}"
    assert p1_money > 0.0, f"Candidate terminal cash must be positive, got ${p1_money}"
    assert p2_money > 0.0, f"Ghost terminal cash must be positive, got ${p2_money}"


# =============================================================================
# Tier 4: Master Tournament Acceptance Gate & Fallback Tuning Tests
# =============================================================================

def test_tier4_tournament_results_parser_passing():
    """Tier 4: Verify acceptance gate parser succeeds on compliant tournament data."""
    summary_fixture = {
        "name": "gauntlet_1000",
        "schedule": "gauntlet",
        "focus": {
            "agent": "hybrid_grandmaster_v2",
            "overall": {"games": 1000, "wins": 650, "draws": 0, "losses": 350, "score": 0.65},
        },
        "standings": [
            {
                "agent": "hybrid_grandmaster_v2",
                "score": 0.65,
                "act_ms_mean": 1.85,
                "act_ms_max": 4.2,
                "errors": 0,
            }
        ],
    }

    # Generate 1,000 synthetic rows averaging $142,500 terminal cash
    results_fixture = [
        {
            "game_id": f"game_{i:04d}",
            "seed": i,
            "agents": ["hybrid_grandmaster_v2", f"ladder_ghost_{i % 20}"],
            "roles": ["candidate", "panel"],
            "banks": [142500.0 + (i % 1000), 95000.0],
            "scores": [1.0 if (i % 10 < 6) else 0.0, 0.0 if (i % 10 < 6) else 1.0],
            "act_ms_mean": [1.85, 0.45],
            "act_ms_max": [4.2, 1.1],
            "error": None,
            "forfeit": False,
        }
        for i in range(1000)
    ]

    report = evaluate_tournament_acceptance(summary_fixture, results_fixture)
    assert report["gate_passed"] is True
    assert report["win_rate"] == 0.65
    assert report["avg_terminal_cash"] >= 130000.0
    assert report["mean_latency_ms"] < 5.0


def test_tier4_tournament_results_parser_failing():
    """Tier 4: Verify acceptance gate raises explicit AssertionError on breached thresholds."""
    # Case A: Low Win Rate (55.0% < 60.0%)
    summary_low_win = {
        "focus": {"agent": "candidate", "overall": {"score": 0.55}},
        "standings": [{"agent": "candidate", "score": 0.55, "act_ms_mean": 1.5, "errors": 0}],
    }
    results_low_win = [
        {"agents": ["candidate", "ghost"], "roles": ["candidate", "panel"], "banks": [140000.0, 100000.0], "scores": [0.55, 0.45]}
    ]
    with pytest.raises(AssertionError, match="win rate .* breached acceptance threshold"):
        evaluate_tournament_acceptance(summary_low_win, results_low_win)

    # Case B: Low Terminal Cash ($115,000 < $130,000)
    summary_low_cash = {
        "focus": {"agent": "candidate", "overall": {"score": 0.65}},
        "standings": [{"agent": "candidate", "score": 0.65, "act_ms_mean": 1.5, "errors": 0}],
    }
    results_low_cash = [
        {"agents": ["candidate", "ghost"], "roles": ["candidate", "panel"], "banks": [115000.0, 80000.0], "scores": [1.0, 0.0]}
    ]
    with pytest.raises(AssertionError, match="terminal cash .* breached acceptance threshold"):
        evaluate_tournament_acceptance(summary_low_cash, results_low_cash)

    # Case C: Latency Violation (6.2ms >= 5.0ms)
    summary_high_latency = {
        "focus": {"agent": "candidate", "overall": {"score": 0.65}},
        "standings": [{"agent": "candidate", "score": 0.65, "act_ms_mean": 6.2, "errors": 0}],
    }
    results_high_latency = [
        {"agents": ["candidate", "ghost"], "roles": ["candidate", "panel"], "banks": [140000.0, 80000.0], "scores": [1.0, 0.0]}
    ]
    with pytest.raises(AssertionError, match="mean turn latency .* breached limit"):
        evaluate_tournament_acceptance(summary_high_latency, results_high_latency)


def test_tier4_live_tournament_acceptance_gate():
    """Tier 4: Live verification of gauntlet_1000 tournament results if present."""
    gauntlet_dir = ROOT / "tournaments" / "gauntlet_1000"
    summary_file = gauntlet_dir / "summary.json"
    results_file = gauntlet_dir / "results.jsonl"

    assert summary_file.is_file(), f"Missing tournament summary: {summary_file}"
    assert results_file.is_file(), f"Missing tournament results: {results_file}"

    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    results = [
        json.loads(line)
        for line in results_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    report = evaluate_tournament_acceptance(summary, results)
    assert report["gate_passed"] is True


def test_tier4_autopsy_pipeline_loss_validation():
    """Tier 4: Verify loss autopsy pipeline extracts crossover turn and loss catalysts."""
    replay_file = ROOT / "replays" / "live_losses" / "episode-114793445-replay.json"
    if not replay_file.is_file():
        pytest.skip(f"Loss replay {replay_file} not found")

    analysis = analyze_loss_match(replay_file, player_query="Pranav")

    assert analysis["match_metadata"]["match_result"] == "LOSS"
    assert analysis["crossover_analysis"]["crossover_turn"] > 0
    assert analysis["crossover_analysis"]["crossover_day"] >= 0
    assert "opponent_macro_milestones" in analysis
    assert analysis["opponent_macro_milestones"]["peak_herd_size"] >= 0


def test_tier4_optuna_hpo_loop_validation(tmp_path: Path):
    """Tier 4: Verify Optuna HPO parameter sampling and fallback tuning execution."""
    # 1. Hyperparameter search space bounds check
    class DummyTrial:
        def suggest_float(self, name: str, low: float, high: float, log: bool = False):
            return low

        def suggest_int(self, name: str, low: int, high: int):
            return low

    weights = sample_beam_weights(DummyTrial())
    required_weights = {
        "milk_prior_scale",
        "berry_prior_scale",
        "poison_prior_scale",
        "holding_penalty_mult",
        "milk_sell_margin",
        "berry_sell_margin",
        "top_k_intents",
    }
    assert required_weights.issubset(set(weights.keys()))

    # 2. Fast 2-trial mock HPO execution check
    ghost_spec = {
        "name": "ladder_ghost_mock",
        "type": "python",
        "path": str(ROOT / "submissions" / "ladder_ghost_112542379" / "main.py"),
    }

    def mock_evaluator(w: Dict[str, Any], trial_no: int) -> float:
        # Mock score function favoring higher prior scales
        return 0.5 + 0.1 * min(1.0, float(w.get("milk_prior_scale", 1.0)))

    hpo_res = run_hpo(
        ghost_spec=ghost_spec,
        out_dir=tmp_path / "hpo_test",
        n_trials=2,
        trial_seeds=[1, 2],
        workers=1,
        kagg=None,
        optuna_seed=42,
        evaluator=mock_evaluator,
    )

    assert "best_value" in hpo_res
    assert hpo_res["best_value"] >= 0.5
    assert (tmp_path / "hpo_test" / "hpo_summary.json").is_file()
    assert (tmp_path / "hpo_test" / "counter_best" / "main.py").is_file()
