"""Tests for phase inference and transition logging."""

from kaggriculture.helpers.phase_log import (
    Phase,
    PhaseLog,
    infer_phase,
    log_transition,
    start_phase_log,
    clear_phase_log,
)


def _obs(day: int, hour: int = 0, step: int = 0, money: float = 3000.0, shed: int = 0):
    return {
        "step": step,
        "day": day,
        "hour": hour,
        "player": 0,
        "farms": [{"money": money, "tiles": []}],
        "private": {"shed": {"WHEAT": shed}},
        "market": {"inventory": {"MILK": 10000}, "prices": {"MILK": 160}},
    }


def test_infer_phase_by_day():
    assert infer_phase(_obs(0)) == Phase.OPENING
    assert infer_phase(_obs(5)) == Phase.EXPANSION
    assert infer_phase(_obs(15)) == Phase.ADAPTIVE_MID
    assert infer_phase(_obs(28)) == Phase.TERMINAL


def test_log_transition_dedupes():
    clear_phase_log()
    log = start_phase_log()
    obs = _obs(0, step=1)
    log_transition(obs, Phase.OPENING)
    log_transition(obs, Phase.OPENING)
    log_transition(_obs(5, step=120), Phase.EXPANSION)
    assert len(log.transitions) == 2
    clear_phase_log()
