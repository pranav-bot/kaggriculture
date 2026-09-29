"""Tests for the combinatorial strawberry scheduler."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kaggriculture.scheduling import (
    WORKERS_PER_HOUR,
    batch_daily_ops,
    build_schedule,
    harvest_days,
    optimize_stagger,
)


def test_lifecycle_constants_match_engine():
    assert harvest_days(2) == [12, 14, 16, 18]
    assert batch_daily_ops(5, -1) == {}
    assert batch_daily_ops(5, 6) == {"WATER": 5, "FERTILIZE": 5}
    assert batch_daily_ops(5, 10) == {"WATER": 5, "HARVEST": 5}
    assert batch_daily_ops(5, 17) == {"DIG": 5, "PLANT": 5}


def test_simultaneous_planting_spikes_harvest_day():
    sched = build_schedule([(2, 14)])
    assert sched.peak_harvest_day == 14  # all 14 collide every harvest day


def test_staggered_plan_beats_simultaneous_and_proves_fit():
    plan = optimize_stagger(n_tiles=14, herd_size=18)
    assert sum(s for _, s in plan.batches) == 14
    assert plan.peak_harvest_day < 14  # batches interleave, never all collide
    proof = plan.verify()
    assert proof["within_bandwidth"] and proof["no_harvest_overlap"]
    assert proof["feeding_fits"]  # 36 dawn ops into 4x9 block
    assert plan.peak_load <= WORKERS_PER_HOUR


def test_no_feeding_overlap_slot_by_slot():
    plan = optimize_stagger()
    for (day, hour), ops in plan.slots.items():
        assert len(ops) <= WORKERS_PER_HOUR
        if hour in (0, 1, 2, 3):
            assert "HARVEST" not in ops


def test_routing_interface_serializable():
    plan = optimize_stagger()
    d = plan.to_dict()
    assert len(d["batches"]) == 3 and d["peak_load"] <= WORKERS_PER_HOUR
    assert isinstance(plan.jobs_for_slot(12, 4), list)


def test_infeasible_plan_rejected():
    with pytest.raises(ValueError):
        build_schedule([(2, 14)], herd_size=200)  # dawn block cannot fit
