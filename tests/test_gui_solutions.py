"""Focused contracts for persisted GUI solutions and comparisons."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mamut_routing_tools.gui.solutions import SolutionStore, compare_solution_records


def test_comparison_rejects_runs_from_different_metrics() -> None:
    instance = SimpleNamespace(depot=0, demands=[0, 1, 1])
    candidate = {
        "run_id": "candidate",
        "objective_function": "MonoCost",
        "cost": 10,
        "routes": [[1, 2]],
        "validation": {"valid": True},
        "metadata": {"metric": "fastest"},
    }
    reference = {
        **candidate,
        "run_id": "reference",
        "metadata": {"metric": "shortest"},
    }

    with pytest.raises(ValueError, match="same metric"):
        compare_solution_records(instance, candidate, reference)


def test_solution_store_rejects_non_run_identifiers(tmp_path: Path) -> None:
    store = SolutionStore(tmp_path)

    with pytest.raises(KeyError):
        store.get("instance", "../../state/jobs/a-job")


def _cost_instance(matrix):
    from mamut_routing_lib.models import BenchmarkInstanceCVRP

    n = len(matrix) - 1
    return BenchmarkInstanceCVRP(
        instance_name="compare", benchmark_name="Dimacs2021", instance_origin="Solomon1987",
        num_customers=n, vehicle_capacity=n, demands=[0] + [1] * n,
        coordinates=[(x, 0) for x in range(n + 1)], arc_costs=matrix,
    )


def test_comparison_reprices_legacy_runs_with_tied_display_costs() -> None:
    high = 2**53
    instance = _cost_instance([[0, high, high], [0, 0, 0], [0, 0.5, 0]])
    candidate = {"routes": [[1, 2]], "cost": float(high), "objective_function": "MonoCost"}
    reference = {**candidate, "routes": [[2, 1]]}
    comparison = compare_solution_records(instance, candidate, reference)
    assert comparison["profile"] == "binary64-exact-v1"
    assert comparison["candidate"]["cost"] == comparison["reference"]["cost"] == high
    assert comparison["reference"]["authoritative_cost"] == {"numerator": str(2 * high + 1), "denominator": "2"}
    assert comparison["ordering"] == "better"
    assert comparison["cost_delta"] == -0.5
    assert comparison["relative_gap_percent"] < 0


def test_fleet_priority_precedes_cost_in_the_workbench() -> None:
    from mamut_routing_lib.models import BenchmarkInstanceVRPTW

    static = _cost_instance([[0, 1, 1], [1, 0, 10], [1, 10, 0]])
    instance = BenchmarkInstanceVRPTW(**static.model_dump(), time_windows=[(0, 100)] * 3, service_times=[0] * 3)
    one_route = {"routes": [[1, 2]], "objective_function": "HierarchicalVehicleCost"}
    two_routes = {**one_route, "routes": [[1], [2]]}
    comparison = compare_solution_records(instance, one_route, two_routes)
    assert comparison["ordering"] == "better"
    assert comparison["cost_delta"] == 8
    assert comparison["route_count_delta"] == -1


def test_comparison_accepts_routes_whose_display_overflows() -> None:
    high = sys.float_info.max
    instance = _cost_instance([[0, high], [high, 0]])
    run = {"routes": [[1]], "cost": None, "objective_function": "MonoCost"}
    comparison = compare_solution_records(instance, run, run)
    assert comparison["candidate"]["cost"] is None
    assert comparison["ordering"] == "equal"
    assert comparison["cost_delta"] == comparison["relative_gap_percent"] == 0


def test_comparison_rejects_incompatible_or_unavailable_inputs(tmp_path: Path) -> None:
    instance = _cost_instance([[0, 1], [1, 0]])
    run = {"routes": [[1]], "objective_function": "MonoCost", "instance_path": str(tmp_path / "a.vrp.json")}
    with pytest.raises(ValueError, match="same objective"):
        compare_solution_records(instance, run, {**run, "objective_function": "Duration"})
    with pytest.raises(ValueError, match="same current instance file"):
        compare_solution_records(instance, run, {**run, "instance_path": str(tmp_path / "b.vrp.json")}, instance_path=Path(run["instance_path"]))
    with pytest.raises(FileNotFoundError):
        compare_solution_records(instance, run, run, instance_path=Path(run["instance_path"]))
    with pytest.raises(ValueError, match="instance edition"):
        compare_solution_records(instance, run, {**run, "metadata": {"standard": {"manifest_sha256": "ab" * 32}}})
