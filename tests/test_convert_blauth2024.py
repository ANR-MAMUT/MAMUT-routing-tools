"""Tests for the Blauth2024 upstream converter (synthetic mini-upstream)."""

from __future__ import annotations

import bz2
import gzip
import json

import pytest

from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.models import BenchmarkSolution
from mamut_routing_lib.td import check_td_solution, load_td_instance

from mamut_routing_tools.conversion.blauth2024 import (
    DEPOT_DUE_DATE,
    FLEET_FIXED_COST,
    HORIZON_END,
    HORIZON_START,
    Blauth2024ConversionError,
    convert_instance,
)

TRAVEL = 600_000  # 10 min constant travel on every synthetic arc


def _identity_atf() -> dict:
    return {
        "atf_leave_time": [0, HORIZON_END],
        "atf_arrive_time": [0, HORIZON_END],
        "atf_dist_end_time": [HORIZON_END],
        "atf_distance": [0],
    }


def _constant_atf(travel: int = TRAVEL) -> dict:
    return {
        "atf_leave_time": [HORIZON_START, HORIZON_END],
        "atf_arrive_time": [HORIZON_START + travel, HORIZON_END + travel],
        "atf_dist_end_time": [HORIZON_END],
        "atf_distance": [5000],
    }


def write_upstream(tmp_path, *, n=2, mutate=None):
    """Write a minimal upstream checkout for city 'toyville'; return its root."""
    root = tmp_path / "upstream"
    (root / "instances").mkdir(parents=True)
    items = {
        str(k): {
            "earliest_delivery": 55_800_000,
            "latest_delivery": 75_600_000,
            "latitude": 45.0 + k,
            "longitude": 4.0 + k,
        }
        for k in range(1, n + 1)
    }
    instance = {"depot": {"latitude": 45.0, "longitude": 4.0}, "items": items}
    vertices = ["depot"] + [str(k) for k in range(1, n + 1)]
    entries = []
    for source in vertices:
        for target in vertices:
            atf = _identity_atf() if source == target else _constant_atf()
            entries.append({"from": source, "to": target, "atf": atf})
    if mutate is not None:
        mutate(instance, entries)
    (root / "instances" / f"toyville_{n}.json").write_text(json.dumps(instance))
    with bz2.open(root / "instances" / f"toyville_{n}_tt.json.bz2", "wt") as handle:
        json.dump(entries, handle)
    return root


def test_convert_round_trips_and_prices_exactly(tmp_path):
    root = write_upstream(tmp_path)
    result = convert_instance(root, tmp_path / "family", "toyville", 2, upstream_commit="deadbeef")

    assert result.instance_name == "Blauth-toyville"
    assert result.instance_path.name == "Blauth-toyville.vrp.json"
    assert result.atf_path.name == "Blauth-toyville.atf.json.gz"
    assert result.max_arrival == HORIZON_END + TRAVEL

    loaded = load_td_instance(result.instance_path)  # verifies atf_sha256
    instance = loaded.instance
    assert instance.num_vehicles is None
    assert instance.vehicle_capacity == 2
    assert instance.demands == [0, 1, 1]
    assert instance.service_times == [0, 180_000, 180_000]
    assert instance.fleet_fixed_cost == FLEET_FIXED_COST
    assert tuple(instance.horizon) == (HORIZON_START, HORIZON_END)
    assert tuple(instance.time_windows[0]) == (HORIZON_START, DEPOT_DUE_DATE)
    assert instance.metadata["generator"]["upstream_commit"] == "deadbeef"

    # Constant arcs: each singleton route costs travel + service + travel.
    solution = BenchmarkSolution(instance_name=instance.instance_name, routes=[[1], [2]])
    check = check_td_solution(loaded, solution, ObjectiveFunction.FLEET_COST_DURATION)
    assert check.is_valid()
    per_route = 2 * TRAVEL + 180_000
    assert check.routing_cost == 2 * per_route + 2 * FLEET_FIXED_COST


def test_conversion_is_deterministic(tmp_path):
    root = write_upstream(tmp_path)
    first = convert_instance(root, tmp_path / "a", "toyville", 2)
    second = convert_instance(root, tmp_path / "b", "toyville", 2)
    assert first.atf_sha256 == second.atf_sha256
    assert first.instance_path.read_bytes() == second.instance_path.read_bytes()
    assert first.atf_path.read_bytes() == second.atf_path.read_bytes()
    # The gz payload really is the canonical bytes (mtime=0 gzip).
    assert gzip.decompress(first.atf_path.read_bytes()) == gzip.decompress(second.atf_path.read_bytes())


def _expect_gate_failure(tmp_path, mutate, match):
    root = write_upstream(tmp_path, mutate=mutate)
    with pytest.raises(Blauth2024ConversionError, match=match):
        convert_instance(root, tmp_path / "family", "toyville", 2)


def test_co_located_pair_becomes_zero_travel_arc(tmp_path):
    # Upstream encodes duplicate-address pairs as the full identity (like
    # self-arcs); the converter keeps them as horizon-restricted zero travel.
    def mutate(instance, entries):
        instance["items"]["2"]["latitude"] = instance["items"]["1"]["latitude"]
        instance["items"]["2"]["longitude"] = instance["items"]["1"]["longitude"]
        for entry in entries:
            if {entry["from"], entry["to"]} == {"1", "2"}:
                entry["atf"] = _identity_atf()

    root = write_upstream(tmp_path, mutate=mutate)
    result = convert_instance(root, tmp_path / "family", "toyville", 2)
    loaded = load_td_instance(result.instance_path)
    assert loaded.instance.metadata["co_located_arcs"] == 2
    arc = loaded.atfs.arcs[(1, 2)]
    assert list(arc.xs) == [HORIZON_START, HORIZON_END]
    assert list(arc.ys) == [HORIZON_START, HORIZON_END]
    # Zero travel between the pair: route [1, 2] costs travel + 2 services + travel.
    solution = BenchmarkSolution(instance_name=loaded.instance.instance_name, routes=[[1, 2]])
    check = check_td_solution(loaded, solution, ObjectiveFunction.FLEET_COST_DURATION)
    assert check.is_valid()
    assert check.routing_cost == 2 * TRAVEL + 2 * 180_000 + FLEET_FIXED_COST


def test_gate_zero_start_non_identity_rejected(tmp_path):
    def mutate(_instance, entries):
        for entry in entries:
            if entry["from"] == "1" and entry["to"] == "2":
                entry["atf"]["atf_leave_time"] = [0, HORIZON_END]
                entry["atf"]["atf_arrive_time"] = [1000, HORIZON_END + 1000]

    _expect_gate_failure(tmp_path, mutate, "co-located identity")


def test_short_domain_arc_gets_constant_travel_extension(tmp_path):
    # Upstream guarantees coverage only past 21:33; a short arc is completed
    # to 22:00 at constant travel time (one integer breakpoint, dead region).
    short_end = HORIZON_END - 74_195

    def mutate(_instance, entries):
        for entry in entries:
            if entry["from"] == "1" and entry["to"] == "2":
                entry["atf"]["atf_leave_time"] = [HORIZON_START, short_end]
                entry["atf"]["atf_arrive_time"] = [HORIZON_START + TRAVEL, short_end + TRAVEL]

    root = write_upstream(tmp_path, mutate=mutate)
    result = convert_instance(root, tmp_path / "family", "toyville", 2)
    loaded = load_td_instance(result.instance_path)
    assert loaded.instance.metadata["domain_extended_arcs"] == 1
    arc = loaded.atfs.arcs[(1, 2)]
    assert list(arc.xs) == [HORIZON_START, short_end, HORIZON_END]
    assert list(arc.ys) == [HORIZON_START + TRAVEL, short_end + TRAVEL, HORIZON_END + TRAVEL]


def test_gate_domain_end_before_2103_rejected(tmp_path):
    def mutate(_instance, entries):
        for entry in entries:
            if entry["from"] == "1" and entry["to"] == "2":
                entry["atf"]["atf_leave_time"] = [HORIZON_START, 75_000_000]
                entry["atf"]["atf_arrive_time"] = [HORIZON_START + TRAVEL, 75_000_000 + TRAVEL]

    _expect_gate_failure(tmp_path, mutate, "extension-safety")


def test_gate_fifo_violation(tmp_path):
    def mutate(_instance, entries):
        for entry in entries:
            if entry["from"] == "1" and entry["to"] == "2":
                entry["atf"]["atf_arrive_time"] = [70_000_000, 69_000_000]

    _expect_gate_failure(tmp_path, mutate, "FIFO violation")


def test_gate_domain_mismatch(tmp_path):
    def mutate(_instance, entries):
        for entry in entries:
            if entry["from"] == "1" and entry["to"] == "2":
                entry["atf"]["atf_leave_time"] = [HORIZON_START + 1, HORIZON_END]

    _expect_gate_failure(tmp_path, mutate, "domain")


def test_gate_self_arc_identity(tmp_path):
    def mutate(_instance, entries):
        for entry in entries:
            if entry["from"] == "1" and entry["to"] == "1":
                entry["atf"]["atf_arrive_time"] = [0, HORIZON_END + 1]

    _expect_gate_failure(tmp_path, mutate, "self-arc")


def test_gate_midnight_reachable_by_feasible_return(tmp_path):
    # A return arc arriving at/after midnight when departing at 21:03 breaks
    # the non-binding-L proof and must be refused.
    def mutate(_instance, entries):
        for entry in entries:
            if entry["from"] == "2" and entry["to"] == "depot":
                entry["atf"]["atf_arrive_time"] = [DEPOT_DUE_DATE, DEPOT_DUE_DATE + 1]

    _expect_gate_failure(tmp_path, mutate, "midnight")


def test_dead_region_extension_past_midnight_is_allowed(tmp_path):
    # A slow short return arc whose constant-travel extension lands past
    # midnight is fine: the extension lives in the dead post-21:03 region and
    # the exact 21:03 evaluation stays far below midnight.
    short_end = HORIZON_END - 74_195
    slow = 7_500_000  # ~2 h 5 min

    def mutate(_instance, entries):
        for entry in entries:
            if entry["from"] == "2" and entry["to"] == "depot":
                entry["atf"]["atf_leave_time"] = [HORIZON_START, short_end]
                entry["atf"]["atf_arrive_time"] = [HORIZON_START + slow, short_end + slow]

    root = write_upstream(tmp_path, mutate=mutate)
    result = convert_instance(root, tmp_path / "family", "toyville", 2)
    assert result.max_arrival == HORIZON_END + slow  # past midnight, dead region
    assert result.max_arrival > DEPOT_DUE_DATE
    loaded = load_td_instance(result.instance_path)
    meta = loaded.instance.metadata["depot_due_date"]
    assert meta["max_feasible_return_ceil"] < DEPOT_DUE_DATE


def test_gate_non_integer_values(tmp_path):
    def mutate(_instance, entries):
        for entry in entries:
            if entry["from"] == "1" and entry["to"] == "2":
                entry["atf"]["atf_arrive_time"] = [54_600_000.5, HORIZON_END + TRAVEL]

    _expect_gate_failure(tmp_path, mutate, "non-integer")


def test_gate_bad_item_ids(tmp_path):
    def mutate(instance, _entries):
        instance["items"]["7"] = instance["items"].pop("2")

    _expect_gate_failure(tmp_path, mutate, "item ids")


def test_gate_tw_outside_horizon(tmp_path):
    def mutate(instance, _entries):
        instance["items"]["1"]["latest_delivery"] = HORIZON_END + 1

    _expect_gate_failure(tmp_path, mutate, "time window")
