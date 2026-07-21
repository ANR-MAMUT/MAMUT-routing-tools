"""Generation port tests: demand bands, capacity formula, TW feasibility,
writer round-trips, and the fixture-city end-to-end pipeline."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from mamut_routing_tools.generation.demands import (
    avg_route_size_bounds,
    capacity_from_avg_route_size,
    demand_distribution_bounds,
    generate_demands,
)
from mamut_routing_tools.generation.single import GenerationRequest, build_generation_selection, generate_single_instance
from mamut_routing_tools.generation.vrptw import (
    HORIZON_END,
    HORIZON_START,
    derive_vrptw_from_cvrp,
    generate_vrptw_fields,
    repair_time_window,
    stable_seed,
)
from mamut_routing_tools.generation.writers import parse_cvrp_vrp


@pytest.mark.parametrize("demand_type", range(1, 8))
def test_demands_respect_type_bands(demand_type: int) -> None:
    rng = random.Random(11)
    customer_ll = [(45.0 + i * 0.001, 4.0 + (i % 7) * 0.001) for i in range(60)]
    demands, total, max_demand, r = generate_demands(rng, customer_ll, demand_type, 4)
    assert len(demands) == 60 and total == sum(demands) and max_demand == max(demands)
    rlo, rhi = avg_route_size_bounds(4)
    assert rlo <= r <= rhi
    if demand_type == 1:
        assert set(demands) == {1}
    elif demand_type in (6, 7):
        assert all(1 <= d <= 100 for d in demands)
    else:
        lo, hi = demand_distribution_bounds(demand_type)
        assert all(lo <= d <= hi for d in demands)


def test_capacity_formula_bounds() -> None:
    demands = [3, 8, 5, 9, 2, 7]
    capacity = capacity_from_avg_route_size(4.0, demands)
    assert max(demands) <= capacity <= sum(demands) - 1
    # Unit demands: capacity is the route size itself.
    assert capacity_from_avg_route_size(4.7, [1] * 10) == 4


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"cluster_seeds": 0}, "cluster_seeds"),
        ({"cluster_decay_meters": 0}, "cluster_decay_meters"),
        ({"hybrid_poi_share": -0.1}, "hybrid_poi_share"),
        ({"hybrid_poi_share": 1.1}, "hybrid_poi_share"),
    ],
)
def test_generation_request_rejects_invalid_parametric_controls(
    fixture_osm_path: Path,
    overrides: dict,
    message: str,
) -> None:
    request = GenerationRequest(city="Testville", osm_path=fixture_osm_path, **overrides)

    with pytest.raises(ValueError, match=message):
        request.validate()


def test_repair_time_window_keeps_depot_roundtrip_feasible() -> None:
    e, latest = repair_time_window(100, 200, 400, 300, 50, 0, 1000)
    assert e >= 400 and latest <= 1000 - 50 - 300 and e <= latest
    # Infeasible customer collapses to a clamped point window.
    e2, l2 = repair_time_window(0, 86400, 50000, 50000, 4000, 0, 86400)
    assert e2 == l2


@pytest.mark.parametrize("tw_method", ["route_centered", "reachable_interval"])
def test_vrptw_fields_are_deterministic_and_feasible(tw_method: str) -> None:
    rng = random.Random(3)
    n = 12
    travel = [[0 if i == j else rng.randint(60, 1800) for j in range(n)] for i in range(n)]
    seed_parts = ("base", "place", 0, tw_method, HORIZON_START, HORIZON_END, "v1")
    service_a, windows_a, params_a = generate_vrptw_fields(seed_parts, travel, HORIZON_START, HORIZON_END, tw_method)
    service_b, windows_b, _params_b = generate_vrptw_fields(seed_parts, travel, HORIZON_START, HORIZON_END, tw_method)
    assert service_a == service_b and windows_a == windows_b
    assert windows_a[0] == (HORIZON_START, HORIZON_END)
    for i in range(1, n):
        e, latest = windows_a[i]
        assert HORIZON_START <= e <= latest <= HORIZON_END
        # Every window is reachable from the depot and allows the return trip.
        assert e >= HORIZON_START + travel[0][i] or e == latest
        assert latest <= HORIZON_END - service_a[i] - travel[i][0] or e == latest
    assert params_a["tw_method"] == tw_method


def test_stable_seed_is_deterministic_and_sensitive() -> None:
    assert stable_seed("lyon", 10, 7) == stable_seed("lyon", 10, 7)
    assert stable_seed("lyon", 10, 7) != stable_seed("lyon", 10, 8)


def test_fixture_city_generation_end_to_end(tmp_path: Path, fixture_osm_path: Path) -> None:
    request = GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method="parametric_attach",
        n_customers=4,
        seed=7,
    )
    # Selection is deterministic per seed.
    first = build_generation_selection(request).vertices
    second = build_generation_selection(request).vertices
    assert first == second

    result = generate_single_instance(request, tmp_path / "instances")
    folder = Path(result["folder"])
    base = result["base_name"]
    assert result["summary"]["customers"] == 4

    parsed = parse_cvrp_vrp(folder / f"{base}_fastest.vrp")
    assert parsed.dimension == 5 and parsed.capacity == result["summary"]["capacity"]
    assert parsed.arc_costs[0][0] == 0 and parsed.depot_node_index == 1

    meta = json.loads((folder / f"{base}_meta.json").read_text())
    assert meta["schema_version"] == 2 and meta["depot_instance_node_id"] == 1
    assert len(meta["nodes"]) == 5 and set(meta["road_cache"]) == {"shortest", "fastest"}
    for key in meta["road_cache"]["fastest"]:
        u, v = key.split("_")
        vertex_ids = {str(node["graph_vertex_id"]) for node in meta["nodes"]}
        assert u in vertex_ids and v in vertex_ids

    derived = derive_vrptw_from_cvrp(folder, base, place_slug="testville", source_seed=7)
    vrptw_text = (folder / derived["vrptw_file"]).read_text()
    assert "TYPE : CVRPTW" in vrptw_text and "TIME_WINDOW_SECTION" in vrptw_text
