"""Generation port tests: demand bands, capacity formula, TW feasibility,
writer round-trips, and the fixture-city end-to-end pipeline."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from mamut_routing_tools.generation.bulk import (
    BulkRow,
    generate_bulk_from_rows,
    generate_bulk_instances,
    preflight_rows,
)
from mamut_routing_tools.generation.demands import (
    avg_route_size_bounds,
    capacity_from_avg_route_size,
    demand_distribution_bounds,
    generate_demands,
)
from mamut_routing_tools.generation.pois import POI_CATEGORIES, NoPoiFoundError, find_pois
from mamut_routing_tools.generation.select import POI_ATTACH_NEAREST_VERTEX
from mamut_routing_tools.generation.single import (
    GenerationRequest,
    build_generation_selection,
    composition_notice,
    generate_single_instance,
)
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


def test_capacity_never_clamps_into_a_two_route_instance() -> None:
    """A route-size target larger than n must not become ``Q = total - 1``.

    That clamp used to sit at the end of the formula to "guarantee two routes".
    What it actually guaranteed was a TSP with a cosmetic second route holding
    the overflow -- five published v2 instances had best known solutions of
    shape [99, 1] or [147, 1]. Capacity now answers only to the route-size
    target; whether the configuration is a genuine VRP is the campaign's
    admissibility rule to decide, where it is visible.
    """
    demands = [5] * 100
    total = sum(demands)
    # r far beyond n: the old code returned total - 1 and forced K = 2.
    assert capacity_from_avg_route_size(180.0, demands) > total
    # A sane target is unaffected, and stays well clear of the old clamp.
    assert capacity_from_avg_route_size(10.0, demands) == 50


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
    assert meta["schema_version"] == 3 and meta["depot_instance_node_id"] == 1
    assert len(meta["nodes"]) == 5 and set(meta["road_cache"]) == {"shortest", "fastest"}
    # v3 nodes always carry the POI columns; parametric customers leave them null.
    for node in meta["nodes"]:
        assert {"poi_category", "poi_osm_id", "poi_name", "snap_distance_m"} <= set(node)
    for key in meta["road_cache"]["fastest"]:
        u, v = key.split("_")
        vertex_ids = {str(node["graph_vertex_id"]) for node in meta["nodes"]}
        assert u in vertex_ids and v in vertex_ids

    derived = derive_vrptw_from_cvrp(folder, base, place_slug="testville", source_seed=7)
    vrptw_text = (folder / derived["vrptw_file"]).read_text()
    assert "TYPE : CVRPTW" in vrptw_text and "TIME_WINDOW_SECTION" in vrptw_text


def test_find_pois_keeps_names_and_osm_ids(fixture_osm_path: Path) -> None:
    pois = {poi.osm_id: poi for poi in find_pois(fixture_osm_path, list(POI_CATEGORIES))}
    assert pois[1].category == "restaurant" and pois[1].name == "Chez Un"
    assert pois[2].name == "Café Deux"
    # An amenity without a name tag stays usable, just anonymous.
    assert pois[3].category == "pharmacy" and pois[3].name is None
    # Filtering by category must not leak other amenities.
    only_cafes = find_pois(fixture_osm_path, ["cafe"])
    assert [poi.osm_id for poi in only_cafes] == [2]


def test_manual_selection_snaps_picks_and_reports_them(fixture_osm_path: Path) -> None:
    request = GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method="manual",
        # Node 9 is the far-away museum: it must be reported, not silently kept.
        manual_poi_ids=[2, 3, 4, 9],
        manual_depot_poi_id=1,
        seed=7,
    )
    selection = build_generation_selection(request)
    report = selection.params["manual_selection"]
    assert report["unreachable_poi_ids"] == [9]
    assert report["depot_poi_osm_id"] == 1 and report["resolved"] == 3

    assert selection.source_tags == ["depot", "poi_manual", "poi_manual", "poi_manual"]
    names = [poi.name if poi else None for poi in selection.poi_meta]
    assert names == ["Chez Un", "Café Deux", None, "École Quatre"]
    # These POIs sit exactly on road nodes, so snapping is a no-op.
    assert all(distance < 1.0 for distance in selection.snap_distances_m)


def test_poi_shortfall_is_reported_instead_of_silently_filled(fixture_osm_path: Path) -> None:
    # Only the cafe of node 2 matches, so 3 of the 4 customers must fall back to
    # parametric road points -- the case the GUI has to warn about. The depot is
    # taken off-center so it does not claim the single matching POI itself.
    request = GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method="poi_categories",
        n_customers=4,
        categories=["cafe"],
        depot_mode="corner",
        seed=5,
    )
    with pytest.warns(UserWarning, match="POI-attached"):
        selection = build_generation_selection(request)
    composition = selection.params["composition"]
    assert composition["poi_customers"] == 1
    assert composition["parametric_customers"] == 3
    assert composition["poi_target"] == 4 and composition["poi_shortfall"] == 3
    # The whole pool was scanned, so 1 is the ceiling, not just where it stopped.
    assert composition["poi_pool_matching"] == 1 and composition["poi_pool_attachable"] == 1
    assert composition["category_count"] == 1

    notice = composition_notice(composition)
    assert notice is not None
    assert "3 are parametric road points" in notice
    assert "reduce the customer count to 1" in notice
    assert "select more POI categories (1 of 49 selected)" in notice


def test_no_notice_when_every_customer_sits_on_a_poi(fixture_osm_path: Path) -> None:
    request = GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method="poi_categories",
        n_customers=2,
        categories=["restaurant", "cafe", "school", "bar"],
        depot_mode="corner",
        seed=5,
    )
    selection = build_generation_selection(request)
    composition = selection.params["composition"]
    assert composition["poi_shortfall"] == 0 and composition["parametric_customers"] == 0
    # The whole pool is walked even once the request is served, so the ceiling
    # is reported for what it is: 4 matching POIs on 4 distinct vertices.
    assert composition["poi_pool_matching"] == 4
    assert composition["poi_pool_attachable"] == 4
    assert composition_notice(composition) is None


def test_hybrid_shortfall_advises_on_the_share_as_well(fixture_osm_path: Path) -> None:
    request = GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method="hybrid",
        n_customers=4,
        categories=["cafe"],
        hybrid_poi_share=0.5,
        depot_mode="corner",
        seed=5,
    )
    with pytest.warns(UserWarning):
        selection = build_generation_selection(request)
    composition = selection.params["composition"]
    assert composition["poi_target"] == 2 and composition["poi_customers"] == 1
    notice = composition_notice(composition)
    assert notice is not None and "lower the POI share" in notice


def test_manual_notice_names_the_picks_that_could_not_be_used(fixture_osm_path: Path) -> None:
    request = GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method="manual",
        # Node 9 is the far-away museum; node 1 doubles as the depot pick.
        manual_poi_ids=[2, 3, 4, 9],
        manual_depot_poi_id=1,
        seed=7,
    )
    composition = build_generation_selection(request).params["composition"]
    assert composition["requested"] == 4 and composition["delivered"] == 3
    assert composition["dropped"]["unreachable"] == 1
    notice = composition_notice(composition)
    assert notice is not None
    assert "1 of your 4 picked POIs cannot become customers" in notice
    assert "too far from any road" in notice


def test_manual_selection_rejects_too_few_resolvable_picks(fixture_osm_path: Path) -> None:
    request = GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method="manual",
        manual_poi_ids=[2, 9],
        manual_depot_poi_id=1,
        seed=7,
    )
    with pytest.raises(ValueError, match="fewer than 2 customers"):
        build_generation_selection(request)


def test_bulk_rows_share_a_pool_and_honour_per_row_settings(
    tmp_path: Path, fixture_osm_path: Path
) -> None:
    def row(n: int, demand_type: int, band: int, seed: int, problem: str = "cvrp") -> BulkRow:
        return BulkRow(
            request=GenerationRequest(
                city="Testville",
                osm_path=fixture_osm_path,
                method="parametric_attach",
                n_customers=n,
                demand_type=demand_type,
                avg_route_size=band,
            ),
            problem_type=problem,
            explicit_seed=seed,
        )

    calls: list[str] = []

    class Recorder:
        def progress(self, message: str, *, current: int | None = None, total: int | None = None) -> None:
            calls.append(message)

        def check_cancelled(self) -> None:
            return None

    rows = [row(3, 1, 1, 101), row(4, 5, 2, 102, problem="vrptw"), row(3, 1, 1, 103)]
    result = generate_bulk_from_rows(
        rows, output_root=tmp_path / "instances", base_seed=0, context=Recorder()
    )
    assert result["generated"] == 3
    # One selection group -> one pool build, whatever the per-row differences.
    assert sum(1 for message in calls if "customer pool" in message) == 1
    assert len(result["city_reports"]) == 1

    seeds, demand_types = [], []
    for entry in result["results"]:
        meta = json.loads(
            (Path(entry["folder"]) / f"{entry['base_name']}_meta.json").read_text(encoding="utf-8")
        )
        seeds.append(meta["generation_params"]["seed"])
        demand_types.append(meta["generation_params"]["demand_type"])
    assert seeds == [101, 102, 103] and demand_types == [1, 5, 1]
    # Only the row that asked for VRPTW gets a twin.
    assert [bool(entry.get("vrptw")) for entry in result["results"]] == [False, True, False]

    # Rows 1 and 3 differ only by seed, so they must not share a file name; the
    # row with a name of its own keeps the plain convention.
    names = [entry["base_name"] for entry in result["results"]]
    assert len(set(names)) == 3
    assert names[0].endswith("-s101") and names[2].endswith("-s103")
    assert "-s" not in names[1]


def test_cartesian_bulk_derives_vrptw_twins_when_asked(
    tmp_path: Path, fixture_osm_path: Path
) -> None:
    """Regression: bulk used to drop the problem type and emit CVRP only."""
    base = GenerationRequest(
        city="Testville", osm_path=fixture_osm_path, method="parametric_attach", seed=5
    )
    result = generate_bulk_instances(
        base,
        cities=[("Testville", fixture_osm_path)],
        n_list=[3],
        demand_types=[2],
        avg_route_sizes=[1],
        output_root=tmp_path / "instances",
        problem_type="vrptw",
    )
    assert result["generated"] == 1
    twin = result["results"][0]["vrptw"]
    assert "TYPE : CVRPTW" in (
        Path(result["results"][0]["folder"]) / twin["vrptw_file"]
    ).read_text(encoding="utf-8")


def test_preflight_reports_sizes_the_pool_cannot_serve(fixture_osm_path: Path) -> None:
    rows = [
        BulkRow(
            request=GenerationRequest(
                city="Testville",
                osm_path=fixture_osm_path,
                method="parametric_attach",
                n_customers=n,
            )
        )
        for n in (3, 900)
    ]
    report = preflight_rows(rows)
    group = report["groups"][0]
    assert group["skipped_sizes"] == [900] and group["status"] == "partial"
    assert group["pool_total"] >= 3


def test_bulk_rows_that_would_share_a_file_name_never_overwrite(
    tmp_path: Path, fixture_osm_path: Path
) -> None:
    """The instance name encodes route_count, not the request fields, so rows can
    collide even when their parameters differ. Nothing may be silently lost."""

    def row() -> BulkRow:
        return BulkRow(
            request=GenerationRequest(
                city="Testville",
                osm_path=fixture_osm_path,
                method="parametric_attach",
                n_customers=3,
                demand_type=1,
                avg_route_size=1,
            ),
            explicit_seed=99,
        )

    result = generate_bulk_from_rows(
        [row(), row()], output_root=tmp_path / "instances", base_seed=0
    )
    assert result["generated"] == 2
    names = [entry["base_name"] for entry in result["results"]]
    # Same parameters and same seed: the -s<seed> suffix cannot separate these.
    assert len(set(names)) == 2, names
    for entry in result["results"]:
        assert (Path(entry["folder"]) / f"{entry['base_name']}_meta.json").is_file()


def test_bulk_results_come_back_in_input_row_order(
    tmp_path: Path, fixture_osm_path: Path
) -> None:
    def row(depot_mode: str, n: int) -> BulkRow:
        return BulkRow(
            request=GenerationRequest(
                city="Testville",
                osm_path=fixture_osm_path,
                method="parametric_attach",
                n_customers=n,
                depot_mode=depot_mode,
                demand_type=1,
                avg_route_size=1,
            )
        )

    # Rows 0 and 2 pool together, row 1 does not: grouping alone would emit
    # them as 3, 4, 3 instead of the order the caller listed.
    rows = [row("center", 3), row("corner", 3), row("center", 4)]
    result = generate_bulk_from_rows(rows, output_root=tmp_path / "instances", base_seed=0)
    assert result["generated"] == 3
    assert [entry["summary"]["customers"] for entry in result["results"]] == [3, 3, 4]
    assert len({entry["base_name"] for entry in result["results"]}) == 3


def test_notice_warns_when_the_road_graph_is_too_small(fixture_osm_path: Path) -> None:
    # The fixture graph holds 5 vertices, so a parametric request for 20 cannot
    # be served: the user must hear that before the job fails on it.
    request = GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method="parametric_attach",
        n_customers=20,
        seed=5,
    )
    with pytest.warns(UserWarning, match="candidate graph vertices"):
        composition = build_generation_selection(request).params["composition"]
    assert composition["delivered"] < composition["requested"]
    notice = composition_notice(composition)
    assert notice is not None and "Generation will fail unless" in notice


def test_bulk_reports_the_outcome_of_every_input_row(
    tmp_path: Path, fixture_osm_path: Path
) -> None:
    def row(n: int) -> BulkRow:
        return BulkRow(
            request=GenerationRequest(
                city="Testville",
                osm_path=fixture_osm_path,
                method="parametric_attach",
                n_customers=n,
                seed=3,
            )
        )

    # The fixture pool cannot serve 900, so that row is dropped by the driver and
    # the total alone would not say which row went missing.
    result = generate_bulk_from_rows([row(3), row(900), row(4)], output_root=tmp_path)
    reports = result["row_reports"]
    assert [entry["index"] for entry in reports] == [0, 1, 2]
    assert [entry["status"] for entry in reports] == ["generated", "skipped", "generated"]
    assert "pool holds" in reports[1]["reason"]
    assert reports[0]["base_name"] and reports[0]["parametric_customers"] == 3


def test_graph_options_recorded_are_the_ones_the_loader_used(fixture_osm_path: Path) -> None:
    request = GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method="parametric_attach",
        n_customers=3,
        seed=3,
    )
    selection = build_generation_selection(request)
    params = selection.params
    # The trim rebuilds itself with trim_to_connected=False once the nodes are
    # removed, so only the loader's own record can be trusted here.
    assert selection.graph.loaded_with == (True, True)
    assert params["only_intersections"] is True
    assert params["trim_to_connected_graph"] is True
    assert params["requested_graph_options"] == {
        "only_intersections": True,
        "trim_to_connected_graph": True,
    }
    assert params["composition"]["graph_options_relaxed"] is None
    assert composition_notice(params["composition"]) is None

    # A relaxation must be spelled out rather than left in the metadata.
    relaxed = dict(params["composition"])
    relaxed["graph_options_relaxed"] = {
        "only_intersections": True,
        "trim_to_connected_graph": True,
    }
    notice = composition_notice(relaxed)
    assert notice is not None and "loaded with relaxed options" in notice


def test_poi_taken_by_the_depot_is_named_as_the_cause(fixture_osm_path: Path) -> None:
    # The cafe of node 2 sits at the center of the fixture, so a centered depot
    # claims the only matching POI and the shortfall has nothing to do with the
    # extract being short of amenities.
    request = GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method="poi_categories",
        n_customers=3,
        categories=["cafe"],
        depot_mode="center",
        seed=5,
    )
    with pytest.warns(UserWarning):
        composition = build_generation_selection(request).params["composition"]
    assert composition["poi_customers"] == 0 and composition["poi_used_as_depot"] == 1
    notice = composition_notice(composition)
    assert notice is not None
    assert "1 matching POI(s) became the depot rather than a customer" in notice


def test_the_depot_does_not_cost_the_request_a_poi_customer(fixture_osm_path: Path) -> None:
    """A depot standing on an amenity must not turn a POI run into a hybrid one.

    The depot is picked first, and it can land on a vertex an amenity attaches
    to. Filtering that POI out of the selection *after* the fact silently costs
    one customer: the run returns the n it was asked for, one is discarded, and
    the shortfall is topped up with a sampled road point -- so the instance is
    relabelled ``hybrid`` and stops being POI-only. Measured on the real
    campaign, this hit 5 of 50 POI rungs, including one in a city with four
    times the amenities it needed. Excluding the depot from the draw instead
    lets the walk continue to the next amenity.
    """
    request = GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method="poi_categories",
        n_customers=2,
        # Every category, so the fixture has amenities to spare and any shortfall
        # can only come from the depot collision.
        categories=list(POI_CATEGORIES),
        depot_mode="center",
        poi_attach_mode=POI_ATTACH_NEAREST_VERTEX,
        seed=5,
    )
    selection = build_generation_selection(request)

    assert selection.params["method"] == "poi_categories"
    assert selection.source_tags[1:] == ["poi", "poi"]
    assert selection.params["composition"]["poi_customers"] == 2
    assert selection.vertices[0] not in selection.vertices[1:], "depot served itself"


def test_poi_run_completed_parametrically_is_recorded_as_hybrid(
    tmp_path: Path, fixture_osm_path: Path
) -> None:
    # One matching POI for four customers: the instance is three quarters
    # parametric, so calling it a POI instance would misname the artifact.
    request = GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method="poi_categories",
        n_customers=4,
        categories=["cafe"],
        depot_mode="corner",
        seed=5,
    )
    with pytest.warns(UserWarning):
        params = build_generation_selection(request).params
    assert params["method"] == "hybrid" and params["requested_method"] == "poi_categories"
    assert params["composition"]["effective_method"] == "hybrid"

    with pytest.warns(UserWarning):
        result = generate_single_instance(request, tmp_path / "instances")
    assert "_hyb-" in result["base_name"]
    meta = json.loads((Path(result["folder"]) / f"{result['base_name']}_meta.json").read_text())
    assert meta["method"] == "hybrid"
    assert "recorded and named as 'hybrid'" in result["notice"]


@pytest.mark.parametrize("method", ["poi_categories", "hybrid"])
def test_a_run_without_any_poi_is_recorded_as_parametric(
    fixture_osm_path: Path, method: str
) -> None:
    # A centered depot claims the only cafe, so no customer sits on an amenity.
    request = GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method=method,
        n_customers=3,
        categories=["cafe"],
        hybrid_poi_share=0.5,
        depot_mode="center",
        seed=5,
    )
    with pytest.warns(UserWarning):
        params = build_generation_selection(request).params
    assert params["method"] == "parametric_attach"
    assert params["requested_method"] == method
    assert params["composition"]["poi_customers"] == 0


def test_bulk_rows_take_every_available_poi_before_parametric_points(
    tmp_path: Path, fixture_osm_path: Path
) -> None:
    def row(n: int) -> BulkRow:
        return BulkRow(
            request=GenerationRequest(
                city="Testville",
                osm_path=fixture_osm_path,
                method="poi_categories",
                n_customers=n,
                categories=["restaurant", "cafe", "school", "bar"],
                depot_mode="corner",
                seed=3,
            )
        )

    # Both rows share one pool sized for the larger of them. The small row must
    # still be served POIs first rather than drawn from the mixed pool.
    result = generate_bulk_from_rows([row(2), row(4)], output_root=tmp_path, base_seed=1)
    compositions = [entry["summary"]["composition"] for entry in result["results"]]
    assert compositions[0]["parametric_customers"] == 0
    assert compositions[0]["poi_customers"] == 2
    # The larger row exhausts the POIs and is named for what it really is.
    assert compositions[1]["poi_customers"] == 3
    assert compositions[1]["parametric_customers"] == 1
    assert "_poi-" in result["results"][0]["base_name"]
    assert "_hyb-" in result["results"][1]["base_name"]


def test_hybrid_degrades_to_parametric_when_no_category_has_a_poi(
    tmp_path: Path, fixture_osm_path: Path
) -> None:
    # The fixture has no ferry terminal: the POI half of the request cannot be
    # served at all, but its parametric half can, so the run goes through.
    request = GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method="hybrid",
        n_customers=3,
        categories=["ferry_terminal"],
        hybrid_poi_share=0.5,
        depot_mode="corner",
        seed=5,
    )
    selection = build_generation_selection(request)
    composition = selection.params["composition"]
    assert selection.source_tags == ["depot", "param", "param", "param"]
    assert composition["poi_customers"] == 0 and composition["poi_pool_matching"] == 0
    assert selection.params["method"] == "parametric_attach"
    assert selection.params["requested_method"] == "hybrid"

    notice = composition_notice(composition)
    assert notice is not None
    assert "None of the 1 selected category has a POI in this extract" in notice
    assert "Select other POI categories" in notice
    assert "recorded and named as 'parametric_attach'" in notice
    # Neither remedy applies when the pool is empty, so neither is offered.
    assert "lower the POI share" not in notice and "reduce the customer count" not in notice

    result = generate_single_instance(request, tmp_path / "instances")
    assert "_par-" in result["base_name"]


def test_poi_categories_still_refuses_a_city_without_any_matching_poi(
    fixture_osm_path: Path,
) -> None:
    # Only hybrid degrades: a pure POI run with nothing to draw from is a
    # request that cannot be honoured, not one to quietly reinterpret.
    request = GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method="poi_categories",
        n_customers=3,
        categories=["ferry_terminal"],
        seed=5,
    )
    with pytest.raises(NoPoiFoundError, match="No POI found for selected categories"):
        build_generation_selection(request)


def test_find_pois_reads_ways_and_relations_not_just_nodes(fixture_osm_path: Path) -> None:
    pois = find_pois(fixture_osm_path, list(POI_CATEGORIES))
    by_ref = {(poi.osm_type, poi.osm_id): poi for poi in pois}
    # The building-outline pub and the multipolygon marketplace are located by
    # the <center> Overpass computes, since they have no position of their own.
    pub = by_ref[("way", 4)]
    assert pub.category == "pub" and pub.name == "Le Quatre"
    assert (pub.lat, pub.lon) == (45.0056, 4.005)
    market = by_ref[("relation", 20)]
    assert market.category == "marketplace" and market.name == "Marché Vingt"
    # Node 4 and way 4 are different places: the id alone cannot identify either.
    assert by_ref[("node", 4)].category == "school"
    assert all(poi.osm_type == "node" for poi in find_pois(fixture_osm_path, ["cafe"]))


def test_way_mapped_pois_become_customers_with_their_type_recorded(
    tmp_path: Path, fixture_osm_path: Path
) -> None:
    request = GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method="poi_categories",
        n_customers=2,
        categories=["pub", "marketplace"],
        depot_mode="corner",
        seed=5,
    )
    selection = build_generation_selection(request)
    types = [poi.osm_type if poi else None for poi in selection.poi_meta]
    assert "way" in types or "relation" in types

    result = generate_single_instance(request, tmp_path / "instances")
    meta = json.loads((Path(result["folder"]) / f"{result['base_name']}_meta.json").read_text())
    poi_nodes = [node for node in meta["nodes"] if node["poi_osm_id"] is not None]
    assert poi_nodes, "the way-mapped POIs must reach the metadata"
    for node in poi_nodes:
        assert node["poi_osm_type"] in ("node", "way", "relation")
    # The depot is parametric here, so its POI columns stay null on both halves.
    assert meta["nodes"][0]["poi_osm_id"] is None
    assert meta["nodes"][0]["poi_osm_type"] is None


def test_manual_picks_distinguish_a_way_from_a_node_of_the_same_id(
    fixture_osm_path: Path,
) -> None:
    # Way 4 (the pub) and node 4 (the school) are different places; picking one
    # must never resolve to the other. They sit on the same intersection, so the
    # two are picked in separate runs rather than collapsed into one.
    def pick(*refs: tuple[str, int]) -> list[tuple[str | None, str]]:
        selection = build_generation_selection(
            GenerationRequest(
                city="Testville",
                osm_path=fixture_osm_path,
                method="manual",
                manual_poi_refs=list(refs),
                manual_depot_poi_ref=("node", 1),
                seed=7,
            )
        )
        assert selection.params["manual_selection"]["depot_poi_osm_type"] == "node"
        return [(poi.name, poi.osm_type) for poi in selection.poi_meta[1:] if poi]

    assert pick(("way", 4), ("node", 2)) == [("Le Quatre", "way"), ("Café Deux", "node")]
    assert pick(("node", 4), ("node", 2)) == [("École Quatre", "node"), ("Café Deux", "node")]

    # Bare ids still mean nodes, so payloads written before ways were selectable
    # keep resolving to exactly what they used to.
    legacy = GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method="manual",
        manual_poi_ids=[2, 3, 4],
        manual_depot_poi_id=1,
        seed=7,
    )
    legacy_picked = [
        (poi.osm_type, poi.osm_id) for poi in build_generation_selection(legacy).poi_meta[1:] if poi
    ]
    assert all(kind == "node" for kind, _ in legacy_picked)


def test_pois_sharing_a_road_point_are_named_on_the_surviving_customer(
    tmp_path: Path, fixture_osm_path: Path
) -> None:
    # Two pairs of amenities sit on the same intersections: the school and the
    # pub outline on one, the pharmacy and the marketplace relation on the
    # other. Each pair yields one customer, and the loser must not vanish.
    request = GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method="poi_categories",
        n_customers=2,
        categories=["school", "pub", "pharmacy", "marketplace"],
        depot_mode="corner",
        seed=5,
    )
    selection = build_generation_selection(request)
    assert len(selection.vertices) == 3
    for index in (1, 2):
        chosen = selection.poi_meta[index]
        merged = selection.poi_merged[index]
        assert chosen is not None and len(merged) == 1, "one amenity lost this point"
        # Whichever won, the pair is the same set of places.
        names = {chosen.name, merged[0].name}
        assert names in ({"École Quatre", "Le Quatre"}, {None, "Marché Vingt"})

    result = generate_single_instance(request, tmp_path / "instances")
    meta = json.loads((Path(result["folder"]) / f"{result['base_name']}_meta.json").read_text())
    merged_rows = [node["poi_merged"] for node in meta["nodes"] if node["poi_merged"]]
    assert len(merged_rows) == 2
    assert {row[0]["osm_type"] for row in merged_rows} <= {"node", "way", "relation"}
    assert all("category" in row[0] and "osm_id" in row[0] for row in merged_rows)
    # A node standing for a single amenity keeps an empty list, not a null.
    assert all(isinstance(node["poi_merged"], list) for node in meta["nodes"])


def test_manual_picks_that_collapse_are_named_on_the_customer_that_kept_the_point(
    fixture_osm_path: Path,
) -> None:
    # Node 4 (the school) and way 4 (the pub) snap to the same intersection, so
    # one pick becomes a customer and the other has to be accounted for on it.
    request = GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method="manual",
        manual_poi_refs=[("node", 4), ("way", 4), ("node", 2), ("node", 3)],
        manual_depot_poi_ref=("node", 1),
        seed=7,
    )
    selection = build_generation_selection(request)
    report = selection.params["manual_selection"]
    assert report["collapsed_poi_ids"] == [4]

    merged_by_name = {
        (poi.name if poi else None): [other.name for other in selection.poi_merged[index]]
        for index, poi in enumerate(selection.poi_meta)
    }
    assert merged_by_name["École Quatre"] == ["Le Quatre"]
    # The picks that kept a point of their own carry nothing extra.
    assert merged_by_name["Café Deux"] == []


def test_pool_ceiling_counts_every_attachable_vertex(fixture_osm_path: Path) -> None:
    # Six POIs in these categories, but two pairs share a point: the ceiling on
    # POI customers is four, not six, and it is reported even though this small
    # request is served in full.
    request = GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method="poi_categories",
        n_customers=2,
        categories=["school", "pub", "pharmacy", "marketplace", "restaurant", "cafe"],
        depot_mode="corner",
        seed=5,
    )
    composition = build_generation_selection(request).params["composition"]
    assert composition["poi_pool_matching"] == 6
    assert composition["poi_pool_attachable"] == 4


def test_poi_attachment_is_computed_once_per_extract_and_graph(
    fixture_osm_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mamut_routing_tools.generation import select as select_module
    from mamut_routing_tools.roadgraph.build import load_road_graph

    select_module._ATTACHMENT_CACHE.clear()
    graph = load_road_graph(fixture_osm_path)
    calls = 0
    real = graph.nearest_nodes

    def counted(points, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return real(points, *args, **kwargs)

    monkeypatch.setattr(graph, "nearest_nodes", counted)
    first = select_module.catalog_attachment(graph, fixture_osm_path)
    second = select_module.catalog_attachment(graph, fixture_osm_path)
    # Preview, preflight and generate all ask the same question of the same
    # extract, and one batched pass answers it for every category filter.
    assert calls == 1 and first is second
    assert any(vertex is not None for vertex in first)


def _bank_request(fixture_osm_path: Path, **overrides) -> GenerationRequest:
    return GenerationRequest(
        city="Testville",
        osm_path=fixture_osm_path,
        method="poi_categories",
        n_customers=3,
        categories=["bank", "cafe", "restaurant"],
        depot_mode="corner",
        seed=5,
        **overrides,
    )


def test_strict_attachment_discards_a_poi_whose_nearest_node_is_mid_segment(
    fixture_osm_path: Path,
) -> None:
    # The bank sits beside the disconnected island, so its nearest road node is
    # not a graph vertex. The published rule drops it and tops the instance up
    # parametrically -- the reason an amenity present in the extract can never
    # show up anywhere.
    with pytest.warns(UserWarning, match="POI-attached"):
        selection = build_generation_selection(_bank_request(fixture_osm_path))
    names = [poi.name for poi in selection.poi_meta if poi]
    assert "Banque de l'île" not in names
    composition = selection.params["composition"]
    assert composition["poi_customers"] == 2 and composition["parametric_customers"] == 1
    assert composition["poi_pool_unattached"] == 1
    assert composition["poi_attach_mode"] == "nearest_node"
    # The way out is named, with the count that makes it worth trying.
    notice = composition_notice(composition)
    assert notice is not None
    assert "switch POI attachment to 'nearest vertex'" in notice
    assert "1 POI(s) here sit closest to a mid-segment road node" in notice


def test_snapping_attachment_makes_that_poi_a_customer_and_records_the_distance(
    tmp_path: Path, fixture_osm_path: Path
) -> None:
    request = _bank_request(
        fixture_osm_path, poi_attach_mode="nearest_vertex", poi_attach_radius_m=400.0
    )
    selection = build_generation_selection(request)
    names = [poi.name for poi in selection.poi_meta if poi]
    assert "Banque de l'île" in names
    assert selection.params["composition"]["parametric_customers"] == 0
    assert selection.params["poi_attach_mode"] == "nearest_vertex"
    assert selection.params["poi_attach_radius_m"] == 400.0

    index = next(i for i, poi in enumerate(selection.poi_meta) if poi and "Banque" in (poi.name or ""))
    # How far the amenity had to move to reach a routable point, as manual
    # picking has always recorded it.
    assert 300 < selection.snap_distances_m[index] < 400
    others = [
        distance
        for i, distance in enumerate(selection.snap_distances_m)
        if i != index
    ]
    assert all(distance == 0.0 for distance in others), "POIs on a vertex never move"

    result = generate_single_instance(request, tmp_path / "instances")
    # utf-8 explicitly: the name carries an accent, and the platform default is
    # cp1252 on Windows, which would read it back as something else entirely.
    meta = json.loads(
        (Path(result["folder"]) / f"{result['base_name']}_meta.json").read_text(encoding="utf-8")
    )
    bank = next(node for node in meta["nodes"] if node["poi_name"] == "Banque de l'île")
    assert 300 < bank["snap_distance_m"] < 400
    assert meta["generation_params"]["poi_attach_mode"] == "nearest_vertex"


def test_a_radius_too_short_to_reach_a_vertex_still_drops_the_poi(
    fixture_osm_path: Path,
) -> None:
    # Snapping is bounded: the customer must stay where the amenity is.
    with pytest.warns(UserWarning, match="POI-attached"):
        selection = build_generation_selection(
            _bank_request(
                fixture_osm_path, poi_attach_mode="nearest_vertex", poi_attach_radius_m=50.0
            )
        )
    assert "Banque de l'île" not in [poi.name for poi in selection.poi_meta if poi]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"poi_attach_mode": "teleport"}, "Unsupported POI attach mode"),
        ({"poi_attach_radius_m": 0}, "poi_attach_radius_m"),
    ],
)
def test_generation_request_rejects_invalid_attachment_controls(
    fixture_osm_path: Path, overrides: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GenerationRequest(city="Testville", osm_path=fixture_osm_path, **overrides).validate()
