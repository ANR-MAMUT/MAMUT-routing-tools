"""Published benchmark instances in the workbench.

The GUI only knew the instances generated in its workspace, so nobody could
check and look at a solution of their own for a Mamut2026 or Poryos2026
instance. A benchmarks folder is now a second, read-only source: its static
instances are served as records of the shape the workspace ones have, and the
existing endpoints (runs, import with the reference checker, render, compare)
work on them unchanged. The fixture is a five-node collection written by hand
in the published layout.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mamut_routing_tools.gui import benchmarks as benchmark_source
from mamut_routing_tools.gui.benchmarks import BenchmarkCatalog, SidecarRoads, resolve_benchmarks_root
from mamut_routing_tools.gui.server import create_app

TOKEN = "test-token"
BASE = "mini-town-n4"
COORDINATES = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0], [-100.0, 50.0]]
DEMANDS = [0, 1, 1, 1, 1]
BKS_ROUTES = [[1, 2], [3, 4]]
# Node i sits on graph vertex i; vertices 5 and 6 are road bends. The direct road
# from the depot to customer 1 is short but slow; the one through vertex 5 is longer and fast.
VERTEX_LONLAT = [(2.0, 48.0), (2.002, 48.0), (2.002, 48.002), (2.0, 48.002), (1.998, 48.001), (2.001, 47.999), (2.001, 48.003)]
EDGES = sorted(
    [(0, 1, 100.0, 1.0), (1, 0, 100.0, 1.0)]
    + [edge for u, v, length in ((0, 5, 60.0), (5, 1, 60.0), (1, 2, 100.0), (2, 6, 55.0), (6, 3, 55.0), (3, 4, 112.0), (4, 0, 112.0), (2, 0, 150.0))
       for edge in ((u, v, length, 10.0), (v, u, length, 10.0))]
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_gzip_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        handle.write(json.dumps(payload).encode("utf-8"))


def _instance(name: str, problem: str, **extra) -> dict:
    return {
        "instance_name": name,
        "instance_origin": "OsmCvrpGen",
        "benchmark_name": "Poryos2026",
        "num_customers": 4,
        "num_vehicles": None,
        "vehicle_capacity": 2,
        "coordinates": COORDINATES,
        "demands": DEMANDS,
        "depot": 0,
        "reference_lla": {"lat": 48.0, "lon": 2.0, "alt": 0.0},
        "metric_variant": "euclidean",
        "arc_costs_source": {"model": "euclidean", "decimals": 3},
        "metadata": {
            "authors": "fixture", "generated_at": "2026-01-01", "problem_type": problem, "metric_variant": "euclidean",
            "city": "town", "method": "poi", "base_instance_name": BASE, "num_vehicles_lb": 2,
            "sidecars": {"geo": {"path": f"sidecars/town/n=4/{BASE}/{BASE}.geo.json.gz", "sha256": "0" * 64}},
        },
        **extra,
    }


def _bks(name: str) -> dict:
    return {
        "instance_name": name, "routes": BKS_ROUTES, "cost": 665.027, "objective_function": "MonoCost",
        "metadata": {"authors": "fixture", "source": "by hand", "validated_by": "fixture checker", "date": "2026-01-01T00:00:00",
                     "validated_cost": 665.027, "validated_num_routes": 2},
    }


@pytest.fixture
def benchmarks_dir(tmp_path: Path) -> Path:
    from mamut_routing_lib.td.roadgraph import InstanceRoadGraph, save_instance_road_graph

    root = tmp_path / "benchmarks"
    family = root / "Poryos2026"
    _write_json(family / "mamut-collection.json", {"format": "mamut-collection", "format_version": 1, "family": "Poryos2026", "layout_version": 1})
    cvrp = family / "CVRP" / "euclidean" / "town" / "n=4" / BASE
    _write_json(cvrp / f"{BASE}.vrp.json", _instance(BASE, "CVRP"))
    _write_json(cvrp / f"{BASE}.bks.MonoCost.json", _bks(BASE))
    windowed = family / "VRPTW" / "euclidean" / "town" / "n=4" / BASE
    wide = [[0.0, 10000.0]] * 5
    _write_json(windowed / f"{BASE}-tw-tight.vrp.json", _instance(f"{BASE}-tw-tight", "VRPTW", time_windows=wide, service_times=[0.0, 5.0, 5.0, 5.0, 5.0]))
    _write_json(windowed / f"{BASE}-tw-tight.bks.MonoCost.json", _bks(f"{BASE}-tw-tight"))
    sidecars = family / "sidecars" / "town" / "n=4" / BASE
    _write_gzip_json(sidecars / f"{BASE}.geo.json.gz", {
        "format": "mamut-geo", "format_version": 3, "base_name": BASE, "benchmark_name": "Poryos2026", "city": "town",
        "source_osm_file": "osmdata/Town.osm", "road_cache": None,
        "nodes": [
            {"instance_node_id": index, "poi_lon": VERTEX_LONLAT[index][0], "poi_lat": VERTEX_LONLAT[index][1],
             "demand": DEMANDS[index], "source_tag": "depot" if index == 0 else "poi", "graph_vertex_id": index}
            for index in range(5)
        ],
    })
    save_instance_road_graph(InstanceRoadGraph(
        base_name=BASE, benchmark_name="Poryos2026", num_customers=4, horizon=(0.0, 86400.0), extension_end=172800.0,
        bin_edges=[0.0, 86400.0], sample_step=60.0, simplify_tolerance=1.0, num_vertices=7,
        vertex_osm_ids=[10, 11, 12, 13, 14, 15, 16], vertex_lonlat=VERTEX_LONLAT, node_vertices=[0, 1, 2, 3, 4], edges=EDGES,
    ), sidecars / f"{BASE}.road.json.gz")
    # A classic family beside it: no metric variants, no sidecars, not offered here.
    classic = root / "Sintef2008"
    _write_json(classic / "mamut-collection.json", {"format": "mamut-collection", "format_version": 1, "family": "Sintef2008", "layout_version": 2})
    _write_json(classic / "VRPTW" / "n=2" / "tiny.vrp.json", {
        "instance_name": "tiny", "instance_origin": "Solomon1987", "benchmark_name": "Sintef2008", "num_customers": 2,
        "num_vehicles": 2, "vehicle_capacity": 1, "coordinates": [[0, 0], [1, 0], [2, 0]], "demands": [0, 1, 1],
        "arc_costs": [[0, 1, 2], [1, 0, 1], [2, 1, 0]], "service_times": [0, 0, 0], "time_windows": [[0, 9], [0, 9], [0, 9]],
    })
    return root


@pytest.fixture
def client(tmp_path: Path, benchmarks_dir: Path):  # type: ignore[no-untyped-def]
    app = create_app(tmp_path / "workspace", TOKEN, benchmarks_dir)
    with TestClient(app, base_url="http://localhost", headers={"X-Mamut-Token": TOKEN}) as test_client:
        yield test_client


def _tree(root: Path) -> dict[str, int]:
    return {path.relative_to(root).as_posix(): path.stat().st_mtime_ns for path in sorted(root.rglob("*")) if path.is_file()}


def _entry(client: TestClient, problem: str) -> dict:
    listing = client.get("/api/workbench/benchmarks").json()
    return next(entry for entry in listing["instances"] if entry["problem_type"] == problem)


def _sol(routes: list[list[int]]) -> str:
    return "\n".join(f"Route #{index + 1}: {' '.join(map(str, route))}" for index, route in enumerate(routes))


def test_the_catalog_lists_static_instances_of_family_first_collections(benchmarks_dir: Path) -> None:
    listing = BenchmarkCatalog(benchmarks_dir).listing()
    assert listing["families"] == ["Poryos2026"] and listing["hint"] is None
    by_problem = {entry["problem_type"]: entry for entry in listing["instances"]}
    assert set(by_problem) == {"CVRP", "VRPTW"}
    cvrp = by_problem["CVRP"]
    assert cvrp["source"] == "benchmark" and cvrp["base_name"] == BASE and cvrp["city"] == "town" and cvrp["n_customers"] == 4
    # The metric variants of a base are one instance; paths are relative to the collection, which is the record's folder.
    assert cvrp["metrics"] == ["euclidean"]
    assert cvrp["files"]["vrp_json"] == {"euclidean": f"CVRP/euclidean/town/n=4/{BASE}/{BASE}.vrp.json"}
    assert Path(cvrp["folder"]) == benchmarks_dir / "Poryos2026"
    # A time-window set is an instance of its own, under the other problem class.
    windowed = by_problem["VRPTW"]
    assert windowed["base_name"] == f"{BASE}-tw-tight" and windowed["tw_set"] == "tight"
    assert windowed["instance_id"] != cvrp["instance_id"] and cvrp["instance_id"].startswith("bench-")
    # The id names the instance, not its place on disk.
    assert cvrp["instance_id"] == benchmark_source.benchmark_instance_id("Poryos2026", "CVRP", BASE)


def test_the_benchmarks_folder_is_explicit_then_environment_then_found_above(
    tmp_path: Path, benchmarks_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (benchmark_source.BENCHMARKS_DIR_ENV, "MAMUT_ROUTING_BENCHMARKS_ROOT", "MAMUT_ROUTING_ROOT"):
        monkeypatch.delenv(name, raising=False)
    nested = benchmarks_dir.parent / "tools" / "deep"
    nested.mkdir(parents=True)
    assert resolve_benchmarks_root(start=nested) == benchmarks_dir
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "benchmarks").mkdir()  # a folder of that name without collections is not one
    assert resolve_benchmarks_root(start=elsewhere / "benchmarks") != elsewhere / "benchmarks"
    monkeypatch.setenv("MAMUT_ROUTING_BENCHMARKS_ROOT", str(tmp_path / "from-env"))
    assert resolve_benchmarks_root(start=nested) == (tmp_path / "from-env").resolve()
    assert resolve_benchmarks_root(tmp_path / "explicit", start=nested) == (tmp_path / "explicit").resolve()
    # A configured folder that does not exist is reported, not silently replaced.
    assert "does not exist" in BenchmarkCatalog(tmp_path / "explicit").listing()["hint"]
    assert "No benchmarks folder" in BenchmarkCatalog(None).listing()["hint"]


def test_a_folder_that_was_empty_is_read_again(tmp_path: Path, benchmarks_dir: Path) -> None:
    """A listing that found nothing is not kept for the life of the server."""
    later = tmp_path / "later"
    catalog = BenchmarkCatalog(later)
    assert catalog.listing()["instances"] == [] and "does not exist" in catalog.listing()["hint"]
    benchmarks_dir.rename(later)  # the collection is cloned after the GUI was started
    listing = catalog.listing()
    assert listing["hint"] is None and len(listing["instances"]) == 2
    # A listing that found instances is kept until a rescan is asked for.
    import shutil

    shutil.rmtree(later / "Poryos2026" / "VRPTW")
    assert len(catalog.listing()["instances"]) == 2
    assert [entry["problem_type"] for entry in catalog.listing(refresh=True)["instances"]] == ["CVRP"]


def test_the_server_and_its_helpers_read_one_catalog(tmp_path: Path, benchmarks_dir: Path) -> None:
    """The listing endpoint and the lookup by instance id go through the same catalog."""
    workspace = tmp_path / "workspace"
    first = create_app(workspace, TOKEN, tmp_path / "nowhere")
    create_app(workspace, TOKEN, benchmarks_dir)  # the same workspace, configured again
    with TestClient(first, base_url="http://localhost", headers={"X-Mamut-Token": TOKEN}) as test_client:
        entries = test_client.get("/api/workbench/benchmarks").json()["instances"]
        assert len(entries) == 2
        for entry in entries:
            assert test_client.get(f"/api/instances/{entry['instance_id']}/map-data").json()["ok"]
            assert test_client.get(f"/api/workbench/benchmarks/{entry['instance_id']}").json()["ok"]


def test_a_benchmark_instance_is_served_like_a_workspace_one(client: TestClient, benchmarks_dir: Path) -> None:
    entry = _entry(client, "CVRP")
    instance_id = entry["instance_id"]
    assert entry["solution_count"] == 0
    # Only the fields the panel shows: a real manifest also pins the instance and its matrix by hash.
    _write_json(
        benchmarks_dir / "Poryos2026" / "CVRP" / "euclidean" / "town" / "n=4" / BASE / f"{BASE}.manifest.json",
        {"benchmark_revision": "r9", "default_profile": "binary64-exact-v1", "objective": "MonoCost", "units": {"cost": "metre"}},
    )

    detail = client.get(f"/api/workbench/benchmarks/{instance_id}").json()["instance"]
    assert detail["summary"]["capacity"] == 2 and detail["summary"]["route_count"] == 2
    assert detail["summary"]["revision"] == "r9" and detail["summary"]["units"] == {"euclidean": "metre"}
    assert detail["summary"]["bks"] == {"euclidean": ["MonoCost"]}
    assert client.get("/api/workbench/benchmarks/bench-0000").status_code == 404

    # The geo sidecar numbers nodes from 0; the map gets the generator's convention, depot named.
    nodes = client.get(f"/api/instances/{instance_id}/map-data").json()
    assert nodes["depot_instance_node_id"] == 1
    properties = [feature["properties"] for feature in nodes["geojson"]["features"]]
    assert [p["instance_node_id"] for p in properties] == [1, 2, 3, 4, 5] and [p["model_node_id"] for p in properties] == [0, 1, 2, 3, 4]
    assert [p["role"] for p in properties] == ["depot", "customer", "customer", "customer", "customer"]
    assert nodes["geojson"]["features"][1]["geometry"]["coordinates"] == [2.002, 48.0]

    # The workspace listing is untouched: benchmarks are a separate source.
    assert client.get("/api/workbench/instances").json()["instances"] == []


def test_the_published_bks_is_the_reference_without_running_the_checker(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mamut_routing_tools.gui.server as server

    instance_id = _entry(client, "CVRP")["instance_id"]

    def refuse(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("listing the published BKS must not load the cost matrix")

    monkeypatch.setattr(server, "validate_solution", refuse)
    runs = client.get(f"/api/instances/{instance_id}/solutions").json()
    assert runs["runs"] == []
    (reference,) = runs["references"]
    assert reference["run_id"] == "bks:MonoCost:euclidean" and reference["source"] == "bks"
    assert reference["cost"] == 665.027 and reference["num_routes"] == 2 and reference["routes"] == BKS_ROUTES
    assert reference["validation"] == {
        "valid": True, "status": "published", "recorded": True, "error_message": None,
        "routing_cost": 665.027, "num_routes": 2, "validated_by": "fixture checker",
    }


def test_a_solution_is_checked_stored_in_the_workspace_and_compared_with_the_bks(
    client: TestClient, benchmarks_dir: Path, tmp_path: Path,
) -> None:
    instance_id = _entry(client, "CVRP")["instance_id"]
    before = _tree(benchmarks_dir)

    # A customer left out: the checker's reason comes back and nothing is stored.
    rejected = client.post(f"/api/instances/{instance_id}/solutions/import", json={"metric": "euclidean", "text": _sol([[1, 2], [3]])})
    assert rejected.status_code == 400 and "never visited" in rejected.json()["error"]
    assert client.get(f"/api/instances/{instance_id}/solutions").json()["runs"] == []
    # A variant this instance does not have is refused too.
    assert client.post(f"/api/instances/{instance_id}/solutions/import", json={"metric": "fastest", "text": _sol(BKS_ROUTES)}).status_code == 400

    # A valid one, worse than the BKS: customers paired the long way round.
    mine = [[1, 3], [2, 4]]
    imported = client.post(
        f"/api/instances/{instance_id}/solutions/import",
        json={"metric": "euclidean", "text": _sol(mine), "filename": "mine.sol", "label": "my heuristic"},
    ).json()
    assert imported["ok"], imported
    solution = imported["solution"]
    assert solution["source"] == "imported" and solution["solver"] == "my heuristic" and solution["routes"] == mine
    assert solution["validation"]["valid"] and solution["cost"] > 665.027
    assert solution["metadata"]["metric"] == "euclidean"

    # It lives in the workspace; the benchmarks folder is exactly as it was.
    stored = tmp_path / "workspace" / "solutions" / instance_id / f"{solution['run_id']}.json"
    assert stored.is_file()
    assert _tree(benchmarks_dir) == before
    assert _entry(client, "CVRP")["solution_count"] == 1

    comparison = client.post(
        f"/api/instances/{instance_id}/solutions/compare",
        json={"candidate_run_id": solution["run_id"], "reference_run_id": "bks:MonoCost:euclidean"},
    ).json()
    assert comparison["ok"], comparison
    assert comparison["comparison"]["ordering"] == "worse" and comparison["comparison"]["cost_delta"] > 0
    assert comparison["comparison"]["route_count_delta"] == 0


def test_routes_are_drawn_on_the_published_road_graph(client: TestClient) -> None:
    instance_id = _entry(client, "CVRP")["instance_id"]
    render = lambda metric: client.post(f"/api/instances/{instance_id}/solutions/bks:MonoCost:euclidean/render?metric={metric}", json={}).json()  # noqa: E731

    fastest = render("fastest")
    assert fastest["ok"], fastest
    assert fastest["summary"]["render_mode"] == "cached_road" and fastest["summary"]["road_source"] == "road-graph sidecar"
    assert fastest["summary"]["straight_fallback_count"] == 0
    first, second = (feature["geometry"]["coordinates"] for feature in fastest["geojson"]["features"])
    # Depot to customer 1 takes the fast road through the bend (vertex 5), then 1 -> 2 and 2 -> depot directly.
    assert first == [list(VERTEX_LONLAT[0]), list(VERTEX_LONLAT[5]), list(VERTEX_LONLAT[1]), list(VERTEX_LONLAT[2]), list(VERTEX_LONLAT[0])]
    # Each arc's first point is addressable, for depot legs and the animation.
    assert fastest["geojson"]["features"][0]["properties"]["arc_starts"] == [0, 2, 3]
    assert fastest["geojson"]["features"][0]["properties"]["load"] == 2
    # 2 -> 3 goes round by vertex 6.
    assert list(VERTEX_LONLAT[6]) not in first and second[0] == list(VERTEX_LONLAT[0]) and second[-1] == list(VERTEX_LONLAT[0])

    # By length the slow direct road wins.
    shortest = render("shortest")["geojson"]["features"][0]["geometry"]["coordinates"]
    assert shortest[:2] == [list(VERTEX_LONLAT[0]), list(VERTEX_LONLAT[1])]
    # The euclidean variant has no road to follow.
    straight = render("euclidean")
    assert straight["summary"]["render_mode"] == "straight_line"
    # Depot, two customers, depot: one point per stop.
    assert straight["geojson"]["features"][0]["geometry"]["coordinates"] == [list(VERTEX_LONLAT[v]) for v in (0, 1, 2, 0)]


def test_road_paths_are_the_library_s_pinned_paths() -> None:
    """Stopping the search at the target must not change the path the published cost was measured on."""
    import random

    from mamut_routing_lib.td.roadgraph import InstanceRoadGraph, build_adjacency, compute_fastest_path_tree

    rng = random.Random(7)
    count = 60
    pairs = {(u, v) for u in range(count) for v in rng.sample(range(count), 4) if u != v}
    pairs |= {(v, u) for u, v in pairs}
    # Few distinct weights, so equal-cost paths are common and the tie rule matters.
    edges = [(u, v, float(rng.choice([10, 20, 30])), float(rng.choice([1, 2]))) for u, v in sorted(pairs)]
    road = InstanceRoadGraph(
        base_name="random", benchmark_name="Poryos2026", num_customers=5, horizon=(0.0, 86400.0), extension_end=172800.0,
        bin_edges=[0.0, 86400.0], sample_step=60.0, simplify_tolerance=1.0, num_vertices=count,
        vertex_osm_ids=list(range(1, count + 1)), vertex_lonlat=[(float(v), 0.0) for v in range(count)],
        node_vertices=[0, 11, 23, 37, 48, 59], edges=edges,
    )
    roads = SidecarRoads.__new__(SidecarRoads)
    roads.road, roads.adjacency, roads._segments = road, build_adjacency(road), {}
    for i, source in enumerate(road.node_vertices):
        dist, pred_edge = compute_fastest_path_tree(road, roads.adjacency, source)
        for j, target in enumerate(road.node_vertices):
            if i == j or dist[target] == float("inf"):
                continue
            expected = [target]
            while expected[-1] != source:
                expected.append(road.edges[pred_edge[expected[-1]]][0])
            expected.reverse()
            assert roads.segment(i, j, "fastest") == [[float(v), 0.0] for v in expected]
    assert roads.segment(0, 1, "euclidean") is None


def test_a_time_window_instance_is_checked_against_its_windows(client: TestClient) -> None:
    entry = _entry(client, "VRPTW")
    instance_id = entry["instance_id"]
    assert entry["metrics"] == ["euclidean"] and entry["tw_set"] == "tight"
    assert client.get(f"/api/workbench/benchmarks/{instance_id}").json()["instance"]["summary"]["has_time_windows"] is True
    imported = client.post(f"/api/instances/{instance_id}/solutions/import", json={"metric": "euclidean", "text": _sol(BKS_ROUTES)}).json()
    assert imported["ok"], imported
    assert imported["solution"]["validation"]["valid"] and imported["solution"]["num_routes"] == 2


def test_the_frontend_offers_the_benchmarks_source(client: TestClient) -> None:
    html = client.get("/").text
    script = client.get("/static/workbench.js").text
    assert 'id="sourceWorkspace"' in html and 'id="sourceBenchmarks"' in html
    assert 'id="benchmarkSource" class="source-view" hidden' in html and 'id="benchList"' in html
    assert "/api/workbench/benchmarks" in script and "function selectBenchmark(" in script
    # A benchmark's primary action, and how a solution stands against the published BKS.
    assert "Check my solution" in script and "function bksStandingHtml(run)" in script
    # A selected benchmark instance survives a refresh of the workspace list.
    assert 'state.selected.source !== "benchmark"' in script
    # The folder can be read again, and the list learns of a run made in this session.
    assert 'id="bench-refresh"' in html and "loadBenchmarks(true)" in script
    assert "if (listed) listed.solution_count = instance.solution_count;" in script
    # A finished solve opens the instance it belongs to; it is never loaded into another one.
    solve = script[script.index('} else if (job.kind === "solve") {'):]
    solve = solve[:solve.index("activateTab(")]
    assert "await selectBenchmark(result.instance_id, preferredRunId);" in solve
    assert solve.count("loadSolutionRuns(preferredRunId)") == 1 and "state.selected?.instance_id === result.instance_id" in solve
    # The standing against the BKS says no more than the rounded costs show.
    assert '"less than 0.01%"' in script and "Compare prices both exactly" in script and "Equals the published BKS" not in script
