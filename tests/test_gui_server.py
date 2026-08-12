"""GUI server tests: security guards and the workbench endpoint shapes."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mamut_routing_tools import osm
from mamut_routing_tools.generation.pois import POI_CATEGORIES
from mamut_routing_tools.gui.server import create_app
from mamut_routing_tools.workspace import jobs_dir, osmdata_dir

TOKEN = "test-token"


@pytest.fixture
def client(tmp_path: Path, fixture_osm_path: Path):  # type: ignore[no-untyped-def]
    workspace = tmp_path / "workspace"
    (osmdata_dir(workspace)).mkdir(parents=True, exist_ok=True)
    (osmdata_dir(workspace) / "Testville.osm").write_text(fixture_osm_path.read_text())
    app = create_app(workspace, TOKEN)
    with TestClient(app, base_url="http://localhost", headers={"X-Mamut-Token": TOKEN}) as test_client:
        yield test_client


def _wait_for_job(client: TestClient, job_id: str) -> dict:
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()["job"]
        if job["status"] in {"completed", "failed", "cancelled", "interrupted"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish")


def test_requests_without_token_or_wrong_host_are_rejected(client: TestClient) -> None:
    assert client.get("/healthz", headers={"X-Mamut-Token": "wrong"}).status_code == 403
    assert client.get("/healthz", headers={"Host": "evil.example"}).status_code == 403
    assert client.get("/healthz").status_code == 200


def test_cities_endpoint_lists_workspace_extracts(client: TestClient) -> None:
    payload = client.get("/api/workbench/generation/cities").json()
    assert payload["ok"] and payload["preview_available"]
    assert [city["slug"] for city in payload["cities"]] == ["testville"]


def test_gui_shell_exposes_restored_generation_and_instance_only_controls(client: TestClient) -> None:
    html = client.get("/").text

    assert 'id="depot-mode"' in html
    assert 'id="poi-list"' in html
    assert 'id="hybrid-poi-share"' in html
    assert 'id="customer-mode"' in html
    assert 'id="cluster-seeds"' in html
    assert 'id="cluster-decay"' in html
    assert "POI / ${100 - poiPercent}% parametric" in html
    assert "Excentered (corner)" in html
    assert "Instance only · customer locations" in html
    assert "City fetches store every listed category" in html
    assert "poiCategories: [...POI_CATEGORIES]" in html

    # Bulk configuration overlay with the per-instance table.
    assert 'id="bulk-modal"' in html and 'id="bulk-table-body"' in html
    for control in ("bulk-expand", "bulk-add-row", "bulk-import-csv", "bulk-export-csv", "bulk-check"):
        assert f'id="{control}"' in html
    # Per-row outcome, so a row the city pool could not serve is visible as such.
    assert "<th>Outcome</th>" in html and "bulkOutcomeCell" in html
    # The pre-generation composition line and the dialog that confirms a shortfall.
    assert 'id="gen-notice"' in html
    assert 'id="confirm-modal"' in html and 'id="confirm-ok"' in html
    assert "/api/workbench/generation/preflight" in html
    # POI attachment: the strict rule and the snapping alternative.
    assert 'id="poi-attach-mode"' in html and 'id="poi-attach-radius"' in html
    assert 'value="nearest_vertex"' in html and 'id="help-attach"' in html
    # Stored-extract coverage: check what is on disk, refill what predates ways.
    assert 'id="osm-audit-panel"' in html and 'id="osm-audit-check"' in html
    assert 'id="osm-audit-refresh"' in html
    assert "/api/workbench/osmdata/audit" in html
    # Manual POI picking and the solution import panel.
    assert 'id="manual-options"' in html and 'id="manual-list"' in html
    assert 'value="manual"' in html
    assert 'id="import-panel"' in html and 'id="import-run"' in html
    # Inline help for the dials that used to be bare numbers.
    for help_id in ("help-demand", "help-band", "help-method", "help-depot", "help-metric"):
        assert f'id="{help_id}"' in html
    # The right panel must reflow on narrow screens instead of disappearing.
    assert ".panel-right { display: none; }" not in html
    assert 'id="sheet-toggle"' in html


def test_frontend_demand_and_band_labels_match_the_generator(client: TestClient) -> None:
    """The JS tables are a copy of demands.py; keep them honest."""
    from mamut_routing_tools.generation.demands import (
        DEMAND_TYPES,
        avg_route_size_bounds,
        demand_distribution_bounds,
    )

    html = client.get("/").text
    for demand_type in DEMAND_TYPES:
        assert f"{{ value: {demand_type}," in html
        low, high = demand_distribution_bounds(demand_type)
        band_low, band_high = avg_route_size_bounds(demand_type)
        # Every bound is quoted somewhere in the help tables (en dash separated).
        assert f"{low}–{high}" in html or demand_type in (1, 6, 7)
        assert f"{int(band_low)}–{int(band_high)}" in html


def test_fetch_job_acquires_the_full_poi_catalog(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def fake_fetch(city: str, **kwargs):  # type: ignore[no-untyped-def]
        captured["city"] = city
        captured.update(kwargs)
        return {
            "ok": True,
            "city": city,
            "poi_categories": list(kwargs["poi_categories"]),
        }

    monkeypatch.setattr(osm, "fetch_and_store_city_osm", fake_fetch)
    requested_categories = ["hospital", "library", "charging_station"]
    submitted = client.post(
        "/api/jobs",
        json={
            "kind": "fetch-osm",
            "payload": {"city": "New Poi City", "poiCategories": requested_categories},
        },
    ).json()
    job = _wait_for_job(client, submitted["job"]["job_id"])

    assert job["status"] == "completed", job.get("error")
    assert captured["profile"] == "generation"
    assert captured["poi_categories"] == POI_CATEGORIES
    assert job["result"]["poi_categories"] == POI_CATEGORIES
    assert set(requested_categories) <= set(captured["poi_categories"])


def test_preview_generate_solve_render_round_trip(client: TestClient) -> None:
    body = {
        "city": "Testville",
        "nCustomers": 4,
        "seed": 7,
        "method": "parametric_attach",
        "depotMode": "excentered",
        "customerMode": "clustered",
        "clusterSeeds": 2,
        "clusterDecayMeters": 375,
        "hybridPoiShare": 0.65,
        "categories": ["hospital", "library"],
    }
    preview = client.post("/api/workbench/generation/preview", json=body).json()
    assert preview["ok"] and len(preview["geojson"]["features"]) == 5

    parametric_only_hybrid = client.post(
        "/api/workbench/generation/preview",
        json={
            **body,
            "method": "hybrid",
            "categories": [],
            "hybridPoiShare": 0,
        },
    ).json()
    assert parametric_only_hybrid["ok"]
    assert parametric_only_hybrid["summary"]["poi_customers"] == 0
    assert parametric_only_hybrid["summary"]["parametric_customers"] == 4

    single = client.post("/api/workbench/generation/single", json=body).json()
    assert single["ok"]
    folder = single["folder"]

    listing = client.get("/api/workbench/instances").json()
    assert listing["ok"]
    listed = {entry["base_name"]: entry for entry in listing["instances"]}
    assert single["base_name"] in listed
    entry = listed[single["base_name"]]
    assert entry["folder"] == folder
    assert entry["files"]["vrp_json"]["fastest"] == single["files"]["vrp_json"]["fastest"]
    assert entry["summary"]["capacity"] == single["summary"]["capacity"]
    assert entry["has_vrptw_twin"] is False
    assert entry["solution_count"] == 0

    map_data = client.get(f"/api/instances/{entry['instance_id']}/map-data").json()
    assert map_data["ok"] and len(map_data["geojson"]["features"]) == 5
    assert [feature["properties"]["model_node_id"] for feature in map_data["geojson"]["features"]] == list(range(5))
    assert [feature["properties"]["role"] for feature in map_data["geojson"]["features"]].count("depot") == 1

    vrp_json = client.get(
        "/instances-file", params={"path": f"{folder}/{single['files']['vrp_json']['fastest']}"}
    ).json()
    solved = client.post("/api/workbench/solve", json={"vrp_json": vrp_json, "time_limit": 1}).json()
    assert solved["ok"] and solved["n_routes"] >= 1 and solved["input_source"] == "vrp_json"
    assert sorted(stop for route in solved["routes"] for stop in route) == [1, 2, 3, 4]

    meta = client.get("/instances-file", params={"path": f"{folder}/{single['files']['meta']}"}).json()
    assert meta["generation_params"]["depot_mode"] == "corner"
    assert meta["generation_params"]["customer_mode"] == "clustered"
    assert meta["generation_params"]["cluster_seeds"] == 2
    assert meta["generation_params"]["cluster_decay_meters"] == 375
    assert meta["generation_params"]["hybrid_poi_share"] == 0.65
    assert meta["generation_params"]["categories"] == ["hospital", "library"]
    rendered = client.post(
        "/api/workbench/render-routes", json={"meta": meta, "routes": solved["routes"], "metric": "fastest"}
    ).json()
    assert rendered["ok"] and rendered["summary"]["render_mode"] in ("cached_road", "mixed")

    rendered_euclidean = client.post(
        "/api/workbench/render-routes", json={"meta": meta, "routes": solved["routes"], "metric": "euclidean"}
    ).json()
    drawn_points = {
        tuple(point)
        for feature in rendered_euclidean["geojson"]["features"]
        for point in feature["geometry"]["coordinates"]
    }
    expected_points = {(node["poi_lon"], node["poi_lat"]) for node in meta["nodes"]}
    assert expected_points <= drawn_points

    download = client.post(
        "/api/workbench/generation/single-download",
        json={"folder": folder, "base_name": single["base_name"]},
    )
    assert download.status_code == 200 and download.headers["content-type"] == "application/zip"


def test_td_build_derives_tdvrp_and_tdvrptw_twins(client: TestClient) -> None:
    generated = client.post(
        "/api/workbench/generation/single",
        json={
            "city": "Testville",
            "nCustomers": 3,
            "seed": 5,
            "method": "parametric_attach",
            "deriveVrptw": True,
        },
    ).json()
    assert generated["ok"] and generated["instance_id"]

    built = client.post(
        "/api/workbench/generation/td-build",
        json={"instance_id": generated["instance_id"], "model": "wave", "intensity": "moderate"},
    ).json()
    assert built["ok"] is True
    assert built["action"] in ("derived", "kept")
    assert len(built["combos"]) == 1
    combo = built["combos"][0]
    assert combo["model"] == "wave" and combo["intensity"] == "moderate"
    folder = Path(generated["folder"])
    assert (folder / combo["tdvrptw_twin"]).is_file()
    assert (folder / combo["tdvrp_twin"]).is_file()


def test_td_build_rejects_unknown_instance(client: TestClient) -> None:
    assert client.post("/api/workbench/generation/td-build", json={"instance_id": "nope"}).status_code == 404


def test_instances_file_refuses_paths_outside_the_workspace(client: TestClient) -> None:
    assert client.get("/instances-file", params={"path": "/etc/passwd"}).status_code == 404


def test_solve_jobs_persist_validated_runs_and_compare_them(client: TestClient) -> None:
    generated = client.post(
        "/api/workbench/generation/single",
        json={"city": "Testville", "nCustomers": 4, "seed": 7, "method": "parametric_attach"},
    ).json()
    assert generated["ok"] and generated["instance_id"]
    instance_id = generated["instance_id"]

    run_ids = []
    for seed in (11, 12):
        submitted = client.post(
            "/api/jobs",
            json={
                "kind": "solve",
                "payload": {
                    "instance_id": instance_id,
                    "metric": "fastest",
                    "objective_function": "MonoCost",
                    "seed": seed,
                    "time_limit": 1,
                },
            },
        ).json()
        job = _wait_for_job(client, submitted["job"]["job_id"])
        assert job["status"] == "completed", job.get("error")
        assert job["result"]["validation"]["valid"] is True
        assert job["result"]["solution"]["seed"] == seed
        run_ids.append(job["result"]["solution"]["run_id"])

    listing = client.get(f"/api/instances/{instance_id}/solutions").json()
    assert [run["run_id"] for run in listing["runs"]] == list(reversed(run_ids))
    assert all(run["validation"]["status"] == "valid" for run in listing["runs"])
    workspace_listing = client.get("/api/workbench/instances").json()["instances"]
    assert next(entry for entry in workspace_listing if entry["instance_id"] == instance_id)["solution_count"] == 2

    comparison = client.post(
        f"/api/instances/{instance_id}/solutions/compare",
        json={"candidate_run_id": run_ids[1], "reference_run_id": run_ids[0]},
    ).json()["comparison"]
    assert comparison["ordering"] in {"better", "equal", "worse"}
    assert comparison["candidate"]["valid"] and comparison["reference"]["valid"]
    assert "directed_edges_added" in comparison["route_difference"]

    rendered = client.post(
        f"/api/instances/{instance_id}/solutions/{run_ids[1]}/render",
        params={"metric": "fastest"},
    ).json()
    assert rendered["ok"] and rendered["summary"]["route_count"] >= 1

    jobs = client.get("/api/jobs").json()["jobs"]
    assert len([job for job in jobs if job["kind"] == "solve" and job["status"] == "completed"]) == 2
    log = client.get(f"/api/jobs/{jobs[0]['job_id']}/log").json()["log"]
    assert "Job completed" in log

    workspace = Path(client.get("/healthz").json()["workspace"])
    with TestClient(
        create_app(workspace, TOKEN),
        base_url="http://localhost",
        headers={"X-Mamut-Token": TOKEN},
    ) as restarted:
        restarted_runs = restarted.get(f"/api/instances/{instance_id}/solutions").json()["runs"]
        assert {run["run_id"] for run in restarted_runs} == set(run_ids)
        restarted_instance = next(
            entry
            for entry in restarted.get("/api/workbench/instances").json()["instances"]
            if entry["instance_id"] == instance_id
        )
        assert restarted_instance["solution_count"] == 2
        restarted_jobs = restarted.get("/api/jobs").json()["jobs"]
        assert all(job["status"] == "completed" for job in restarted_jobs)


def test_restart_marks_unfinished_jobs_as_interrupted(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    jobs_dir(workspace).mkdir(parents=True, exist_ok=True)
    job_id = "unfinished-job"
    (jobs_dir(workspace) / f"{job_id}.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "kind": "solve",
                "status": "running",
                "created_at": "2026-07-21T08:00:00+00:00",
                "started_at": "2026-07-21T08:00:01+00:00",
                "finished_at": None,
                "progress": {"message": "Solving", "current": None, "total": None},
                "cancel_requested": False,
                "request": {},
                "result": None,
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    with TestClient(
        create_app(workspace, TOKEN),
        base_url="http://localhost",
        headers={"X-Mamut-Token": TOKEN},
    ) as restarted:
        job = restarted.get(f"/api/jobs/{job_id}").json()["job"]

    assert job["status"] == "interrupted"
    assert job["finished_at"]
    assert job["error"] == "GUI server stopped before this job finished."


def test_poi_endpoint_serves_named_pois_and_filters_by_category(client: TestClient) -> None:
    everything = client.get("/api/workbench/generation/pois", params={"city": "Testville"}).json()
    assert everything["ok"]
    by_ref = {
        (feature["properties"]["osm_type"], feature["properties"]["osm_id"]): feature["properties"]
        for feature in everything["geojson"]["features"]
    }
    assert by_ref[("node", 1)]["name"] == "Chez Un"
    assert by_ref[("node", 1)]["category"] == "restaurant"
    # A nameless amenity is still offered; the picker labels it itself.
    assert by_ref[("node", 3)]["name"] is None
    assert by_ref[("node", 3)]["category"] == "pharmacy"
    # Amenities mapped as a building outline or a multipolygon are offered too,
    # and are told apart from the node sharing their id.
    assert by_ref[("way", 4)]["name"] == "Le Quatre"
    assert by_ref[("node", 4)]["category"] == "school"
    assert by_ref[("relation", 20)]["category"] == "marketplace"
    assert everything["category_counts"]["cafe"] == 1

    filtered = client.get(
        "/api/workbench/generation/pois", params={"city": "Testville", "categories": "cafe,bar"}
    ).json()
    assert {feature["properties"]["category"] for feature in filtered["geojson"]["features"]} == {
        "cafe",
        "bar",
    }

    unknown = client.get(
        "/api/workbench/generation/pois", params={"city": "Testville", "categories": "not_an_amenity"}
    )
    assert unknown.status_code == 400


def test_manual_generation_uses_the_picked_pois(client: TestClient) -> None:
    body = {
        "city": "Testville",
        "method": "manual",
        "seed": 3,
        "manualPoiIds": [2, 3, 4],
        "manualDepotPoiId": 1,
    }
    preview = client.post("/api/workbench/generation/preview", json=body).json()
    assert preview["ok"]
    features = preview["geojson"]["features"]
    assert [feature["properties"]["role"] for feature in features].count("depot") == 1
    assert [feature["properties"]["poi_name"] for feature in features] == [
        "Chez Un",
        "Café Deux",
        None,
        "École Quatre",
    ]

    generated = client.post("/api/workbench/generation/single", json=body).json()
    assert generated["ok"] and generated["summary"]["customers"] == 3
    map_data = client.get(f"/api/instances/{generated['instance_id']}/map-data").json()
    named = [feature["properties"]["poi_name"] for feature in map_data["geojson"]["features"]]
    assert "École Quatre" in named


def test_manual_generation_accepts_typed_picks_including_ways(client: TestClient) -> None:
    # The picker sends {type, id} pairs, because way 4 and node 4 are different
    # places that a bare id could not tell apart.
    body = {
        "city": "Testville",
        "method": "manual",
        "seed": 3,
        "manualPois": [
            {"type": "way", "id": 4},
            {"type": "relation", "id": 20},
            {"type": "node", "id": 2},
        ],
        "manualDepotPoi": {"type": "node", "id": 1},
    }
    preview = client.post("/api/workbench/generation/preview", json=body).json()
    assert preview["ok"]
    picked = [
        (feature["properties"]["poi_name"], feature["properties"]["poi_osm_type"])
        for feature in preview["geojson"]["features"]
    ]
    assert ("Le Quatre", "way") in picked
    assert ("Marché Vingt", "relation") in picked

    generated = client.post("/api/workbench/generation/single", json=body).json()
    assert generated["ok"] and generated["summary"]["customers"] == 3
    map_data = client.get(f"/api/instances/{generated['instance_id']}/map-data").json()
    types = {
        feature["properties"]["poi_osm_type"]
        for feature in map_data["geojson"]["features"]
        if feature["properties"]["poi_osm_type"]
    }
    assert {"way", "relation"} <= types

    unknown_type = client.post(
        "/api/workbench/generation/preview",
        json={**body, "manualPois": [{"type": "cloud", "id": 4}, {"type": "node", "id": 2}]},
    )
    assert unknown_type.json()["ok"] is False


def test_preflight_warns_before_topping_a_poi_request_up_parametrically(
    client: TestClient,
) -> None:
    body = {
        "city": "Testville",
        "nCustomers": 4,
        "seed": 5,
        "method": "poi_categories",
        "depotMode": "excentered",
        "categories": ["cafe"],
    }
    report = client.post("/api/workbench/generation/preflight", json=body).json()
    assert report["ok"]
    # The map is the preview's job; the pre-generation check is about the numbers.
    assert "geojson" not in report
    composition = report["summary"]["composition"]
    assert composition["poi_customers"] == 1 and composition["parametric_customers"] == 3
    assert composition["poi_pool_attachable"] == 1
    assert "parametric road points" in report["notice"]
    assert "select more POI categories" in report["notice"]

    # A request the categories can serve must not nag.
    served = client.post(
        "/api/workbench/generation/preflight",
        json={**body, "nCustomers": 2, "categories": ["restaurant", "cafe", "school", "bar"]},
    ).json()
    assert served["ok"] and served["notice"] is None
    assert served["summary"]["composition"]["parametric_customers"] == 0

    # The generated instance carries the same verdict, so the job report agrees
    # with what the user confirmed.
    generated = client.post("/api/workbench/generation/single", json=body).json()
    assert generated["ok"]
    assert generated["summary"]["composition"]["parametric_customers"] == 3
    assert "parametric road points" in generated["notice"]


def test_bulk_preflight_reports_sizes_the_pool_cannot_serve(client: TestClient) -> None:
    report = client.post(
        "/api/workbench/generation/bulk-preflight",
        json={
            "city": "Testville",
            "method": "parametric_attach",
            "instances": [
                {"city": "Testville", "nCustomers": 3},
                {"city": "Testville", "nCustomers": 900},
            ],
        },
    ).json()
    assert report["ok"] and report["instances"] == 2
    group = report["groups"][0]
    assert group["skipped_sizes"] == [900] and group["status"] == "partial"


def test_bulk_rows_generate_per_row_instances(client: TestClient) -> None:
    job = client.post(
        "/api/jobs",
        json={
            "kind": "bulk-generate",
            "payload": {
                "city": "Testville",
                "method": "parametric_attach",
                "seed": 0,
                "instances": [
                    {"city": "Testville", "nCustomers": 3, "demandType": 1, "avgRouteSize": 1},
                    {
                        "city": "Testville",
                        "nCustomers": 4,
                        "demandType": 5,
                        "avgRouteSize": 2,
                        "problemType": "vrptw",
                    },
                ],
            },
        },
    ).json()["job"]
    finished = _wait_for_job(client, job["job_id"])
    assert finished["status"] == "completed", finished
    result = finished["result"]
    assert result["generated"] == 2
    assert [bool(entry.get("vrptw")) for entry in result["results"]] == [False, True]
    # Per-row demand type and size really reached the generator.
    assert [entry["summary"]["demand_type"] for entry in result["results"]] == [1, 5]
    assert [entry["summary"]["customers"] for entry in result["results"]] == [3, 4]

    listed = {entry["base_name"] for entry in client.get("/api/workbench/instances").json()["instances"]}
    assert {entry["base_name"] for entry in result["results"]} <= listed


def _generate_for_import(client: TestClient) -> dict:
    body = {"city": "Testville", "nCustomers": 4, "seed": 7, "method": "parametric_attach"}
    return client.post("/api/workbench/generation/single", json=body).json()


def test_importing_an_external_solution_validates_before_storing(client: TestClient) -> None:
    instance = _generate_for_import(client)
    instance_id = instance["instance_id"]

    accepted = client.post(
        f"/api/instances/{instance_id}/solutions/import",
        json={
            "metric": "fastest",
            "text": "Route #1: 1 2\nRoute #2: 3 4\nCost 999\n",
            "filename": "external.sol",
            "label": "some-other-solver",
        },
    ).json()
    assert accepted["ok"], accepted
    run = accepted["solution"]
    assert run["source"] == "imported" and run["metadata"]["metric"] == "fastest"
    assert run["validation"]["valid"] and run["num_routes"] == 2
    # The declared cost was wrong, so the checked value is stored and flagged.
    assert accepted["warning"] and run["cost"] == run["validation"]["routing_cost"]

    runs = client.get(f"/api/instances/{instance_id}/solutions").json()["runs"]
    assert [entry["run_id"] for entry in runs] == [run["run_id"]]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("Route #1: 1 2\nRoute #2: 3 3\n", "more than once"),
        ("Route #1: 1 2\n", "never visited"),
        ("Route #1: 1 2 3 4 77\n", "fits neither"),
        ("nothing useful here\n", "No 'Route #k"),
        ("", "Paste a solution"),
    ],
)
def test_importing_a_broken_solution_is_refused(client: TestClient, text: str, message: str) -> None:
    instance = _generate_for_import(client)
    instance_id = instance["instance_id"]
    response = client.post(
        f"/api/instances/{instance_id}/solutions/import",
        json={"metric": "fastest", "text": text},
    )
    assert response.status_code == 400
    assert message in response.json()["error"]
    # Nothing is stored when the import is refused.
    assert client.get(f"/api/instances/{instance_id}/solutions").json()["runs"] == []


def test_importing_a_zero_based_solution_is_accepted(client: TestClient) -> None:
    """Files numbering customers 0..n-1 must be understood, not rejected as
    incomplete because their customer 0 was mistaken for a depot marker."""
    instance = _generate_for_import(client)
    accepted = client.post(
        f"/api/instances/{instance['instance_id']}/solutions/import",
        json={"metric": "fastest", "text": "Route #1: 0 1\nRoute #2: 2 3\n"},
    ).json()
    assert accepted["ok"], accepted
    # Shifted onto model ids 1..n, every customer visited exactly once.
    assert sorted(stop for route in accepted["solution"]["routes"] for stop in route) == [1, 2, 3, 4]


def test_importing_a_one_based_solution_with_explicit_depots_is_accepted(
    client: TestClient,
) -> None:
    """The other convention: 1..n with 0 written at the route boundaries."""
    instance = _generate_for_import(client)
    accepted = client.post(
        f"/api/instances/{instance['instance_id']}/solutions/import",
        json={"metric": "fastest", "text": "Route #1: 0 1 2 0\nRoute #2: 0 3 4 0\n"},
    ).json()
    assert accepted["ok"], accepted
    assert accepted["solution"]["routes"] == [[1, 2], [3, 4]]


def test_poi_endpoint_caps_how_many_features_it_returns(client: TestClient) -> None:
    """A real extract holds tens of thousands of amenities; the picker must not
    be handed all of them at once."""
    capped = client.get(
        "/api/workbench/generation/pois", params={"city": "Testville", "limit": 2}
    ).json()
    assert capped["returned"] == 2 and capped["truncated"] is True
    assert capped["matching"] > 2
    assert len(capped["geojson"]["features"]) == 2

    full = client.get("/api/workbench/generation/pois", params={"city": "Testville"}).json()
    assert full["truncated"] is False and full["returned"] == full["matching"]


def test_osmdata_audit_flags_extracts_missing_outline_pois(
    client: TestClient, tmp_path: Path
) -> None:
    # A second extract in the shape produced before ways were fetched.
    (osmdata_dir(tmp_path / "workspace") / "Oldtown.osm").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6">\n'
        '  <bounds minlat="44.99" minlon="3.99" maxlat="45.01" maxlon="4.01"/>\n'
        '<node id="1" lat="45.0" lon="4.0"><tag k="amenity" v="cafe"/></node>\n'
        '<way id="10"><nd ref="1"/><tag k="highway" v="residential"/></way>\n'
        "</osm>\n",
        encoding="utf-8",
    )
    report = client.get("/api/workbench/osmdata/audit").json()
    assert report["ok"]
    by_city = {entry["city"]: entry for entry in report["extracts"]}
    assert by_city["Testville"]["status"] == "complete"
    assert by_city["Testville"]["poi_ways"] == 1
    assert by_city["Oldtown"]["status"] == "nodes_only"
    assert report["outdated"] == 1


def test_refresh_pois_job_backfills_the_outdated_extracts(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mamut_routing_tools.osm import fetch as fetch_module

    (osmdata_dir(tmp_path / "workspace") / "Oldtown.osm").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6">\n'
        '  <bounds minlat="44.99" minlon="3.99" maxlat="45.01" maxlon="4.01"/>\n'
        '<node id="1" lat="45.0" lon="4.0"><tag k="amenity" v="cafe"/></node>\n'
        '<way id="10"><nd ref="1"/><tag k="highway" v="residential"/></way>\n'
        "</osm>\n",
        encoding="utf-8",
    )
    queries: list[str] = []

    def fake_fetch(query: str, **kwargs):  # type: ignore[no-untyped-def]
        queries.append(query)
        return fetch_module.OverpassResult(
            body='<?xml version="1.0"?><osm version="0.6">'
            '<way id="88"><center lat="45.001" lon="4.001"/>'
            '<tag k="amenity" v="restaurant"/><tag k="name" v="Le Contour"/></way></osm>'
        )

    monkeypatch.setattr(fetch_module, "fetch_overpass_body", fake_fetch)
    submitted = client.post(
        "/api/jobs", json={"kind": "refresh-pois", "payload": {}}
    ).json()
    job = _wait_for_job(client, submitted["job"]["job_id"])

    assert job["status"] == "completed", job.get("error")
    result = job["result"]
    # Only the extract that needed it: Testville already holds way-mapped POIs.
    assert [entry["city"] for entry in result["results"]] == ["Oldtown"]
    assert result["refreshed"] == 1 and result["gained"] >= 1
    assert queries and all("highway" not in query for query in queries)

    after = client.get("/api/workbench/osmdata/audit").json()
    assert {entry["status"] for entry in after["extracts"]} == {"complete"}
    assert after["outdated"] == 0
    # The refilled POI is immediately selectable in the picker.
    pois = client.get(
        "/api/workbench/generation/pois", params={"city": "Oldtown", "categories": "restaurant"}
    ).json()
    assert [feature["properties"]["name"] for feature in pois["geojson"]["features"]] == [
        "Le Contour"
    ]


def test_poi_attach_mode_travels_from_the_payload_to_the_instance(client: TestClient) -> None:
    body = {
        "city": "Testville",
        "nCustomers": 3,
        "seed": 5,
        "method": "poi_categories",
        "depotMode": "excentered",
        "categories": ["bank", "cafe", "restaurant"],
    }
    # The bank's nearest road node is not an intersection, so the strict rule
    # cannot serve it and the request is topped up parametrically.
    strict = client.post("/api/workbench/generation/preflight", json=body).json()
    assert strict["summary"]["composition"]["parametric_customers"] == 1
    assert "switch POI attachment to 'nearest vertex'" in strict["notice"]

    snapped = client.post(
        "/api/workbench/generation/preflight",
        json={**body, "poiAttachMode": "nearest_vertex", "poiAttachRadiusM": 400},
    ).json()
    assert snapped["ok"] and snapped["notice"] is None
    assert snapped["summary"]["composition"]["parametric_customers"] == 0

    generated = client.post(
        "/api/workbench/generation/single",
        json={**body, "poiAttachMode": "nearest_vertex", "poiAttachRadiusM": 400},
    ).json()
    assert generated["ok"]
    map_data = client.get(f"/api/instances/{generated['instance_id']}/map-data").json()
    snaps = [feature["properties"]["snap_distance_m"] for feature in map_data["geojson"]["features"]]
    assert any(distance > 300 for distance in snaps), "the moved POI records how far it went"

    rejected = client.post(
        "/api/workbench/generation/preflight", json={**body, "poiAttachMode": "teleport"}
    )
    assert rejected.json()["ok"] is False
