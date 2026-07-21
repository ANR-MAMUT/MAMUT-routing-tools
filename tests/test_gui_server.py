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
    assert "Excentered (corner)" in html
    assert "Instance only · customer locations" in html
    assert "City fetches store every listed category" in html
    assert "poiCategories: [...POI_CATEGORIES]" in html


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
        "categories": ["hospital", "library"],
    }
    preview = client.post("/api/workbench/generation/preview", json=body).json()
    assert preview["ok"] and len(preview["geojson"]["features"]) == 5

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


def test_td_build_is_explicitly_unsupported(client: TestClient) -> None:
    assert client.post("/api/workbench/generation/td-build", json={}).status_code == 501


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
