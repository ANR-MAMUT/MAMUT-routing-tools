"""GUI server tests: security guards and the workbench endpoint shapes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mamut_routing_tools.gui.server import create_app
from mamut_routing_tools.workspace import osmdata_dir

TOKEN = "test-token"


@pytest.fixture
def client(tmp_path: Path, fixture_osm_path: Path) -> TestClient:
    workspace = tmp_path / "workspace"
    (osmdata_dir(workspace)).mkdir(parents=True, exist_ok=True)
    (osmdata_dir(workspace) / "Testville.osm").write_text(fixture_osm_path.read_text())
    app = create_app(workspace, TOKEN)
    return TestClient(app, base_url="http://localhost", headers={"X-Mamut-Token": TOKEN})


def test_requests_without_token_or_wrong_host_are_rejected(client: TestClient) -> None:
    assert client.get("/healthz", headers={"X-Mamut-Token": "wrong"}).status_code == 403
    assert client.get("/healthz", headers={"Host": "evil.example"}).status_code == 403
    assert client.get("/healthz").status_code == 200


def test_cities_endpoint_lists_workspace_extracts(client: TestClient) -> None:
    payload = client.get("/api/workbench/generation/cities").json()
    assert payload["ok"] and payload["preview_available"]
    assert [city["slug"] for city in payload["cities"]] == ["testville"]


def test_preview_generate_solve_render_round_trip(client: TestClient) -> None:
    body = {"city": "Testville", "nCustomers": 4, "seed": 7, "method": "parametric_attach"}
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

    vrp_json = client.get(
        "/instances-file", params={"path": f"{folder}/{single['files']['vrp_json']['fastest']}"}
    ).json()
    solved = client.post("/api/workbench/solve", json={"vrp_json": vrp_json, "time_limit": 1}).json()
    assert solved["ok"] and solved["n_routes"] >= 1 and solved["input_source"] == "vrp_json"
    assert sorted(stop for route in solved["routes"] for stop in route) == [1, 2, 3, 4]

    meta = client.get("/instances-file", params={"path": f"{folder}/{single['files']['meta']}"}).json()
    rendered = client.post(
        "/api/workbench/render-routes", json={"meta": meta, "routes": solved["routes"], "metric": "fastest"}
    ).json()
    assert rendered["ok"] and rendered["summary"]["render_mode"] in ("cached_road", "mixed")

    download = client.post(
        "/api/workbench/generation/single-download",
        json={"folder": folder, "base_name": single["base_name"]},
    )
    assert download.status_code == 200 and download.headers["content-type"] == "application/zip"


def test_td_build_is_explicitly_unsupported(client: TestClient) -> None:
    assert client.post("/api/workbench/generation/td-build", json={}).status_code == 501


def test_instances_file_refuses_paths_outside_the_workspace(client: TestClient) -> None:
    assert client.get("/instances-file", params={"path": "/etc/passwd"}).status_code == 404
