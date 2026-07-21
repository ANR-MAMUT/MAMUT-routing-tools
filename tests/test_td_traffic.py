"""Wave traffic model and TD-bridge emitters on the synthetic city.

The structural assertions mirror the publisher's bridge loaders (5-tuple
edges, 3-tuple vertices, positive lengths/speeds, one profile per edge, 24
bins per profile), so a passing test guarantees the emitted files round-trip
through those loaders without importing the publisher.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mamut_routing_tools.roadgraph.build import SPEED_ROADS_URBAN, clear_caches, load_road_graph
from mamut_routing_tools.td.traffic import (
    TD_MIN_SPEED_FACTOR,
    TD_NUM_BINS,
    TrafficModelError,
    bridge_seed,
    collect_edges,
    export_bridge,
    td_free_speed_ms,
    td_rush_curve,
    vertex_latlon,
    wave_speeds,
)


def _assert_graph_file_loadable(payload: dict) -> list[tuple]:
    """Replicate the publisher's ``load_bridge_graph`` checks."""
    assert payload["schema_version"] == 2
    assert payload["num_bins"] == TD_NUM_BINS
    assert float(payload["bin_seconds"]) == 3600.0
    edges = []
    for entry in payload["edges"]:
        assert len(entry) == 5
        osm_u, osm_v, length_m, road_class, free_speed = entry
        assert float(length_m) > 0
        assert float(free_speed) > 0
        edges.append((int(osm_u), int(osm_v), float(length_m), int(road_class), float(free_speed)))
    assert edges
    coords: dict[int, tuple[float, float]] = {}
    for entry in payload["vertices"]:
        assert len(entry) == 3
        osm_id, lon, lat = entry
        coords[int(osm_id)] = (float(lon), float(lat))
    for osm_u, osm_v, *_ in edges:
        assert osm_u in coords and osm_v in coords
    return edges


def _assert_speeds_file_loadable(payload: dict, n_edges: int) -> list[list[float]]:
    """Replicate the publisher's ``load_bridge_speeds`` checks."""
    assert payload["schema_version"] == 2
    speeds = [[float(v) for v in row] for row in payload["speeds"]]
    assert len(speeds) == n_edges
    for row in speeds:
        assert len(row) == TD_NUM_BINS
        assert all(v > 0 for v in row)
    return speeds


def test_rush_curve_shape() -> None:
    # Peaks near the morning (bin 8, 08:00-09:00) and evening (bin 17) rushes;
    # near zero in the small hours; capped at 1.
    assert td_rush_curve(3) < 0.05  # 03:00-04:00
    assert td_rush_curve(8) > 0.8  # 08:00-09:00, morning peak
    assert td_rush_curve(17) > 0.6  # 17:00-18:00, evening peak
    assert all(0.0 <= td_rush_curve(b) <= 1.0 for b in range(TD_NUM_BINS))


def test_collect_edges_matches_road_graph(fixture_osm_path: Path) -> None:
    clear_caches()
    graph = load_road_graph(fixture_osm_path, only_intersections=True, trim_to_connected=True)
    edges = collect_edges(graph)
    # RoadGraph.edges is already unique per (u, v); no self-loops, all positive.
    assert len(edges) == graph.edge_count
    assert len({(e.u, e.v) for e in edges}) == len(edges)
    assert all(e.u != e.v and e.length_m > 0 for e in edges)
    # Sorted by (u, v) so speed rows align by position.
    assert [(e.u, e.v) for e in edges] == sorted((e.u, e.v) for e in edges)
    for edge in edges:
        assert graph.node_of[edge.u] == edge.osm_u
        assert graph.node_of[edge.v] == edge.osm_v


def test_wave_profiles_structural(fixture_osm_path: Path) -> None:
    clear_caches()
    graph = load_road_graph(fixture_osm_path, only_intersections=True, trim_to_connected=True)
    edges = collect_edges(graph)
    vertex_ll = vertex_latlon(graph)
    center = (sum(t[0] for t in vertex_ll) / len(vertex_ll), sum(t[1] for t in vertex_ll) / len(vertex_ll))

    speeds = wave_speeds(edges, vertex_ll, center, "moderate", bridge_seed(42, "wave", "moderate"))
    assert len(speeds) == len(edges)
    for edge, profile in zip(edges, speeds):
        assert len(profile) == TD_NUM_BINS
        free = td_free_speed_ms(edge.road_class)
        floor = round(free * TD_MIN_SPEED_FACTOR, 3)
        assert all(v > 0 for v in profile)
        # Never below the free-flow floor (bar rounding).
        assert min(profile) >= floor - 1e-6
        # Rush dips are non-negative: peak bins are no faster than the small
        # hours (same per-edge jitter cancels out).
        assert profile[8] <= profile[3] + 1e-9  # morning peak <= 03:00
        assert profile[17] <= profile[3] + 1e-9  # evening peak <= 03:00


def test_wave_intensity_deepens_dip(fixture_osm_path: Path) -> None:
    clear_caches()
    graph = load_road_graph(fixture_osm_path, only_intersections=True, trim_to_connected=True)
    edges = collect_edges(graph)
    vertex_ll = vertex_latlon(graph)
    center = (sum(t[0] for t in vertex_ll) / len(vertex_ll), sum(t[1] for t in vertex_ll) / len(vertex_ll))

    # Same seed per combination via bridge_seed so only the amplitude changes;
    # compare the morning-peak speed of the busiest (most central) edge.
    def peak_min(intensity: str) -> float:
        speeds = wave_speeds(edges, vertex_ll, center, intensity, bridge_seed(42, "wave", intensity))
        return min(profile[8] for profile in speeds)

    assert peak_min("heavy") <= peak_min("moderate") <= peak_min("light")


def test_wave_is_seed_deterministic(fixture_osm_path: Path) -> None:
    clear_caches()
    graph = load_road_graph(fixture_osm_path, only_intersections=True, trim_to_connected=True)
    edges = collect_edges(graph)
    vertex_ll = vertex_latlon(graph)
    center = (sum(t[0] for t in vertex_ll) / len(vertex_ll), sum(t[1] for t in vertex_ll) / len(vertex_ll))

    a = wave_speeds(edges, vertex_ll, center, "moderate", 12345)
    b = wave_speeds(edges, vertex_ll, center, "moderate", 12345)
    c = wave_speeds(edges, vertex_ll, center, "moderate", 999)
    assert a == b
    assert a != c


def test_export_bridge_round_trips(fixture_osm_path: Path, tmp_path: Path) -> None:
    clear_caches()
    out_root = tmp_path / "td-bridge"
    out_dir = export_bridge(
        osm_path=fixture_osm_path,
        city_slug="Testville",
        out_root=out_root,
        models=["wave"],
        intensities=["light", "moderate", "heavy"],
        seed=42,
    )
    assert out_dir == out_root / "Testville"

    graph_payload_json = json.loads((out_dir / "graph.json").read_text())
    edges = _assert_graph_file_loadable(graph_payload_json)
    assert graph_payload_json["city"] == "Testville"
    assert graph_payload_json["osm_file"] == "Testville.osm"

    for intensity in ("light", "moderate", "heavy"):
        speeds_json = json.loads((out_dir / f"speeds-wave-{intensity}.json").read_text())
        _assert_speeds_file_loadable(speeds_json, len(edges))
        assert speeds_json["model"] == "wave"
        assert speeds_json["intensity"] == intensity
        assert speeds_json["seed"] == bridge_seed(42, "wave", intensity)
        assert speeds_json["num_trips"] == 0

    manifest = json.loads((out_dir / "bridge-manifest.json").read_text())
    assert manifest["num_edges"] == len(edges)
    assert len(manifest["speed_files"]) == 3

    # free_speed_ms in graph.json is derived purely from the road class.
    for osm_u, osm_v, length_m, road_class, free_speed in edges:
        assert free_speed == pytest.approx(round(SPEED_ROADS_URBAN[road_class] / 3.6, 3))


def test_export_bridge_reuses_unless_forced(fixture_osm_path: Path, tmp_path: Path) -> None:
    clear_caches()
    out_root = tmp_path / "td-bridge"
    export_bridge(osm_path=fixture_osm_path, city_slug="Testville", out_root=out_root, models=["wave"], intensities=["moderate"])
    speeds_path = out_root / "Testville" / "speeds-wave-moderate.json"
    marker = speeds_path.read_text()
    speeds_path.write_text(json.dumps({"schema_version": 2, "sentinel": True, "speeds": []}))

    # Without force, the existing file is kept.
    export_bridge(osm_path=fixture_osm_path, city_slug="Testville", out_root=out_root, models=["wave"], intensities=["moderate"])
    assert json.loads(speeds_path.read_text()).get("sentinel") is True

    # With force, it is regenerated.
    export_bridge(osm_path=fixture_osm_path, city_slug="Testville", out_root=out_root, models=["wave"], intensities=["moderate"], force=True)
    assert "sentinel" not in speeds_path.read_text()
    assert json.loads(speeds_path.read_text())["speeds"]


def test_unknown_model_and_intensity_rejected(fixture_osm_path: Path, tmp_path: Path) -> None:
    with pytest.raises(TrafficModelError):
        export_bridge(osm_path=fixture_osm_path, city_slug="X", out_root=tmp_path, models=["nope"], intensities=["light"])
    with pytest.raises(TrafficModelError):
        export_bridge(osm_path=fixture_osm_path, city_slug="X", out_root=tmp_path, models=["wave"], intensities=["extreme"])


def test_bpr_not_yet_ported(fixture_osm_path: Path, tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        export_bridge(osm_path=fixture_osm_path, city_slug="X", out_root=tmp_path, models=["bpr"], intensities=["light"])
