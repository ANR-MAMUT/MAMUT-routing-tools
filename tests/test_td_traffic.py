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
from mamut_routing_tools.td import (
    build_bridge,
    load_bridge_graph,
    load_bridge_nodes,
    load_bridge_speeds,
)
from mamut_routing_tools.td.traffic import (
    TD_MIN_SPEED_FACTOR,
    TD_NUM_BINS,
    TrafficModelError,
    _bpr_profiles,
    bpr_speeds,
    bpr_work_pool,
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


def test_bpr_work_pool_falls_back_to_all_vertices(fixture_osm_path: Path) -> None:
    # The synthetic city has no amenity POIs, so the pool falls back to all
    # graph vertices (uniform workplaces), like the Julia td_work_pool.
    clear_caches()
    graph = load_road_graph(fixture_osm_path, only_intersections=True, trim_to_connected=True)
    assert bpr_work_pool(graph, fixture_osm_path) == list(range(graph.vertex_count))


def test_bpr_volume_delay_math(fixture_osm_path: Path) -> None:
    # Deterministic check of the BPR function itself, independent of the RNG:
    # zero flow -> free flow; saturating flow -> the multiplier cap.
    import numpy as np

    clear_caches()
    graph = load_road_graph(fixture_osm_path, only_intersections=True, trim_to_connected=True)
    edges = collect_edges(graph)

    zero = _bpr_profiles(edges, np.zeros((len(edges), TD_NUM_BINS)))
    for edge, profile in zip(edges, zero):
        free = round(td_free_speed_ms(edge.road_class), 3)
        assert all(v == free for v in profile)  # multiplier == 1 everywhere

    saturating = _bpr_profiles(edges, np.full((len(edges), TD_NUM_BINS), 1e9))
    for edge, profile in zip(edges, saturating):
        free = td_free_speed_ms(edge.road_class)
        capped = round(max(free / 6.0, free * TD_MIN_SPEED_FACTOR), 3)  # cap = 6.0
        assert all(v == capped for v in profile)


def test_bpr_speeds_never_exceed_free_flow(fixture_osm_path: Path) -> None:
    clear_caches()
    graph = load_road_graph(fixture_osm_path, only_intersections=True, trim_to_connected=True)
    edges = collect_edges(graph)
    speeds, num_trips = bpr_speeds(graph, edges, fixture_osm_path, "heavy", bridge_seed(42, "bpr", "heavy"))
    assert num_trips > 0
    assert len(speeds) == len(edges)
    for edge, profile in zip(edges, speeds):
        free = td_free_speed_ms(edge.road_class)
        assert len(profile) == TD_NUM_BINS
        # BPR only slows traffic: speeds never exceed free flow (bar rounding),
        # and never fall below the cap-implied minimum free/6.
        assert max(profile) <= round(free, 3) + 1e-9
        assert min(profile) >= round(free / 6.0, 3) - 1e-6


def test_bpr_is_seed_deterministic(fixture_osm_path: Path) -> None:
    clear_caches()
    graph = load_road_graph(fixture_osm_path, only_intersections=True, trim_to_connected=True)
    edges = collect_edges(graph)
    a, ta = bpr_speeds(graph, edges, fixture_osm_path, "heavy", 777)
    b, tb = bpr_speeds(graph, edges, fixture_osm_path, "heavy", 777)
    assert a == b and ta == tb


def _fake_meta(graph, vertices: list[int], base: str) -> dict:
    """A minimal stage-1 meta for the given graph vertices (depot first)."""
    return {
        "schema_version": 2,
        "city": "Testville",
        "instance_name": base,
        "map_options": {"only_intersections": True, "trim_to_connected_graph": True},
        "nodes": [
            {
                "instance_node_id": i + 1,
                "graph_vertex_id": v,
                "poi_lat": 0.0,
                "poi_lon": 0.0,
                "enu_x": 0.0,
                "enu_y": 0.0,
                "demand": 0 if i == 0 else 5,
                "source_tag": "depot" if i == 0 else "param",
            }
            for i, v in enumerate(vertices)
        ],
    }


def test_nodes_emitter_maps_to_osm_ids(fixture_osm_path: Path, tmp_path: Path) -> None:
    clear_caches()
    graph = load_road_graph(fixture_osm_path, only_intersections=True, trim_to_connected=True)
    vertices = [0, 1, 2]
    base = "Testville_par-n2-k1"
    meta_path = tmp_path / f"{base}_meta.json"
    meta_path.write_text(json.dumps(_fake_meta(graph, vertices, base)))

    out_dir = export_bridge(
        osm_path=fixture_osm_path,
        city_slug="Testville",
        out_root=tmp_path / "td-bridge",
        models=["wave"],
        intensities=["moderate"],
        meta_paths=[meta_path],
    )
    nodes_json = json.loads((out_dir / f"nodes-{base}.json").read_text())
    assert nodes_json["schema_version"] == 2
    assert nodes_json["instance_base"] == base
    assert nodes_json["depot_first"] is True
    assert nodes_json["node_osm_ids"] == [graph.node_of[v] for v in vertices]
    # The publisher's load_bridge_nodes contract: >= 2, distinct, depot first.
    assert len(nodes_json["node_osm_ids"]) >= 2
    assert len(set(nodes_json["node_osm_ids"])) == len(nodes_json["node_osm_ids"])
    manifest = json.loads((out_dir / "bridge-manifest.json").read_text())
    assert manifest["node_files"] == [f"nodes-{base}.json"]


def test_nodes_emitter_rejects_option_mismatch(fixture_osm_path: Path, tmp_path: Path) -> None:
    clear_caches()
    graph = load_road_graph(fixture_osm_path, only_intersections=True, trim_to_connected=True)
    meta = _fake_meta(graph, [0, 1], "Testville_par-n1-k1")
    meta["map_options"]["only_intersections"] = False  # numbering would not align
    meta_path = tmp_path / "bad_meta.json"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(TrafficModelError):
        export_bridge(
            osm_path=fixture_osm_path,
            city_slug="Testville",
            out_root=tmp_path / "td-bridge",
            models=["wave"],
            intensities=["moderate"],
            meta_paths=[meta_path],
        )


def test_nodes_emitter_rejects_out_of_range_vertex(fixture_osm_path: Path, tmp_path: Path) -> None:
    clear_caches()
    graph = load_road_graph(fixture_osm_path, only_intersections=True, trim_to_connected=True)
    meta = _fake_meta(graph, [0, graph.vertex_count + 10], "Testville_par-n1-k1")
    meta_path = tmp_path / "oob_meta.json"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(TrafficModelError):
        export_bridge(
            osm_path=fixture_osm_path,
            city_slug="Testville",
            out_root=tmp_path / "td-bridge",
            models=["wave"],
            intensities=["moderate"],
            meta_paths=[meta_path],
        )


def test_build_bridge_matches_disk_round_trip(fixture_osm_path: Path, tmp_path: Path) -> None:
    # The streamlined in-memory build must equal the serialize-then-load path,
    # so the per-instance derivation (build_bridge) and any cached disk export
    # (export_bridge + load_bridge_*) never diverge. Both models exercised.
    clear_caches()
    graph = load_road_graph(fixture_osm_path, only_intersections=True, trim_to_connected=True)
    base = "Testville_par-n2-k1"
    meta = _fake_meta(graph, [0, 1, 2], base)
    meta_path = tmp_path / f"{base}_meta.json"
    meta_path.write_text(json.dumps(meta))
    models = ["wave", "bpr"]
    intensities = ["moderate"]

    built = build_bridge(
        osm_path=fixture_osm_path,
        city_slug="Testville",
        models=models,
        intensities=intensities,
        metas=[meta],
    )

    out_dir = export_bridge(
        osm_path=fixture_osm_path,
        city_slug="Testville",
        out_root=tmp_path / "td-bridge",
        models=models,
        intensities=intensities,
        meta_paths=[meta_path],
    )
    disk_graph = load_bridge_graph(out_dir / "graph.json")
    disk_nodes = load_bridge_nodes(out_dir / f"nodes-{base}.json")

    assert built.graph == disk_graph
    assert built.nodes[base] == disk_nodes
    for model in models:
        for intensity in intensities:
            disk_speeds = load_bridge_speeds(out_dir / f"speeds-{model}-{intensity}.json", disk_graph)
            assert built.speeds[(model, intensity)] == disk_speeds


def test_export_bridge_bpr_round_trips(fixture_osm_path: Path, tmp_path: Path) -> None:
    clear_caches()
    out_dir = export_bridge(
        osm_path=fixture_osm_path,
        city_slug="Testville",
        out_root=tmp_path / "td-bridge",
        models=["bpr"],
        intensities=["moderate"],
    )
    graph_json = json.loads((out_dir / "graph.json").read_text())
    edges = _assert_graph_file_loadable(graph_json)
    speeds_json = json.loads((out_dir / "speeds-bpr-moderate.json").read_text())
    _assert_speeds_file_loadable(speeds_json, len(edges))
    assert speeds_json["model"] == "bpr"
    assert speeds_json["seed"] == bridge_seed(42, "bpr", "moderate")
    assert speeds_json["num_trips"] > 0
