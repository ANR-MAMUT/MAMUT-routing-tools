"""Route-geometry materialization contract on the synthetic city."""

from __future__ import annotations

from pathlib import Path

from mamut_routing_tools.geometry.materialize import (
    EdgeAnchor,
    materialize_group,
    node_edge_cache_key,
    point_distance_meters,
)
from mamut_routing_tools.roadgraph.build import clear_caches, load_road_graph


def test_point_distance_meters_lonlat() -> None:
    # 0.001 degrees of latitude is about 111 m.
    assert abs(point_distance_meters([4.0, 45.0], [4.0, 45.001]) - 111.32) < 1.0


def test_materialize_group_roads_and_fallback(fixture_osm_path: Path, tmp_path: Path) -> None:
    clear_caches()
    meta = {
        "source_osm_file": fixture_osm_path.name,
        "map_options": {"only_intersections": True, "trim_to_connected_graph": True},
        "nodes": [
            {"instance_node_id": 0, "poi_lon": 4.000, "poi_lat": 45.000},
            {"instance_node_id": 1, "poi_lon": 4.008, "poi_lat": 45.000},
            {"instance_node_id": 2, "poi_lon": 4.001, "poi_lat": 44.9995},
        ],
    }
    group = {
        "result_file": "group-000.json",
        "geo_path": fixture_osm_path.name,
        "metric": "shortest",
        "meta": meta,
        "entries": [
            {"bks_path": "some/road.bks.json", "routes": [[1]]},
            {"bks_path": "some/faraway.bks.json", "routes": [[2]]},
        ],
    }
    result = materialize_group(tmp_path, group)

    road_forward = result["edge_cache"][node_edge_cache_key(0, 1)]
    assert len(road_forward) == 3  # follows the road through node 2
    assert point_distance_meters(road_forward[0], [4.000, 45.000]) < 10
    assert point_distance_meters(road_forward[-1], [4.008, 45.000]) < 10
    road_back = result["edge_cache"][node_edge_cache_key(1, 0)]
    assert road_back == list(reversed(road_forward))

    # Instance node 2 sits 55 m off the road but its nearest road node is
    # node 1 again: the segment 0 -> 2 would end ~140 m from node 2's
    # coordinates... it actually maps to node 1, so routing yields a
    # single-vertex path that fails the endpoint test and falls back.
    assert node_edge_cache_key(0, 2) in result["straight_fallback_edges"]
    straight = result["edge_cache"][node_edge_cache_key(0, 2)]
    assert straight == [[4.000, 45.000], [4.001, 44.9995]]

    entry_keys = {entry["bks_path"]: entry["edge_keys"] for entry in result["entries"]}
    assert entry_keys["some/road.bks.json"] == sorted([node_edge_cache_key(0, 1), node_edge_cache_key(1, 0)])


def test_mid_segment_node_anchors_to_the_road(fixture_osm_path: Path, tmp_path: Path) -> None:
    """A node on a road but ~200 m from every road node (no node in its 100 m
    box) is projected onto the segment and routed from there instead of
    falling back to a straight line. This is the Chartres depot: a synthetic
    crop node of the generation-time extract that the re-fetched extract no
    longer has."""
    clear_caches()
    # Way 10 runs 4.000 -> 4.005 -> 4.008 along lat 45; 4.0025 is the middle of
    # its first segment, ~196 m from both node 1 and node 2.
    mid = {"instance_node_id": 0, "poi_lon": 4.0025, "poi_lat": 45.0000}
    group = {
        "result_file": "group-000.json",
        "geo_path": fixture_osm_path.name,
        "metric": "shortest",
        "meta": {
            "source_osm_file": fixture_osm_path.name,
            "map_options": {"only_intersections": True, "trim_to_connected_graph": True},
            "nodes": [mid, {"instance_node_id": 1, "poi_lon": 4.008, "poi_lat": 45.000}],
        },
        "entries": [{"bks_path": "some/mid.bks.json", "routes": [[1]]}],
    }
    result = materialize_group(tmp_path, group)

    assert result["straight_fallback_edges"] == []
    out = result["edge_cache"][node_edge_cache_key(0, 1)]
    # Starts at the projection (on the road, at the node), passes node 2, ends at node 3.
    assert point_distance_meters(out[0], [4.0025, 45.0]) < 1.0
    assert point_distance_meters(out[1], [4.005, 45.0]) < 1.0
    assert point_distance_meters(out[-1], [4.008, 45.0]) < 1.0
    assert len(out) == 3
    back = result["edge_cache"][node_edge_cache_key(1, 0)]
    assert back == list(reversed(out))


def test_edge_projection_only_on_node_level_graphs(fixture_osm_path: Path) -> None:
    clear_caches()
    intersections = load_road_graph(fixture_osm_path, only_intersections=True, trim_to_connected=True)
    assert intersections.nearest_edge_projection(45.0, 4.0025) is None
    full = load_road_graph(fixture_osm_path, only_intersections=False, trim_to_connected=True)
    projection = full.nearest_edge_projection(45.0, 4.0025)
    assert projection is not None
    edge_index, fraction, distance = projection
    assert {full.edges[edge_index][0], full.edges[edge_index][1]} == {1, 2}
    assert abs(fraction - 0.5) < 0.01
    assert distance < 1.0
    # 60 m north of the road: outside the anchor radius.
    assert full.nearest_edge_projection(45.00054, 4.0025) is None
    assert isinstance(EdgeAnchor(edge_index, fraction, [4.0025, 45.0]), EdgeAnchor)
