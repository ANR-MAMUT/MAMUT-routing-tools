"""Route-geometry materialization contract on the synthetic city."""

from __future__ import annotations

from pathlib import Path

from mamut_routing_tools.geometry.materialize import (
    materialize_group,
    node_edge_cache_key,
    point_distance_meters,
)
from mamut_routing_tools.roadgraph.build import clear_caches


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
