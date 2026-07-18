"""Road-graph construction semantics on a synthetic city."""

from __future__ import annotations

from pathlib import Path

from mamut_routing_tools.roadgraph.build import build_road_graph, clear_caches, load_road_graph
from mamut_routing_tools.roadgraph.osmxml import OsmWay, crop_to_bounds, parse_osm
from mamut_routing_tools.roadgraph.router import route_lonlat


def test_crop_trim_and_intersection_graph(fixture_osm_path: Path) -> None:
    clear_caches()
    graph = load_road_graph(fixture_osm_path, only_intersections=True, trim_to_connected=True)

    # The footway and the invisible way are filtered; the island (7, 8) and
    # the boundary-crossing oneway spur fall to the SCC trim; the cross
    # 1-2-3 / 4-2-5 splits at the shared interior node 2.
    assert sorted(graph.vertex_of) == [1, 2, 3, 4, 5]
    assert graph.edge_count == 8

    class_counts: dict[int, int] = {}
    for cls in graph.edge_class:
        class_counts[cls] = class_counts.get(cls, 0) + 1
    assert class_counts == {6: 4, 3: 4}

    # 0.005 degrees of longitude at latitude 45 is about 393 m.
    edge_length = graph.edge_weight[graph.edges.index((1, 2))]
    assert 380 < edge_length < 405

    segment = route_lonlat(graph, graph.vertex_of[1], graph.vertex_of[3], "shortest")
    assert segment is not None
    assert len(segment) == 3  # 1 -> 2 -> 3


def test_crop_creates_boundary_node(fixture_osm_path: Path) -> None:
    osm_data = parse_osm(fixture_osm_path)
    crop_to_bounds(osm_data)
    way12 = next(way for way in osm_data.ways if way.way_id == 12)
    assert len(way12.nodes) == 2
    synthetic = way12.nodes[1]
    assert synthetic not in range(1, 9)
    lat, lon = osm_data.nodes[synthetic]
    assert lon == 4.01  # clipped exactly on the east boundary
    assert abs(lat - 45.0) < 1e-9
    assert 6 not in osm_data.nodes


def test_oneway_directions(fixture_osm_path: Path) -> None:
    osm_data = parse_osm(fixture_osm_path)
    crop_to_bounds(osm_data)
    graph = build_road_graph(osm_data, fixture_osm_path, only_intersections=True, trim_to_connected=False)
    synthetic = next(way for way in osm_data.ways if way.way_id == 12).nodes[1]
    assert (3, synthetic) in graph.edges  # oneway=yes: forward only
    assert (synthetic, 3) not in graph.edges

    reversed_ways = [
        OsmWay(way_id=way.way_id, nodes=list(way.nodes), tags={**way.tags, "oneway": "-1"} if way.way_id == 12 else way.tags)
        for way in osm_data.ways
    ]
    osm_data.ways = reversed_ways
    graph = build_road_graph(osm_data, fixture_osm_path, only_intersections=True, trim_to_connected=False)
    assert (synthetic, 3) in graph.edges  # oneway=-1: reversed only
    assert (3, synthetic) not in graph.edges


def test_nearest_node_uses_box_inclusion(fixture_osm_path: Path) -> None:
    clear_caches()
    graph = load_road_graph(fixture_osm_path, only_intersections=True, trim_to_connected=True)
    # 72 m east and 72 m north of node 1: 102 m Euclidean, but inside the
    # +-100 m box, so it must match (the findnode contract).
    lat = 45.000 + 72.0 / 111_320.0
    lon = 4.000 + 72.0 / (111_320.0 * 0.7071067811865476)
    assert graph.nearest_node(lat, lon) == 1
    # 120 m straight east: outside the box, no match.
    lon_far = 4.000 + 120.0 / (111_320.0 * 0.7071067811865476)
    assert graph.nearest_node(45.000, lon_far) is None
