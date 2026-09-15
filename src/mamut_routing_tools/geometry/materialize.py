"""Route-geometry materialization: the engine behind the website's
hash-addressed BKS road geometry.

Consumes the same group plan the website's ``route_geometry.py`` used to
hand to the Julia materializer, and produces byte-compatible result payloads:
per group, an ``edge_cache`` of ``node:{from}_{to}`` polylines, per-BKS
``edge_keys``, and the set of straight-line fallback edges.

Semantics ported from ``site_api.jl``: per-edge resolution walks the map
candidates in cascade order, requires both endpoints to map to a graph
vertex (nearest road node within 100 m must itself be a vertex), routes with
the group metric, accepts a segment only when its endpoints land within
250 m of the instance node coordinates, tries the reversed edge before
giving up, and falls back to a straight line between the node coordinates.

One addition over the Julia semantics: an instance node with no road node in
its 100 m box (or whose nearest node is not a vertex) is projected onto the
nearest straight edge of a node-level candidate graph and routed from that
virtual point (:class:`EdgeAnchor`). Mamut2026 ``corner`` depots are often
synthetic crop nodes of the generation-time extract; when the website
re-fetches the city with other bounds, that node no longer exists and the
depot sits mid-segment on a road, which used to turn every depot leg into a
straight line. Nodes that resolve to a vertex keep the exact former path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable

from mamut_routing_tools.geo import ENU, lla_from_enu
from mamut_routing_tools.roadgraph.build import RoadGraph, road_graph_candidates
from mamut_routing_tools.roadgraph.router import route_lonlat, route_vertices

ENDPOINT_TOLERANCE_METERS = 250.0
#: How far off the nearest road an unsnappable node may sit and still be
#: anchored to that road. Well under the 100 m node box: a node this close to
#: a road but farther than that from every road node is on a long straight
#: segment, which is exactly the case the anchor exists for.
ANCHOR_MAX_DISTANCE_METERS = 50.0


@dataclass(frozen=True)
class EdgeAnchor:
    """An instance node resolved to a point ON a directed graph edge rather
    than to a vertex: ``fraction`` along ``edge_index`` (0 = tail, 1 = head),
    ``lonlat`` the projected point the polyline starts or ends at."""

    edge_index: int
    fraction: float
    lonlat: list[float]


#: A candidate graph with its two node resolutions: instance node -> vertex,
#: and, for the nodes that map to no vertex, instance node -> edge anchor.
#: A bare ``(graph, vertex_map)`` pair is still accepted (no anchors).
Candidate = (
    tuple[RoadGraph, dict[int, int]] | tuple[RoadGraph, dict[int, int], dict[int, EdgeAnchor]]
)


def node_edge_cache_key(from_node: int, to_node: int) -> str:
    return f"node:{from_node}_{to_node}"


def _is_lonlat_point(point: list[float]) -> bool:
    return abs(float(point[0])) <= 180.0 and abs(float(point[1])) <= 90.0


def point_distance_meters(first_point: list[float], second_point: list[float]) -> float:
    if _is_lonlat_point(first_point) and _is_lonlat_point(second_point):
        mean_lat = (float(first_point[1]) + float(second_point[1])) / 2.0
        lon_scale = 111_320.0 * math.cos(math.radians(mean_lat))
        lat_scale = 111_320.0
        return math.hypot(
            (float(first_point[0]) - float(second_point[0])) * lon_scale,
            (float(first_point[1]) - float(second_point[1])) * lat_scale,
        )
    return math.hypot(
        float(first_point[0]) - float(second_point[0]),
        float(first_point[1]) - float(second_point[1]),
    )


def _segment_matches_endpoints(
    segment: list[list[float]],
    from_coordinates: list[float] | None,
    to_coordinates: list[float] | None,
) -> bool:
    if from_coordinates is None or to_coordinates is None:
        return True
    if len(segment) < 2:
        return False
    return (
        point_distance_meters(segment[0], from_coordinates) <= ENDPOINT_TOLERANCE_METERS
        and point_distance_meters(segment[-1], to_coordinates) <= ENDPOINT_TOLERANCE_METERS
    )


def node_coordinates_map(meta: dict[str, Any]) -> dict[int, list[float]]:
    nodes = meta.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("Request meta is missing its nodes list")
    coordinates: dict[int, list[float]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("instance_node_id")
        if node_id is None:
            continue
        poi_lon, poi_lat = node.get("poi_lon"), node.get("poi_lat")
        if poi_lon is not None and poi_lat is not None:
            coordinates[int(node_id)] = [float(poi_lon), float(poi_lat)]
            continue
        enu_x, enu_y = node.get("enu_x"), node.get("enu_y")
        if enu_x is not None and enu_y is not None:
            coordinates[int(node_id)] = [float(enu_x), float(enu_y)]
    if not coordinates:
        raise ValueError("Request meta does not expose any previewable node coordinates")
    return coordinates


def resolve_source_osm_path(repo_root: Path, meta: dict[str, Any], meta_file_path: str) -> Path:
    source_osm_file = meta.get("source_osm_file")
    if not source_osm_file:
        raise ValueError(f"Sidecar '{meta_file_path}' is missing 'source_osm_file'")
    source = Path(str(source_osm_file))
    if source.is_absolute() and source.is_file():
        return source
    for candidate in (
        (repo_root / meta_file_path).parent / source,
        repo_root / source,
    ):
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"Unable to resolve source OSM file '{source_osm_file}' for sidecar '{meta_file_path}'")


def _graph_vertex_map(graph: RoadGraph, node_coordinates: dict[int, list[float]]) -> dict[int, int]:
    """Instance node id -> graph vertex, via nearest road node within 100 m
    that is itself a graph vertex (the NodeSpatIndex + map.v contract)."""
    mapping: dict[int, int] = {}
    for node_id, point in node_coordinates.items():
        if not _is_lonlat_point(point):
            continue
        osm_id = graph.nearest_node(point[1], point[0])
        if osm_id is not None and osm_id in graph.vertex_of:
            mapping[node_id] = graph.vertex_of[osm_id]
    return mapping


def _graph_edge_anchor_map(
    graph: RoadGraph,
    node_coordinates: dict[int, list[float]],
    vertex_map: dict[int, int],
) -> dict[int, EdgeAnchor]:
    """Edge anchors for the instance nodes ``vertex_map`` left unresolved.

    Empty on intersection-level graphs, whose edges have no straight geometry
    to project onto; the cascade always also probes a node-level graph."""
    anchors: dict[int, EdgeAnchor] = {}
    if graph.only_intersections:
        return anchors
    for node_id, point in node_coordinates.items():
        if node_id in vertex_map or not _is_lonlat_point(point):
            continue
        projection = graph.nearest_edge_projection(point[1], point[0], ANCHOR_MAX_DISTANCE_METERS)
        if projection is None:
            continue
        edge_index, fraction, _distance = projection
        tail = graph.node_enu[graph.edges[edge_index][0]]
        head = graph.node_enu[graph.edges[edge_index][1]]
        east = tail[0] + fraction * (head[0] - tail[0])
        north = tail[1] + fraction * (head[1] - tail[1])
        lla = lla_from_enu(ENU(east, north, 0.0), graph.ref_lla)
        anchors[node_id] = EdgeAnchor(edge_index, fraction, [lla.lon, lla.lat])
    return anchors


def graph_node_maps(
    graph: RoadGraph, node_coordinates: dict[int, list[float]]
) -> tuple[dict[int, int], dict[int, EdgeAnchor]]:
    """Both node resolutions of one candidate graph: vertices first, edge
    anchors only for what the vertex rule could not place."""
    vertex_map = _graph_vertex_map(graph, node_coordinates)
    return vertex_map, _graph_edge_anchor_map(graph, node_coordinates, vertex_map)


def _metric_weight(graph: RoadGraph, metric: str) -> Callable[[int], float]:
    if metric == "fastest":
        return graph.time_weight
    if metric == "shortest":
        return lambda edge_index: graph.edge_weight[edge_index]
    raise ValueError(f"Unsupported road metric '{metric}'")


def _reverse_edge_index(graph: RoadGraph, edge_index: int) -> int | None:
    """The cheapest edge running head -> tail of ``edge_index``, if any."""
    tail, head = graph.edges[edge_index]
    tail_vertex, head_vertex = graph.vertex_of[tail], graph.vertex_of[head]
    if not graph.graph.has_edge(head_vertex, tail_vertex):
        return None
    return min(graph.graph.get_all_edge_data(head_vertex, tail_vertex), key=lambda index: graph.edge_weight[index])


def _path_cost(graph: RoadGraph, vertices: list[int], weight: Callable[[int], float]) -> float:
    return sum(
        min(weight(index) for index in graph.graph.get_all_edge_data(vertices[i - 1], vertices[i]))
        for i in range(1, len(vertices))
    )


def _departures(graph: RoadGraph, source: int | EdgeAnchor, weight: Callable[[int], float]) -> list[tuple[int, float, list[list[float]]]]:
    """Ways to leave ``source`` onto a vertex: (vertex, cost so far, prefix)."""
    if isinstance(source, int):
        return [(source, 0.0, [])]
    tail, head = graph.edges[source.edge_index]
    options = [(graph.vertex_of[head], (1.0 - source.fraction) * weight(source.edge_index), [source.lonlat])]
    reverse = _reverse_edge_index(graph, source.edge_index)
    if reverse is not None:
        options.append((graph.vertex_of[tail], source.fraction * weight(reverse), [source.lonlat]))
    return options


def _arrivals(graph: RoadGraph, target: int | EdgeAnchor, weight: Callable[[int], float]) -> list[tuple[int, float, list[list[float]]]]:
    """Ways to reach ``target`` from a vertex: (vertex, remaining cost, suffix)."""
    if isinstance(target, int):
        return [(target, 0.0, [])]
    tail, head = graph.edges[target.edge_index]
    options = [(graph.vertex_of[tail], target.fraction * weight(target.edge_index), [target.lonlat])]
    reverse = _reverse_edge_index(graph, target.edge_index)
    if reverse is not None:
        options.append((graph.vertex_of[head], (1.0 - target.fraction) * weight(reverse), [target.lonlat]))
    return options


def _same_edge_segment(
    graph: RoadGraph, source: EdgeAnchor, target: EdgeAnchor, weight: Callable[[int], float]
) -> tuple[float, list[list[float]]] | None:
    """Both anchors on one road segment, travelled directly along it."""
    if target.edge_index == source.edge_index:
        target_fraction = target.fraction
    elif _reverse_edge_index(graph, target.edge_index) == source.edge_index:
        target_fraction = 1.0 - target.fraction
    else:
        return None
    if target_fraction >= source.fraction:
        cost = (target_fraction - source.fraction) * weight(source.edge_index)
    else:
        reverse = _reverse_edge_index(graph, source.edge_index)
        if reverse is None:
            return None
        cost = (source.fraction - target_fraction) * weight(reverse)
    return cost, [source.lonlat, target.lonlat]


def _anchored_route_lonlat(
    graph: RoadGraph,
    source: int | EdgeAnchor,
    target: int | EdgeAnchor,
    metric: str,
) -> list[list[float]] | None:
    """Cheapest polyline between two resolutions, at least one an anchor:
    every way of leaving the source edge times every way of reaching the
    target edge, each completed by a vertex-to-vertex Dijkstra."""
    weight = _metric_weight(graph, metric)
    best: tuple[float, list[list[float]]] | None = None
    if isinstance(source, EdgeAnchor) and isinstance(target, EdgeAnchor):
        best = _same_edge_segment(graph, source, target, weight)
    for departure_vertex, departure_cost, prefix in _departures(graph, source, weight):
        for arrival_vertex, arrival_cost, suffix in _arrivals(graph, target, weight):
            vertices = route_vertices(graph, departure_vertex, arrival_vertex, metric)
            if vertices is None:
                continue
            cost = departure_cost + _path_cost(graph, vertices, weight) + arrival_cost
            if best is None or cost < best[0]:
                best = (cost, [*prefix, *(graph.node_lonlat(graph.node_of[vertex]) for vertex in vertices), *suffix])
    return None if best is None else best[1]


def candidate_route_segment(
    candidates: list[Candidate],
    from_node: int,
    to_node: int,
    from_coordinates: list[float],
    to_coordinates: list[float],
    metric: str,
) -> list[list[float]] | None:
    for candidate in candidates:
        graph, vertex_map = candidate[0], candidate[1]
        anchor_map: dict[int, EdgeAnchor] = candidate[2] if len(candidate) > 2 else {}
        source = vertex_map.get(from_node, anchor_map.get(from_node))
        target = vertex_map.get(to_node, anchor_map.get(to_node))
        if source is None or target is None:
            continue
        try:
            if isinstance(source, int) and isinstance(target, int):
                segment = route_lonlat(graph, source, target, metric)
            else:
                segment = _anchored_route_lonlat(graph, source, target, metric)
        except (ValueError, KeyError):
            segment = None
        if segment is None:
            continue
        if _segment_matches_endpoints(segment, from_coordinates, to_coordinates):
            return segment
    return None


def materialize_group(repo_root: Path, group: dict[str, Any]) -> dict[str, Any]:
    meta = group["meta"]
    metric = str(group["metric"])
    geo_path = str(group["geo_path"])
    node_coordinates = node_coordinates_map(meta)
    map_options = meta.get("map_options") or {}
    only_intersections = bool(map_options.get("only_intersections", True))
    trim_to_connected = bool(map_options.get("trim_to_connected_graph", True))

    osm_path = resolve_source_osm_path(repo_root, meta, geo_path)
    graphs = road_graph_candidates(
        osm_path,
        only_intersections=only_intersections,
        trim_to_connected=trim_to_connected,
    )
    if not graphs:
        raise ValueError(f"No usable OSM road graph was available for {geo_path}")
    candidates: list[Candidate] = [(graph, *graph_node_maps(graph, node_coordinates)) for graph in graphs]

    required_edges: set[tuple[int, int]] = set()
    entry_edges: list[dict[str, Any]] = []
    for entry in group["entries"]:
        required_keys: set[str] = set()
        for raw_route in entry["routes"]:
            route = [int(stop) for stop in raw_route]
            full_route = [0, *route, 0]
            for index in range(len(full_route) - 1):
                edge = (full_route[index], full_route[index + 1])
                required_edges.add(edge)
                required_keys.add(node_edge_cache_key(*edge))
        entry_edges.append({"bks_path": str(entry["bks_path"]), "edge_keys": sorted(required_keys)})

    edge_cache: dict[str, list[list[float]]] = {}
    straight_fallback_edges: list[str] = []
    for from_node, to_node in sorted(required_edges):
        segment = candidate_route_segment(
            candidates,
            from_node,
            to_node,
            node_coordinates[from_node],
            node_coordinates[to_node],
            metric,
        )
        if segment is None:
            reverse_segment = candidate_route_segment(
                candidates,
                to_node,
                from_node,
                node_coordinates[to_node],
                node_coordinates[from_node],
                metric,
            )
            segment = list(reversed(reverse_segment)) if reverse_segment is not None else None
        key = node_edge_cache_key(from_node, to_node)
        if segment is None:
            segment = [node_coordinates[from_node], node_coordinates[to_node]]
            straight_fallback_edges.append(key)
        edge_cache[key] = segment

    return {
        "edge_cache": edge_cache,
        "entries": entry_edges,
        "straight_fallback_edges": sorted(straight_fallback_edges),
    }


def materialize_plan(repo_root: str | Path, plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Process every group; returns {result_file: result_payload}."""
    root = Path(repo_root)
    results: dict[str, dict[str, Any]] = {}
    for group in plan["groups"]:
        results[str(group["result_file"])] = materialize_group(root, group)
    return results
