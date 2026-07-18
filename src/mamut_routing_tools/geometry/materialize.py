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
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from mamut_routing_tools.roadgraph.build import RoadGraph, road_graph_candidates
from mamut_routing_tools.roadgraph.router import route_lonlat

ENDPOINT_TOLERANCE_METERS = 250.0


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


def candidate_route_segment(
    candidates: list[tuple[RoadGraph, dict[int, int]]],
    from_node: int,
    to_node: int,
    from_coordinates: list[float],
    to_coordinates: list[float],
    metric: str,
) -> list[list[float]] | None:
    for graph, vertex_map in candidates:
        if from_node not in vertex_map or to_node not in vertex_map:
            continue
        try:
            segment = route_lonlat(graph, vertex_map[from_node], vertex_map[to_node], metric)
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
    candidates = [(graph, _graph_vertex_map(graph, node_coordinates)) for graph in graphs]

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
