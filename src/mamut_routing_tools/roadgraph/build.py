"""Road-graph construction: a direct Python port of OpenStreetMapX.jl's
``get_map_data`` as the MAMUT Julia pipeline used it.

The port preserves the exact graph semantics rather than approximating them
with a generic OSM library: the same way filter (``ROAD_CLASSES`` + visible +
not services), the same oneway/reverse rules, the same intersection-splitting
with ENU Euclidean segment lengths, the same largest-strongly-connected-
component trim (computed on the full node-level graph, then rebuilt at
intersection granularity), and the same option-fallback chain. Graph parity
with the Julia engine is then a floating-point question, not a modeling one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rustworkx as rx
from scipy.spatial import cKDTree

from mamut_routing_tools.geo import ENU, LLA, bounds_center, enu_distance, enu_from_lla, lla_from_enu
from mamut_routing_tools.roadgraph.osmxml import OsmData, OsmWay, crop_to_bounds, ensure_bounds, parse_osm

#: Default speed limits in km/h by road class (OpenStreetMapX SPEED_ROADS_URBAN).
SPEED_ROADS_URBAN: dict[int, float] = {
    1: 100.0,  # motorway
    2: 90.0,  # trunk
    3: 90.0,  # primary
    4: 70.0,  # secondary
    5: 50.0,  # tertiary
    6: 40.0,  # residential/unclassified
    7: 20.0,  # service
    8: 10.0,  # living street
}

ROAD_CLASSES: dict[str, int] = {
    "motorway": 1,
    "trunk": 2,
    "primary": 3,
    "secondary": 4,
    "tertiary": 5,
    "unclassified": 6,
    "residential": 6,
    "service": 7,
    "motorway_link": 1,
    "trunk_link": 2,
    "primary_link": 3,
    "secondary_link": 4,
    "tertiary_link": 5,
    "living_street": 8,
    "pedestrian": 8,
    "road": 6,
}


class EmptyRoadGraphError(ValueError):
    """The requested options produced no usable road graph (triggers the
    option-fallback chain, mirroring the Julia ArgumentError path)."""


def _visible(way: OsmWay) -> bool:
    return way.tags.get("visible", "") != "false"


def _valid_roadway(way: OsmWay) -> bool:
    highway = way.tags.get("highway", "")
    if not highway or highway == "services" or highway not in ROAD_CLASSES:
        return False
    return _visible(way)


def _oneway(way: OsmWay) -> bool:
    value = way.tags.get("oneway", "")
    if value in ("false", "no", "0"):
        return False
    if value in ("-1", "true", "yes", "1"):
        return True
    highway = way.tags.get("highway", "")
    junction = way.tags.get("junction", "")
    return highway in ("motorway", "motorway_link") or junction == "roundabout"


def _reverseway(way: OsmWay) -> bool:
    return way.tags.get("oneway", "") == "-1"


def _classify(way: OsmWay) -> int:
    return ROAD_CLASSES[way.tags["highway"]]


def _find_intersections(roadways: list[OsmWay]) -> set[int]:
    seen: set[int] = set()
    intersections: set[int] = set()
    for way in roadways:
        last_index = len(way.nodes) - 1
        for index, node in enumerate(way.nodes):
            if index == 0 or index == last_index or node in seen:
                intersections.add(node)
            else:
                seen.add(node)
    return intersections


@dataclass
class RoadGraph:
    osm_path: Path
    only_intersections: bool
    trim_to_connected: bool
    ref_lla: LLA
    node_enu: dict[int, tuple[float, float, float]]
    vertex_of: dict[int, int]  # osm node id -> graph vertex index (0-based)
    node_of: list[int]  # graph vertex index -> osm node id
    edges: list[tuple[int, int]]  # (osm u, osm v)
    edge_class: list[int]
    edge_weight: list[float]  # metres (ENU Euclidean along the segment)
    graph: rx.PyDiGraph = field(repr=False)
    _kdtree: cKDTree | None = field(default=None, repr=False)
    _kdtree_nodes: list[int] = field(default_factory=list, repr=False)

    @property
    def vertex_count(self) -> int:
        return len(self.node_of)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def node_lla(self, osm_id: int) -> LLA:
        east, north, up = self.node_enu[osm_id]
        return lla_from_enu((east, north, up), self.ref_lla)  # type: ignore[arg-type]

    def node_lonlat(self, osm_id: int) -> list[float]:
        lla = self.node_lla(osm_id)
        return [lla.lon, lla.lat]

    def time_weight(self, edge_index: int, speeds: dict[int, float] = SPEED_ROADS_URBAN) -> float:
        return 3.6 * self.edge_weight[edge_index] / speeds[self.edge_class[edge_index]]

    def nearest_node(self, lat: float, lon: float, node_range_m: float = 100.0) -> int | None:
        """Nearest road node (any node, not only graph vertices), matching
        OSMToolset's NodeSpatIndex/findnode contract exactly: candidate
        inclusion is an axis-aligned +-range box (L-infinity) around the
        query point, and the pick is the minimum EUCLIDEAN distance among
        the box hits. A node at 102 m Euclidean but inside the box still
        matches; one at 99 m outside the box does not exist geometrically."""
        if self._kdtree is None:
            self._kdtree_nodes = list(self.node_enu.keys())
            points = np.array(
                [(self.node_enu[node][0], self.node_enu[node][1]) for node in self._kdtree_nodes]
            )
            self._kdtree = cKDTree(points)
        enu = enu_from_lla(LLA(lat, lon), self.ref_lla)
        query = np.array([enu.east, enu.north])
        hits = self._kdtree.query_ball_point(query, r=node_range_m, p=np.inf)
        if not hits:
            return None
        best_index = min(hits, key=lambda i: float(np.sum((self._kdtree.data[i] - query) ** 2)))
        return self._kdtree_nodes[int(best_index)]


def _add_intersection_edges(
    roadways: list[OsmWay],
    node_enu: dict[int, tuple[float, float, float]],
    intersections: set[int],
) -> tuple[list[tuple[int, int]], list[int], list[float]]:
    back: dict[tuple[int, int], int] = {}
    edges: list[tuple[int, int]] = []
    classes: list[int] = []
    weights: list[float] = []

    def add_segment(way: OsmWay, path: list[int]) -> None:
        edge = (path[0], path[-1])
        weight = sum(
            enu_distance(node_enu[path[i - 1]], node_enu[path[i]])  # type: ignore[arg-type]
            for i in range(1, len(path))
        )
        if edge in back:
            index = back[edge]
            if weight < weights[index]:
                classes[index] = _classify(way)
                weights[index] = weight
        else:
            edges.append(edge)
            classes.append(_classify(way))
            weights.append(weight)
            back[edge] = len(edges) - 1

    for way in roadways:
        first = 0
        for j in range(1, len(way.nodes)):
            if way.nodes[first] != way.nodes[j] and (way.nodes[j] in intersections or j == len(way.nodes) - 1):
                reverse = _reverseway(way)
                if not reverse:
                    add_segment(way, way.nodes[first : j + 1])
                if reverse or not _oneway(way):
                    add_segment(way, way.nodes[first : j + 1][::-1])
                first = j
    return edges, classes, weights


def _add_full_edges(
    roadways: list[OsmWay],
    node_enu: dict[int, tuple[float, float, float]],
) -> tuple[list[tuple[int, int]], list[int], list[float]]:
    edge_class: dict[tuple[int, int], int] = {}
    for way in roadways:
        one = _oneway(way)
        reverse = _reverseway(way)
        cls = _classify(way)
        for j in range(1, len(way.nodes)):
            n0, n1 = way.nodes[j - 1], way.nodes[j]
            start, fin = (n1, n0) if reverse else (n0, n1)
            edge_class[(start, fin)] = cls
            if not one:
                edge_class[(fin, start)] = cls
    edges = list(edge_class.keys())
    classes = list(edge_class.values())
    weights = [
        enu_distance(node_enu[edge[1]], node_enu[edge[0]])  # type: ignore[arg-type]
        for edge in edges
    ]
    return edges, classes, weights


def build_road_graph(
    osm_data: OsmData,
    osm_path: Path,
    *,
    only_intersections: bool = True,
    trim_to_connected: bool = True,
    remove_nodes: set[int] | None = None,
) -> RoadGraph:
    roadways = [way for way in osm_data.ways if _valid_roadway(way)]
    if remove_nodes:
        pruned: list[OsmWay] = []
        for way in roadways:
            kept = [node for node in way.nodes if node not in remove_nodes]
            if kept:
                pruned.append(OsmWay(way_id=way.way_id, nodes=kept, tags=way.tags))
        roadways = pruned

    min_lat, min_lon, max_lat, max_lon = osm_data.bounds
    ref_lla = bounds_center(min_lat, min_lon, max_lat, max_lon)
    node_enu: dict[int, ENU] = {}
    for way in roadways:
        for node in way.nodes:
            if node not in node_enu:
                lat, lon = osm_data.nodes[node]
                node_enu[node] = enu_from_lla(LLA(lat, lon), ref_lla)

    if only_intersections and not trim_to_connected:
        intersections = _find_intersections(roadways)
        edges, classes, weights = _add_intersection_edges(roadways, node_enu, intersections)
    else:
        edges, classes, weights = _add_full_edges(roadways, node_enu)
    if not edges:
        raise EmptyRoadGraphError(f"OSM file '{osm_path}' produced an empty road graph")

    vertex_of: dict[int, int] = {}
    node_of: list[int] = []
    for edge in edges:
        for osm_id in edge:
            if osm_id not in vertex_of:
                vertex_of[osm_id] = len(node_of)
                node_of.append(osm_id)

    graph = rx.PyDiGraph()
    graph.add_nodes_from(range(len(node_of)))
    for index, (u_osm, v_osm) in enumerate(edges):
        graph.add_edge(vertex_of[u_osm], vertex_of[v_osm], index)

    if trim_to_connected:
        components = rx.strongly_connected_components(graph)
        largest = max(components, key=len)
        if len(largest) < len(node_of):
            keep = set(largest)
            removed = {node_of[vertex] for vertex in range(len(node_of)) if vertex not in keep}
            return build_road_graph(
                osm_data,
                osm_path,
                only_intersections=only_intersections,
                trim_to_connected=False,
                remove_nodes=(remove_nodes or set()) | removed,
            )
        # Already strongly connected: rebuild without the trim flag so the
        # only_intersections edge granularity is honored (the Julia pipeline
        # reaches the same state through its fallback chain).
        if only_intersections:
            return build_road_graph(
                osm_data,
                osm_path,
                only_intersections=True,
                trim_to_connected=False,
                remove_nodes=remove_nodes,
            )

    return RoadGraph(
        osm_path=osm_path,
        only_intersections=only_intersections,
        trim_to_connected=trim_to_connected,
        ref_lla=ref_lla,
        node_enu=node_enu,
        vertex_of=vertex_of,
        node_of=node_of,
        edges=edges,
        edge_class=classes,
        edge_weight=weights,
        graph=graph,
    )


_OSM_DATA_CACHE: dict[Path, OsmData] = {}
_GRAPH_CACHE: dict[tuple[Path, bool, bool], RoadGraph] = {}

#: get_map_data fallback chain (OpenStreetMapX order).
_FALLBACK_CHAIN = ((True, True), (True, False), (False, True), (False, False))
#: Candidate order used by the render/materialize cascade (site_api.jl order).
CANDIDATE_OPTION_ORDER = ((True, True), (False, True), (True, False), (False, False))


def _load_osm_data(osm_path: Path) -> OsmData:
    resolved = osm_path.resolve()
    if resolved not in _OSM_DATA_CACHE:
        ensure_bounds(resolved)
        osm_data = parse_osm(resolved)
        crop_to_bounds(osm_data)
        _OSM_DATA_CACHE[resolved] = osm_data
    return _OSM_DATA_CACHE[resolved]


def load_road_graph(
    osm_path: str | Path,
    *,
    only_intersections: bool = True,
    trim_to_connected: bool = True,
) -> RoadGraph:
    """Load with the same progressive option relaxation as the Julia servers:
    requested options first, then (oi, False), (False, trim), (False, False)."""
    resolved = Path(osm_path).resolve()
    key = (resolved, only_intersections, trim_to_connected)
    if key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]
    osm_data = _load_osm_data(resolved)

    attempts: list[tuple[bool, bool]] = []
    for option in ((only_intersections, trim_to_connected), *_FALLBACK_CHAIN):
        if option not in attempts:
            attempts.append(option)
    last_error: Exception | None = None
    for oi, trim in attempts:
        try:
            graph = build_road_graph(osm_data, resolved, only_intersections=oi, trim_to_connected=trim)
        except EmptyRoadGraphError as error:
            last_error = error
            continue
        _GRAPH_CACHE[(resolved, oi, trim)] = graph
        _GRAPH_CACHE[key] = graph
        return graph
    raise EmptyRoadGraphError(
        f"OSM file '{resolved}' produced an empty road graph for every option combination"
    ) from last_error


def road_graph_candidates(
    osm_path: str | Path,
    *,
    only_intersections: bool = True,
    trim_to_connected: bool = True,
) -> list[RoadGraph]:
    """Candidate graphs in the cascade order the render path probes them."""
    options: list[tuple[bool, bool]] = []
    for option in (
        (only_intersections, trim_to_connected),
        (False, trim_to_connected),
        (only_intersections, False),
        (False, False),
    ):
        if option not in options:
            options.append(option)
    candidates: list[RoadGraph] = []
    seen_ids: set[int] = set()
    for oi, trim in options:
        try:
            graph = load_road_graph(osm_path, only_intersections=oi, trim_to_connected=trim)
        except (EmptyRoadGraphError, OSError):
            continue
        if id(graph) not in seen_ids:
            seen_ids.add(id(graph))
            candidates.append(graph)
    return candidates


def clear_caches() -> None:
    _OSM_DATA_CACHE.clear()
    _GRAPH_CACHE.clear()
