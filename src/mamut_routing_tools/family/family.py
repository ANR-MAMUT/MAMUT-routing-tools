"""Poryos2026 collection builder (v2, Stream 12').

The Python side of the deliberately 3-stepped generation. Per base
(city x n x method):

1. ``build_base`` (stage ``generate-base``, after the Python sampling +
   bridge export): builds the full-city road graph from the bridge, trims it
   to the union of pinned free-flow fastest-path edges (static per-edge
   ``speed_limit`` weights; bit-exact distance-preservation re-check),
   computes both distance matrices and the complete indexed geo road cache,
   and publishes the 3 slim CVRP metric instances + the ``geo`` / ``road`` /
   ``distances-*`` sidecars into the collection trees.

2. ``derive_vrptw`` (stage ``derive-vrptw``): the single name-seeded
   synthesis (service times + TWs over the published free-flow fastest
   matrix) emits one VRPTW instance per TW set: the ``td-shared``
   candidate (route-centered, finalized by ``build_td``) and the
   static-only ``tight`` / ``spread`` sets (final as written, never
   audited under traffic, NOT the TDVRPTW windows).

3. ``build_td`` (stage ``build-td``): aligns the 6 traffic overlays to the
   trimmed graph (clamped at each edge's free-flow limit), materializes the
   canonical ATFs per subinstance, certifies the complete anchor routes under
   all 6 overlays and applies minimal shared deadline lifts (finalizing the
   VRPTW instance, ``metadata.tw_repair``), then emits the 12 slim TD twins
   with sha-pinned td blocks. Generation gates: the published
   ``distances-fastest`` equals the road graph's free-flow node times after
   rounding, and the post-lift audit holds for every overlay (hard assert).

Python is the sole serializer (language-boundary tier 1): every published
byte is canonical JSON written here; the git-ignored bridge is produced in Python (mamut_routing_tools.td.traffic).
"""

from __future__ import annotations

import math
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Mapping

from mamut_routing_tools.generation.binpacking import minimum_bins

from mamut_routing_lib.distances import (
    InstanceDistances,
    compute_distances_sha256,
    load_instance_distances,
    save_instance_distances,
)
from mamut_routing_lib.geo import (
    GeoNode,
    GeoRoadCache,
    InstanceGeo,
    compute_geo_sha256,
    save_instance_geo,
)
from mamut_routing_lib.json_utils import load_json_from_file, save_json_to_file
from mamut_routing_lib.sidecars import (
    COLLECTION_MARKER_FILENAME,
    CollectionMarker,
    find_collection_root,
    save_collection_marker,
)
from mamut_routing_lib.td import (
    InstanceRoadGraph,
    TrafficOverlay,
    build_adjacency,
    compute_atf_sha256,
    compute_fastest_path_tree,
    compute_road_graph_sha256,
    compute_traffic_overlay_sha256,
    free_flow_node_times,
    load_instance_road_graph,
    load_traffic_overlay,
    load_td_instance,
    materialize_instance_atfs_roadgraph,
    materialize_selected_atfs_roadgraph,
    save_instance_road_graph,
    save_traffic_overlay,
    td_instance_from_payload,
)

from mamut_routing_tools.td.bridge import BridgeGraph, BridgeNodes, BridgeSpeeds
from mamut_routing_tools.family.naming import (
    ALL_TW_SETS,
    FAMILY,
    TW_SET_SPREAD,
    TW_SET_TD_SHARED,
    TW_SET_TIGHT,
    FAMILIES_WITH_FLEET_IN_NAME,
    base_instance_name,
    cvrp_dir,
    sidecar_dir,
    sidecar_relpath,
    subinstance_name,
    td_instance_dir,
    td_instance_name,
    vrptw_dir,
    vrptw_instance_name,
)
from mamut_routing_tools.family.tw_synthesis import (
    TIGHT_TW_WIDTH_RATIO_MAX,
    TIGHT_TW_WIDTH_RATIO_MEAN,
    TIGHT_TW_WIDTH_RATIO_MIN,
    TIGHT_TW_WIDTH_RATIO_STD,
    construct_anchor_routes,
    synthesize_service_times,
    synthesize_time_windows,
    synthesize_time_windows_spread,
    validate_static_anchor,
)

TD_HORIZON = (0.0, 86400.0)
DEFAULT_EXTENSION_END = 172800.0
DEFAULT_SAMPLE_STEP = 60.0
DISTANCE_DECIMALS = 3
#: Above this, the geo sidecar carries node coordinates but no pinned paths, so
#: the website draws routes as straight lines rather than along real streets.
#: Held at 100 because the cache is the single most expensive artifact per
#: instance: measured on the published collection, a geo sidecar is 2.23 MB with
#: it and 0.01 MB without -- a factor of ~200 -- and it grows with n^2.
ROAD_CACHE_MAX_N = 100
#: Above this, an instance is ``.vrp.json``-only: the CVRPLIB explicit-matrix
#: format writes n x n costs as text, so the file is quadratic and quickly
#: absurd (~9 MB per metric at n=1000). Raised from 100 to 200 for the size
#: ladder, where every instance has a distinct n and a cutoff at 100 would leave
#: exactly one of a hundred instances readable by a CVRPLIB-only solver. At 200
#: it is thirty of them, for ~17 MB.
VRP_EXPORT_MAX_N = 200
TD_MODELS = ("bpr", "wave")
TD_INTENSITIES = ("light", "moderate", "heavy")
GENERATOR_NAME = "mamut-routing-tools"
PIPELINE_VERSION = 3
AUTHORS = "MAMUT-routing-tools (generated instance)"


def capacity_lower_bound(demands: list[int], capacity: int) -> int:
    """Return the standard capacity lower bound, excluding the depot demand."""
    if capacity <= 0:
        raise ValueError("vehicle capacity must be positive")
    return math.ceil(sum(demands[1:]) / capacity)


def simplify_tolerance_for(num_customers: int) -> float:
    """Frozen family policy: 1.0 s for n <= 100, 2.0 s for n >= 500."""
    return 1.0 if num_customers <= 100 else 2.0


def _stable_seed(*parts: str) -> int:
    return zlib.crc32("|".join(parts).encode("utf-8"))


def sampling_seed(base: str) -> int:
    """Stage-1 sampling seed, derived from the base name (recorded provenance;
    reproducible under the pinned Julia toolchain only)."""
    return _stable_seed(base, "sample")


def ensure_collection_root(collection_root: str | Path, family: str = FAMILY) -> Path:
    root = Path(collection_root)
    marker_path = root / COLLECTION_MARKER_FILENAME
    if not marker_path.exists():
        save_collection_marker(CollectionMarker(family=family), root)
    return root


# ---------------------------------------------------------------------------
# Road-graph assembly (full city -> trimmed base graph)
# ---------------------------------------------------------------------------


def _full_city_road_graph(
    graph: BridgeGraph,
    nodes: BridgeNodes,
    *,
    base: str,
    sample_step: float,
    simplify_tolerance: float,
    extension_end: float,
    generator: dict[str, Any],
    family: str = FAMILY,
) -> InstanceRoadGraph:
    osm_ids = sorted({osm for osm_u, osm_v, _, _, _ in graph.edges for osm in (osm_u, osm_v)})
    index_of = {osm: index for index, osm in enumerate(osm_ids)}
    edges = sorted(
        (index_of[osm_u], index_of[osm_v], length_m, free_speed)
        for osm_u, osm_v, length_m, _, free_speed in graph.edges
    )
    missing = [osm for osm in nodes.node_osm_ids if osm not in index_of]
    if missing:
        raise ValueError(f"instance nodes not present in the bridge graph: OSM {missing[:5]}...")
    bin_edges = [TD_HORIZON[0] + k * graph.bin_seconds for k in range(graph.num_bins + 1)]
    if bin_edges[-1] != TD_HORIZON[1]:
        raise ValueError(f"bridge bins {graph.num_bins} x {graph.bin_seconds}s do not tile the horizon")
    return InstanceRoadGraph(
        base_name=base,
        benchmark_name=family,
        num_customers=len(nodes.node_osm_ids) - 1,
        horizon=TD_HORIZON,
        extension_end=extension_end,
        bin_edges=bin_edges,
        sample_step=sample_step,
        simplify_tolerance=simplify_tolerance,
        num_vertices=len(osm_ids),
        vertex_osm_ids=osm_ids,
        vertex_lonlat=[graph.vertex_lonlat[osm] for osm in osm_ids],
        node_vertices=[index_of[osm] for osm in nodes.node_osm_ids],
        edges=edges,
        generator=generator,
    )


def _collect_tree_paths(
    road: InstanceRoadGraph,
    adjacency: list[list[int]],
    source: int,
) -> tuple[list[float], list[int]]:
    return compute_fastest_path_tree(road, adjacency, source)


def _pinned_tree_edges(
    graph_ir: InstanceRoadGraph,
    adjacency: list[list[int]],
) -> tuple[set[int], list[list[float]]]:
    """Edges lying on the pinned shortest-path trees between instance nodes.

    Also returns the node-to-node distance matrix those trees induce, under
    whatever weighting ``graph_ir`` carries: free-flow times on the graph as
    built, lengths on its :func:`_length_weighted` twin.
    """
    node_set = graph_ir.node_vertices
    used_edges: set[int] = set()
    node_dists: list[list[float]] = []
    for source in node_set:
        dist, pred_edge = _collect_tree_paths(graph_ir, adjacency, source)
        node_dists.append([dist[target] for target in node_set])
        walked: set[int] = {source}
        for target in node_set:
            if target == source:
                continue
            vertex = target
            # Paths merge toward the source in the tree: once a vertex was
            # walked for this source, its chain to the source is collected.
            while vertex not in walked:
                edge_index = pred_edge[vertex]
                if edge_index < 0:
                    raise ValueError(
                        f"vertex OSM {graph_ir.vertex_osm_ids[target]} unreachable from "
                        f"OSM {graph_ir.vertex_osm_ids[source]} in the bridge graph"
                    )
                used_edges.add(edge_index)
                walked.add(vertex)
                vertex = graph_ir.edges[edge_index][0]
    return used_edges, node_dists


def _trim_road_graph(
    full: InstanceRoadGraph,
    *,
    pin_shortest_paths: bool = False,
) -> tuple[InstanceRoadGraph, list[list[float]]]:
    """Trim to the union of pinned free-flow fastest-path edges between
    instance nodes, verify node-to-node free-flow times are bit-identical,
    and return the trimmed graph plus the full-graph node time matrix.

    ``pin_shortest_paths`` widens the trim to also cover the shortest-*distance*
    trees. The two metrics follow different roads, so the default trim -- built
    from the fastest-time trees alone -- cannot reproduce the shortest matrix,
    and the published ``distances-shortest`` sidecar is therefore computed on
    the untrimmed city graph instead. That is fine while the sidecar is
    committed, and not fine when it is only sha-pinned and regenerated from
    published bytes: reproducing it would then need the OSM extract, which is
    gitignored and not stable over time. Set this for any instance whose
    distance sidecars are not committed.
    """
    adjacency = build_adjacency(full)
    used_edges, full_dists = _pinned_tree_edges(full, adjacency)
    if pin_shortest_paths:
        length_weighted = _length_weighted(full)
        shortest_edges, _ = _pinned_tree_edges(
            length_weighted, build_adjacency(length_weighted)
        )
        # Edge indices are shared: _length_weighted rewrites the speeds in place
        # and preserves the edge order.
        used_edges |= shortest_edges

    kept = sorted(used_edges)
    kept_osm_ids = sorted(
        {full.vertex_osm_ids[endpoint] for index in kept for endpoint in full.edges[index][:2]}
    )
    index_of = {osm: index for index, osm in enumerate(kept_osm_ids)}
    old_index_of_osm = {osm: index for index, osm in enumerate(full.vertex_osm_ids)}
    old_to_new = {
        old: index_of[full.vertex_osm_ids[old]]
        for index in kept
        for old in full.edges[index][:2]
    }
    trimmed_edges = sorted(
        (old_to_new[u], old_to_new[v], length, speed_limit)
        for u, v, length, speed_limit in (full.edges[index] for index in kept)
    )
    trimmed = InstanceRoadGraph(
        base_name=full.base_name,
        benchmark_name=full.benchmark_name,
        num_customers=full.num_customers,
        horizon=full.horizon,
        extension_end=full.extension_end,
        bin_edges=list(full.bin_edges),
        sample_step=full.sample_step,
        simplify_tolerance=full.simplify_tolerance,
        num_vertices=len(kept_osm_ids),
        vertex_osm_ids=kept_osm_ids,
        vertex_lonlat=[full.vertex_lonlat[old_index_of_osm[osm]] for osm in kept_osm_ids],
        node_vertices=[old_to_new[vertex] for vertex in full.node_vertices],
        edges=trimmed_edges,
        generator=dict(full.generator),
    )

    trimmed_adjacency = build_adjacency(trimmed)
    for row, source in zip(full_dists, trimmed.node_vertices):
        dist, _ = compute_fastest_path_tree(trimmed, trimmed_adjacency, source)
        trimmed_row = [dist[target] for target in trimmed.node_vertices]
        if trimmed_row != row:
            raise AssertionError(
                f"trimmed-graph free-flow times diverge from the full graph for {full.base_name}"
            )
    return trimmed, full_dists


def _length_weighted(graph_ir: InstanceRoadGraph) -> InstanceRoadGraph:
    """The same graph with unit speed limits: pinned Dijkstra weights become
    edge lengths, giving the canonical shortest-path metric."""
    return InstanceRoadGraph(
        base_name=graph_ir.base_name,
        benchmark_name=graph_ir.benchmark_name,
        num_customers=graph_ir.num_customers,
        horizon=graph_ir.horizon,
        extension_end=graph_ir.extension_end,
        bin_edges=list(graph_ir.bin_edges),
        sample_step=graph_ir.sample_step,
        simplify_tolerance=graph_ir.simplify_tolerance,
        num_vertices=graph_ir.num_vertices,
        vertex_osm_ids=list(graph_ir.vertex_osm_ids),
        vertex_lonlat=list(graph_ir.vertex_lonlat),
        node_vertices=list(graph_ir.node_vertices),
        edges=[(u, v, length, 1.0) for u, v, length, _ in graph_ir.edges],
        generator=dict(graph_ir.generator),
    )


def _round_matrix(matrix: list[list[float]]) -> list[list[float]]:
    return [[round(value, DISTANCE_DECIMALS) for value in row] for row in matrix]


def _node_paths(
    graph_ir: InstanceRoadGraph,
    *,
    want_paths: bool = True,
) -> tuple[list[list[float]], dict[tuple[int, int], list[int]]]:
    """Pinned node-to-node distances and vertex paths on ``graph_ir``.

    Returns the node time/length matrix (pinned Dijkstra weights) and, per
    ordered node pair, the pinned path as graph vertex indices. Path
    extraction is skipped with ``want_paths=False`` (large n: the road cache
    is capped, only the matrix is needed).
    """
    adjacency = build_adjacency(graph_ir)
    nodes = graph_ir.node_vertices
    matrix: list[list[float]] = []
    paths: dict[tuple[int, int], list[int]] = {}
    for i, source in enumerate(nodes):
        dist, pred_edge = compute_fastest_path_tree(graph_ir, adjacency, source)
        matrix.append([dist[target] for target in nodes])
        for j, target in enumerate(nodes):
            if i == j:
                continue
            if dist[target] == float("inf"):
                raise ValueError(
                    f"node {j} unreachable from node {i} on {graph_ir.base_name}"
                )
            if not want_paths:
                continue
            path = []
            vertex = target
            while vertex != source:
                path.append(vertex)
                vertex = graph_ir.edges[pred_edge[vertex]][0]
            path.append(source)
            path.reverse()
            paths[(i, j)] = path
    return matrix, paths


# ---------------------------------------------------------------------------
# Stage: generate-base (Python publication half)
# ---------------------------------------------------------------------------


@dataclass
class BuiltBase:
    base: str
    city: str
    num_customers: int
    method_tag: str
    road_sha256: str
    geo_sha256: str
    distances_sha256: dict[str, str]
    num_road_vertices: int
    num_road_edges: int
    build_seconds: float
    cvrp_paths: dict[str, Path] = field(default_factory=dict)
    #: False when the distance sidecars were pinned but not written, so the
    #: caller can report which instances need local materialization.
    distances_committed: bool = True


def _write_cvrplib(
    path: Path,
    *,
    name: str,
    comment: str,
    coordinates: list[list[float]],
    demands: list[int],
    matrix: list[list[float]],
    capacity: int,
) -> None:
    lines = [
        f"NAME : {name}",
        f"COMMENT : {comment}",
        "TYPE : CVRP",
        f"DIMENSION : {len(coordinates)}",
        "EDGE_WEIGHT_TYPE : EXPLICIT",
        "EDGE_WEIGHT_FORMAT : FULL_MATRIX",
        f"CAPACITY : {capacity}",
        "EDGE_WEIGHT_SECTION",
    ]
    lines.extend(" ".join(f"{value:.3f}" for value in row) for row in matrix)
    lines.append("NODE_COORD_SECTION")
    lines.extend(
        f"{index + 1} {x:.6f} {y:.6f}" for index, (x, y) in enumerate(coordinates)
    )
    lines.append("DEMAND_SECTION")
    lines.extend(f"{index + 1} {demand}" for index, demand in enumerate(demands))
    lines.extend(["DEPOT_SECTION", "1", "-1", "EOF", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _base_generator(stage: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    generator = {"name": GENERATOR_NAME, "stage": stage, "version": PIPELINE_VERSION}
    if extra:
        generator.update(extra)
    return generator


def _static_instance_payload(
    *,
    base: str,
    meta: dict[str, Any],
    manifest: dict[str, Any],
    metric: str,
    arc_costs_source: dict[str, Any],
    city: str,
    method_tag: str,
    geo_ref: dict[str, Any],
    generated_at: str,
    family: str = FAMILY,
) -> dict[str, Any]:
    nodes = meta["nodes"]
    reference_lla = meta["reference_lla"]
    demands = [int(node["demand"]) for node in nodes]
    capacity = int(manifest["capacity"])
    # The exact bin-packing minimum, which is what the base name carries. The
    # continuous bound ``ceil(sum / Q)`` is only its relaxation, and publishing
    # the two side by side would have the payload disagree with its own name.
    fleet = minimum_bins(demands[1:], capacity)
    lb_cap = fleet.lower
    floor = MINIMUM_ROUTES_BY_FAMILY.get(family, 2)
    if lb_cap < floor:
        raise ValueError(
            f"{base}: LB_cap={lb_cap}; {family} instances must need at least "
            f"{floor} routes"
        )
    if max(demands[1:], default=0) > capacity:
        raise ValueError(f"{base}: a customer demand exceeds vehicle capacity")
    return {
        "instance_name": base,
        "instance_origin": "OsmCvrpGen",
        "benchmark_name": family,
        "num_customers": len(nodes) - 1,
        "num_vehicles": None,
        "vehicle_capacity": capacity,
        "coordinates": [[float(node["enu_x"]), float(node["enu_y"])] for node in nodes],
        "demands": demands,
        "depot": 0,
        "reference_lla": {
            "lat": float(reference_lla["lat"]),
            "lon": float(reference_lla["lon"]),
            "alt": float(reference_lla.get("alt", 0.0)),
        },
        "metric_variant": metric,
        "arc_costs_source": arc_costs_source,
        "metadata": {
            "authors": AUTHORS,
            "generated_at": generated_at,
            "problem_type": "CVRP",
            "metric_variant": metric,
            "city": city,
            "method": method_tag,
            "base_instance_name": base,
            # A lower bound on a solution's route count, following CVRPLIB's
            # ``X-n101-k25``. Not a fleet cap: ``num_vehicles`` stays null.
            "num_vehicles_lb": lb_cap,
            "num_vehicles_lb_proven": fleet.proven,
            "num_vehicles_lb_method": fleet.method,
            "generator": _base_generator(
                "generate-base",
                {
                    "city": city,
                    "method": method_tag,
                    "sampling_seed": sampling_seed(base),
                    "stage1_base": str(meta.get("instance_name", "")),
                },
            ),
            "sidecars": {"geo": dict(geo_ref)},
        },
    }


#: Fewest routes a published instance of each family may require.
#:
#: Two only means "not a TSP", and it is not enough: 22 of the 110 bases in the
#: v2 Mamut2026 collection cleared it and were still trivially partitioned, 19 of
#: them into exactly two routes, five with a second route holding a single
#: customer. Mamut2026 v3 requires a real fleet. Poryos2026 keeps the old floor
#: because its instances are published and its sizes are drawn differently; this
#: gate must not retroactively invalidate them.
#:
#: This is the last line of defence, not the mechanism: the campaign's route-size
#: admissibility rule is what makes the property hold by construction.
MINIMUM_ROUTES_BY_FAMILY = {"Mamut2026": 6}


def _cvrplib_comment(
    family: str, metric: str, city: str, base: str, manifest: Mapping[str, Any]
) -> str:
    """The COMMENT line of a published ``.vrp``.

    "No of trucks" is how CVRPLIB records the fleet in a comment, and how the
    readers that care about it find the value. Ours is a lower bound, so the
    wording says so rather than letting a reader assume a cap.

    Only families that already put the fleet in their names get the clause.
    Poryos2026's ``.vrp`` files are published, and a comment is bytes: adding a
    field there would change every one of their digests to say something their
    names do not.
    """
    head = f"{family} {metric} metric; city {city}; "
    fleet = ""
    if family in FAMILIES_WITH_FLEET_IN_NAME:
        fleet = (
            f"No of trucks: {int(manifest['route_count'])} "
            "(lower bound, fleet not fixed); "
        )
    return f"{head}{fleet}3-decimal seconds/meters; ENU ref in {base}.vrp.json"


def build_base(
    *,
    graph: BridgeGraph,
    nodes: BridgeNodes,
    meta: dict[str, Any],
    manifest: dict[str, Any],
    city: str,
    method_tag: str,
    collection_root: str | Path,
    sample_step: float = DEFAULT_SAMPLE_STEP,
    extension_end: float = DEFAULT_EXTENSION_END,
    generated_at: str | None = None,
    force: bool = False,
    family: str = FAMILY,
    commit_distances: bool = True,
) -> BuiltBase | None:
    """Publish the base: road + geo + distances sidecars and the 3 CVRP instances.

    ``commit_distances=False`` computes and hashes the distance matrices exactly
    as usual, records the pins in every instance's ``arc_costs_source``, and then
    does not write the two ``.distances-*.json.gz`` files. The instance is a
    complete descriptor that a consumer regenerates its matrices for, verified
    against the pins -- the treatment the Blauth2024 family already uses for its
    oversized sidecars. It implies ``pin_shortest_paths``, since a regenerated
    shortest matrix has to come from the published road graph.
    """
    started = time.perf_counter()
    num_customers = len(nodes.node_osm_ids) - 1
    # Generation already solved the bin-packing minimum and recorded it; families
    # that put the fleet in the name read it from there rather than recomputing.
    base = base_instance_name(
        city,
        num_customers,
        method_tag,
        family,
        route_count=(
            int(manifest["route_count"])
            if family in FAMILIES_WITH_FLEET_IN_NAME
            else None
        ),
    )
    root = ensure_collection_root(collection_root, family)
    generated_at = generated_at or time.strftime("%Y-%m-%d")

    side_dir = sidecar_dir(root, city, num_customers, base)
    road_path = side_dir / f"{base}.road.json.gz"
    geo_path = side_dir / f"{base}.geo.json.gz"
    cvrp_json_paths = {
        metric: cvrp_dir(root, metric, city, num_customers, base) / f"{base}.vrp.json"
        for metric in ("euclidean", "shortest", "fastest")
    }
    if not force and road_path.exists() and all(p.exists() for p in cvrp_json_paths.values()):
        return None

    tolerance = simplify_tolerance_for(num_customers)
    generator = _base_generator(
        "generate-base",
        {"city": city, "method": method_tag, "sampling_seed": sampling_seed(base)},
    )
    full = _full_city_road_graph(
        graph,
        nodes,
        base=base,
        sample_step=sample_step,
        simplify_tolerance=tolerance,
        extension_end=extension_end,
        generator=generator,
        family=family,
    )
    # An uncommitted distance sidecar has to be reproducible from published
    # bytes alone, which means the road graph must pin the shortest-distance
    # paths as well as the fastest-time ones.
    road, _ = _trim_road_graph(full, pin_shortest_paths=not commit_distances)
    save_instance_road_graph(road, road_path)
    road_sha = compute_road_graph_sha256(road)

    # Distance matrices: fastest on the trimmed graph (identical to the full
    # graph by the trim gate), shortest on the full graph (its paths may use
    # edges outside the fastest trim).
    fastest_matrix = _round_matrix(free_flow_node_times(road))
    shortest_full = _length_weighted(full)
    shortest_values, shortest_paths = _node_paths(
        shortest_full, want_paths=num_customers <= ROAD_CACHE_MAX_N
    )
    shortest_matrix = _round_matrix(shortest_values)

    if not commit_distances:
        # The pin is a promise that these bytes can be rebuilt from what is
        # published. Check it here rather than discovering at load time that the
        # trim lost an edge the shortest metric needed.
        rebuilt, _ = _node_paths(_length_weighted(road), want_paths=False)
        if _round_matrix(rebuilt) != shortest_matrix:
            raise AssertionError(
                f"{base}: the published road graph does not reproduce "
                "distances-shortest; the trim is missing shortest-path edges"
            )

    distances_sha: dict[str, str] = {}
    for metric, values in (("fastest", fastest_matrix), ("shortest", shortest_matrix)):
        distances = InstanceDistances(
            base_name=base,
            benchmark_name=family,
            metric=metric,
            num_customers=num_customers,
            values=values,
            generator=_base_generator("generate-base"),
        )
        distances_sha[metric] = compute_distances_sha256(distances)
        if commit_distances:
            save_instance_distances(distances, side_dir / f"{base}.distances-{metric}.json.gz")

    # Geo sidecar: nodes + (n <= 100) the complete indexed road cache with
    # fastest paths pinned on the trimmed graph and shortest paths on the
    # full graph, over one shared local vertex table keyed by OSM id.
    road_cache = None
    if num_customers <= ROAD_CACHE_MAX_N:
        _, fastest_paths = _node_paths(road)
        used_osm: set[int] = set()
        for path in fastest_paths.values():
            used_osm.update(road.vertex_osm_ids[v] for v in path)
        for path in shortest_paths.values():
            used_osm.update(shortest_full.vertex_osm_ids[v] for v in path)
        local_osm = sorted(used_osm)
        local_of = {osm: index for index, osm in enumerate(local_osm)}
        cache_paths = {
            "fastest": {
                f"{i}-{j}": [local_of[road.vertex_osm_ids[v]] for v in path]
                for (i, j), path in fastest_paths.items()
            },
            "shortest": {
                f"{i}-{j}": [local_of[shortest_full.vertex_osm_ids[v]] for v in path]
                for (i, j), path in shortest_paths.items()
            },
        }
        road_cache = GeoRoadCache(
            vertex_lonlat=[graph.vertex_lonlat[osm] for osm in local_osm],
            paths=cache_paths,
        )
    meta_nodes = meta["nodes"]
    geo = InstanceGeo(
        base_name=base,
        benchmark_name=family,
        city=city,
        method=method_tag,
        source_osm_file=str(meta.get("source_osm_file", graph.osm_file)),
        reference_lla={
            "lat": float(meta["reference_lla"]["lat"]),
            "lon": float(meta["reference_lla"]["lon"]),
            "alt": float(meta["reference_lla"].get("alt", 0.0)),
        },
        map_options=dict(meta.get("map_options", graph.map_options)),
        nodes=[
            GeoNode(
                instance_node_id=index,
                poi_lon=float(node["poi_lon"]),
                poi_lat=float(node["poi_lat"]),
                enu_x=float(node["enu_x"]),
                enu_y=float(node["enu_y"]),
                demand=int(node["demand"]),
                source_tag=str(node["source_tag"]),
                graph_vertex_id=road.node_vertices[index],
            )
            for index, node in enumerate(meta_nodes)
        ],
        road_cache=road_cache,
        generator=_base_generator("generate-base", {"city": city, "method": method_tag}),
    )
    save_instance_geo(geo, geo_path)
    geo_sha = compute_geo_sha256(geo)
    geo_ref = {
        "path": sidecar_relpath(city, num_customers, base, f"{base}.geo.json.gz"),
        "sha256": geo_sha,
    }

    # The 3 slim CVRP metric instances (+ CVRPLIB .vrp exports for n <= 100).
    matrices = {"shortest": shortest_matrix, "fastest": fastest_matrix}
    coordinates = [[float(node["enu_x"]), float(node["enu_y"])] for node in meta_nodes]
    demands = [int(node["demand"]) for node in meta_nodes]
    for metric in ("euclidean", "shortest", "fastest"):
        if metric == "euclidean":
            source = {"model": "euclidean", "decimals": DISTANCE_DECIMALS}
        else:
            source = {
                "model": "distances-sidecar",
                "distances": {
                    "path": sidecar_relpath(
                        city, num_customers, base, f"{base}.distances-{metric}.json.gz"
                    ),
                    "sha256": distances_sha[metric],
                },
            }
        payload = _static_instance_payload(
            base=base,
            meta=meta,
            manifest=manifest,
            metric=metric,
            arc_costs_source=source,
            city=city,
            method_tag=method_tag,
            geo_ref=geo_ref,
            generated_at=generated_at,
            family=family,
        )
        target = cvrp_json_paths[metric]
        target.parent.mkdir(parents=True, exist_ok=True)
        save_json_to_file(payload, target)
        if num_customers <= VRP_EXPORT_MAX_N:
            if metric == "euclidean":
                import math

                matrix = [
                    [
                        0.0 if i == j else round(math.hypot(bx - ax, by - ay), DISTANCE_DECIMALS)
                        for j, (bx, by) in enumerate(coordinates)
                    ]
                    for i, (ax, ay) in enumerate(coordinates)
                ]
            else:
                matrix = matrices[metric]
            _write_cvrplib(
                target.parent / f"{base}.vrp",
                name=base,
                comment=_cvrplib_comment(family, metric, city, base, manifest),
                coordinates=coordinates,
                demands=demands,
                matrix=matrix,
                capacity=int(manifest["capacity"]),
            )

    return BuiltBase(
        base=base,
        city=city,
        num_customers=num_customers,
        method_tag=method_tag,
        road_sha256=road_sha,
        geo_sha256=geo_sha,
        distances_sha256=distances_sha,
        num_road_vertices=road.num_vertices,
        num_road_edges=len(road.edges),
        build_seconds=time.perf_counter() - started,
        cvrp_paths=cvrp_json_paths,
        distances_committed=commit_distances,
    )


# ---------------------------------------------------------------------------
# Stage: derive-vrptw
# ---------------------------------------------------------------------------


TW_SET_METADATA: dict[str, dict[str, Any]] = {
    TW_SET_TD_SHARED: {
        "name": TW_SET_TD_SHARED,
        "td_paired": True,
        "policy": "route-centered",
        "note": "windows shared verbatim with the base's TDVRPTW twins (post minimal-shared-tw-repair)",
    },
    TW_SET_TIGHT: {
        "name": TW_SET_TIGHT,
        "td_paired": False,
        "policy": "route-centered-tight",
        "note": "static-only TW set; NOT the TDVRPTW windows; never audited under traffic",
    },
    TW_SET_SPREAD: {
        "name": TW_SET_SPREAD,
        "td_paired": False,
        "policy": "uniform-spread",
        "note": "static-only TW set; NOT the TDVRPTW windows; never audited under traffic",
    },
}


def _annotate_tw_set(path: Path) -> bool:
    """Insert ``metadata.tw_set`` into an already-published td-shared VRPTW
    instance, at the position a fresh pipeline run produces (just before
    ``tw_repair``); returns whether the file changed. Everything else is
    preserved byte-for-byte (canonical rewrite of identical content)."""
    import json as _json

    payload = _json.loads(path.read_text())
    metadata = payload["metadata"]
    if "tw_set" in metadata:
        return False
    ordered: dict[str, Any] = {}
    for key, value in metadata.items():
        if key == "tw_repair":
            ordered["tw_set"] = TW_SET_METADATA[TW_SET_TD_SHARED]
        ordered[key] = value
    if "tw_set" not in ordered:
        ordered["tw_set"] = TW_SET_METADATA[TW_SET_TD_SHARED]
    payload["metadata"] = ordered
    save_json_to_file(payload, path)
    return True


def materialize_distances(
    instance_path: str | Path,
    *,
    collection_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Rebuild an instance's distance sidecar from its published road graph.

    The counterpart of ``build_base(commit_distances=False)``. Large instances
    ship as descriptors: the ``.vrp.json`` records the sidecar's path and its
    sha256, and the bytes themselves are regenerated locally rather than stored.
    This recomputes them from the committed ``.road.json.gz`` and refuses to
    write anything whose digest does not match the recorded pin, so a
    materialized sidecar is exactly the one the instance was published against.

    Returns a record describing what happened; ``status`` is ``written``,
    ``kept`` (already present) or ``euclidean`` (nothing to do -- that variant
    derives its costs from the coordinates).
    """
    path = Path(instance_path)
    payload = load_json_from_file(path)
    base = str(payload["instance_name"])
    source = payload.get("arc_costs_source") or {}
    if source.get("model") != "distances-sidecar":
        return {"instance": base, "status": "euclidean", "path": None}

    reference = source["distances"]
    if collection_root is not None:
        root = Path(collection_root)
    else:
        discovered = find_collection_root(path)
        if discovered is None:
            raise FileNotFoundError(
                f"{path}: no {COLLECTION_MARKER_FILENAME} above it; "
                "pass collection_root explicitly"
            )
        root = Path(discovered)
    target = root / reference["path"]
    if target.exists() and not force:
        return {"instance": base, "status": "kept", "path": target}

    metric = str(payload["metric_variant"])
    road_path = target.parent / f"{base}.road.json.gz"
    if not road_path.is_file():
        raise FileNotFoundError(
            f"{base}: road sidecar {road_path} is required to rebuild {metric} distances"
        )
    road = load_instance_road_graph(road_path)

    if metric == "fastest":
        values = free_flow_node_times(road)
    elif metric == "shortest":
        values, _ = _node_paths(_length_weighted(road), want_paths=False)
    else:
        raise ValueError(f"{base}: cannot rebuild distances for metric {metric!r}")

    distances = InstanceDistances(
        base_name=base,
        benchmark_name=str(payload["benchmark_name"]),
        metric=metric,
        num_customers=int(payload["num_customers"]),
        values=_round_matrix(values),
        generator=_base_generator("generate-base"),
    )
    digest = compute_distances_sha256(distances)
    if digest != reference["sha256"]:
        raise AssertionError(
            f"{base}: rebuilt {metric} distances hash {digest}, "
            f"but the instance pins {reference['sha256']}"
        )
    save_instance_distances(distances, target)
    return {"instance": base, "status": "written", "path": target, "sha256": digest}


def derive_vrptw(
    *,
    collection_root: str | Path,
    city: str,
    num_customers: int,
    method_tag: str,
    tw_set: str = TW_SET_TD_SHARED,
    generated_at: str | None = None,
    force: bool = False,
    family: str = FAMILY,
    #: Bin-packing minimum fleet, for families whose base names carry it.
    #: Required for those and rejected for the others -- see
    #: ``naming.base_instance_name``.
    route_count: int | None = None,
) -> tuple[str, Path]:
    """Emit one VRPTW TW-set instance of a base.

    ``td-shared`` emits the candidate finalized by ``build_td`` (audit +
    minimal shared TW repair); the static-only sets (``tight``, ``spread``)
    are final as written and never touched by ``build_td``. Returns
    ``(action, path)`` with action in ``derived`` / ``kept`` / ``annotated``
    (the latter: an existing td-shared instance received the ``tw_set``
    metadata block in place, windows untouched).
    """
    if tw_set not in ALL_TW_SETS:
        raise ValueError(f"unknown TW set {tw_set!r} (expected one of {ALL_TW_SETS})")
    root = Path(collection_root)
    base = base_instance_name(
        city, num_customers, method_tag, family, route_count=route_count
    )
    name = vrptw_instance_name(base, tw_set)
    target_dir = vrptw_dir(root, city, num_customers, base)
    target = target_dir / f"{name}.vrp.json"
    if not force and target.exists():
        if tw_set == TW_SET_TD_SHARED and _annotate_tw_set(target):
            return "annotated", target
        return "kept", target

    cvrp_fastest = cvrp_dir(root, "fastest", city, num_customers, base) / f"{base}.vrp.json"
    if not cvrp_fastest.exists():
        raise FileNotFoundError(f"CVRP fastest instance missing for {base}: run generate-base first")
    import json as _json

    cvrp_payload = _json.loads(cvrp_fastest.read_text())
    fastest = load_instance_distances(
        root / cvrp_payload["arc_costs_source"]["distances"]["path"]
    ).values

    service_times = synthesize_service_times(Random(_stable_seed(base, "service")), num_customers)
    tw_seed_label = "tw" if tw_set == TW_SET_TD_SHARED else f"tw:{tw_set}"
    tw_rng = Random(_stable_seed(base, tw_seed_label))
    if tw_set == TW_SET_SPREAD:
        tw_anchor = "uniform-feasible"
        time_windows = synthesize_time_windows_spread(tw_rng, fastest, service_times)
    else:
        tw_anchor = "free-flow-fastest"
        demands = [int(value) for value in cvrp_payload["demands"]]
        capacity = int(cvrp_payload["vehicle_capacity"])
        anchor_routes, visit_times = construct_anchor_routes(
            fastest, demands, capacity, service_times
        )
        if tw_set == TW_SET_TIGHT:
            time_windows = synthesize_time_windows(
                tw_rng,
                fastest,
                service_times,
                visit_times,
                width_ratio_mean=TIGHT_TW_WIDTH_RATIO_MEAN,
                width_ratio_std=TIGHT_TW_WIDTH_RATIO_STD,
                width_ratio_min=TIGHT_TW_WIDTH_RATIO_MIN,
                width_ratio_max=TIGHT_TW_WIDTH_RATIO_MAX,
            )
        else:
            time_windows = synthesize_time_windows(tw_rng, fastest, service_times, visit_times)
        validate_static_anchor(
            anchor_routes,
            fastest,
            demands,
            capacity,
            service_times,
            time_windows,
        )

    generator_extra: dict[str, Any] = {
        "city": city,
        "method": method_tag,
        "service_seed": _stable_seed(base, "service"),
        "tw_seed": _stable_seed(base, tw_seed_label),
        "tw_anchor": tw_anchor,
    }
    if tw_set != TW_SET_TD_SHARED:
        generator_extra["tw_set"] = tw_set

    payload = dict(cvrp_payload)
    payload["instance_name"] = name
    payload["service_times"] = service_times
    payload["time_windows"] = [list(window) for window in time_windows]
    metadata = dict(cvrp_payload["metadata"])
    metadata["problem_type"] = "VRPTW"
    metadata["generator"] = _base_generator("derive-vrptw", generator_extra)
    if generated_at:
        metadata["generated_at"] = generated_at
    metadata["tw_set"] = TW_SET_METADATA[tw_set]
    if tw_set != TW_SET_SPREAD:
        metadata["tw_anchor"] = {
            "policy": "deterministic-capacity-horizon-nearest-neighbour",
            "routes": anchor_routes,
            "num_routes": len(anchor_routes),
            "min_customers_per_route": min(map(len, anchor_routes)),
            "max_customers_per_route": max(map(len, anchor_routes)),
        }
    payload["metadata"] = metadata
    target_dir.mkdir(parents=True, exist_ok=True)
    save_json_to_file(payload, target)
    return "derived", target


# ---------------------------------------------------------------------------
# Stage: build-td
# ---------------------------------------------------------------------------


@dataclass
class BuiltTDBase:
    base: str
    subinstances: list[str]
    atf_sha256: dict[str, str]
    traffic_sha256: dict[str, str]
    lifted_customers: int
    max_lift_seconds: int
    reduced_customers: int
    max_reduction_seconds: int
    build_seconds: float


def _align_overlay(
    road: InstanceRoadGraph,
    graph: BridgeGraph,
    speeds: BridgeSpeeds,
    family: str = FAMILY,
) -> TrafficOverlay:
    """Project the citywide speed field onto the trimmed graph's edge order,
    clamped at each edge's static free-flow limit."""
    row_of = {(osm_u, osm_v): index for index, (osm_u, osm_v, _, _, _) in enumerate(graph.edges)}
    edge_speeds: list[list[float]] = []
    for u, v, _, speed_limit in road.edges:
        key = (road.vertex_osm_ids[u], road.vertex_osm_ids[v])
        row = speeds.speeds[row_of[key]]
        edge_speeds.append([speed if speed <= speed_limit else speed_limit for speed in row])
    return TrafficOverlay(
        base_name=road.base_name,
        benchmark_name=family,
        traffic_model=speeds.model,
        intensity=speeds.intensity,
        bin_edges=list(road.bin_edges),
        edge_speeds=edge_speeds,
        generator=_base_generator(
            "traffic-sim",
            {
                "seed": speeds.seed,
                "num_trips": speeds.num_trips,
                "params": speeds.params,
                "clamped_at_free_flow": True,
            },
        ),
    )


def _audit_and_lift(
    time_windows: list[tuple[int, int]],
    service_times: list[int],
    anchor_routes: list[list[int]],
    anchor_arcs_by_sub: dict[str, dict[tuple[int, int], Any]],
    horizon_end: float,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Lift shared deadlines enough to certify the full anchor solution.

    Each anchor route is simulated from time zero with waiting under every
    traffic overlay. Earliest bounds are never reduced. Deadlines receive the
    smallest shared integer lift containing every resulting service start.
    The same routes are then rechecked globally, including their depot return.
    """
    lifted = [tuple(window) for window in time_windows]
    repairs: dict[str, Any] = {}
    needed_deadlines = [latest for _, latest in lifted]
    binding: list[str | None] = [None] * len(lifted)
    for sub, arcs in anchor_arcs_by_sub.items():
        for route in anchor_routes:
            clock = 0.0
            previous = 0
            for customer in route:
                arrival = arcs[(previous, customer)].evaluate(clock)
                clock = max(arrival, float(lifted[customer][0]))
                needed = math.ceil(clock)
                if needed > needed_deadlines[customer]:
                    needed_deadlines[customer] = needed
                    binding[customer] = sub
                clock += service_times[customer]
                previous = customer
            returned = arcs[(previous, 0)].evaluate(clock)
            if returned > horizon_end:
                raise AssertionError(
                    f"anchor route {route} returns at {returned} after the horizon under {sub}"
                )

    for customer in range(1, len(lifted)):
        earliest, latest = lifted[customer]
        needed = needed_deadlines[customer]
        if needed > latest:
            repairs[str(customer)] = {
                "deadline_before": latest,
                "deadline_after": needed,
                "deadline_binding_overlay": binding[customer],
            }
            lifted[customer] = (earliest, needed)
        if lifted[customer][0] >= lifted[customer][1]:
            raise AssertionError(f"customer {customer} has a zero-width time window")

    for sub, arcs in anchor_arcs_by_sub.items():
        for route in anchor_routes:
            clock = 0.0
            previous = 0
            for customer in route:
                arrival = arcs[(previous, customer)].evaluate(clock)
                clock = max(arrival, float(lifted[customer][0]))
                if clock > lifted[customer][1]:
                    raise AssertionError(
                        f"anchor misses customer {customer} under {sub}: {clock} > {lifted[customer][1]}"
                    )
                clock += service_times[customer]
                previous = customer
            if arcs[(previous, 0)].evaluate(clock) > horizon_end:
                raise AssertionError(f"anchor route returns after the horizon under {sub}")
    return lifted, repairs


def build_td(
    *,
    collection_root: str | Path,
    graph: BridgeGraph,
    speeds_by_combo: dict[tuple[str, str], BridgeSpeeds],
    city: str,
    num_customers: int,
    method_tag: str,
    generated_at: str | None = None,
    force: bool = False,
    verify: bool = True,
    tdvrptw_only: bool = False,
    reuse_traffic: bool = False,
    family: str = FAMILY,
    #: Bin-packing minimum fleet, for families whose base names carry it.
    #: Required for those and rejected for the others -- see
    #: ``naming.base_instance_name``.
    route_count: int | None = None,
) -> BuiltTDBase | None:
    """Publish the TD layer of a base: 6 overlays, TW lift, 12 slim twins."""
    reuse_traffic = reuse_traffic or tdvrptw_only
    started = time.perf_counter()
    root = Path(collection_root)
    base = base_instance_name(
        city, num_customers, method_tag, family, route_count=route_count
    )
    side_dir = sidecar_dir(root, city, num_customers, base)
    road_path = side_dir / f"{base}.road.json.gz"
    if not road_path.exists():
        raise FileNotFoundError(f"road sidecar missing for {base}: run generate-base first")
    vrptw_path = vrptw_dir(root, city, num_customers, base) / f"{base}.vrp.json"
    if not vrptw_path.exists():
        raise FileNotFoundError(f"VRPTW candidate missing for {base}: run derive-vrptw first")

    combos = [
        (model, intensity)
        for model in TD_MODELS
        for intensity in TD_INTENSITIES
    ]
    missing = [combo for combo in combos if combo not in speeds_by_combo] if not reuse_traffic else []
    if missing:
        raise ValueError(f"missing traffic speeds for {base}: {missing}")

    subs = [subinstance_name(model, intensity) for model, intensity in combos]
    twin_paths = {
        (pt, sub): td_instance_dir(root, pt, city, num_customers, base, sub)
        / f"{td_instance_name(base, *sub.split('-', 1))}.vrp.json"
        for pt in ("TDVRP", "TDVRPTW")
        for sub in subs
    }
    target_problem_types = ("TDVRPTW",) if tdvrptw_only else ("TDVRP", "TDVRPTW")
    if not force and all(
        twin_paths[(problem_type, sub)].exists()
        for problem_type in target_problem_types
        for sub in subs
    ):
        return None

    road = load_instance_road_graph(road_path)
    road_sha = compute_road_graph_sha256(road)

    # Gate: the published distances-fastest equals the road graph's free-flow
    # node times after the family rounding.
    stored = load_instance_distances(side_dir / f"{base}.distances-fastest.json.gz")
    recomputed = _round_matrix(free_flow_node_times(road))
    if stored.values != recomputed:
        raise AssertionError(f"distances-fastest gate failed for {base}")

    import json as _json

    vrptw_payload = _json.loads(vrptw_path.read_text())
    service_times = [int(v) for v in vrptw_payload["service_times"]]
    time_windows = [tuple(int(v) for v in window) for window in vrptw_payload["time_windows"]]
    anchor_routes = [
        [int(customer) for customer in route]
        for route in vrptw_payload.get("metadata", {}).get("tw_anchor", {}).get("routes", [])
    ]
    if not anchor_routes:
        raise ValueError(f"route-centered TW anchor missing for {base}: re-run derive-vrptw")
    anchor_arc_keys = {
        (previous, customer)
        for route in anchor_routes
        for previous, customer in zip([0, *route], [*route, 0])
    }

    # Overlays + per-subinstance materialization (ATFs are TW-independent).
    # Only the anchor arcs are retained for the global TW certificate: full
    # n=1000 ATF sets are heavy and all other arcs are irrelevant here.
    overlay_sha: dict[str, str] = {}
    atf_sha: dict[str, str] = {}
    anchor_arcs_by_sub: dict[str, dict[tuple[int, int], Any]] = {}
    overlay_refs: dict[str, dict[str, str]] = {}
    existing_tdvrp_payloads: dict[str, dict[str, Any]] = {}
    for model, intensity in combos:
        sub = subinstance_name(model, intensity)
        overlay_file = f"{base}.traffic-{sub}.json.gz"
        overlay_path = side_dir / overlay_file
        if reuse_traffic and overlay_path.exists():
            overlay = load_traffic_overlay(overlay_path)
        else:
            if (model, intensity) not in speeds_by_combo:
                raise ValueError(f"missing traffic speeds for {base}: {(model, intensity)}")
            overlay = _align_overlay(road, graph, speeds_by_combo[(model, intensity)], family)
            save_traffic_overlay(overlay, overlay_path)
        overlay_sha[sub] = compute_traffic_overlay_sha256(overlay)
        overlay_refs[sub] = {
            "path": sidecar_relpath(city, num_customers, base, overlay_file),
            "sha256": overlay_sha[sub],
        }
        name = td_instance_name(base, model, intensity)
        probe_payload = {
            "instance_name": name,
            "instance_origin": "OsmCvrpGen",
            "benchmark_name": family,
            "num_customers": num_customers,
            "num_vehicles": None,
            "vehicle_capacity": int(vrptw_payload["vehicle_capacity"]),
            "coordinates": vrptw_payload["coordinates"],
            "demands": vrptw_payload["demands"],
            "service_times": service_times,
            "depot": 0,
            "horizon": list(TD_HORIZON),
            "td": {
                "model": "road-graph",
                "graph": {
                    "path": sidecar_relpath(city, num_customers, base, f"{base}.road.json.gz"),
                    "sha256": road_sha,
                },
                "traffic": dict(overlay_refs[sub]),
                "sample_step": road.sample_step,
                "simplify_tolerance": road.simplify_tolerance,
            },
            "metadata": {},
        }
        instance = td_instance_from_payload(probe_payload)
        anchor_arcs_by_sub[sub] = materialize_selected_atfs_roadgraph(
            instance, road, overlay, anchor_arc_keys
        )
        existing_tdvrp_path = twin_paths[("TDVRP", sub)]
        if existing_tdvrp_path.exists():
            existing_tdvrp = _json.loads(existing_tdvrp_path.read_text())
            existing_tdvrp_payloads[sub] = existing_tdvrp
            atf_sha[sub] = str(existing_tdvrp["td"]["atf_sha256"])
        else:
            atfs = materialize_instance_atfs_roadgraph(instance, road, overlay)
            atf_sha[sub] = compute_atf_sha256(atfs)
            del atfs
        missing_anchor_arcs = anchor_arc_keys - anchor_arcs_by_sub[sub].keys()
        if missing_anchor_arcs:
            raise AssertionError(f"missing anchor arcs under {sub}: {sorted(missing_anchor_arcs)}")

    # Global anchor audit + minimal shared deadline repair; finalize VRPTW.
    lifted, repairs = _audit_and_lift(
        time_windows,
        service_times,
        anchor_routes,
        anchor_arcs_by_sub,
        TD_HORIZON[1],
    )
    vrptw_payload["time_windows"] = [list(window) for window in lifted]
    vrptw_metadata = dict(vrptw_payload["metadata"])
    lift_entries = [e for e in repairs.values() if "deadline_after" in e]
    cut_entries = [e for e in repairs.values() if "earliest_after" in e]
    vrptw_metadata["tw_repair"] = {
        "policy": "global-anchor-minimal-shared-deadline-lift",
        "overlays_audited": subs,
        "lifted_customers": len(lift_entries),
        "max_lift_seconds": max(
            (e["deadline_after"] - e["deadline_before"] for e in lift_entries), default=0
        ),
        "reduced_customers": 0,
        "max_reduction_seconds": 0,
        "anchor_routes": len(anchor_routes),
        "repairs": repairs,
    }
    if generated_at:
        vrptw_metadata["generated_at"] = generated_at
    vrptw_payload["metadata"] = vrptw_metadata
    save_json_to_file(vrptw_payload, vrptw_path)

    # Emit the 12 slim TD twins.
    reference_lla = vrptw_payload.get("reference_lla")
    geo_ref = vrptw_metadata.get("sidecars", {}).get("geo")
    for model, intensity in combos:
        sub = subinstance_name(model, intensity)
        name = td_instance_name(base, model, intensity)
        if reuse_traffic:
            td_metadata = dict(existing_tdvrp_payloads[sub]["metadata"])
            td_metadata.pop("problem_type", None)
            td_metadata["generated_at"] = generated_at or td_metadata.get("generated_at", "")
            td_metadata["generator"] = _base_generator(
                "build-td", {"city": city, "method": method_tag}
            )
            td_metadata["sidecars"] = {"geo": dict(geo_ref)} if geo_ref else {}
        else:
            speeds = speeds_by_combo[(model, intensity)]
            td_metadata = {
                "authors": AUTHORS,
                "generated_at": generated_at or vrptw_metadata.get("generated_at", ""),
                "city": city,
                "method": method_tag,
                "base_instance_name": base,
                "subinstance": sub,
                "generator": _base_generator(
                    "build-td", {"city": city, "method": method_tag}
                ),
                "traffic": {
                    "model": speeds.model,
                    "intensity": speeds.intensity,
                    "seed": speeds.seed,
                    "num_trips": speeds.num_trips,
                    "params": speeds.params,
                },
                "sidecars": {"geo": dict(geo_ref)} if geo_ref else {},
                "notes": (
                    "Time dependence derives from the city's OSM road network: the shared "
                    "road-graph sidecar pins free-flow fastest paths per base; this "
                    "subinstance's traffic overlay carries per-edge hourly speeds; "
                    "arrival-time functions are materialized deterministically on load "
                    "and pinned by td.atf_sha256."
                ),
            }
        common = {
            "instance_name": name,
            "instance_origin": "OsmCvrpGen",
            "benchmark_name": family,
            "num_customers": num_customers,
            "num_vehicles": None,
            "vehicle_capacity": int(vrptw_payload["vehicle_capacity"]),
            "coordinates": vrptw_payload["coordinates"],
            "demands": vrptw_payload["demands"],
            "service_times": service_times,
            "depot": 0,
            "reference_lla": reference_lla,
            "horizon": list(TD_HORIZON),
            "td": {
                "model": "road-graph",
                "graph": {
                    "path": sidecar_relpath(city, num_customers, base, f"{base}.road.json.gz"),
                    "sha256": road_sha,
                },
                "traffic": dict(overlay_refs[sub]),
                "atf_sha256": atf_sha[sub],
                "sample_step": road.sample_step,
                "simplify_tolerance": road.simplify_tolerance,
            },
        }
        tdvrp_payload = dict(common)
        tdvrp_payload["metadata"] = {**td_metadata, "problem_type": "TDVRP"}
        tdvrptw_payload = dict(common)
        tdvrptw_payload["time_windows"] = [list(window) for window in lifted]
        tdvrptw_payload["metadata"] = {
            **td_metadata,
            "problem_type": "TDVRPTW",
            "tw_repair": vrptw_metadata["tw_repair"],
        }
        payloads = (("TDVRPTW", tdvrptw_payload),) if tdvrptw_only else (
            ("TDVRP", tdvrp_payload),
            ("TDVRPTW", tdvrptw_payload),
        )
        for pt, payload in payloads:
            target = twin_paths[(pt, sub)]
            target.parent.mkdir(parents=True, exist_ok=True)
            save_json_to_file(payload, target)
            if verify:
                load_td_instance(target, verify_sha256=True)

    return BuiltTDBase(
        base=base,
        subinstances=subs,
        atf_sha256=atf_sha,
        traffic_sha256=overlay_sha,
        lifted_customers=len(lift_entries),
        max_lift_seconds=max(
            (e["deadline_after"] - e["deadline_before"] for e in lift_entries), default=0
        ),
        reduced_customers=0,
        max_reduction_seconds=0,
        build_seconds=time.perf_counter() - started,
    )
