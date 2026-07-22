"""Time-dependent traffic generation on the tool's own road graph.

A Python port of the workbench's stage-3 traffic stage: per-edge hourly speed
profiles over a 24 h day from two traffic models, exposed as the "TD bridge"
records (:mod:`mamut_routing_tools.td.bridge`) the family builder turns into
TDVRP/TDVRPTW instances.

Traffic models:

- ``wave``: no simulation. Each edge gets a bimodal rush-hour speed dip scaled
  by road class, distance to the city centre and a seeded per-edge jitter.
- ``bpr``: a synthetic commuter population routed on the free-flow fastest
  path; per-edge hourly flows drive the Bureau of Public Roads volume-delay
  function. Routing is grouped by the trip's workplace endpoint (the small POI
  pool), so Dijkstra runs ~twice per workplace instead of once per home:
  ``work -> home`` trips keep a forward tree (bit-identical to a per-origin
  forward routing) and ``home -> work`` trips use a reverse tree over the
  transposed graph, which matches forward routing except on equal-cost
  shortest-path ties. (Ported separately; see :func:`bpr_speeds`.)

:func:`build_bridge` assembles the bridge records in memory (the streamlined
path a per-instance derivation uses); :func:`export_bridge` serializes the same
records to the git-ignored ``graph.json`` / ``speeds-*.json`` / ``nodes-*.json``
intermediate for the cached / inspectable path. The canonical published data is
whatever the family builder freezes into the road-graph speed sidecars, so
speeds are rounded here to keep sidecar size down. This generator is additive:
cross-language RNG means numpy cannot reproduce the original Julia
MersenneTwister stream, so regenerated speeds differ from any previously frozen
overlays by design. Validation is therefore structural (same edge set and bin
count, free-flow clamp, plausible per-intensity speed distributions) and
downstream (materialize ATFs, run the Duration checker), never a byte-for-byte
diff.

Wave speeds may exceed the static free-flow limit by up to the jitter fraction;
the builder clamps every bridge speed to its edge's free-flow limit when it
canonicalizes the overlay sidecar (overlays are slowdowns by contract), so
over-limit bridge speeds are expected.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from mamut_routing_tools.generation.matrices import _reconstruct_path
from mamut_routing_tools.generation.pois import DEFAULT_CATEGORIES, find_pois
from mamut_routing_tools.geo import haversine_m
from mamut_routing_tools.roadgraph.build import SPEED_ROADS_URBAN, RoadGraph, load_road_graph
from mamut_routing_tools.td.bridge import (
    BRIDGE_SCHEMA_VERSION,
    BridgeGraph,
    BridgeNodes,
    BridgeSpeeds,
    serialize_bridge_graph,
    serialize_bridge_nodes,
    serialize_bridge_speeds,
)

# --- bridge schema / time discretization -----------------------------------
TD_NUM_BINS = 24
TD_BIN_SECONDS = 3600.0
TD_SPEED_DECIMALS = 3  # exported speeds are rounded to mm/s
TD_MIN_SPEED_FACTOR = 0.12  # never slower than this fraction of free flow

# --- BPR parameters (Bureau of Public Roads volume-delay function) ----------
TD_BPR_ALPHA = 0.15
TD_BPR_BETA = 4
TD_BPR_MULTIPLIER_CAP = 6.0
# Practical hourly capacity (veh/h) per road class (1 motorway, 2 trunk,
# 3 primary, 4 secondary, 5 tertiary, 6 residential, 7 service, 8 living
# street / pedestrian).
TD_CAPACITY_VEH_H: dict[int, float] = {
    1: 1900.0,
    2: 1600.0,
    3: 1400.0,
    4: 1100.0,
    5: 900.0,
    6: 600.0,
    7: 400.0,
    8: 300.0,
}
# Commuters per graph vertex, per intensity level.
TD_BPR_TRIPS_PER_VERTEX: dict[str, float] = {"light": 2.5, "moderate": 5.0, "heavy": 10.0}

# --- wave-model parameters --------------------------------------------------
# Peak relative speed drop on central arterials, per intensity.
TD_WAVE_AMPLITUDE: dict[str, float] = {"light": 0.25, "moderate": 0.45, "heavy": 0.65}
TD_WAVE_JITTER = 0.08
TD_WAVE_CENTER_DECAY_M = 3000.0
TD_WAVE_FLOOR_SHARE = 0.35  # peripheral edges still see this share of the dip

TD_MODELS: tuple[str, ...] = ("bpr", "wave")
TD_INTENSITIES: tuple[str, ...] = ("light", "moderate", "heavy")


class TrafficModelError(ValueError):
    """Raised on an unknown traffic model, intensity, or an empty edge set."""


@dataclass(frozen=True)
class BridgeEdge:
    """One deduplicated directed edge of the bridge graph.

    ``u`` / ``v`` are 0-based graph vertex indices; ``osm_u`` / ``osm_v`` are
    the OSM node ids that key the bridge.
    """

    u: int
    v: int
    osm_u: int
    osm_v: int
    length_m: float
    road_class: int


def td_free_speed_ms(road_class: int) -> float:
    """Static free-flow limit of a road class in m/s."""
    return SPEED_ROADS_URBAN[road_class] / 3.6


def td_round_speed(value: float) -> float:
    """Round a speed to mm/s, keeping it strictly positive."""
    return max(round(value, TD_SPEED_DECIMALS), 10.0 ** (-TD_SPEED_DECIMALS))


def td_rush_curve(bin_index: int) -> float:
    """Bimodal rush-hour curve at the centre of hourly ``bin_index`` (0..23),
    in [0, 1]. Gaussians peak near 08:15 and 17:45."""
    h = bin_index + 0.5
    g = math.exp(-((h - 8.25) ** 2) / (2 * 1.1**2)) + 0.85 * math.exp(-((h - 17.75) ** 2) / (2 * 1.5**2))
    return min(g, 1.0)


def collect_edges(graph: RoadGraph) -> list[BridgeEdge]:
    """Deduplicated directed edge list, one entry per ``(u, v)`` graph-vertex
    pair, keeping the fastest free-flow representative (min free-flow time,
    then min length). Self-loops and non-positive lengths are dropped.

    ``RoadGraph`` already stores a single edge per ``(u, v)``, so the dedup is
    a safety net; the filters mirror ``td_collect_edges`` exactly. Edges are
    returned sorted by ``(u, v)`` so speed profiles align by position.
    """
    best: dict[tuple[int, int], BridgeEdge] = {}
    for index, (osm_u, osm_v) in enumerate(graph.edges):
        u = graph.vertex_of[osm_u]
        v = graph.vertex_of[osm_v]
        if u == v:
            continue
        length_m = float(graph.edge_weight[index])
        if length_m <= 0:
            continue
        candidate = BridgeEdge(u, v, int(osm_u), int(osm_v), length_m, int(graph.edge_class[index]))
        incumbent = best.get((u, v))
        if incumbent is None:
            best[(u, v)] = candidate
            continue
        new_time = candidate.length_m / SPEED_ROADS_URBAN[candidate.road_class]
        old_time = incumbent.length_m / SPEED_ROADS_URBAN[incumbent.road_class]
        if new_time < old_time or (new_time == old_time and candidate.length_m < incumbent.length_m):
            best[(u, v)] = candidate
    return sorted(best.values(), key=lambda e: (e.u, e.v))


def vertex_latlon(graph: RoadGraph) -> list[tuple[float, float]]:
    """Per-vertex ``(lat, lon)`` indexed by 0-based graph vertex."""
    out: list[tuple[float, float]] = []
    for osm_id in graph.node_of:
        lla = graph.node_lla(osm_id)
        out.append((lla.lat, lla.lon))
    return out


def _center_latlon(vertex_ll: list[tuple[float, float]]) -> tuple[float, float]:
    n = len(vertex_ll)
    return sum(t[0] for t in vertex_ll) / n, sum(t[1] for t in vertex_ll) / n


def bridge_seed(base_seed: int, model: str, intensity: str) -> int:
    """Per-combination seed, matching the workbench's seed arithmetic so the
    recorded ``seed`` field is stable across regenerations."""
    model_index = TD_MODELS.index(model) + 1
    intensity_index = TD_INTENSITIES.index(intensity) + 1
    return base_seed + 101 * model_index + 10007 * intensity_index


# ---------------------------------------------------------------------------
# wave model
# ---------------------------------------------------------------------------


def wave_speeds(
    edges: list[BridgeEdge],
    vertex_ll: list[tuple[float, float]],
    center_latlon: tuple[float, float],
    intensity: str,
    seed: int,
) -> list[list[float]]:
    """Per-edge 24-bin wave speed profiles (m/s), aligned with ``edges``.

    Each edge draws one uniform jitter in edge order, then applies a bimodal
    rush dip scaled by amplitude, road-centre proximity and jitter, clamped to
    a free-flow floor.
    """
    amplitude = TD_WAVE_AMPLITUDE[intensity]
    center_lat, center_lon = center_latlon
    rng = np.random.Generator(np.random.PCG64(seed))
    # One jitter draw per edge, in edge order (a block draw is the same PCG64
    # stream as consecutive scalar draws).
    jitter = (2.0 * rng.random(len(edges)) - 1.0) * TD_WAVE_JITTER
    rush = [td_rush_curve(b) for b in range(TD_NUM_BINS)]

    speeds: list[list[float]] = []
    for index, edge in enumerate(edges):
        u_lat, u_lon = vertex_ll[edge.u]
        v_lat, v_lon = vertex_ll[edge.v]
        mid_lat = (u_lat + v_lat) / 2.0
        mid_lon = (u_lon + v_lon) / 2.0
        centrality = math.exp(-haversine_m(mid_lat, mid_lon, center_lat, center_lon) / TD_WAVE_CENTER_DECAY_M)
        dip_share = TD_WAVE_FLOOR_SHARE + (1.0 - TD_WAVE_FLOOR_SHARE) * centrality
        free = td_free_speed_ms(edge.road_class)
        floor = free * TD_MIN_SPEED_FACTOR
        jitter_factor = 1.0 + float(jitter[index])
        profile = [
            td_round_speed(max(free * (1.0 - amplitude * rush[b] * dip_share) * jitter_factor, floor))
            for b in range(TD_NUM_BINS)
        ]
        speeds.append(profile)
    return speeds


# ---------------------------------------------------------------------------
# bpr model
# ---------------------------------------------------------------------------

# Memory budget for one chunked multi-source Dijkstra call: dist (float64) +
# predecessors (int32) is 12 bytes per (source, vertex) cell. Chunking bounds
# transient memory while amortizing the scipy C-call overhead over many
# sources; the flow accumulation is exact and chunk-size independent.
_DIJKSTRA_CHUNK_BYTES = 256_000_000


def bpr_work_pool(graph: RoadGraph, osm_path: str | Path) -> list[int]:
    """Workplace vertex pool: amenity-POI-snapped graph vertices when the OSM
    file has enough of them (>= 50), otherwise all vertices (uniform
    fallback). Mirrors ``td_work_pool``: each POI snaps to its nearest road
    node, which must itself be a graph vertex (the ``findnode`` + ``md.v``
    membership rule); first-seen dedup; sorted."""
    pool: list[int] = []
    seen: set[int] = set()
    try:
        for poi in find_pois(osm_path, DEFAULT_CATEGORIES):
            osm_id = graph.nearest_node(poi.lat, poi.lon)
            if osm_id is None or osm_id not in graph.vertex_of:
                continue
            vertex = graph.vertex_of[osm_id]
            if vertex not in seen:
                seen.add(vertex)
                pool.append(vertex)
    except Exception:  # noqa: BLE001 - keep the partial pool and warn, as td_work_pool does
        warnings.warn("POI workplace pool failed; falling back to uniform workplaces", stacklevel=2)
    if len(pool) >= 50:
        return sorted(pool)
    return list(range(graph.vertex_count))


def _departure_s(rng: np.random.Generator, mu_h: float, sigma_h: float) -> float:
    """A departure time in seconds: a normal hour clamped into the day."""
    hour = min(max(mu_h + sigma_h * float(rng.standard_normal()), 0.25), 23.75)
    return hour * TD_BIN_SECONDS


def _sample_trips(
    rng: np.random.Generator, commuters: int, num_vertices: int, work_pool: list[int]
) -> list[tuple[int, int, float, bool]]:
    """Commuter trip list ``(origin, destination, departure_s, to_work)``: a
    morning home->work and evening work->home per commuter, plus a lunch round
    trip with probability 0.25. ``to_work`` is ``True`` for the home->work legs
    (morning, lunch back) and ``False`` for the work->home legs (evening, lunch
    return); it lets the router group trips by their workplace endpoint (which
    stays in the small work pool) so each direction needs only a per-workplace
    Dijkstra tree. Draw order matches the workbench for reproducibility."""
    n_work = len(work_pool)
    trips: list[tuple[int, int, float, bool]] = []
    for _ in range(commuters):
        home = int(rng.integers(0, num_vertices))
        work = work_pool[int(rng.integers(0, n_work))]
        if work == home:
            continue
        trips.append((home, work, _departure_s(rng, 8.0, 0.75), True))
        trips.append((work, home, _departure_s(rng, 17.5, 1.0), False))
        if rng.random() < 0.25:
            trips.append((work, home, _departure_s(rng, 12.25, 0.5), False))
            trips.append((home, work, _departure_s(rng, 13.5, 0.5), True))
    return trips


def _reconstruct_reverse_path(pred_row: np.ndarray, home: int, work: int) -> list[int] | None:
    """Rebuild the ``home -> work`` path (in original-graph order) from a
    reverse-Dijkstra predecessor row rooted at ``work`` over the TRANSPOSED
    graph.

    In the transposed graph a shortest-path tree rooted at ``work`` gives, for
    each vertex, its predecessor toward ``work``; because the graph is
    transposed, that predecessor is the *next* vertex on the original ``home ->
    work`` path. Walking predecessors from ``home`` therefore yields the path
    in ``home ... work`` order directly (no reversal). Returns ``[home]`` when
    ``home == work`` (the caller skips ``len < 2``) or ``None`` if unreachable.
    """
    if home == work:
        return [home]
    path = [home]
    cur = home
    while True:
        nxt = int(pred_row[cur])
        if nxt < 0:
            return None
        path.append(nxt)
        if nxt == work:
            break
        cur = nxt
    return path


def _edge_clock_bins(
    path: list[int],
    departure: float,
    times: np.ndarray,
    edge_index: dict[tuple[int, int], int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Map an origin->dest ``path`` to its traversed edge ids and the hourly
    bin each edge is *entered* at, given the trip ``departure`` (seconds).

    The entry clock is a left-fold of the departure time over the traversed
    edge free-flow times, which ``cumsum([departure, *times])`` reproduces
    exactly. This is the single clock/bin routine shared by both routing
    directions, so the only difference from a pure-forward routing is which
    equal-cost shortest path the reverse tree picks. Consecutive vertex pairs
    are looked up in ``edge_index``, breaking on the first missing edge (mirror
    of the graph's directed-edge coverage). Returns ``(None, None)`` when no
    edge is traversed.
    """
    edge_ids: list[int] = []
    for k in range(len(path) - 1):
        edge_id = edge_index.get((path[k], path[k + 1]))
        if edge_id is None:
            break
        edge_ids.append(edge_id)
    if not edge_ids:
        return None, None
    ids = np.asarray(edge_ids, dtype=np.int64)
    # Entry clock per traversed edge: departure, then the running sum of the
    # edge times before it -- a left-fold identical to the sequential
    # ``clock += times[edge]`` accumulation.
    clocks = np.cumsum(np.concatenate(([departure], times[ids])))[:-1]
    bins = np.clip((clocks // TD_BIN_SECONDS).astype(np.int64), 0, TD_NUM_BINS - 1)
    return ids, bins


def _route_and_accumulate(
    graph_csr: csr_matrix,
    by_source: dict[int, list[tuple[int, float]]],
    edge_index: dict[tuple[int, int], int],
    times: np.ndarray,
    num_vertices: int,
    counts: np.ndarray,
    *,
    reverse: bool,
) -> None:
    """Route every trip grouped under a common ``source`` workplace and add its
    per-edge entry-time flows into ``counts`` (flat ``edge_id * TD_NUM_BINS +
    bin`` layout).

    One Dijkstra tree per ``source`` (batched in memory-bounded chunks) serves
    all trips sharing that source. ``by_source[source]`` holds ``(home,
    departure)`` pairs: with ``reverse=False`` the source is the workplace and
    each path is the forward ``work -> home`` route; with ``reverse=True``
    ``graph_csr`` is the transposed graph, the source is still the workplace,
    and each path is rebuilt as the original-graph ``home -> work`` route. In
    both cases the reconstructed ``path`` is already in origin->dest order and
    ``departure`` is that trip's departure, so the shared clock helper applies
    unchanged.
    """
    sources = sorted(by_source)
    if not sources:
        return
    chunk = max(1, min(len(sources), _DIJKSTRA_CHUNK_BYTES // (12 * max(num_vertices, 1))))
    sources_arr = np.asarray(sources, dtype=np.int64)
    for start in range(0, len(sources_arr), chunk):
        block = sources_arr[start : start + chunk]
        dist, pred = dijkstra(graph_csr, directed=True, indices=block, return_predecessors=True)
        edge_id_parts: list[np.ndarray] = []
        bin_parts: list[np.ndarray] = []
        for row, src in enumerate(block):
            pred_row = pred[row]
            dist_row = dist[row]
            source = int(src)
            for home, departure in by_source[source]:
                if not np.isfinite(dist_row[home]):
                    continue
                if reverse:
                    path = _reconstruct_reverse_path(pred_row, home, source)
                else:
                    path = _reconstruct_path(pred_row, source, home)
                if path is None or len(path) < 2:
                    continue
                ids, bins = _edge_clock_bins(path, departure, times, edge_index)
                if ids is None:
                    continue
                edge_id_parts.append(ids)
                bin_parts.append(bins)
        if edge_id_parts:
            flat = np.concatenate(edge_id_parts) * TD_NUM_BINS + np.concatenate(bin_parts)
            counts += np.bincount(flat, minlength=counts.size)


def _accumulate_flows(
    csr: csr_matrix,
    trips: list[tuple[int, int, float, bool]],
    edge_index: dict[tuple[int, int], int],
    times: np.ndarray,
    num_vertices: int,
) -> np.ndarray:
    """Per-edge hourly flows over the free-flow times, grouped by the trip's
    workplace endpoint so Dijkstra runs once per workplace rather than once per
    home.

    Trips come in two directions and both endpoints' workplace side is drawn
    from the small POI pool:

    - ``work -> home`` (``to_work is False``): grouped by origin = workplace and
      routed with a FORWARD Dijkstra tree. Bit-identical to a per-origin
      forward routing.
    - ``home -> work`` (``to_work is True``): grouped by destination = workplace
      and routed with a REVERSE Dijkstra tree over the transposed graph. One
      reverse tree per workplace yields the ``home -> work`` shortest path for
      every home. This diverges from a pure-forward routing only when several
      shortest paths tie on cost (the reverse tree may keep a different
      equal-cost path).

    Both directions feed the same clock/bin accumulation, so the entry-time
    flows stay a left-fold of departure over edge times. Flows are integer
    counts, so per-chunk ``bincount`` gives the same totals as scalar ``+= 1``;
    accumulation order does not matter (exact integer sums).
    """
    n_edges = len(edge_index)
    counts = np.zeros(n_edges * TD_NUM_BINS, dtype=np.int64)
    to_home_by_work: dict[int, list[tuple[int, float]]] = defaultdict(list)
    to_work_by_work: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for origin, destination, departure, to_work in trips:
        if to_work:
            # home -> work: destination is the workplace; store (home, dep).
            to_work_by_work[destination].append((origin, departure))
        else:
            # work -> home: origin is the workplace; store (home, dep).
            to_home_by_work[origin].append((destination, departure))
    _route_and_accumulate(
        csr, to_home_by_work, edge_index, times, num_vertices, counts, reverse=False
    )
    csr_t = csr.transpose().tocsr()
    _route_and_accumulate(
        csr_t, to_work_by_work, edge_index, times, num_vertices, counts, reverse=True
    )
    return counts.reshape(n_edges, TD_NUM_BINS).astype(np.float64)


def _bpr_profiles(edges: list[BridgeEdge], flows: np.ndarray) -> list[list[float]]:
    """BPR volume-delay speeds: ``t = t_free * (1 + alpha*(flow/cap)^beta)``,
    multiplier capped, speed floored at a free-flow fraction, mm/s rounded."""
    free = np.array([td_free_speed_ms(edge.road_class) for edge in edges], dtype=np.float64)[:, None]
    capacity = np.array([TD_CAPACITY_VEH_H.get(edge.road_class, 600.0) for edge in edges], dtype=np.float64)[:, None]
    multiplier = np.minimum(
        1.0 + TD_BPR_ALPHA * (flows / capacity) ** TD_BPR_BETA, TD_BPR_MULTIPLIER_CAP
    )
    raw = np.maximum(free / multiplier, free * TD_MIN_SPEED_FACTOR)
    rounded = np.maximum(np.round(raw, TD_SPEED_DECIMALS), 10.0 ** (-TD_SPEED_DECIMALS))
    return rounded.tolist()


def bpr_speeds(
    graph: RoadGraph,
    edges: list[BridgeEdge],
    osm_path: str | Path,
    intensity: str,
    seed: int,
) -> tuple[list[list[float]], int]:
    """Per-edge 24-bin BPR speed profiles and the total number of trips routed.

    Samples ``trips_per_vertex * |V|`` commuters (homes uniform on vertices,
    workplaces from the POI pool with uniform fallback), routes every trip on
    the free-flow fastest path, accumulates per-edge hourly flows at edge
    entry time, then applies the BPR volume-delay function with class-based
    capacities. Deterministic per seed within Python.

    Routing is grouped by each trip's workplace endpoint (the small POI pool)
    rather than by home, so Dijkstra runs ~2*|work pool| times instead of once
    per home: ``work -> home`` trips keep a forward tree per workplace (bit-
    identical to a per-origin forward routing) and ``home -> work`` trips use a
    reverse tree per workplace over the transposed graph, which matches forward
    routing except on equal-cost shortest-path ties.
    """
    num_vertices = graph.vertex_count
    rng = np.random.Generator(np.random.PCG64(seed))
    commuters = round(TD_BPR_TRIPS_PER_VERTEX[intensity] * num_vertices)
    work_pool = bpr_work_pool(graph, osm_path)
    trips = _sample_trips(rng, commuters, num_vertices, work_pool)

    edge_index = {(edge.u, edge.v): index for index, edge in enumerate(edges)}
    times = np.array([edge.length_m / td_free_speed_ms(edge.road_class) for edge in edges], dtype=np.float64)
    rows = np.fromiter((edge.u for edge in edges), dtype=np.int64, count=len(edges))
    cols = np.fromiter((edge.v for edge in edges), dtype=np.int64, count=len(edges))
    csr = csr_matrix((times, (rows, cols)), shape=(num_vertices, num_vertices))

    flows = _accumulate_flows(csr, trips, edge_index, times, num_vertices)
    return _bpr_profiles(edges, flows), len(trips)


# ---------------------------------------------------------------------------
# bridge payloads and export
# ---------------------------------------------------------------------------


def model_params(model: str, intensity: str) -> dict:
    """Parameter record embedded in a speeds file (provenance only)."""
    if model == "wave":
        return {
            "amplitude": TD_WAVE_AMPLITUDE[intensity],
            "jitter": TD_WAVE_JITTER,
            "center_decay_m": TD_WAVE_CENTER_DECAY_M,
            "floor_share": TD_WAVE_FLOOR_SHARE,
            "min_speed_factor": TD_MIN_SPEED_FACTOR,
        }
    if model == "bpr":
        return {
            "trips_per_vertex": TD_BPR_TRIPS_PER_VERTEX[intensity],
            "bpr_alpha": TD_BPR_ALPHA,
            "bpr_beta": TD_BPR_BETA,
            "multiplier_cap": TD_BPR_MULTIPLIER_CAP,
            # String keys so the record equals its JSON-serialized form (JSON
            # object keys are strings); this is a provenance record only.
            "capacity_veh_h": {str(cls): cap for cls, cap in TD_CAPACITY_VEH_H.items()},
            "min_speed_factor": TD_MIN_SPEED_FACTOR,
            "departures": {
                "morning": [8.0, 0.75],
                "evening": [17.5, 1.0],
                "lunch_return": [12.25, 0.5],
                "lunch_back": [13.5, 0.5],
                "lunch_probability": 0.25,
            },
        }
    raise TrafficModelError(f"unknown traffic model {model!r}; known: {TD_MODELS}")


def build_bridge_graph(
    graph: RoadGraph,
    edges: list[BridgeEdge],
    city_slug: str,
    osm_path: str | Path,
    only_intersections: bool,
    trim_to_connected: bool,
) -> BridgeGraph:
    """Assemble the shared :class:`BridgeGraph` for a road graph and its edges.

    Edges carry the static free-flow limit (m/s, same rounding as the speed
    profiles) so the consumer never needs the class table; every vertex
    incident to an edge ships its WGS84 position for the consumer's geo cache.
    Serializing this record (``serialize_bridge_graph``) then loading it back
    (``load_bridge_graph``) returns an equal record, so the in-memory and disk
    paths never diverge.
    """
    used = sorted({v for edge in edges for v in (edge.u, edge.v)})
    vertex_lonlat: dict[int, tuple[float, float]] = {}
    for graph_vertex in used:
        osm_id = int(graph.node_of[graph_vertex])
        lon, lat = graph.node_lonlat(osm_id)
        vertex_lonlat[osm_id] = (float(lon), float(lat))
    edge_tuples = [
        (
            int(edge.osm_u),
            int(edge.osm_v),
            float(edge.length_m),
            int(edge.road_class),
            td_round_speed(td_free_speed_ms(edge.road_class)),
        )
        for edge in edges
    ]
    return BridgeGraph(
        city=city_slug,
        osm_file=Path(osm_path).name,
        map_options={
            "only_intersections": only_intersections,
            "trim_to_connected_graph": trim_to_connected,
        },
        num_bins=TD_NUM_BINS,
        bin_seconds=TD_BIN_SECONDS,
        edges=edge_tuples,
        vertex_lonlat=vertex_lonlat,
    )


def build_bridge_speeds(
    city_slug: str,
    model: str,
    intensity: str,
    seed: int,
    num_trips: int,
    speeds: list[list[float]],
) -> BridgeSpeeds:
    """Wrap one combination's per-edge speed profiles in a :class:`BridgeSpeeds`."""
    return BridgeSpeeds(
        city=city_slug,
        model=model,
        intensity=intensity,
        seed=seed,
        num_trips=num_trips,
        params=model_params(model, intensity),
        speeds=speeds,
    )


def _write_json(path: Path, payload: dict) -> None:
    """Atomic per-file write (per-process tmp name + rename), so concurrent
    exporters targeting one city directory never clobber each other's tmp."""
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def node_osm_ids_from_meta(
    graph: RoadGraph, meta: dict, *, only_intersections: bool, trim_to_connected: bool
) -> list[int]:
    """Map a stage-1 meta's instance nodes (depot first) to OSM node ids on
    ``graph``.

    The meta's ``graph_vertex_id`` values are indices into the road graph the
    instance was sampled on, so the meta's ``map_options`` must match the graph
    the bridge is built on for the numbering to align (this is the stage-1a /
    stage-3 shared-numbering invariant).
    """
    options = meta.get("map_options", {})
    if (
        bool(options.get("only_intersections", only_intersections)) != only_intersections
        or bool(options.get("trim_to_connected_graph", trim_to_connected)) != trim_to_connected
    ):
        raise TrafficModelError(
            f"meta {meta.get('instance_name')!r} map_options {options} do not match the bridge graph "
            f"options (only_intersections={only_intersections}, trim_to_connected_graph={trim_to_connected}); "
            "the graph_vertex_id numbering would not align"
        )
    node_osm_ids: list[int] = []
    for node in meta["nodes"]:
        gvid = int(node["graph_vertex_id"])
        if not 0 <= gvid < graph.vertex_count:
            raise TrafficModelError(
                f"meta {meta.get('instance_name')!r} graph_vertex_id {gvid} out of range [0, {graph.vertex_count})"
            )
        node_osm_ids.append(graph.node_of[gvid])
    return node_osm_ids


def build_bridge_nodes(
    graph: RoadGraph,
    meta: dict,
    city_slug: str,
    *,
    only_intersections: bool,
    trim_to_connected: bool,
) -> BridgeNodes:
    """Map one stage-1 ``meta``'s instance nodes (depot first) to a :class:`BridgeNodes`."""
    return BridgeNodes(
        city=city_slug,
        instance_base=str(meta["instance_name"]),
        node_osm_ids=node_osm_ids_from_meta(
            graph, meta, only_intersections=only_intersections, trim_to_connected=trim_to_connected
        ),
    )


def _compute_speeds(
    graph: RoadGraph,
    edges: list[BridgeEdge],
    vertex_ll: list[tuple[float, float]],
    center: tuple[float, float],
    osm_path: str | Path,
    model: str,
    intensity: str,
    combo_seed: int,
) -> tuple[list[list[float]], int]:
    """One combination's per-edge speed profiles and trip count (0 for ``wave``)."""
    if model == "wave":
        return wave_speeds(edges, vertex_ll, center, intensity, combo_seed), 0
    return bpr_speeds(graph, edges, osm_path, intensity, combo_seed)


def _validate_combos(
    models: list[str] | tuple[str, ...], intensities: list[str] | tuple[str, ...]
) -> None:
    for model in models:
        if model not in TD_MODELS:
            raise TrafficModelError(f"unknown traffic model {model!r}; known: {TD_MODELS}")
    for intensity in intensities:
        if intensity not in TD_INTENSITIES:
            raise TrafficModelError(f"unknown intensity {intensity!r}; known: {TD_INTENSITIES}")


def _load_bridge_road_graph(
    osm_path: str | Path, only_intersections: bool, trim_to_connected: bool
) -> tuple[RoadGraph, list[BridgeEdge], list[tuple[float, float]], tuple[float, float]]:
    """Load the city road graph and derive the shared bridge inputs once."""
    graph = load_road_graph(
        osm_path, only_intersections=only_intersections, trim_to_connected=trim_to_connected
    )
    edges = collect_edges(graph)
    if not edges:
        raise TrafficModelError(f"road graph for {Path(osm_path).name} produced no usable edges")
    vertex_ll = vertex_latlon(graph)
    return graph, edges, vertex_ll, _center_latlon(vertex_ll)


@dataclass
class BridgeBuild:
    """The full TD bridge for one city, held in memory.

    ``graph`` is shared by every combination; ``speeds`` is keyed by ``(model,
    intensity)``; ``nodes`` is keyed by instance base name (one entry per
    stage-1 meta). The consumer (``mamut_routing_tools.family``) takes these
    records directly, so a per-instance derivation skips the JSON round-trip
    entirely; :func:`export_bridge` serializes the same records to disk for the
    cached / inspectable path.
    """

    graph: BridgeGraph
    speeds: dict[tuple[str, str], BridgeSpeeds]
    nodes: dict[str, BridgeNodes]


def build_bridge(
    *,
    osm_path: str | Path,
    city_slug: str,
    models: list[str] | tuple[str, ...] = TD_MODELS,
    intensities: list[str] | tuple[str, ...] = TD_INTENSITIES,
    seed: int = 42,
    only_intersections: bool = True,
    trim_to_connected: bool = True,
    metas: list[dict] | tuple[dict, ...] = (),
) -> BridgeBuild:
    """Build the full TD bridge for one city in memory (no disk round-trip).

    Loads the city road graph once, collects its deduplicated edges, then
    assembles the shared :class:`BridgeGraph`, a :class:`BridgeSpeeds` for every
    ``model x intensity`` combination, and a :class:`BridgeNodes` for every
    stage-1 ``meta`` dict in ``metas`` (keyed by instance base name). This is the
    streamlined path the per-instance TD derivation uses; it produces exactly the
    records :func:`export_bridge` writes to and reads back from disk.

    Each meta's ``graph_vertex_id`` values must index the same road graph the
    bridge is built on, so the metas' ``map_options`` must match
    ``only_intersections`` / ``trim_to_connected``.
    """
    _validate_combos(models, intensities)
    graph, edges, vertex_ll, center = _load_bridge_road_graph(
        osm_path, only_intersections, trim_to_connected
    )
    bridge_graph = build_bridge_graph(
        graph, edges, city_slug, osm_path, only_intersections, trim_to_connected
    )
    speeds: dict[tuple[str, str], BridgeSpeeds] = {}
    for model in models:
        for intensity in intensities:
            combo_seed = bridge_seed(seed, model, intensity)
            profiles, num_trips = _compute_speeds(
                graph, edges, vertex_ll, center, osm_path, model, intensity, combo_seed
            )
            speeds[(model, intensity)] = build_bridge_speeds(
                city_slug, model, intensity, combo_seed, num_trips, profiles
            )
    nodes: dict[str, BridgeNodes] = {}
    for meta in metas:
        record = build_bridge_nodes(
            graph,
            meta,
            city_slug,
            only_intersections=only_intersections,
            trim_to_connected=trim_to_connected,
        )
        nodes[record.instance_base] = record
    return BridgeBuild(graph=bridge_graph, speeds=speeds, nodes=nodes)


def export_bridge(
    *,
    osm_path: str | Path,
    city_slug: str,
    out_root: str | Path,
    models: list[str] | tuple[str, ...] = TD_MODELS,
    intensities: list[str] | tuple[str, ...] = TD_INTENSITIES,
    seed: int = 42,
    force: bool = False,
    only_intersections: bool = True,
    trim_to_connected: bool = True,
    meta_paths: list[str | Path] | tuple[str | Path, ...] = (),
) -> Path:
    """Serialize the TD bridge for one city under ``<out_root>/<city_slug>/``.

    Writes ``graph.json`` (deduplicated directed edges keyed by OSM node ids),
    ``speeds-<model>-<intensity>.json`` for every requested combination (speed
    profiles aligned with the graph edge order, m/s), one
    ``nodes-<instance_base>.json`` per stage-1 meta in ``meta_paths`` (instance
    node -> OSM node ids, depot first), and a ``bridge-manifest.json``. The bytes
    written are exactly the serialized :func:`build_bridge` records, so
    ``load_bridge_*`` reads them back unchanged. Existing per-combination speed
    files are reused unless ``force=True`` (the disk path's only advantage over
    building in memory: skipping an expensive recompute). Returns the city output
    directory.

    Each meta's ``graph_vertex_id`` values must index the same road graph the
    bridge is built on, so the metas' ``map_options`` must match
    ``only_intersections`` / ``trim_to_connected``.
    """
    _validate_combos(models, intensities)
    graph, edges, vertex_ll, center = _load_bridge_road_graph(
        osm_path, only_intersections, trim_to_connected
    )

    out_dir = Path(out_root) / city_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        out_dir / "graph.json",
        serialize_bridge_graph(
            build_bridge_graph(graph, edges, city_slug, osm_path, only_intersections, trim_to_connected)
        ),
    )

    written: list[str] = []
    for model in models:
        for intensity in intensities:
            speeds_path = out_dir / f"speeds-{model}-{intensity}.json"
            if speeds_path.is_file() and not force:
                written.append(f"{speeds_path.name} (kept)")
                continue
            combo_seed = bridge_seed(seed, model, intensity)
            profiles, num_trips = _compute_speeds(
                graph, edges, vertex_ll, center, osm_path, model, intensity, combo_seed
            )
            _write_json(
                speeds_path,
                serialize_bridge_speeds(
                    build_bridge_speeds(city_slug, model, intensity, combo_seed, num_trips, profiles)
                ),
            )
            written.append(speeds_path.name)

    node_files: list[str] = []
    for meta_path in meta_paths:
        meta = json.loads(Path(meta_path).read_text())
        record = build_bridge_nodes(
            graph, meta, city_slug, only_intersections=only_intersections, trim_to_connected=trim_to_connected
        )
        nodes_path = out_dir / f"nodes-{record.instance_base}.json"
        _write_json(nodes_path, serialize_bridge_nodes(record))
        node_files.append(nodes_path.name)

    _write_json(
        out_dir / "bridge-manifest.json",
        {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "city": city_slug,
            "num_vertices": graph.vertex_count,
            "num_edges": len(edges),
            "speed_files": written,
            "node_files": node_files,
        },
    )
    return out_dir
