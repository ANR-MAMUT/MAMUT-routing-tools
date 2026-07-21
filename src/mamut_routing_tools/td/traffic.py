"""Time-dependent traffic generation on the tool's own road graph.

A Python port of the workbench's stage-3 traffic stage: per-edge hourly speed
profiles over a 24 h day from two traffic models, plus the "TD bridge" export
(plain-JSON intermediates the MAMUT-routing publisher turns into TDVRP/TDVRPTW
instances).

Traffic models:

- ``wave``: no simulation. Each edge gets a bimodal rush-hour speed dip scaled
  by road class, distance to the city centre and a seeded per-edge jitter.
- ``bpr``: a synthetic commuter population routed on the free-flow fastest
  path; per-edge hourly flows drive the Bureau of Public Roads volume-delay
  function. (Ported separately; see :func:`bpr_speeds`.)

The bridge is a git-ignored intermediate. The canonical published data is
whatever the publisher freezes into the road-graph speed sidecars, so speeds
are rounded here to keep sidecar size down. This generator is additive: cross-
language RNG means numpy cannot reproduce the original Julia MersenneTwister
stream, so regenerated speeds differ from any previously frozen overlays by
design. Validation is therefore structural (same edge set and bin count,
free-flow clamp, plausible per-intensity speed distributions) and downstream
(materialize ATFs, run the Duration checker), never a byte-for-byte diff.

The emitted files round-trip unchanged through the publisher's bridge loaders;
that JSON contract is fixed and this module emits exactly what those loaders
read. Wave speeds may exceed the static free-flow limit by up to the jitter
fraction; the publisher clamps every bridge speed to its edge's free-flow
limit when it canonicalizes the overlay sidecar (overlays are slowdowns by
contract), so over-limit bridge speeds are expected and match the Julia
bridge.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mamut_routing_tools.geo import haversine_m
from mamut_routing_tools.roadgraph.build import SPEED_ROADS_URBAN, RoadGraph, load_road_graph

# --- bridge schema / time discretization -----------------------------------
TD_BRIDGE_SCHEMA_VERSION = 2
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
# bpr model (ported in M9.2)
# ---------------------------------------------------------------------------


def bpr_speeds(
    graph: RoadGraph,
    edges: list[BridgeEdge],
    osm_path: str | Path,
    center_latlon: tuple[float, float],
    intensity: str,
    seed: int,
) -> tuple[list[list[float]], int]:
    """Per-edge 24-bin BPR speed profiles and the total number of trips routed.

    Not yet ported (M9.2): commuter sampling, POI work pool, per-origin
    Dijkstra with entry-time flow accumulation, and the BPR volume-delay
    function.
    """
    raise NotImplementedError("bpr traffic model not yet ported (Plan 9 M9.2); use model='wave'")


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
            "capacity_veh_h": TD_CAPACITY_VEH_H,
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


def graph_payload(
    graph: RoadGraph,
    edges: list[BridgeEdge],
    city_slug: str,
    osm_path: str | Path,
    only_intersections: bool,
    trim_to_connected: bool,
) -> dict:
    """The ``graph.json`` payload: bridge schema v2.

    Edges carry the static free-flow limit (m/s, same rounding as the speed
    profiles) so the consumer never needs the class table; every vertex
    incident to an edge ships its WGS84 position ``[osm_id, lon, lat]`` (sorted
    by osm_id) for the consumer's geo cache.
    """
    used = sorted({v for edge in edges for v in (edge.u, edge.v)})
    vertices: list[list] = []
    for graph_vertex in used:
        osm_id = graph.node_of[graph_vertex]
        lon, lat = graph.node_lonlat(osm_id)
        vertices.append([osm_id, lon, lat])
    vertices.sort(key=lambda row: row[0])
    edge_rows = [
        [edge.osm_u, edge.osm_v, edge.length_m, edge.road_class, td_round_speed(td_free_speed_ms(edge.road_class))]
        for edge in edges
    ]
    return {
        "schema_version": TD_BRIDGE_SCHEMA_VERSION,
        "city": city_slug,
        "osm_file": Path(osm_path).name,
        "map_options": {
            "only_intersections": only_intersections,
            "trim_to_connected_graph": trim_to_connected,
        },
        "num_bins": TD_NUM_BINS,
        "bin_seconds": TD_BIN_SECONDS,
        "speed_unit": "m/s",
        "length_unit": "m",
        "vertices": vertices,
        "edges": edge_rows,
    }


def speeds_payload(
    city_slug: str,
    model: str,
    intensity: str,
    seed: int,
    num_trips: int,
    speeds: list[list[float]],
) -> dict:
    """The ``speeds-<model>-<intensity>.json`` payload."""
    return {
        "schema_version": TD_BRIDGE_SCHEMA_VERSION,
        "city": city_slug,
        "model": model,
        "intensity": intensity,
        "seed": seed,
        "num_trips": num_trips,
        "params": model_params(model, intensity),
        "speeds": speeds,
    }


def _write_json(path: Path, payload: dict) -> None:
    """Atomic per-file write (per-process tmp name + rename), so concurrent
    exporters targeting one city directory never clobber each other's tmp."""
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


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
) -> Path:
    """Write the TD bridge for one city under ``<out_root>/<city_slug>/``.

    Writes ``graph.json`` (deduplicated directed edges keyed by OSM node ids),
    ``speeds-<model>-<intensity>.json`` for every requested combination (speed
    profiles aligned with the graph edge order, m/s), and a
    ``bridge-manifest.json``. Existing per-combination speed files are reused
    unless ``force=True``. Returns the city output directory.

    ``nodes-<instance_base>.json`` (instance node -> OSM node ids) is not
    emitted here; it depends on the stage-1 meta numbering and is wired in a
    later milestone.
    """
    for model in models:
        if model not in TD_MODELS:
            raise TrafficModelError(f"unknown traffic model {model!r}; known: {TD_MODELS}")
    for intensity in intensities:
        if intensity not in TD_INTENSITIES:
            raise TrafficModelError(f"unknown intensity {intensity!r}; known: {TD_INTENSITIES}")

    graph = load_road_graph(osm_path, only_intersections=only_intersections, trim_to_connected=trim_to_connected)
    edges = collect_edges(graph)
    if not edges:
        raise TrafficModelError(f"road graph for {Path(osm_path).name} produced no usable edges")
    vertex_ll = vertex_latlon(graph)
    center = _center_latlon(vertex_ll)

    out_dir = Path(out_root) / city_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        out_dir / "graph.json",
        graph_payload(graph, edges, city_slug, osm_path, only_intersections, trim_to_connected),
    )

    written: list[str] = []
    for model in models:
        for intensity in intensities:
            speeds_path = out_dir / f"speeds-{model}-{intensity}.json"
            if speeds_path.is_file() and not force:
                written.append(f"{speeds_path.name} (kept)")
                continue
            combo_seed = bridge_seed(seed, model, intensity)
            num_trips = 0
            if model == "wave":
                speeds = wave_speeds(edges, vertex_ll, center, intensity, combo_seed)
            else:
                speeds, num_trips = bpr_speeds(graph, edges, osm_path, center, intensity, combo_seed)
            _write_json(
                speeds_path,
                speeds_payload(city_slug, model, intensity, combo_seed, num_trips, speeds),
            )
            written.append(speeds_path.name)

    _write_json(
        out_dir / "bridge-manifest.json",
        {
            "schema_version": TD_BRIDGE_SCHEMA_VERSION,
            "city": city_slug,
            "num_vertices": graph.vertex_count,
            "num_edges": len(edges),
            "speed_files": written,
            "node_files": [],
        },
    )
    return out_dir
