"""The TD-bridge contract: in-memory types and their JSON serialization.

The bridge is the hand-off between the traffic generator (the producer,
:mod:`mamut_routing_tools.td.traffic`) and the family builder (the consumer,
:mod:`mamut_routing_tools.family`). Three record types carry it:

- :class:`BridgeGraph` — deduplicated directed edges ``(osm_u, osm_v,
  length_m, class, free_speed_ms)`` keyed by OSM node ids (the free-flow limit
  uses the same 3-decimal rounding as the speed profiles) plus every incident
  vertex's WGS84 position;
- :class:`BridgeSpeeds` — per-edge hourly speed profiles (m/s), aligned with
  ``BridgeGraph.edges``;
- :class:`BridgeNodes` — one stage-1 instance's nodes mapped to OSM node ids,
  depot first.

These records ARE the contract; the ``graph.json`` /
``speeds-<model>-<intensity>.json`` / ``nodes-<instance_base>.json`` files are
just their serialized form. ``serialize_bridge_*`` and ``load_bridge_*`` are
exact inverses, so building the records in memory (the streamlined default) and
round-tripping them through disk (the cached / inspectable path) yield equal
records. The JSON is a git-ignored intermediate; every published byte is
canonicalized downstream by the Python builder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

BRIDGE_SCHEMA_VERSION = 2


class BridgeFormatError(ValueError):
    """Raised when a TD bridge file violates the expected format."""


@dataclass
class BridgeGraph:
    city: str
    osm_file: str
    map_options: dict
    num_bins: int
    bin_seconds: float
    edges: list[tuple[int, int, float, int, float]]  # (osm_u, osm_v, length_m, class, free_speed_ms)
    vertex_lonlat: dict[int, tuple[float, float]]  # osm_id -> (lon, lat)


@dataclass
class BridgeSpeeds:
    city: str
    model: str
    intensity: str
    seed: int
    num_trips: int
    params: dict
    speeds: list[list[float]]  # aligned with BridgeGraph.edges


@dataclass
class BridgeNodes:
    city: str
    instance_base: str
    node_osm_ids: list[int]  # depot first


# ---------------------------------------------------------------------------
# serialization (record -> JSON-ready dict)
# ---------------------------------------------------------------------------


def serialize_bridge_graph(graph: BridgeGraph) -> dict:
    """The ``graph.json`` payload for a :class:`BridgeGraph`.

    Vertices ship as ``[osm_id, lon, lat]`` sorted by OSM id (the consumer
    reads them into a dict, so order is cosmetic but kept stable); edges ship as
    ``[osm_u, osm_v, length_m, class, free_speed_ms]`` in ``BridgeGraph.edges``
    order (the speed profiles align with it).
    """
    vertices = sorted(
        ([int(osm_id), float(lon), float(lat)] for osm_id, (lon, lat) in graph.vertex_lonlat.items()),
        key=lambda row: row[0],
    )
    edges = [
        [int(osm_u), int(osm_v), float(length_m), int(road_class), float(free_speed)]
        for osm_u, osm_v, length_m, road_class, free_speed in graph.edges
    ]
    return {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "city": graph.city,
        "osm_file": graph.osm_file,
        "map_options": dict(graph.map_options),
        "num_bins": graph.num_bins,
        "bin_seconds": graph.bin_seconds,
        "speed_unit": "m/s",
        "length_unit": "m",
        "vertices": vertices,
        "edges": edges,
    }


def serialize_bridge_speeds(speeds: BridgeSpeeds) -> dict:
    """The ``speeds-<model>-<intensity>.json`` payload for a :class:`BridgeSpeeds`."""
    return {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "city": speeds.city,
        "model": speeds.model,
        "intensity": speeds.intensity,
        "seed": speeds.seed,
        "num_trips": speeds.num_trips,
        "params": dict(speeds.params),
        "speeds": [list(row) for row in speeds.speeds],
    }


def serialize_bridge_nodes(nodes: BridgeNodes) -> dict:
    """The ``nodes-<instance_base>.json`` payload for a :class:`BridgeNodes`."""
    return {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "city": nodes.city,
        "instance_base": nodes.instance_base,
        "depot_first": True,
        "node_osm_ids": [int(v) for v in nodes.node_osm_ids],
    }


# ---------------------------------------------------------------------------
# deserialization (JSON -> record) with the consumer's validity checks
# ---------------------------------------------------------------------------


def _read_bridge_payload(path: Path) -> dict:
    payload = json.loads(path.read_text())
    version = payload.get("schema_version")
    if version != BRIDGE_SCHEMA_VERSION:
        raise BridgeFormatError(f"{path.name}: unsupported bridge schema_version {version!r}")
    return payload


def load_bridge_graph(path: str | Path) -> BridgeGraph:
    source = Path(path)
    payload = _read_bridge_payload(source)
    edges: list[tuple[int, int, float, int, float]] = []
    for index, entry in enumerate(payload["edges"]):
        if len(entry) != 5:
            raise BridgeFormatError(
                f"{source.name}: edge {index} must be [osm_u, osm_v, length_m, class, free_speed_ms]"
            )
        osm_u, osm_v, length_m, road_class, free_speed = entry
        length_m = float(length_m)
        free_speed = float(free_speed)
        if length_m <= 0:
            raise BridgeFormatError(f"{source.name}: edge {index} has non-positive length {length_m}")
        if free_speed <= 0:
            raise BridgeFormatError(f"{source.name}: edge {index} has non-positive free speed {free_speed}")
        edges.append((int(osm_u), int(osm_v), length_m, int(road_class), free_speed))
    if not edges:
        raise BridgeFormatError(f"{source.name}: no edges")
    vertex_lonlat: dict[int, tuple[float, float]] = {}
    for index, entry in enumerate(payload["vertices"]):
        if len(entry) != 3:
            raise BridgeFormatError(f"{source.name}: vertex {index} must be [osm_id, lon, lat]")
        osm_id, lon, lat = entry
        vertex_lonlat[int(osm_id)] = (float(lon), float(lat))
    for osm_u, osm_v, _, _, _ in edges:
        if osm_u not in vertex_lonlat or osm_v not in vertex_lonlat:
            raise BridgeFormatError(f"{source.name}: edge endpoint without vertex coordinates")
    return BridgeGraph(
        city=str(payload["city"]),
        osm_file=str(payload["osm_file"]),
        map_options=dict(payload["map_options"]),
        num_bins=int(payload["num_bins"]),
        bin_seconds=float(payload["bin_seconds"]),
        edges=edges,
        vertex_lonlat=vertex_lonlat,
    )


def load_bridge_speeds(path: str | Path, graph: BridgeGraph) -> BridgeSpeeds:
    source = Path(path)
    payload = _read_bridge_payload(source)
    speeds = [[float(v) for v in row] for row in payload["speeds"]]
    if len(speeds) != len(graph.edges):
        raise BridgeFormatError(
            f"{source.name}: {len(speeds)} speed rows do not match {len(graph.edges)} graph edges"
        )
    for index, row in enumerate(speeds):
        if len(row) != graph.num_bins:
            raise BridgeFormatError(
                f"{source.name}: row {index} has {len(row)} bins, expected {graph.num_bins}"
            )
        if any(v <= 0 for v in row):
            raise BridgeFormatError(f"{source.name}: row {index} has a non-positive speed")
    return BridgeSpeeds(
        city=str(payload["city"]),
        model=str(payload["model"]),
        intensity=str(payload["intensity"]),
        seed=int(payload["seed"]),
        num_trips=int(payload.get("num_trips", 0)),
        params=dict(payload.get("params", {})),
        speeds=speeds,
    )


def load_bridge_nodes(path: str | Path) -> BridgeNodes:
    source = Path(path)
    payload = _read_bridge_payload(source)
    node_osm_ids = [int(v) for v in payload["node_osm_ids"]]
    if len(node_osm_ids) < 2:
        raise BridgeFormatError(f"{source.name}: need at least depot + 1 customer")
    if len(set(node_osm_ids)) != len(node_osm_ids):
        raise BridgeFormatError(f"{source.name}: node_osm_ids must be distinct")
    if not payload.get("depot_first", False):
        raise BridgeFormatError(f"{source.name}: depot_first marker missing")
    return BridgeNodes(
        city=str(payload["city"]),
        instance_base=str(payload["instance_base"]),
        node_osm_ids=node_osm_ids,
    )
