"""Time-dependent traffic generation for the MAMUT road-graph model.

Ports the workbench's stage-3 traffic stage to Python on the tool's own road
graph. :func:`build_bridge` assembles the TD-bridge records in memory (the
streamlined path the family builder and the per-instance derivation consume);
:func:`export_bridge` serializes the same records to the git-ignored JSON
intermediate. The bridge record types and their JSON I/O live in
:mod:`mamut_routing_tools.td.bridge`.
"""

from mamut_routing_tools.td.bridge import (
    BRIDGE_SCHEMA_VERSION,
    BridgeFormatError,
    BridgeGraph,
    BridgeNodes,
    BridgeSpeeds,
    load_bridge_graph,
    load_bridge_nodes,
    load_bridge_speeds,
    serialize_bridge_graph,
    serialize_bridge_nodes,
    serialize_bridge_speeds,
)
from mamut_routing_tools.td.traffic import (
    BridgeBuild,
    BridgeEdge,
    TD_INTENSITIES,
    TD_MODELS,
    TrafficModelError,
    bpr_speeds,
    bpr_work_pool,
    bridge_seed,
    build_bridge,
    build_bridge_graph,
    build_bridge_nodes,
    build_bridge_speeds,
    collect_edges,
    export_bridge,
    node_osm_ids_from_meta,
    vertex_latlon,
    wave_speeds,
)

__all__ = [
    "BRIDGE_SCHEMA_VERSION",
    "BridgeBuild",
    "BridgeEdge",
    "BridgeFormatError",
    "BridgeGraph",
    "BridgeNodes",
    "BridgeSpeeds",
    "TD_INTENSITIES",
    "TD_MODELS",
    "TrafficModelError",
    "bpr_speeds",
    "bpr_work_pool",
    "bridge_seed",
    "build_bridge",
    "build_bridge_graph",
    "build_bridge_nodes",
    "build_bridge_speeds",
    "collect_edges",
    "export_bridge",
    "load_bridge_graph",
    "load_bridge_nodes",
    "load_bridge_speeds",
    "node_osm_ids_from_meta",
    "serialize_bridge_graph",
    "serialize_bridge_nodes",
    "serialize_bridge_speeds",
    "vertex_latlon",
    "wave_speeds",
]
