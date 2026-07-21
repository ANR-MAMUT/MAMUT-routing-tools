"""Time-dependent traffic generation for the MAMUT road-graph model.

Ports the workbench's stage-3 traffic stage to Python on the tool's own road
graph, and exports the plain-JSON "TD bridge" the MAMUT-routing publisher
turns into TDVRP/TDVRPTW instances.
"""

from mamut_routing_tools.td.traffic import (
    BridgeEdge,
    TD_INTENSITIES,
    TD_MODELS,
    TrafficModelError,
    bpr_speeds,
    bpr_work_pool,
    bridge_seed,
    collect_edges,
    export_bridge,
    graph_payload,
    speeds_payload,
    vertex_latlon,
    wave_speeds,
)

__all__ = [
    "BridgeEdge",
    "TD_INTENSITIES",
    "TD_MODELS",
    "TrafficModelError",
    "bpr_speeds",
    "bpr_work_pool",
    "bridge_seed",
    "collect_edges",
    "export_bridge",
    "graph_payload",
    "speeds_payload",
    "vertex_latlon",
    "wave_speeds",
]
