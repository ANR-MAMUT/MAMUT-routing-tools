"""Python-side parity metrics + comparison against the Julia reference.

Usage:
  uv run python tests/parity/compare_roadgraph.py <city.osm> <julia_metrics.json>

Prints a comparison table and exits non-zero when any metric exceeds its
tolerance. Sampled route endpoints come from the Julia output so both sides
route the exact same OSM node pairs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import rustworkx as rx

from mamut_routing_tools.roadgraph import SPEED_ROADS_URBAN, load_road_graph

REL_TOL_LENGTH = 2e-3
REL_TOL_ROUTE = 5e-3


def route_cost(graph, from_osm: int, to_osm: int, metric: str) -> float | None:
    if from_osm not in graph.vertex_of or to_osm not in graph.vertex_of:
        return None
    if metric == "fastest":
        def weight(edge_index: int) -> float:
            return graph.time_weight(edge_index, SPEED_ROADS_URBAN)
    else:
        def weight(edge_index: int) -> float:
            return graph.edge_weight[edge_index]
    lengths = rx.dijkstra_shortest_path_lengths(
        graph.graph, graph.vertex_of[from_osm], weight, goal=graph.vertex_of[to_osm]
    )
    target = graph.vertex_of[to_osm]
    if target not in lengths:
        return None
    return lengths[target]


def main() -> int:
    osm_path = Path(sys.argv[1])
    julia = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

    graph = load_road_graph(osm_path, only_intersections=True, trim_to_connected=True)

    failures: list[str] = []

    def check(label: str, python_value, julia_value, rel_tol: float | None = None) -> None:
        if rel_tol is None or python_value is None or julia_value in (None, 0):
            ok = python_value == julia_value
        else:
            ok = abs(python_value - julia_value) / abs(julia_value) <= rel_tol
        status = "OK " if ok else "FAIL"
        print(f"  {status} {label}: python={python_value} julia={julia_value}")
        if not ok:
            failures.append(label)

    print(f"== {osm_path.name} ==")
    check("vertices", graph.vertex_count, julia["vertices"])
    check("edges", graph.edge_count, julia["edges"])
    check(
        "total_edge_length_km",
        round(sum(graph.edge_weight) / 1000.0, 3),
        julia["total_edge_length_km"],
        REL_TOL_LENGTH,
    )
    python_classes = {}
    for cls in graph.edge_class:
        python_classes[str(cls)] = python_classes.get(str(cls), 0) + 1
    check("edge_class_counts", python_classes, dict(julia["edge_class_counts"]))
    check("ref_lat", graph.ref_lla.lat, julia["ref_lla"]["lat"], 1e-9)
    check("ref_lon", graph.ref_lla.lon, julia["ref_lla"]["lon"], 1e-9)

    for sample in julia["sampled_routes"]:
        from_osm, to_osm = int(sample["from"]), int(sample["to"])
        for metric, key, unit in (("shortest", "shortest_m", "m"), ("fastest", "fastest_s", "s")):
            julia_cost = sample[key]
            python_cost = route_cost(graph, from_osm, to_osm, metric)
            label = f"route {from_osm}->{to_osm} {metric} ({unit})"
            if julia_cost is None:
                check(label, python_cost, None)
            else:
                check(label, round(python_cost, 3) if python_cost is not None else None, round(julia_cost, 3), REL_TOL_ROUTE)

    if failures:
        print(f"PARITY FAILURES: {len(failures)}")
        return 1
    print("PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
