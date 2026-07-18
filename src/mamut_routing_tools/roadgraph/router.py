"""Point-to-point routing on a RoadGraph (shortest = metres, fastest =
seconds with the urban speed table), matching the Julia server's
``shortest_route``/``fastest_route`` semantics."""

from __future__ import annotations

import rustworkx as rx

from mamut_routing_tools.roadgraph.build import SPEED_ROADS_URBAN, RoadGraph


def route_vertices(
    graph: RoadGraph,
    from_vertex: int,
    to_vertex: int,
    metric: str,
) -> list[int] | None:
    """Vertex-index path for one edge of a route, or None when unreachable."""
    if metric == "fastest":
        def weight(edge_index: int) -> float:
            return graph.time_weight(edge_index, SPEED_ROADS_URBAN)
    elif metric == "shortest":
        def weight(edge_index: int) -> float:
            return graph.edge_weight[edge_index]
    else:
        raise ValueError(f"Unsupported road metric '{metric}'")

    paths = rx.dijkstra_shortest_paths(graph.graph, from_vertex, target=to_vertex, weight_fn=weight)
    if to_vertex not in paths:
        return None if from_vertex != to_vertex else [from_vertex]
    return list(paths[to_vertex])


def route_lonlat(
    graph: RoadGraph,
    from_vertex: int,
    to_vertex: int,
    metric: str,
) -> list[list[float]] | None:
    vertices = route_vertices(graph, from_vertex, to_vertex, metric)
    if vertices is None:
        return None
    return [graph.node_lonlat(graph.node_of[vertex]) for vertex in vertices]
