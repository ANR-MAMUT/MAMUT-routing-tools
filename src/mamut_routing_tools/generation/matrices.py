"""Distance/time/euclidean matrices for a set of instance vertices, plus the
depot-edge road geometry the workbench meta files embed.

Shortest = metres, fastest = seconds (3.6 * metres / km-per-hour class speed),
both ceil'd to int, exactly like the Julia workbench. Matrix rows are computed
with scipy's C Dijkstra over the road graph in CSR form.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from mamut_routing_tools.roadgraph.build import SPEED_ROADS_URBAN, RoadGraph

EdgeGeometry = dict[str, list[list[float]]]


def _metric_csr(graph: RoadGraph, metric: str) -> csr_matrix:
    rows = np.empty(graph.edge_count, dtype=np.int64)
    cols = np.empty(graph.edge_count, dtype=np.int64)
    data = np.empty(graph.edge_count, dtype=np.float64)
    for index, (osm_u, osm_v) in enumerate(graph.edges):
        rows[index] = graph.vertex_of[osm_u]
        cols[index] = graph.vertex_of[osm_v]
        if metric == "fastest":
            data[index] = 3.6 * graph.edge_weight[index] / SPEED_ROADS_URBAN[graph.edge_class[index]]
        else:
            data[index] = graph.edge_weight[index]
    size = graph.vertex_count
    return csr_matrix((data, (rows, cols)), shape=(size, size))


def _reconstruct_path(predecessors: np.ndarray, source: int, target: int) -> list[int] | None:
    if source == target:
        return [source]
    path = [target]
    current = target
    while True:
        parent = int(predecessors[current])
        if parent < 0:
            return None
        path.append(parent)
        if parent == source:
            break
        current = parent
    path.reverse()
    return path


def _path_lonlat(graph: RoadGraph, path: list[int]) -> list[list[float]]:
    return [graph.node_lonlat(graph.node_of[vertex]) for vertex in path]


def compute_matrices(
    graph: RoadGraph,
    vertices: list[int],
) -> tuple[list[list[int]], list[list[int]], EdgeGeometry, EdgeGeometry]:
    """(D_short, D_fast, depot-edge geometry per metric) for instance vertices.

    Geometry covers exactly the depot edges the workbench meta embeds: from the
    depot (vertices[0]) to every customer, and from every customer back to the
    depot, keyed ``"{u}_{v}"`` on graph vertex ids with [lon, lat] polylines.
    """
    n = len(vertices)
    index_array = np.asarray(vertices, dtype=np.int64)
    matrices: dict[str, list[list[int]]] = {}
    geometry: dict[str, EdgeGeometry] = {}

    for metric in ("shortest", "fastest"):
        csr = _metric_csr(graph, metric)
        dist = dijkstra(csr, directed=True, indices=index_array)
        block = dist[:, index_array]
        if not np.all(np.isfinite(block)):
            raise ValueError("Some instance vertices are mutually unreachable on the road graph")
        matrices[metric] = [[math.ceil(value) for value in row] for row in block]

        edge_geom: EdgeGeometry = {}
        depot = vertices[0]
        _, pred_out = dijkstra(csr, directed=True, indices=depot, return_predecessors=True)
        _, pred_in = dijkstra(csr.T.tocsr(), directed=True, indices=depot, return_predecessors=True)
        for k in range(1, n):
            target = vertices[k]
            path_out = _reconstruct_path(pred_out, depot, target)
            if path_out is not None:
                edge_geom[f"{depot}_{target}"] = _path_lonlat(graph, path_out)
            path_back = _reconstruct_path(pred_in, depot, target)
            if path_back is not None:
                edge_geom[f"{target}_{depot}"] = _path_lonlat(graph, list(reversed(path_back)))
        geometry[metric] = edge_geom

    return matrices["shortest"], matrices["fastest"], geometry["shortest"], geometry["fastest"]


def euclidean_matrix_from_vertices(
    graph: RoadGraph,
    vertices: list[int],
) -> tuple[list[list[int]], list[tuple[float, float]]]:
    """ENU-plane euclidean matrix (ceil'd metres) + per-vertex ENU coords."""
    coords: list[tuple[float, float]] = []
    for vertex in vertices:
        east, north, _up = graph.node_enu[graph.node_of[vertex]]
        coords.append((east, north))
    n = len(vertices)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            matrix[i][j] = math.ceil(math.hypot(dx, dy))
    return matrix, coords
