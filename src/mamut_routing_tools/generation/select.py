"""Depot and customer selection on a road graph (port of the workbench
selection methods: parametric attach, POI categories, hybrid)."""

from __future__ import annotations

import random
import warnings
from pathlib import Path

from mamut_routing_tools.geo import LLA, haversine_m
from mamut_routing_tools.generation.pois import DEFAULT_CATEGORIES, find_pois
from mamut_routing_tools.roadgraph.build import RoadGraph


def vertex_latlon(graph: RoadGraph) -> list[tuple[float, float]]:
    """Per-vertex (lat, lon), indexed by 0-based graph vertex."""
    out: list[tuple[float, float]] = []
    for osm_id in graph.node_of:
        lla = graph.node_lla(osm_id)
        out.append((lla.lat, lla.lon))
    return out


def pick_depot_vertex(mode: str, vertex_ll: list[tuple[float, float]], rng: random.Random) -> int:
    n = len(vertex_ll)
    if n < 1:
        raise ValueError("Cannot pick depot from empty vertex list")
    if mode == "random":
        return rng.randrange(n)
    if mode == "center":
        c_lat = sum(t[0] for t in vertex_ll) / n
        c_lon = sum(t[1] for t in vertex_ll) / n
        return min(range(n), key=lambda v: haversine_m(vertex_ll[v][0], vertex_ll[v][1], c_lat, c_lon))
    if mode == "corner":
        min_lat = min(t[0] for t in vertex_ll)
        min_lon = min(t[1] for t in vertex_ll)
        return min(range(n), key=lambda v: haversine_m(vertex_ll[v][0], vertex_ll[v][1], min_lat, min_lon))
    raise ValueError(f"Unsupported depot mode '{mode}'")


def sample_clustered_vertices(
    candidates: list[int],
    vertex_ll: list[tuple[float, float]],
    target: int,
    n_seeds: int,
    decay_m: float,
    rng: random.Random,
) -> list[int]:
    if target <= 0:
        return []
    n_seeds = max(1, min(n_seeds, max(1, min(target, len(candidates)))))
    seeds = [rng.choice(candidates) for _ in range(n_seeds)]
    selected: set[int] = set(seeds)

    max_weight = 0.0
    for s in seeds:
        slat, slon = vertex_ll[s]
        w = sum(2.0 ** (-haversine_m(slat, slon, vertex_ll[t][0], vertex_ll[t][1]) / decay_m) for t in seeds)
        max_weight = max(max_weight, w)
    max_weight = max(max_weight, 1e-9)

    attempts = 0
    max_attempts = max(5000, 300 * target)
    while len(selected) < target and attempts < max_attempts:
        attempts += 1
        v = rng.choice(candidates)
        if v in selected:
            continue
        vlat, vlon = vertex_ll[v]
        w = sum(2.0 ** (-haversine_m(vlat, vlon, vertex_ll[s][0], vertex_ll[s][1]) / decay_m) for s in seeds)
        p = min(1.0, max(0.0, w / max_weight))
        if rng.random() <= p:
            selected.add(v)

    if len(selected) < target:
        remainder = [v for v in candidates if v not in selected]
        rng.shuffle(remainder)
        for v in remainder[: target - len(selected)]:
            selected.add(v)

    out = list(selected)
    rng.shuffle(out)
    return out[: min(target, len(out))]


def select_customers_parametric(
    graph: RoadGraph,
    vertex_ll: list[tuple[float, float]],
    depot_vertex: int,
    n_customers: int,
    customer_mode: str,
    n_seeds: int,
    decay_m: float,
    rng: random.Random,
) -> tuple[list[int], list[str]]:
    candidates = [v for v in range(graph.vertex_count) if v != depot_vertex]
    if not candidates:
        raise ValueError("No candidate vertices in map")
    if n_customers > len(candidates):
        warnings.warn(
            f"Requested {n_customers} customers but only {len(candidates)} candidate graph vertices; using all available",
            stacklevel=2,
        )
        n_customers = len(candidates)

    if customer_mode == "random":
        selected = [rng.choice(candidates) for _ in range(n_customers)]
    elif customer_mode == "clustered":
        selected = sample_clustered_vertices(candidates, vertex_ll, n_customers, n_seeds, decay_m, rng)
    elif customer_mode == "random_clustered":
        n_rand = n_customers // 2
        rand_part = [rng.choice(candidates) for _ in range(n_rand)]
        rand_set = set(rand_part)
        rem_candidates = [v for v in candidates if v not in rand_set]
        cl_part = sample_clustered_vertices(rem_candidates, vertex_ll, n_customers - n_rand, n_seeds, decay_m, rng)
        selected = rand_part + cl_part
        rng.shuffle(selected)
    else:
        raise ValueError(f"Unsupported customer mode '{customer_mode}'")

    unique_selected = list(dict.fromkeys(selected))
    if len(unique_selected) < n_customers:
        seen = set(unique_selected)
        remainder = [v for v in candidates if v not in seen]
        rng.shuffle(remainder)
        unique_selected.extend(remainder[: n_customers - len(unique_selected)])
    unique_selected = unique_selected[:n_customers]
    return unique_selected, ["param"] * len(unique_selected)


def select_customers_poi(
    graph: RoadGraph,
    osm_path: str | Path,
    n_customers: int,
    categories: list[str],
    rng: random.Random,
) -> tuple[list[int], list[float], list[float], list[str]]:
    cats = categories or DEFAULT_CATEGORIES
    pois = find_pois(osm_path, cats)
    if not pois:
        raise ValueError(f"No POI found for selected categories: {', '.join(cats)}")

    rows = list(range(len(pois)))
    rng.shuffle(rows)

    taken: set[int] = set()
    verts: list[int] = []
    poi_lats: list[float] = []
    poi_lons: list[float] = []
    for i in rows:
        # The nearest ROAD NODE must itself be a graph vertex, matching the
        # Julia findnode + md.v membership semantic (POIs whose nearest node
        # is mid-segment are dropped, not snapped to the nearest vertex).
        osm_id = graph.nearest_node(pois[i].lat, pois[i].lon)
        if osm_id is None or osm_id not in graph.vertex_of:
            continue
        v = graph.vertex_of[osm_id]
        if v in taken:
            continue
        taken.add(v)
        verts.append(v)
        poi_lats.append(pois[i].lat)
        poi_lons.append(pois[i].lon)
        if len(verts) >= n_customers:
            break

    if len(verts) < n_customers:
        warnings.warn(
            f"Only found {len(verts)} POI-attached unique graph vertices; requested {n_customers}",
            stacklevel=2,
        )
    n_actual = min(n_customers, len(verts))
    return verts[:n_actual], poi_lats[:n_actual], poi_lons[:n_actual], ["poi"] * n_actual


def select_customers_hybrid(
    graph: RoadGraph,
    osm_path: str | Path,
    vertex_ll: list[tuple[float, float]],
    depot_vertex: int,
    n_customers: int,
    categories: list[str],
    poi_share: float,
    customer_mode: str,
    n_seeds: int,
    decay_m: float,
    rng: random.Random,
) -> tuple[list[int], list[float], list[float], list[str]]:
    n_poi = min(n_customers, max(0, round(n_customers * poi_share)))
    n_param = n_customers - n_poi

    poi_v: list[int] = []
    poi_lat: list[float] = []
    poi_lon: list[float] = []
    poi_src: list[str] = []
    if n_poi > 0:
        poi_v, poi_lat, poi_lon, poi_src = select_customers_poi(graph, osm_path, n_poi, categories, rng)

    param_v, _ = select_customers_parametric(
        graph, vertex_ll, depot_vertex, max(n_param, 0), customer_mode, n_seeds, decay_m, rng
    )

    seen = {depot_vertex}
    out_v: list[int] = []
    out_lat: list[float] = []
    out_lon: list[float] = []
    out_src: list[str] = []
    for i, v in enumerate(poi_v):
        if v not in seen:
            seen.add(v)
            out_v.append(v)
            out_lat.append(poi_lat[i])
            out_lon.append(poi_lon[i])
            out_src.append(poi_src[i])
    for v in param_v:
        if v not in seen:
            seen.add(v)
            lat, lon = vertex_ll[v]
            out_v.append(v)
            out_lat.append(lat)
            out_lon.append(lon)
            out_src.append("param")

    if len(out_v) < n_customers:
        candidates = [v for v in range(graph.vertex_count) if v not in seen]
        rng.shuffle(candidates)
        for v in candidates:
            lat, lon = vertex_ll[v]
            out_v.append(v)
            out_lat.append(lat)
            out_lon.append(lon)
            out_src.append("param_fill")
            if len(out_v) >= n_customers:
                break

    if len(out_v) < n_customers:
        raise ValueError("Hybrid method could not gather enough unique customers")
    return out_v[:n_customers], out_lat[:n_customers], out_lon[:n_customers], out_src[:n_customers]
