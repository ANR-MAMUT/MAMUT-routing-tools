"""Depot and customer selection on a road graph (port of the workbench
selection methods: parametric attach, POI categories, hybrid)."""

from __future__ import annotations

import random
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any

from mamut_routing_tools.geo import LLA, haversine_m
from mamut_routing_tools.generation.pois import (
    DEFAULT_CATEGORIES,
    POI_CATEGORIES,
    NoPoiFoundError,
    Poi,
    find_pois,
    load_poi_catalog,
)
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


#: How a POI is bound to the road graph. ``nearest_node`` is the strict Julia
#: rule the port was built on; ``nearest_vertex`` snaps to the closest routable
#: point instead, the way manual picking always has.
POI_ATTACH_NEAREST_NODE = "nearest_node"
POI_ATTACH_NEAREST_VERTEX = "nearest_vertex"
POI_ATTACH_MODES = (POI_ATTACH_NEAREST_NODE, POI_ATTACH_NEAREST_VERTEX)
#: Snapping radius when a POI does not sit on a routable point itself. 50 m is
#: about a city block: far enough to catch a shop whose nearest road node is
#: mid-segment, short enough that the customer stays where the amenity is.
DEFAULT_POI_ATTACH_RADIUS_M = 50.0

#: Attachment maps keyed by (extract version, graph identity). Preview,
#: preflight and generate each build a selection from the same extract and the
#: same graph, and the answer to "where does every POI land" is identical for
#: all three -- and independent of the categories, the seed and the size, so one
#: batched pass serves every later call. Bounded like the POI catalog cache, for
#: the same reason: a long session browsing many cities must not grow it forever.
_ATTACHMENT_CACHE: OrderedDict[tuple[Any, ...], list[int | None]] = OrderedDict()
_ATTACHMENT_CACHE_MAX = 4


def _graph_identity(graph: RoadGraph) -> tuple[Any, ...]:
    """Enough of a graph to tell two builds of the same extract apart."""
    return (
        str(Path(graph.osm_path).resolve()),
        graph.only_intersections,
        graph.trim_to_connected,
        graph.loaded_with,
        graph.vertex_count,
        graph.edge_count,
    )


def attach_pois(
    graph: RoadGraph,
    pois: list[Poi],
    mode: str = POI_ATTACH_NEAREST_NODE,
    radius_m: float = DEFAULT_POI_ATTACH_RADIUS_M,
) -> list[tuple[int, float] | None]:
    """Per POI, the ``(vertex, snap_distance_m)`` it attaches to, or ``None``.

    ``nearest_node`` is the Julia ``findnode`` + ``md.v`` membership semantic:
    the nearest ROAD NODE must itself be a graph vertex, so a POI whose closest
    node sits mid-segment is dropped even when an intersection stands a metre
    further. It keeps generated instances identical to the published pipeline,
    at the cost of discarding most of a city's amenities.

    ``nearest_vertex`` instead snaps each POI to the closest routable vertex
    within ``radius_m``, which is what manual picking has always done. Far more
    of the pool becomes usable; the distance is returned so every customer can
    record how far its amenity was moved.
    """
    if mode == POI_ATTACH_NEAREST_VERTEX:
        return graph.nearest_vertices([(poi.lat, poi.lon) for poi in pois], radius_m)
    if mode != POI_ATTACH_NEAREST_NODE:
        raise ValueError(f"Unsupported POI attach mode '{mode}'")
    nodes = graph.nearest_nodes([(poi.lat, poi.lon) for poi in pois])
    out: list[tuple[int, float] | None] = []
    for osm_id in nodes:
        vertex = graph.vertex_of.get(osm_id) if osm_id is not None else None
        # Zero by construction: the POI attached to the node it is nearest to.
        out.append(None if vertex is None else (vertex, 0.0))
    return out


def catalog_attachment(
    graph: RoadGraph,
    osm_path: str | Path,
    mode: str = POI_ATTACH_NEAREST_NODE,
    radius_m: float = DEFAULT_POI_ATTACH_RADIUS_M,
) -> list[tuple[int, float] | None]:
    """``attach_pois`` over the whole cached catalog, computed at most once."""
    path = Path(osm_path)
    stat = path.stat()
    key = (
        str(path.resolve()),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        _graph_identity(graph),
        mode,
        # Only the snapping mode reads the radius; folding it in unconditionally
        # would split the cache for a value the strict mode ignores.
        radius_m if mode == POI_ATTACH_NEAREST_VERTEX else None,
    )
    cached = _ATTACHMENT_CACHE.get(key)
    if cached is not None:
        _ATTACHMENT_CACHE.move_to_end(key)
        return cached
    attachment = attach_pois(graph, load_poi_catalog(path), mode, radius_m)
    _ATTACHMENT_CACHE[key] = attachment
    while len(_ATTACHMENT_CACHE) > _ATTACHMENT_CACHE_MAX:
        _ATTACHMENT_CACHE.popitem(last=False)
    return attachment


def select_customers_poi(
    graph: RoadGraph,
    osm_path: str | Path,
    n_customers: int,
    categories: list[str],
    rng: random.Random,
    stats: dict[str, Any] | None = None,
    attach_mode: str = POI_ATTACH_NEAREST_NODE,
    attach_radius_m: float = DEFAULT_POI_ATTACH_RADIUS_M,
) -> tuple[list[int], list[float], list[float], list[str], list[Poi | None]]:
    """Chosen vertices plus, per vertex, the POI it came from.

    The trailing ``list[Poi | None]`` carries the amenity value, OSM node id and
    display name so callers can persist them; ``None`` marks a non-POI customer.

    ``stats``, when given, is filled with how big the pool actually was:
    ``pool_matching`` (POIs of these categories in the extract), ``attached``
    (the number of distinct graph vertices the whole pool can reach, which is
    the hard ceiling on POI customers) and ``co_located`` (per chosen vertex,
    the other POIs that snap to it and are therefore *not* customers of their
    own). Callers need the first two to explain a shortfall and the last to say
    which amenities a single customer stands for.
    """
    cats = categories or DEFAULT_CATEGORIES
    # The cached catalog holds every known category in file order, so filtering
    # it gives exactly what find_pois would return without re-parsing an extract
    # that preview, preflight and generate each walk in turn -- and the cached
    # attachment map spares them the spatial search too. A category outside the
    # catalog's own list is not in there, so that case still parses and searches.
    if set(cats) <= set(POI_CATEGORIES):
        wanted = set(cats)
        catalog = load_poi_catalog(osm_path)
        attachment = catalog_attachment(graph, osm_path, attach_mode, attach_radius_m)
        selected = [index for index, poi in enumerate(catalog) if poi.category in wanted]
        pois = [catalog[index] for index in selected]
        poi_hits = [attachment[index] for index in selected]
    else:
        pois = find_pois(osm_path, list(cats))
        poi_hits = attach_pois(graph, pois, attach_mode, attach_radius_m)
    if stats is not None:
        stats.update(
            {"pool_matching": len(pois), "attached": 0, "co_located": {}, "snap_m": {}}
        )
    if not pois:
        raise NoPoiFoundError(f"No POI found for selected categories: {', '.join(cats)}")

    rows = list(range(len(pois)))
    rng.shuffle(rows)

    taken: set[int] = set()
    verts: list[int] = []
    poi_lats: list[float] = []
    poi_lons: list[float] = []
    sources: list[Poi | None] = []
    # Two amenities on the same street corner collapse into one customer. The
    # ones that lose are kept here, per winning vertex, so the instance can say
    # which places that single customer actually stands for.
    co_located: dict[int, list[Poi]] = {}
    attachable: set[int] = set()
    # How far each winning POI had to move to reach its vertex; zero under the
    # strict rule, which never moves a POI at all.
    snap_m: dict[int, float] = {}
    for i in rows:
        hit = poi_hits[i]
        if hit is None:
            continue
        v, distance = hit
        attachable.add(v)
        if v in taken:
            co_located.setdefault(v, []).append(pois[i])
            continue
        # The pool keeps being walked past the requested count, which costs one
        # spatial lookup per remaining POI and buys two things the selection
        # cannot report otherwise: the true attachable ceiling, and the full set
        # of amenities standing behind each chosen point.
        if len(verts) >= n_customers:
            continue
        taken.add(v)
        verts.append(v)
        poi_lats.append(pois[i].lat)
        poi_lons.append(pois[i].lon)
        sources.append(pois[i])
        snap_m[v] = distance

    if stats is not None:
        stats.update(
            {"attached": len(attachable), "co_located": co_located, "snap_m": snap_m}
        )
    if len(verts) < n_customers:
        warnings.warn(
            f"Only found {len(verts)} POI-attached unique graph vertices; requested {n_customers}",
            stacklevel=2,
        )
    n_actual = min(n_customers, len(verts))
    return (
        verts[:n_actual],
        poi_lats[:n_actual],
        poi_lons[:n_actual],
        ["poi"] * n_actual,
        sources[:n_actual],
    )


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
    stats: dict[str, Any] | None = None,
    attach_mode: str = POI_ATTACH_NEAREST_NODE,
    attach_radius_m: float = DEFAULT_POI_ATTACH_RADIUS_M,
) -> tuple[list[int], list[float], list[float], list[str], list[Poi | None]]:
    n_poi = min(n_customers, max(0, round(n_customers * poi_share)))
    n_param = n_customers - n_poi

    poi_v: list[int] = []
    poi_lat: list[float] = []
    poi_lon: list[float] = []
    poi_src: list[str] = []
    poi_meta: list[Poi | None] = []
    if n_poi > 0:
        try:
            poi_v, poi_lat, poi_lon, poi_src, poi_meta = select_customers_poi(
                graph, osm_path, n_poi, categories, rng, stats, attach_mode, attach_radius_m
            )
        except NoPoiFoundError:
            # These categories hold nothing in this extract. The parametric half
            # of a hybrid run is perfectly buildable, so the whole selection
            # becomes parametric rather than failing; ``stats`` already carries
            # the empty pool, which is what the caller warns about.
            n_param = n_customers
    elif stats is not None:
        stats.update({"pool_matching": 0, "attached": 0, "co_located": {}})

    param_v, _ = select_customers_parametric(
        graph, vertex_ll, depot_vertex, max(n_param, 0), customer_mode, n_seeds, decay_m, rng
    )

    seen = {depot_vertex}
    out_v: list[int] = []
    out_lat: list[float] = []
    out_lon: list[float] = []
    out_src: list[str] = []
    out_meta: list[Poi | None] = []
    used_as_depot = 0
    depot_pois: list[Poi] = []
    for i, v in enumerate(poi_v):
        if v not in seen:
            seen.add(v)
            out_v.append(v)
            out_lat.append(poi_lat[i])
            out_lon.append(poi_lon[i])
            out_src.append(poi_src[i])
            out_meta.append(poi_meta[i])
        elif v == depot_vertex:
            # A chosen POI that is the depot is one POI customer fewer, which
            # the caller has to be able to explain -- and a name for the depot.
            used_as_depot += 1
            if poi_meta[i] is not None:
                depot_pois.append(poi_meta[i])
    if stats is not None:
        stats["used_as_depot"] = used_as_depot
        stats["depot_pois"] = depot_pois
    for v in param_v:
        if v not in seen:
            seen.add(v)
            lat, lon = vertex_ll[v]
            out_v.append(v)
            out_lat.append(lat)
            out_lon.append(lon)
            out_src.append("param")
            out_meta.append(None)

    if len(out_v) < n_customers:
        candidates = [v for v in range(graph.vertex_count) if v not in seen]
        rng.shuffle(candidates)
        for v in candidates:
            lat, lon = vertex_ll[v]
            out_v.append(v)
            out_lat.append(lat)
            out_lon.append(lon)
            out_src.append("param_fill")
            out_meta.append(None)
            if len(out_v) >= n_customers:
                break

    if len(out_v) < n_customers:
        raise ValueError("Hybrid method could not gather enough unique customers")
    return (
        out_v[:n_customers],
        out_lat[:n_customers],
        out_lon[:n_customers],
        out_src[:n_customers],
        out_meta[:n_customers],
    )
