"""Single-instance generation driver (selection, demands, matrices, artifacts),
the port of the workbench's build_generation_selection + generate_single_instance."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from mamut_routing_tools.generation.binpacking import minimum_bins
from mamut_routing_tools.generation.demands import capacity_from_avg_route_size, generate_demands
from mamut_routing_tools.generation.matrices import compute_matrices, euclidean_matrix_from_vertices
from mamut_routing_tools.generation.pois import (
    DEFAULT_CATEGORIES,
    POI_CATEGORIES,
    Poi,
    is_poi_source_tag,
    load_poi_catalog,
    poi_ref,
)
from mamut_routing_tools.generation.select import (
    DEFAULT_POI_ATTACH_RADIUS_M,
    POI_ATTACH_MODES,
    POI_ATTACH_NEAREST_NODE,
    POI_ATTACH_NEAREST_VERTEX,
    pick_depot_vertex,
    select_customers_hybrid,
    select_customers_parametric,
    select_customers_poi,
    vertex_latlon,
)
from mamut_routing_tools.generation.writers import (
    build_vrp_json_payload,
    instance_path_plan,
    parse_cvrp_vrp,
    slugify,
    write_cvrplib,
    write_instance_metadata,
    write_json,
)
from mamut_routing_tools.roadgraph.build import RoadGraph, load_road_graph

METHODS = ("poi_categories", "parametric_attach", "hybrid", "manual")
# Distance beyond which a hand-picked POI is reported as unreachable rather than
# snapped; generous, because the picker only offers POIs inside the extract.
MANUAL_SNAP_MAX_M = 2000.0
DEPOT_MODES = ("random", "center", "corner")
CUSTOMER_MODES = ("random", "clustered", "random_clustered")
METRICS = ("shortest", "fastest", "euclidean")


@dataclass
class GenerationRequest:
    city: str
    osm_path: Path
    method: str = "poi_categories"
    n_customers: int = 50
    seed: int = 0
    demand_type: int = 7
    avg_route_size: int = 4
    depot_mode: str = "center"
    customer_mode: str = "random_clustered"
    cluster_seeds: int | None = None
    cluster_decay_meters: float = 800.0
    categories: list[str] = field(default_factory=lambda: list(DEFAULT_CATEGORIES))
    hybrid_poi_share: float = 0.5
    only_intersections: bool = True
    trim_to_connected_graph: bool = True
    # How a POI binds to the road graph. The default is the strict rule the port
    # inherited, which keeps instances identical to the published pipeline;
    # "nearest_vertex" snaps within poi_attach_radius_m instead and makes most
    # of a city's amenities usable. See generation.select.attach_pois.
    poi_attach_mode: str = POI_ATTACH_NEAREST_NODE
    poi_attach_radius_m: float = DEFAULT_POI_ATTACH_RADIUS_M
    # method="manual" only: OSM ids of the hand-picked POIs, and optionally the
    # one to use as the depot (falls back to depot_mode when unset). Ids alone
    # mean nodes, which is what every pick was before ways and relations became
    # selectable; ``manual_poi_refs`` carries ``(osm_type, osm_id)`` pairs and is
    # what a caller aware of the other element types should send.
    manual_poi_ids: list[int] = field(default_factory=list)
    manual_depot_poi_id: int | None = None
    manual_poi_refs: list[tuple[str, int]] = field(default_factory=list)
    manual_depot_poi_ref: tuple[str, int] | None = None

    def picked_refs(self) -> list[tuple[str, int]]:
        """Every hand-picked POI as a ``(osm_type, osm_id)`` pair, in pick order."""
        refs = [(str(kind), int(value)) for kind, value in self.manual_poi_refs]
        seen = set(refs)
        for osm_id in self.manual_poi_ids:
            ref = ("node", int(osm_id))
            if ref not in seen:
                seen.add(ref)
                refs.append(ref)
        return refs

    def depot_ref(self) -> tuple[str, int] | None:
        if self.manual_depot_poi_ref is not None:
            kind, value = self.manual_depot_poi_ref
            return (str(kind), int(value))
        if self.manual_depot_poi_id is not None:
            return ("node", int(self.manual_depot_poi_id))
        return None

    def validate(self) -> None:
        if not self.city:
            raise ValueError("Missing 'city'")
        if not Path(self.osm_path).is_file():
            raise FileNotFoundError(f"OSM file not found: {self.osm_path}")
        if self.method not in METHODS:
            raise ValueError(f"Unsupported method '{self.method}'")
        if self.method == "manual":
            picked = set(self.picked_refs()) - {self.depot_ref()}
            if len(picked) < 2:
                raise ValueError(
                    "Manual selection needs at least 2 picked POIs besides the depot"
                )
        if self.n_customers < 2:
            raise ValueError("n_customers must be >= 2")
        if self.demand_type not in range(1, 8):
            raise ValueError("demand_type must be between 1 and 7")
        if self.avg_route_size not in range(1, 8):
            raise ValueError("avg_route_size must be between 1 and 7")
        if self.depot_mode not in DEPOT_MODES:
            raise ValueError(f"Unsupported depot mode '{self.depot_mode}'")
        if self.customer_mode not in CUSTOMER_MODES:
            raise ValueError(f"Unsupported customer mode '{self.customer_mode}'")
        if self.cluster_seeds is not None and self.cluster_seeds < 1:
            raise ValueError("cluster_seeds must be >= 1")
        if not math.isfinite(self.cluster_decay_meters) or self.cluster_decay_meters <= 0:
            raise ValueError("cluster_decay_meters must be a positive finite number")
        if not math.isfinite(self.hybrid_poi_share) or not 0.0 <= self.hybrid_poi_share <= 1.0:
            raise ValueError("hybrid_poi_share must be between 0 and 1")
        if self.poi_attach_mode not in POI_ATTACH_MODES:
            raise ValueError(
                f"Unsupported POI attach mode '{self.poi_attach_mode}'; "
                f"expected one of {', '.join(POI_ATTACH_MODES)}"
            )
        if not math.isfinite(self.poi_attach_radius_m) or self.poi_attach_radius_m <= 0:
            raise ValueError("poi_attach_radius_m must be a positive finite number")


def effective_method(requested_method: str, customer_tags: list[str]) -> str:
    """The method that describes what was actually selected.

    A POI request the extract cannot serve is completed with parametric road
    points, which makes the instance a hybrid one -- or a parametric one when no
    amenity could be used at all. Naming it after the request would put ``_poi``
    on a file whose customers sit on plain intersections, so the composition has
    the last word. Only the fallback direction is reclassified: a hybrid that
    reaches its POI target stays hybrid, because that mix was asked for.
    """
    if requested_method not in ("poi_categories", "hybrid"):
        return requested_method
    has_poi = any(is_poi_source_tag(tag) for tag in customer_tags)
    has_parametric = any(not is_poi_source_tag(tag) for tag in customer_tags)
    if not has_poi:
        return "parametric_attach"
    if requested_method == "poi_categories" and has_parametric:
        return "hybrid"
    return requested_method


def graph_option_params(request: GenerationRequest, graph: RoadGraph) -> dict[str, Any]:
    """The graph options actually in force, plus the ones that were asked for.

    ``load_road_graph`` relaxes ``only_intersections`` / ``trim_to_connected``
    when the requested combination yields an empty graph, so recording the
    request would make an instance claim a graph it does not have. The effective
    values are the ones written to the metadata; the request is kept beside them
    and the difference is what the caller warns about.

    ``graph.loaded_with`` is the loader's own record. A graph built without it
    went through no fallback chain, so the request stands.
    """
    requested = {
        "only_intersections": request.only_intersections,
        "trim_to_connected_graph": request.trim_to_connected_graph,
    }
    loaded = graph.loaded_with
    effective = (
        {"only_intersections": loaded[0], "trim_to_connected_graph": loaded[1]}
        if loaded is not None
        else dict(requested)
    )
    return {
        **effective,
        "requested_graph_options": requested,
        "graph_options_relaxed": requested if requested != effective else None,
    }


def selection_composition(
    *,
    method: str,
    requested: int,
    customer_tags: list[str],
    poi_stats: dict[str, Any],
    poi_share: float,
    categories: list[str],
    graph_options_relaxed: dict[str, Any] | None = None,
    effective: str | None = None,
    attach_mode: str = POI_ATTACH_NEAREST_NODE,
) -> dict[str, Any]:
    """What the selection really is made of, versus what was asked for.

    A POI-backed method that runs out of amenities keeps going with parametric
    road points, which is the right fallback but changes the nature of the
    instance. This record is what lets the caller say so before generating:
    ``poi_shortfall`` is how many customers meant to sit on a real amenity did
    not, and ``poi_pool_attachable`` is the ceiling that caused it (``None``
    when the pool was never exhausted, so no ceiling was observed).
    """
    poi_customers = sum(1 for tag in customer_tags if is_poi_source_tag(tag))
    delivered = len(customer_tags)
    if method == "poi_categories":
        poi_target = requested
    elif method == "hybrid":
        # Mirrors select_customers_hybrid, so the target matches what it aimed for.
        poi_target = min(requested, max(0, round(requested * poi_share)))
    else:
        poi_target = 0
    return {
        # The requested method: the advice below is about the controls the user
        # actually has. ``effective_method`` is what the instance is recorded as.
        "method": method,
        "effective_method": effective if effective is not None else method,
        "requested": requested,
        "delivered": delivered,
        "poi_customers": poi_customers,
        "parametric_customers": delivered - poi_customers,
        "poi_target": poi_target,
        "poi_shortfall": max(0, poi_target - poi_customers),
        "poi_share": poi_share if method == "hybrid" else None,
        "poi_pool_matching": int(poi_stats.get("pool_matching", 0)),
        # The whole pool is walked, so this is the real ceiling on POI customers
        # for these categories, not just where the search happened to stop.
        "poi_pool_attachable": (
            int(poi_stats["attached"]) if "attached" in poi_stats else None
        ),
        # A POI whose vertex is the depot cannot also be a customer; without this
        # the resulting shortfall looks like a missing amenity.
        "poi_used_as_depot": int(poi_stats.get("used_as_depot", 0)),
        # POIs the attachment rule discarded outright. Under the strict rule
        # this is most of a city, and it is the first thing to explain when a
        # user cannot find an amenity they know is in the extract.
        "poi_pool_unattached": max(
            0, int(poi_stats.get("pool_matching", 0)) - int(poi_stats.get("attached", 0))
        ),
        "poi_attach_mode": attach_mode,
        "category_count": len(categories) if poi_target else 0,
        "category_total": len(POI_CATEGORIES),
        "graph_options_relaxed": graph_options_relaxed,
    }


def composition_notice(composition: dict[str, Any] | None) -> str | None:
    """Everything about this selection the user did not ask for, or ``None``.

    Names the final split, the pool that capped it, and the ways out (fewer
    customers, more categories to draw from), plus any option the loader had to
    relax to build the graph at all.
    """
    if not composition:
        return None
    relaxed = composition.get("graph_options_relaxed")
    relaxed_note = (
        "The requested road-graph options (only intersections="
        f"{relaxed['only_intersections']}, trim to connected graph="
        f"{relaxed['trim_to_connected_graph']}) yield an empty graph for this extract, so it is "
        "loaded with relaxed options; the instance records the ones actually used."
        if relaxed
        else None
    )
    if composition.get("method") == "manual":
        notes: list[str | None] = []
        missing = int(composition["requested"]) - int(composition["delivered"])
        if missing > 0:
            dropped = composition.get("dropped") or {}
            reasons = [
                f"{dropped[key]} {label}"
                for key, label in (
                    ("unknown", "not found in this extract"),
                    ("unreachable", "too far from any road"),
                    ("collapsed", "snapped onto an already-used intersection"),
                )
                if dropped.get(key)
            ]
            notes.append(
                f"{missing} of your {composition['requested']} picked POIs cannot become customers"
                + (f" ({', '.join(reasons)})" if reasons else "")
                + f". The instance will hold {composition['delivered']} customers."
            )
        fallback_mode = composition.get("depot_fell_back_to_mode")
        if fallback_mode:
            notes.append(
                "The POI marked as depot could not be resolved, so the depot falls back to the "
                f"'{fallback_mode}' rule."
            )
        notes.append(relaxed_note)
        return " ".join(note for note in notes if note) or None
    shortfall = int(composition.get("poi_shortfall") or 0)
    requested = int(composition["requested"])
    delivered = int(composition["delivered"])
    # A road graph too small for the request is fatal at generation time; saying
    # so here beats letting the job fail after the user waited for it.
    too_small = (
        f"Only {delivered} of the {requested} customers requested can be placed: the road graph "
        f"of this city does not offer enough distinct vertices. Generation will fail unless the "
        f"customer count drops to {delivered}."
        if delivered < requested
        else None
    )
    if shortfall <= 0:
        return " ".join(note for note in (too_small, relaxed_note) if note) or None
    poi = int(composition["poi_customers"])
    target = int(composition["poi_target"])
    pool = int(composition["poi_pool_matching"])
    ticked = int(composition["category_count"])
    total = int(composition["category_total"])
    ceiling = composition.get("poi_pool_attachable")
    effective = composition.get("effective_method")
    became = (
        f"The instance will be recorded and named as '{effective}' rather than "
        f"'{composition['method']}', since that is what its customers are."
        if effective and effective != composition["method"]
        else None
    )
    if pool == 0:
        # Nothing to draw from at all: the POI share and the customer count are
        # both beside the point, so neither is offered as a remedy.
        return " ".join(
            note
            for note in (
                too_small,
                f"None of the {ticked} selected "
                + ("category has" if ticked == 1 else "categories have")
                + f" a POI in this extract, so all {delivered} customers are parametric road "
                "points.",
                f"Select other POI categories ({ticked} of {total} selected), or re-fetch the "
                "city with its amenities.",
                became,
                relaxed_note,
            )
            if note
        )
    parts = [
        *([too_small] if too_small else []),
        f"Only {poi} of the {target} POI customers this method asked for sit on a real amenity; "
        f"the remaining {shortfall} are parametric road points. "
        f"Final instance: {delivered} customers = {poi} POI + "
        f"{int(composition['parametric_customers'])} parametric."
    ]
    ceiling_text = (
        f", of which {ceiling} attach to a distinct road intersection"
        if ceiling is not None
        else ""
    )
    as_depot = int(composition.get("poi_used_as_depot") or 0)
    depot_text = (
        f" {as_depot} matching POI(s) became the depot rather than a customer."
        if as_depot
        else ""
    )
    parts.append(
        f"The {ticked} selected "
        + (f"category holds {pool}" if ticked == 1 else f"categories hold {pool}")
        + f" POI(s) in this extract{ceiling_text}.{depot_text}"
    )
    options = []
    if ceiling:
        share = composition.get("poi_share")
        max_customers = (
            int(ceiling / share) if composition["method"] == "hybrid" and share else int(ceiling)
        )
        if max_customers < requested:
            options.append(f"reduce the customer count to {max_customers}")
    if composition["method"] == "hybrid" and composition.get("poi_share"):
        options.append("lower the POI share")
    unattached = int(composition.get("poi_pool_unattached") or 0)
    if unattached and composition.get("poi_attach_mode") == POI_ATTACH_NEAREST_NODE:
        # By far the largest source of "but that amenity is in the extract".
        options.append(
            f"switch POI attachment to 'nearest vertex' ({unattached} POI(s) here sit closest "
            "to a mid-segment road node and are discarded by the strict rule)"
        )
    options.append(f"select more POI categories ({ticked} of {total} selected)")
    joined = options[0] if len(options) == 1 else ", ".join(options[:-1]) + f" or {options[-1]}"
    parts.append(f"To keep every customer on a POI, {joined}.")
    parts.append(became)
    parts.append(relaxed_note)
    return " ".join(part for part in parts if part)


@dataclass
class Selection:
    graph: RoadGraph
    vertices: list[int]  # [depot, customers...]
    poi_lats: list[float]
    poi_lons: list[float]
    source_tags: list[str]  # ["depot", ...]
    params: dict[str, Any]
    # Per-vertex originating POI (None for the depot and parametric customers),
    # parallel to ``vertices``. Carries amenity value, OSM id and display name.
    poi_meta: list[Poi | None] = field(default_factory=list)
    # Manual mode only: metres between the picked POI and the vertex it snapped
    # to, parallel to ``vertices`` (0.0 elsewhere).
    snap_distances_m: list[float] = field(default_factory=list)
    # Per vertex, the *other* POIs that snap to the same point and so did not
    # become customers of their own, parallel to ``vertices``. A customer with a
    # non-empty list stands for several amenities, which the map and the
    # metadata say rather than leaving one arbitrary winner's name on it.
    poi_merged: list[list[Poi]] = field(default_factory=list)


def build_generation_selection(request: GenerationRequest) -> Selection:
    request.validate()
    rng = random.Random(request.seed)

    graph = load_road_graph(
        request.osm_path,
        only_intersections=request.only_intersections,
        trim_to_connected=request.trim_to_connected_graph,
    )
    vertex_ll = vertex_latlon(graph)

    n_seeds = request.cluster_seeds if request.cluster_seeds is not None else rng.randint(2, 6)
    categories = request.categories or list(DEFAULT_CATEGORIES)
    poi_share = min(1.0, max(0.0, request.hybrid_poi_share))

    cust_meta: list[Poi | None] = []
    snap_distances: list[float] = []
    # Filled by the POI-backed selectors with how big the usable pool really was,
    # so a request the categories cannot serve is reported instead of being
    # quietly topped up with parametric points.
    poi_stats: dict[str, Any] = {}

    if request.method == "manual":
        return _build_manual_selection(request, graph, vertex_ll, categories, rng, n_seeds, poi_share)

    depot_vertex = pick_depot_vertex(request.depot_mode, vertex_ll, rng)
    depot_lat, depot_lon = vertex_ll[depot_vertex]

    if request.method == "poi_categories":
        verts, lats, lons, sources, metas = select_customers_poi(
            graph, request.osm_path, request.n_customers, categories, rng, poi_stats,
            request.poi_attach_mode, request.poi_attach_radius_m,
            exclude={depot_vertex},
        )
        cust = [(v, lats[i], lons[i], sources[i], metas[i]) for i, v in enumerate(verts)]
        # Amenities that snapped to the depot vertex. They are not customers --
        # the depot is excluded from the draw, so the selection reaches its full
        # count without them -- but they are still the reason a sparse request
        # can come up short, and the depot is standing on a real place worth
        # naming. Read from the co-located map rather than from the returned
        # list, which by construction no longer holds the depot.
        on_depot = (poi_stats.get("co_located") or {}).get(depot_vertex, [])
        poi_stats["used_as_depot"] = len(on_depot)
        poi_stats["depot_pois"] = list(on_depot)
        cust_vertices = [c[0] for c in cust]
        cust_lat = [c[1] for c in cust]
        cust_lon = [c[2] for c in cust]
        cust_src = [c[3] for c in cust]
        cust_meta = [c[4] for c in cust]
        if len(cust_vertices) < request.n_customers:
            # Parametric top-up must not duplicate POI-chosen vertices.
            chosen = set(cust_vertices) | {depot_vertex}
            attempts = 0
            while len(cust_vertices) < request.n_customers and attempts < 20:
                need = request.n_customers - len(cust_vertices)
                rem_vertices, rem_src = select_customers_parametric(
                    graph, vertex_ll, depot_vertex, need + 32,
                    request.customer_mode, n_seeds, request.cluster_decay_meters, rng,
                )
                for vtx, src in zip(rem_vertices, rem_src):
                    if vtx in chosen:
                        continue
                    chosen.add(vtx)
                    cust_vertices.append(vtx)
                    cust_lat.append(vertex_ll[vtx][0])
                    cust_lon.append(vertex_ll[vtx][1])
                    cust_src.append(src)
                    cust_meta.append(None)
                    if len(cust_vertices) >= request.n_customers:
                        break
                attempts += 1
    elif request.method == "parametric_attach":
        cust_vertices, cust_src = select_customers_parametric(
            graph, vertex_ll, depot_vertex, request.n_customers,
            request.customer_mode, n_seeds, request.cluster_decay_meters, rng,
        )
        cust_lat = [vertex_ll[v][0] for v in cust_vertices]
        cust_lon = [vertex_ll[v][1] for v in cust_vertices]
        cust_meta = [None] * len(cust_vertices)
    else:
        cust_vertices, cust_lat, cust_lon, cust_src, cust_meta = select_customers_hybrid(
            graph, request.osm_path, vertex_ll, depot_vertex, request.n_customers,
            categories, poi_share, request.customer_mode, n_seeds, request.cluster_decay_meters, rng,
            poi_stats, request.poi_attach_mode, request.poi_attach_radius_m,
        )

    n_actual = min(request.n_customers, len(cust_vertices))
    # Non-zero only under nearest_vertex attachment, and only for the POI
    # customers: how far the amenity had to move to reach a routable point.
    snapped: dict[int, float] = poi_stats.get("snap_m") or {}
    snap_distances = [0.0] + [
        float(snapped.get(vertex, 0.0)) for vertex in cust_vertices[:n_actual]
    ]
    # Every POI that snapped to a chosen point, minus the one that won it. The
    # depot also gets its list: a POI standing on it was dropped as a customer,
    # so naming what is there is just as useful.
    co_located: dict[int, list[Poi]] = poi_stats.get("co_located") or {}
    # ``depot_pois`` is already exactly the co-located list for the depot on the
    # POI path, and empty elsewhere, so take whichever is populated rather than
    # concatenating and double-counting.
    depot_merged = list(poi_stats.get("depot_pois") or co_located.get(depot_vertex, []))
    poi_merged = [depot_merged] + [
        list(co_located.get(vertex, [])) for vertex in cust_vertices[:n_actual]
    ]
    graph_options = graph_option_params(request, graph)
    method = effective_method(request.method, cust_src[:n_actual])
    params: dict[str, Any] = {
        "city": request.city,
        "osm_path": str(request.osm_path),
        # What the selection is, with what was asked for kept beside it: the
        # instance name and metadata follow the former.
        "method": method,
        "requested_method": request.method,
        "n_customers": request.n_customers,
        "seed": request.seed,
        "demand_type": request.demand_type,
        "avg_route_size": request.avg_route_size,
        "depot_mode": request.depot_mode,
        "customer_mode": request.customer_mode,
        "cluster_seeds": n_seeds,
        "cluster_decay_meters": request.cluster_decay_meters,
        "categories": categories,
        "hybrid_poi_share": poi_share,
        "poi_attach_mode": request.poi_attach_mode,
        "poi_attach_radius_m": request.poi_attach_radius_m,
        **graph_options,
        "composition": selection_composition(
            method=request.method,
            requested=request.n_customers,
            customer_tags=cust_src[:n_actual],
            poi_stats=poi_stats,
            poi_share=poi_share,
            categories=categories,
            graph_options_relaxed=graph_options["graph_options_relaxed"],
            effective=method,
            attach_mode=request.poi_attach_mode,
        ),
    }
    return Selection(
        graph=graph,
        vertices=[depot_vertex, *cust_vertices[:n_actual]],
        poi_lats=[depot_lat, *cust_lat[:n_actual]],
        poi_lons=[depot_lon, *cust_lon[:n_actual]],
        source_tags=["depot", *cust_src[:n_actual]],
        params=params,
        poi_meta=[None, *cust_meta[:n_actual]],
        snap_distances_m=snap_distances,
        poi_merged=poi_merged,
    )


def _build_manual_selection(
    request: GenerationRequest,
    graph: RoadGraph,
    vertex_ll: list[tuple[float, float]],
    categories: list[str],
    rng: random.Random,
    n_seeds: int,
    poi_share: float,
) -> Selection:
    """Selection from hand-picked POIs, snapping each to its nearest vertex.

    Unlike ``select_customers_poi``, which drops a POI whose nearest road node
    is not a graph vertex, a manual pick is never discarded silently: it snaps
    to the nearest routable vertex and the distance is recorded per node. Two
    picks that snap to the same vertex necessarily collapse into one customer,
    which is reported rather than raised.
    """
    # Resolved against the whole catalog, not the current category filter, so a
    # pick stays valid when the user unticks its category afterwards. Keyed by
    # (type, id): node 123 and way 123 are different places in OSM.
    by_ref = {
        ref: poi
        for poi in load_poi_catalog(request.osm_path)
        if (ref := poi_ref(poi.osm_type, poi.osm_id)) is not None
    }

    picked_refs = request.picked_refs()
    depot_pick = request.depot_ref()
    unknown_ids: list[int] = []
    unreachable_ids: list[int] = []
    collapsed_ids: list[int] = []

    def snap(ref: tuple[str, int]) -> tuple[int, float, Poi] | None:
        poi = by_ref.get(ref)
        if poi is None:
            unknown_ids.append(ref[1])
            return None
        hit = graph.nearest_vertex(poi.lat, poi.lon, MANUAL_SNAP_MAX_M)
        if hit is None:
            unreachable_ids.append(ref[1])
            return None
        vertex, distance = hit
        return vertex, distance, poi

    depot_meta: Poi | None = None
    depot_snap = 0.0
    depot_resolved = snap(depot_pick) if depot_pick is not None else None
    if depot_resolved is not None:
        depot_vertex, depot_snap, depot_meta = depot_resolved
        depot_lat, depot_lon = depot_meta.lat, depot_meta.lon
    else:
        depot_vertex = pick_depot_vertex(request.depot_mode, vertex_ll, rng)
        depot_lat, depot_lon = vertex_ll[depot_vertex]
    # A pick that could not be resolved must not be reported as the depot: the
    # instance would claim a hand-placed depot it does not have.
    depot_fell_back = depot_pick is not None and depot_resolved is None

    taken = {depot_vertex}
    cust_vertices: list[int] = []
    cust_lat: list[float] = []
    cust_lon: list[float] = []
    cust_src: list[str] = []
    cust_meta: list[Poi | None] = []
    cust_snap: list[float] = []
    # Which picks collapsed onto which point: the count alone leaves the user
    # guessing which of their picks the surviving customer now stands for.
    co_located: dict[int, list[Poi]] = {}
    for ref in picked_refs:
        if ref == depot_pick:
            continue
        resolved = snap(ref)
        if resolved is None:
            continue
        vertex, distance, poi = resolved
        if vertex in taken:
            collapsed_ids.append(ref[1])
            co_located.setdefault(vertex, []).append(poi)
            continue
        taken.add(vertex)
        cust_vertices.append(vertex)
        cust_lat.append(poi.lat)
        cust_lon.append(poi.lon)
        cust_src.append("poi_manual")
        cust_meta.append(poi)
        cust_snap.append(distance)

    if len(cust_vertices) < 2:
        raise ValueError(
            "Manual selection resolved fewer than 2 customers "
            f"({len(unknown_ids)} unknown, {len(unreachable_ids)} unreachable, "
            f"{len(collapsed_ids)} collapsed onto an already-used vertex)"
        )

    graph_options = graph_option_params(request, graph)
    params: dict[str, Any] = {
        "city": request.city,
        "osm_path": str(request.osm_path),
        "method": request.method,
        "requested_method": request.method,
        # The picks are the instance size; a requested n is meaningless here.
        "n_customers": len(cust_vertices),
        "seed": request.seed,
        "demand_type": request.demand_type,
        "avg_route_size": request.avg_route_size,
        "depot_mode": "manual" if depot_resolved is not None else request.depot_mode,
        "customer_mode": request.customer_mode,
        "cluster_seeds": n_seeds,
        "cluster_decay_meters": request.cluster_decay_meters,
        "categories": categories,
        "hybrid_poi_share": poi_share,
        # Manual picks have always snapped to the nearest routable vertex, which
        # is what the nearest_vertex mode does for the sampled methods.
        "poi_attach_mode": POI_ATTACH_NEAREST_VERTEX,
        "poi_attach_radius_m": MANUAL_SNAP_MAX_M,
        **graph_options,
        "manual_selection": {
            "requested": len(set(picked_refs) - {depot_pick}),
            "resolved": len(cust_vertices),
            # Only set when a picked POI really became the depot. The id keeps
            # its historical shape; the type beside it is what disambiguates it.
            "depot_poi_osm_id": depot_pick[1] if depot_resolved is not None else None,
            "depot_poi_osm_type": depot_pick[0] if depot_resolved is not None else None,
            "requested_depot_poi_id": depot_pick[1] if depot_pick is not None else None,
            "requested_depot_poi_type": depot_pick[0] if depot_pick is not None else None,
            "depot_fell_back_to_mode": request.depot_mode if depot_fell_back else None,
            "unknown_poi_ids": unknown_ids,
            "unreachable_poi_ids": unreachable_ids,
            "collapsed_poi_ids": collapsed_ids,
        },
        # Same shape as the sampled methods, so one caller can report both.
        "composition": {
            "method": "manual",
            "effective_method": "manual",
            "requested": len(set(picked_refs) - {depot_pick}),
            "delivered": len(cust_vertices),
            "poi_customers": len(cust_vertices),
            "parametric_customers": 0,
            "poi_target": len(cust_vertices),
            "poi_shortfall": 0,
            "dropped": {
                "unknown": len(unknown_ids),
                "unreachable": len(unreachable_ids),
                "collapsed": len(collapsed_ids),
            },
            "depot_fell_back_to_mode": request.depot_mode if depot_fell_back else None,
            "graph_options_relaxed": graph_options["graph_options_relaxed"],
        },
    }
    return Selection(
        graph=graph,
        vertices=[depot_vertex, *cust_vertices],
        poi_lats=[depot_lat, *cust_lat],
        poi_lons=[depot_lon, *cust_lon],
        source_tags=["depot", *cust_src],
        params=params,
        poi_meta=[depot_meta, *cust_meta],
        snap_distances_m=[depot_snap, *cust_snap],
        poi_merged=[
            list(co_located.get(vertex, []))
            for vertex in (depot_vertex, *cust_vertices)
        ],
    )


def poi_attributes(
    selection: Selection, index: int
) -> tuple[str | None, int | None, str | None, str | None]:
    """``(category, osm_id, name, osm_type)`` of the POI behind a selected vertex."""
    poi = selection.poi_meta[index] if index < len(selection.poi_meta) else None
    if poi is None:
        return None, None, None, None
    return poi.category, poi.osm_id, poi.name, poi.osm_type


def merged_poi_records(selection: Selection, index: int) -> list[dict[str, Any]]:
    """The other amenities standing on this node, as plain serialisable rows."""
    merged = selection.poi_merged[index] if index < len(selection.poi_merged) else []
    return [
        {
            "name": poi.name,
            "category": poi.category,
            "osm_id": poi.osm_id,
            "osm_type": poi.osm_type,
        }
        for poi in merged
    ]


def poi_attribute_columns(
    selection: Selection, count: int
) -> tuple[
    list[str | None],
    list[int | None],
    list[str | None],
    list[str | None],
    list[list[dict[str, Any]]],
    list[float],
]:
    """The six per-node POI columns in one pass over ``selection``."""
    categories: list[str | None] = []
    osm_ids: list[int | None] = []
    names: list[str | None] = []
    osm_types: list[str | None] = []
    merged: list[list[dict[str, Any]]] = []
    snaps: list[float] = []
    for index in range(count):
        category, osm_id, name, osm_type = poi_attributes(selection, index)
        categories.append(category)
        osm_ids.append(osm_id)
        names.append(name)
        osm_types.append(osm_type)
        merged.append(merged_poi_records(selection, index))
        snaps.append(
            selection.snap_distances_m[index]
            if index < len(selection.snap_distances_m)
            else 0.0
        )
    return categories, osm_ids, names, osm_types, merged, snaps


def _claim_unique_name(folder: Path, base: str, reserved: set[str]) -> str:
    """``base``, or the first ``base-2``, ``base-3``... free in this run."""
    candidate = base
    counter = 2
    while str(folder / candidate) in reserved:
        candidate = f"{base}-{counter}"
        counter += 1
    reserved.add(str(folder / candidate))
    return candidate


def preview_geojson(selection: Selection) -> dict[str, Any]:
    graph = selection.graph
    features = []
    for i, vertex in enumerate(selection.vertices):
        lon, lat = graph.node_lonlat(graph.node_of[vertex])
        category, osm_id, name, osm_type = poi_attributes(selection, i)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "instance_node_id": i + 1,
                    "graph_vertex_id": vertex,
                    "role": "depot" if i == 0 else "customer",
                    "source_tag": selection.source_tags[i],
                    "poi_category": category,
                    "poi_osm_id": osm_id,
                    "poi_osm_type": osm_type,
                    "poi_name": name,
                    "poi_merged": merged_poi_records(selection, i),
                    "snap_distance_m": (
                        selection.snap_distances_m[i]
                        if i < len(selection.snap_distances_m)
                        else 0.0
                    ),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def generate_single_instance(
    request: GenerationRequest,
    output_root: str | Path,
    *,
    name_suffix: str = "",
    reserved_names: set[str] | None = None,
) -> dict[str, Any]:
    selection = build_generation_selection(request)
    n_got = len(selection.vertices) - 1
    # Manual mode is sized by the picks themselves, so a requested n never applies.
    if request.method != "manual" and n_got < request.n_customers:
        raise ValueError(
            f"Generation method produced only {n_got} customers out of requested {request.n_customers}"
        )
    return materialize_instance(
        selection,
        output_root,
        demand_type=request.demand_type,
        avg_route_size=request.avg_route_size,
        rng=random.Random(request.seed),
        name_suffix=name_suffix,
        reserved_names=reserved_names,
    )


def materialize_instance(
    selection: Selection,
    output_root: str | Path,
    *,
    demand_type: int,
    avg_route_size: int,
    rng: random.Random,
    precomputed: dict[str, Any] | None = None,
    name_suffix: str = "",
    reserved_names: set[str] | None = None,
) -> dict[str, Any]:
    """Write the 3-metric .vrp set, _meta.json, _manifest.json and .vrp.json
    files for a selection; ``precomputed`` lets the bulk driver pass sliced
    matrices/geometry instead of recomputing them, and ``name_suffix`` keeps
    same-parameter-different-seed instances from overwriting one another.

    ``reserved_names`` is the set of instance paths a caller has already written
    in this run. The final name is only known here -- it encodes ``route_count``,
    which follows from the drawn demands rather than from any request field -- so
    this is the only place that can guarantee two instances of one run never land
    on the same files. Names claimed by *earlier* runs are untouched, keeping
    regeneration of the same configuration idempotent."""
    graph = selection.graph
    params = selection.params
    vertices = selection.vertices
    ref = graph.ref_lla

    customer_ll = list(zip(selection.poi_lats[1:], selection.poi_lons[1:]))
    demand_values, sum_demands, _max_demand, r = generate_demands(rng, customer_ll, demand_type, avg_route_size)
    demands = [0, *demand_values]
    capacity = capacity_from_avg_route_size(r, demand_values)
    # The exact bin-packing minimum, the way CVRPLIB derives the ``k`` in
    # ``X-n101-k25`` -- not ``ceil(sum / capacity)``, which is only the
    # continuous relaxation of it and can understate the fleet by a wide margin
    # when the demands do not divide the capacity. It is a *lower bound* on a
    # solution's route count, not a cap: ``num_vehicles`` stays unset.
    fleet = minimum_bins(demand_values, capacity)
    route_count = fleet.lower

    if precomputed is None:
        d_short, d_fast, geom_short, geom_fast = compute_matrices(graph, vertices)
        d_eucl, coords = euclidean_matrix_from_vertices(graph, vertices)
    else:
        d_short = precomputed["d_short"]
        d_fast = precomputed["d_fast"]
        geom_short = precomputed["geom_short"]
        geom_fast = precomputed["geom_fast"]
        d_eucl = precomputed["d_eucl"]
        coords = precomputed["coords"]

    n_customers = len(vertices) - 1
    folder, base = instance_path_plan(
        str(params["city"]),
        str(params["method"]),
        n_customers,
        route_count,
        output_root,
        name_suffix,
    )
    if reserved_names is not None:
        base = _claim_unique_name(folder, base, reserved_names)
    folder.mkdir(parents=True, exist_ok=True)

    files = {
        "shortest": f"{base}_shortest.vrp",
        "fastest": f"{base}_fastest.vrp",
        "euclidean": f"{base}_euclidean.vrp",
        "meta": f"{base}_meta.json",
    }
    manifest_name = f"{base}_manifest.json"
    ref_str = f"LLA({ref.lat}, {ref.lon}, {ref.alt})"
    write_cvrplib(folder / files["shortest"], f"{base}_shortest", f"Shortest distances; ENU ref: {ref_str}", coords, demands, d_short, capacity)
    write_cvrplib(folder / files["fastest"], f"{base}_fastest", f"Fastest distances; ENU ref: {ref_str}", coords, demands, d_fast, capacity)
    write_cvrplib(folder / files["euclidean"], f"{base}_euclidean", f"Euclidean distances; ENU ref: {ref_str}", coords, demands, d_eucl, capacity)

    generation_params = dict(params)
    generation_params["demand_type"] = demand_type
    generation_params["avg_route_size"] = avg_route_size
    (
        poi_categories,
        poi_osm_ids,
        poi_names,
        poi_osm_types,
        poi_merged,
        poi_snaps,
    ) = poi_attribute_columns(selection, len(vertices))
    write_instance_metadata(
        folder / files["meta"],
        city=str(params["city"]),
        osm_file=str(params["osm_path"]),
        instance_name=base,
        metric_files=[files["shortest"], files["fastest"], files["euclidean"]],
        ref_lla=ref,
        vertices=vertices,
        poi_lats=selection.poi_lats,
        poi_lons=selection.poi_lons,
        coords=coords,
        demands=demands,
        method=str(params["method"]),
        source_tags=selection.source_tags,
        only_intersections=bool(params["only_intersections"]),
        trim_to_connected_graph=bool(params["trim_to_connected_graph"]),
        generation_params=generation_params,
        poi_categories=poi_categories,
        poi_osm_ids=poi_osm_ids,
        poi_names=poi_names,
        poi_osm_types=poi_osm_types,
        poi_merged=poi_merged,
        snap_distances_m=poi_snaps,
        road_cache={"shortest": geom_short, "fastest": geom_fast},
    )

    generated_at = datetime.now().isoformat()
    place_slug = slugify(str(params["city"]))
    vrp_json_files: dict[str, str] = {}
    for metric in ("shortest", "fastest", "euclidean"):
        parsed = parse_cvrp_vrp(folder / files[metric])
        json_name = files[metric].replace(".vrp", ".vrp.json")
        payload = build_vrp_json_payload(
            parsed,
            instance_name=f"{base}_{metric}",
            metric_variant=metric,
            place_slug=place_slug,
            source_base_name=base,
            source_city=str(params["city"]),
            source_seed=int(params["seed"]),
            source_folder=str(folder),
            num_vehicles_lb=route_count,
            artifact_paths={m: files[m] for m in files},
            sibling_variant_paths={
                m: files[m].replace(".vrp", ".vrp.json") for m in ("shortest", "fastest", "euclidean")
            },
            reference_lla={"lat": ref.lat, "lon": ref.lon, "alt": ref.alt},
            generated_at=generated_at,
        )
        write_json(folder / json_name, payload)
        vrp_json_files[metric] = json_name

    manifest = {
        "generated_at": generated_at,
        "base_name": base,
        "folder": str(folder),
        "files": {**files, "vrp_json": vrp_json_files},
        "params": generation_params,
        "generator": "mamut-routing-tools",
        "demand_type": demand_type,
        "avg_route_size": avg_route_size,
        "route_count": route_count,
        "route_count_proven": fleet.proven,
        "route_count_method": fleet.method,
        "capacity": capacity,
        "total_demand": sum_demands,
    }
    write_json(folder / manifest_name, manifest)

    poi_count = sum(1 for tag in selection.source_tags[1:] if is_poi_source_tag(tag))
    composition = params.get("composition")
    return {
        "ok": True,
        "base_name": base,
        "folder": str(folder),
        "files": manifest["files"],
        "manifest": manifest_name,
        # Not None only when the selection could not be served as asked; the
        # caller shows it so a parametric top-up never passes unnoticed.
        "notice": composition_notice(composition),
        "summary": {
            "customers": n_customers,
            "composition": composition,
            "capacity": capacity,
            "total_demand": sum_demands,
            "method": str(params["method"]),
            "demand_type": demand_type,
            "avg_route_size": avg_route_size,
            "route_count": route_count,
            "route_count_proven": fleet.proven,
            "route_count_method": fleet.method,
            "poi_customers": poi_count,
            "parametric_customers": n_customers - poi_count,
        },
    }
