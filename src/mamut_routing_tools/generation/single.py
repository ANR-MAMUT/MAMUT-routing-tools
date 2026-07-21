"""Single-instance generation driver (selection, demands, matrices, artifacts),
the port of the workbench's build_generation_selection + generate_single_instance."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from mamut_routing_tools.generation.demands import capacity_from_avg_route_size, generate_demands
from mamut_routing_tools.generation.matrices import compute_matrices, euclidean_matrix_from_vertices
from mamut_routing_tools.generation.pois import DEFAULT_CATEGORIES
from mamut_routing_tools.generation.select import (
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

METHODS = ("poi_categories", "parametric_attach", "hybrid")
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

    def validate(self) -> None:
        if not self.city:
            raise ValueError("Missing 'city'")
        if not Path(self.osm_path).is_file():
            raise FileNotFoundError(f"OSM file not found: {self.osm_path}")
        if self.method not in METHODS:
            raise ValueError(f"Unsupported method '{self.method}'")
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


@dataclass
class Selection:
    graph: RoadGraph
    vertices: list[int]  # [depot, customers...]
    poi_lats: list[float]
    poi_lons: list[float]
    source_tags: list[str]  # ["depot", ...]
    params: dict[str, Any]


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

    depot_vertex = pick_depot_vertex(request.depot_mode, vertex_ll, rng)
    depot_lat, depot_lon = vertex_ll[depot_vertex]

    if request.method == "poi_categories":
        verts, lats, lons, sources = select_customers_poi(
            graph, request.osm_path, request.n_customers, categories, rng
        )
        cust = [(v, lats[i], lons[i], sources[i]) for i, v in enumerate(verts) if v != depot_vertex]
        cust_vertices = [c[0] for c in cust]
        cust_lat = [c[1] for c in cust]
        cust_lon = [c[2] for c in cust]
        cust_src = [c[3] for c in cust]
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
    else:
        cust_vertices, cust_lat, cust_lon, cust_src = select_customers_hybrid(
            graph, request.osm_path, vertex_ll, depot_vertex, request.n_customers,
            categories, poi_share, request.customer_mode, n_seeds, request.cluster_decay_meters, rng,
        )

    n_actual = min(request.n_customers, len(cust_vertices))
    params: dict[str, Any] = {
        "city": request.city,
        "osm_path": str(request.osm_path),
        "method": request.method,
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
        "only_intersections": request.only_intersections,
        "trim_to_connected_graph": request.trim_to_connected_graph,
    }
    return Selection(
        graph=graph,
        vertices=[depot_vertex, *cust_vertices[:n_actual]],
        poi_lats=[depot_lat, *cust_lat[:n_actual]],
        poi_lons=[depot_lon, *cust_lon[:n_actual]],
        source_tags=["depot", *cust_src[:n_actual]],
        params=params,
    )


def preview_geojson(selection: Selection) -> dict[str, Any]:
    graph = selection.graph
    features = []
    for i, vertex in enumerate(selection.vertices):
        lon, lat = graph.node_lonlat(graph.node_of[vertex])
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "instance_node_id": i + 1,
                    "graph_vertex_id": vertex,
                    "role": "depot" if i == 0 else "customer",
                    "source_tag": selection.source_tags[i],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def generate_single_instance(request: GenerationRequest, output_root: str | Path) -> dict[str, Any]:
    selection = build_generation_selection(request)
    n_got = len(selection.vertices) - 1
    if n_got < request.n_customers:
        raise ValueError(
            f"Generation method produced only {n_got} customers out of requested {request.n_customers}"
        )
    return materialize_instance(
        selection,
        output_root,
        demand_type=request.demand_type,
        avg_route_size=request.avg_route_size,
        rng=random.Random(request.seed),
    )


def materialize_instance(
    selection: Selection,
    output_root: str | Path,
    *,
    demand_type: int,
    avg_route_size: int,
    rng: random.Random,
    precomputed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the 3-metric .vrp set, _meta.json, _manifest.json and .vrp.json
    files for a selection; ``precomputed`` lets the bulk driver pass sliced
    matrices/geometry instead of recomputing them."""
    graph = selection.graph
    params = selection.params
    vertices = selection.vertices
    ref = graph.ref_lla

    customer_ll = list(zip(selection.poi_lats[1:], selection.poi_lons[1:]))
    demand_values, sum_demands, _max_demand, r = generate_demands(rng, customer_ll, demand_type, avg_route_size)
    demands = [0, *demand_values]
    capacity = capacity_from_avg_route_size(r, demand_values)
    route_count = -(-sum_demands // capacity)

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
    folder, base = instance_path_plan(str(params["city"]), str(params["method"]), n_customers, route_count, output_root)
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
        "capacity": capacity,
        "total_demand": sum_demands,
    }
    write_json(folder / manifest_name, manifest)

    poi_count = sum(1 for tag in selection.source_tags[1:] if tag == "poi")
    return {
        "ok": True,
        "base_name": base,
        "folder": str(folder),
        "files": manifest["files"],
        "manifest": manifest_name,
        "summary": {
            "customers": n_customers,
            "capacity": capacity,
            "total_demand": sum_demands,
            "method": str(params["method"]),
            "demand_type": demand_type,
            "avg_route_size": avg_route_size,
            "route_count": route_count,
            "poi_customers": poi_count,
            "parametric_customers": n_customers - poi_count,
        },
    }
