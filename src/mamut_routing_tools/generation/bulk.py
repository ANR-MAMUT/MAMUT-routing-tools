"""Bulk generation driver: one POI pool + one full matrix computation per
city, sliced per (n, demand_type, avg_route_size) combination, exactly like
the workbench's cartesian bulk mode."""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

from mamut_routing_tools.generation.matrices import compute_matrices, euclidean_matrix_from_vertices
from mamut_routing_tools.generation.single import (
    GenerationRequest,
    Selection,
    build_generation_selection,
    materialize_instance,
)
from mamut_routing_tools.generation.vrptw import stable_seed


def generate_bulk_instances(
    base_request: GenerationRequest,
    *,
    cities: list[tuple[str, Path]],
    n_list: list[int],
    demand_types: list[int],
    avg_route_sizes: list[int],
    output_root: str | Path,
) -> dict[str, Any]:
    """Cartesian bulk generation over cities x sizes x demand types x route sizes."""
    if not cities:
        raise ValueError("Bulk generation requires at least one city")
    base_seed = base_request.seed
    results: list[dict[str, Any]] = []
    city_reports: list[dict[str, Any]] = []

    for city, osm_path in cities:
        max_nc = max(n_list)
        pool_size = math.ceil(max_nc * 1.5)
        pool_request = replace(
            base_request,
            city=city,
            osm_path=Path(osm_path),
            n_customers=pool_size,
            seed=stable_seed(city, max_nc, base_seed),
        )
        try:
            pool = build_generation_selection(pool_request)
        except (ValueError, FileNotFoundError) as error:
            warnings.warn(f"City {city}: selection failed, skipping ({error})", stacklevel=2)
            city_reports.append({"city": city, "status": "skipped", "error": str(error)})
            continue

        actual_max_nc = len(pool.vertices) - 1
        valid_n_list = [nc for nc in n_list if nc <= actual_max_nc]
        skipped_sizes = [nc for nc in n_list if nc > actual_max_nc]
        pool_poi = sum(1 for tag in pool.source_tags[1:] if tag == "poi")
        city_reports.append(
            {
                "city": city,
                "poi_available": pool_poi,
                "parametric_filled": actual_max_nc - pool_poi,
                "pool_total": actual_max_nc,
                "requested_sizes": n_list,
                "valid_sizes": valid_n_list,
                "skipped_sizes": skipped_sizes,
                "status": "skipped" if not valid_n_list else ("partial" if skipped_sizes else "ok"),
            }
        )
        if not valid_n_list:
            continue

        d_short_full, d_fast_full, geom_short_full, geom_fast_full = compute_matrices(pool.graph, pool.vertices)
        d_eucl_full, coords_full = euclidean_matrix_from_vertices(pool.graph, pool.vertices)
        total = len(pool.vertices)

        for nc in valid_n_list:
            for demand_type in demand_types:
                for avg_route_size in avg_route_sizes:
                    inst_seed = stable_seed(city, nc, demand_type, avg_route_size, base_seed)
                    rng = random.Random(inst_seed)
                    if nc < actual_max_nc:
                        perm = list(range(1, total))
                        rng.shuffle(perm)
                        sel_indices = [0, *sorted(perm[:nc])]
                    else:
                        sel_indices = list(range(total))

                    vertices = [pool.vertices[i] for i in sel_indices]
                    subset = set(vertices)
                    params = dict(pool.params)
                    params["seed"] = inst_seed
                    params["n_customers"] = nc
                    selection = Selection(
                        graph=pool.graph,
                        vertices=vertices,
                        poi_lats=[pool.poi_lats[i] for i in sel_indices],
                        poi_lons=[pool.poi_lons[i] for i in sel_indices],
                        source_tags=[pool.source_tags[i] for i in sel_indices],
                        params=params,
                    )
                    precomputed = {
                        "d_short": [[d_short_full[i][j] for j in sel_indices] for i in sel_indices],
                        "d_fast": [[d_fast_full[i][j] for j in sel_indices] for i in sel_indices],
                        "d_eucl": [[d_eucl_full[i][j] for j in sel_indices] for i in sel_indices],
                        "coords": [coords_full[i] for i in sel_indices],
                        "geom_short": {
                            key: value
                            for key, value in geom_short_full.items()
                            if all(int(part) in subset for part in key.split("_"))
                        },
                        "geom_fast": {
                            key: value
                            for key, value in geom_fast_full.items()
                            if all(int(part) in subset for part in key.split("_"))
                        },
                    }
                    result = materialize_instance(
                        selection,
                        output_root,
                        demand_type=demand_type,
                        avg_route_size=avg_route_size,
                        rng=rng,
                        precomputed=precomputed,
                    )
                    results.append(result)

    return {"ok": True, "generated": len(results), "results": results, "city_reports": city_reports}
