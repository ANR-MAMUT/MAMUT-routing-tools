"""Bulk generation driver.

Bulk work is expressed as an explicit list of rows, each a fully specified
``GenerationRequest`` plus its problem type. Rows that agree on every parameter
affecting *customer selection* are grouped, so a group still costs one POI pool
and one full matrix computation and is then sliced per row -- the optimization
the cartesian mode has always had, now available to arbitrary row lists.

``generate_bulk_instances`` remains as the cartesian front door and simply
expands its product into rows.
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from mamut_routing_tools.generation.matrices import compute_matrices, euclidean_matrix_from_vertices
from mamut_routing_tools.generation.pois import is_poi_source_tag
from mamut_routing_tools.generation.single import (
    GenerationRequest,
    Selection,
    build_generation_selection,
    generate_single_instance,
    materialize_instance,
)
from mamut_routing_tools.generation.vrptw import derive_vrptw_from_cvrp, stable_seed
from mamut_routing_tools.generation.writers import slugify

PROBLEM_TYPES = ("cvrp", "vrptw")


class ProgressContext(Protocol):
    """The subset of the GUI's JobContext this module uses."""

    def progress(self, message: str, *, current: int | None = ..., total: int | None = ...) -> None: ...

    def check_cancelled(self) -> None: ...


@dataclass
class BulkRow:
    """One instance to generate."""

    request: GenerationRequest
    problem_type: str = "cvrp"
    tw_method: str = "route_centered"
    # None means "derive the seed from (city, n, demand type, route band)", which
    # is what the cartesian mode has always done. An int pins this row's seed.
    explicit_seed: int | None = None

    def validate(self) -> None:
        if self.problem_type not in PROBLEM_TYPES:
            raise ValueError(f"Unsupported problem type '{self.problem_type}'")


def _pool_key(request: GenerationRequest) -> tuple[Any, ...]:
    """Everything that changes which customers a pool can offer.

    Demand type, route band and n are deliberately absent: they slice or label a
    pool rather than change it, which is exactly what makes grouping pay off.
    """
    return (
        request.city,
        str(request.osm_path),
        request.method,
        tuple(request.categories or ()),
        request.depot_mode,
        request.customer_mode,
        request.cluster_seeds,
        request.cluster_decay_meters,
        request.hybrid_poi_share,
        request.only_intersections,
        request.trim_to_connected_graph,
    )


def _name_key(request: GenerationRequest) -> tuple[Any, ...]:
    """Everything the generated file name encodes, seed excluded."""
    return (
        request.city,
        request.method,
        request.n_customers,
        request.demand_type,
        request.avg_route_size,
    )


@dataclass
class _Group:
    key: tuple[Any, ...]
    city: str
    osm_path: Path
    rows: list[tuple[int, BulkRow]] = field(default_factory=list)


def _derive_twin(result: dict[str, Any], row: BulkRow, seed: int) -> None:
    """Attach the VRPTW twin in place when the row asks for one."""
    result["vrptw"] = derive_vrptw_from_cvrp(
        result["folder"],
        result["base_name"],
        tw_method=row.tw_method,
        place_slug=slugify(row.request.city),
        source_seed=seed,
    )


def generate_bulk_instances(
    base_request: GenerationRequest,
    *,
    cities: list[tuple[str, Path]],
    n_list: list[int],
    demand_types: list[int],
    avg_route_sizes: list[int],
    output_root: str | Path,
    problem_type: str = "cvrp",
    tw_method: str = "route_centered",
    context: ProgressContext | None = None,
) -> dict[str, Any]:
    """Cartesian bulk generation over cities x sizes x demand types x route sizes."""
    if not cities:
        raise ValueError("Bulk generation requires at least one city")
    rows = [
        BulkRow(
            request=replace(
                base_request,
                city=city,
                osm_path=Path(osm_path),
                n_customers=nc,
                demand_type=demand_type,
                avg_route_size=avg_route_size,
            ),
            problem_type=problem_type,
            tw_method=tw_method,
        )
        for city, osm_path in cities
        for nc in n_list
        for demand_type in demand_types
        for avg_route_size in avg_route_sizes
    ]
    return generate_bulk_from_rows(
        rows,
        output_root=output_root,
        base_seed=base_request.seed,
        context=context,
    )


def generate_bulk_from_rows(
    rows: list[BulkRow],
    *,
    output_root: str | Path,
    base_seed: int = 0,
    context: ProgressContext | None = None,
) -> dict[str, Any]:
    """Generate an explicit list of instances, reusing a pool per selection group."""
    if not rows:
        raise ValueError("Bulk generation requires at least one instance row")
    for row in rows:
        row.validate()

    groups: dict[tuple[Any, ...], _Group] = {}
    for index, row in enumerate(rows):
        key = _pool_key(row.request)
        group = groups.get(key)
        if group is None:
            group = _Group(key=key, city=row.request.city, osm_path=Path(row.request.osm_path))
            groups[key] = group
        group.rows.append((index, row))

    # An instance is named <city>_<abbr>-n<N>-k<K>, which does not mention the
    # seed: rows agreeing on everything but the seed would overwrite each other
    # on disk. Those rows -- and only those -- get a seed suffix.
    name_counts: dict[tuple[Any, ...], int] = {}
    for row in rows:
        name_counts[_name_key(row.request)] = name_counts.get(_name_key(row.request), 0) + 1
    ambiguous = {key for key, count in name_counts.items() if count > 1}

    results: list[dict[str, Any]] = []
    city_reports: list[dict[str, Any]] = []
    done = 0
    total = len(rows)

    def advance(row: BulkRow) -> None:
        nonlocal done
        done += 1
        if context is not None:
            context.progress(
                f"Generated {done}/{total}: {row.request.city} n={row.request.n_customers} "
                f"(demand {row.request.demand_type}, band {row.request.avg_route_size})",
                current=done,
                total=total,
            )

    for group in groups.values():
        if context is not None:
            context.check_cancelled()

        # Hand-picked selections are per-row by construction; nothing to pool.
        if group.rows[0][1].request.method == "manual":
            for _, row in group.rows:
                if context is not None:
                    context.check_cancelled()
                seed = row.explicit_seed if row.explicit_seed is not None else row.request.seed
                result = generate_single_instance(
                    replace(row.request, seed=seed),
                    output_root,
                    name_suffix=f"-s{seed}" if _name_key(row.request) in ambiguous else "",
                )
                if row.problem_type == "vrptw":
                    _derive_twin(result, row, seed)
                results.append(result)
                advance(row)
            continue

        max_nc = max(row.request.n_customers for _, row in group.rows)
        pool_seed = stable_seed(group.city, max_nc, base_seed)
        pool_request = replace(
            group.rows[0][1].request,
            n_customers=math.ceil(max_nc * 1.5),
            seed=pool_seed,
        )
        if context is not None:
            context.progress(
                f"Building the {group.city} customer pool for {len(group.rows)} instance(s)",
                current=done,
                total=total,
            )
        try:
            pool = build_generation_selection(pool_request)
        except (ValueError, FileNotFoundError) as error:
            warnings.warn(f"City {group.city}: selection failed, skipping ({error})", stacklevel=2)
            city_reports.append({"city": group.city, "status": "skipped", "error": str(error)})
            done += len(group.rows)
            continue

        actual_max_nc = len(pool.vertices) - 1
        requested_sizes = sorted({row.request.n_customers for _, row in group.rows})
        valid_rows = [(i, row) for i, row in group.rows if row.request.n_customers <= actual_max_nc]
        skipped_sizes = sorted({n for n in requested_sizes if n > actual_max_nc})
        pool_poi = sum(1 for tag in pool.source_tags[1:] if is_poi_source_tag(tag))
        city_reports.append(
            {
                "city": group.city,
                "method": pool_request.method,
                "poi_available": pool_poi,
                "parametric_filled": actual_max_nc - pool_poi,
                "pool_total": actual_max_nc,
                "requested_sizes": requested_sizes,
                "valid_sizes": sorted({row.request.n_customers for _, row in valid_rows}),
                "skipped_sizes": skipped_sizes,
                "status": "skipped" if not valid_rows else ("partial" if skipped_sizes else "ok"),
            }
        )
        done += len(group.rows) - len(valid_rows)
        if not valid_rows:
            continue

        d_short_full, d_fast_full, geom_short_full, geom_fast_full = compute_matrices(
            pool.graph, pool.vertices
        )
        d_eucl_full, coords_full = euclidean_matrix_from_vertices(pool.graph, pool.vertices)
        total_vertices = len(pool.vertices)

        for _, row in valid_rows:
            if context is not None:
                context.check_cancelled()
            nc = row.request.n_customers
            inst_seed = (
                row.explicit_seed
                if row.explicit_seed is not None
                else stable_seed(
                    group.city, nc, row.request.demand_type, row.request.avg_route_size, base_seed
                )
            )
            rng = random.Random(inst_seed)
            if nc < actual_max_nc:
                perm = list(range(1, total_vertices))
                rng.shuffle(perm)
                sel_indices = [0, *sorted(perm[:nc])]
            else:
                sel_indices = list(range(total_vertices))

            vertices = [pool.vertices[i] for i in sel_indices]
            subset = set(vertices)
            params = dict(pool.params)
            params["seed"] = inst_seed
            params["n_customers"] = nc
            params["demand_type"] = row.request.demand_type
            params["avg_route_size"] = row.request.avg_route_size
            selection = Selection(
                graph=pool.graph,
                vertices=vertices,
                poi_lats=[pool.poi_lats[i] for i in sel_indices],
                poi_lons=[pool.poi_lons[i] for i in sel_indices],
                source_tags=[pool.source_tags[i] for i in sel_indices],
                params=params,
                poi_meta=[
                    pool.poi_meta[i] if i < len(pool.poi_meta) else None for i in sel_indices
                ],
                snap_distances_m=[
                    pool.snap_distances_m[i] if i < len(pool.snap_distances_m) else 0.0
                    for i in sel_indices
                ],
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
                demand_type=row.request.demand_type,
                avg_route_size=row.request.avg_route_size,
                rng=rng,
                precomputed=precomputed,
                name_suffix=f"-s{inst_seed}" if _name_key(row.request) in ambiguous else "",
            )
            if row.problem_type == "vrptw":
                _derive_twin(result, row, inst_seed)
            results.append(result)
            advance(row)

    return {"ok": True, "generated": len(results), "results": results, "city_reports": city_reports}


def preflight_rows(rows: list[BulkRow], *, base_seed: int = 0) -> dict[str, Any]:
    """Report per selection group how many customers its pool can actually offer.

    Loads each group's road graph and POI pool but generates nothing, so the UI
    can warn about sizes that would be skipped before committing to a long run.
    """
    groups: dict[tuple[Any, ...], _Group] = {}
    for index, row in enumerate(rows):
        key = _pool_key(row.request)
        group = groups.setdefault(
            key, _Group(key=key, city=row.request.city, osm_path=Path(row.request.osm_path))
        )
        group.rows.append((index, row))

    reports: list[dict[str, Any]] = []
    for group in groups.values():
        requested_sizes = sorted({row.request.n_customers for _, row in group.rows})
        if group.rows[0][1].request.method == "manual":
            reports.append(
                {
                    "city": group.city,
                    "method": "manual",
                    "status": "ok",
                    "requested_sizes": requested_sizes,
                    "skipped_sizes": [],
                }
            )
            continue
        max_nc = max(requested_sizes)
        pool_request = replace(
            group.rows[0][1].request,
            n_customers=math.ceil(max_nc * 1.5),
            # Same pool seed generation will use, so the estimate matches the run.
            seed=stable_seed(group.city, max_nc, base_seed),
        )
        try:
            pool = build_generation_selection(pool_request)
        except (ValueError, FileNotFoundError) as error:
            reports.append(
                {
                    "city": group.city,
                    "method": pool_request.method,
                    "status": "skipped",
                    "error": str(error),
                    "requested_sizes": requested_sizes,
                    "skipped_sizes": requested_sizes,
                }
            )
            continue
        available = len(pool.vertices) - 1
        skipped = [n for n in requested_sizes if n > available]
        pool_poi = sum(1 for tag in pool.source_tags[1:] if is_poi_source_tag(tag))
        reports.append(
            {
                "city": group.city,
                "method": pool_request.method,
                "pool_total": available,
                "poi_available": pool_poi,
                "parametric_filled": available - pool_poi,
                "instances": len(group.rows),
                "requested_sizes": requested_sizes,
                "skipped_sizes": skipped,
                "status": "skipped" if len(skipped) == len(requested_sizes) else ("partial" if skipped else "ok"),
            }
        )
    return {"ok": True, "groups": reports, "instances": len(rows)}
