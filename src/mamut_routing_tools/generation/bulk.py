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
    effective_method,
    generate_single_instance,
    materialize_instance,
    selection_composition,
)
from mamut_routing_tools.generation.vrptw import TW_METHODS, derive_vrptw_from_cvrp, stable_seed
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
        # Checked up front rather than at derivation time: a bad method would
        # otherwise only surface once part of the batch is already on disk.
        if self.problem_type == "vrptw" and self.tw_method not in TW_METHODS:
            raise ValueError(f"Unsupported time window method '{self.tw_method}'")


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
        # Two rows that bind POIs to the graph differently draw from genuinely
        # different pools and must not share one.
        request.poi_attach_mode,
        request.poi_attach_radius_m,
    )


def _name_key(request: GenerationRequest) -> tuple[Any, ...]:
    """Rows that obviously want a readable ``-s<seed>`` suffix, seed excluded.

    Only a hint for nicer names: it cannot decide collisions on its own, because
    the real name encodes ``route_count`` (roughly n/r, so two demand types on
    one band collide) and, in manual mode, the *resolved* customer count rather
    than the requested one. ``materialize_instance`` holds the actual guarantee
    through its reserved-name set.
    """
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
    """Attach the VRPTW twin in place when the row asks for one.

    A twin that cannot be derived is reported on its own instance instead of
    aborting the batch: the CVRP files are already written and valid, and losing
    the other N-1 instances of a long run to one bad row is far worse than
    finishing with one row flagged.
    """
    try:
        result["vrptw"] = derive_vrptw_from_cvrp(
            result["folder"],
            result["base_name"],
            tw_method=row.tw_method,
            place_slug=slugify(row.request.city),
            source_seed=seed,
        )
    except (ValueError, OSError) as error:
        warnings.warn(
            f"{result['base_name']}: VRPTW derivation failed ({error})", stacklevel=2
        )
        result["vrptw"] = None
        result["vrptw_error"] = str(error)


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

    # Results are collected against their input index and re-sorted at the end,
    # so callers get them in the order they asked for rather than grouped by
    # pool key -- the bulk table has no other way to line rows up with results.
    indexed_results: list[tuple[int, dict[str, Any]]] = []
    city_reports: list[dict[str, Any]] = []
    # One entry per input row, so a caller with a row table can say which of its
    # rows produced nothing instead of only reporting a total that came up short.
    row_reports: dict[int, dict[str, Any]] = {}

    def row_generated(index: int, result: dict[str, Any]) -> None:
        composition = (result.get("summary") or {}).get("composition") or {}
        row_reports[index] = {
            "index": index,
            "status": "generated",
            "base_name": result.get("base_name"),
            "poi_customers": composition.get("poi_customers"),
            "parametric_customers": composition.get("parametric_customers"),
            "notice": result.get("notice"),
            "vrptw_error": result.get("vrptw_error"),
        }

    def row_skipped(index: int, reason: str) -> None:
        row_reports[index] = {"index": index, "status": "skipped", "reason": reason}
    # Guarantees no two rows of this run write the same files; see
    # materialize_instance. Empty at the start, so earlier runs still overwrite.
    reserved_names: set[str] = set()
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

    def skip(count: int, reason: str) -> None:
        """Account for rows that will never be generated, and say so."""
        nonlocal done
        done += count
        if context is not None:
            context.progress(f"{done}/{total} handled: {reason}", current=done, total=total)

    for group in groups.values():
        if context is not None:
            context.check_cancelled()

        # Hand-picked selections are per-row by construction; nothing to pool.
        if group.rows[0][1].request.method == "manual":
            for index, row in group.rows:
                if context is not None:
                    context.check_cancelled()
                seed = row.explicit_seed if row.explicit_seed is not None else row.request.seed
                result = generate_single_instance(
                    replace(row.request, seed=seed),
                    output_root,
                    name_suffix=f"-s{seed}" if _name_key(row.request) in ambiguous else "",
                    reserved_names=reserved_names,
                )
                if row.problem_type == "vrptw":
                    _derive_twin(result, row, seed)
                indexed_results.append((index, result))
                row_generated(index, result)
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
            for index, _row in group.rows:
                row_skipped(index, f"{group.city} selection failed: {error}")
            skip(len(group.rows), f"{group.city} selection failed, {len(group.rows)} row(s) skipped")
            continue

        actual_max_nc = len(pool.vertices) - 1
        pool_composition = pool.params.get("composition") or {}
        # The pool may have been reclassified by its own fallback; rows are sized
        # differently and each has to be judged against what was asked for.
        pool_requested_method = str(pool.params.get("requested_method") or pool.params["method"])
        pool_poi_indices = [
            i for i in range(1, len(pool.source_tags)) if is_poi_source_tag(pool.source_tags[i])
        ]
        pool_parametric_indices = [
            i for i in range(1, len(pool.source_tags)) if not is_poi_source_tag(pool.source_tags[i])
        ]
        # Reconstructed from the pool report so each sliced row can be described
        # against the same POI pool without re-scanning the extract.
        pool_poi_stats = {
            "pool_matching": pool_composition.get("poi_pool_matching", 0),
            "attached": pool_composition.get("poi_pool_attachable") or 0,
            "exhausted": pool_composition.get("poi_pool_attachable") is not None,
        }
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
                "poi_pool_matching": pool_composition.get("poi_pool_matching"),
                "poi_pool_attachable": pool_composition.get("poi_pool_attachable"),
                "requested_sizes": requested_sizes,
                "valid_sizes": sorted({row.request.n_customers for _, row in valid_rows}),
                "skipped_sizes": skipped_sizes,
                "status": "skipped" if not valid_rows else ("partial" if skipped_sizes else "ok"),
            }
        )
        if len(valid_rows) < len(group.rows):
            valid_indices = {index for index, _row in valid_rows}
            for index, row in group.rows:
                if index not in valid_indices:
                    row_skipped(
                        index,
                        f"{group.city} pool holds {actual_max_nc} customers, "
                        f"short of the {row.request.n_customers} this row asks for",
                    )
            skip(
                len(group.rows) - len(valid_rows),
                f"{group.city} pool holds {actual_max_nc}, so n = "
                f"{', '.join(str(n) for n in skipped_sizes)} cannot be served",
            )
        if not valid_rows:
            continue

        d_short_full, d_fast_full, geom_short_full, geom_fast_full = compute_matrices(
            pool.graph, pool.vertices
        )
        d_eucl_full, coords_full = euclidean_matrix_from_vertices(pool.graph, pool.vertices)
        total_vertices = len(pool.vertices)

        for index, row in valid_rows:
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
            if nc >= actual_max_nc:
                sel_indices = list(range(total_vertices))
            elif pool_poi_indices:
                # POIs first, then parametric points: drawing from the union
                # would hand a small row an almost purely parametric instance
                # just because a larger row in the batch grew the pool.
                poi_part = list(pool_poi_indices)
                param_part = list(pool_parametric_indices)
                rng.shuffle(poi_part)
                rng.shuffle(param_part)
                sel_indices = [0, *sorted((poi_part + param_part)[:nc])]
            else:
                # No POI in the pool: keep the original single-shuffle draw, so
                # parametric batches reproduce byte for byte across this change.
                perm = list(range(1, total_vertices))
                rng.shuffle(perm)
                sel_indices = [0, *sorted(perm[:nc])]

            vertices = [pool.vertices[i] for i in sel_indices]
            subset = set(vertices)
            source_tags = [pool.source_tags[i] for i in sel_indices]
            params = dict(pool.params)
            params["seed"] = inst_seed
            params["n_customers"] = nc
            params["demand_type"] = row.request.demand_type
            params["avg_route_size"] = row.request.avg_route_size
            # The pool's own composition describes the oversized pool, not this
            # slice: recompute it from the tags this row actually drew, and let
            # the row be named after what it holds rather than after the pool.
            params["method"] = effective_method(pool_requested_method, source_tags[1:])
            params["composition"] = selection_composition(
                method=pool_requested_method,
                requested=nc,
                customer_tags=source_tags[1:],
                poi_stats=pool_poi_stats,
                poi_share=float(pool.params.get("hybrid_poi_share") or 0.0),
                categories=list(pool.params.get("categories") or []),
                graph_options_relaxed=pool.params.get("graph_options_relaxed"),
                effective=params["method"],
            )
            selection = Selection(
                graph=pool.graph,
                vertices=vertices,
                poi_lats=[pool.poi_lats[i] for i in sel_indices],
                poi_lons=[pool.poi_lons[i] for i in sel_indices],
                source_tags=source_tags,
                params=params,
                poi_meta=[
                    pool.poi_meta[i] if i < len(pool.poi_meta) else None for i in sel_indices
                ],
                snap_distances_m=[
                    pool.snap_distances_m[i] if i < len(pool.snap_distances_m) else 0.0
                    for i in sel_indices
                ],
                poi_merged=[
                    pool.poi_merged[i] if i < len(pool.poi_merged) else []
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
                reserved_names=reserved_names,
            )
            if row.problem_type == "vrptw":
                _derive_twin(result, row, inst_seed)
            indexed_results.append((index, result))
            row_generated(index, result)
            advance(row)

    results = [result for _, result in sorted(indexed_results, key=lambda pair: pair[0])]
    return {
        "ok": True,
        "generated": len(results),
        "results": results,
        "city_reports": city_reports,
        "row_reports": [
            row_reports.get(index, {"index": index, "status": "skipped", "reason": "not reached"})
            for index in range(len(rows))
        ],
    }


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
        pool_composition = pool.params.get("composition") or {}
        reports.append(
            {
                "city": group.city,
                "method": pool_request.method,
                "pool_total": available,
                "poi_available": pool_poi,
                "parametric_filled": available - pool_poi,
                "poi_pool_matching": pool_composition.get("poi_pool_matching"),
                "poi_pool_attachable": pool_composition.get("poi_pool_attachable"),
                # Rows are served POIs first, so only sizes past the POI count
                # have to take parametric points.
                "sizes_needing_parametric": [
                    n for n in requested_sizes if n <= available and n > pool_poi
                ],
                "instances": len(group.rows),
                "requested_sizes": requested_sizes,
                "skipped_sizes": skipped,
                "status": "skipped" if len(skipped) == len(requested_sizes) else ("partial" if skipped else "ok"),
            }
        )
    return {"ok": True, "groups": reports, "instances": len(rows)}
