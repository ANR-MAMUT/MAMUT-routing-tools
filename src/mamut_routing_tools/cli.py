"""mamut-tools: command-line interface of MAMUT-routing-tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="mamut-tools",
    help="Local MAMUT-routing tool suite: OSM acquisition, road-graph engine, route geometry, and instance generation.",
    no_args_is_help=True,
    add_completion=False,
)

roadgraph_app = typer.Typer(help="Road-graph engine (OpenStreetMapX-compatible construction).", no_args_is_help=True)
geometry_app = typer.Typer(help="BKS route-geometry materialization.", no_args_is_help=True)
osm_app = typer.Typer(help="OSM city acquisition (Nominatim + Overpass).", no_args_is_help=True)
generate_app = typer.Typer(help="Interactive CVRP/VRPTW instance generation on city road graphs.", no_args_is_help=True)
app.add_typer(roadgraph_app, name="roadgraph")
app.add_typer(geometry_app, name="geometry")
app.add_typer(osm_app, name="osm")
app.add_typer(generate_app, name="generate")


def _resolve_city_osm(city: str, osm_path: Path | None, workspace: Path) -> Path:
    """The city's extract: an explicit path, else <workspace>/osmdata/<City>.osm."""
    from mamut_routing_tools.workspace import osmdata_dir

    if osm_path is not None:
        return osm_path
    candidate = osmdata_dir(workspace, create=False) / f"{city}.osm"
    if candidate.is_file():
        return candidate
    raise typer.BadParameter(
        f"No OSM extract for '{city}' at {candidate}. Fetch it first: mamut-tools osm fetch-city '{city}' --osm-dir {candidate.parent}"
    )


@generate_app.command("single")
def generate_single_cmd(
    city: Annotated[str, typer.Argument(help="City name (matching <workspace>/osmdata/<City>.osm unless --osm-path).")],
    n_customers: Annotated[int, typer.Option("--n", help="Number of customers.")] = 50,
    method: Annotated[str, typer.Option("--method", help="Sampling method: poi_categories, parametric_attach, hybrid.")] = "poi_categories",
    seed: Annotated[int, typer.Option("--seed")] = 0,
    demand_type: Annotated[int, typer.Option("--demand-type", min=1, max=7)] = 7,
    avg_route_size: Annotated[int, typer.Option("--avg-route-size", min=1, max=7)] = 4,
    depot_mode: Annotated[str, typer.Option("--depot-mode", help="random, center, corner.")] = "center",
    customer_mode: Annotated[str, typer.Option("--customer-mode", help="random, clustered, random_clustered.")] = "random_clustered",
    vrptw: Annotated[bool, typer.Option("--vrptw/--no-vrptw", help="Also derive the fastest-metric VRPTW twin.")] = False,
    tw_method: Annotated[str, typer.Option("--tw-method", help="route_centered or reachable_interval.")] = "route_centered",
    osm_path: Annotated[Optional[Path], typer.Option("--osm-path", help="Explicit OSM extract path.")] = None,
    output_dir: Annotated[Optional[Path], typer.Option("--output-dir", help="Workspace directory (default: the resolved workspace).")] = None,
) -> None:
    """Generate one instance (3 metric .vrp files + meta + manifest + .vrp.json)."""
    from mamut_routing_tools.generation.single import GenerationRequest, generate_single_instance
    from mamut_routing_tools.generation.vrptw import derive_vrptw_from_cvrp
    from mamut_routing_tools.generation.writers import slugify
    from mamut_routing_tools.workspace import instances_dir, resolve_workspace

    workspace = resolve_workspace(output_dir)
    request = GenerationRequest(
        city=city,
        osm_path=_resolve_city_osm(city, osm_path, workspace),
        method=method,
        n_customers=n_customers,
        seed=seed,
        demand_type=demand_type,
        avg_route_size=avg_route_size,
        depot_mode=depot_mode,
        customer_mode=customer_mode,
    )
    result = generate_single_instance(request, instances_dir(workspace))
    if vrptw:
        result["vrptw"] = derive_vrptw_from_cvrp(
            result["folder"],
            result["base_name"],
            tw_method=tw_method,
            place_slug=slugify(city),
            source_seed=seed,
        )
    typer.echo(json.dumps(result, indent=1))


@generate_app.command("preview")
def generate_preview_cmd(
    city: Annotated[str, typer.Argument(help="City name.")],
    n_customers: Annotated[int, typer.Option("--n")] = 50,
    method: Annotated[str, typer.Option("--method")] = "poi_categories",
    seed: Annotated[int, typer.Option("--seed")] = 0,
    depot_mode: Annotated[str, typer.Option("--depot-mode")] = "center",
    customer_mode: Annotated[str, typer.Option("--customer-mode")] = "random_clustered",
    osm_path: Annotated[Optional[Path], typer.Option("--osm-path")] = None,
    output_dir: Annotated[Optional[Path], typer.Option("--output-dir")] = None,
) -> None:
    """Preview a selection as GeoJSON (no artifacts written)."""
    from mamut_routing_tools.generation.single import GenerationRequest, build_generation_selection, preview_geojson
    from mamut_routing_tools.workspace import resolve_workspace

    workspace = resolve_workspace(output_dir)
    request = GenerationRequest(
        city=city,
        osm_path=_resolve_city_osm(city, osm_path, workspace),
        method=method,
        n_customers=n_customers,
        seed=seed,
        depot_mode=depot_mode,
        customer_mode=customer_mode,
    )
    typer.echo(json.dumps(preview_geojson(build_generation_selection(request)), indent=1))


@generate_app.command("bulk")
def generate_bulk_cmd(
    cities: Annotated[list[str], typer.Argument(help="City names (extracts under <workspace>/osmdata/).")],
    n_list: Annotated[str, typer.Option("--n-list", help="Comma-separated customer counts, e.g. 10,25,50.")] = "50",
    demand_types: Annotated[str, typer.Option("--demand-types", help="Comma-separated demand types (1-7).")] = "7",
    avg_route_sizes: Annotated[str, typer.Option("--avg-route-sizes", help="Comma-separated route-size bands (1-7).")] = "4",
    method: Annotated[str, typer.Option("--method")] = "poi_categories",
    seed: Annotated[int, typer.Option("--seed", help="Base seed; per-instance seeds derive from it.")] = 0,
    depot_mode: Annotated[str, typer.Option("--depot-mode")] = "center",
    customer_mode: Annotated[str, typer.Option("--customer-mode")] = "random_clustered",
    output_dir: Annotated[Optional[Path], typer.Option("--output-dir")] = None,
) -> None:
    """Bulk-generate over cities x sizes x demand types x route-size bands."""
    from mamut_routing_tools.generation.bulk import generate_bulk_instances
    from mamut_routing_tools.generation.single import GenerationRequest
    from mamut_routing_tools.workspace import instances_dir, resolve_workspace

    workspace = resolve_workspace(output_dir)
    parsed_cities = [(name, _resolve_city_osm(name, None, workspace)) for name in cities]
    base_request = GenerationRequest(
        city=parsed_cities[0][0],
        osm_path=parsed_cities[0][1],
        method=method,
        seed=seed,
        depot_mode=depot_mode,
        customer_mode=customer_mode,
    )
    result = generate_bulk_instances(
        base_request,
        cities=parsed_cities,
        n_list=[int(part) for part in n_list.split(",") if part.strip()],
        demand_types=[int(part) for part in demand_types.split(",") if part.strip()],
        avg_route_sizes=[int(part) for part in avg_route_sizes.split(",") if part.strip()],
        output_root=instances_dir(workspace),
    )
    summary = {
        "ok": result["ok"],
        "generated": result["generated"],
        "city_reports": result["city_reports"],
        "bases": [item["base_name"] for item in result["results"]],
    }
    typer.echo(json.dumps(summary, indent=1))


@generate_app.command("derive-vrptw")
def generate_derive_vrptw_cmd(
    folder: Annotated[Path, typer.Argument(help="Folder holding the generated CVRP base files.")],
    base: Annotated[str, typer.Argument(help="Instance base name (e.g. lyon_poi-n51-k5).")],
    tw_method: Annotated[str, typer.Option("--tw-method")] = "route_centered",
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    """Derive the fastest-metric VRPTW twin of an already generated CVRP base."""
    from mamut_routing_tools.generation.vrptw import derive_vrptw_from_cvrp

    result = derive_vrptw_from_cvrp(folder, base, tw_method=tw_method, source_seed=seed)
    typer.echo(json.dumps(result, indent=1))


@osm_app.command("fetch-city")
def osm_fetch_city_cmd(
    city: Annotated[str, typer.Argument(help="City or locality name to geocode and download.")],
    country: Annotated[str, typer.Option("--country", help="Optional country to disambiguate the geocode.")] = "",
    osm_dir: Annotated[Path, typer.Option("--osm-dir", help="Directory for the downloaded <city>.osm extract.")] = Path("osmdata"),
    padding_km: Annotated[float, typer.Option("--padding-km", help="Extra bbox padding in km.")] = 0.0,
    max_radius_km: Annotated[float, typer.Option("--max-radius-km", help="Clamp the administrative bbox to a square of this radius around the place's geocode point (0 = no clamp).")] = 0.0,
) -> None:
    """Download an OSM extract (roads + amenities) for a city by name."""
    from mamut_routing_tools.osm import fetch_and_store_city_osm

    summary = fetch_and_store_city_osm(
        city,
        country=country,
        osm_dir=osm_dir,
        padding_km=padding_km,
        max_radius_km=max_radius_km,
    )
    typer.echo(json.dumps(summary, indent=1))


@roadgraph_app.command("info")
def roadgraph_info_cmd(
    osm_path: Annotated[Path, typer.Argument(help="OSM XML extract to build the road graph from.")],
    only_intersections: Annotated[bool, typer.Option("--only-intersections/--all-nodes")] = True,
    trim_to_connected: Annotated[bool, typer.Option("--trim/--no-trim")] = True,
) -> None:
    """Build the road graph and print vertex/edge statistics."""
    from mamut_routing_tools.roadgraph import load_road_graph

    graph = load_road_graph(
        osm_path,
        only_intersections=only_intersections,
        trim_to_connected=trim_to_connected,
    )
    class_counts: dict[int, int] = {}
    for cls in graph.edge_class:
        class_counts[cls] = class_counts.get(cls, 0) + 1
    typer.echo(
        json.dumps(
            {
                "osm_path": str(graph.osm_path),
                "only_intersections": graph.only_intersections,
                "trim_to_connected": graph.trim_to_connected,
                "vertices": graph.vertex_count,
                "edges": graph.edge_count,
                "total_edge_length_km": round(sum(graph.edge_weight) / 1000.0, 3),
                "edge_class_counts": {str(k): class_counts[k] for k in sorted(class_counts)},
                "ref_lla": {"lat": graph.ref_lla.lat, "lon": graph.ref_lla.lon},
            },
            indent=1,
        )
    )


@geometry_app.command("materialize-plan")
def geometry_materialize_plan_cmd(
    plan_path: Annotated[Path, typer.Argument(help="Group plan JSON (route_geometry.py contract).")],
    repo_root: Annotated[Path, typer.Option("--repo-root", help="MAMUT-routing repo root the plan paths are relative to.")],
    result_dir: Annotated[Optional[Path], typer.Option("--result-dir", help="Directory for per-group result files. Prints to stdout when omitted.")] = None,
) -> None:
    """Materialize a route-geometry group plan (website build contract)."""
    from mamut_routing_tools.geometry import materialize_plan

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    results = materialize_plan(repo_root, plan)
    if result_dir is None:
        typer.echo(json.dumps(results, sort_keys=True))
        return
    result_dir.mkdir(parents=True, exist_ok=True)
    for result_file, payload in results.items():
        target = result_dir / result_file
        target.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        typer.echo(str(target))


if __name__ == "__main__":  # pragma: no cover
    app()
