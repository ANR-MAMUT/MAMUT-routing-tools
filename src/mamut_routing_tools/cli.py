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
app.add_typer(roadgraph_app, name="roadgraph")
app.add_typer(geometry_app, name="geometry")


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
