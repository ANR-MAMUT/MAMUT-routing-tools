"""Instance artifact writers and parsers (CVRPLIB .vrp, workbench _meta.json,
_manifest.json, and the lib-contract .vrp.json payloads)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mamut_routing_tools.geo import LLA


def slugify(value: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "x"


def method_abbrev(method: str) -> str:
    m = method.strip().lower()
    return {"poi_categories": "poi", "parametric_attach": "par", "hybrid": "hyb"}.get(m, slugify(m))


def instance_path_plan(
    city: str,
    method: str,
    n_customers: int,
    route_count: int,
    output_root: str | Path,
) -> tuple[Path, str]:
    city_slug = slugify(city)
    n_nodes = n_customers + 1
    folder = Path(output_root) / "osm" / city_slug / f"n{n_nodes}"
    base = f"{city_slug}_{method_abbrev(method)}-n{n_nodes}-k{route_count}"
    return folder, base


def write_cvrplib(
    path: str | Path,
    name: str,
    comment: str,
    coords: list[tuple[float, float]],
    demands: list[int],
    matrix: list[list[int]],
    capacity: int,
) -> None:
    lines = [
        f"NAME : {name}",
        "TYPE : CVRP",
        f"COMMENT : {comment}",
        f"DIMENSION : {len(demands)}",
        f"CAPACITY : {capacity}",
        "EDGE_WEIGHT_TYPE : EXPLICIT",
        "EDGE_WEIGHT_FORMAT : FULL_MATRIX",
        "EDGE_WEIGHT_SECTION",
    ]
    lines.extend(" ".join(str(value) for value in row) for row in matrix)
    lines.append("NODE_COORD_SECTION")
    lines.extend(f"{i + 1} {x} {y}" for i, (x, y) in enumerate(coords))
    lines.append("DEMAND_SECTION")
    lines.extend(f"{i + 1} {demand}" for i, demand in enumerate(demands))
    lines.append("DEPOT_SECTION\n1\n-1\nEOF")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_instance_metadata(
    meta_path: str | Path,
    *,
    city: str,
    osm_file: str,
    instance_name: str,
    metric_files: list[str],
    ref_lla: LLA,
    vertices: list[int],
    poi_lats: list[float],
    poi_lons: list[float],
    coords: list[tuple[float, float]],
    demands: list[int],
    method: str,
    source_tags: list[str],
    only_intersections: bool = True,
    trim_to_connected_graph: bool = True,
    generation_params: dict[str, Any] | None = None,
    road_cache: dict[str, Any] | None = None,
) -> None:
    n = len(vertices)
    assert n == len(coords) == len(demands) == len(poi_lats) == len(poi_lons) == len(source_tags)
    nodes = [
        {
            "instance_node_id": i + 1,
            "graph_vertex_id": vertices[i],
            "poi_lat": poi_lats[i],
            "poi_lon": poi_lons[i],
            "enu_x": coords[i][0],
            "enu_y": coords[i][1],
            "demand": demands[i],
            "source_tag": source_tags[i],
        }
        for i in range(n)
    ]
    payload: dict[str, Any] = {
        "schema_version": 2,
        "city": city,
        "instance_name": instance_name,
        "source_osm_file": osm_file,
        "metric_files": metric_files,
        "depot_instance_node_id": 1,
        "method": method,
        "reference_lla": {"lat": ref_lla.lat, "lon": ref_lla.lon, "alt": ref_lla.alt},
        "map_options": {
            "only_intersections": only_intersections,
            "trim_to_connected_graph": trim_to_connected_graph,
        },
        "generation_params": generation_params or {},
        "nodes": nodes,
    }
    if road_cache is not None:
        payload["road_cache"] = road_cache
    Path(meta_path).write_text(json.dumps(payload, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ParsedCvrpInstance:
    name: str
    comment: str
    dimension: int
    capacity: int
    arc_costs: list[list[int]]
    coordinates: list[tuple[float, float]]
    demands: list[int]
    depot_node_index: int  # 1-based, as written in DEPOT_SECTION


def parse_cvrp_vrp(path: str | Path) -> ParsedCvrpInstance:
    headers: dict[str, str] = {}
    edge_tokens: list[str] = []
    coordinates: list[tuple[float, float]] = []
    demands: list[int] = []
    depot_indices: list[int] = []
    section = ""
    section_headers = {"EDGE_WEIGHT_SECTION", "NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION", "EOF"}

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in section_headers:
            section = "" if line == "EOF" else line
            continue
        if not section:
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            headers[key] = value
            continue
        if section == "EDGE_WEIGHT_SECTION":
            edge_tokens.extend(line.split())
        elif section == "NODE_COORD_SECTION":
            parts = line.split()
            if len(parts) >= 3:
                coordinates.append((float(parts[1]), float(parts[2])))
        elif section == "DEMAND_SECTION":
            parts = line.split()
            if len(parts) >= 2:
                demands.append(int(parts[1]))
        elif section == "DEPOT_SECTION":
            if line == "-1":
                section = ""
                continue
            depot_indices.append(int(line))

    if "DIMENSION" not in headers or "CAPACITY" not in headers:
        raise ValueError(f"Missing DIMENSION/CAPACITY header in {path}")
    dimension = int(headers["DIMENSION"])
    capacity = int(headers["CAPACITY"])
    if len(edge_tokens) != dimension * dimension:
        raise ValueError(
            f"EDGE_WEIGHT_SECTION has {len(edge_tokens)} tokens, expected {dimension * dimension} in {path}"
        )
    arc_costs = [
        [int(edge_tokens[row * dimension + col]) for col in range(dimension)] for row in range(dimension)
    ]
    if len(coordinates) != dimension or len(demands) != dimension:
        raise ValueError(f"NODE_COORD/DEMAND section size mismatch in {path}")
    return ParsedCvrpInstance(
        name=headers.get("NAME", ""),
        comment=headers.get("COMMENT", ""),
        dimension=dimension,
        capacity=capacity,
        arc_costs=arc_costs,
        coordinates=coordinates,
        demands=demands,
        depot_node_index=depot_indices[0] if depot_indices else 1,
    )


def build_vrp_json_payload(
    parsed: ParsedCvrpInstance,
    *,
    instance_name: str,
    metric_variant: str,
    place_slug: str,
    source_base_name: str,
    source_city: str,
    source_seed: int,
    source_folder: str,
    num_vehicles_lb: int | None,
    artifact_paths: dict[str, str],
    sibling_variant_paths: dict[str, str],
    reference_lla: dict[str, float] | None,
    generated_at: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "authors": "OSM CVRP Workbench",
        "generated_at": generated_at,
        "problem_type": "CVRP",
        "metric_variant": metric_variant,
        "place_slug": place_slug,
        "source_base_name": source_base_name,
        "source_city": source_city,
        "source_seed": source_seed,
        "source_folder": source_folder,
        "generator_version": "mamut-routing-tools-v2",
        "artifact_paths": artifact_paths,
        "sibling_variant_paths": sibling_variant_paths,
        "derived_problem_paths": {},
        "source_problem_paths": {},
    }
    if num_vehicles_lb is not None:
        metadata["num_vehicles_lb"] = num_vehicles_lb
    payload: dict[str, Any] = {
        "instance_name": instance_name,
        "instance_origin": "OsmCvrpGen",
        "benchmark_name": "Mamut2026",
        "num_customers": parsed.dimension - 1,
        "vehicle_capacity": parsed.capacity,
        "coordinates": [[x, y] for x, y in parsed.coordinates],
        "demands": list(parsed.demands),
        "depot": parsed.depot_node_index - 1,
        "arc_costs": [list(row) for row in parsed.arc_costs],
        "metadata": metadata,
    }
    if reference_lla is not None:
        payload["reference_lla"] = reference_lla
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
