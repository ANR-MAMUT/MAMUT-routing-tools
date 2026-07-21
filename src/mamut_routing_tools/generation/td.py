"""Standalone per-instance TDVRP / TDVRPTW derivation.

The per-instance counterpart of the family builder's ``build_td`` stage
(``mamut_routing_tools.family.family``). Where ``build_td`` publishes the whole
Mamut2026 family into a marker-rooted collection tree, this module derives the
time-dependent twins of a *single* already-generated instance in place, next to
its ``generate single`` / ``derive-vrptw`` artifacts.

Given a folder holding

- ``<base>_meta.json``          (from ``generate single``),
- ``<base>_fastest.cvrptw.vrp`` (from ``generate derive-vrptw``: the PROVIDED
  time windows + service times to reuse), and
- ``<base>_vrptw_manifest.json`` (its nearest-neighbour anchor route),

:func:`derive_td_from_vrptw` rebuilds the TD bridge for the instance's city,
assembles the trimmed road-graph sidecar and one traffic overlay per
``model x intensity`` combination, lifts the provided deadlines just enough to
certify the anchor route under every derived overlay (earliest bounds are never
reduced), and writes the canonical ``road-graph`` v2 TD twins:

- ``<base>-<model>-<intensity>.vrp.json``        (TDVRPTW: lifted windows),
- ``<base>-<model>-<intensity>.tdvrp.vrp.json``  (TDVRP: no windows).

Both mirror ``build_td``'s twin payload (same td block, sha-pinned road +
traffic sidecars and materialized ``atf_sha256``), so each loads and
self-verifies through ``mamut_routing_lib.td.load_td_instance(path,
verify_sha256=True)``. A ``mamut-collection.json`` marker is dropped in the
folder so the collection-root-relative sidecar refs resolve with no extra
arguments (the road-graph td model always resolves its sidecars against a
collection root).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mamut_routing_lib.json_utils import save_json_to_file
from mamut_routing_lib.td import (
    compute_atf_sha256,
    compute_road_graph_sha256,
    compute_traffic_overlay_sha256,
    materialize_instance_atfs_roadgraph,
    materialize_selected_atfs_roadgraph,
    save_instance_road_graph,
    save_traffic_overlay,
    td_instance_from_payload,
)

from mamut_routing_tools.family.bridge import (
    load_bridge_graph,
    load_bridge_nodes,
    load_bridge_speeds,
)
from mamut_routing_tools.family.family import (
    AUTHORS,
    DEFAULT_EXTENSION_END,
    DEFAULT_SAMPLE_STEP,
    FAMILY,
    PIPELINE_VERSION,
    TD_HORIZON,
    TD_INTENSITIES,
    TD_MODELS,
    _align_overlay,
    _audit_and_lift,
    _full_city_road_graph,
    _trim_road_graph,
    ensure_collection_root,
    simplify_tolerance_for,
)
from mamut_routing_tools.family.naming import subinstance_name, td_instance_name
from mamut_routing_tools.generation.vrptw import nearest_neighbour_route
from mamut_routing_tools.generation.writers import slugify
from mamut_routing_tools.roadgraph.build import load_road_graph
from mamut_routing_tools.td.traffic import export_bridge

BRIDGE_DIRNAME = ".td-bridge"
GENERATOR_NAME = "mamut-routing-tools"


class TDDerivationError(ValueError):
    """Raised when the inputs of a standalone TD derivation are inconsistent."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _parse_cvrptw_vrp(path: Path) -> dict[str, Any]:
    """Parse a TSPLIB CVRPTW ``.vrp`` (as written by ``write_cvrptw_vrp``).

    ``generation.writers.parse_cvrp_vrp`` cannot be reused directly: its
    section scanner does not know ``TIME_WINDOW_SECTION`` /
    ``SERVICE_TIME_SECTION`` and would fold those rows into the demand list.
    This reader understands all five node sections and returns capacity, the
    explicit matrix, coordinates, demands, per-node time windows and service
    times (all node-indexed, depot first).
    """
    headers: dict[str, str] = {}
    edge_tokens: list[str] = []
    coordinates: list[tuple[float, float]] = []
    demands: list[int] = []
    time_windows: list[tuple[int, int]] = []
    service_times: list[int] = []
    depot_indices: list[int] = []
    section = ""
    section_headers = {
        "EDGE_WEIGHT_SECTION",
        "NODE_COORD_SECTION",
        "DEMAND_SECTION",
        "TIME_WINDOW_SECTION",
        "SERVICE_TIME_SECTION",
        "DEPOT_SECTION",
        "EOF",
    }
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in section_headers:
            section = "" if line == "EOF" else line
            continue
        if not section:
            if ":" in line:
                key, value = (part.strip() for part in line.split(":", 1))
                headers[key] = value
            continue
        parts = line.split()
        if section == "EDGE_WEIGHT_SECTION":
            edge_tokens.extend(parts)
        elif section == "NODE_COORD_SECTION" and len(parts) >= 3:
            coordinates.append((float(parts[1]), float(parts[2])))
        elif section == "DEMAND_SECTION" and len(parts) >= 2:
            demands.append(int(parts[1]))
        elif section == "TIME_WINDOW_SECTION" and len(parts) >= 3:
            time_windows.append((int(parts[1]), int(parts[2])))
        elif section == "SERVICE_TIME_SECTION" and len(parts) >= 2:
            service_times.append(int(parts[1]))
        elif section == "DEPOT_SECTION":
            if line == "-1":
                section = ""
            else:
                depot_indices.append(int(line))

    if "DIMENSION" not in headers or "CAPACITY" not in headers:
        raise TDDerivationError(f"missing DIMENSION/CAPACITY header in {path}")
    dimension = int(headers["DIMENSION"])
    capacity = int(headers["CAPACITY"])
    if len(edge_tokens) != dimension * dimension:
        raise TDDerivationError(
            f"EDGE_WEIGHT_SECTION has {len(edge_tokens)} tokens, expected {dimension * dimension} in {path}"
        )
    matrix = [
        [int(edge_tokens[row * dimension + col]) for col in range(dimension)]
        for row in range(dimension)
    ]
    for name, seq in (
        ("NODE_COORD", coordinates),
        ("DEMAND", demands),
        ("TIME_WINDOW", time_windows),
        ("SERVICE_TIME", service_times),
    ):
        if len(seq) != dimension:
            raise TDDerivationError(
                f"{name}_SECTION has {len(seq)} rows, expected {dimension} in {path}"
            )
    return {
        "dimension": dimension,
        "capacity": capacity,
        "matrix": matrix,
        "coordinates": coordinates,
        "demands": demands,
        "time_windows": time_windows,
        "service_times": service_times,
        "depot_index": depot_indices[0] if depot_indices else 1,
    }


def _resolve_osm_path(source_osm_file: str) -> Path:
    """Resolve ``meta['source_osm_file']`` against the tool OSM caches.

    Accepts an absolute path, a ``osmdata/<City>.osm`` relative path, or a bare
    ``<City>.osm`` name; probes the literal path, then the repo/env workspace
    ``osmdata/`` and the user cache ``~/.cache/mamut-tools/osmdata/``.
    """
    from mamut_routing_tools.workspace import osmdata_dir, resolve_workspace

    literal = Path(source_osm_file).expanduser()
    candidates = [literal]
    name = literal.name
    for create in (False,):
        candidates.append(osmdata_dir(resolve_workspace(None, create=False), create=create) / name)
    candidates.append(Path.home() / ".cache" / "mamut-tools" / "osmdata" / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"could not resolve OSM extract {source_osm_file!r}; tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _anchor_customers(manifest: dict[str, Any], fastest_matrix: list[list[int]]) -> tuple[list[int], str]:
    """The anchor route as a customer sequence (depot stripped) plus its source.

    Uses the persisted ``derivation.anchor_route`` when present (the
    route-centered nearest-neighbour route the windows were centred on),
    otherwise regenerates it deterministically on the fastest matrix (the
    ``reachable_interval`` case persists no route)."""
    route = (manifest.get("derivation") or {}).get("anchor_route")
    source = "manifest"
    if not route:
        route = nearest_neighbour_route(fastest_matrix)
        source = "regenerated"
    route = [int(node) for node in route]
    if route and route[0] == 0:
        route = route[1:]
    return route, source


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def _generator(stage: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    generator = {"name": GENERATOR_NAME, "stage": stage, "version": PIPELINE_VERSION}
    if extra:
        generator.update(extra)
    return generator


def derive_td_from_vrptw(
    folder: str | Path,
    base: str,
    *,
    model: str = "bpr",
    intensity: str = "moderate",
    all_combos: bool = False,
    seed: int = 42,
    force: bool = False,
) -> dict[str, Any]:
    """Derive the TDVRP + TDVRPTW twins of a generated instance in place.

    Reuses the PROVIDED VRPTW windows (lifting deadlines to time-dependent
    feasibility, never regenerating them) and produces canonical road-graph
    v2 twins that self-verify through ``load_td_instance(path,
    verify_sha256=True)``.
    """
    if model not in TD_MODELS:
        raise TDDerivationError(f"unknown traffic model {model!r}; known: {TD_MODELS}")
    if intensity not in TD_INTENSITIES:
        raise TDDerivationError(f"unknown intensity {intensity!r}; known: {TD_INTENSITIES}")

    folder = Path(folder)
    meta_path = folder / f"{base}_meta.json"
    cvrptw_path = folder / f"{base}_fastest.cvrptw.vrp"
    manifest_path = folder / f"{base}_vrptw_manifest.json"
    for required in (meta_path, cvrptw_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"missing input for {base}: {required}")

    meta = _load_json(meta_path)
    manifest = _load_json(manifest_path)
    cvrptw = _parse_cvrptw_vrp(cvrptw_path)

    nodes_meta = meta["nodes"]
    num_customers = len(nodes_meta) - 1
    if cvrptw["dimension"] != num_customers + 1:
        raise TDDerivationError(
            f"{base}: CVRPTW dimension {cvrptw['dimension']} does not match "
            f"meta node count {num_customers + 1}"
        )

    options = meta.get("map_options", {})
    only_intersections = bool(options.get("only_intersections", True))
    trim_to_connected = bool(options.get("trim_to_connected_graph", True))
    city_slug = slugify(str(meta.get("city", base.split("_", 1)[0])))
    method_tag = str(meta.get("method", meta.get("generation_params", {}).get("method", "")))

    combos = (
        [(m, i) for m in TD_MODELS for i in TD_INTENSITIES]
        if all_combos
        else [(model, intensity)]
    )
    subs = [subinstance_name(m, i) for m, i in combos]

    twin_paths: dict[tuple[str, str], dict[str, Path]] = {}
    for (m, i), sub in zip(combos, subs):
        name = td_instance_name(base, m, i)
        twin_paths[(m, i)] = {
            "TDVRPTW": folder / f"{name}.vrp.json",
            "TDVRP": folder / f"{name}.tdvrp.vrp.json",
        }
    all_targets = [p for paths in twin_paths.values() for p in paths.values()]
    if not force and all(p.exists() for p in all_targets):
        return {
            "ok": True,
            "base": base,
            "folder": str(folder),
            "action": "kept",
            "num_customers": num_customers,
            "combos": [
                {
                    "model": m,
                    "intensity": i,
                    "sub": sub,
                    "tdvrptw_twin": twin_paths[(m, i)]["TDVRPTW"].name,
                    "tdvrp_twin": twin_paths[(m, i)]["TDVRP"].name,
                }
                for (m, i), sub in zip(combos, subs)
            ],
        }

    osm_path = _resolve_osm_path(str(meta["source_osm_file"]))

    # 1. TD bridge for this city (graph + requested speed profiles + this
    #    instance's node -> OSM mapping), under a per-folder temp root.
    bridge_root = folder / BRIDGE_DIRNAME
    bridge_models = sorted({m for m, _ in combos})
    bridge_intensities = sorted({i for _, i in combos})
    bridge_dir = export_bridge(
        osm_path=osm_path,
        city_slug=city_slug,
        out_root=bridge_root,
        models=bridge_models,
        intensities=bridge_intensities,
        meta_paths=[meta_path],
        seed=seed,
        force=force,
        only_intersections=only_intersections,
        trim_to_connected=trim_to_connected,
    )
    bridge_graph = load_bridge_graph(bridge_dir / "graph.json")
    bridge_nodes = load_bridge_nodes(bridge_dir / f"nodes-{base}.json")

    # 2. Instance road graph: full city graph from the bridge, trimmed to the
    #    union of pinned free-flow fastest paths, saved next to the instance.
    ensure_collection_root(folder)  # collection marker: sidecar refs resolve here
    full = _full_city_road_graph(
        bridge_graph,
        bridge_nodes,
        base=base,
        sample_step=DEFAULT_SAMPLE_STEP,
        simplify_tolerance=simplify_tolerance_for(num_customers),
        extension_end=DEFAULT_EXTENSION_END,
        generator=_generator("derive-td", {"city": city_slug, "method": method_tag}),
    )
    road, _ = _trim_road_graph(full)
    road_file = f"{base}.road.json.gz"
    save_instance_road_graph(road, folder / road_file)
    road_sha = compute_road_graph_sha256(road)

    # 3. Per-combo overlays + ATF materialization. Anchor arcs certify the TW
    #    lift; the full ATF set pins atf_sha256.
    anchor_custs, anchor_source = _anchor_customers(manifest, cvrptw["matrix"])
    anchor_arc_keys = {
        (previous, customer)
        for previous, customer in zip([0, *anchor_custs], [*anchor_custs, 0])
    }
    provided_time_windows = [tuple(w) for w in cvrptw["time_windows"]]
    service_times = [int(s) for s in cvrptw["service_times"]]
    coordinates = [[float(node["enu_x"]), float(node["enu_y"])] for node in nodes_meta]
    demands = [int(node["demand"]) for node in nodes_meta]
    reference_lla = meta.get("reference_lla")

    overlay_refs: dict[str, dict[str, Any]] = {}
    overlay_sha: dict[str, str] = {}
    atf_sha: dict[str, str] = {}
    traffic_info: dict[str, dict[str, Any]] = {}
    anchor_arcs_by_sub: dict[str, dict[tuple[int, int], Any]] = {}
    for m, i in combos:
        sub = subinstance_name(m, i)
        speeds = load_bridge_speeds(bridge_dir / f"speeds-{m}-{i}.json", bridge_graph)
        overlay = _align_overlay(road, bridge_graph, speeds)
        overlay_file = f"{base}.traffic-{sub}.json.gz"
        save_traffic_overlay(overlay, folder / overlay_file)
        overlay_sha[sub] = compute_traffic_overlay_sha256(overlay)
        overlay_refs[sub] = {"path": overlay_file, "sha256": overlay_sha[sub]}
        traffic_info[sub] = {
            "model": speeds.model,
            "intensity": speeds.intensity,
            "seed": speeds.seed,
            "num_trips": speeds.num_trips,
            "params": speeds.params,
        }

        probe_payload = {
            "instance_name": td_instance_name(base, m, i),
            "instance_origin": "OsmCvrpGen",
            "benchmark_name": FAMILY,
            "num_customers": num_customers,
            "num_vehicles": None,
            "vehicle_capacity": int(cvrptw["capacity"]),
            "coordinates": coordinates,
            "demands": demands,
            "service_times": service_times,
            "depot": 0,
            "horizon": list(TD_HORIZON),
            "td": {
                "model": "road-graph",
                "graph": {"path": road_file, "sha256": road_sha},
                "traffic": dict(overlay_refs[sub]),
                "sample_step": road.sample_step,
                "simplify_tolerance": road.simplify_tolerance,
            },
            "metadata": {},
        }
        instance = td_instance_from_payload(probe_payload)
        anchor_arcs_by_sub[sub] = materialize_selected_atfs_roadgraph(
            instance, road, overlay, anchor_arc_keys
        )
        missing = anchor_arc_keys - anchor_arcs_by_sub[sub].keys()
        if missing:
            raise TDDerivationError(f"missing anchor arcs under {sub}: {sorted(missing)}")
        atfs = materialize_instance_atfs_roadgraph(instance, road, overlay)
        atf_sha[sub] = compute_atf_sha256(atfs)
        del atfs

    # 4. Shared deadline lift certifying the anchor route under every derived
    #    overlay. Earliest bounds are never reduced; zero-width windows raise.
    lifted, repairs = _audit_and_lift(
        provided_time_windows,
        service_times,
        [anchor_custs],
        anchor_arcs_by_sub,
        TD_HORIZON[1],
    )
    for customer in range(1, len(lifted)):
        earliest_before = provided_time_windows[customer][0]
        earliest_after, latest_after = lifted[customer]
        if earliest_after < earliest_before:
            raise TDDerivationError(
                f"customer {customer}: lifted earliest {earliest_after} below provided {earliest_before}"
            )
        if earliest_after >= latest_after:
            raise TDDerivationError(f"customer {customer}: non-positive lifted window width")
    lift_entries = [e for e in repairs.values() if "deadline_after" in e]
    tw_repair = {
        "policy": "anchor-minimal-shared-deadline-lift",
        "overlays_audited": subs,
        "lifted_customers": len(lift_entries),
        "max_lift_seconds": max(
            (e["deadline_after"] - e["deadline_before"] for e in lift_entries), default=0
        ),
        "anchor_route_source": anchor_source,
        "repairs": repairs,
    }

    # 5. Emit the twins (TDVRP without windows, TDVRPTW with the lifted set).
    generated_at = meta.get("generated_at") or manifest.get("generated_at", "")
    combos_out: list[dict[str, Any]] = []
    for m, i in combos:
        sub = subinstance_name(m, i)
        name = td_instance_name(base, m, i)
        td_metadata = {
            "authors": AUTHORS,
            "generated_at": generated_at,
            "city": city_slug,
            "method": method_tag,
            "base_instance_name": base,
            "subinstance": sub,
            "generator": _generator("derive-td", {"city": city_slug, "method": method_tag}),
            "traffic": traffic_info[sub],
            "notes": (
                "Time dependence derives from the city's OSM road network: the shared "
                "road-graph sidecar pins free-flow fastest paths for this base; this "
                "subinstance's traffic overlay carries per-edge hourly speeds; arrival-time "
                "functions are materialized deterministically on load and pinned by "
                "td.atf_sha256. Windows reuse the derive-vrptw set, lifted to time-dependent "
                "feasibility along the anchor route."
            ),
        }
        common = {
            "instance_name": name,
            "instance_origin": "OsmCvrpGen",
            "benchmark_name": FAMILY,
            "num_customers": num_customers,
            "num_vehicles": None,
            "vehicle_capacity": int(cvrptw["capacity"]),
            "coordinates": coordinates,
            "demands": demands,
            "service_times": service_times,
            "depot": 0,
            "reference_lla": reference_lla,
            "horizon": list(TD_HORIZON),
            "td": {
                "model": "road-graph",
                "graph": {"path": road_file, "sha256": road_sha},
                "traffic": dict(overlay_refs[sub]),
                "atf_sha256": atf_sha[sub],
                "sample_step": road.sample_step,
                "simplify_tolerance": road.simplify_tolerance,
            },
        }
        tdvrp_payload = dict(common)
        tdvrp_payload["metadata"] = {**td_metadata, "problem_type": "TDVRP"}
        tdvrptw_payload = dict(common)
        tdvrptw_payload["time_windows"] = [list(window) for window in lifted]
        tdvrptw_payload["metadata"] = {
            **td_metadata,
            "problem_type": "TDVRPTW",
            "tw_repair": tw_repair,
        }
        save_json_to_file(tdvrp_payload, twin_paths[(m, i)]["TDVRP"])
        save_json_to_file(tdvrptw_payload, twin_paths[(m, i)]["TDVRPTW"])
        combos_out.append(
            {
                "model": m,
                "intensity": i,
                "sub": sub,
                "traffic_sidecar": overlay_refs[sub]["path"],
                "traffic_sha256": overlay_sha[sub],
                "atf_sha256": atf_sha[sub],
                "tdvrptw_twin": twin_paths[(m, i)]["TDVRPTW"].name,
                "tdvrp_twin": twin_paths[(m, i)]["TDVRP"].name,
            }
        )

    return {
        "ok": True,
        "base": base,
        "folder": str(folder),
        "action": "derived",
        "num_customers": num_customers,
        "collection_marker": "mamut-collection.json",
        "road_sidecar": road_file,
        "road_sha256": road_sha,
        "anchor_route_source": anchor_source,
        "lifted_customers": len(lift_entries),
        "max_lift_seconds": tw_repair["max_lift_seconds"],
        "combos": combos_out,
    }
