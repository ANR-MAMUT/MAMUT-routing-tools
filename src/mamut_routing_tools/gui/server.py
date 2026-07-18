"""The loopback workbench server.

Endpoint shapes follow the retired Julia site_api.jl workbench API so the
copied workbench frontend keeps working unchanged. Security: loopback bind,
Host allow-list, and a Jupyter-style random URL token (cookie after first
visit); loopback alone does not stop DNS-rebinding against a server that
executes jobs and writes files.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from mamut_routing_tools.workspace import instances_dir, osmdata_dir

_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]")
STATIC_DIR = Path(__file__).parent / "static"


def _payload_error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message}, status_code=status)


def _city_slug(name: str) -> str:
    from mamut_routing_tools.generation.writers import slugify

    return slugify(name)


def _city_label(name: str) -> str:
    return name.replace("-", " ").replace("_", " ").strip()


def _request_to_generation(payload: dict[str, Any], workspace: Path) -> "Any":
    from mamut_routing_tools.generation.single import GenerationRequest

    city = str(payload.get("city") or "")
    osm_path = payload.get("osmPath")
    if osm_path:
        resolved = Path(str(osm_path))
        if not resolved.is_absolute():
            resolved = workspace / resolved
    else:
        resolved = _find_city_osm(city, workspace)
    request = GenerationRequest(city=city, osm_path=resolved)
    if payload.get("method"):
        request.method = str(payload["method"]).lower()
    if payload.get("nCustomers") is not None:
        request.n_customers = int(payload["nCustomers"])
    if payload.get("seed") is not None:
        request.seed = int(payload["seed"])
    if payload.get("demandType") is not None:
        request.demand_type = int(payload["demandType"])
    if payload.get("avgRouteSize") is not None:
        request.avg_route_size = int(payload["avgRouteSize"])
    if payload.get("depotMode"):
        request.depot_mode = str(payload["depotMode"]).lower()
    if payload.get("customerMode"):
        request.customer_mode = str(payload["customerMode"]).lower()
    if payload.get("clusterSeeds") is not None:
        request.cluster_seeds = int(payload["clusterSeeds"])
    if payload.get("clusterDecayMeters") is not None:
        request.cluster_decay_meters = float(payload["clusterDecayMeters"])
    if payload.get("categories"):
        raw = payload["categories"]
        request.categories = (
            [part.strip() for part in raw.split(",") if part.strip()] if isinstance(raw, str) else [str(v) for v in raw]
        )
    if payload.get("hybridPoiShare") is not None:
        request.hybrid_poi_share = float(payload["hybridPoiShare"])
    return request


def _find_city_osm(city: str, workspace: Path) -> Path:
    root = osmdata_dir(workspace, create=False)
    if not city:
        raise ValueError("Missing 'city'")
    slug = _city_slug(city)
    if root.is_dir():
        for candidate in sorted(root.glob("*.osm")):
            if _city_slug(candidate.stem) == slug:
                return candidate
    raise FileNotFoundError(f"No OSM extract for '{city}' under {root}. Fetch it from the Generate tab first.")


def _contained(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def create_app(workspace: Path, token: str) -> FastAPI:
    app = FastAPI(title="mamut-tools workbench", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def guard(request: Request, call_next):  # type: ignore[no-untyped-def]
        host = (request.headers.get("host") or "").split(":")[0]
        if host not in _ALLOWED_HOSTS:
            return _payload_error(403, "Host not allowed")
        origin = request.headers.get("origin")
        if origin and not any(f"//{allowed}" in origin for allowed in _ALLOWED_HOSTS):
            return _payload_error(403, "Origin not allowed")
        presented = (
            request.query_params.get("token")
            or request.headers.get("x-mamut-token")
            or request.cookies.get("mamut_token")
        )
        if presented != token:
            return _payload_error(403, "Missing or invalid token; open the URL printed by 'mamut-tools gui start'")
        response = await call_next(request)
        if request.query_params.get("token") == token:
            response.set_cookie("mamut_token", token, httponly=True, samesite="strict")
        return response

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "workspace": str(workspace)}

    @app.get("/api/workbench/generation/cities")
    async def generation_cities() -> dict[str, Any]:
        root = osmdata_dir(workspace, create=False)
        cities = []
        if root.is_dir():
            for path in sorted(root.glob("*.osm")):
                cities.append(
                    {
                        "slug": _city_slug(path.stem),
                        "label": _city_label(path.stem),
                        "customer_counts": [],
                        "osm_filename": path.name,
                        "osm_path": str(path),
                    }
                )
        return {
            "ok": True,
            "preview_available": bool(cities),
            "local_osmdata_dir": str(root),
            "cities": cities,
        }

    @app.get("/api/workbench/instances")
    async def workbench_instances() -> dict[str, Any]:
        """List generated instances already on disk (workspace instances/),
        so the GUI Visualize tab survives server restarts. CVRP manifests
        only: the VRPTW twin derivation manifests mark their base instead."""
        root = instances_dir(workspace, create=False)
        instances: list[dict[str, Any]] = []
        if root.is_dir():
            for manifest_path in sorted(root.rglob("*_manifest.json")):
                if manifest_path.name.endswith("_vrptw_manifest.json"):
                    continue
                try:
                    manifest = json.loads(manifest_path.read_text())
                    params = manifest.get("params") or {}
                    base_name = str(manifest["base_name"])
                    folder = manifest_path.parent
                    instances.append(
                        {
                            "base_name": base_name,
                            "folder": str(folder),
                            "files": manifest["files"],
                            "generated_at": manifest.get("generated_at"),
                            "city": params.get("city"),
                            "n_customers": params.get("n_customers"),
                            "seed": params.get("seed"),
                            "method": params.get("method"),
                            "summary": {
                                "capacity": manifest.get("capacity"),
                                "route_count": manifest.get("route_count"),
                                "customers": params.get("n_customers"),
                                "demand_type": manifest.get("demand_type"),
                                "avg_route_size": manifest.get("avg_route_size"),
                            },
                            "has_vrptw_twin": (folder / f"{base_name}_vrptw_manifest.json").is_file(),
                        }
                    )
                except (OSError, ValueError, KeyError):
                    continue
        instances.sort(key=lambda entry: str(entry.get("generated_at") or ""), reverse=True)
        return {"ok": True, "instances": instances}

    @app.post("/api/workbench/generation/fetch-osm-city")
    async def generation_fetch_city(request: Request) -> Any:
        payload = await request.json()
        from mamut_routing_tools.osm import fetch_and_store_city_osm

        try:
            result = fetch_and_store_city_osm(
                str(payload.get("city") or ""),
                country=str(payload.get("country") or ""),
                osm_dir=osmdata_dir(workspace),
                padding_km=float(payload.get("paddingKm") or 0.0),
                max_radius_km=float(payload.get("maxRadiusKm") or 0.0),
            )
        except Exception as error:  # noqa: BLE001 - surfaced as the API error contract
            return _payload_error(400, str(error))
        result["local_osmdata_dir"] = str(osmdata_dir(workspace, create=False))
        result["cities"] = [
            _city_slug(path.stem) for path in sorted(osmdata_dir(workspace, create=False).glob("*.osm"))
        ]
        return result

    async def _preview(request: Request) -> Any:
        payload = await request.json()
        from mamut_routing_tools.generation.single import build_generation_selection, preview_geojson

        try:
            generation_request = _request_to_generation(payload, workspace)
            selection = build_generation_selection(generation_request)
        except Exception as error:  # noqa: BLE001
            return _payload_error(400, str(error))
        tags = selection.source_tags[1:]
        poi_count = sum(1 for tag in tags if tag == "poi")
        return {
            "ok": True,
            "geojson": preview_geojson(selection),
            "summary": {
                "preview_mode": "osm",
                "city": selection.params["city"],
                "method": selection.params["method"],
                "customers": int(selection.params["n_customers"]),
                "poi_customers": poi_count,
                "parametric_customers": len(tags) - poi_count,
            },
        }

    app.post("/api/workbench/generation/preview")(_preview)
    app.post("/api/workbench/generation/generate")(_preview)

    @app.post("/api/workbench/generation/single")
    async def generation_single(request: Request) -> Any:
        payload = await request.json()
        from mamut_routing_tools.generation.single import generate_single_instance
        from mamut_routing_tools.generation.vrptw import derive_vrptw_from_cvrp

        try:
            generation_request = _request_to_generation(payload, workspace)
            result = generate_single_instance(generation_request, instances_dir(workspace))
            if payload.get("deriveVrptw"):
                result["vrptw"] = derive_vrptw_from_cvrp(
                    result["folder"],
                    result["base_name"],
                    tw_method=str(payload.get("twMethod") or "route_centered"),
                    place_slug=_city_slug(generation_request.city),
                    source_seed=generation_request.seed,
                )
        except Exception as error:  # noqa: BLE001
            return _payload_error(400, str(error))
        return result

    def _zip_folder_bases(folder: Path, bases: list[str]) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(folder.iterdir()):
                if any(path.name.startswith(base) for base in bases):
                    archive.write(path, path.name)
        return buffer.getvalue()

    @app.post("/api/workbench/generation/single-download")
    async def generation_single_download(request: Request) -> Any:
        payload = await request.json()
        folder = Path(str(payload.get("folder") or ""))
        base = str(payload.get("base_name") or payload.get("baseName") or "")
        if not base or not folder.is_dir() or not _contained(folder, instances_dir(workspace, create=False)):
            return _payload_error(400, "Unknown generated instance folder")
        data = _zip_folder_bases(folder, [base])
        return Response(
            content=data,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{base}.zip"'},
        )

    @app.post("/api/workbench/generation/bulk")
    async def generation_bulk(request: Request) -> Any:
        payload = await request.json()
        from mamut_routing_tools.generation.bulk import generate_bulk_instances

        try:
            base_request = _request_to_generation(payload, workspace)
            raw_cities = payload.get("cities") or [payload.get("city")]
            if isinstance(raw_cities, str):
                raw_cities = [part.strip() for part in raw_cities.split(",") if part.strip()]
            cities = [(str(name), _find_city_osm(str(name), workspace)) for name in raw_cities if name]

            def _int_list(key: str, fallback: int) -> list[int]:
                value = payload.get(key)
                if value is None:
                    return [fallback]
                if isinstance(value, str):
                    return [int(part) for part in value.split(",") if part.strip()]
                return [int(v) for v in value]

            result = generate_bulk_instances(
                base_request,
                cities=cities,
                n_list=_int_list("nCustomersList", base_request.n_customers),
                demand_types=_int_list("demandTypesList", base_request.demand_type),
                avg_route_sizes=_int_list("avgRouteSizesList", base_request.avg_route_size),
                output_root=instances_dir(workspace),
            )
        except Exception as error:  # noqa: BLE001
            return _payload_error(400, str(error))
        return result

    @app.post("/api/workbench/generation/bulk-download")
    async def generation_bulk_download(request: Request) -> Any:
        payload = await request.json()
        bases = [str(name) for name in (payload.get("base_names") or payload.get("baseNames") or [])]
        root = instances_dir(workspace, create=False)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file() and (not bases or any(path.name.startswith(base) for base in bases)):
                    archive.write(path, path.relative_to(root).as_posix())
        return Response(
            content=buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="mamut-generated-instances.zip"'},
        )

    @app.post("/api/workbench/generation/td-build")
    async def generation_td_build() -> Any:
        return _payload_error(
            501,
            "The official TD family pipeline stays in the MAMUT-routing repository for now: "
            "run 'mamut-routing-publish workbench build-family <City>' there.",
        )

    @app.post("/api/workbench/solve")
    async def workbench_solve(request: Request) -> Any:
        payload = await request.json()
        try:
            return _solve_payload(payload)
        except Exception as error:  # noqa: BLE001
            return _payload_error(400, str(error))

    @app.post("/api/workbench/render-routes")
    async def workbench_render_routes(request: Request) -> Any:
        payload = await request.json()
        try:
            return _render_routes_payload(payload, workspace)
        except Exception as error:  # noqa: BLE001
            return _payload_error(400, str(error))

    @app.get("/instances-file")
    async def instances_file(path: str) -> Any:
        target = Path(path)
        if not target.is_absolute():
            target = instances_dir(workspace, create=False) / target
        if not target.is_file() or not _contained(target, instances_dir(workspace, create=False)):
            return _payload_error(404, "Unknown generated artifact")
        return JSONResponse(json.loads(target.read_text(encoding="utf-8")))

    @app.get("/")
    async def index() -> Any:
        page = STATIC_DIR / "index.html"
        if page.is_file():
            return FileResponse(page)
        return _payload_error(404, "GUI frontend assets are missing")

    @app.get("/static/{asset_path:path}")
    async def static_asset(asset_path: str) -> Any:
        target = STATIC_DIR / asset_path
        if not target.is_file() or not _contained(target, STATIC_DIR):
            return _payload_error(404, "Unknown asset")
        return FileResponse(target)

    return app


def _solve_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """PyVRP-backed port of the workbench solve endpoint: accepts TSPLIB
    FULL_MATRIX text or a vrp_json object, returns the same response shape
    (1-based customer indices in routes)."""
    import tempfile

    from mamut_routing_lib.artifacts import load_benchmark_instance
    from mamut_routing_lib.solvers.pyvrp import solve_instance

    time_limit = float(payload.get("time_limit") or 30.0)
    if not (time_limit > 0):
        raise ValueError("'time_limit' must be a positive number")

    vrp_json = payload.get("vrp_json")
    vrp_text = payload.get("vrp_text")
    if vrp_json is not None:
        input_source = "vrp_json"
        with tempfile.NamedTemporaryFile("w", suffix=".vrp.json", delete=False) as handle:
            json.dump(vrp_json, handle)
            instance_file = Path(handle.name)
        try:
            instance = load_benchmark_instance(instance_file)
        finally:
            instance_file.unlink(missing_ok=True)
    elif vrp_text:
        input_source = "vrp_text"
        instance = _instance_from_vrp_text(str(vrp_text))
    else:
        raise ValueError("Missing required 'vrp_text' or 'vrp_json' field")

    result = solve_instance(instance, time_limit_s=max(1, int(time_limit)), seed=0)
    return {
        "ok": result.solver_is_feasible,
        "cost": result.solver_cost,
        "time": round(result.wall_time, 3),
        "routes": result.routes,
        "n_routes": result.route_count,
        "dimension": instance.num_customers + 1,
        "capacity": instance.vehicle_capacity,
        "input_source": input_source,
        "method": result.method,
    }


def _instance_from_vrp_text(vrp_text: str) -> Any:
    import tempfile

    from mamut_routing_lib.models import BenchmarkInstanceCVRP

    from mamut_routing_tools.generation.writers import parse_cvrp_vrp

    with tempfile.NamedTemporaryFile("w", suffix=".vrp", delete=False) as handle:
        handle.write(vrp_text)
        path = Path(handle.name)
    try:
        parsed = parse_cvrp_vrp(path)
    finally:
        path.unlink(missing_ok=True)
    return BenchmarkInstanceCVRP(
        instance_name=parsed.name or "uploaded",
        instance_origin="OsmCvrpGen",
        benchmark_name="Mamut2026",
        num_customers=parsed.dimension - 1,
        vehicle_capacity=parsed.capacity,
        coordinates=parsed.coordinates,
        demands=parsed.demands,
        depot=parsed.depot_node_index - 1,
        arc_costs=parsed.arc_costs,
        metadata={},
    )


def _render_routes_payload(payload: dict[str, Any], workspace: Path) -> dict[str, Any]:
    """Road polylines for arbitrary routes over a workbench meta object:
    cache-first on meta.road_cache, live Dijkstra fill on the meta's OSM
    graph for shortest/fastest, straight lines otherwise."""
    from mamut_routing_tools.geometry.materialize import (
        candidate_route_segment,
        node_coordinates_map,
        resolve_source_osm_path,
    )
    from mamut_routing_tools.roadgraph.build import road_graph_candidates

    metric = str(payload.get("metric") or "shortest")
    if metric not in ("shortest", "fastest", "euclidean"):
        raise ValueError(f"Unsupported metric '{metric}'")
    routes = payload.get("routes")
    if not isinstance(routes, list) or not routes:
        raise ValueError("Request must contain 'routes'")
    routes = [[int(stop) for stop in route] for route in routes]
    meta = payload.get("meta")
    if meta is None:
        raise ValueError("Missing required 'meta' object")

    nodes = {int(node["instance_node_id"]): node for node in meta.get("nodes", [])}
    depot_id = int(meta.get("depot_instance_node_id") or 1)

    def node_lonlat(instance_node_id: int) -> list[float]:
        node = nodes[instance_node_id]
        return [float(node["poi_lon"]), float(node["poi_lat"])]

    road_cache = (meta.get("road_cache") or {}).get(metric) or {}
    candidates = None
    if metric in ("shortest", "fastest"):
        try:
            osm_path = resolve_source_osm_path(workspace, meta, "meta")
            candidates = [
                (graph, _graph_vertex_map_cached(graph, meta))
                for graph in road_graph_candidates(
                    osm_path,
                    only_intersections=bool((meta.get("map_options") or {}).get("only_intersections", True)),
                    trim_to_connected=bool((meta.get("map_options") or {}).get("trim_to_connected_graph", True)),
                )
            ]
        except Exception:  # noqa: BLE001 - degrade to cache + straight lines
            candidates = None

    features = []
    used_cache = False
    cache_miss_count = 0
    render_modes: set[str] = set()
    for route_index, route in enumerate(routes):
        sequence = [depot_id, *route, depot_id]
        coordinates: list[list[float]] = []
        segment_modes: set[str] = set()
        for i in range(len(sequence) - 1):
            from_id, to_id = sequence[i], sequence[i + 1]
            from_node, to_node = nodes[from_id], nodes[to_id]
            segment = None
            if metric in ("shortest", "fastest"):
                key = f"{from_node['graph_vertex_id']}_{to_node['graph_vertex_id']}"
                cached = road_cache.get(key)
                if cached:
                    segment = [list(map(float, point)) for point in cached]
                    segment_modes.add("cached_road")
                elif candidates:
                    segment = candidate_route_segment(
                        candidates,
                        from_id,
                        to_id,
                        node_lonlat(from_id),
                        node_lonlat(to_id),
                        metric,
                    )
                    if segment is not None:
                        segment_modes.add("cached_road")
            if segment is None:
                segment = [node_lonlat(from_id), node_lonlat(to_id)]
                segment_modes.add("straight_line")
                if metric in ("shortest", "fastest"):
                    cache_miss_count += 1
            if coordinates:
                coordinates.extend(segment[1:])
            else:
                coordinates.extend(segment)
        if segment_modes == {"cached_road"}:
            render_mode = "cached_road"
            used_cache = True
        elif "cached_road" in segment_modes:
            render_mode = "mixed"
            used_cache = True
        else:
            render_mode = "straight_line"
        render_modes.add(render_mode)
        load = sum(int(nodes[stop].get("demand") or 0) for stop in route)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coordinates},
                "properties": {
                    "route_index": route_index + 1,
                    "stops": len(route),
                    "load": load,
                    "metric": metric,
                    "render_mode": render_mode,
                },
            }
        )

    overall = (
        "mixed" if "mixed" in render_modes else "cached_road" if "cached_road" in render_modes else "straight_line"
    )
    return {
        "ok": True,
        "geojson": {"type": "FeatureCollection", "features": features},
        "summary": {
            "metric": metric,
            "route_count": len(routes),
            "render_mode": overall,
            "used_cache": used_cache,
            "cache_miss_count": cache_miss_count,
            "straight_fallback_count": cache_miss_count,
        },
    }


_VERTEX_MAP_CACHE: dict[tuple[int, int], dict[int, int]] = {}


def _graph_vertex_map_cached(graph: Any, meta: dict[str, Any]) -> dict[int, int]:
    from mamut_routing_tools.geometry.materialize import _graph_vertex_map, node_coordinates_map

    key = (id(graph), id(meta))
    if key not in _VERTEX_MAP_CACHE:
        _VERTEX_MAP_CACHE[key] = _graph_vertex_map(graph, node_coordinates_map(meta))
    return _VERTEX_MAP_CACHE[key]
