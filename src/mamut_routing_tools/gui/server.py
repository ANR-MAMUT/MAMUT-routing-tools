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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from mamut_routing_tools.gui.jobs import JobContext, JobManager
from mamut_routing_tools.gui.solutions import (
    SolutionStore,
    compare_solution_records,
    instance_id_for,
    validate_solution,
)
from mamut_routing_tools.workspace import instances_dir, osmdata_dir

_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]")
STATIC_DIR = Path(__file__).parent / "static"


class JobSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["fetch-osm", "generate", "bulk-generate", "solve"]
    payload: dict[str, Any] = Field(default_factory=dict)


class SolutionComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_run_id: str
    reference_run_id: str


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
        depot_mode = str(payload["depotMode"]).lower()
        request.depot_mode = {"centered": "center", "excentered": "corner"}.get(depot_mode, depot_mode)
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


def _workspace_instances(workspace: Path) -> list[dict[str, Any]]:
    """Manifest-backed generated instances with stable opaque GUI ids."""

    root = instances_dir(workspace, create=False)
    records: list[dict[str, Any]] = []
    if not root.is_dir():
        return records
    for manifest_path in sorted(root.rglob("*_manifest.json")):
        if manifest_path.name.endswith("_vrptw_manifest.json"):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            params = manifest.get("params") or {}
            base_name = str(manifest["base_name"])
            folder = manifest_path.parent
            records.append(
                {
                    "instance_id": instance_id_for(folder, base_name),
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
    records.sort(key=lambda entry: str(entry.get("generated_at") or ""), reverse=True)
    return records


def _workspace_instance(workspace: Path, instance_id: str) -> dict[str, Any]:
    for record in _workspace_instances(workspace):
        if record["instance_id"] == instance_id:
            return record
    raise KeyError(instance_id)


def _instance_variant_path(record: dict[str, Any], metric: str) -> Path:
    variants = record.get("files", {}).get("vrp_json", {})
    filename = variants.get(metric)
    if not filename:
        raise ValueError(f"Instance has no '{metric}' JSON variant")
    folder = Path(record["folder"])
    path = folder / str(filename)
    if not path.is_file() or not _contained(path, folder):
        raise FileNotFoundError(path)
    return path


def _instance_map_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Small, geometry-only view of instance metadata for the GUI map."""

    folder = Path(record["folder"])
    meta_path = folder / str(record["files"]["meta"])
    if not meta_path.is_file() or not _contained(meta_path, folder):
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    depot_id = int(meta.get("depot_instance_node_id") or 1)
    raw_nodes = sorted(meta.get("nodes") or [], key=lambda node: int(node["instance_node_id"]))
    features = []
    for model_node_id, node in enumerate(raw_nodes):
        metadata_node_id = int(node["instance_node_id"])
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(node["poi_lon"]), float(node["poi_lat"])],
                },
                "properties": {
                    "instance_node_id": metadata_node_id,
                    "model_node_id": model_node_id,
                    "role": "depot" if metadata_node_id == depot_id else "customer",
                    "demand": int(node.get("demand") or 0),
                    "source_tag": str(node.get("source_tag") or "unknown"),
                },
            }
        )
    return {
        "ok": True,
        "instance_id": record["instance_id"],
        "base_name": record["base_name"],
        "depot_instance_node_id": depot_id,
        "geojson": {"type": "FeatureCollection", "features": features},
    }


def _routes_for_metadata(meta: dict[str, Any], routes: list[list[int]]) -> list[list[int]]:
    """Map normalized library node indexes onto metadata's persisted IDs.

    Generated metadata uses one-based ``instance_node_id`` values, while the
    library solver uses array positions (depot 0, customers 1..n). Historical
    callers may already provide metadata IDs, so complete customer sets make
    the convention explicit without guessing route by route.
    """

    ordered_ids = [
        int(node["instance_node_id"])
        for node in sorted(meta.get("nodes") or [], key=lambda node: int(node["instance_node_id"]))
    ]
    depot_metadata_id = int(meta.get("depot_instance_node_id") or 1)
    if depot_metadata_id not in ordered_ids:
        raise ValueError("Instance metadata does not contain its depot node")
    model_depot_id = ordered_ids.index(depot_metadata_id)
    model_to_metadata = dict(enumerate(ordered_ids))
    metadata_customers = set(ordered_ids) - {depot_metadata_id}
    model_customers = set(model_to_metadata) - {model_depot_id}
    stops = {stop for route in routes for stop in route}

    if stops == model_customers or (stops <= model_customers and not stops <= metadata_customers):
        return [[model_to_metadata[stop] for stop in route] for route in routes]
    if stops <= metadata_customers:
        return routes
    raise ValueError("Solution routes contain node IDs that do not match the instance metadata")


def _fetch_osm_payload(payload: dict[str, Any], workspace: Path, context: JobContext | None = None) -> dict[str, Any]:
    from mamut_routing_tools.generation.pois import POI_CATEGORIES
    from mamut_routing_tools.osm import fetch_and_store_city_osm

    def report_progress(event: dict[str, Any]) -> None:
        if context is None:
            return
        context.check_cancelled()
        context.progress(
            f"Downloading {event.get('phase', 'OSM')} tiles",
            current=int(event.get("current") or 0),
            total=int(event.get("total") or 0),
        )

    if context is not None:
        context.progress("Geocoding city")
    profile = str(payload.get("profile") or "generation")
    # A GUI city extract is a reusable local asset. Fetch the whole category
    # catalog once so changing the generation filter later never requires the
    # user to remember which boxes were checked during acquisition.
    poi_categories = list(POI_CATEGORIES) if profile == "generation" else None
    result = fetch_and_store_city_osm(
        str(payload.get("city") or ""),
        country=str(payload.get("country") or ""),
        osm_dir=osmdata_dir(workspace),
        padding_km=float(payload.get("paddingKm") or 0.0),
        max_radius_km=float(payload.get("maxRadiusKm") or 0.0),
        profile=profile,
        poi_categories=poi_categories,
        progress=report_progress,
    )
    result["local_osmdata_dir"] = str(osmdata_dir(workspace, create=False))
    result["cities"] = [
        _city_slug(path.stem) for path in sorted(osmdata_dir(workspace, create=False).glob("*.osm"))
    ]
    return result


def _generate_single_payload(
    payload: dict[str, Any], workspace: Path, context: JobContext | None = None
) -> dict[str, Any]:
    from mamut_routing_tools.generation.single import generate_single_instance
    from mamut_routing_tools.generation.vrptw import derive_vrptw_from_cvrp

    if context is not None:
        context.progress("Selecting customers and materializing matrices")
        context.check_cancelled()
    generation_request = _request_to_generation(payload, workspace)
    result = generate_single_instance(generation_request, instances_dir(workspace))
    if payload.get("deriveVrptw"):
        if context is not None:
            context.progress("Deriving VRPTW twin")
            context.check_cancelled()
        result["vrptw"] = derive_vrptw_from_cvrp(
            result["folder"],
            result["base_name"],
            tw_method=str(payload.get("twMethod") or "route_centered"),
            place_slug=_city_slug(generation_request.city),
            source_seed=generation_request.seed,
        )
    result["instance_id"] = instance_id_for(Path(result["folder"]), result["base_name"])
    return result


def _generate_bulk_payload(
    payload: dict[str, Any], workspace: Path, context: JobContext | None = None
) -> dict[str, Any]:
    from mamut_routing_tools.generation.bulk import generate_bulk_instances

    if context is not None:
        context.progress("Preparing bulk generation")
        context.check_cancelled()
    base_request = _request_to_generation(payload, workspace)
    raw_cities = payload.get("cities") or [payload.get("city")]
    if isinstance(raw_cities, str):
        raw_cities = [part.strip() for part in raw_cities.split(",") if part.strip()]
    cities = [(str(name), _find_city_osm(str(name), workspace)) for name in raw_cities if name]

    def int_list(key: str, fallback: int) -> list[int]:
        value = payload.get(key)
        if value is None:
            return [fallback]
        if isinstance(value, str):
            return [int(part) for part in value.split(",") if part.strip()]
        return [int(item) for item in value]

    result = generate_bulk_instances(
        base_request,
        cities=cities,
        n_list=int_list("nCustomersList", base_request.n_customers),
        demand_types=int_list("demandTypesList", base_request.demand_type),
        avg_route_sizes=int_list("avgRouteSizesList", base_request.avg_route_size),
        output_root=instances_dir(workspace),
    )
    for item in result.get("results", []):
        item["instance_id"] = instance_id_for(Path(item["folder"]), item["base_name"])
    return result


def create_app(workspace: Path, token: str) -> FastAPI:
    jobs = JobManager(workspace)
    solutions = SolutionStore(workspace)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
        yield
        jobs.shutdown()

    app = FastAPI(
        title="mamut-tools workbench",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.job_manager = jobs
    app.state.solution_store = solutions

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
        records = _workspace_instances(workspace)
        for record in records:
            record["solution_count"] = len(solutions.list(str(record["instance_id"])))
        return {"ok": True, "instances": records}

    @app.post("/api/workbench/generation/fetch-osm-city")
    async def generation_fetch_city(request: Request) -> Any:
        payload = await request.json()
        try:
            return _fetch_osm_payload(payload, workspace)
        except Exception as error:  # noqa: BLE001 - surfaced as the API error contract
            return _payload_error(400, str(error))

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
        try:
            return _generate_single_payload(payload, workspace)
        except Exception as error:  # noqa: BLE001
            return _payload_error(400, str(error))

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
        try:
            return _generate_bulk_payload(payload, workspace)
        except Exception as error:  # noqa: BLE001
            return _payload_error(400, str(error))

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

    def submit_job(submission: JobSubmission) -> dict[str, Any]:
        payload = submission.payload

        def runner(context: JobContext) -> dict[str, Any]:
            context.log(f"Request: {json.dumps(payload, sort_keys=True)}")
            if submission.kind == "fetch-osm":
                return _fetch_osm_payload(payload, workspace, context)
            if submission.kind == "generate":
                return _generate_single_payload(payload, workspace, context)
            if submission.kind == "bulk-generate":
                return _generate_bulk_payload(payload, workspace, context)
            if submission.kind == "solve":
                instance_id = str(payload.get("instance_id") or "")
                if not instance_id:
                    raise ValueError("A solve job requires 'instance_id'")
                context.progress("Loading instance")
                return _solve_workspace_payload(
                    workspace,
                    instance_id,
                    payload,
                    solutions,
                    context=context,
                )
            raise ValueError(f"Unsupported job kind '{submission.kind}'")

        return jobs.submit(submission.kind, payload, runner)

    @app.post("/api/jobs")
    async def create_job(submission: JobSubmission) -> dict[str, Any]:
        return {"ok": True, "job": submit_job(submission)}

    @app.get("/api/jobs")
    async def list_jobs(limit: int = 50) -> dict[str, Any]:
        return {"ok": True, "jobs": jobs.list(limit=limit)}

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> Any:
        try:
            return {"ok": True, "job": jobs.get(job_id)}
        except KeyError:
            return _payload_error(404, "Unknown job")

    @app.get("/api/jobs/{job_id}/log")
    async def get_job_log(job_id: str) -> Any:
        try:
            return {"ok": True, "job_id": job_id, "log": jobs.log_text(job_id)}
        except KeyError:
            return _payload_error(404, "Unknown job")

    @app.delete("/api/jobs/{job_id}")
    async def cancel_job(job_id: str) -> Any:
        try:
            return {"ok": True, "job": jobs.cancel(job_id)}
        except KeyError:
            return _payload_error(404, "Unknown job")

    @app.get("/api/instances/{instance_id}/solutions")
    async def list_solution_runs(instance_id: str) -> Any:
        try:
            record = _workspace_instance(workspace, instance_id)
            references = _bks_references(record)
        except KeyError:
            return _payload_error(404, "Unknown workspace instance")
        except Exception as error:  # noqa: BLE001
            return _payload_error(400, str(error))
        return {
            "ok": True,
            "instance_id": instance_id,
            "runs": solutions.list(instance_id),
            "references": references,
        }

    @app.get("/api/instances/{instance_id}/map-data")
    async def instance_map_data(instance_id: str) -> Any:
        try:
            return _instance_map_payload(_workspace_instance(workspace, instance_id))
        except KeyError:
            return _payload_error(404, "Unknown workspace instance")
        except Exception as error:  # noqa: BLE001
            return _payload_error(400, str(error))

    @app.post("/api/instances/{instance_id}/solutions/compare")
    async def compare_solution_runs(instance_id: str, request: SolutionComparisonRequest) -> Any:
        try:
            record = _workspace_instance(workspace, instance_id)
            candidate = _solution_or_reference(solutions, record, instance_id, request.candidate_run_id)
            reference = _solution_or_reference(solutions, record, instance_id, request.reference_run_id)
            instance_path = Path(str(candidate.get("instance_path") or reference.get("instance_path") or ""))
            from mamut_routing_lib.artifacts import load_benchmark_instance
            from mamut_routing_lib.solvers.pyvrp import hydrate_collection_instance

            instance = load_benchmark_instance(instance_path)
            checkable = hydrate_collection_instance(instance, instance_path)
            comparison = compare_solution_records(checkable, candidate, reference)
            return {"ok": True, "comparison": comparison}
        except KeyError:
            return _payload_error(404, "Unknown instance or solution run")
        except Exception as error:  # noqa: BLE001
            return _payload_error(400, str(error))

    @app.post("/api/instances/{instance_id}/solutions/{run_id}/render")
    async def render_solution_run(instance_id: str, run_id: str, metric: str = "fastest") -> Any:
        try:
            record = _workspace_instance(workspace, instance_id)
            solution = _solution_or_reference(solutions, record, instance_id, run_id)
            folder = Path(record["folder"])
            meta_path = folder / str(record["files"]["meta"])
            if not meta_path.is_file() or not _contained(meta_path, folder):
                raise FileNotFoundError(meta_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return _render_routes_payload(
                {"meta": meta, "routes": solution["routes"], "metric": metric}, workspace
            )
        except KeyError:
            return _payload_error(404, "Unknown instance or solution run")
        except Exception as error:  # noqa: BLE001
            return _payload_error(400, str(error))

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


def _solve_payload(payload: dict[str, Any], *, instance_path: Path | None = None) -> dict[str, Any]:
    """Solve and canonically validate a static CVRP/VRPTW instance.

    The compatibility endpoint still accepts inline TSPLIB/JSON.  Persistent
    jobs pass ``instance_path`` so collection sidecars and provenance remain
    available to the solver and checker.
    """
    import tempfile

    from mamut_routing_lib.artifacts import load_benchmark_instance
    from mamut_routing_lib.enums import ObjectiveFunction
    from mamut_routing_lib.solvers.pyvrp import solve_instance

    time_limit = float(payload.get("time_limit") or 30.0)
    if not (time_limit > 0):
        raise ValueError("'time_limit' must be a positive number")
    seed = int(payload.get("seed") if payload.get("seed") is not None else 0)
    objective = ObjectiveFunction(str(payload.get("objective_function") or "MonoCost"))

    temporary_path: Path | None = None
    try:
        if instance_path is not None:
            input_source = "workspace"
            instance_file = instance_path
            instance = load_benchmark_instance(instance_file)
        else:
            vrp_json = payload.get("vrp_json")
            vrp_text = payload.get("vrp_text")
            if vrp_json is not None:
                input_source = "vrp_json"
                with tempfile.NamedTemporaryFile("w", suffix=".vrp.json", delete=False) as handle:
                    json.dump(vrp_json, handle)
                    temporary_path = Path(handle.name)
                instance_file = temporary_path
                instance = load_benchmark_instance(instance_file)
            elif vrp_text:
                input_source = "vrp_text"
                instance_file = Path("uploaded.vrp.json")
                instance = _instance_from_vrp_text(str(vrp_text))
            else:
                raise ValueError("Missing required 'vrp_text' or 'vrp_json' field")

        result = solve_instance(
            instance,
            time_limit_s=max(1, int(time_limit)),
            seed=seed,
            objective_function=objective,
            instance_path=instance_file if input_source != "vrp_text" else None,
        )
        validation = validate_solution(instance, result.routes, instance_path=instance_file)
        canonical_cost = validation.get("routing_cost")
        return {
            "ok": bool(result.solver_is_feasible and validation["valid"]),
            "cost": canonical_cost if canonical_cost is not None else result.solver_cost,
            "solver_cost": result.solver_cost,
            "time": round(result.wall_time, 3),
            "routes": result.routes,
            "n_routes": result.route_count,
            "dimension": instance.num_customers + 1,
            "capacity": instance.vehicle_capacity,
            "input_source": input_source,
            "method": result.method,
            "objective_function": result.objective_function,
            "seed": result.seed,
            "time_limit_s": result.time_limit_s,
            "validation": validation,
            "metadata": result.metadata,
        }
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _solve_workspace_payload(
    workspace: Path,
    instance_id: str,
    payload: dict[str, Any],
    store: SolutionStore,
    *,
    context: JobContext | None = None,
) -> dict[str, Any]:
    record = _workspace_instance(workspace, instance_id)
    metric = str(payload.get("metric") or "fastest").lower()
    instance_path = _instance_variant_path(record, metric)
    if context is not None:
        context.progress(f"Solving {record['base_name']} ({metric})")
        context.check_cancelled()
    result = _solve_payload(payload, instance_path=instance_path)
    if context is not None:
        context.progress("Persisting validated solution")
        context.check_cancelled()
    solution = store.record(
        instance_id=instance_id,
        instance_name=str(record["base_name"]),
        instance_path=instance_path,
        routes=result["routes"],
        cost=result["cost"],
        objective_function=str(result["objective_function"]),
        solver="pyvrp",
        method=str(result["method"]),
        seed=int(result["seed"]),
        time_limit_s=int(result["time_limit_s"]),
        wall_time_s=float(result["time"]),
        validation=result["validation"],
        metadata={**(result.get("metadata") or {}), "metric": metric},
        job_id=context.job_id if context is not None else None,
    )
    return {**result, "instance_id": instance_id, "metric": metric, "solution": solution}


def _bks_references(record: dict[str, Any]) -> list[dict[str, Any]]:
    from mamut_routing_lib.artifacts import get_bks_path_for_instance, load_benchmark_instance, load_bks
    from mamut_routing_lib.enums import ObjectiveFunction

    references: list[dict[str, Any]] = []
    for metric in ("fastest", "shortest", "euclidean"):
        try:
            instance_path = _instance_variant_path(record, metric)
        except (ValueError, FileNotFoundError):
            continue
        instance = load_benchmark_instance(instance_path)
        for objective in (ObjectiveFunction.MONO_COST, ObjectiveFunction.HIERARCHICAL_VEHICLE_COST):
            path = get_bks_path_for_instance(instance_path, objective)
            if not path.is_file():
                continue
            bks = load_bks(path)
            validation = validate_solution(instance, bks.routes, instance_path=instance_path)
            references.append(
                {
                    "schema_version": 1,
                    "run_id": f"bks:{objective.value}:{metric}",
                    "instance_id": record["instance_id"],
                    "instance_name": record["base_name"],
                    "instance_path": str(instance_path),
                    "created_at": None,
                    "source": "bks",
                    "solver": "published-bks",
                    "method": str(bks.metadata.get("method") or "BKS"),
                    "objective_function": objective.value,
                    "seed": None,
                    "time_limit_s": None,
                    "wall_time_s": None,
                    "cost": validation.get("routing_cost") if validation["valid"] else bks.cost,
                    "num_routes": len(bks.routes),
                    "routes": bks.routes,
                    "validation": validation,
                    "metadata": {**bks.metadata, "metric": metric, "bks_path": str(path)},
                    "job_id": None,
                }
            )
    return references


def _solution_or_reference(
    store: SolutionStore,
    record: dict[str, Any],
    instance_id: str,
    run_id: str,
) -> dict[str, Any]:
    if run_id.startswith("bks:"):
        for reference in _bks_references(record):
            if reference["run_id"] == run_id:
                return reference
        raise KeyError(run_id)
    return store.get(instance_id, run_id)


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
    routes = _routes_for_metadata(meta, routes)

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
