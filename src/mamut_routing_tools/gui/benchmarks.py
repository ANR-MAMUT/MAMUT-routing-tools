"""Published benchmark collections as read-only instances of the workbench.

The workbench was built around instances generated into its workspace. A
published geographic collection (Mamut2026, Poryos2026) has everything the
same screens need -- the slim ``.vrp.json`` of each metric variant, the BKS
next to it, a geo sidecar with the node positions and a road-graph sidecar --
only laid out differently. This module reads a benchmarks folder and presents
each static instance as a record of the shape the server already works with,
so importing, checking, drawing, comparing and animating a solution need no
second code path.

Nothing is ever written into the benchmarks folder: runs live in the
workspace, under an id derived from the family, problem and instance name.

Scope: static CVRP and VRPTW instances of family-first collections (layout 1,
the one with a metric slot and sidecars). Time-dependent instances and classic
families are left out.
"""

from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

BENCHMARK_ID_PREFIX = "bench-"
#: Set by ``mamut-tools gui start/run --benchmarks-dir`` for the server process.
BENCHMARKS_DIR_ENV = "MAMUT_GUI_BENCHMARKS_DIR"
METRICS = ("fastest", "shortest", "euclidean")
_STATIC_PROBLEMS = ("CVRP", "VRPTW")


def is_benchmark_id(instance_id: str) -> bool:
    return str(instance_id).startswith(BENCHMARK_ID_PREFIX)


def benchmark_instance_id(family: str, problem: str, instance_name: str) -> str:
    """Stable across checkouts: it names the instance, not where it sits on disk."""
    identity = f"{family}\0{problem}\0{instance_name}".encode()
    return BENCHMARK_ID_PREFIX + hashlib.sha256(identity).hexdigest()[:20]


def _has_collections(directory: Path) -> bool:
    from mamut_routing_lib.sidecars import COLLECTION_MARKER_FILENAME

    return (directory / COLLECTION_MARKER_FILENAME).is_file() or any(directory.glob(f"*/{COLLECTION_MARKER_FILENAME}"))


def resolve_benchmarks_root(explicit: str | Path | None = None, *, start: Path | None = None) -> Path | None:
    """The folder holding the collections, or ``None`` when there is none to read.

    1. An explicit path (``--benchmarks-dir``, or ``MAMUT_GUI_BENCHMARKS_DIR``).
    2. The library's convention: ``MAMUT_ROUTING_BENCHMARKS_ROOT``, else
       ``MAMUT_ROUTING_ROOT/benchmarks``.
    3. A ``benchmarks`` folder with collections at or above the current
       directory: the GUI started inside a MAMUT-routing checkout.
    """
    candidates: list[Path] = []
    for value in (explicit, os.environ.get(BENCHMARKS_DIR_ENV), os.environ.get("MAMUT_ROUTING_BENCHMARKS_ROOT")):
        if value:
            candidates.append(Path(value).expanduser())
    if os.environ.get("MAMUT_ROUTING_ROOT"):
        candidates.append(Path(os.environ["MAMUT_ROUTING_ROOT"]).expanduser() / "benchmarks")
    if candidates:
        # A configured folder is used as given, also when it is wrong: the listing then says so.
        return candidates[0].resolve()
    origin = (start or Path.cwd()).resolve()
    for directory in (origin, *origin.parents):
        candidate = directory / "benchmarks"
        if candidate.is_dir() and _has_collections(candidate):
            return candidate
    return None


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as handle:
        return json.loads(handle.read())


class SidecarRoads:
    """Road polylines between the nodes of one instance, from its road-graph sidecar.

    A generated instance draws its routes on the city's OSM extract; a published
    one ships the road graph its distances were computed on instead. Paths use
    the pinned Dijkstra of ``mamut_routing_lib.td.roadgraph`` (distance then
    vertex id on the heap, edges in list order, strict improvement), stopped as
    soon as the target is settled, so the line drawn for an arc is the path its
    published cost was measured on.
    """

    def __init__(self, path: Path):
        from mamut_routing_lib.td.roadgraph import build_adjacency, load_instance_road_graph

        self.road = load_instance_road_graph(path)
        self.adjacency = build_adjacency(self.road)
        self._segments: dict[tuple[int, int, str], list[list[float]] | None] = {}

    def segment(self, from_index: int, to_index: int, metric: str) -> list[list[float]] | None:
        """``[lon, lat]`` points of the road path from node ``from_index`` to node ``to_index``.

        Indices are model node indices (depot 0). ``None`` when the metric has no
        road path (euclidean) or the target cannot be reached.
        """
        if metric not in ("fastest", "shortest"):
            return None
        key = (from_index, to_index, metric)
        if key not in self._segments:
            vertices = self.road.node_vertices
            path = self._path(vertices[from_index], vertices[to_index], fastest=metric == "fastest")
            lonlat = self.road.vertex_lonlat
            self._segments[key] = None if path is None else [[float(lonlat[v][0]), float(lonlat[v][1])] for v in path]
        return self._segments[key]

    def _path(self, source: int, target: int, *, fastest: bool) -> list[int] | None:
        if source == target:
            return [source]
        edges = self.road.edges
        infinity = float("inf")
        dist: dict[int, float] = {source: 0.0}
        pred: dict[int, int] = {}
        heap: list[tuple[float, int]] = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, infinity):
                continue
            if u == target:
                break
            for edge_index in self.adjacency[u]:
                _, v, length, speed_limit = edges[edge_index]
                candidate = d + (length / speed_limit if fastest else length)
                if candidate < dist.get(v, infinity):
                    dist[v] = candidate
                    pred[v] = u
                    heapq.heappush(heap, (candidate, v))
        if target not in pred:
            return None
        path = [target]
        while path[-1] != source:
            path.append(pred[path[-1]])
        path.reverse()
        return path


class BenchmarkCatalog:
    """The static instances of the collections under one benchmarks folder."""

    def __init__(self, root: Path | None):
        self.root = root
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] | None = None
        self._error: str | None = None
        self._meta: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()
        self._roads: OrderedDict[tuple[str, int], SidecarRoads] = OrderedDict()

    # ── Discovery ──

    def _scan(self) -> None:
        from mamut_routing_lib.artifacts import discover_benchmark_instances, find_collection_roots
        from mamut_routing_lib.enums import ProblemType
        from mamut_routing_lib.sidecars import COLLECTION_MARKER_FILENAME, load_collection_marker

        records: dict[str, dict[str, Any]] = {}
        self._error = None
        if self.root is None:
            self._records = records
            return
        if not self.root.is_dir():
            self._error = f"The benchmarks folder {self.root} does not exist."
            self._records = records
            return
        try:
            roots = find_collection_roots(self.root)
            # Layout 1 is the family-first shape with metric variants and sidecars.
            families = {
                family: root for root, family in roots.items()
                if load_collection_marker(root / COLLECTION_MARKER_FILENAME).layout_version == 1
            }
            found = discover_benchmark_instances(
                self.root,
                problem_types=[ProblemType(problem) for problem in _STATIC_PROBLEMS],
                benchmark_names=sorted(families),
            ) if families else []
        except Exception as error:  # noqa: BLE001 - a malformed checkout must not take the GUI down
            self._error = f"Cannot read the benchmarks folder {self.root}: {error}"
            self._records = records
            return
        for item in found:
            if item.metric_variant is None or not item.base_instance_name:
                continue
            family = str(item.benchmark_name)
            problem = item.problem_type.value
            collection_root = families[family]
            instance_id = benchmark_instance_id(family, problem, item.instance_name)
            record = records.setdefault(instance_id, {
                "instance_id": instance_id,
                "source": "benchmark",
                "family": family,
                "problem_type": problem,
                "base_name": item.instance_name,
                "base_instance_name": item.base_instance_name,
                "tw_set": item.tw_set,
                "city": item.place_slug,
                "n_customers": item.num_customers,
                "folder": str(collection_root),
                "files": {"vrp_json": {}},
                "summary": {},
                "has_vrptw_twin": False,
            })
            record["files"]["vrp_json"][item.metric_variant.value] = item.instance_path.relative_to(collection_root).as_posix()
        for record in records.values():
            variants = record["files"]["vrp_json"]
            record["files"]["vrp_json"] = {metric: variants[metric] for metric in METRICS if metric in variants}
            record["metrics"] = list(record["files"]["vrp_json"])
        self._records = records

    def _loaded(self, refresh: bool = False) -> dict[str, dict[str, Any]]:
        with self._lock:
            # A scan that found nothing (no folder yet, an unreadable one) is not kept:
            # the folder may have been cloned or repaired since, and looking again is cheap.
            if self._records is None or refresh or self._error or not self._records:
                self._scan()
            assert self._records is not None
            return self._records

    def listing(self, refresh: bool = False) -> dict[str, Any]:
        records = sorted(
            self._loaded(refresh).values(),
            key=lambda record: (record["family"], record["problem_type"], str(record["city"]), int(record["n_customers"] or 0), record["base_name"]),
        )
        hint = None
        if self.root is None:
            hint = (
                "No benchmarks folder was found. Start the GUI inside a MAMUT-routing checkout, "
                "pass --benchmarks-dir, or set MAMUT_ROUTING_BENCHMARKS_ROOT."
            )
        elif self._error:
            hint = self._error
        elif not records:
            hint = f"{self.root} holds no geographic collection (Mamut2026, Poryos2026) with static instances."
        return {
            "root": str(self.root) if self.root else None,
            "families": sorted({record["family"] for record in records}),
            "instances": [{key: value for key, value in record.items() if key != "summary"} for record in records],
            "hint": hint,
        }

    def record(self, instance_id: str) -> dict[str, Any]:
        """The record of one instance; ``KeyError`` when the catalog does not hold it."""
        return self._loaded()[instance_id]

    # ── One instance ──

    def variant_path(self, record: dict[str, Any], metric: str | None = None) -> Path:
        variants = record["files"]["vrp_json"]
        chosen = metric if metric in variants else next(iter(variants))
        return Path(record["folder"]) / variants[chosen]

    def detail(self, instance_id: str) -> dict[str, Any]:
        """The record with what the panel shows about the instance itself."""
        record = self.record(instance_id)
        path = self.variant_path(record)
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata") or {}
        summary: dict[str, Any] = {
            "capacity": payload.get("vehicle_capacity"),
            "customers": payload.get("num_customers"),
            "route_count": payload.get("num_vehicles") or metadata.get("num_vehicles_lb"),
            "generated_at": metadata.get("generated_at"),
            "method": metadata.get("method"),
            "has_time_windows": "time_windows" in payload,
            "units": {},
            "bks": {},
        }
        for metric, relative in record["files"]["vrp_json"].items():
            variant = Path(record["folder"]) / relative
            manifest_path = variant.with_name(variant.name.removesuffix(".vrp.json") + ".manifest.json")
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                summary["revision"] = manifest.get("benchmark_revision")
                summary["profile"] = manifest.get("default_profile")
                summary["objective"] = manifest.get("objective")
                summary["units"][metric] = (manifest.get("units") or {}).get("cost")
            summary["bks"][metric] = sorted(
                bks.name.removeprefix(variant.name.removesuffix(".vrp.json") + ".bks.").removesuffix(".json")
                for bks in variant.parent.glob(variant.name.removesuffix(".vrp.json") + ".bks.*.json")
            )
        return {**record, "summary": summary}

    def _sidecar_path(self, record: dict[str, Any], kind: str) -> Path:
        payload = json.loads(self.variant_path(record).read_text(encoding="utf-8"))
        geo = ((payload.get("metadata") or {}).get("sidecars") or {}).get("geo") or {}
        relative = str(geo.get("path") or "")
        if not relative:
            raise FileNotFoundError(f"{record['base_name']} names no geo sidecar")
        root = Path(record["folder"])
        path = (root / relative).resolve()
        if root.resolve() not in path.parents:
            raise ValueError(f"{record['base_name']}: sidecar path leaves the collection")
        if kind != "geo":
            path = path.with_name(path.name.replace(".geo.json.gz", f".{kind}.json.gz"))
        return path

    def meta(self, record: dict[str, Any]) -> dict[str, Any]:
        """The geo sidecar in the shape of a generated instance's metadata.

        Generated metadata numbers its nodes from 1 and names the depot; the
        sidecar numbers them from 0, the model's own indices. One convention
        downstream keeps the route and map code free of a second case.
        """
        path = self._sidecar_path(record, "geo")
        if not path.is_file():
            raise FileNotFoundError(f"{record['base_name']}: the geo sidecar {path.name} is missing from the benchmarks folder")
        key = (str(path), path.stat().st_mtime_ns)
        with self._lock:
            cached = self._meta.get(key)
            if cached is not None:
                self._meta.move_to_end(key)
                return cached
        geo = _read_gzip_json(path)
        ordered = sorted(geo.get("nodes") or [], key=lambda node: int(node["instance_node_id"]))
        depot_index = next((index for index, node in enumerate(ordered) if node.get("source_tag") == "depot"), 0)
        meta = {
            **geo,
            "nodes": [{**node, "instance_node_id": index + 1} for index, node in enumerate(ordered)],
            "depot_instance_node_id": depot_index + 1,
        }
        with self._lock:
            self._meta[key] = meta
            while len(self._meta) > 8:
                self._meta.popitem(last=False)
        return meta

    def roads(self, record: dict[str, Any]) -> SidecarRoads | None:
        """The instance's road graph, or ``None`` when its sidecar is not in the folder."""
        try:
            path = self._sidecar_path(record, "road")
        except (FileNotFoundError, ValueError):
            return None
        if not path.is_file():
            return None
        key = (str(path), path.stat().st_mtime_ns)
        with self._lock:
            cached = self._roads.get(key)
            if cached is not None:
                self._roads.move_to_end(key)
                return cached
        roads = SidecarRoads(path)
        with self._lock:
            self._roads[key] = roads
            while len(self._roads) > 3:
                self._roads.popitem(last=False)
        return roads


#: One catalog per workspace and per process. The server and its module-level
#: helpers all reach it through ``catalog_for``, so they cannot disagree on which
#: folder an instance id belongs to.
_CATALOGS: dict[str, BenchmarkCatalog] = {}


def configure_catalog(workspace: Path, root: Path | None) -> BenchmarkCatalog:
    """Set the folder the server of ``workspace`` reads its benchmark instances from."""
    catalog = BenchmarkCatalog(root)
    _CATALOGS[str(Path(workspace).resolve())] = catalog
    return catalog


def catalog_for(workspace: Path) -> BenchmarkCatalog:
    key = str(Path(workspace).resolve())
    if key not in _CATALOGS:
        _CATALOGS[key] = BenchmarkCatalog(None)
    return _CATALOGS[key]
