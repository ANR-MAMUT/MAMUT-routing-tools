"""Persistent, checker-validated solution runs and comparisons.

A run is tied to the instance file it was validated on: schema 2 records
``instance_sha256`` (the file's sha256 when the solve or import started). A
run whose instance file no longer has that hash -- the base was regenerated
or replaced -- is *stale*: it is listed with ``stale: true`` and refused by
compare and render, instead of showing an old verdict and cost on new
customers. Schema 1 runs have no hash; they are re-validated against the
current file once per file content and the verdict is cached in the record.
"""

from __future__ import annotations

import hashlib
import json
import string
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mamut_routing_lib.solutions import ParsedSolution, SolutionFormatError

from mamut_routing_tools.workspace import solutions_dir


def _now() -> str:
    return datetime.now(UTC).isoformat()


#: sha256 of instance files keyed by (path, size, mtime_ns): listings hash
#: every run's instance, and an n=1000 instance file is megabytes.
_FILE_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


def file_sha256(path: str | Path) -> str | None:
    """sha256 of a file's bytes (cached on size and mtime); ``None`` if it does not exist."""
    target = Path(path)
    try:
        stat = target.stat()
    except FileNotFoundError:
        return None
    key = (str(target.resolve()), stat.st_size, stat.st_mtime_ns)
    cached = _FILE_SHA256_CACHE.get(key)
    if cached is None:
        cached = hashlib.sha256(target.read_bytes()).hexdigest()
        _FILE_SHA256_CACHE[key] = cached
    return cached


def instance_id_for(folder: Path, base_name: str) -> str:
    identity = f"{folder.resolve()}\0{base_name}".encode()
    return hashlib.sha256(identity).hexdigest()[:20]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class SolutionImportError(SolutionFormatError):
    """A pasted or uploaded solution that cannot be read or does not fit.

    The parsing itself lives in ``mamut_routing_lib.solutions`` (shared with
    ``mamut-routing check`` / ``bks submit``); this subclass keeps the GUI's
    error type for its callers.
    """


def parse_solution_text(text: str, *, filename: str = "") -> tuple[list[list[int]], float | None]:
    """Routes and declared cost from a CVRPLIB ``.sol`` or a JSON solution (``mamut_routing_lib.solutions``)."""
    from mamut_routing_lib.solutions import parse_solution_text as parse

    try:
        return parse(text, filename=filename)
    except SolutionFormatError as error:
        raise SolutionImportError(str(error)) from error


def normalize_imported_routes(routes: list[list[int]], num_customers: int) -> list[list[int]]:
    """Map an external stop-id convention onto library model ids (``mamut_routing_lib.solutions``)."""
    from mamut_routing_lib.solutions import normalize_routes

    try:
        return normalize_routes(routes, num_customers)
    except SolutionFormatError as error:
        raise SolutionImportError(str(error)) from error


def read_imported_solution(text: str, instance: Any, *, filename: str = "") -> tuple[list[list[int]], float | None, str]:
    """(library routes, declared cost, format) of a pasted or uploaded solution in any supported format.

    CVRPLIB ``.sol``, JSON routes and LKH-3 MTSP solution files are recognised.
    """
    parsed = parse_imported_solution(text, instance, filename=filename)
    return parsed.routes, parsed.declared_cost, parsed.format


def parse_imported_solution(
    text: str, instance: Any, *, filename: str = "", preserve_solver_report: bool = False,
) -> ParsedSolution:
    """Read routes and any numeric provenance supplied by the imported file."""
    from mamut_routing_lib.solutions import read_solution_text

    try:
        return read_solution_text(
            text,
            int(instance.num_customers),
            filename=filename,
            depot=int(getattr(instance, "depot", 0) or 0),
            preserve_solver_report=preserve_solver_report,
        )
    except SolutionFormatError as error:
        raise SolutionImportError(str(error)) from error


def validate_solution(
    instance: Any,
    routes: list[list[int]],
    *,
    instance_path: Path,
    objective_function=None,
    marked_input=None,
) -> dict[str, Any]:
    """Run the canonical library checker (static or TD) and return a JSON-friendly report."""
    from mamut_routing_lib.solutions import check_routes

    checked = check_routes(instance, instance_path, routes, objective_function, marked_input=marked_input)
    return {
        "valid": checked.is_valid(),
        **checked.model_dump(mode="json"),
    }


class SolutionStore:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.root = solutions_dir(workspace)

    def _instance_dir(self, instance_id: str) -> Path:
        return self.root / instance_id

    def record(
        self,
        *,
        instance_id: str,
        instance_name: str,
        instance_path: Path,
        routes: list[list[int]],
        cost: int | float | None,
        objective_function: str,
        solver: str,
        method: str,
        seed: int,
        time_limit_s: int,
        wall_time_s: float,
        validation: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        job_id: str | None = None,
        source: str = "solver",
        instance_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Persist a run. ``instance_sha256`` should be hashed *before* solving or
        importing, so a concurrent regeneration marks the run stale at once."""
        run_id = uuid.uuid4().hex
        record = {
            "schema_version": 2,
            "run_id": run_id,
            "instance_id": instance_id,
            "instance_name": instance_name,
            "instance_path": str(instance_path),
            "instance_sha256": instance_sha256 or file_sha256(instance_path),
            "created_at": _now(),
            "source": source,
            "solver": solver,
            "method": method,
            "objective_function": objective_function,
            "seed": seed,
            "time_limit_s": time_limit_s,
            "wall_time_s": wall_time_s,
            "cost": cost,
            "num_routes": len(routes),
            "routes": routes,
            "validation": validation,
            "metadata": metadata or {},
            "job_id": job_id,
        }
        _atomic_json(self._instance_dir(instance_id) / f"{run_id}.json", record)
        return {**record, "stale": False, "stale_reason": None}

    def get(self, instance_id: str, run_id: str) -> dict[str, Any]:
        if len(run_id) != 32 or any(character not in string.hexdigits for character in run_id):
            raise KeyError(run_id)
        path = self._instance_dir(instance_id) / f"{run_id}.json"
        if not path.is_file():
            raise KeyError(run_id)
        return self._annotated(json.loads(path.read_text(encoding="utf-8")), path)

    def list(self, instance_id: str) -> list[dict[str, Any]]:
        """Every run of the instance, newest first, each with ``stale`` / ``stale_reason``."""
        records: list[dict[str, Any]] = []
        for path in self._instance_dir(instance_id).glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            records.append(self._annotated(record, path))
        records.sort(key=lambda value: str(value.get("created_at") or ""), reverse=True)
        return records

    def current(self, instance_id: str) -> list[dict[str, Any]]:
        """The runs still valid for the instance on disk."""
        return [record for record in self.list(instance_id) if not record["stale"]]

    def _annotated(self, record: dict[str, Any], path: Path) -> dict[str, Any]:
        stale, reason = self.run_staleness(record, path)
        return {**record, "stale": stale, "stale_reason": reason}

    def run_staleness(self, record: dict[str, Any], path: Path | None = None) -> tuple[bool, str | None]:
        """Whether ``record`` still describes the instance file on disk, and why not."""
        instance_path = Path(str(record.get("instance_path") or ""))
        current = file_sha256(instance_path) if str(instance_path) else None
        if current is None:
            return True, "the instance file no longer exists"
        recorded = record.get("instance_sha256")
        if recorded:
            if recorded != current:
                return True, "the instance was regenerated after this run"
            return False, None
        # Schema 1: no hash. Re-validate once per instance content, cache the verdict.
        cached = record.get("legacy_check") or {}
        if cached.get("instance_sha256") == current:
            return bool(cached.get("stale")), cached.get("reason")
        stale, reason = _recheck_legacy_run(record, instance_path)
        record["legacy_check"] = {"instance_sha256": current, "stale": stale, "reason": reason, "checked_at": _now()}
        target = path or self._instance_dir(str(record.get("instance_id"))) / f"{record.get('run_id')}.json"
        if target.is_file():
            _atomic_json(target, {key: value for key, value in record.items() if key not in ("stale", "stale_reason")})
        return stale, reason


def _recheck_legacy_run(record: dict[str, Any], instance_path: Path) -> tuple[bool, str | None]:
    """A hash-less (schema 1) run is current iff the checker still gives its verdict and cost."""
    from mamut_routing_lib.artifacts import load_benchmark_instance

    try:
        instance = load_benchmark_instance(instance_path)
        routes = [[int(node) for node in route] for route in record.get("routes") or []]
        report = validate_solution(instance, routes, instance_path=instance_path)
    except Exception as error:  # noqa: BLE001 - any failure means the run cannot be trusted
        return True, f"the run no longer fits the instance ({error})"
    was_valid = bool((record.get("validation") or {}).get("valid"))
    if bool(report.get("valid")) != was_valid:
        return True, "the checker's verdict changed: the instance was regenerated after this run"
    if was_valid and record.get("cost") is not None and report.get("routing_cost") != record.get("cost"):
        return True, "the checker's cost changed: the instance was regenerated after this run"
    return False, None


def _route_edges(routes: list[list[int]], depot: int) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for route in routes:
        sequence = [depot, *route, depot]
        edges.update(zip(sequence, sequence[1:]))
    return edges


def _co_routed_pairs(routes: list[list[int]]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for route in routes:
        ordered = sorted(route)
        for index, left in enumerate(ordered):
            pairs.update((left, right) for right in ordered[index + 1 :])
    return pairs


def compare_solution_records(
    instance: Any,
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Compare objective value, fleet, loads, edges and route partition."""

    from mamut_routing_lib.checker import get_objective_tuple
    from mamut_routing_lib.enums import ObjectiveFunction

    candidate_objective = str(candidate["objective_function"])
    reference_objective = str(reference["objective_function"])
    if candidate_objective != reference_objective:
        raise ValueError(
            f"Cannot compare {candidate_objective} against {reference_objective}; select runs with the same objective."
        )
    candidate_metric = str(candidate.get("metadata", {}).get("metric") or "")
    reference_metric = str(reference.get("metadata", {}).get("metric") or "")
    if candidate_metric != reference_metric:
        raise ValueError(
            f"Cannot compare {candidate_metric or 'unknown'} against {reference_metric or 'unknown'}; "
            "select runs for the same metric."
        )
    objective = ObjectiveFunction(candidate_objective)
    candidate_cost = candidate.get("cost")
    reference_cost = reference.get("cost")
    if candidate_cost is None or reference_cost is None:
        raise ValueError("Both solution runs need a cost before they can be compared.")
    candidate_routes = [[int(node) for node in route] for route in candidate["routes"]]
    reference_routes = [[int(node) for node in route] for route in reference["routes"]]
    candidate_tuple = get_objective_tuple(candidate_routes, candidate_cost, objective)
    reference_tuple = get_objective_tuple(reference_routes, reference_cost, objective)
    cost_delta = candidate_cost - reference_cost
    relative_gap = None if reference_cost == 0 else (cost_delta / abs(reference_cost)) * 100.0

    def route_loads(routes: list[list[int]]) -> list[int]:
        return [sum(int(instance.demands[node]) for node in route) for route in routes]

    candidate_edges = _route_edges(candidate_routes, int(instance.depot))
    reference_edges = _route_edges(reference_routes, int(instance.depot))
    candidate_pairs = _co_routed_pairs(candidate_routes)
    reference_pairs = _co_routed_pairs(reference_routes)
    return {
        "candidate_run_id": candidate.get("run_id"),
        "reference_run_id": reference.get("run_id"),
        "objective_function": candidate_objective,
        "metric": candidate_metric,
        "ordering": "better" if candidate_tuple < reference_tuple else "equal" if candidate_tuple == reference_tuple else "worse",
        "cost_delta": cost_delta,
        "relative_gap_percent": relative_gap,
        "route_count_delta": len(candidate_routes) - len(reference_routes),
        "candidate": {
            "cost": candidate_cost,
            "num_routes": len(candidate_routes),
            "route_loads": route_loads(candidate_routes),
            "valid": bool(candidate.get("validation", {}).get("valid")),
        },
        "reference": {
            "cost": reference_cost,
            "num_routes": len(reference_routes),
            "route_loads": route_loads(reference_routes),
            "valid": bool(reference.get("validation", {}).get("valid")),
        },
        "route_difference": {
            "directed_edges_added": len(candidate_edges - reference_edges),
            "directed_edges_removed": len(reference_edges - candidate_edges),
            "co_routed_customer_pairs_added": len(candidate_pairs - reference_pairs),
            "co_routed_customer_pairs_removed": len(reference_pairs - candidate_pairs),
        },
    }
