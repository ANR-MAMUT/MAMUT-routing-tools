"""Persistent, checker-validated solution runs and comparisons."""

from __future__ import annotations

import hashlib
import json
import re
import string
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mamut_routing_tools.workspace import solutions_dir


def _now() -> str:
    return datetime.now(UTC).isoformat()


def instance_id_for(folder: Path, base_name: str) -> str:
    identity = f"{folder.resolve()}\0{base_name}".encode()
    return hashlib.sha256(identity).hexdigest()[:20]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


_ROUTE_LINE = re.compile(r"^\s*Route\s*#\s*\d+\s*:\s*(.*)$", re.IGNORECASE)
_COST_LINE = re.compile(r"^\s*cost\s*[:=]?\s*(-?\d+(?:\.\d+)?)\s*$", re.IGNORECASE)


class SolutionImportError(ValueError):
    """A pasted or uploaded solution that cannot be read or does not fit."""


def parse_solution_text(text: str, *, filename: str = "") -> tuple[list[list[int]], float | None]:
    """Routes and declared cost from a CVRPLIB ``.sol`` or a JSON solution.

    Accepts ``Route #k: a b c`` lines with an optional ``Cost <value>`` line, or
    JSON as either ``{"routes": [[...]]}`` or a bare array of routes.
    """
    stripped = text.strip()
    if not stripped:
        raise SolutionImportError("The solution is empty.")

    looks_json = stripped[0] in "[{" or filename.lower().endswith(".json")
    if looks_json:
        try:
            payload = json.loads(stripped)
        except ValueError as error:
            raise SolutionImportError(f"Not valid JSON: {error}") from error
        raw_routes = payload if isinstance(payload, list) else payload.get("routes")
        if not isinstance(raw_routes, list):
            raise SolutionImportError("JSON solution has no 'routes' array.")
        declared = None if isinstance(payload, list) else payload.get("cost")
        routes = []
        for route in raw_routes:
            if not isinstance(route, list):
                raise SolutionImportError("Every entry of 'routes' must be a list of stops.")
            try:
                routes.append([int(stop) for stop in route])
            except (TypeError, ValueError) as error:
                raise SolutionImportError(f"Route stops must be integers: {error}") from error
        return routes, (float(declared) if declared is not None else None)

    routes = []
    declared_cost: float | None = None
    for line in stripped.splitlines():
        match = _ROUTE_LINE.match(line)
        if match:
            try:
                routes.append([int(part) for part in match.group(1).split()])
            except ValueError as error:
                raise SolutionImportError(f"Unreadable stop id in '{line.strip()}': {error}") from error
            continue
        cost_match = _COST_LINE.match(line)
        if cost_match:
            declared_cost = float(cost_match.group(1))
    if not routes:
        raise SolutionImportError(
            "No 'Route #k: ...' lines found. Provide a CVRPLIB .sol file or JSON routes."
        )
    return routes, declared_cost


def normalize_imported_routes(routes: list[list[int]], num_customers: int) -> list[list[int]]:
    """Map an external stop-id convention onto library model ids (depot 0).

    External files disagree on whether customers are numbered 1..n (CVRPLIB, the
    common case) or 0..n-1, and some include the depot at the route ends. A file
    is either understood exactly or rejected -- never silently reinterpreted into
    a different instance.

    The two conventions are told apart *before* any depot stripping, because a
    literal 0 is a depot marker under one and a customer under the other. Only a
    file that covers 0..n-1 exactly once, zero included, is read as 0-based;
    everything else is read as 1..n with 0 meaning the depot.
    """
    raw_stops = [stop for route in routes for stop in route]
    if not raw_stops:
        raise SolutionImportError("The solution contains no customer stops.")

    zero_based = (
        0 in raw_stops
        and len(raw_stops) == len(set(raw_stops))
        and set(raw_stops) == set(range(num_customers))
    )
    if zero_based:
        normalized = [[stop + 1 for stop in route] for route in routes]
    else:
        # 0 is the depot written at the route boundaries: drop it, ids are 1..n.
        normalized = [[stop for stop in route if stop != 0] for route in routes]

    flat = [stop for route in normalized if route for stop in route]
    if not flat:
        raise SolutionImportError("The solution contains no customer stops.")
    low, high = min(flat), max(flat)
    if low < 1 or high > num_customers:
        raise SolutionImportError(
            f"Stop ids run {min(raw_stops)}..{max(raw_stops)}, which fits neither "
            f"1..{num_customers} nor 0..{num_customers - 1}; this solution is not "
            f"for an instance with {num_customers} customers."
        )

    seen = Counter(flat)
    duplicates = sorted(stop for stop, count in seen.items() if count > 1)
    if duplicates:
        raise SolutionImportError(
            f"Customer(s) {', '.join(str(value) for value in duplicates[:10])} visited more than once."
        )
    missing = sorted(set(range(1, num_customers + 1)) - set(flat))
    if missing:
        raise SolutionImportError(
            f"{len(missing)} customer(s) never visited (first missing: "
            f"{', '.join(str(value) for value in missing[:10])})."
        )
    return [route for route in normalized if route]


def validate_solution(
    instance: Any,
    routes: list[list[int]],
    *,
    instance_path: Path,
) -> dict[str, Any]:
    """Run the canonical library checker and return a JSON-friendly report."""

    from mamut_routing_lib.checker import check_solution
    from mamut_routing_lib.models import BenchmarkSolution
    from mamut_routing_lib.solvers.pyvrp import hydrate_collection_instance

    candidate = BenchmarkSolution(
        instance_name=str(instance.instance_name),
        routes=routes,
        cost=None,
        metadata={},
    )
    checkable = hydrate_collection_instance(instance, instance_path)
    checked = check_solution(checkable, candidate)
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
    ) -> dict[str, Any]:
        run_id = uuid.uuid4().hex
        record = {
            "schema_version": 1,
            "run_id": run_id,
            "instance_id": instance_id,
            "instance_name": instance_name,
            "instance_path": str(instance_path),
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
        return record

    def get(self, instance_id: str, run_id: str) -> dict[str, Any]:
        if len(run_id) != 32 or any(character not in string.hexdigits for character in run_id):
            raise KeyError(run_id)
        path = self._instance_dir(instance_id) / f"{run_id}.json"
        if not path.is_file():
            raise KeyError(run_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self, instance_id: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in self._instance_dir(instance_id).glob("*.json"):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        records.sort(key=lambda value: str(value.get("created_at") or ""), reverse=True)
        return records


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
