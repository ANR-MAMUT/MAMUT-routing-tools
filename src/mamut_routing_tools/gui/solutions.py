"""Persistent, checker-validated solution runs and comparisons."""

from __future__ import annotations

import hashlib
import json
import string
import uuid
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
