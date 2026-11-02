"""From a campaign spec to the list of runs still to do.

Every (instance, run entry, seed) becomes a task with its budget, objective
and run key. Tasks whose record already exists are done (resumption), tasks a
solver cannot take are listed with the reason instead of failing at run time,
and unavailable solvers are reported once. Longest budgets go first so the
pool's tail is short.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mamut_routing_tools.bks_campaign.archive import run_key, run_path
from mamut_routing_tools.bks_campaign.spec import CampaignSpec

__all__ = ["CampaignPlan", "Task", "build_plan", "default_memory_gb", "select_instances"]


@dataclass(frozen=True)
class Task:
    key: str
    campaign: str
    out: str
    instance_id: str
    instance_name: str
    instance_path: str
    num_customers: int
    problem_type: str
    objective: str
    kind: str
    solver: str
    seed: int
    time_limit_s: int
    memory_gb: float
    solvers_config: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    warm_start_routes: list[list[int]] | None = None
    warm_start_key: str | None = None
    warm_start_sha256: str | None = None
    nice: int | None = None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CampaignPlan:
    tasks: list[Task]
    done: list[str]
    unsupported: list[tuple[str, str, str]]  # (instance_id, solver, reason)
    unavailable: dict[str, str]  # solver -> reason
    retried: list[str]

    @property
    def core_hours(self) -> float:
        return sum(task.time_limit_s for task in self.tasks) / 3600.0

    def makespan_lower_bound_h(self, workers: int) -> float:
        if not self.tasks:
            return 0.0
        longest = max(task.time_limit_s for task in self.tasks)
        return max(longest, sum(task.time_limit_s for task in self.tasks) / max(1, workers)) / 3600.0

    def worst_memory_gb(self, workers: int) -> float:
        return sum(sorted((task.memory_gb for task in self.tasks), reverse=True)[:workers])

    def summary(self, workers: int) -> dict[str, Any]:
        per_solver: dict[str, int] = {}
        for task in self.tasks:
            per_solver[task.kind] = per_solver.get(task.kind, 0) + 1
        return {
            "todo": len(self.tasks),
            "done": len(self.done),
            "retried": len(self.retried),
            "per_kind": per_solver,
            "core_hours": round(self.core_hours, 2),
            "makespan_lower_bound_h": round(self.makespan_lower_bound_h(workers), 2),
            "worst_case_memory_gb": round(self.worst_memory_gb(workers), 2),
            "unsupported": len(self.unsupported),
            "unavailable": self.unavailable,
        }


def default_memory_gb(solver: str, num_customers: int) -> float:
    """Rough RSS per run, from the Mamut2026 campaigns (PyVRP 1.6 GB at n = 4000; AILS-II -Xmx4g)."""
    if solver == "pyvrp":
        return 0.2 + 0.4e-3 * num_customers
    if solver == "ails2":
        return 4.5
    if solver == "hgs-cvrp":
        return 0.1 + 8.0 * num_customers**2 / 1e9
    return 0.5 + 8.0 * num_customers**2 / 1e9


def select_instances(spec: CampaignSpec) -> list[Any]:
    """The instances the spec selects (``DiscoveredBenchmarkInstance``), static and TD alike."""
    from mamut_routing_lib.artifacts import discover_benchmark_instances

    select = spec.select
    found = discover_benchmark_instances(
        select.benchmarks_dir,
        problem_types=select.problem_types or None,
        benchmark_names=select.families or None,
        metric_variants=select.metrics or None,
        instance_ids=select.instance_ids or None,
    )
    pattern = re.compile(select.name_regex) if select.name_regex else None
    chosen = []
    for item in found:
        n = item.num_customers
        if n is None:
            n = int(item.load().num_customers)
        if select.min_n is not None and n < select.min_n:
            continue
        if select.max_n is not None and n > select.max_n:
            continue
        if pattern is not None and not (pattern.search(item.instance_name) or pattern.search(item.instance_id)):
            continue
        chosen.append(item)
    return chosen


def _best_known_routes(spec: CampaignSpec, instance_path: Path, instance_id: str, objective: Any) -> tuple[list[list[int]] | None, str | None]:
    """Routes of the best valid archived run or the stored BKS (warm starts), with their origin."""
    from mamut_routing_lib.artifacts import get_bks_path_for_instance, load_bks

    from mamut_routing_tools.bks_campaign.archive import iter_runs

    best: tuple[float, list[list[int]], str] | None = None
    for record in iter_runs(spec.out):
        if record.get("instance_id") != instance_id or record.get("objective_function") != objective.value:
            continue
        if not record.get("valid") or record.get("validated_cost") is None or not record.get("routes"):
            continue
        cost = float(record["validated_cost"])
        if best is None or cost < best[0]:
            best = (cost, record["routes"], f"run:{record['key']}")
    bks_path = get_bks_path_for_instance(instance_path, objective)
    if bks_path.is_file():
        bks = load_bks(bks_path)
        if bks.cost is not None and (best is None or float(bks.cost) < best[0]):
            best = (float(bks.cost), bks.routes, f"bks:{bks.metadata.get('method') or 'stored'}")
    if best is None:
        return None, None
    return [list(route) for route in best[1]], best[2]


def build_plan(spec: CampaignSpec, *, only: str | None = None, retry_failed: bool = False) -> CampaignPlan:
    """Expand the spec into tasks, skipping done runs and unsupported (instance, solver) pairs."""
    from mamut_routing_lib.artifacts import load_benchmark_instance
    from mamut_routing_lib.objectives import base_objective
    from mamut_routing_lib.solvers import get_solver, profile_instance

    only_pattern = re.compile(only) if only else None
    config = str(spec.campaign.solvers_config) if spec.campaign.solvers_config else None
    solvers = {}
    unavailable: dict[str, str] = {}
    for run in spec.runs:
        if run.solver in solvers or run.solver in unavailable:
            continue
        solver = get_solver(run.solver, config=config)
        availability = solver.availability()
        if not availability.available:
            unavailable[run.solver] = f"{availability.reason}" + (f" ({availability.hint})" if availability.hint else "")
            continue
        if run.warm_start == "best" and not solver.capabilities.warm_start:
            raise ValueError(f"[[runs]] {run.run_kind}: {run.solver} does not take warm starts")
        solvers[run.solver] = solver

    tasks: list[Task] = []
    done: list[str] = []
    retried: list[str] = []
    unsupported: list[tuple[str, str, str]] = []
    for item in select_instances(spec):
        if only_pattern and not (only_pattern.search(item.instance_id) or only_pattern.search(item.instance_name)):
            continue
        instance = load_benchmark_instance(item.instance_path)
        problem = getattr(item.problem_type, "value", str(item.problem_type))
        objective = spec.select.objective.get(item.problem_type) or base_objective(instance)
        profile = profile_instance(instance, item.instance_path)
        n = int(instance.num_customers)
        warm: dict[str, tuple[list[list[int]] | None, str | None]] = {}
        for run in spec.runs:
            solver = solvers.get(run.solver)
            if solver is None:
                continue
            support = solver.supports(profile, objective)
            if not support.supported:
                unsupported.append((item.instance_id, run.solver, "; ".join(support.reasons)))
                continue
            budget = spec.budget.seconds(n, run.budget_factor)
            routes = sha = origin = None
            if run.warm_start == "best":
                if objective.value not in warm:
                    warm[objective.value] = _best_known_routes(spec, item.instance_path, item.instance_id, objective)
                routes, origin = warm[objective.value]
                if routes is None:
                    unsupported.append((item.instance_id, run.solver, "warm_start = best but no archived run or stored BKS"))
                    continue
                sha = hashlib.sha256(json.dumps(routes, separators=(",", ":")).encode()).hexdigest()
            for seed in run.seed_list():
                key = run_key(item.instance_id, objective.value, run.run_kind, seed, budget)
                record_path = run_path(spec.out, key)
                if record_path.is_file():
                    if retry_failed and _failed(record_path):
                        retried.append(key)
                    else:
                        done.append(key)
                        continue
                tasks.append(
                    Task(
                        key=key,
                        campaign=spec.campaign.name,
                        out=str(spec.out),
                        instance_id=item.instance_id,
                        instance_name=item.instance_name,
                        instance_path=str(item.instance_path),
                        num_customers=n,
                        problem_type=problem,
                        objective=objective.value,
                        kind=run.run_kind,
                        solver=run.solver,
                        seed=seed,
                        time_limit_s=budget,
                        memory_gb=run.memory_gb if run.memory_gb is not None else default_memory_gb(run.solver, n),
                        solvers_config=config,
                        options=dict(run.options),
                        warm_start_routes=routes,
                        warm_start_key=origin,
                        warm_start_sha256=sha,
                        nice=spec.resources.nice,
                    )
                )
    tasks.sort(key=lambda task: (-task.time_limit_s, task.key))
    return CampaignPlan(tasks=tasks, done=done, unsupported=unsupported, unavailable=unavailable, retried=retried)


def _failed(path: Path) -> bool:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    return record.get("status") in ("timeout", "failed") or not record.get("valid")
