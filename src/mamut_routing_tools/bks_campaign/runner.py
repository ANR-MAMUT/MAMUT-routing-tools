"""Run a campaign's tasks: a resumable process pool with memory-aware admission.

- One task per worker process (``spawn``, ``max_tasks_per_child=1``): a solver's
  memory is returned to the system after every run, and a fixed driver is
  picked up by the next worker.
- Admission: a task starts only while the memory estimates of the running
  tasks plus its own fit the budget (``resources.memory_gb``, or 80 % of
  ``MemAvailable``); at least one task always runs.
- Workers solve, check and write their own run record (atomically); they never
  write a BKS. Harvest is a separate, single-writer step.
- First Ctrl-C / SIGTERM: stop admitting, let the running tasks finish. A second
  one aborts: workers get SIGTERM, which kills their solver's process group.
  Workers ignore the terminal's SIGINT themselves, so the first Ctrl-C never
  interrupts a run halfway.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
import socket
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from mamut_routing_tools.bks_campaign.archive import SCHEMA_VERSION, atomic_write_json, raw_dir, run_path
from mamut_routing_tools.bks_campaign.plan import CampaignPlan, Task

__all__ = ["execute_task", "memory_budget_gb", "run_tasks"]

#: Solver input files are large for big explicit matrices (130 MB at n = 4000); keep only outputs.
_INPUT_FILES = ("problem.vrp", "problem.par")


def _loadavg() -> float | None:
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:  # pragma: no cover
        return None


def memory_budget_gb(setting: float | str) -> float:
    """``resources.memory_gb``: a number, or ``auto`` = 80 % of MemAvailable (Linux; 8 GB elsewhere)."""
    if setting != "auto":
        return float(setting)
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return 0.8 * int(line.split()[1]) / (1024 * 1024)
    except OSError:  # pragma: no cover
        pass
    return 8.0  # pragma: no cover


def _worker_init(nice: int | None) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _terminate(signum: int, frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _terminate)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    if nice:
        try:
            os.nice(int(nice))
        except OSError:  # pragma: no cover
            pass


def execute_task(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker: solve one task, check it, write its record; returns a short summary."""
    from importlib.metadata import version

    from mamut_routing_lib.enums import ObjectiveFunction
    from mamut_routing_lib.solvers import SolveMonitor, run_solver
    from mamut_routing_lib.solvers.provenance import environment, file_sha256

    task = Task(**payload)
    out = Path(task.out)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "key": task.key,
        "campaign": task.campaign,
        "instance_id": task.instance_id,
        "instance_name": task.instance_name,
        "instance_path": task.instance_path,
        "instance_sha256": file_sha256(task.instance_path),
        "n": task.num_customers,
        "problem_type": task.problem_type,
        "objective_function": task.objective,
        "kind": task.kind,
        "solver": task.solver,
        "seed": task.seed,
        "time_limit_s": task.time_limit_s,
        "warm_start_key": task.warm_start_key,
        "warm_start_sha256": task.warm_start_sha256,
        "started": started,
        "loadavg_start": _loadavg(),
        **environment(include_host=True),
    }
    try:
        record["tools_version"] = version("mamut-routing-tools")
    except Exception:  # noqa: BLE001
        record["tools_version"] = None
    monitor = SolveMonitor(None, interval_s=3600.0, record_trajectory=True)
    workdir = raw_dir(out, task.key)
    try:
        solved = run_solver(
            task.solver,
            task.instance_path,
            time_limit_s=task.time_limit_s,
            seed=task.seed,
            objective_function=ObjectiveFunction(task.objective),
            monitor=monitor,
            initial_routes=task.warm_start_routes,
            workdir=workdir,
            keep_artifacts=True,
            options=task.options,
            config=task.solvers_config,
        )
        result = solved.result
        record.update(
            method=result.method,
            solver_version=result.solver_version,
            status=result.status,
            error=result.error,
            wall_time=round(result.wall_time, 3),
            valid=solved.valid,
            validated_cost=solved.cost,
            num_routes=len(result.routes) if result.routes else 0,
            check_status=solved.validation_status,
            check_error=(solved.check.error_message or None) if solved.check is not None else None,
            solver_cost=result.solver_cost,
            solver_cost_raw=result.solver_cost_raw,
            cost_scale=result.cost_scale,
            cost_exact=result.cost_exact,
            iterations=monitor.iterations or result.metadata.get("iterations"),
            trajectory=[list(point) for point in monitor.trajectory],
            routes=result.routes,
            provenance={key: value for key, value in result.provenance.items() if key not in ("host",)},
        )
    except Exception as error:  # noqa: BLE001 - one bad run must not take the pool down
        import traceback

        record.update(status="failed", valid=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        for name in _INPUT_FILES:
            (workdir / name).unlink(missing_ok=True)
    record.update(finished=datetime.now(UTC).isoformat(timespec="seconds"), loadavg_end=_loadavg())
    atomic_write_json(run_path(out, task.key), record)
    return {key: record.get(key) for key in ("key", "status", "valid", "validated_cost", "num_routes", "wall_time", "error")}


def run_tasks(
    plan: CampaignPlan,
    *,
    out: Path,
    workers: int,
    memory_gb: float,
    progress: Callable[[str], None] = print,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """Run the plan's tasks; returns (and writes to ``run.DONE``) a summary."""
    tasks = list(plan.tasks[:max_tasks] if max_tasks is not None else plan.tasks)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = {"stop": False, "abort": False}
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def _handle(signum: int, frame: Any) -> None:
        if state["stop"]:
            state["abort"] = True
            progress("second interrupt: aborting the running tasks")
        else:
            state["stop"] = True
            progress("interrupt: no new task starts; the running ones finish (interrupt again to abort)")

    previous = {sig: signal.signal(sig, _handle) for sig in (signal.SIGINT, signal.SIGTERM)}
    nice = tasks[0].nice if tasks else None
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        max_tasks_per_child=1,
        initializer=_worker_init,
        initargs=(nice,),
    )
    running: dict[Future, Task] = {}
    pending = list(tasks)
    try:
        while pending or running:
            if state["abort"]:
                break
            while pending and not state["stop"] and len(running) < workers:
                in_use = sum(task.memory_gb for task in running.values())
                candidate = pending[0]
                if running and in_use + candidate.memory_gb > memory_gb:
                    # The longest task does not fit: start the first smaller one that does.
                    fitting = next((task for task in pending if in_use + task.memory_gb <= memory_gb), None)
                    if fitting is None:
                        break
                    candidate = fitting
                pending.remove(candidate)
                running[executor.submit(execute_task, candidate.to_payload())] = candidate
                progress(f"start {candidate.key} ({candidate.time_limit_s}s, ~{candidate.memory_gb:.1f} GB)")
            if not running:
                break
            done, _ = wait(list(running), timeout=1.0, return_when=FIRST_COMPLETED)
            for future in done:
                task = running.pop(future)
                try:
                    summary = future.result()
                except Exception as error:  # noqa: BLE001 - a dead worker (OOM kill) is reported, not fatal
                    summary = {"key": task.key, "status": "failed", "valid": False, "error": f"worker died: {error}"}
                    failures.append(summary)
                results.append(summary)
                cost = summary.get("validated_cost")
                progress(f"done  {task.key}: {summary.get('status')}" + (f" cost {cost}" if cost is not None else ""))
    finally:
        if state["abort"]:
            for process in list(getattr(executor, "_processes", {}).values()):
                try:
                    os.kill(process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):  # pragma: no cover
                    pass
        executor.shutdown(wait=True, cancel_futures=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    summary = {
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "planned": len(tasks),
        "completed": len(results),
        "valid": sum(1 for item in results if item.get("valid")),
        "failed_workers": len(failures),
        "interrupted": bool(state["stop"] or state["abort"]),
        "aborted": bool(state["abort"]),
        "not_started": len(pending),
        "wall_time_s": round(time.monotonic() - started, 1),
    }
    (out / "run.DONE").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary
