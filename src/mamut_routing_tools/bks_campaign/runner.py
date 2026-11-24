"""Run a campaign's tasks: resumable worker processes with memory-aware admission.

- One spawned worker process per task: a solver's memory is returned to the
  system after every run, and a fixed driver is picked up by the next worker.
  Workers are managed one by one (no shared pool), so a worker that dies (the
  out-of-memory killer) costs only its own run: it gets a ``failed`` record
  (``--retry-failed`` runs it again), its orphaned solver is killed, and the
  other runs carry on.
- Admission: a task starts only while the memory estimates of the running
  tasks plus its own fit the budget (``resources.memory_gb``, or 80 % of
  ``MemAvailable``); at least one task always runs.
- A ``warm_start = "best"`` task starts only once the other runs of its
  instance and objective have finished, and looks its start routes up at that
  moment: the best valid archived run, or the stored BKS.
- Workers solve, check and write their own run record (atomically); they never
  write a BKS. Harvest is a separate, single-writer step.
- First Ctrl-C: stop admitting, let the running tasks finish. A second one
  aborts: workers get SIGTERM, which kills their solver's process group.
  Workers ignore the terminal's SIGINT, from the moment they start (it is
  blocked while they boot), so the first Ctrl-C never interrupts a run halfway.
- SIGTERM (``kill``, ``timeout``, systemd, a batch scheduler before its SIGKILL)
  aborts at once: the running tasks are stopped and leave no record, so the next
  ``run`` redoes them.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import multiprocessing as mp
import multiprocessing.connection
import os
import signal
import socket
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from mamut_routing_tools.bks_campaign.archive import SCHEMA_VERSION, atomic_write_json, raw_dir, run_path
from mamut_routing_tools.bks_campaign.plan import CampaignPlan, Task, best_known_routes

__all__ = ["RUN_MARKER_ENV", "execute_task", "memory_budget_gb", "run_tasks"]

#: Solver input files are large for big explicit matrices (130 MB at n = 4000); keep only outputs.
_INPUT_FILES = ("problem.vrp", "problem.par")
#: A worker sets this to its run key before it solves; its solver's whole process tree inherits it,
#: which is how the solver of a worker that died is told apart from any other process.
RUN_MARKER_ENV = "MAMUT_BKS_CAMPAIGN_RUN"


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


#: Signals held back while a worker boots: the terminal's Ctrl-C must not kill it before it ignores SIGINT.
_BOOT_SIGNALS = {signal.SIGINT, signal.SIGTERM}
#: Seconds an aborted worker gets to kill its solver before it is killed itself.
ABORT_GRACE_S = 15.0


@contextlib.contextmanager
def _signals_blocked() -> Iterator[None]:
    """Block SIGINT/SIGTERM in this thread; a process started meanwhile inherits the mask."""
    if not hasattr(signal, "pthread_sigmask"):  # pragma: no cover - Windows
        yield
        return
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, _BOOT_SIGNALS)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _worker_init(nice: int | None) -> None:
    # Ignoring SIGINT discards a Ctrl-C that arrived while the worker booted with it blocked.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _terminate(signum: int, frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _terminate)
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, _BOOT_SIGNALS)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    if nice:
        try:
            os.nice(int(nice))
        except OSError:  # pragma: no cover
            pass


def _base_record(task: Task, started: str) -> dict[str, Any]:
    from mamut_routing_lib.solvers.provenance import environment, file_sha256

    return {
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
        "warm_start_key": None,  # set by the worker for a warm-started run
        "warm_start_sha256": None,
        "started": started,
        "loadavg_start": _loadavg(),
        **environment(include_host=True),
    }


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in ("key", "status", "valid", "validated_cost", "num_routes", "wall_time", "error")}


def execute_task(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker: solve one task, check it, write its record; returns a short summary."""
    from importlib.metadata import version

    from mamut_routing_lib.enums import ObjectiveFunction
    from mamut_routing_lib.solvers import SolveMonitor, run_solver
    from mamut_routing_lib.solvers.report import solver_report_metadata

    task = Task(**payload)
    out = Path(task.out)
    record = _base_record(task, datetime.now(UTC).isoformat(timespec="seconds"))
    try:
        record["tools_version"] = version("mamut-routing-tools")
    except Exception:  # noqa: BLE001
        record["tools_version"] = None
    monitor = SolveMonitor(None, interval_s=3600.0, record_trajectory=True)
    workdir = raw_dir(out, task.key)
    objective = ObjectiveFunction(task.objective)
    initial_routes = None
    if task.warm_start == "best":
        # Looked up now, not when the plan was made: the runs this one starts from have finished.
        initial_routes, origin = best_known_routes(
            out, Path(task.instance_path), task.instance_id, objective, record["instance_sha256"]
        )
        if initial_routes is None:
            record.update(
                status="failed",
                valid=False,
                error="warm_start = best: no valid archived run or stored BKS to start from",
                finished=datetime.now(UTC).isoformat(timespec="seconds"),
                loadavg_end=_loadavg(),
            )
            atomic_write_json(run_path(out, task.key), record)
            return _summary(record)
        record["warm_start_key"] = origin
        record["warm_start_sha256"] = hashlib.sha256(json.dumps(initial_routes, separators=(",", ":")).encode()).hexdigest()
    try:
        solved = run_solver(
            task.solver,
            task.instance_path,
            time_limit_s=task.time_limit_s,
            seed=task.seed,
            objective_function=objective,
            monitor=monitor,
            initial_routes=initial_routes,
            workdir=workdir,
            keep_artifacts=True,
            options=task.options,
            config=task.solvers_config,
        )
        result = solved.result
        if solved.valid:
            evaluation = solved.check.exact_evaluation
            record.update(check_profile=evaluation["profile"], authoritative_cost=evaluation["authoritative_cost"])
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
            cost_scale=result.cost_scale,
            iterations=monitor.iterations or result.metadata.get("iterations"),
            trajectory=[list(point) for point in monitor.trajectory],
            routes=result.routes,
            provenance={key: value for key, value in result.provenance.items() if key not in ("host",)},
            **solver_report_metadata(result.solver_report),
        )
    except Exception as error:  # noqa: BLE001 - one bad run must not take the pool down
        import traceback

        record.update(status="failed", valid=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        for name in _INPUT_FILES:
            (workdir / name).unlink(missing_ok=True)
    record.update(finished=datetime.now(UTC).isoformat(timespec="seconds"), loadavg_end=_loadavg())
    atomic_write_json(run_path(out, task.key), record)
    return _summary(record)


def _task_main(payload: dict[str, Any], nice: int | None, connection: Any) -> None:
    """Worker process entry: ignore Ctrl-C, run the task, send its summary back."""
    _worker_init(nice)
    os.environ[RUN_MARKER_ENV] = str(payload["key"])  # inherited by the solver's process tree
    summary = execute_task(payload)
    try:
        connection.send(summary)
    finally:
        connection.close()


def _death_reason(exitcode: int | None) -> str:
    if exitcode is not None and exitcode < 0:
        number = -exitcode
        try:
            name = signal.Signals(number).name
        except ValueError:  # pragma: no cover
            name = f"signal {number}"
        hint = " (the out-of-memory killer?)" if number == signal.SIGKILL else ""
        return f"the worker was killed by {name}{hint}"
    return f"the worker exited with code {exitcode} before recording the run"


def _kill_orphans(key: str) -> int:
    """Kill the process groups the worker of run ``key`` left behind (its solver); Linux only.

    Only processes that carry the worker's mark in their environment (``RUN_MARKER_ENV``) are
    touched: the solver and whatever it spawned. Working in the run's directory is not enough --
    a shell following ``solver.log`` there is none of our business.
    """
    proc = Path("/proc")
    if not proc.is_dir():  # pragma: no cover - not Linux
        return 0
    marker = f"{RUN_MARKER_ENV}={key}".encode()
    own_group = os.getpgid(0)
    groups: set[int] = set()
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if marker not in (entry / "environ").read_bytes().split(b"\0"):
                continue
            group = os.getpgid(int(entry.name))
        except (OSError, ValueError):
            continue
        if group != own_group:
            groups.add(group)
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):  # pragma: no cover
            pass
    return len(groups)


@dataclass
class _Worker:
    task: Task
    process: Any
    connection: Any

    def summary(self) -> dict[str, Any] | None:
        try:
            if self.connection.poll():
                return self.connection.recv()
        except (EOFError, OSError):
            pass
        return None


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
    context = mp.get_context("spawn")

    def _handle(signum: int, frame: Any) -> None:
        if signum == signal.SIGTERM:
            # Whoever sends SIGTERM expects the process gone soon (often SIGKILL follows): abort now.
            state["stop"] = state["abort"] = True
            progress("SIGTERM: aborting the running tasks (they leave no record; the next run redoes them)")
        elif state["stop"]:
            state["abort"] = True
            progress("second interrupt: aborting the running tasks")
        else:
            state["stop"] = True
            progress("interrupt: no new task starts; the running ones finish (interrupt again to abort)")

    previous = {sig: signal.signal(sig, _handle) for sig in (signal.SIGINT, signal.SIGTERM)}
    nice = tasks[0].nice if tasks else None
    running: dict[Any, _Worker] = {}
    pending = list(tasks)
    # Runs of each (instance, objective) that a warm start waits for: not started, or running.
    cold_open: dict[tuple[str, str], int] = {}
    for task in tasks:
        if task.warm_start != "best":
            cold_open[task.group] = cold_open.get(task.group, 0) + 1

    def _ready(task: Task) -> bool:
        return task.warm_start != "best" or not cold_open.get(task.group)

    def _start(task: Task) -> None:
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_task_main, args=(task.to_payload(), nice, sender), name=f"bks-campaign {task.key}"
        )
        with _signals_blocked():
            process.start()
        sender.close()
        running[process.sentinel] = _Worker(task=task, process=process, connection=receiver)
        progress(f"start {task.key} ({task.time_limit_s}s, ~{task.memory_gb:.1f} GB)")

    def _finish(worker: _Worker) -> None:
        worker.process.join()
        summary = worker.summary()
        worker.connection.close()
        task = worker.task
        if task.warm_start != "best":
            cold_open[task.group] -= 1  # whatever its outcome, nobody waits for it any more
        if summary is None:
            killed = _kill_orphans(task.key)
            terminated = worker.process.exitcode in (-signal.SIGTERM, 128 + signal.SIGTERM)
            if state["abort"] or terminated:
                # An aborted or terminated run (SIGTERM to the whole group: systemd, a scheduler)
                # leaves no record: the next `run` picks it up.
                progress(f"stop  {task.key}: interrupted, no record")
                return
            reason = _death_reason(worker.process.exitcode)
            if killed:
                reason += f"; its solver was left running and has been killed ({killed} process group(s))"
            record = _base_record(task, datetime.now(UTC).isoformat(timespec="seconds"))
            record.update(status="failed", valid=False, error=reason, finished=datetime.now(UTC).isoformat(timespec="seconds"))
            atomic_write_json(run_path(out, task.key), record)
            summary = _summary(record)
            failures.append(summary)
        results.append(summary)
        cost = summary.get("validated_cost")
        detail = f" cost {cost}" if cost is not None else (f" ({summary['error']})" if summary.get("error") else "")
        progress(f"done  {task.key}: {summary.get('status')}{detail}")

    try:
        while pending or running:
            if state["abort"]:
                break
            while pending and not state["stop"] and len(running) < workers:
                in_use = sum(worker.task.memory_gb for worker in running.values())
                candidate = next((task for task in pending if _ready(task)), None)
                if candidate is None:
                    break  # only warm starts are left, waiting for the runs they start from
                if running and in_use + candidate.memory_gb > memory_gb:
                    # The longest task does not fit: start the first smaller one that does.
                    fitting = next((task for task in pending if _ready(task) and in_use + task.memory_gb <= memory_gb), None)
                    if fitting is None:
                        break
                    candidate = fitting
                pending.remove(candidate)
                _start(candidate)
            if not running:
                break
            for sentinel in mp.connection.wait(list(running), timeout=1.0):
                _finish(running.pop(sentinel))
    finally:
        if running:
            # Abort (or an unexpected error here): SIGTERM lets each worker kill its solver's group.
            for worker in running.values():
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(worker.process.pid, signal.SIGTERM)
            deadline = time.monotonic() + ABORT_GRACE_S
            for worker in running.values():
                worker.process.join(max(0.0, deadline - time.monotonic()))
                if worker.process.is_alive():
                    worker.process.kill()
                    worker.process.join()
                _kill_orphans(worker.task.key)
                worker.connection.close()
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
