"""Persistent background jobs for the local workbench.

The GUI server is deliberately small, but generation, OSM acquisition and
solving are not request-sized operations.  This module provides an in-process
worker pool with durable state and logs.  A server restart marks unfinished
work as interrupted; completed results remain queryable.
"""

from __future__ import annotations

import json
import threading
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from mamut_routing_tools.workspace import jobs_dir, logs_dir

JobRunner = Callable[["JobContext"], dict[str, Any]]
_TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class JobCancelled(RuntimeError):
    """Raised cooperatively when a job observes a cancellation request."""


class JobContext:
    def __init__(self, manager: "JobManager", job_id: str):
        self._manager = manager
        self.job_id = job_id

    def progress(
        self,
        message: str,
        *,
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        self._manager._progress(self.job_id, message, current=current, total=total)

    def log(self, message: str) -> None:
        self._manager._log(self.job_id, message)

    def check_cancelled(self) -> None:
        if self._manager.cancel_requested(self.job_id):
            raise JobCancelled("Cancellation requested")


class JobManager:
    """Thread-backed job manager with JSON records and append-only logs."""

    def __init__(self, workspace: Path, *, max_workers: int = 2):
        self.workspace = workspace
        self._root = jobs_dir(workspace)
        self._logs = logs_dir(workspace)
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, Future[None]] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mamut-job")
        self._load_records()

    def _record_path(self, job_id: str) -> Path:
        return self._root / f"{job_id}.json"

    def _log_path(self, job_id: str) -> Path:
        return self._logs / f"{job_id}.log"

    def _load_records(self) -> None:
        for path in sorted(self._root.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                job_id = str(record["job_id"])
            except (OSError, ValueError, KeyError):
                continue
            if record.get("status") not in _TERMINAL:
                record["status"] = "interrupted"
                record["finished_at"] = _now()
                record["error"] = "GUI server stopped before this job finished."
                _atomic_json(path, record)
            self._records[job_id] = record

    def _persist(self, record: dict[str, Any]) -> None:
        _atomic_json(self._record_path(str(record["job_id"])), record)

    def submit(self, kind: str, payload: dict[str, Any], runner: JobRunner) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        record: dict[str, Any] = {
            "job_id": job_id,
            "kind": kind,
            "status": "queued",
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "progress": {"message": "Queued", "current": None, "total": None},
            "cancel_requested": False,
            "request": payload,
            "result": None,
            "error": None,
        }
        with self._lock:
            self._records[job_id] = record
            self._persist(record)
            self._futures[job_id] = self._executor.submit(self._run, job_id, runner)
        return self.get(job_id)

    def _run(self, job_id: str, runner: JobRunner) -> None:
        with self._lock:
            record = self._records[job_id]
            if record["cancel_requested"]:
                record["status"] = "cancelled"
                record["finished_at"] = _now()
                record["progress"] = {"message": "Cancelled", "current": None, "total": None}
                self._persist(record)
                return
            record["status"] = "running"
            record["started_at"] = _now()
            record["progress"] = {"message": "Starting", "current": None, "total": None}
            self._persist(record)

        context = JobContext(self, job_id)
        context.log(f"Job {job_id} ({self._records[job_id]['kind']}) started")
        try:
            result = runner(context)
            context.check_cancelled()
        except JobCancelled as error:
            with self._lock:
                record = self._records[job_id]
                record["status"] = "cancelled"
                record["finished_at"] = _now()
                record["progress"] = {"message": "Cancelled", "current": None, "total": None}
                record["error"] = str(error)
                self._persist(record)
            context.log(str(error))
        except Exception as error:  # noqa: BLE001 - durable boundary for job failures
            context.log(traceback.format_exc())
            with self._lock:
                record = self._records[job_id]
                record["status"] = "failed"
                record["finished_at"] = _now()
                record["progress"] = {"message": "Failed", "current": None, "total": None}
                record["error"] = str(error)
                self._persist(record)
        else:
            with self._lock:
                record = self._records[job_id]
                record["status"] = "completed"
                record["finished_at"] = _now()
                record["progress"] = {"message": "Completed", "current": 1, "total": 1}
                record["result"] = result
                self._persist(record)
            context.log("Job completed")

    def _progress(
        self,
        job_id: str,
        message: str,
        *,
        current: int | None,
        total: int | None,
    ) -> None:
        with self._lock:
            record = self._records[job_id]
            record["progress"] = {"message": message, "current": current, "total": total}
            self._persist(record)

    def _log(self, job_id: str, message: str) -> None:
        line = f"{_now()} {message.rstrip()}\n"
        with self._lock:
            with self._log_path(job_id).open("a", encoding="utf-8") as handle:
                handle.write(line)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._records:
                raise KeyError(job_id)
            return json.loads(json.dumps(self._records[job_id]))

    def list(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            records = sorted(
                self._records.values(), key=lambda value: str(value.get("created_at") or ""), reverse=True
            )
            return json.loads(json.dumps(records[: max(1, min(limit, 200))]))

    def log_text(self, job_id: str) -> str:
        if job_id not in self._records:
            raise KeyError(job_id)
        path = self._log_path(job_id)
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            return bool(self._records[job_id].get("cancel_requested"))

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._records:
                raise KeyError(job_id)
            record = self._records[job_id]
            if record["status"] in _TERMINAL:
                return self.get(job_id)
            record["cancel_requested"] = True
            record["progress"] = {
                "message": "Cancellation requested; waiting for the current safe checkpoint",
                "current": record.get("progress", {}).get("current"),
                "total": record.get("progress", {}).get("total"),
            }
            future = self._futures.get(job_id)
            if future is not None and future.cancel():
                record["status"] = "cancelled"
                record["finished_at"] = _now()
                record["progress"]["message"] = "Cancelled"
            self._persist(record)
            return self.get(job_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
