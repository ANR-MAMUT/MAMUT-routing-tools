"""The run archive: one JSON record per run, written atomically, the unit of resumption.

Layout under ``campaign.out``::

    runs/<key>.json      one record per finished run (winner or not), schema_version 1
    raw/<key>/           the solver's log and solution file (external solvers)
    baseline.csv         the stored BKS before the campaign (``baseline``)
    ledger.csv           one row per run (``report``)
    cross-solver.csv     per instance: best per kind, gaps, seed spread (``report``)
    harvest.json         what the single-writer harvest did
    run.DONE             summary of the last ``run`` invocation

A run is *done* when its record exists; an interrupted run leaves no record and
is picked up by the next ``run``. The run key is
``<instance_id>__<objective>__<kind>__seed<S>__t<T>``: the budget is part of it,
so changing ``time_scale`` schedules new runs instead of mixing budgets.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "SCHEMA_VERSION",
    "atomic_write_json",
    "iter_runs",
    "load_runs",
    "raw_dir",
    "run_key",
    "run_path",
    "runs_dir",
]

SCHEMA_VERSION = 1
_UNSAFE = re.compile(r"[^A-Za-z0-9._=-]+")


def run_key(instance_id: str, objective: str, kind: str, seed: int, time_limit_s: int) -> str:
    raw = f"{instance_id}__{objective}__{kind}__seed{int(seed)}__t{int(time_limit_s)}"
    return _UNSAFE.sub("-", raw)


def runs_dir(out: Path) -> Path:
    return out / "runs"


def raw_dir(out: Path, key: str) -> Path:
    return out / "raw" / key


def run_path(out: Path, key: str) -> Path:
    return runs_dir(out) / f"{key}.json"


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(payload, separators=(",", ":"), default=str), encoding="utf-8")
    os.replace(temporary, path)


def iter_runs(out: Path) -> Iterator[dict[str, Any]]:
    directory = runs_dir(out)
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        record["_file"] = str(path)
        yield record


def load_runs(out: Path, *, with_routes: bool = False) -> list[dict[str, Any]]:
    records = []
    for record in iter_runs(out):
        if not with_routes:
            record.pop("routes", None)
            record.pop("trajectory", None)
        records.append(record)
    return records
