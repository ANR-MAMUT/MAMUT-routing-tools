"""Baseline, harvest and reports of a campaign.

- ``write_baseline`` freezes the stored BKS of every selected instance before
  the campaign (``baseline.csv``) and refuses to overwrite it: gains are
  measured against what was published, not against the campaign's own writes.
- ``harvest`` is the only step that writes BKS files. It sets aside the runs
  whose instance file changed since (``instance_sha256``), takes the best valid
  remaining run per (instance, objective) (fewer routes first under
  ``HierarchicalVehicleCost``), and offers it to the BKS store unless it is
  worse than the baseline by more than float noise (``--offer-all`` offers
  every best run), one process per instance, credited to ``campaign.authors``.
  Ties and near-ties are the store's call: it re-validates, compares exact
  decimal costs and keeps the incumbent on a tie. Every harvest is kept in
  ``harvests/<NNNN>-<time>.json``; ``harvest.json`` is the latest one.
- ``write_reports`` writes ``ledger.csv`` (one row per run; ``bks_action`` is
  what the store did with the run in any harvest, ``created`` over a later
  ``kept_existing``) and
  ``cross-solver.csv`` (per instance: best per kind, gap to the best, seed
  spread), ranking by the objective like the harvest.
"""

from __future__ import annotations

import csv
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mamut_routing_tools.bks_campaign.archive import atomic_write_json, load_runs
from mamut_routing_tools.bks_campaign.plan import objective_rank
from mamut_routing_tools.bks_campaign.spec import CampaignSpec

__all__ = ["BASELINE_FIELDS", "LEDGER_FIELDS", "harvest", "harvest_history", "read_baseline", "write_baseline", "write_reports"]

BASELINE_FIELDS = ["instance_id", "objective_function", "cost", "num_routes", "method", "bks_sha256", "bks_path", "recorded"]
LEDGER_FIELDS = [
    "instance_id", "objective_function", "n", "kind", "solver", "method", "solver_version", "seed", "time_limit_s",
    "wall_time", "status", "valid", "validated_cost", "num_routes", "baseline_cost", "vs_baseline_pct", "is_best",
    "bks_action", "bks_action_at", "cost_exact", "check_error", "error", "host", "machine", "key",
]


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #


def write_baseline(spec: CampaignSpec, *, force: bool = False) -> Path:
    """Snapshot the stored BKS of the selected instances into ``baseline.csv``."""
    from mamut_routing_lib.artifacts import get_bks_path_for_instance, load_benchmark_instance, load_bks
    from mamut_routing_lib.objectives import base_objective
    from mamut_routing_lib.solvers.provenance import file_sha256

    from mamut_routing_tools.bks_campaign.plan import select_instances

    path = spec.out / "baseline.csv"
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists: the baseline must predate the campaign (pass --force to replace it)")
    rows = []
    recorded = datetime.now(UTC).isoformat(timespec="seconds")
    for item in select_instances(spec):
        instance = load_benchmark_instance(item.instance_path)
        objective = spec.select.objective.get(item.problem_type) or base_objective(instance)
        bks_path = get_bks_path_for_instance(item.instance_path, objective)
        row = {
            "instance_id": item.instance_id,
            "objective_function": objective.value,
            "cost": "",
            "num_routes": "",
            "method": "",
            "bks_sha256": "",
            "bks_path": str(bks_path),
            "recorded": recorded,
        }
        if bks_path.is_file():
            bks = load_bks(bks_path)
            row.update(cost=bks.cost, num_routes=bks.num_routes, method=bks.metadata.get("method") or "", bks_sha256=file_sha256(bks_path))
        rows.append(row)
    spec.out.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASELINE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_baseline(out: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = out / "baseline.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {(row["instance_id"], row["objective_function"]): row for row in csv.DictReader(handle)}


# --------------------------------------------------------------------------- #
# Harvest
# --------------------------------------------------------------------------- #


def _best_runs(runs: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Lowest checker cost per (instance, objective) among valid runs (ties: the first key)."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for record in sorted(runs, key=lambda item: item.get("key", "")):
        if not record.get("valid") or record.get("validated_cost") is None:
            continue
        group = (record["instance_id"], record["objective_function"])
        current = best.get(group)
        if current is None or _objective_key(record) < _objective_key(current):
            best[group] = record
    return best


def _objective_key(record: dict[str, Any]) -> tuple:
    return objective_rank(str(record.get("objective_function")), record["validated_cost"], int(record.get("num_routes") or 0))


def _current_runs(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(runs of the instance files as they are now, runs of a file that changed since)."""
    from mamut_routing_lib.solvers.provenance import file_sha256

    shas: dict[str, str | None] = {}
    current, stale = [], []
    for record in runs:
        path = record.get("instance_path")
        if path and path not in shas:
            shas[path] = file_sha256(path) if Path(path).is_file() else None
        recorded = record.get("instance_sha256")
        (stale if recorded and path and shas.get(path) != recorded else current).append(record)
    return current, stale


#: Relative margin of the baseline pre-filter. A float total depends on the order and the arithmetic
#: that summed the arcs (by at most about n * 2**-53 of the cost, 5e-13 at n = 4000), and the
#: baseline may have been priced by another checker release than the runs; the store compares exact
#: decimal costs. So the pre-filter only drops runs that are worse by more than this margin and
#: leaves ties and near-ties to the store.
BASELINE_MARGIN = 1e-10


def _may_beat_baseline(record: dict[str, Any], baseline: dict[str, Any] | None) -> bool:
    """False only when the run is worse than the baseline beyond float noise (``BASELINE_MARGIN``)."""
    if baseline is None or baseline.get("cost") in ("", None):
        return True
    if record.get("objective_function") == "HierarchicalVehicleCost":
        routes, reference_routes = int(record.get("num_routes") or 0), int(baseline.get("num_routes") or 0)
        if routes != reference_routes:
            return routes < reference_routes
    cost, reference = float(record["validated_cost"]), float(baseline["cost"])
    return cost - reference <= BASELINE_MARGIN * max(1.0, abs(reference))


def _submit_one(job: dict[str, Any]) -> dict[str, Any]:
    """One instance's store call (run in its own process: the store is an unlocked read/compare/write)."""
    from mamut_routing_lib.enums import ObjectiveFunction
    from mamut_routing_lib.solutions import preview_bks_update, submit_bks

    record = job["record"]
    objective = ObjectiveFunction(record["objective_function"])
    if job["dry_run"]:
        preview = preview_bks_update(record["instance_path"], record["routes"], objective_function=objective)
        return {"key": record["key"], "instance_id": record["instance_id"], "action": preview["action"], "tie": preview["tie"],
                "candidate_cost": preview["candidate_cost"], "path": preview["path"]}
    try:
        update = submit_bks(
            record["instance_path"],
            record["routes"],
            authors=job["authors"],
            objective_function=objective,
            metadata=job["metadata"],
        )
    except ValueError as error:
        return {"key": record["key"], "instance_id": record["instance_id"], "action": "error", "error": str(error)}
    return {"key": record["key"], "instance_id": record["instance_id"], "action": update.action, "tie": update.tie,
            "candidate_cost": update.candidate_cost, "path": str(update.path)}


def _bks_metadata(spec: CampaignSpec, record: dict[str, Any]) -> dict[str, Any]:
    provenance = record.get("provenance") or {}
    metadata = {
        "campaign": spec.campaign.name,
        "method": record.get("method") or record.get("solver"),
        "solver": record.get("solver"),
        "solver_version": record.get("solver_version"),
        "seed": record.get("seed"),
        "time_limit_s": record.get("time_limit_s"),
        "wall_time": record.get("wall_time"),
        "run_key": record.get("key"),
        "machine": record.get("machine"),
        "cost_exact": record.get("cost_exact"),
        "executable_sha256": provenance.get("executable_sha256"),
        "input_sha256": provenance.get("input_sha256"),
        "warm_start": record.get("warm_start_key"),
    }
    return {key: value for key, value in metadata.items() if value is not None}


def harvest(
    spec: CampaignSpec,
    *,
    offer_all: bool = False,
    dry_run: bool = False,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Offer each instance's best run to the BKS store (the campaign's only writer)."""
    from mamut_routing_lib.artifacts import load_benchmark_instance

    runs, stale = _current_runs(load_runs(spec.out, with_routes=True))
    baseline = read_baseline(spec.out)
    results: list[dict[str, Any]] = []
    jobs = []
    best = _best_runs(runs)
    for group, record in sorted(_best_runs(stale).items()):
        if group not in best:  # only runs of an older instance file: nothing current to offer
            results.append({"key": record["key"], "instance_id": group[0], "action": "skipped",
                            "reason": "the instance file changed after the run (plan runs it again)"})
    for group, record in sorted(best.items()):
        if not offer_all and not _may_beat_baseline(record, baseline.get(group)):
            results.append({"key": record["key"], "instance_id": group[0], "action": "not_offered",
                            "reason": "worse than the baseline"})
            continue
        instance = load_benchmark_instance(record["instance_path"])
        metadata = getattr(instance, "metadata", None)
        lower = metadata.get("num_vehicles_lb") if isinstance(metadata, dict) else getattr(metadata, "num_vehicles_lb", None)
        if lower and int(record.get("num_routes") or 0) < int(lower):
            results.append({"key": record["key"], "instance_id": group[0], "action": "skipped",
                            "reason": f"{record.get('num_routes')} routes < published lower bound {lower}"})
            continue
        jobs.append({"record": record, "authors": spec.campaign.authors, "metadata": _bks_metadata(spec, record), "dry_run": dry_run})
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp

        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            results.extend(pool.map(_submit_one, jobs))
    else:
        results.extend(_submit_one(job) for job in jobs)
    if not dry_run:
        _record_harvest(spec.out, results)
    return results


#: Ledger ``bks_action`` precedence across harvests: what the store did that matters most.
_ACTION_RANK = {"replaced": 0, "created": 1, "kept_existing": 2, "error": 3, "not_offered": 4, "skipped": 5}


def _record_harvest(out: Path, results: list[dict[str, Any]]) -> Path:
    """Write ``harvests/<NNNN>-<UTC time>.json`` (kept, numbered in order) and ``harvest.json`` (the latest)."""
    history = out / "harvests"
    history.mkdir(parents=True, exist_ok=True)
    latest = out / "harvest.json"
    numbers = [int(path.name[:4]) for path in history.glob("[0-9][0-9][0-9][0-9]-*.json")]
    if latest.is_file() and not numbers:
        # A harvest from before the history existed: keep it first, under its file time.
        when = datetime.fromtimestamp(latest.stat().st_mtime, UTC)
        legacy = json.loads(latest.read_text(encoding="utf-8"))
        atomic_write_json(
            history / f"0000-{when:%Y-%m-%dT%H-%M-%SZ}.json", {"harvested": when.isoformat(timespec="seconds"), "results": legacy}
        )
        numbers = [0]
    now = datetime.now(UTC)
    path = history / f"{max(numbers, default=0) + 1:04d}-{now:%Y-%m-%dT%H-%M-%SZ}.json"
    atomic_write_json(path, {"harvested": now.isoformat(timespec="seconds"), "results": results})
    latest.write_text(json.dumps(results, indent=1, default=str) + "\n", encoding="utf-8")
    return path


def harvest_history(out: Path) -> list[dict[str, Any]]:
    """Every recorded harvest, oldest first: ``{"harvested": time, "results": [...]}``."""
    history = sorted((out / "harvests").glob("[0-9][0-9][0-9][0-9]-*.json"))
    if history:
        return [json.loads(path.read_text(encoding="utf-8")) for path in history]
    latest = out / "harvest.json"  # a campaign harvested before the history existed
    if latest.is_file():
        return [{"harvested": None, "results": json.loads(latest.read_text(encoding="utf-8"))}]
    return []


def _harvested_actions(out: Path) -> dict[str, tuple[str, str | None]]:
    """run key -> (most significant store action over every harvest, when it happened)."""
    actions: dict[str, tuple[str, str | None]] = {}
    for entry in harvest_history(out):
        for item in entry.get("results", []):
            key, action = item.get("key"), item.get("action")
            if not key or not action:
                continue
            known = actions.get(key)
            if known is None or _ACTION_RANK.get(action, 9) <= _ACTION_RANK.get(known[0], 9):
                actions[key] = (action, entry.get("harvested"))
    return actions


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def write_reports(spec: CampaignSpec) -> tuple[Path, Path]:
    """``ledger.csv`` (one row per run) and ``cross-solver.csv`` (per instance and objective)."""
    spec.out.mkdir(parents=True, exist_ok=True)  # before any `run`: empty reports, not a traceback
    runs = load_runs(spec.out)
    current, _stale = _current_runs(runs)
    baseline = read_baseline(spec.out)
    harvested = _harvested_actions(spec.out)
    best = {group: record["key"] for group, record in _best_runs(current).items()}

    ledger = spec.out / "ledger.csv"
    with ledger.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in sorted(runs, key=lambda item: (item.get("instance_id", ""), item.get("kind", ""), item.get("seed", 0))):
            group = (record.get("instance_id"), record.get("objective_function"))
            reference = baseline.get(group, {}).get("cost")
            cost = record.get("validated_cost")
            gap = None
            if cost is not None and reference not in (None, ""):
                gap = round(100.0 * (float(cost) - float(reference)) / float(reference), 4) if float(reference) else None
            writer.writerow({
                **record,
                "baseline_cost": reference,
                "vs_baseline_pct": gap,
                "is_best": best.get(group) == record.get("key"),
                "bks_action": harvested.get(record.get("key"), ("", None))[0],
                "bks_action_at": harvested.get(record.get("key"), ("", None))[1] or "",
            })

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in current:  # runs of an instance file that changed since are not compared
        groups.setdefault((record.get("instance_id"), record.get("objective_function")), []).append(record)
    kinds = sorted({record.get("kind") for record in runs if record.get("kind")})
    compare = spec.out / "cross-solver.csv"
    fields = ["instance_id", "objective_function", "baseline_cost", "overall_best", "winner"]
    for kind in kinds:
        fields += [f"{kind}_best", f"{kind}_routes", f"{kind}_gap_pct", f"{kind}_spread_pct", f"{kind}_valid_runs"]
    with compare.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group, records in sorted(groups.items()):
            valid = [record for record in records if record.get("valid") and record.get("validated_cost") is not None]
            overall = min(valid, key=_objective_key, default=None)
            row: dict[str, Any] = {
                "instance_id": group[0],
                "objective_function": group[1],
                "baseline_cost": baseline.get(group, {}).get("cost"),
                "overall_best": overall["validated_cost"] if overall else None,
                "winner": overall["kind"] if overall else None,
            }
            for kind in kinds:
                kind_runs = [record for record in valid if record.get("kind") == kind]
                if not kind_runs:
                    row[f"{kind}_valid_runs"] = 0
                    continue
                costs = [float(record["validated_cost"]) for record in kind_runs]
                kind_best_run = min(kind_runs, key=_objective_key)
                kind_best = float(kind_best_run["validated_cost"])
                row[f"{kind}_best"] = kind_best
                row[f"{kind}_routes"] = kind_best_run.get("num_routes")
                row[f"{kind}_gap_pct"] = round(100.0 * (kind_best - float(overall["validated_cost"])) / float(overall["validated_cost"]), 4) if overall and float(overall["validated_cost"]) else None
                row[f"{kind}_spread_pct"] = round(100.0 * (max(costs) - kind_best) / kind_best, 4) if kind_best else None
                row[f"{kind}_valid_runs"] = len(costs)
            writer.writerow(row)
    return ledger, compare


def campaign_status(spec: CampaignSpec) -> dict[str, Any]:
    runs = load_runs(spec.out)
    by_status: dict[str, int] = {}
    for record in runs:
        by_status[str(record.get("status"))] = by_status.get(str(record.get("status")), 0) + 1
    wall = [float(record["wall_time"]) for record in runs if record.get("wall_time") is not None]
    done = spec.out / "run.DONE"
    return {
        "runs": len(runs),
        "valid": sum(1 for record in runs if record.get("valid")),
        "by_status": by_status,
        "median_wall_time_s": round(statistics.median(wall), 2) if wall else None,
        "baseline": (spec.out / "baseline.csv").is_file(),
        "last_run": json.loads(done.read_text()) if done.is_file() else None,
    }
