"""Baseline, harvest and reports of a campaign.

- ``write_baseline`` freezes the stored BKS of every selected instance before
  the campaign (``baseline.csv``) and refuses to overwrite it: gains are
  measured against what was published, not against the campaign's own writes.
- ``harvest`` is the only step that writes BKS files. It takes the best valid
  run per (instance, objective), skips runs whose instance file changed since
  the run (``instance_sha256``), offers it to the BKS store only if it beats
  the baseline (``--offer-all`` offers every best run), one process per
  instance, credited to ``campaign.authors``. The store itself re-validates and
  keeps the incumbent on a tie.
- ``write_reports`` writes ``ledger.csv`` (one row per run) and
  ``cross-solver.csv`` (per instance: best per kind, gap to the best, seed
  spread).
"""

from __future__ import annotations

import csv
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mamut_routing_tools.bks_campaign.archive import load_runs
from mamut_routing_tools.bks_campaign.spec import CampaignSpec

__all__ = ["BASELINE_FIELDS", "LEDGER_FIELDS", "harvest", "read_baseline", "write_baseline", "write_reports"]

BASELINE_FIELDS = ["instance_id", "objective_function", "cost", "num_routes", "method", "bks_sha256", "bks_path", "recorded"]
LEDGER_FIELDS = [
    "instance_id", "objective_function", "n", "kind", "solver", "method", "solver_version", "seed", "time_limit_s",
    "wall_time", "status", "valid", "validated_cost", "num_routes", "baseline_cost", "vs_baseline_pct", "is_best",
    "bks_action", "cost_exact", "check_error", "error", "host", "machine", "key",
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
    if record.get("objective_function") == "HierarchicalVehicleCost":
        return (int(record.get("num_routes") or 0), float(record["validated_cost"]))
    return (float(record["validated_cost"]),)


def _beats_baseline(record: dict[str, Any], baseline: dict[str, Any] | None) -> bool:
    if baseline is None or baseline.get("cost") in ("", None):
        return True
    reference = {"objective_function": record["objective_function"], "validated_cost": float(baseline["cost"]),
                 "num_routes": int(baseline["num_routes"] or 0)}
    return _objective_key(record) < _objective_key(reference)


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
    from mamut_routing_lib.solvers.provenance import file_sha256

    runs = load_runs(spec.out, with_routes=True)
    baseline = read_baseline(spec.out)
    results: list[dict[str, Any]] = []
    jobs = []
    for group, record in sorted(_best_runs(runs).items()):
        if record.get("instance_sha256") and file_sha256(record["instance_path"]) != record["instance_sha256"]:
            results.append({"key": record["key"], "instance_id": group[0], "action": "skipped",
                            "reason": "the instance file changed after the run"})
            continue
        if not offer_all and not _beats_baseline(record, baseline.get(group)):
            results.append({"key": record["key"], "instance_id": group[0], "action": "not_offered",
                            "reason": "does not beat the baseline"})
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
        (spec.out / "harvest.json").write_text(json.dumps(results, indent=1, default=str) + "\n", encoding="utf-8")
    return results


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def write_reports(spec: CampaignSpec) -> tuple[Path, Path]:
    """``ledger.csv`` (one row per run) and ``cross-solver.csv`` (per instance and objective)."""
    runs = load_runs(spec.out)
    baseline = read_baseline(spec.out)
    harvest_path = spec.out / "harvest.json"
    harvested = {}
    if harvest_path.is_file():
        harvested = {item["key"]: item.get("action") for item in json.loads(harvest_path.read_text(encoding="utf-8"))}
    best = {group: record["key"] for group, record in _best_runs(runs).items()}

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
                "bks_action": harvested.get(record.get("key"), ""),
            })

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in runs:
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
            overall = min(valid, key=lambda record: float(record["validated_cost"]), default=None)
            row: dict[str, Any] = {
                "instance_id": group[0],
                "objective_function": group[1],
                "baseline_cost": baseline.get(group, {}).get("cost"),
                "overall_best": overall["validated_cost"] if overall else None,
                "winner": overall["kind"] if overall else None,
            }
            for kind in kinds:
                costs = [float(record["validated_cost"]) for record in valid if record.get("kind") == kind]
                if not costs:
                    row[f"{kind}_valid_runs"] = 0
                    continue
                kind_best = min(costs)
                routes = next(record.get("num_routes") for record in valid if record.get("kind") == kind and float(record["validated_cost"]) == kind_best)
                row[f"{kind}_best"] = kind_best
                row[f"{kind}_routes"] = routes
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
