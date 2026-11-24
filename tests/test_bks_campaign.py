"""BKS campaigns end to end on a toy tree: spec, plan, resumable run, baseline, single-writer harvest, reports."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mamut_routing_lib.sidecars import CollectionMarker, save_collection_marker
from mamut_routing_lib.models import BenchmarkInstanceVRPTW
from mamut_routing_tools.bks_campaign import (
    build_plan,
    harvest,
    load_runs,
    load_spec,
    parse_seeds,
    run_tasks,
    write_baseline,
    write_reports,
)
from mamut_routing_tools.cli import app

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")

SOLVERS = """schema_version = 1
[solvers.fake]
label = "Fake"
runtime = "python"
input_format = "lkh"
output_format = "lkh-mtsp"
problem_types = ["CVRP", "VRPTW"]
symmetric_only = false
argv = ["{python}", "-m", "mamut_routing_lib.solvers.external._fake_solver", "--params", "{params}"]
"""


def _instance(name: str, shift: int) -> BenchmarkInstanceVRPTW:
    return BenchmarkInstanceVRPTW(
        instance_name=name,
        instance_origin="Solomon1987",
        benchmark_name="Dimacs2021",
        num_customers=3,
        vehicle_capacity=10,
        coordinates=[(0, 0), (1, 1), (2, 2), (3, 3)],
        demands=[0, 3, 4, 2],
        service_times=[0, 1, 1, 1],
        time_windows=[(0, 1000), (0, 500), (0, 500), (0, 500)],
        depot=0,
        arc_costs=[
            [0, 10 + shift, 20, 30],
            [10 + shift, 0, 12, 22],
            [20, 12, 0, 11],
            [30, 22, 11, 0],
        ],
        metadata={"metric_variant": "euclidean", "authors": "toy"},
    )


@pytest.fixture
def campaign(tmp_path: Path) -> Path:
    benchmarks = tmp_path / "benchmarks"
    save_collection_marker(CollectionMarker(family="Dimacs2021", layout_version=2), benchmarks / "Dimacs2021")
    for name, shift in (("A1", 0), ("B2", 5)):
        path = benchmarks / "Dimacs2021" / "VRPTW" / "n=3" / f"{name}.vrp.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_instance(name, shift).model_dump(mode="json")))
    (tmp_path / "solvers.toml").write_text(SOLVERS)
    spec = tmp_path / "campaign.toml"
    spec.write_text(
        """
[campaign]
name = "toy-campaign"
out = "out"
authors = "Campaign Tester"
solvers_config = "solvers.toml"

[select]
benchmarks_dir = "benchmarks"
problem_types = ["VRPTW"]

[budget]
fixed_s = 2

[[runs]]
solver = "fake"
seeds = "1-2"

[resources]
workers = 2
memory_gb = 8
nice = 0
"""
    )
    return spec


def test_seed_ranges() -> None:
    assert parse_seeds("1-3,7") == [1, 2, 3, 7]
    assert parse_seeds(5) == [5] and parse_seeds([4, 2]) == [4, 2]
    assert parse_seeds("1-3,2") == [1, 2, 3] and parse_seeds([4, 4, 2]) == [4, 2]  # no duplicate runs
    with pytest.raises(ValueError):
        parse_seeds("5-1")


def test_a_repeated_seed_is_planned_once(campaign: Path) -> None:
    campaign.write_text(campaign.read_text().replace('seeds = "1-2"', 'seeds = "1-2,2"'))
    keys = [task.key for task in build_plan(load_spec(campaign), only="A1").tasks]
    assert len(keys) == len(set(keys)) == 2  # two tasks sharing one raw directory would overwrite each other


def test_spec_rules(campaign: Path, tmp_path: Path) -> None:
    spec = load_spec(campaign)
    assert spec.out == (tmp_path / "out").resolve()
    assert spec.budget.seconds(3) == 2
    text = campaign.read_text()
    campaign.write_text(text.replace('out = "out"', 'out = "benchmarks/runs"'))
    with pytest.raises(ValueError, match="inside a benchmark tree"):
        load_spec(campaign)
    campaign.write_text(text.replace('authors = "Campaign Tester"', 'authors = " "'))
    with pytest.raises(ValueError, match="authors"):
        load_spec(campaign)
    campaign.write_text(text + '\n[[runs]]\nsolver = "fake"\nseeds = "3"\n')
    with pytest.raises(ValueError, match="distinct kind"):
        load_spec(campaign)


def test_budget_rule_reproduces_campaign_3() -> None:
    from mamut_routing_tools.bks_campaign.spec import BudgetSpec

    rule = BudgetSpec(per_customer_s=21.6, min_s=7200, max_s=86400)
    assert rule.seconds(100) == 7200 and rule.seconds(705) == 15228 and rule.seconds(5000) == 86400
    assert rule.seconds(705, factor=1.5) == 22842


def test_plan_lists_runs_unavailable_solvers_and_unsupported_pairs(campaign: Path) -> None:
    campaign.write_text(campaign.read_text() + '\n[[runs]]\nsolver = "filo2"\n\n[[runs]]\nsolver = "pyvrp"\nseeds = "1"\nkind = "pv"\n')
    spec = load_spec(campaign)
    plan = build_plan(spec)
    keys = [task.key for task in plan.tasks]
    assert len([task for task in plan.tasks if task.kind == "fake-seed"]) == 4
    assert any(key.endswith("__fake-seed__seed2__t2") for key in keys)
    assert "filo2" in plan.unavailable  # not on PATH in the test environment
    assert {task.kind for task in plan.tasks} == {"fake-seed", "pv"}
    summary = plan.summary(spec.resources.workers)
    assert summary["todo"] == 6 and summary["per_kind"] == {"fake-seed": 4, "pv": 2}


def test_run_resume_baseline_harvest_and_reports(campaign: Path) -> None:
    spec = load_spec(campaign)
    write_baseline(spec)
    with pytest.raises(FileExistsError):
        write_baseline(spec)

    plan = build_plan(spec)
    assert len(plan.tasks) == 4
    lines: list[str] = []
    summary = run_tasks(plan, out=spec.out, workers=2, memory_gb=8, progress=lines.append)
    assert summary["completed"] == 4 and summary["valid"] == 4 and not summary["interrupted"]
    assert (spec.out / "run.DONE").is_file()

    runs = load_runs(spec.out)
    assert len(runs) == 4 and all(run["valid"] and run["schema_version"] == 2 for run in runs)
    record = runs[0]
    assert record["solver"] == "fake" and record["host"] and record["machine"]
    assert record["check_profile"] == "binary64-exact-v1"
    assert record["authoritative_cost"] == {"numerator": str(int(record["validated_cost"])), "denominator": "1"}
    assert record["provenance"]["scale"]["exact"] is True
    assert record["solver_report"]["objective"] == "total-arc-cost"
    assert record["solver_report"]["value"] == str(int(record["solver_cost"]))
    assert "cost_exact" not in record and "solver_cost_raw" not in record
    assert not (spec.out / "raw" / record["key"] / "problem.vrp").exists()  # inputs are not kept
    assert (spec.out / "raw" / record["key"] / "solver.log").is_file()

    assert build_plan(spec).tasks == []  # resumption: every run has its record

    preview = harvest(spec, dry_run=True)
    assert {item["action"] for item in preview} == {"would_create"}
    assert not list((campaign.parent / "benchmarks").rglob("*.bks.*.json"))
    results = harvest(spec)
    assert sorted(item["action"] for item in results) == ["created", "created"]
    bks_files = sorted((campaign.parent / "benchmarks").rglob("*.bks.MonoCost.json"))
    assert len(bks_files) == 2
    stored = json.loads(bks_files[0].read_text())
    assert stored["metadata"]["authors"] == "Campaign Tester"
    assert stored["metadata"]["campaign"] == "toy-campaign" and stored["metadata"]["solver"] == "fake"
    assert "host" not in stored["metadata"] and stored["metadata"]["machine"]
    winner = next(run for run in runs if run["key"] == stored["metadata"]["run_key"])
    assert stored["metadata"]["solver_report"] == winner["solver_report"]
    assert stored["metadata"]["solver_cost"] == winner["solver_cost"]
    assert "reported_cost_matches_exported_model" not in stored["metadata"]

    again = harvest(spec)  # the baseline had no BKS, so the best runs are offered again: ties keep the incumbent
    assert {item["action"] for item in again} == {"kept_existing"} and all(item["tie"] for item in again)
    write_baseline(spec, force=True)  # now the baseline holds the campaign's own BKS
    tied = harvest(spec)  # ties are the store's call, not the float pre-filter's
    assert {item["action"] for item in tied} == {"kept_existing"} and all(item["tie"] for item in tied)

    ledger, compare = write_reports(spec)
    rows = list(csv.DictReader(ledger.open()))
    assert len(rows) == 4 and {row["kind"] for row in rows} == {"fake-seed"}
    assert float(rows[0]["solver_cost"]) == record["solver_cost"] and "cost_exact" not in rows[0]
    assert sum(row["is_best"] == "True" for row in rows) == 2
    table = list(csv.DictReader(compare.open()))
    assert len(table) == 2 and table[0]["winner"] == "fake-seed" and table[0]["fake-seed_valid_runs"] == "2"


@pytest.mark.parametrize("objective", ["MonoCost", "HierarchicalVehicleCost"])
def test_harvest_reprices_legacy_and_repeated_routes_on_a_display_tie(campaign: Path, objective: str) -> None:
    from mamut_routing_lib.artifacts import get_bks_path_for_instance, load_bks
    from mamut_routing_lib.enums import ObjectiveFunction
    from mamut_routing_lib.solvers.provenance import file_sha256

    spec = load_spec(campaign)
    path = next((campaign.parent / "benchmarks").rglob("A1.vrp.json"))
    instance = json.loads(path.read_text())
    costs = [[0] * 4 for _ in range(4)]
    costs[0][1], costs[1][2] = 1, 2**-55
    instance["arc_costs"] = costs
    path.write_text(json.dumps(instance))
    folder = spec.out / "runs"
    folder.mkdir(parents=True)
    base = dict(instance_id="A1", instance_path=str(path), instance_sha256=file_sha256(path),
                objective_function=objective, kind="demo", valid=True, validated_cost=1.0, num_routes=1)

    def write(key, routes, **fields):
        (folder / (key + ".json")).write_text(json.dumps({**base, "key": key, "routes": routes, **fields}))

    write("bad", [[1, 2]])  # an incomplete answer previously marked valid
    write("a", [[1, 2, 3]])  # exact 1 + 2**-55, displayed as 1.0
    write("b", [[1, 2, 3]], check_profile="legacy", validated_cost=0.5)
    # Schema 1 used solver_cost for two different units. Preserve the source and recheck its routes.
    write("z", [[1, 3, 2]], schema_version=1, solver_cost=999, solver_cost_raw=999, cost_exact=True)
    write("zz", [[1, 3, 2]], authoritative_cost={"numerator": "1", "denominator": "1"})
    if objective == "HierarchicalVehicleCost":
        write("cheaper", [[2, 1], [3]], validated_cost=0.0, num_routes=2)

    selected, = harvest(spec)
    assert (selected["key"], selected["action"]) == ("z", "created")
    bks_path = get_bks_path_for_instance(path, ObjectiveFunction(objective))
    bks = load_bks(bks_path)
    assert bks.routes == [[1, 3, 2]]
    assert {"solver_report", "solver_cost", "cost_exact"}.isdisjoint(bks.metadata)
    assert json.loads((folder / "z.json").read_text())["solver_cost"] == 999
    ledger, _ = write_reports(spec)
    with ledger.open() as handle:
        legacy = next(row for row in csv.DictReader(handle) if row["key"] == "z")
    assert legacy["solver_cost"] == ""
    assert harvest(spec)[0]["tie"]

    # The file digest identifies the frozen baseline routes even when its historical display is wrong.
    baseline = spec.out / "baseline.csv"
    with baseline.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["instance_id", "objective_function", "cost", "bks_sha256"])
        writer.writeheader()
        writer.writerow(dict(instance_id="A1", objective_function=objective, cost=-100, bks_sha256=file_sha256(bks_path)))
    assert harvest(spec)[0]["tie"]
    for key in ("z", "zz"):
        (folder / (key + ".json")).unlink()
    assert harvest(spec, dry_run=True)[0]["action"] == "not_offered"
    assert harvest(spec, offer_all=True)[0]["action"] == "kept_existing"
    bks_path.unlink()  # the frozen route bytes are unavailable
    assert harvest(spec)[0]["action"] == "created"


def test_harvest_verifies_current_manifest_before_ranking_archived_costs(campaign: Path) -> None:
    import shutil

    from mamut_routing_lib.artifacts import get_bks_path_for_instance
    from mamut_routing_lib.enums import ObjectiveFunction
    from mamut_routing_lib.manifests import manifest_path_for
    from mamut_routing_tools.static_manifest import stage_static

    spec = load_spec(campaign)
    run_tasks(build_plan(spec, only="A1"), out=spec.out, workers=1, memory_gb=8, progress=lambda line: None)
    assert harvest(spec)[0]["action"] == "created"
    root = campaign.parent / "benchmarks" / "Dimacs2021"
    path = next(root.rglob("A1.vrp.json"))
    stage = campaign.parent / "stage"
    stage_static(root, path.relative_to(root), stage, recipes={}, revision="r1")
    manifest = manifest_path_for(path)
    staged_manifest = stage / manifest.relative_to(root)
    shutil.copyfile(staged_manifest, manifest)
    bks = get_bks_path_for_instance(path, ObjectiveFunction.MONO_COST)
    bks.unlink()
    assert harvest(spec)[0]["action"] == "created"
    assert json.loads(bks.read_text())["metadata"]["standard"]["benchmark_revision"] == "r1"
    changed = json.loads(manifest.read_text())
    changed["materialization"]["matrix_sha256"] = "0" * 64
    manifest.write_text(json.dumps(changed))
    result, = harvest(spec)
    assert result["action"] == "error" and "matrix_sha256" in result["error"]


def test_harvest_skips_runs_of_a_changed_instance(campaign: Path) -> None:
    spec = load_spec(campaign)
    run_tasks(build_plan(spec, only="A1"), out=spec.out, workers=1, memory_gb=8, progress=lambda line: None)
    path = next((campaign.parent / "benchmarks").rglob("A1.vrp.json"))
    payload = json.loads(path.read_text())
    payload["metadata"]["note"] = "regenerated"
    path.write_text(json.dumps(payload))
    results = harvest(spec)
    assert results and all(item["action"] == "skipped" and "changed" in item["reason"] for item in results)

    # The old runs are not done any more: the plan runs them again, and the harvest offers the new ones.
    plan = build_plan(spec, only="A1")
    assert len(plan.tasks) == 2 and len(plan.stale) == 2 and plan.summary(1)["stale"] == 2
    run_tasks(plan, out=spec.out, workers=1, memory_gb=8, progress=lambda line: None)
    assert build_plan(spec, only="A1").tasks == []
    assert [item["action"] for item in harvest(spec)] == ["created"]


def test_harvest_prefers_a_current_run_over_a_better_stale_one(campaign: Path) -> None:
    spec = load_spec(campaign)
    run_tasks(build_plan(spec, only="A1"), out=spec.out, workers=1, memory_gb=8, progress=lambda line: None)
    runs = sorted((spec.out / "runs").glob("*.json"))
    stale = json.loads(runs[0].read_text())
    stale.update(instance_sha256="0" * 64, validated_cost=1.0)  # a cheaper run, but of another version of A1
    runs[0].write_text(json.dumps(stale))
    current = json.loads(runs[1].read_text())
    results = harvest(spec, dry_run=True)
    assert [(item["key"], item["action"]) for item in results] == [(current["key"], "would_create")]


def test_hierarchical_rankings_put_fewer_routes_first(campaign: Path) -> None:
    from mamut_routing_tools.bks_campaign.harvest import _best_runs
    from mamut_routing_tools.bks_campaign.plan import objective_rank

    def run(key: str, cost: float, routes: int) -> dict:
        return {"key": key, "instance_id": "A1", "objective_function": "HierarchicalVehicleCost",
                "valid": True, "validated_cost": cost, "num_routes": routes, "kind": "k"}

    runs = [run("a", 900.0, 11), run("b", 950.0, 10)]
    assert _best_runs(runs)[("A1", "HierarchicalVehicleCost")]["key"] == "b"
    assert objective_rank("HierarchicalVehicleCost", 950.0, 10) < objective_rank("HierarchicalVehicleCost", 900.0, 11)
    assert objective_rank("MonoCost", 900.0, 11) < objective_rank("MonoCost", 950.0, 10)

    high, low = run("a", 1.0, 1), run("z", 1.0, 1)
    high["authoritative_cost"] = {"numerator": str(2**55 + 1), "denominator": str(2**55)}
    low["authoritative_cost"] = {"numerator": "1", "denominator": "1"}
    assert _best_runs([high, low])[("A1", "HierarchicalVehicleCost")]["key"] == "z"


POLISH = '''
[[runs]]
solver = "pyvrp"
kind = "polish"
seeds = "3000"
budget_factor = 1.5
warm_start = "best"
'''


def test_a_warm_start_runs_after_the_campaign_runs_and_starts_from_them(campaign: Path) -> None:
    """One `run` of a fresh campaign: the polish waits for the seeds and starts from their best."""
    campaign.write_text(campaign.read_text().replace("[[runs]]", POLISH + "\n[[runs]]", 1))  # listed first, on purpose
    spec = load_spec(campaign)
    plan = build_plan(spec, only="A1")
    assert plan.unsupported == []  # no archived run and no stored BKS yet, but runs to start from are planned
    assert [task.kind for task in plan.tasks] == ["fake-seed", "fake-seed", "polish"]  # despite its longer budget
    lines: list[str] = []
    summary = run_tasks(plan, out=spec.out, workers=3, memory_gb=8, progress=lines.append)
    assert summary["completed"] == 3 and summary["valid"] == 3
    starts = [line.split()[1] for line in lines if line.startswith("start ")]
    done = [line.split()[1].rstrip(":") for line in lines if line.startswith("done ")]
    polish = next(key for key in starts if "polish" in key)
    assert set(done[:2]) == {key for key in starts if "polish" not in key}  # both seeds finished ...
    assert lines.index(f"start {polish} (3s, ~{plan.tasks[-1].memory_gb:.1f} GB)") > max(
        index for index, line in enumerate(lines) if line.startswith("done ") and "polish" not in line
    )  # ... before the polish started
    runs = {run["kind"]: run for run in load_runs(spec.out, with_routes=True)}
    record = runs["polish"]
    assert record["valid"] and record["warm_start_key"].startswith("run:") and record["warm_start_sha256"]
    assert record["validated_cost"] <= runs["fake-seed"]["validated_cost"]
    assert build_plan(spec, only="A1").tasks == []  # done, like any run


def test_a_warm_start_alone_needs_something_to_start_from(campaign: Path) -> None:
    text = campaign.read_text()
    first = text.index("[[runs]]")
    campaign.write_text(text[:first] + POLISH + text[text.index("[resources]"):])  # the polish is the only run entry
    spec = load_spec(campaign)
    plan = build_plan(spec, only="A1")
    assert plan.tasks == [] and "no archived run, stored BKS or planned run" in plan.unsupported[0][2]


def test_a_warm_start_whose_runs_all_failed_is_a_failed_run(campaign: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_MODE", "crash")
    campaign.write_text(campaign.read_text() + POLISH)
    spec = load_spec(campaign)
    summary = run_tasks(build_plan(spec, only="A1"), out=spec.out, workers=2, memory_gb=8, progress=lambda line: None)
    assert summary["completed"] == 3 and summary["valid"] == 0
    record = next(run for run in load_runs(spec.out) if run["kind"] == "polish")
    assert record["status"] == "failed" and "no valid archived run or stored BKS" in record["error"]
    assert len(build_plan(spec, only="A1", retry_failed=True).tasks) == 3


def test_orphan_kill_only_touches_the_runs_own_processes(tmp_path: Path) -> None:
    """A shell following the log in the run's directory is not the dead worker's solver."""
    import os
    import subprocess
    import time

    from mamut_routing_tools.bks_campaign.runner import RUN_MARKER_ENV, _kill_orphans

    if not Path("/proc").is_dir():
        pytest.skip("needs /proc")
    raw = tmp_path / "raw" / "run-key"
    raw.mkdir(parents=True)
    sleep = [sys.executable, "-c", "import time; time.sleep(60)"]
    bystander = subprocess.Popen(sleep, cwd=raw, start_new_session=True)  # someone working in the directory
    other_run = subprocess.Popen(sleep, cwd=raw, start_new_session=True, env={**os.environ, RUN_MARKER_ENV: "run-key-2"})
    solver = subprocess.Popen(sleep, cwd=tmp_path, start_new_session=True, env={**os.environ, RUN_MARKER_ENV: "run-key"})
    try:
        assert _kill_orphans("run-key") == 1
        assert solver.wait(timeout=10) < 0  # killed, wherever it was working
        time.sleep(0.2)
        assert bystander.poll() is None and other_run.poll() is None
    finally:
        for process in (bystander, other_run, solver):
            if process.poll() is None:
                process.kill()
            process.wait()


def test_a_store_error_fails_one_instance_and_the_harvest_is_still_recorded(campaign: Path, monkeypatch) -> None:
    from mamut_routing_lib import solutions

    from mamut_routing_tools.bks_campaign import harvest_history

    spec = load_spec(campaign)
    run_tasks(build_plan(spec), out=spec.out, workers=2, memory_gb=8, progress=lambda line: None)
    real_submit = solutions.submit_bks
    failure: list[BaseException] = [PermissionError("read-only collection")]

    def submit(instance_path, *args, **kwargs):
        if "B2" in str(instance_path):
            raise failure[0]
        return real_submit(instance_path, *args, **kwargs)

    monkeypatch.setattr(solutions, "submit_bks", submit)
    results = {item["instance_id"].rsplit("-", 1)[-1].upper(): item for item in harvest(spec)}
    assert results["A1"]["action"] == "created"
    assert results["B2"]["action"] == "error" and "PermissionError: read-only collection" in results["B2"]["error"]
    history = harvest_history(spec.out)
    assert len(history) == 1 and {item["action"] for item in history[0]["results"]} == {"created", "error"}

    # Interrupted between two instances: what was written before is in the history all the same.
    for path in (campaign.parent / "benchmarks").rglob("*.bks.*.json"):
        path.unlink()
    failure[0] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        harvest(spec)
    history = harvest_history(spec.out)
    assert len(history) == 2 and [item["action"] for item in history[1]["results"]] == ["created"]
    ledger, _ = write_reports(spec)
    created = [row for row in csv.DictReader(ledger.open()) if row["bks_action"] == "created"]
    assert len(created) == 1 and "A1" in created[0]["instance_id"].upper()


def test_memory_admission_still_runs_everything(campaign: Path) -> None:
    spec = load_spec(campaign)
    plan = build_plan(spec)
    summary = run_tasks(plan, out=spec.out, workers=2, memory_gb=0.1, progress=lambda line: None)
    assert summary["completed"] == 4  # at least one task always runs


def test_cli_commands(campaign: Path) -> None:
    runner = CliRunner()
    planned = runner.invoke(app, ["bks-campaign", "plan", str(campaign), "--json"])
    assert planned.exit_code == 0, planned.output
    assert json.loads(planned.stdout)["todo"] == 4
    assert runner.invoke(app, ["bks-campaign", "baseline", str(campaign)]).exit_code == 0
    assert runner.invoke(app, ["bks-campaign", "baseline", str(campaign)]).exit_code == 1
    ran = runner.invoke(app, ["bks-campaign", "run", str(campaign), "--only", "B2", "--workers", "1"])
    assert ran.exit_code == 0, ran.output
    status = json.loads(runner.invoke(app, ["bks-campaign", "status", str(campaign)]).stdout)
    assert status["runs"] == 2 and status["valid"] == 2 and status["baseline"]
    harvested = runner.invoke(app, ["bks-campaign", "harvest", str(campaign)])
    assert harvested.exit_code == 0 and json.loads(harvested.stdout)["counts"] == {"created": 1}
    assert runner.invoke(app, ["bks-campaign", "report", str(campaign)]).exit_code == 0
    broken = campaign.parent / "broken.toml"
    broken.write_text("[campaign]\nname = 'x'\n")
    assert runner.invoke(app, ["bks-campaign", "plan", str(broken)]).exit_code == 2


def _fake_solvers_in(directory: Path) -> list[int]:
    """Live processes whose working directory lies under ``directory`` (Linux /proc)."""
    import os

    proc = Path("/proc")
    if not proc.is_dir():
        pytest.skip("needs /proc")
    found = []
    for entry in proc.iterdir():
        if entry.name.isdigit():
            try:
                if Path(os.readlink(entry / "cwd")).is_relative_to(directory.resolve()):
                    found.append(int(entry.name))
            except OSError:
                pass
    return found


def test_a_killed_worker_costs_only_its_own_run(campaign: Path, monkeypatch) -> None:
    import multiprocessing as mp
    import os
    import signal
    import threading
    import time

    monkeypatch.setenv("FAKE_MODE", "slow")
    monkeypatch.setenv("FAKE_SLEEP", "4")
    spec = load_spec(campaign)
    plan = build_plan(spec, only="A1")
    assert len(plan.tasks) == 2
    victims: list[int] = []

    def _kill_first_worker_once_its_solver_runs() -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not victims:
            for child in mp.active_children():
                if _fake_solvers_in(spec.out / "raw" / child.name.removeprefix("bks-campaign ")):
                    victims.append(child.pid)
                    os.kill(child.pid, signal.SIGKILL)  # what the out-of-memory killer does
                    return
            time.sleep(0.1)

    killer = threading.Thread(target=_kill_first_worker_once_its_solver_runs, daemon=True)
    killer.start()
    summary = run_tasks(plan, out=spec.out, workers=2, memory_gb=8, progress=lambda line: None)
    killer.join()
    assert victims, "no worker was killed"
    assert summary["completed"] == 2 and summary["valid"] == 1 and summary["failed_workers"] == 1
    assert (spec.out / "run.DONE").is_file()
    runs = {run["status"]: run for run in load_runs(spec.out)}
    assert set(runs) == {"feasible", "failed"}
    assert "killed by SIGKILL" in runs["failed"]["error"] and "has been killed" in runs["failed"]["error"]
    assert _fake_solvers_in(spec.out / "raw") == []  # the dead worker's solver did not outlive it
    assert len(build_plan(spec, only="A1").tasks) == 0  # a failed run is done ...
    assert len(build_plan(spec, only="A1", retry_failed=True).tasks) == 1  # ... until --retry-failed


def test_ctrl_c_while_workers_boot_does_not_kill_them(campaign: Path) -> None:
    import multiprocessing as mp
    import os
    import signal

    spec = load_spec(campaign)
    plan = build_plan(spec)
    hit: set[int] = set()

    def _interrupt_booting_workers(line: str) -> None:
        # Called right after each worker process starts: a terminal Ctrl-C reaches it mid-boot.
        if line.startswith("start"):
            for child in mp.active_children():
                if child.pid not in hit:
                    hit.add(child.pid)
                    os.kill(child.pid, signal.SIGINT)

    summary = run_tasks(plan, out=spec.out, workers=2, memory_gb=8, progress=_interrupt_booting_workers)
    assert len(hit) == 4
    assert summary["completed"] == 4 and summary["valid"] == 4 and summary["failed_workers"] == 0


def test_every_harvest_is_kept_and_the_ledger_remembers_creations(campaign: Path) -> None:
    from mamut_routing_tools.bks_campaign import harvest_history

    spec = load_spec(campaign)
    run_tasks(build_plan(spec), out=spec.out, workers=2, memory_gb=8, progress=lambda line: None)
    first = harvest(spec)
    assert sorted(item["action"] for item in first) == ["created", "created"]
    second = harvest(spec)  # the same best runs again: the store keeps its (equal) BKS
    assert {item["action"] for item in second} == {"kept_existing"}
    history = harvest_history(spec.out)
    assert len(history) == 2 and history[0]["results"] == first and history[1]["results"] == second
    assert json.loads((spec.out / "harvest.json").read_text()) == second  # the latest, as before
    ledger, _ = write_reports(spec)
    rows = [row for row in csv.DictReader(ledger.open()) if row["is_best"] == "True"]
    assert [row["bks_action"] for row in rows] == ["created", "created"]
    assert all(row["bks_action_at"] == history[0]["harvested"] for row in rows)


def test_a_harvest_from_before_the_history_is_kept(campaign: Path) -> None:
    from mamut_routing_tools.bks_campaign import harvest_history

    spec = load_spec(campaign)
    run_tasks(build_plan(spec, only="A1"), out=spec.out, workers=1, memory_gb=8, progress=lambda line: None)
    old = [{"key": "legacy-key", "instance_id": "A1", "action": "created"}]
    (spec.out / "harvest.json").write_text(json.dumps(old))
    harvest(spec)
    history = harvest_history(spec.out)
    assert len(history) == 2 and history[0]["results"] == old


def test_sigterm_aborts_the_running_tasks_at_once(campaign: Path, monkeypatch) -> None:
    import os
    import signal
    import threading
    import time

    monkeypatch.setenv("FAKE_MODE", "slow")
    monkeypatch.setenv("FAKE_SLEEP", "30")
    spec = load_spec(campaign)
    plan = build_plan(spec)
    sent: list[float] = []

    def _terminate_once_solvers_run() -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if _fake_solvers_in(spec.out / "raw"):
                sent.append(time.monotonic())
                os.kill(os.getpid(), signal.SIGTERM)  # a scheduler signalling the runner only
                return
            time.sleep(0.1)

    lines: list[str] = []
    killer = threading.Thread(target=_terminate_once_solvers_run, daemon=True)
    killer.start()
    summary = run_tasks(plan, out=spec.out, workers=2, memory_gb=8, progress=lines.append)
    killer.join()
    assert sent, "the solvers never started"
    assert time.monotonic() - sent[0] < 20  # not the 30 s the solvers would take
    assert summary["aborted"] and summary["completed"] == 0 and load_runs(spec.out) == []
    assert any(line.startswith("SIGTERM: aborting") for line in lines)
    assert _fake_solvers_in(spec.out / "raw") == []
    assert len(build_plan(spec).tasks) == 4  # the next run redoes them
