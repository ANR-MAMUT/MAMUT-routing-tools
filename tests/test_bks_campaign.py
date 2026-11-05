"""BKS campaigns end to end on a toy tree: spec, plan, resumable run, baseline, single-writer harvest, reports."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mamut_routing_lib.models import BenchmarkInstance
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


def _instance(name: str, shift: int) -> BenchmarkInstance:
    return BenchmarkInstance(
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
    for name, shift in (("A1", 0), ("B2", 5)):
        path = benchmarks / "VRPTW" / "Dimacs2021" / "n=3" / f"{name}.vrp.json"
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
    with pytest.raises(ValueError):
        parse_seeds("5-1")


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
    assert len(runs) == 4 and all(run["valid"] and run["schema_version"] == 1 for run in runs)
    record = runs[0]
    assert record["solver"] == "fake" and record["host"] and record["machine"]
    assert record["provenance"]["scale"]["exact"] is True
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

    again = harvest(spec)  # the baseline had no BKS, so the best runs are offered again: ties keep the incumbent
    assert {item["action"] for item in again} == {"kept_existing"} and all(item["tie"] for item in again)
    write_baseline(spec, force=True)  # now the baseline holds the campaign's own BKS
    tied = harvest(spec)  # ties are the store's call, not the float pre-filter's
    assert {item["action"] for item in tied} == {"kept_existing"} and all(item["tie"] for item in tied)

    ledger, compare = write_reports(spec)
    rows = list(csv.DictReader(ledger.open()))
    assert len(rows) == 4 and {row["kind"] for row in rows} == {"fake-seed"}
    assert sum(row["is_best"] == "True" for row in rows) == 2
    table = list(csv.DictReader(compare.open()))
    assert len(table) == 2 and table[0]["winner"] == "fake-seed" and table[0]["fake-seed_valid_runs"] == "2"


def test_baseline_pre_filter_only_drops_runs_clearly_worse() -> None:
    from mamut_routing_tools.bks_campaign.harvest import _may_beat_baseline

    def run(cost: float, routes: int = 10, objective: str = "MonoCost") -> dict:
        return {"objective_function": objective, "validated_cost": cost, "num_routes": routes}

    baseline = {"cost": "1547830.702", "num_routes": "39"}
    assert _may_beat_baseline(run(1547659.77), baseline)
    assert _may_beat_baseline(run(1547830.702), baseline)  # a tie: the store compares exact costs
    assert _may_beat_baseline(run(1547830.702 + 2e-9), baseline)  # the same routes summed in another order
    assert not _may_beat_baseline(run(1547830.703), baseline)  # one unit of the last decimal worse
    assert _may_beat_baseline(run(5.0), {"cost": "", "num_routes": ""})  # no BKS yet
    hierarchical = {"cost": "828.94", "num_routes": "10"}
    assert _may_beat_baseline(run(900.0, 9, "HierarchicalVehicleCost"), hierarchical)
    assert not _may_beat_baseline(run(800.0, 11, "HierarchicalVehicleCost"), hierarchical)
    assert not _may_beat_baseline(run(829.0, 10, "HierarchicalVehicleCost"), hierarchical)


def test_harvest_skips_runs_of_a_changed_instance(campaign: Path) -> None:
    spec = load_spec(campaign)
    run_tasks(build_plan(spec, only="A1"), out=spec.out, workers=1, memory_gb=8, progress=lambda line: None)
    path = next((campaign.parent / "benchmarks").rglob("A1.vrp.json"))
    payload = json.loads(path.read_text())
    payload["metadata"]["note"] = "regenerated"
    path.write_text(json.dumps(payload))
    results = harvest(spec)
    assert results and all(item["action"] == "skipped" and "changed" in item["reason"] for item in results)


def test_warm_start_from_the_best_run(campaign: Path) -> None:
    spec = load_spec(campaign)
    run_tasks(build_plan(spec, only="A1"), out=spec.out, workers=1, memory_gb=8, progress=lambda line: None)
    campaign.write_text(
        campaign.read_text() + '\n[[runs]]\nsolver = "pyvrp"\nkind = "polish"\nseeds = "3000"\nwarm_start = "best"\n'
    )
    spec = load_spec(campaign)
    plan = build_plan(spec, only="A1")
    polish = [task for task in plan.tasks if task.kind == "polish"]
    assert len(polish) == 1 and polish[0].warm_start_routes
    assert polish[0].warm_start_key.startswith("run:") and polish[0].warm_start_sha256
    run_tasks(plan, out=spec.out, workers=1, memory_gb=8, progress=lambda line: None)
    record = next(run for run in load_runs(spec.out) if run["kind"] == "polish")
    assert record["valid"] and record["warm_start_key"] == polish[0].warm_start_key


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
