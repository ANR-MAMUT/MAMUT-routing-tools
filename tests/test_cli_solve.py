"""``mamut-tools solve`` reports the checker's cost and the store's real action.

Audit 2026-09-24 N-cross-repo-04: it printed the solver's integer-scaled
objective as "cost" and read ``improved`` / ``bks_path`` attributes the lib's
``BKSUpdateResult`` does not have, so every run said
``{"improved": false, "path": ""}`` while a BKS file was written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mamut_routing_tools.cli import app
from mamut_routing_tools.generation.single import (
    GenerationRequest,
    generate_single_instance,
)


def test_solve_reports_checker_cost_and_bks_action(tmp_path: Path, fixture_osm_path: Path) -> None:
    generated = generate_single_instance(
        GenerationRequest(city="Testville", osm_path=fixture_osm_path, method="parametric_attach", n_customers=3, seed=7),
        tmp_path / "instances",
    )
    instance = Path(generated["folder"]) / generated["files"]["vrp_json"]["shortest"]
    runner = CliRunner()

    first = runner.invoke(app, ["solve", str(instance), "--time-limit", "1", "--update-bks", "--authors", "Tester"])
    assert first.exit_code == 0, first.output
    payload = json.loads(first.stdout)
    bks_path = Path(payload["bks_update"]["path"])
    assert payload["bks_update"]["action"] == "created" and bks_path.is_file()
    stored = json.loads(bks_path.read_text(encoding="utf-8"))
    assert payload["cost"] == stored["cost"]
    assert payload["validation"] == "valid"

    again = json.loads(
        runner.invoke(app, ["solve", str(instance), "--time-limit", "1", "--update-bks", "--authors", "Tester"]).stdout
    )
    assert again["bks_update"]["action"] == "kept_existing"
    assert again["bks_update"]["tie"] is (again["cost"] == stored["cost"])
    assert payload["solver"] == "pyvrp" and payload["status"] == "feasible"
    assert stored["metadata"]["authors"] == "Tester" and stored["metadata"]["solver"] == "pyvrp"


def _generated(tmp_path: Path, fixture_osm_path: Path, metric: str = "shortest") -> Path:
    generated = generate_single_instance(
        GenerationRequest(city="Testville", osm_path=fixture_osm_path, method="parametric_attach", n_customers=3, seed=7),
        tmp_path / "instances",
    )
    return Path(generated["folder"]) / generated["files"]["vrp_json"][metric]


def test_update_bks_needs_authors(tmp_path: Path, fixture_osm_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MAMUT_ROUTING_BKS_AUTHORS", raising=False)
    result = CliRunner().invoke(app, ["solve", str(_generated(tmp_path, fixture_osm_path)), "--update-bks"])
    assert result.exit_code == 1 and "--save-bks needs --authors" in result.stderr


def test_solve_takes_the_lib_flag_names_and_its_json_is_a_solution_file(tmp_path: Path, fixture_osm_path: Path) -> None:
    from mamut_routing_lib.cli import app as lib_app

    instance = _generated(tmp_path, fixture_osm_path)
    result = CliRunner().invoke(app, ["solve", str(instance), "--time-limit-s", "1", "--save-bks", "--authors", "Tester"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["bks_update"]["action"] == "created"
    solution = tmp_path / "run.json"
    solution.write_text(result.stdout)
    checked = CliRunner().invoke(lib_app, ["check", str(instance), str(solution)])
    assert checked.exit_code == 0 and "Warning" not in checked.output, checked.output


def test_a_zero_time_limit_is_a_usage_error(tmp_path: Path, fixture_osm_path: Path) -> None:
    result = CliRunner().invoke(app, ["solve", str(_generated(tmp_path, fixture_osm_path)), "--time-limit", "0"])
    assert result.exit_code == 2 and "Traceback" not in result.output


def test_solve_with_a_configured_external_solver(tmp_path: Path, fixture_osm_path: Path, monkeypatch) -> None:
    config = tmp_path / "solvers.toml"
    config.write_text(
        'schema_version = 1\n[solvers.fake]\nruntime = "python"\ninput_format = "lkh"\noutput_format = "lkh-mtsp"\n'
        'problem_types = ["CVRP"]\nsymmetric_only = false\n'
        'argv = ["{python}", "-m", "mamut_routing_lib.solvers.external._fake_solver", "--params", "{params}"]\n'
    )
    monkeypatch.setenv("MAMUT_ROUTING_SOLVERS_CONFIG", str(config))
    from mamut_routing_lib.solvers import reload_solvers

    reload_solvers()
    result = CliRunner().invoke(app, ["solve", str(_generated(tmp_path, fixture_osm_path)), "--solver", "fake", "--time-limit", "5"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] and payload["solver"] == "fake" and payload["validation"] == "valid"
    assert payload["provenance"]["scale"]["exact"] is True
    assert payload["cost"] == pytest.approx(payload["solver_cost"])

    monkeypatch.setenv("FAKE_MODE", "crash")
    crashed = CliRunner().invoke(app, ["solve", str(_generated(tmp_path, fixture_osm_path)), "--solver", "fake", "--time-limit", "5"])
    assert crashed.exit_code == 1  # the JSON still says why
    failed = json.loads(crashed.stdout)
    assert not failed["ok"] and failed["status"] == "failed" and "no solution file" in failed["error"]


def test_unknown_or_unsuitable_solvers_are_errors(tmp_path: Path, fixture_osm_path: Path) -> None:
    instance = _generated(tmp_path, fixture_osm_path)
    unknown = CliRunner().invoke(app, ["solve", str(instance), "--solver", "nope"])
    assert unknown.exit_code == 1 and "unknown solver 'nope'" in unknown.stderr
    kayros = CliRunner().invoke(app, ["solve", str(instance), "--solver", "kayros"])
    assert kayros.exit_code == 1 and "kayros" in kayros.stderr


def test_termination_signal_during_solve(tmp_path, monkeypatch):
    import os
    import signal

    if os.name != "posix":
        pytest.skip("POSIX termination signals")
    def interrupted(*args, **kwargs):
        signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr("mamut_routing_lib.solvers.run_solver", interrupted)
    path = tmp_path / "toy.vrp.json"
    path.write_text("{}")
    result = CliRunner().invoke(app, ["solve", str(path), "--save-bks", "--authors", "Tester"])
    assert result.exit_code == 143 and "SIGTERM" in result.output
    assert not list(tmp_path.glob("*.bks.*.json"))
