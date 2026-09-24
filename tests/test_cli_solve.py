"""``mamut-tools solve`` reports the checker's cost and the store's real action.

Audit 2026-09-24 N-cross-repo-04: it printed the solver's integer-scaled
objective as "cost" and read ``improved`` / ``bks_path`` attributes the lib's
``BKSUpdateResult`` does not have, so every run said
``{"improved": false, "path": ""}`` while a BKS file was written.
"""

from __future__ import annotations

import json
from pathlib import Path

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

    first = runner.invoke(app, ["solve", str(instance), "--time-limit", "1", "--update-bks"])
    assert first.exit_code == 0, first.output
    payload = json.loads(first.stdout)
    bks_path = Path(payload["bks_update"]["path"])
    assert payload["bks_update"]["action"] == "created" and bks_path.is_file()
    stored = json.loads(bks_path.read_text(encoding="utf-8"))
    assert payload["cost"] == stored["cost"]
    assert payload["validation"] == "valid"

    again = json.loads(runner.invoke(app, ["solve", str(instance), "--time-limit", "1", "--update-bks"]).stdout)
    assert again["bks_update"]["action"] == "kept_existing"
    assert again["bks_update"]["tie"] is (again["cost"] == stored["cost"])
