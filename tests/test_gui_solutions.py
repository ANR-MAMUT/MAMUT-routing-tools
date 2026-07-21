"""Focused contracts for persisted GUI solutions and comparisons."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mamut_routing_tools.gui.solutions import SolutionStore, compare_solution_records


def test_comparison_rejects_runs_from_different_metrics() -> None:
    instance = SimpleNamespace(depot=0, demands=[0, 1, 1])
    candidate = {
        "run_id": "candidate",
        "objective_function": "MonoCost",
        "cost": 10,
        "routes": [[1, 2]],
        "validation": {"valid": True},
        "metadata": {"metric": "fastest"},
    }
    reference = {
        **candidate,
        "run_id": "reference",
        "metadata": {"metric": "shortest"},
    }

    with pytest.raises(ValueError, match="same metric"):
        compare_solution_records(instance, candidate, reference)


def test_solution_store_rejects_non_run_identifiers(tmp_path: Path) -> None:
    store = SolutionStore(tmp_path)

    with pytest.raises(KeyError):
        store.get("instance", "../../state/jobs/a-job")
