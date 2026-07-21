"""Standalone per-instance TD derivation on a real cached city (Nantes n=8).

The correctness gate is ``load_td_instance(twin, verify_sha256=True)``: it
resolves the road + traffic sidecars against the collection marker the deriver
drops, rebuilds the canonical ATFs from the published road-graph + overlay, and
checks every sha256 (graph, traffic, materialized atf). A passing load is proof
the twins are canonical road-graph-model TD instances.

Guarded by the tool's cached ``Nantes.osm`` extract; skipped when it is absent
(the suite must stay green without the multi-megabyte extract on disk).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mamut_routing_lib.td import load_td_instance
from mamut_routing_lib.td.models import BenchmarkInstanceTDVRP, BenchmarkInstanceTDVRPTW

from mamut_routing_tools.generation.single import GenerationRequest, generate_single_instance
from mamut_routing_tools.generation.td import derive_td_from_vrptw
from mamut_routing_tools.generation.vrptw import derive_vrptw_from_cvrp

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHED_NANTES = REPO_ROOT / ".cache" / "mamut-tools" / "osmdata" / "Nantes.osm"

pytestmark = pytest.mark.skipif(
    not CACHED_NANTES.is_file(),
    reason=f"cached OSM extract not present at {CACHED_NANTES}",
)


def _provided_windows(cvrptw_path: Path) -> dict[int, tuple[int, int]]:
    """The PROVIDED (pre-lift) time windows straight from the CVRPTW file."""
    windows: dict[int, tuple[int, int]] = {}
    section = ""
    for raw in cvrptw_path.read_text().splitlines():
        line = raw.strip()
        if line in ("TIME_WINDOW_SECTION", "SERVICE_TIME_SECTION", "DEPOT_SECTION", "EOF"):
            section = "" if line in ("DEPOT_SECTION", "EOF") else line
            continue
        if section == "TIME_WINDOW_SECTION":
            parts = line.split()
            if len(parts) >= 3:
                windows[int(parts[0]) - 1] = (int(parts[1]), int(parts[2]))
    return windows


def test_derive_td_nantes_n8_loads_and_verifies(tmp_path: Path) -> None:
    request = GenerationRequest(
        city="Nantes",
        osm_path=CACHED_NANTES,
        method="parametric_attach",
        n_customers=8,
        seed=7,
    )
    result = generate_single_instance(request, tmp_path / "instances")
    folder = Path(result["folder"])
    base = result["base_name"]

    derive_vrptw_from_cvrp(folder, base, source_seed=7)

    out = derive_td_from_vrptw(folder, base, model="bpr", intensity="moderate", seed=42)
    assert out["ok"] and out["action"] == "derived"
    assert out["num_customers"] == 8
    assert (folder / "mamut-collection.json").is_file()
    assert (folder / out["road_sidecar"]).is_file()
    assert len(out["combos"]) == 1
    combo = out["combos"][0]

    tdvrptw_path = folder / combo["tdvrptw_twin"]
    tdvrp_path = folder / combo["tdvrp_twin"]

    # The real correctness gate: rebuild the ATFs and check every sha256.
    tw_loaded = load_td_instance(tdvrptw_path, verify_sha256=True)
    tdvrp_loaded = load_td_instance(tdvrp_path, verify_sha256=True)
    assert isinstance(tw_loaded.instance, BenchmarkInstanceTDVRPTW)
    assert isinstance(tdvrp_loaded.instance, BenchmarkInstanceTDVRP)
    # Both twins share one road graph, one overlay and one materialized ATF set.
    expected_arcs = 9 * 8  # complete customer graph over depot + 8 customers
    assert len(tw_loaded.atfs.arcs) == expected_arcs
    assert len(tdvrp_loaded.atfs.arcs) == expected_arcs
    assert tw_loaded.instance.td.atf_sha256 == combo["atf_sha256"]
    assert tdvrp_loaded.instance.td.atf_sha256 == combo["atf_sha256"]

    # Lifted windows never drop below the provided earliest bounds, and every
    # customer window keeps positive width.
    provided = _provided_windows(folder / f"{base}_fastest.cvrptw.vrp")
    lifted = json.loads(tdvrptw_path.read_text())["time_windows"]
    assert len(lifted) == 9
    for customer in range(1, len(lifted)):
        earliest_after, latest_after = lifted[customer]
        earliest_before, _ = provided[customer]
        assert earliest_after >= earliest_before
        assert latest_after > earliest_after

    # Idempotent: a second call without --force keeps the existing twins.
    kept = derive_td_from_vrptw(folder, base, model="bpr", intensity="moderate")
    assert kept["action"] == "kept"
