"""Standalone derive-td on synthetic grid cities: no OSM download needed.

Audit 2026-09-24:
- I1-tools-generation-01 (high): derive-td crashed from about n=80, because
  its anchor was one capacity-agnostic tour that cannot return by the horizon;
- I1-02: on reachable_interval windows that tour lifted deadlines by hours;
- I1-04: a regenerated base kept stale twins ("kept");
- errors surfaced as raw PWLFError / AssertionError tracebacks.
The n=120 case takes about 20 s.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import grid_osm_text
from mamut_routing_lib.td import NDCPWLF, load_td_instance

from mamut_routing_tools.family.family import AnchorAuditError
from mamut_routing_tools.generation import td as td_module
from mamut_routing_tools.generation.single import (
    GenerationRequest,
    generate_single_instance,
)
from mamut_routing_tools.generation.td import (
    TDDerivationError,
    _split_for_traffic,
    derive_td_from_vrptw,
)
from mamut_routing_tools.generation.vrptw import derive_vrptw_from_cvrp


def _instance(tmp_path: Path, rows: int, n: int, *, tw_method: str = "route_centered", seed: int = 3):
    osm = tmp_path / "Gridville.osm"
    osm.write_text(grid_osm_text(rows, rows), encoding="utf-8")
    result = generate_single_instance(
        GenerationRequest(city="Gridville", osm_path=osm, method="parametric_attach", n_customers=n, seed=seed),
        tmp_path / "instances",
    )
    folder, base = Path(result["folder"]), result["base_name"]
    derive_vrptw_from_cvrp(folder, base, tw_method=tw_method, source_seed=seed)
    return folder, base


def _cvrptw_windows(folder: Path, base: str) -> list[tuple[int, int]]:
    return [tuple(w) for w in td_module._parse_cvrptw_vrp(folder / f"{base}_fastest.cvrptw.vrp")["time_windows"]]


def test_n120_derives_certified_twins(tmp_path: Path) -> None:
    folder, base = _instance(tmp_path, 14, 120)
    manifest = json.loads((folder / f"{base}_vrptw_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["derivation"]["anchor_routes"]) > 1
    out = derive_td_from_vrptw(folder, base, model="bpr", intensity="heavy", seed=42)
    assert out["action"] == "derived" and out["anchor_route_source"] == "manifest"
    combo = out["combos"][0]
    loaded = load_td_instance(folder / combo["tdvrptw_twin"], verify_sha256=True)
    assert loaded.instance.num_customers == out["num_customers"]
    load_td_instance(folder / combo["tdvrp_twin"], verify_sha256=True)
    assert not (folder / ".mamut-staging").exists()


def test_legacy_manifest_and_reachable_windows_get_window_feasible_anchors(tmp_path: Path) -> None:
    folder, base = _instance(tmp_path, 8, 30, tw_method="reachable_interval")
    manifest_path = folder / f"{base}_vrptw_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # A pre-0.6 manifest: one tour through every customer, no anchor_routes.
    manifest["derivation"].pop("anchor_routes")
    manifest["derivation"]["anchor_route"] = [0, *range(1, 31)]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out = derive_td_from_vrptw(folder, base, model="bpr", intensity="moderate", seed=42)
    assert out["anchor_route_source"] == "window-feasible"
    # Windows were built around no route: only traffic may lift them, and little.
    assert out["max_lift_seconds"] < 3600
    load_td_instance(folder / out["combos"][0]["tdvrptw_twin"], verify_sha256=True)


def test_regenerated_inputs_are_rederived_not_kept(tmp_path: Path) -> None:
    folder, base = _instance(tmp_path, 8, 20)
    first = derive_td_from_vrptw(folder, base, model="wave", intensity="light", seed=42)
    assert derive_td_from_vrptw(folder, base, model="wave", intensity="light", seed=42)["action"] == "kept"
    twin_path = folder / first["combos"][0]["tdvrptw_twin"]
    before = json.loads(twin_path.read_text(encoding="utf-8"))["metadata"]["derived_from"]
    # Same base name, new windows: the old twins no longer describe it.
    derive_vrptw_from_cvrp(folder, base, tw_method="reachable_interval", source_seed=3)
    again = derive_td_from_vrptw(folder, base, model="wave", intensity="light", seed=42)
    assert again["action"] == "derived"
    twin = json.loads((folder / again["combos"][0]["tdvrptw_twin"]).read_text(encoding="utf-8"))
    provided = _cvrptw_windows(folder, base)
    assert all(lifted[0] == provided[i][0] for i, lifted in enumerate(twin["time_windows"]))
    assert twin["metadata"]["derived_from"]["cvrptw_sha256"] != before["cvrptw_sha256"]
    # A different traffic seed is a different derivation too.
    assert derive_td_from_vrptw(folder, base, model="wave", intensity="light", seed=7)["action"] == "derived"


@pytest.mark.parametrize("error", [AnchorAuditError("anchor misses customer 3"), AssertionError("zero-width")])
def test_certification_failures_are_td_derivation_errors(tmp_path: Path, monkeypatch, error) -> None:
    folder, base = _instance(tmp_path, 6, 8)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(td_module, "_audit_and_lift", fail)
    with pytest.raises(TDDerivationError, match=str(error)):
        derive_td_from_vrptw(folder, base, seed=42)
    assert not list(folder.glob("*.tdvrp.vrp.json"))
    assert not (folder / ".mamut-staging").exists()


def _constant_travel(minutes: float) -> NDCPWLF:
    return NDCPWLF([0.0, 1000.0], [minutes, 1000.0 + minutes])


def test_split_keeps_the_longest_returning_prefix() -> None:
    arcs = {key: _constant_travel(10.0) for key in [(0, 1), (1, 2), (2, 3), (3, 0), (1, 0), (2, 0), (0, 2), (0, 3)]}
    windows = [(0, 1000), (0, 1000), (0, 1000), (0, 1000)]
    service = [0, 100, 100, 100]
    # Serving 1 then 2 ends at 230 and returns at 240; adding 3 returns at 350.
    assert _split_for_traffic([[1, 2, 3]], windows, service, {"s": arcs}, 300.0) == [[1, 2], [3]]
    assert _split_for_traffic([[1, 2, 3]], windows, service, {"s": arcs}, 400.0) == [[1, 2, 3]]
    with pytest.raises(TDDerivationError, match="customer 1"):
        _split_for_traffic([[1]], windows, service, {"s": arcs}, 100.0)
