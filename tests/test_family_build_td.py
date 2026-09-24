"""``family.build_td`` publishes a loadable TD layer or nothing at all.

Audit 2026-09-24 (I2-tools-family-td-01, -04, -07): a forced rebuild with new
traffic kept the old twins' ``atf_sha256`` next to the new overlays, so nothing
loaded afterwards; a failure part-way left the VRPTW file and some overlays
rewritten; a re-run erased the ``tw_repair`` record; ``tdvrptw_only`` without
TDVRP twins raised ``KeyError`` after rewriting the VRPTW file. These run the
tools' own pipeline (stage 1, ``build_base``, ``derive_vrptw``, ``build_td``) on
a 6x6 grid city with synthetic speed profiles.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from conftest import grid_osm_text
from mamut_routing_lib.td import load_td_instance

from mamut_routing_tools.family import (
    TD_INTENSITIES,
    TD_MODELS,
    build_base,
    build_td,
    derive_vrptw,
)
from mamut_routing_tools.family import family as family_module
from mamut_routing_tools.generation.single import (
    GenerationRequest,
    generate_single_instance,
)
from mamut_routing_tools.staging import STAGING_DIRNAME
from mamut_routing_tools.td import build_static_bridge
from mamut_routing_tools.td.bridge import BridgeSpeeds

CITY = "Gridville"
N = 10
BASE = "poryos-gridville-n10-par"


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    """A collection with one base and its VRPTW candidate, plus the bridge graph."""
    work = tmp_path_factory.mktemp("build-td-template")
    osm = work / f"{CITY}.osm"
    osm.write_text(grid_osm_text(6, 6), encoding="utf-8")
    request = GenerationRequest(
        city=CITY,
        osm_path=osm,
        method="parametric_attach",
        customer_mode="random",
        depot_mode="center",
        n_customers=N,
        demand_type=1,
        avg_route_size=1,
        seed=11,
    )
    staged = generate_single_instance(request, work / "_stage1")
    folder = Path(staged["folder"])
    meta = json.loads((folder / staged["files"]["meta"]).read_text(encoding="utf-8"))
    manifest = json.loads((folder / staged["manifest"]).read_text(encoding="utf-8"))
    bridge = build_static_bridge(osm_path=osm, city_slug=CITY, metas=[meta])
    root = work / "collection"
    build_base(
        graph=bridge.graph,
        nodes=bridge.nodes[meta["instance_name"]],
        meta=meta,
        manifest=manifest,
        city=CITY,
        method_tag="par",
        collection_root=root,
        generated_at="2026-01-01",
    )
    derive_vrptw(collection_root=root, city=CITY, num_customers=N, method_tag="par", generated_at="2026-01-01")
    return root, bridge.graph


def _speeds(graph, heavy_factor: float = 0.5) -> dict:
    """Free-flow speeds, slowed by ``heavy_factor`` all day on the heavy overlays."""
    return {
        (model, intensity): BridgeSpeeds(
            city="gridville",
            model=model,
            intensity=intensity,
            seed=1,
            num_trips=0,
            params={},
            speeds=[
                [edge[4] * (heavy_factor if intensity == "heavy" else 1.0) for _ in range(graph.num_bins)]
                for edge in graph.edges
            ],
        )
        for model in TD_MODELS
        for intensity in TD_INTENSITIES
    }


@pytest.fixture
def collection(template, tmp_path):
    root, graph = template
    copy = tmp_path / "collection"
    shutil.copytree(root, copy)
    return copy, graph


def _build(root: Path, graph, speeds, **kwargs):
    return build_td(
        collection_root=root,
        graph=graph,
        speeds_by_combo=speeds,
        city=CITY,
        num_customers=N,
        method_tag="par",
        generated_at="2026-01-01",
        **kwargs,
    )


def _snapshot(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _twins(root: Path) -> list[Path]:
    return sorted(root.glob("TDVRP*/**/*.vrp.json"))


def _vrptw(root: Path) -> dict:
    return json.loads(next(root.glob(f"VRPTW/**/{BASE}.vrp.json")).read_text(encoding="utf-8"))


def _assert_all_twins_load(root: Path) -> None:
    twins = _twins(root)
    assert len(twins) == 12
    for twin in twins:
        load_td_instance(twin, verify_sha256=True)


def test_build_publishes_loadable_twins(collection) -> None:
    root, graph = collection
    built = _build(root, graph, _speeds(graph))
    assert built is not None and built.base == BASE
    _assert_all_twins_load(root)
    assert not (root / STAGING_DIRNAME).exists()


def test_rebuild_with_new_traffic_repins_every_twin(collection) -> None:
    """The stale-pin regression: new overlays must come with new atf_sha256 pins."""
    root, graph = collection
    first = _build(root, graph, _speeds(graph, 0.5))
    second = _build(root, graph, _speeds(graph, 0.4), force=True, verify=False)
    assert second.atf_sha256["bpr-heavy"] != first.atf_sha256["bpr-heavy"]
    assert second.atf_sha256["bpr-light"] == first.atf_sha256["bpr-light"]
    _assert_all_twins_load(root)


def test_reuse_traffic_keeps_a_matching_pin(collection, monkeypatch) -> None:
    root, graph = collection
    first = _build(root, graph, _speeds(graph))
    calls = []
    real = family_module.materialize_instance_atfs_roadgraph
    monkeypatch.setattr(
        family_module, "materialize_instance_atfs_roadgraph", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    again = _build(root, graph, {}, force=True, reuse_traffic=True)
    assert again.atf_sha256 == first.atf_sha256
    assert calls == []  # every pin was provably reusable, nothing re-materialized
    _assert_all_twins_load(root)


def test_a_rebuild_is_byte_identical(collection) -> None:
    root, graph = collection
    _build(root, graph, _speeds(graph))
    before = _snapshot(root)
    _build(root, graph, _speeds(graph), force=True)
    assert _snapshot(root) == before


def test_a_failed_verification_publishes_nothing(collection, monkeypatch) -> None:
    root, graph = collection
    _build(root, graph, _speeds(graph, 0.5))
    before = _snapshot(root)

    def fail(*args, **kwargs):
        raise RuntimeError("verification failed")

    monkeypatch.setattr(family_module, "load_td_instance", fail)
    with pytest.raises(RuntimeError, match="verification failed"):
        _build(root, graph, _speeds(graph, 0.4), force=True)
    assert _snapshot(root) == before
    assert not (root / STAGING_DIRNAME).exists()


def test_an_anchor_that_cannot_return_publishes_nothing(collection) -> None:
    root, graph = collection
    before = _snapshot(root)
    with pytest.raises(family_module.AnchorAuditError):
        _build(root, graph, _speeds(graph, 0.0018))
    assert _snapshot(root) == before


def test_a_rerun_keeps_the_tw_repair_record(collection) -> None:
    root, graph = collection
    first = _build(root, graph, _speeds(graph, 0.01))
    assert first.lifted_customers > 0
    record = _vrptw(root)["metadata"]["tw_repair"]
    windows = _vrptw(root)["time_windows"]
    for kwargs in ({"force": True, "reuse_traffic": True}, {"force": True, "tdvrptw_only": True}):
        _build(root, graph, {}, **kwargs)
        assert _vrptw(root)["metadata"]["tw_repair"] == record
        assert _vrptw(root)["time_windows"] == windows
        for twin in root.glob("TDVRPTW/**/*.vrp.json"):
            assert json.loads(twin.read_text(encoding="utf-8"))["metadata"]["tw_repair"] == record


def test_tdvrptw_only_without_tdvrp_twins_fails_before_writing(collection) -> None:
    root, graph = collection
    _build(root, graph, _speeds(graph))
    shutil.rmtree(root / "TDVRP")
    before = _snapshot(root)
    with pytest.raises(ValueError, match="published TDVRP twins"):
        _build(root, graph, {}, force=True, tdvrptw_only=True)
    assert _snapshot(root) == before


def test_a_missing_overlay_without_speeds_fails_before_writing(collection) -> None:
    root, graph = collection
    _build(root, graph, _speeds(graph))
    next(root.glob("sidecars/**/*.traffic-wave-light.json.gz")).unlink()
    before = _snapshot(root)
    with pytest.raises(ValueError, match="missing traffic speeds"):
        _build(root, graph, {}, force=True, reuse_traffic=True)
    assert _snapshot(root) == before
