"""Publishing a distance matrix as a pin instead of as bytes.

A distance sidecar grows with n^2: 8 MB at n=1000, ~200 MB at n=5000. Above
some size, storing it stops being reasonable and the instance ships as a
*descriptor* -- the road graph plus the matrix's sha256 -- with the bytes
regenerated locally. That trade is only honest if the regeneration is exact,
which needs two things this module checks end to end:

1. the committed road graph has to contain every edge both metrics use. The
   default trim follows the free-flow-time trees only, and the shortest matrix
   is computed on the *full* city graph, so a shortest path may leave the trim.
   ``pin_shortest_paths`` widens the trim to the union.
2. what comes back out has to be the same bytes. ``materialize_distances``
   verifies its own work against the pin and refuses to write otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mamut_routing_tools.family import build_base, materialize_distances
from mamut_routing_tools.generation.single import GenerationRequest, generate_single_instance
from mamut_routing_tools.td import build_static_bridge

#: A city where "fastest" and "shortest" genuinely disagree, which is what the
#: union trim exists for. Five nodes lie on one slow residential street running
#: due east; a fast trunk bypass arcs north between its two ends. The bypass is
#: two thirds longer but 90 km/h against 40, so it wins on time and loses on
#: distance. The residential street is drawn as a *single* way, so its three
#: interior nodes are not intersections and get contracted away -- so the direct
#: A-E edge is used by no fastest path at all. Trim to the fastest trees and that
#: edge is gone, taking the shortest matrix with it.
#:
#: A comb of dead-end spurs hangs off the bypass. The spurs carry no part of the
#: argument above; they are there because Mamut2026 will not publish an instance
#: needing fewer than six routes, and with a route-size target of at least three
#: customers that means at least eighteen customers, which means at least
#: nineteen vertices. Nothing routes *through* a dead end, so the A-E divergence
#: is exactly what it was with none of them.
_SPUR_COUNT = 24


def _divergent_osm() -> str:
    core_nodes = [
        (1, 45.000, 4.000), (2, 45.000, 4.002), (3, 45.000, 4.004),
        (4, 45.000, 4.006), (5, 45.000, 4.008),
        (6, 45.003, 4.002), (7, 45.003, 4.006),
    ]
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<osm version="0.6" generator="test">',
        '  <bounds minlat="44.99" minlon="3.99" maxlat="45.02" maxlon="4.02"/>',
    ]
    lines += [f'  <node id="{i}" lat="{lat:.5f}" lon="{lon:.5f}"/>' for i, lat, lon in core_nodes]
    for k in range(_SPUR_COUNT):
        lines.append(
            f'  <node id="{100 + k}" lat="{45.005 + 0.0006 * k:.5f}" lon="{4.002 + 0.0002 * k:.5f}"/>'
        )
    lines += [
        '  <way id="101">',
        '    <nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="5"/>',
        '    <tag k="highway" v="residential"/>',
        '  </way>',
        '  <way id="201"><nd ref="1"/><nd ref="6"/><tag k="highway" v="trunk"/></way>',
        '  <way id="202"><nd ref="6"/><nd ref="7"/><tag k="highway" v="trunk"/></way>',
        '  <way id="203"><nd ref="7"/><nd ref="5"/><tag k="highway" v="trunk"/></way>',
    ]
    for k in range(_SPUR_COUNT):
        anchor = 6 if k % 2 == 0 else 7
        lines.append(
            f'  <way id="{300 + k}"><nd ref="{anchor}"/><nd ref="{100 + k}"/>'
            '<tag k="highway" v="residential"/></way>'
        )
    lines.append("</osm>")
    return "\n".join(lines) + "\n"


DIVERGENT_OSM = _divergent_osm()

CITY = "Divergentville"
N_CUSTOMERS = 24


@pytest.fixture()
def divergent_osm_path(tmp_path: Path) -> Path:
    path = tmp_path / f"{CITY}.osm"
    path.write_text(DIVERGENT_OSM, encoding="utf-8")
    return path



def _publish(osm_path: Path, root: Path, *, commit_distances: bool):
    request = GenerationRequest(
        city=CITY,
        osm_path=osm_path,
        method="parametric_attach",
        customer_mode="random",
        depot_mode="center",
        n_customers=N_CUSTOMERS,
        # Unit demands and the shortest route-size band, so the toy instance is a
        # genuine two-route VRP. At n = 3 no legal band can produce a second
        # route: r starts at 3, so capacity always covers the whole demand. That
        # used to be hidden by a ``total - 1`` clamp in the capacity formula,
        # which manufactured a second route holding one customer.
        demand_type=1,
        avg_route_size=1,
        seed=3,
    )
    staged = generate_single_instance(request, root / "_stage1")
    folder = Path(staged["folder"])
    meta = json.loads((folder / staged["files"]["meta"]).read_text(encoding="utf-8"))
    manifest = json.loads((folder / staged["manifest"]).read_text(encoding="utf-8"))

    bridge = build_static_bridge(osm_path=osm_path, city_slug=CITY, metas=[meta])
    collection_root = root / "collection"
    built = build_base(
        graph=bridge.graph,
        nodes=bridge.nodes[meta["instance_name"]],
        meta=meta,
        manifest=manifest,
        city=CITY,
        method_tag="par",
        collection_root=collection_root,
        family="Mamut2026",
        generated_at="2026-01-01",
        commit_distances=commit_distances,
    )
    assert built is not None, "fixture base was unexpectedly already published"
    return collection_root, built


@pytest.fixture()
def committed(divergent_osm_path: Path, tmp_path: Path):
    return _publish(divergent_osm_path, tmp_path / "committed", commit_distances=True)


@pytest.fixture()
def pinned(divergent_osm_path: Path, tmp_path: Path):
    return _publish(divergent_osm_path, tmp_path / "pinned", commit_distances=False)


def _sidecar_dir(root: Path, built) -> Path:
    return root / "sidecars" / CITY.lower() / f"n={N_CUSTOMERS}" / built.base


def test_pinned_build_records_the_hashes_but_writes_no_matrix(pinned) -> None:
    root, built = pinned
    assert built.distances_committed is False

    side = _sidecar_dir(root, built)
    assert not list(side.glob("*.distances-*.json.gz"))
    # The road graph and geometry are still committed: they are what makes the
    # matrices reproducible, and they are small.
    assert (side / f"{built.base}.road.json.gz").is_file()
    assert (side / f"{built.base}.geo.json.gz").is_file()

    for metric in ("fastest", "shortest"):
        payload = json.loads(built.cvrp_paths[metric].read_text(encoding="utf-8"))
        source = payload["arc_costs_source"]
        assert source["model"] == "distances-sidecar"
        assert len(source["distances"]["sha256"]) == 64


def test_pinned_build_hashes_match_the_committed_build(committed, pinned) -> None:
    """The pin is a promise about the same instance, not a different one."""
    committed_root, committed_built = committed
    _, pinned_built = pinned
    assert committed_built.base == pinned_built.base

    for metric in ("fastest", "shortest"):
        left = json.loads(committed_built.cvrp_paths[metric].read_text(encoding="utf-8"))
        right = json.loads(pinned_built.cvrp_paths[metric].read_text(encoding="utf-8"))
        assert (
            left["arc_costs_source"]["distances"]["sha256"]
            == right["arc_costs_source"]["distances"]["sha256"]
        ), f"{metric}: pinned build disagrees with the fully committed one"
        # ... and only the sidecar bytes differ; the instance itself is identical.
        assert left["demands"] == right["demands"]
        assert left["coordinates"] == right["coordinates"]


@pytest.mark.parametrize("metric", ["fastest", "shortest"])
def test_materialize_rebuilds_the_pinned_matrix(pinned, metric: str) -> None:
    root, built = pinned
    result = materialize_distances(built.cvrp_paths[metric], collection_root=root)

    assert result["status"] == "written"
    target = Path(result["path"])
    assert target.is_file()

    payload = json.loads(built.cvrp_paths[metric].read_text(encoding="utf-8"))
    assert result["sha256"] == payload["arc_costs_source"]["distances"]["sha256"]

    # A second call is a no-op rather than a rewrite.
    assert materialize_distances(built.cvrp_paths[metric], collection_root=root)["status"] == "kept"


@pytest.mark.parametrize("metric", ["fastest", "shortest"])
def test_materialized_bytes_equal_the_committed_ones(committed, pinned, metric: str) -> None:
    """The whole claim: regenerating is not merely valid, it is the same file."""
    committed_root, committed_built = committed
    pinned_root, pinned_built = pinned

    materialize_distances(pinned_built.cvrp_paths[metric], collection_root=pinned_root)

    name = f"{committed_built.base}.distances-{metric}.json.gz"
    left = (_sidecar_dir(committed_root, committed_built) / name).read_bytes()
    right = (_sidecar_dir(pinned_root, pinned_built) / name).read_bytes()
    assert left == right


def test_euclidean_variant_has_nothing_to_materialize(pinned) -> None:
    root, built = pinned
    result = materialize_distances(built.cvrp_paths["euclidean"], collection_root=root)
    assert result["status"] == "euclidean"
    assert result["path"] is None


def test_materialize_refuses_a_road_graph_that_does_not_reproduce_the_pin(pinned) -> None:
    """A tampered road sidecar must fail loudly, not publish wrong distances."""
    root, built = pinned
    instance = built.cvrp_paths["shortest"]
    payload = json.loads(instance.read_text(encoding="utf-8"))
    payload["arc_costs_source"]["distances"]["sha256"] = "0" * 64
    instance.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AssertionError, match="but the instance pins"):
        materialize_distances(instance, collection_root=root)


def test_the_union_trim_is_what_makes_the_pin_reproducible(committed, pinned) -> None:
    """Without it the fixture's direct street would be trimmed away.

    ``_trim_road_graph`` follows the free-flow-time trees, and on this city no
    fastest path uses the slow direct edge -- every one of them takes the trunk
    bypass. The shortest matrix, computed on the full graph, does use it. So the
    default road sidecar is strictly smaller than the pinned one, and a pinned
    build against the default trim would fail ``build_base``'s own gate rather
    than publish an unreproducible hash.
    """
    _, committed_built = committed
    _, pinned_built = pinned
    assert pinned_built.num_road_edges > committed_built.num_road_edges, (
        "the fixture no longer separates the two trims, so the round-trip tests "
        "above would pass even if pin_shortest_paths did nothing"
    )
