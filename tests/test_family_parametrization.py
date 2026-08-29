"""``family.build_base`` publishes into the family it is told to.

The family name used to be the module-level constant ``FAMILY``; every artifact
(collection marker, instance payloads, sidecars, the CVRPLIB comment) hardcoded
"Poryos2026" and every base name hardcoded the ``poryos-`` prefix. Mamut2026
reuses this builder, so the name became a parameter.

Two things have to hold, and both are checked end to end on the synthetic
fixture city rather than by reading the source:

1. the default is still Poryos2026, byte for byte -- an existing collection must
   not move because a parameter appeared;
2. passing a family changes the family and *only* the family: the geometry,
   demands, capacity and arc costs of the two builds are identical.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mamut_routing_lib.sidecars import COLLECTION_MARKER_FILENAME

from mamut_routing_tools.family import build_base, family_prefix
from mamut_routing_tools.generation.single import GenerationRequest, generate_single_instance
from mamut_routing_tools.td import build_static_bridge

CITY = "Testville"
N_CUSTOMERS = 24

#: A local road graph rather than the shared ``fixture_osm_path``.
#:
#: Mamut2026 will not publish an instance needing fewer than six routes, and a
#: route-size target starts at three customers, so the smallest publishable
#: instance has eighteen customers and needs nineteen vertices. The shared
#: fixture yields five, and its amenity layout is tuned for the POI-attachment
#: tests; growing it to suit this one would put those at risk.
#:
#: A plain chain, each segment its own way so no node is contracted away. This
#: test is about naming and layout, not about routing.
_CHAIN_LENGTH = 30


def _family_osm() -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<osm version="0.6" generator="test">',
        '  <bounds minlat="44.99" minlon="3.99" maxlat="45.02" maxlon="4.02"/>',
    ]
    for i in range(_CHAIN_LENGTH):
        lines.append(
            f'  <node id="{i + 1}" lat="{45.0 + 0.0004 * i:.5f}" lon="{4.0 + 0.0003 * i:.5f}"/>'
        )
    for i in range(_CHAIN_LENGTH - 1):
        lines.append(
            f'  <way id="{100 + i}"><nd ref="{i + 1}"/><nd ref="{i + 2}"/>'
            '<tag k="highway" v="residential"/></way>'
        )
    lines.append("</osm>")
    return "\n".join(lines) + "\n"


FAMILY_OSM = _family_osm()


@pytest.fixture()
def family_osm_path(tmp_path: Path) -> Path:
    path = tmp_path / f"{CITY}.osm"
    path.write_text(FAMILY_OSM, encoding="utf-8")
    return path


def _build(osm_path: Path, root: Path, **family_kwarg: str) -> tuple[Path, dict]:
    """Generate stage 1 on the fixture city and publish one base into ``root``."""
    request = GenerationRequest(
        city=CITY,
        osm_path=osm_path,
        method="parametric_attach",
        customer_mode="random",
        depot_mode="center",
        n_customers=N_CUSTOMERS,
        # Unit demands in the shortest route-size band: the smallest configuration
        # that is a genuine two-route VRP, which both families require. Below
        # this, capacity covers the whole demand and LB_cap is 1.
        demand_type=1,
        avg_route_size=1,
        seed=11,
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
        generated_at="2026-01-01",
        **family_kwarg,
    )
    assert built is not None, "fixture base was unexpectedly already published"
    return collection_root, {
        metric: json.loads(path.read_text(encoding="utf-8"))
        for metric, path in built.cvrp_paths.items()
    }


@pytest.fixture()
def default_build(family_osm_path: Path, tmp_path: Path):
    return _build(family_osm_path, tmp_path / "default")


def test_default_family_is_still_poryos2026(default_build) -> None:
    root, payloads = default_build
    marker = json.loads((root / COLLECTION_MARKER_FILENAME).read_text(encoding="utf-8"))
    assert marker["family"] == "Poryos2026"

    base = f"poryos-{CITY.lower()}-n{N_CUSTOMERS}-par"
    for metric, payload in payloads.items():
        assert payload["instance_name"] == base
        assert payload["benchmark_name"] == "Poryos2026"
        assert payload["metadata"]["base_instance_name"] == base
        assert (root / "CVRP" / metric / CITY.lower() / f"n={N_CUSTOMERS}" / base).is_dir()

    vrp = (root / "CVRP" / "fastest" / CITY.lower() / f"n={N_CUSTOMERS}" / base / f"{base}.vrp")
    assert vrp.read_text(encoding="utf-8").splitlines()[1].startswith(
        "COMMENT : Poryos2026 fastest metric;"
    )


def test_explicit_poryos2026_matches_the_default(family_osm_path: Path, tmp_path: Path, default_build) -> None:
    _, default_payloads = default_build
    _, explicit_payloads = _build(family_osm_path, tmp_path / "explicit", family="Poryos2026")
    assert explicit_payloads == default_payloads


def test_mamut2026_names_carry_the_fleet_lower_bound(
    family_osm_path: Path, tmp_path: Path, default_build
) -> None:
    """Mamut2026 follows CVRPLIB's ``X-n101-k25``; Poryos2026 does not.

    The ``k`` is the bin-packing minimum, published as a lower bound on the route
    count rather than as a fleet cap -- ``num_vehicles`` stays unset either way.
    Poryos2026 predates the convention and is already published, so its names
    must not move.
    """
    _, default_payloads = default_build
    root, mamut_payloads = _build(family_osm_path, tmp_path / "mamut", family="Mamut2026")

    marker = json.loads((root / COLLECTION_MARKER_FILENAME).read_text(encoding="utf-8"))
    assert marker["family"] == "Mamut2026"

    lb = next(iter(mamut_payloads.values()))["metadata"]["num_vehicles_lb"]
    base = f"mamut-{CITY.lower()}-n{N_CUSTOMERS}-k{lb}-par"
    for metric, payload in mamut_payloads.items():
        assert payload["instance_name"] == base
        assert payload["benchmark_name"] == "Mamut2026"
        assert (root / "CVRP" / metric / CITY.lower() / f"n={N_CUSTOMERS}" / base).is_dir()
        # A lower bound, not a cap: the fleet stays free.
        assert payload["num_vehicles"] is None

    # The same instance under Poryos2026 keeps its k-less name.
    assert all(
        "-k" not in payload["instance_name"] for payload in default_payloads.values()
    )

    # Everything that is not an identity must be untouched: same points, same
    # demands, same capacity, same arc costs. Only the sidecar *paths* differ,
    # because they embed the base name.
    for metric, payload in mamut_payloads.items():
        reference = default_payloads[metric]
        for field in ("coordinates", "demands", "vehicle_capacity", "depot", "reference_lla"):
            assert payload[field] == reference[field], field
        assert payload["metadata"]["num_vehicles_lb"] == reference["metadata"]["num_vehicles_lb"]
        if payload["arc_costs_source"]["model"] == "euclidean":
            assert payload["arc_costs_source"] == reference["arc_costs_source"]
        else:
            assert (
                payload["arc_costs_source"]["distances"]["sha256"]
                != reference["arc_costs_source"]["distances"]["sha256"]
            ), "the sidecar pins its own base name, so the digest must move with it"


@pytest.mark.parametrize(
    ("family", "expected"),
    [("Poryos2026", "poryos"), ("Mamut2026", "mamut"), ("Sintef2008", "sintef")],
)
def test_family_prefix_strips_the_release_year(family: str, expected: str) -> None:
    assert family_prefix(family) == expected


def test_family_prefix_rejects_a_bare_year() -> None:
    with pytest.raises(ValueError, match="no name part"):
        family_prefix("2026")
