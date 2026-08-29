"""The ceiling on POI-only instances, and why it is not the amenity count.

A POI-only instance draws every customer from a real amenity, so its size is
bounded by how many *distinct road vertices* the city's amenities reach. Two
things pull that below the amenity count and one thing is easy to get wrong:

- amenities that share a corner collapse into a single customer;
- amenities too far from any routable point never attach at all;
- and the whole figure moves by a factor of several between attach modes, so a
  capacity quoted without its mode is not a number.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mamut_routing_tools.campaign import (
    PoiCapacity,
    categories_digest,
    cities_supporting,
    load_capacities,
    measure_capacity,
    measure_cities,
    save_capacities,
)
from mamut_routing_tools.generation.select import (
    POI_ATTACH_NEAREST_NODE,
    POI_ATTACH_NEAREST_VERTEX,
)


def _capacity(**overrides) -> PoiCapacity:
    base = dict(
        city="somewhere",
        osm_file="Somewhere.osm",
        num_vertices=1000,
        catalog_size=500,
        attached_pois=400,
        capacity=300,
        collapse_ratio=1.33,
        attach_mode=POI_ATTACH_NEAREST_VERTEX,
        attach_radius_m=50.0,
    )
    base.update(overrides)
    return PoiCapacity(**base)


def test_capacity_counts_vertices_not_amenities(fixture_osm_path: Path) -> None:
    measured = measure_capacity(fixture_osm_path, "testville")

    assert measured.city == "testville"
    assert measured.catalog_size >= measured.attached_pois >= measured.capacity >= 1
    # The fixture holds a museum far from any road and a bank beside a
    # disconnected island: neither can become a customer.
    assert measured.attached_pois < measured.catalog_size
    assert measured.collapse_ratio == pytest.approx(
        measured.attached_pois / measured.capacity, rel=1e-3
    )


def test_the_attach_mode_changes_the_answer(fixture_osm_path: Path) -> None:
    """Which is why the mode is recorded alongside the number.

    ``nearest_node`` demands that a POI's closest road *node* be a graph vertex
    in its own right; ``nearest_vertex`` snaps within a radius. The strict rule
    discards most of a real city's amenities, so a capacity measured under one
    mode says nothing about the other.
    """
    strict = measure_capacity(fixture_osm_path, "testville", attach_mode=POI_ATTACH_NEAREST_NODE)
    lenient = measure_capacity(fixture_osm_path, "testville", attach_mode=POI_ATTACH_NEAREST_VERTEX)

    assert strict.attach_mode == POI_ATTACH_NEAREST_NODE
    assert lenient.attach_mode == POI_ATTACH_NEAREST_VERTEX
    assert lenient.capacity >= strict.capacity


def test_only_the_requested_categories_count(fixture_osm_path: Path) -> None:
    everything = measure_capacity(fixture_osm_path, "testville")
    cafes = measure_capacity(fixture_osm_path, "testville", categories=["cafe"])

    assert cafes.catalog_size < everything.catalog_size
    assert cafes.capacity <= everything.capacity


def test_measure_cities_is_sorted_and_reports_progress(fixture_osm_path: Path) -> None:
    seen: list[str] = []
    results = measure_cities(
        [("b_town", fixture_osm_path), ("a_town", fixture_osm_path)],
        on_progress=seen.append,
    )
    assert [item.city for item in results] == ["a_town", "b_town"]
    assert sorted(seen) == ["a_town", "b_town"]


def test_measure_cities_handles_an_empty_set() -> None:
    assert measure_cities([]) == []


@pytest.mark.parametrize(
    ("capacity", "n", "expected"),
    [(101, 100, True), (100, 100, False), (99, 100, False)],
)
def test_supports_leaves_room_for_the_depot(capacity: int, n: int, expected: bool) -> None:
    assert _capacity(capacity=capacity).supports(n) is expected


def test_cities_supporting_ranks_by_capacity_and_demands_headroom() -> None:
    pool = [
        _capacity(city="huge", capacity=9000),
        _capacity(city="ample", capacity=2600),
        _capacity(city="exact", capacity=2001),
        _capacity(city="short", capacity=1500),
    ]

    # Without headroom, "exact" qualifies -- but its selection would be the whole
    # amenity pool, so the seed stops mattering and the instance stops being a draw.
    assert [c.city for c in cities_supporting(pool, 2000, headroom=1.0)] == [
        "huge",
        "ample",
        "exact",
    ]
    assert [c.city for c in cities_supporting(pool, 2000, headroom=1.25)] == ["huge", "ample"]


def test_capacities_round_trip_through_disk(tmp_path: Path) -> None:
    original = [_capacity(city="a"), _capacity(city="b", capacity=42)]
    path = tmp_path / "nested" / "poi-capacity.json"
    save_capacities(original, path)
    assert load_capacities(path) == original


def test_a_capacity_measured_over_other_categories_is_refused(tmp_path: Path) -> None:
    """Every number in the file is relative to a category list.

    Curating the categories changes all of them, and a stale file is otherwise
    indistinguishable from a fresh one -- so the campaign would classify cities
    by amenities it is no longer drawing from and size instances against them.
    """
    path = tmp_path / "poi-capacity.json"
    save_capacities([_capacity(categories_digest=categories_digest(["cafe", "bar"]))], path)

    # Same list, any order: fine.
    assert load_capacities(path, expect_categories=["bar", "cafe"])

    with pytest.raises(ValueError, match="re-run the poi-capacity phase"):
        load_capacities(path, expect_categories=["cafe", "bar", "restaurant"])

    # Unchecked loading still works, for tools that only want the numbers.
    assert len(load_capacities(path)) == 1


def test_a_file_predating_the_digest_never_passes_the_check(tmp_path: Path) -> None:
    path = tmp_path / "poi-capacity.json"
    save_capacities([_capacity()], path)   # digest defaults to ""
    with pytest.raises(ValueError, match="re-run the poi-capacity phase"):
        load_capacities(path, expect_categories=["cafe"])


def test_the_measurement_records_the_categories_it_used(fixture_osm_path: Path) -> None:
    measured = measure_capacity(fixture_osm_path, "testville", categories=["cafe", "bar"])
    assert measured.categories_digest == categories_digest(["bar", "cafe"])
