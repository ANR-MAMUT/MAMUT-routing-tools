"""Sizes come first; cities are matched to them.

The design this replaces drew ``n``, ``method`` and the city independently and
found out during generation that the city could not serve the draw -- at which
point the generator quietly topped the instance up with sampled road points and
relabelled it. These tests pin the two properties that make the new order
trustworthy: a rung is never given to a city that cannot supply it, and a rung
that nobody can supply is *reported* rather than downgraded.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from mamut_routing_tools.campaign.city_profile import CityProfile
from mamut_routing_tools.campaign.design import CAMPAIGN_MIN_ROUTES
from mamut_routing_tools.campaign.ladder import (
    DEFAULT_METHOD_WEIGHTS,
    LadderPlan,
    admissible_route_size_bands,
    assign_route_size_bands,
    assign_rungs,
    band_size_correlation,
    load_plan,
    plan_summary,
    required_supply,
    save_plan,
    size_ladder,
    weighted_column,
)
from mamut_routing_tools.campaign.poi_capacity import PoiCapacity
from mamut_routing_tools.generation.demands import AVG_ROUTE_SIZES, avg_route_size_bounds

# --------------------------------------------------------------------------
# the ladder
# --------------------------------------------------------------------------


def test_the_ladder_rises_smoothly_from_end_to_end() -> None:
    rungs = size_ladder(100, 100, 1000)

    assert len(rungs) == 100
    assert len(set(rungs)) == 100, "every instance must have its own size"
    assert rungs[0] == 100 and rungs[-1] == 1000, "endpoints are exact"
    assert all(later > earlier for earlier, later in zip(rungs, rungs[1:]))


def test_the_bottom_of_the_ladder_survives_rounding() -> None:
    """Geometric spacing collides down there; the repair pass has to catch it.

    Between 100 and 110 a 100-rung ladder wants steps of ~2.3, and naive
    rounding produces duplicates. A duplicate would silently cost the set a
    distinct size and break the one-instance-per-rung quota.
    """
    rungs = size_ladder(100, 100, 1000)
    bottom = rungs[:10]
    assert len(set(bottom)) == len(bottom), bottom
    assert all(later > earlier for earlier, later in zip(bottom, bottom[1:]))


def test_a_tight_ladder_still_produces_distinct_rungs() -> None:
    """The worst case: exactly as many rungs as there are integers available."""
    rungs = size_ladder(11, 100, 110)
    assert rungs == list(range(100, 111))


@pytest.mark.parametrize(
    ("k", "n_min", "n_max", "message"),
    [
        (1, 100, 1000, "at least two rungs"),
        (10, 1000, 100, "n_min < n_max"),
        (10, 100, 105, "cannot fit"),
    ],
)
def test_size_ladder_rejects_impossible_requests(k, n_min, n_max, message) -> None:
    with pytest.raises(ValueError, match=message):
        size_ladder(k, n_min, n_max)


def test_the_ladder_is_denser_at_the_bottom() -> None:
    """Where a change in n changes the instance most."""
    rungs = size_ladder(100, 100, 1000)
    first_step = rungs[1] - rungs[0]
    last_step = rungs[-1] - rungs[-2]
    assert last_step > first_step * 5


# --------------------------------------------------------------------------
# the sourcing mix
# --------------------------------------------------------------------------


def test_the_sourcing_mix_is_exact_not_approximate() -> None:
    column = weighted_column(DEFAULT_METHOD_WEIGHTS, 100, np.random.default_rng(0))
    assert Counter(column) == {
        "poi_categories": 50,
        "hybrid": 25,
        "parametric_attach": 25,
    }


def test_a_remainder_never_comes_out_of_the_largest_share() -> None:
    """7 slots at 50/25/25 must be 4/2/1 or 3/2/2 -- never 2/3/2."""
    column = weighted_column(DEFAULT_METHOD_WEIGHTS, 7, np.random.default_rng(1))
    counts = Counter(column)
    assert len(column) == 7
    assert counts["poi_categories"] >= counts["hybrid"]
    assert counts["poi_categories"] >= counts["parametric_attach"]


def test_the_mix_is_deterministic_under_a_fixed_seed() -> None:
    left = weighted_column(DEFAULT_METHOD_WEIGHTS, 40, np.random.default_rng(7))
    right = weighted_column(DEFAULT_METHOD_WEIGHTS, 40, np.random.default_rng(7))
    assert left == right


@pytest.mark.parametrize("weights", [{}, {"a": 0.0}])
def test_weighted_column_rejects_a_degenerate_mix(weights) -> None:
    with pytest.raises(ValueError):
        weighted_column(weights, 10, np.random.default_rng(0))


# --------------------------------------------------------------------------
# what a rung demands
# --------------------------------------------------------------------------


def test_a_poi_rung_demands_amenities_and_a_parametric_one_demands_road() -> None:
    """The two resources are not interchangeable, and conflating them is how
    the old design talked itself into instances it could not build."""
    assert required_supply(100, "poi_categories", hybrid_share=0.5, headroom=1.25,
                           vertex_factor=4.0) == ("poi", 127)
    assert required_supply(100, "hybrid", hybrid_share=0.5, headroom=1.25,
                           vertex_factor=4.0) == ("poi", 64)
    assert required_supply(100, "parametric_attach", hybrid_share=0.5, headroom=1.25,
                           vertex_factor=4.0) == ("vertices", 404)


def test_required_supply_counts_the_depot() -> None:
    resource, amount = required_supply(
        99, "poi_categories", hybrid_share=0.5, headroom=1.0, vertex_factor=1.0
    )
    assert (resource, amount) == ("poi", 100)


def test_required_supply_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown sourcing method"):
        required_supply(100, "telepathy", hybrid_share=0.5, headroom=1.25, vertex_factor=4.0)


# --------------------------------------------------------------------------
# the assignment
# --------------------------------------------------------------------------


def _city(name: str, capacity: int, vertices: int = 100_000) -> PoiCapacity:
    return PoiCapacity(
        city=name,
        osm_file=f"{name}.osm",
        num_vertices=vertices,
        catalog_size=capacity * 2,
        attached_pois=int(capacity * 1.5),
        capacity=capacity,
        collapse_ratio=1.5,
        attach_mode="nearest_vertex",
        attach_radius_m=50.0,
        categories_digest="test",
    )


def _profile(name: str, stratum: str) -> CityProfile:
    return CityProfile(
        city=name,
        osm_file=f"{name}.osm",
        num_vertices=100_000,
        num_edges=200_000,
        detour_mean=1.3,
        detour_p90=1.6,
        rank_tau_eucl_fast=0.8,
        num_pairs=100,
        distortion_stratum=stratum,
    )


def test_no_rung_is_ever_given_to_a_city_that_cannot_supply_it() -> None:
    """The property the whole module exists for."""
    cities = [_city(f"c{i}", capacity=200 * (i + 1)) for i in range(20)]
    plan = assign_rungs(size_ladder(20, 100, 400), cities)

    for item in plan.assignments:
        assert item.available >= item.required, item


def test_best_fit_reserves_the_large_cities_for_the_large_rungs() -> None:
    """Taking the biggest available city first would spend it on n=100.

    Two cities, two POI rungs. The small city can only serve the small rung, so
    a correct matching must give the large city the large one -- which a
    largest-first greedy would already have consumed.
    """
    cities = [_city("small", capacity=200), _city("large", capacity=2000)]
    plan = assign_rungs(
        [100, 1000],
        cities,
        method_weights={"poi_categories": 1.0},
        headroom=1.0,
    )

    assert plan.complete, plan.unfilled
    by_n = {item.n: item.city for item in plan.assignments}
    assert by_n == {100: "small", 1000: "large"}


def test_an_unfillable_rung_is_reported_with_what_it_needed() -> None:
    """Not dropped, and above all not quietly downgraded to another method."""
    cities = [_city("tiny", capacity=150)]
    plan = assign_rungs(
        [100, 900],
        cities,
        method_weights={"poi_categories": 1.0},
        headroom=1.0,
    )

    assert not plan.complete
    assert [item.n for item in plan.assignments] == [100]
    (orphan,) = plan.unfilled
    assert orphan.n == 900
    assert orphan.method == "poi_categories"
    assert orphan.required == 901
    # The report names the shortfall precisely enough to act on: 901 amenities
    # wanted, 150 the best city can offer. Note the hardest rung is matched
    # first, so "tiny" is still in the pool here and is reported as the best
    # available even though it goes on to serve the n=100 rung.
    assert (orphan.best_city, orphan.best_available) == ("tiny", 150)


def test_the_shortfall_report_names_the_best_city_still_available() -> None:
    cities = [_city("tiny", capacity=150), _city("small", capacity=400)]
    plan = assign_rungs(
        [100, 5000],
        cities,
        method_weights={"poi_categories": 1.0},
        headroom=1.0,
    )
    (orphan,) = plan.unfilled
    assert orphan.n == 5000
    assert orphan.best_city == "small"
    assert orphan.best_available == 400


def test_each_city_is_used_at_most_once() -> None:
    cities = [_city(f"c{i}", capacity=5000) for i in range(30)]
    plan = assign_rungs(size_ladder(30, 100, 1000), cities)

    used = [item.city for item in plan.assignments]
    assert len(used) == len(set(used))


def test_a_parametric_rung_is_matched_on_road_vertices_not_amenities() -> None:
    """A city with no amenities at all can still carry a parametric instance."""
    cities = [_city("bare", capacity=0, vertices=500_000)]
    plan = assign_rungs(
        [500],
        cities,
        method_weights={"parametric_attach": 1.0},
    )
    assert plan.complete
    assert plan.assignments[0].city == "bare"


def test_parametric_rungs_spend_the_amenity_poorest_cities_first() -> None:
    """Amenity capacity is scarce; road is not. Spend the city that costs least.

    Ranking parametric rungs by road vertices makes a huge, amenity-dead extract
    the *last* candidate for the only kind of rung it can serve, so it is never
    used and the ladder runs out of cities while dead extracts sit unspent. This
    is the case that took the real campaign from 100 filled rungs to 95.
    """
    cities = [
        _city("dead", capacity=0, vertices=500_000),     # useless for POI
        _city("rich", capacity=8000, vertices=60_000),   # precious for POI
    ]
    plan = assign_rungs(
        [200, 400],
        cities,
        method_weights={"parametric_attach": 0.5, "poi_categories": 0.5},
        headroom=1.0,
    )

    assert plan.complete, plan.unfilled
    by_method = {item.method: item.city for item in plan.assignments}
    assert by_method["parametric_attach"] == "dead"
    assert by_method["poi_categories"] == "rich"


def test_ties_break_toward_the_under_represented_stratum() -> None:
    """Amenity counts drive the matching; road-network diversity must survive it."""
    cities = [_city(f"c{i}", capacity=5000) for i in range(9)]
    profiles = [
        _profile("c0", "low"), _profile("c1", "low"), _profile("c2", "low"),
        _profile("c3", "mid"), _profile("c4", "mid"), _profile("c5", "mid"),
        _profile("c6", "high"), _profile("c7", "high"), _profile("c8", "high"),
    ]
    plan = assign_rungs(size_ladder(9, 100, 900), cities, profiles)

    counts = Counter(item.distortion_stratum for item in plan.assignments)
    assert set(counts) == {"low", "mid", "high"}
    assert max(counts.values()) - min(counts.values()) <= 1, counts


def test_the_assignment_is_deterministic() -> None:
    cities = [_city(f"c{i}", capacity=200 * (i + 1)) for i in range(20)]
    ladder = size_ladder(20, 100, 400)
    left = assign_rungs(ladder, cities, base_seed=5)
    right = assign_rungs(ladder, cities, base_seed=5)
    assert left.to_dict() == right.to_dict()
    assert assign_rungs(ladder, cities, base_seed=6).to_dict() != left.to_dict()


def test_assignments_come_back_in_ladder_order() -> None:
    """Matching runs hardest-first; the plan must read as the ladder does."""
    cities = [_city(f"c{i}", capacity=200 * (i + 1)) for i in range(20)]
    plan = assign_rungs(size_ladder(20, 100, 400), cities)
    sizes = [item.n for item in plan.assignments]
    assert sizes == sorted(sizes)
    assert [item.index for item in plan.assignments] == sorted(
        item.index for item in plan.assignments
    )


def test_rung_keys_are_unique() -> None:
    """They become a quota level, one instance each."""
    cities = [_city(f"c{i}", capacity=5000) for i in range(30)]
    plan = assign_rungs(size_ladder(30, 100, 1000), cities)
    keys = [item.rung_key for item in plan.assignments]
    assert len(keys) == len(set(keys))


def test_a_plan_round_trips_through_disk(tmp_path) -> None:
    cities = [_city(f"c{i}", capacity=200 * (i + 1)) for i in range(20)]
    plan = assign_rungs(size_ladder(20, 100, 400), cities)
    path = tmp_path / "nested" / "ladder-main.json"
    save_plan(plan, path)
    assert load_plan(path).to_dict() == plan.to_dict()


def test_plan_summary_reports_the_mix_that_came_out() -> None:
    cities = [_city(f"c{i}", capacity=5000) for i in range(20)]
    plan = assign_rungs(size_ladder(20, 100, 1000), cities)
    summary = plan_summary(plan)

    assert summary["assigned"] == 20
    assert summary["unfilled"] == 0
    assert summary["cities"] == 20
    assert summary["n_min"] == 100 and summary["n_max"] == 1000
    assert summary["methods"]["poi_categories"] == 10


def test_an_empty_plan_summarises_without_crashing() -> None:
    summary = plan_summary(LadderPlan())
    assert summary["assigned"] == 0
    assert summary["n_min"] is None


# --------------------------------------------------------------------------
# enumeration and selection over an assigned ladder
# --------------------------------------------------------------------------


def test_rung_enumeration_pins_the_assignment_and_varies_the_rest() -> None:
    """City, size and method are settled; character is still up for selection."""
    from collections import Counter as _Counter

    from mamut_routing_tools.campaign.design import enumerate_for_rungs

    cities = [_city(f"c{i}", capacity=5000) for i in range(12)]
    plan = assign_rungs(size_ladder(12, 100, 1000), cities)
    specs = enumerate_for_rungs(plan.assignments, per_rung=6)

    assert len(specs) == 72
    by_rung: dict[str, list] = {}
    for spec in specs:
        by_rung.setdefault(spec.rung_key, []).append(spec)
    assert len(by_rung) == 12

    for group in by_rung.values():
        assert len({(s.city, s.n, s.method) for s in group}) == 1, (
            "the assignment is not negotiable at enumeration time"
        )
        # ... but the instance's character has to differ, or the selector has
        # nothing to choose between.
        assert len({s.seed for s in group}) == len(group)

    counts = _Counter(s.demand_type for s in specs)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_rung_enumeration_is_reproducible_and_seed_sensitive() -> None:
    from mamut_routing_tools.campaign.design import enumerate_for_rungs

    cities = [_city(f"c{i}", capacity=5000) for i in range(6)]
    plan = assign_rungs(size_ladder(6, 100, 600), cities)
    same = enumerate_for_rungs(plan.assignments, per_rung=3, base_seed=2)
    assert [s.to_dict() for s in same] == [
        s.to_dict() for s in enumerate_for_rungs(plan.assignments, per_rung=3, base_seed=2)
    ]
    other = enumerate_for_rungs(plan.assignments, per_rung=3, base_seed=3)
    assert [s.to_dict() for s in same] != [s.to_dict() for s in other]


def test_rung_enumeration_rejects_a_degenerate_request() -> None:
    from mamut_routing_tools.campaign.design import enumerate_for_rungs

    assert enumerate_for_rungs([], per_rung=4) == []
    cities = [_city("c0", capacity=5000)]
    plan = assign_rungs([100, 200], cities)
    with pytest.raises(ValueError, match="per_rung must be"):
        enumerate_for_rungs(plan.assignments, per_rung=0)


def test_the_ladder_quota_fills_every_rung_exactly_once() -> None:
    """min == max == 1 turns "pick 100 good ones" into "pick one per slot"."""
    import numpy as _np

    from mamut_routing_tools.campaign.descriptors import InstanceDescriptors
    from mamut_routing_tools.campaign.design import EvaluatedCandidate, enumerate_for_rungs
    from mamut_routing_tools.campaign.quotas import ladder_tier_quotas
    from mamut_routing_tools.campaign.select import select_maxmin

    cities = [_city(f"c{i}", capacity=5000) for i in range(12)]
    profiles = [_profile(f"c{i}", ("low", "mid", "high")[i % 3]) for i in range(12)]
    plan = assign_rungs(size_ladder(12, 100, 1000), cities, profiles)
    specs = enumerate_for_rungs(plan.assignments, per_rung=5)

    rng = _np.random.default_rng(0)
    pool = [
        EvaluatedCandidate(
            spec=spec,
            descriptors=InstanceDescriptors(
                n=spec.n,
                extent_m=1000.0,
                clark_evans_r=float(rng.uniform(0.4, 1.6)),
                nnd_cv=float(rng.uniform(0.1, 1.2)),
                radial_dispersion=float(rng.uniform(0.2, 0.9)),
                depot_centrality=float(rng.uniform(0.0, 2.0)),
                depot_eccentricity=float(rng.uniform(0.0, 1.0)),
                demand_cv=float(rng.uniform(0.1, 1.5)),
                demand_gini=float(rng.uniform(0.05, 0.6)),
                demand_max_over_mean=float(rng.uniform(1.5, 8.0)),
                demand_moran_i=float(rng.uniform(-0.2, 0.6)),
                lb_cap=max(2, spec.n // 20),
                route_size=float(rng.uniform(3.0, 40.0)),
                capacity_slack=float(rng.uniform(0.0, 0.4)),
            ),
            resolved_n=spec.n,
            effective_method=spec.method,
            poi_fraction=1.0 if spec.method == "poi_categories" else 0.0,
        )
        for spec in specs
    ]

    keys = [item.rung_key for item in plan.assignments]
    selected = select_maxmin(pool, len(keys), ladder_tier_quotas(keys))

    assert len(selected) == len(keys)
    assert sorted(c.spec.rung_key for c in selected) == sorted(keys)
    assert len({c.spec.city for c in selected}) == len(keys)
    assert sorted(c.spec.n for c in selected) == size_ladder(12, 100, 1000)


def test_rung_enumeration_resolves_the_extract_directory() -> None:
    """A RungAssignment carries a file name, not a path.

    Capacity measurements are shared across checkouts, so an absolute path would
    not survive the trip -- but a bare name does not open either, and the
    generator's failure is a warning inside a worker rather than an error.
    """
    from pathlib import Path as _Path

    from mamut_routing_tools.campaign.design import enumerate_for_rungs

    plan = assign_rungs([200], [_city("lyon", capacity=5000)])
    (spec,) = enumerate_for_rungs(plan.assignments, osm_dir="extracts", per_rung=1)
    assert spec.osm_path == _Path("extracts") / "lyon.osm"
    assert spec.to_request().osm_path == _Path("extracts") / "lyon.osm"


def test_the_tiers_cannot_produce_colliding_base_names() -> None:
    """A base is named for its city, size and sourcing tag -- nothing else.

    So a POI-tier instance reusing a main-tier city at the same size would
    collide, and build_base returns early on an existing base: the collision
    would read as a successful build while silently republishing the main-tier
    instance. Keeping the grids disjoint is what prevents it.
    """
    from mamut_routing_tools.campaign.design import POI_SIZE_GRID

    main_ladder = set(size_ladder(100, 100, 1000))
    assert not (main_ladder & set(POI_SIZE_GRID))
    assert min(POI_SIZE_GRID) > max(main_ladder)


# --------------------------------------------------------------------------
# route-size bands
# --------------------------------------------------------------------------


def test_a_band_is_admissible_only_where_its_whole_range_fits() -> None:
    """Full containment, not "the draw happened to be small enough".

    ``r`` is drawn uniformly inside the band at generation time, so a band that
    only partly fits would make soundness a coin flip. Band 5 tops out at 25
    customers per route, so it needs n >= 150 to leave six routes.
    """
    assert admissible_route_size_bands(150, min_routes=6) == [1, 2, 3, 4, 5]
    assert admissible_route_size_bands(149, min_routes=6) == [1, 2, 3, 4]
    for n in (100, 250, 1000, 4000):
        for band in admissible_route_size_bands(n, min_routes=6):
            assert avg_route_size_bounds(band)[1] * 6 <= n


def test_the_longest_band_never_reaches_the_main_tier() -> None:
    """Band 7 is CVRPLIB's XL vocabulary, and XL starts at n = 1000.

    v2 drew it anywhere. ``mamut-chicago-n285-poi`` came out of that draw with
    two routes of 142 customers, which is not a vehicle routing problem.
    """
    assert 7 not in admissible_route_size_bands(1000)
    assert 7 in admissible_route_size_bands(1200)


def test_every_rung_of_the_published_ladder_keeps_a_real_choice() -> None:
    for n in size_ladder(100):
        assert len(admissible_route_size_bands(n)) >= 4


def test_a_size_below_the_shortest_band_is_an_error_not_a_silent_downgrade() -> None:
    with pytest.raises(ValueError, match="no route-size band"):
        admissible_route_size_bands(10)


def test_bands_are_apportioned_evenly_and_stay_admissible() -> None:
    ladder = size_ladder(100)
    bands = assign_route_size_bands(ladder, np.random.default_rng(2026))

    assert len(bands) == len(ladder)
    for n, band in zip(ladder, bands):
        assert band in admissible_route_size_bands(n)

    counts = Counter(bands)
    # Band 7 is unreachable below n = 1200, so the split is over bands 1-6.
    assert set(counts) == {1, 2, 3, 4, 5, 6}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_the_assignment_guarantees_the_minimum_fleet() -> None:
    """The property the whole change exists for, at the level of the plan."""
    ladder = size_ladder(100)
    bands = assign_route_size_bands(ladder, np.random.default_rng(7))
    worst = min(n / avg_route_size_bounds(band)[1] for n, band in zip(ladder, bands))
    assert worst >= CAMPAIGN_MIN_ROUTES


def test_band_assignment_is_decorrelated_from_size() -> None:
    """Route size is a design axis, not a proxy for ``n``.

    Admissibility puts a floor under this -- band 6 cannot appear below n = 300 --
    so the swap pass cannot reach exactly zero by right. It gets close enough
    that the axis is usable as an independent covariate.
    """
    ladder = size_ladder(100)
    for seed in (0, 7, 42, 2026):
        bands = assign_route_size_bands(ladder, np.random.default_rng(seed))
        assert abs(band_size_correlation(ladder, bands)) < 0.05


def test_band_assignment_is_deterministic_under_a_fixed_seed() -> None:
    ladder = size_ladder(60)
    first = assign_route_size_bands(ladder, np.random.default_rng(11))
    second = assign_route_size_bands(ladder, np.random.default_rng(11))
    assert first == second


def test_the_plan_carries_a_band_per_rung_and_reports_its_guarantees() -> None:
    ladder = size_ladder(12)
    cities = [_city(f"c{i}", capacity=40_000) for i in range(12)]
    plan = assign_rungs(ladder, cities, base_seed=5)

    assert plan.complete
    assert len(plan.assignments) == 12
    for item in plan.assignments:
        assert item.avg_route_size in admissible_route_size_bands(item.n)
    assert plan.guaranteed_min_routes >= CAMPAIGN_MIN_ROUTES

    payload = plan.to_dict()
    assert payload["min_routes"] == CAMPAIGN_MIN_ROUTES
    assert payload["guaranteed_min_routes"] == plan.guaranteed_min_routes
    # and it survives a round trip
    assert [a.avg_route_size for a in LadderPlan.from_dict(payload).assignments] == [
        a.avg_route_size for a in plan.assignments
    ]
