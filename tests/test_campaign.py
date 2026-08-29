"""Descriptors, selection and city profiling.

The descriptors decide which instances a campaign keeps, so "it returns a
number" is not evidence that they work. Every test here builds a point set or a
demand field whose answer is known from the definition -- a lattice, three tight
blobs, a constant demand vector, a demand field aligned with position -- and
checks the descriptor recovers it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mamut_routing_tools.campaign import (
    POI_SIZE_BUCKETS,
    SELECTION_FEATURES,
    CandidateSpec,
    EvaluatedCandidate,
    InstanceDescriptors,
    Quota,
    QuotaInfeasibleError,
    assign_strata,
    compute_descriptors,
    compute_divergence,
    diversity_report,
    enumerate_candidates,
    profile_city,
    resolve_cities,
    select_maxmin,
    poi_size_bucket,
    poi_tier_quotas,
    size_bucket,
)
from mamut_routing_tools.campaign.design import (
    AVG_ROUTE_SIZES,
    CAMPAIGN_MIN_ROUTES,
    DEMAND_TYPES,
    DEPOT_MODES,
    METHODS,
    _admissible_band,
)
from mamut_routing_tools.campaign.ladder import admissible_route_size_bands

# --------------------------------------------------------------------------
# descriptors
# --------------------------------------------------------------------------


def _lattice(side: int = 10, step: float = 100.0) -> list[list[float]]:
    """A regular grid of customers with the depot at its centre."""
    centre = (side - 1) * step / 2
    return [[centre, centre]] + [
        [i * step, j * step] for i in range(side) for j in range(side)
    ]


def test_a_lattice_reads_as_regular_and_evenly_spaced() -> None:
    points = _lattice()
    descriptors = compute_descriptors(points, [0] + [1] * 100, capacity=10)

    # Clark-Evans on a square lattice is 2 by definition (mean NND = step, the
    # CSR expectation is step/2 at this density); edge effects nudge it up.
    assert descriptors.clark_evans_r == pytest.approx(2.2, abs=0.2)
    assert descriptors.nnd_cv == pytest.approx(0.0, abs=1e-9)
    # depot at the centroid: no offset, and no node is more central than it
    assert descriptors.depot_centrality == pytest.approx(0.0, abs=1e-9)
    assert descriptors.depot_eccentricity == pytest.approx(0.0, abs=1e-9)


def test_tight_clusters_read_as_clustered() -> None:
    rng = np.random.default_rng(0)
    blobs = np.vstack(
        [rng.normal(centre, 30.0, size=(40, 2)) for centre in ([0, 0], [3000, 0], [0, 3000])]
    )
    descriptors = compute_descriptors(
        [[1000.0, 1000.0]] + blobs.tolist(), [0] + [1] * 120, capacity=12
    )
    assert descriptors.clark_evans_r < 0.5, "three tight blobs must read as clustered"
    assert descriptors.nnd_cv > 0.3


def test_uniform_demands_have_no_variation_and_no_autocorrelation() -> None:
    descriptors = compute_descriptors(_lattice(), [0] + [1] * 100, capacity=10)
    assert descriptors.demand_cv == pytest.approx(0.0, abs=1e-12)
    assert descriptors.demand_gini == pytest.approx(0.0, abs=1e-12)
    assert descriptors.demand_max_over_mean == pytest.approx(1.0)
    # A constant field has no variance to correlate: undefined, not zero.
    assert np.isnan(descriptors.demand_moran_i)


def test_spatially_correlated_demand_shows_up_as_positive_morans_i() -> None:
    """The ``demand_type=6`` shape: big demands on one diagonal, small on the other."""
    rng = np.random.default_rng(1)
    customers = rng.uniform(0.0, 1000.0, size=(200, 2))
    diagonal = [90 if (x < 500) == (y < 500) else 10 for x, y in customers]
    independent = [int(v) for v in rng.integers(1, 100, size=200)]

    correlated = compute_descriptors(
        [[500.0, 500.0]] + customers.tolist(), [0] + diagonal, capacity=400
    )
    scattered = compute_descriptors(
        [[500.0, 500.0]] + customers.tolist(), [0] + independent, capacity=400
    )
    assert correlated.demand_moran_i > 0.5
    assert abs(scattered.demand_moran_i) < 0.15


def test_capacity_block_matches_its_definitions() -> None:
    # 100 customers, demand 1 each, capacity 8 -> 13 routes, 7 slack units
    descriptors = compute_descriptors(_lattice(), [0] + [1] * 100, capacity=8)
    assert descriptors.lb_cap == 13
    assert descriptors.route_size == pytest.approx(100 / 13)
    assert descriptors.capacity_slack == pytest.approx(1 - 100 / (13 * 8))


def test_a_remote_depot_is_eccentric_and_off_centre() -> None:
    points = _lattice()
    points[0] = [-5000.0, -5000.0]
    descriptors = compute_descriptors(points, [0] + [1] * 100, capacity=10)
    assert descriptors.depot_centrality > 5.0
    assert descriptors.depot_eccentricity == pytest.approx(1.0)


def test_collinear_customers_leave_the_shape_features_undefined() -> None:
    points = [[0.0, 0.0]] + [[float(i), 0.0] for i in range(1, 21)]
    descriptors = compute_descriptors(points, [0] + [1] * 20, capacity=4)
    assert np.isnan(descriptors.clark_evans_r), "no area means no density to compare against"
    assert descriptors.extent_m == 0.0
    # the demand and capacity blocks are unaffected
    assert descriptors.lb_cap == 5


@pytest.mark.parametrize(
    ("coordinates", "demands", "capacity", "message"),
    [
        ([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], [0, 1], 5, "same length"),
        ([[0.0, 0.0], [1.0, 1.0]], [0, 1], 5, "at least 2 customers"),
        ([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], [0, 1, 1], 0, "capacity must be positive"),
        ([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], [0, 1, -1], 5, "non-negative"),
    ],
)
def test_compute_descriptors_rejects_malformed_input(coordinates, demands, capacity, message) -> None:
    with pytest.raises(ValueError, match=message):
        compute_descriptors(coordinates, demands, capacity)


# --------------------------------------------------------------------------
# metric divergence
# --------------------------------------------------------------------------


def test_identical_matrices_diverge_by_nothing() -> None:
    matrix = [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]
    divergence = compute_divergence(matrix, matrix, matrix)
    assert divergence.detour_mean == pytest.approx(1.0)
    assert divergence.detour_p90 == pytest.approx(1.0)
    assert divergence.rank_tau_eucl_short == pytest.approx(1.0)
    assert divergence.rank_tau_eucl_fast == pytest.approx(1.0)
    assert divergence.asymmetry_shortest == pytest.approx(0.0)


def test_a_uniformly_longer_road_metric_is_pure_detour_without_rank_change() -> None:
    euclidean = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    divergence = compute_divergence(euclidean, euclidean * 1.4, euclidean)
    assert divergence.detour_mean == pytest.approx(1.4)
    # scaling every arc by the same factor cannot reorder anything
    assert divergence.rank_tau_eucl_short == pytest.approx(1.0)


def test_one_way_streets_show_up_as_asymmetry() -> None:
    symmetric = np.array([[0.0, 10.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 0.0]])
    oneway = symmetric.copy()
    oneway[0, 1] = 30.0  # the return leg is three times the outbound
    divergence = compute_divergence(symmetric, oneway, symmetric)
    assert divergence.asymmetry_shortest > 0.0
    assert divergence.asymmetry_fastest == pytest.approx(0.0)


def test_compute_divergence_rejects_mismatched_matrices() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compute_divergence([[0.0, 1.0], [1.0, 0.0]], [[0.0]], [[0.0]])


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def _candidate(city: str, x: float, y: float, *, demand_type: int = 1, n: int = 100, stratum: str = "mid"):
    """A candidate placed at (x, y) in the first two selection features."""
    values = dict.fromkeys(SELECTION_FEATURES, 0.0)
    values[SELECTION_FEATURES[0]] = x
    values[SELECTION_FEATURES[1]] = y
    descriptors = InstanceDescriptors(n=n, extent_m=1000.0, lb_cap=10, **values)
    spec = CandidateSpec(
        city=city,
        osm_path=Path("nowhere.osm"),
        n=n,
        method="hybrid",
        demand_type=demand_type,
        avg_route_size=3,
        depot_mode="center",
        customer_mode="random",
        cluster_seeds=4,
        cluster_decay_meters=800.0,
        hybrid_poi_share=0.5,
        distortion_stratum=stratum,
    )
    return EvaluatedCandidate(spec=spec, descriptors=descriptors, resolved_n=n)


def _grid_pool(side: int = 7):
    return [
        _candidate(f"c{index}", float(x), float(y), demand_type=(index % 7) + 1)
        for index, (x, y) in enumerate(
            (x, y) for x in np.linspace(0, 1, side) for y in np.linspace(0, 1, side)
        )
    ]


def test_max_min_takes_the_extremes_first() -> None:
    selected = select_maxmin(_grid_pool(), 4)
    corners = {
        (round(c.descriptors.clark_evans_r, 6), round(c.descriptors.nnd_cv, 6)) for c in selected
    }
    assert corners == {(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)}


def test_selection_is_deterministic() -> None:
    pool = _grid_pool()
    first = [c.spec.city for c in select_maxmin(pool, 8)]
    assert first == [c.spec.city for c in select_maxmin(pool, 8)]


def test_a_quota_that_fights_max_min_still_wins() -> None:
    """Every demand type must appear even though max-min would take corners."""
    quota = Quota(
        name="demand_type",
        key=lambda c: c.spec.demand_type,
        minimum={value: 1 for value in DEMAND_TYPES},
    )
    selected = select_maxmin(_grid_pool(), 7, [quota])
    assert sorted(c.spec.demand_type for c in selected) == list(DEMAND_TYPES)


def test_a_per_level_cap_is_respected() -> None:
    pool = [_candidate("lyon", 0.0, 0.0), _candidate("lyon", 1.0, 1.0), _candidate("brest", 0.5, 0.5)]
    selected = select_maxmin(pool, 2, [Quota(name="city", key=lambda c: c.spec.city, per_level_maximum=1)])
    assert sorted(c.spec.city for c in selected) == ["brest", "lyon"]


def test_an_unsatisfiable_quota_is_reported_not_silently_violated() -> None:
    quota = Quota(
        name="demand_type",
        key=lambda c: c.spec.demand_type,
        minimum={value: 2 for value in DEMAND_TYPES},
    )
    with pytest.raises(QuotaInfeasibleError, match="demand_type"):
        select_maxmin(_grid_pool(), 5, [quota])


def test_a_budget_larger_than_the_pool_is_rejected() -> None:
    with pytest.raises(QuotaInfeasibleError, match="fewer than the requested"):
        select_maxmin(_grid_pool(side=2), 100)


def test_diversity_report_states_coverage_spread_and_confounding() -> None:
    pool = _grid_pool()
    quota = Quota(
        name="demand_type",
        key=lambda c: c.spec.demand_type,
        minimum={value: 1 for value in DEMAND_TYPES},
    )
    selected = select_maxmin(pool, 7, [quota])
    report = diversity_report(selected, pool, [quota])

    assert report["selected"] == 7
    assert report["pool"] == len(pool)
    assert report["unmet_quotas"] == {}
    assert set(report["coverage"]["demand_type"]) == {str(v) for v in DEMAND_TYPES}
    assert set(report["features"]) == set(SELECTION_FEATURES)
    assert report["min_pairwise_distance"] > 0.0
    # constant features have no correlation to report, and JSON has no NaN
    assert report["feature_correlation"]["matrix"][-1][-1] is None


# --------------------------------------------------------------------------
# design enumeration
# --------------------------------------------------------------------------


def test_enumerate_candidates_balances_every_axis() -> None:
    cities = [(f"city{i}", Path(f"city{i}.osm")) for i in range(12)]
    specs = enumerate_candidates(cities, per_city=7)

    assert len(specs) == 12 * 7
    assert {spec.city for spec in specs} == {city for city, _ in cities}
    for axis, levels in (
        ("demand_type", DEMAND_TYPES),
        ("depot_mode", DEPOT_MODES),
        ("method", METHODS),
    ):
        counts = [sum(getattr(spec, axis) == level for spec in specs) for level in levels]
        assert set(counts) <= {min(counts), min(counts) + 1}, f"{axis} is unbalanced: {counts}"

    # ``avg_route_size`` is deliberately absent from that loop. Its column is
    # drawn balanced like the others and then repaired against each candidate's
    # ``n``, because the long bands are not legal at the small sizes -- so the
    # realized counts lean short, and must.
    assert all(
        spec.avg_route_size in admissible_route_size_bands(spec.n) for spec in specs
    )


def test_a_long_route_band_is_repaired_down_at_a_size_that_cannot_carry_it() -> None:
    """The v2 defect, at the level of a single candidate.

    A 50-200 customer route target is an ordinary instance at n = 4000 and a
    two-route degenerate at n = 100. The band is drawn without reference to size,
    so something has to reconcile them; before, nothing did.
    """
    assert _admissible_band(7, 4000) == 7
    assert _admissible_band(7, 100) == 4
    # Bands the size can carry are passed through untouched.
    assert [_admissible_band(band, 4000) for band in AVG_ROUTE_SIZES] == list(AVG_ROUTE_SIZES)


def test_a_single_method_narrows_the_sourcing_axis() -> None:
    """The POI-only tier is not "mostly POI"; it is POI or nothing."""
    cities = [(f"city{i}", Path(f"city{i}.osm")) for i in range(5)]
    specs = enumerate_candidates(cities, per_city=3, methods=("poi_categories",))

    assert len(specs) == 15
    assert {spec.method for spec in specs} == {"poi_categories"}
    # Narrowing one axis must not disturb the balance of the others.
    counts = [sum(spec.demand_type == level for spec in specs) for level in DEMAND_TYPES]
    assert set(counts) <= {min(counts), min(counts) + 1}


def test_enumerate_candidates_rejects_an_empty_method_list() -> None:
    with pytest.raises(ValueError, match="methods must not be empty"):
        enumerate_candidates([("lyon", Path("lyon.osm"))], per_city=1, methods=())


def test_a_size_ceiling_shrinks_an_over_large_draw_to_the_next_grid_size() -> None:
    """A city cannot be asked for more customers than it can serve.

    The size grid is drawn independently of the city, so a POI-only tier will
    propose n = 5000 in a town with 1500 attachable amenities. The ceiling
    rewrites that to the largest grid size that fits rather than letting the
    evaluation discover the shortfall after paying for it.
    """
    cities = [("big", Path("big.osm")), ("small", Path("small.osm"))]
    grid = (1000, 2000, 5000)
    specs = enumerate_candidates(
        cities, per_city=9, size_grid=grid, size_ceiling={"big": 5000, "small": 2000}
    )

    assert {spec.n for spec in specs if spec.city == "small"} <= {1000, 2000}
    assert max(spec.n for spec in specs if spec.city == "big") == 5000


def test_a_city_under_the_smallest_size_is_dropped_entirely() -> None:
    cities = [("big", Path("big.osm")), ("tiny", Path("tiny.osm"))]
    specs = enumerate_candidates(
        cities, per_city=4, size_grid=(1000, 2000), size_ceiling={"big": 2000, "tiny": 300}
    )
    assert {spec.city for spec in specs} == {"big"}


def test_enumeration_is_reproducible_and_seed_sensitive() -> None:
    cities = [(f"city{i}", Path(f"city{i}.osm")) for i in range(6)]
    same = enumerate_candidates(cities, per_city=4, base_seed=3)
    assert [s.to_dict() for s in same] == [
        s.to_dict() for s in enumerate_candidates(cities, per_city=4, base_seed=3)
    ]
    other = enumerate_candidates(cities, per_city=4, base_seed=4)
    assert [s.to_dict() for s in same] != [s.to_dict() for s in other]


def test_candidate_seeds_follow_the_parameters_not_the_position() -> None:
    cities = [("lyon", Path("lyon.osm"))]
    (spec,) = enumerate_candidates(cities, per_city=1)
    from dataclasses import replace

    assert replace(spec, city="lyon").seed == spec.seed
    assert replace(spec, n=spec.n + 1).seed != spec.seed


def test_candidate_spec_round_trips_through_json_shape() -> None:
    (spec,) = enumerate_candidates([("lyon", Path("lyon.osm"))], per_city=1)
    assert CandidateSpec.from_dict(spec.to_dict()) == spec


@pytest.mark.parametrize(("n", "expected"), [(100, "100-200"), (250, "201-400"), (1000, "701-1000"), (2000, ">1000")])
def test_size_buckets(n: int, expected: str) -> None:
    assert size_bucket(n) == expected


@pytest.mark.parametrize(
    ("n", "expected"),
    [(1000, "1000-1500"), (1500, "1000-1500"), (2000, "1501-2500"), (5000, "2501-5000"),
     (900, "<1000"), (6000, ">5000")],
)
def test_poi_size_buckets(n: int, expected: str) -> None:
    assert poi_size_bucket(n) == expected


def test_the_poi_tier_quotas_on_size_but_not_on_sourcing() -> None:
    """Sourcing is fixed by construction there, so a quota on it says nothing."""
    names = [quota.name for quota in poi_tier_quotas(10)]
    assert names == ["city", "distortion_stratum", "poi_size_bucket", "avg_route_size"]

    size_quota = next(q for q in poi_tier_quotas(10) if q.name == "poi_size_bucket")
    assert sum(size_quota.minimum.values()) <= 10
    assert set(size_quota.minimum) == {name for name, _, _ in POI_SIZE_BUCKETS}


def test_the_poi_tier_caps_route_size_without_flooring_it() -> None:
    """A ceiling guides; a floor over seven levels and ten slots would dictate.

    Without the cap, v2 put five of these ten instances in the longest band and
    published one at 154 customers per route.
    """
    band_quota = next(q for q in poi_tier_quotas(10) if q.name == "avg_route_size")
    assert not band_quota.minimum
    assert band_quota.per_level_maximum == 2
    # It scales with the budget rather than staying at 2 forever.
    assert next(
        q for q in poi_tier_quotas(70) if q.name == "avg_route_size"
    ).per_level_maximum == 16


def test_a_poi_tier_too_small_to_span_its_sizes_drops_the_size_floor() -> None:
    """Four slots cannot honour three size floors and still choose anything."""
    size_quota = next(q for q in poi_tier_quotas(4) if q.name == "poi_size_bucket")
    assert size_quota.minimum == {}


def test_resolve_cities_slugs_extract_names(tmp_path: Path) -> None:
    for name in ("Aix-en-Provence", "New York", "Lyon"):
        (tmp_path / f"{name}.osm").write_text("", encoding="utf-8")
    assert resolve_cities(tmp_path) == [
        ("aix_en_provence", tmp_path / "Aix-en-Provence.osm"),
        ("lyon", tmp_path / "Lyon.osm"),
        ("new_york", tmp_path / "New York.osm"),
    ]
    assert [slug for slug, _ in resolve_cities(tmp_path, ["Lyon"])] == ["lyon"]
    with pytest.raises(FileNotFoundError, match="berlin"):
        resolve_cities(tmp_path, ["Berlin"])


# --------------------------------------------------------------------------
# city profiling
# --------------------------------------------------------------------------


def test_profile_city_measures_detour_on_the_fixture_city(fixture_osm_path: Path) -> None:
    profile = profile_city(fixture_osm_path, "testville", num_sources=4, num_targets=4)
    assert profile.city == "testville"
    assert profile.num_vertices > 0
    assert profile.num_pairs > 0
    # A road always takes at least the straight line, so a detour below 1 would
    # mean the graph and the coordinates disagree.
    assert profile.detour_mean >= 1.0
    assert profile.detour_p90 >= profile.detour_mean
    assert profile.distortion_stratum is None, "strata need the whole city set"


def test_assign_strata_splits_the_set_into_terciles(fixture_osm_path: Path) -> None:
    base = profile_city(fixture_osm_path, "testville", num_sources=4, num_targets=4)
    from dataclasses import replace

    profiles = [replace(base, city=f"c{i}", detour_mean=1.0 + i / 10) for i in range(9)]
    labelled = assign_strata(profiles)
    assert [p.distortion_stratum for p in labelled] == [
        "low", "low", "low", "mid", "mid", "mid", "high", "high", "high"
    ]


# --------------------------------------------------------------------------
# campaign policy: sizes are promises, quotas must be satisfiable
# --------------------------------------------------------------------------


def test_a_candidate_served_short_of_its_requested_size_is_not_usable() -> None:
    """Sparse-amenity cities serve POI requests short; those are not candidates.

    The generator only warns when a pool cannot fill the request. Keeping such a
    candidate would file an instance selected for one size bucket under another.
    """
    full = _candidate("lyon", 0.0, 0.0, n=500)
    assert full.is_usable()

    from dataclasses import replace as _replace

    short = _replace(full, resolved_n=198)
    assert not short.is_usable()


def test_a_candidate_that_barely_needs_a_fleet_is_not_usable() -> None:
    """The gate that v2 set two orders of magnitude too low.

    It read ``lb_cap >= 2`` -- "not a TSP" -- and 22 of the 110 published bases
    cleared it while still being trivially partitioned, 19 into exactly two
    routes. Two is not a fleet.
    """
    from dataclasses import replace as _replace

    candidate = _candidate("lyon", 0.0, 0.0, n=500)
    assert candidate.descriptors.lb_cap >= CAMPAIGN_MIN_ROUTES
    assert candidate.is_usable()

    for lb in range(1, CAMPAIGN_MIN_ROUTES):
        degenerate = _replace(
            candidate, descriptors=_replace(candidate.descriptors, lb_cap=lb)
        )
        assert not degenerate.is_usable(), f"LB_cap={lb} must not be publishable"


def test_route_size_is_spread_on_a_log_scale() -> None:
    """Ratios, not differences -- which is what a route size is.

    Linearly, 50 to 200 customers per route is eleven times the distance from 3
    to 16, so a max-min selector buys most of its spread at the long end and
    parks there. In v2 it put 39 of 100 instances in the longest band against a
    floor of 7. In logs, a factor of four is a factor of four wherever it sits.
    """
    from mamut_routing_tools.campaign.descriptors import selection_features

    index = SELECTION_FEATURES.index("route_size")

    def at(route_size: float) -> float:
        values = dict.fromkeys(SELECTION_FEATURES, 1.0)
        values["route_size"] = route_size
        return selection_features(
            InstanceDescriptors(n=100, extent_m=1000.0, lb_cap=10, **values)
        )[index]

    # 4 is the geometric midpoint of 2 and 8, so it is equidistant from both.
    assert at(4.0) - at(2.0) == pytest.approx(at(8.0) - at(4.0))
    # and the long end no longer dominates the axis
    assert at(200.0) - at(50.0) == pytest.approx(at(12.0) - at(3.0))


def test_quota_minimums_can_always_be_met_within_the_budget() -> None:
    """Every tier quota must be arithmetically satisfiable, at any budget.

    A 5-instance pilot cannot cover 7 demand types; the quota has to ask for
    nothing rather than for something impossible.
    """
    from mamut_routing_tools.campaign import main_tier_quotas

    for k in (5, 7, 12, 50, 100, 240):
        for quota in main_tier_quotas(k):
            assert sum(dict(quota.minimum).values()) <= k, f"{quota.name} over-subscribes k={k}"


def test_the_main_tier_quotas_bite_at_full_size() -> None:
    """...and they must still be real requirements at the intended budget."""
    from mamut_routing_tools.campaign import main_tier_quotas

    by_name = {quota.name: quota for quota in main_tier_quotas(100)}
    assert min(by_name["demand_type"].minimum.values()) >= 5
    assert min(by_name["distortion_stratum"].minimum.values()) >= 20
    assert by_name["city"].per_level_maximum == 1


def test_campaign_candidates_use_a_workable_poi_sourcing_policy() -> None:
    """Wide enough to be usable, narrow enough to be delivery destinations.

    Two departures from the generator's defaults, in opposite directions. The
    default *binding* rule (nearest road node must itself be a graph vertex)
    discards most of a city's amenities, so the campaign snaps to the nearest
    routable point instead. The default *category* list is the first seven,
    which is far too narrow -- but all forty-nine is far too wide, because most
    of what OSM tags ``amenity`` is street furniture.
    """
    from mamut_routing_tools.campaign.design import (
        CAMPAIGN_POI_ATTACH_MODE,
        CAMPAIGN_POI_CATEGORIES,
        EXCLUDED_POI_CATEGORIES,
    )
    from mamut_routing_tools.generation.pois import DEFAULT_CATEGORIES, POI_CATEGORIES
    from mamut_routing_tools.generation.select import POI_ATTACH_NEAREST_VERTEX

    assert CAMPAIGN_POI_ATTACH_MODE == POI_ATTACH_NEAREST_VERTEX
    assert len(CAMPAIGN_POI_CATEGORIES) > len(DEFAULT_CATEGORIES)
    assert set(CAMPAIGN_POI_CATEGORIES) < set(POI_CATEGORIES), (
        "the campaign list must stay a subset, or load_poi_catalog -- which "
        "parses with POI_CATEGORIES -- will silently return nothing for the "
        "categories it does not know"
    )
    # Keep and drop must partition the platform's vocabulary, so a category
    # added upstream cannot fall through unreviewed.
    assert set(CAMPAIGN_POI_CATEGORIES) | set(EXCLUDED_POI_CATEGORIES) == set(POI_CATEGORIES)
    assert not set(CAMPAIGN_POI_CATEGORIES) & set(EXCLUDED_POI_CATEGORIES)

    (spec,) = enumerate_candidates([("lyon", Path("lyon.osm"))], per_city=1)
    request = spec.to_request()
    assert request.poi_attach_mode == POI_ATTACH_NEAREST_VERTEX
    assert set(request.categories) == set(CAMPAIGN_POI_CATEGORIES)


def test_street_furniture_is_not_a_delivery_destination() -> None:
    """Measured on the extracts, bench + waste_basket + recycling alone are
    69.9 % of Lyon's amenities and 41.5 % of Quimper's. Drawing customers from
    that pool makes "the customers are real places" a claim, not a fact."""
    from mamut_routing_tools.campaign.design import CAMPAIGN_POI_CATEGORIES

    for junk in ("bench", "waste_basket", "recycling", "drinking_water",
                 "toilets", "shower", "shelter"):
        assert junk not in CAMPAIGN_POI_CATEGORIES

    # amenity=parking is usually one individual space rather than a car park,
    # and an ATM or a charging point is a machine, not premises with a door.
    for not_premises in ("parking", "atm", "charging_station", "bicycle_rental",
                         "taxi", "bus_station", "ferry_terminal"):
        assert not_premises not in CAMPAIGN_POI_CATEGORIES

    for real in ("restaurant", "pharmacy", "school", "bank", "post_office", "fuel"):
        assert real in CAMPAIGN_POI_CATEGORIES


def test_the_category_list_is_part_of_what_an_instance_is() -> None:
    """Left out of the seed, curating the list would silently repopulate every
    instance while keeping its published name and its recorded seed."""
    (spec,) = enumerate_candidates([("lyon", Path("lyon.osm"))], per_city=1)
    from dataclasses import replace

    narrowed = replace(spec, categories=("restaurant", "cafe"))
    assert narrowed.seed != spec.seed


def test_interacting_quotas_are_detected_before_any_work() -> None:
    """A cap on one axis can make a minimum on another unreachable.

    One instance per city, plus "every stratum represented", is unsatisfiable
    when no candidate exists in one stratum -- and it must be reported up front,
    naming the level that ran out, rather than deadlocking on the last pick.
    """
    pool = [
        _candidate("lyon", 0.0, 0.0, stratum="low"),
        _candidate("brest", 1.0, 0.0, stratum="low"),
        _candidate("nantes", 0.0, 1.0, stratum="high"),
    ]
    quotas = [
        Quota(name="city", key=lambda c: c.spec.city, per_level_maximum=1),
        Quota(
            name="stratum",
            key=lambda c: c.spec.distortion_stratum,
            minimum={"low": 1, "mid": 1, "high": 1},
        ),
    ]
    with pytest.raises(QuotaInfeasibleError, match="mid"):
        select_maxmin(pool, 3, quotas)


def test_the_greedy_does_not_paint_itself_into_a_corner() -> None:
    """A quota that only one late candidate can satisfy must still be met.

    Max-min would take the two extremes first and strand the requirement; the
    lookahead has to keep a slot for the only "mid" candidate, which happens to
    sit in the middle of the cloud where max-min never looks.
    """
    pool = [
        _candidate("a", 0.0, 0.0, stratum="low"),
        _candidate("b", 1.0, 1.0, stratum="low"),
        _candidate("c", 1.0, 0.0, stratum="low"),
        _candidate("d", 0.5, 0.5, stratum="mid"),
    ]
    quotas = [
        Quota(name="city", key=lambda c: c.spec.city, per_level_maximum=1),
        Quota(name="stratum", key=lambda c: c.spec.distortion_stratum, minimum={"mid": 1}),
    ]
    selected = select_maxmin(pool, 3, quotas)
    assert "mid" in {c.spec.distortion_stratum for c in selected}
    assert len({c.spec.city for c in selected}) == 3


def test_a_budget_too_small_for_an_axis_is_not_quota_d_at_all() -> None:
    """A floor of one-per-level is a grid, not a floor.

    Three levels over four slots pins almost the whole selection, and three such
    axes at once pin it completely -- leaving the max-min pass, the entire reason
    for selecting rather than enumerating, nothing to choose. Below the share
    threshold an axis carries no minimum and the selection runs on spread alone.
    """
    from mamut_routing_tools.campaign import main_tier_quotas
    from mamut_routing_tools.campaign.quotas import MIN_SHARE_FOR_QUOTA

    tiny = {quota.name: dict(quota.minimum) for quota in main_tier_quotas(4)}
    assert all(not minimum for minimum in tiny.values()), tiny

    full = {quota.name: dict(quota.minimum) for quota in main_tier_quotas(100)}
    assert min(full["distortion_stratum"].values()) >= 20
    assert min(full["demand_type"].values()) >= 5
    assert MIN_SHARE_FOR_QUOTA >= 2.0


def test_exhausted_per_level_caps_are_reported_as_such() -> None:
    """Running out of cities is a different failure from missing a level."""
    pool = [_candidate("lyon", 0.0, 0.0), _candidate("lyon", 1.0, 1.0)]
    with pytest.raises(QuotaInfeasibleError, match="per-level cap"):
        select_maxmin(pool, 2, [Quota(name="city", key=lambda c: c.spec.city, per_level_maximum=1)])


def test_every_demand_distribution_survives_the_feature_space() -> None:
    """A descriptor that is undefined must not delete a whole design axis.

    ``demand_type=1`` gives every customer demand 1, so Moran's I of the demand
    field has no variance to correlate and is formally undefined. Treating that
    NaN as "unusable" silently removed one of the seven demand distributions
    from an entire campaign, which is how this test came to exist.
    """
    from mamut_routing_tools.campaign.descriptors import selection_features

    points = _lattice()
    for demand_type, demands in (
        (1, [1] * 100),          # constant: Moran's I undefined
        (5, list(range(50, 150))),
    ):
        descriptors = compute_descriptors(points, [0] + demands, capacity=40)
        vector = selection_features(descriptors)
        assert all(np.isfinite(v) for v in vector), f"demand_type={demand_type} -> {vector}"

    # ...and the descriptor itself stays honest about being undefined
    constant = compute_descriptors(points, [0] + [1] * 100, capacity=40)
    assert np.isnan(constant.demand_moran_i)
    assert dict(zip(SELECTION_FEATURES, selection_features(constant)))["demand_moran_i"] == 0.0


def test_a_degenerate_geometry_is_still_rejected() -> None:
    """The imputation is narrow: an unusable *shape* must stay unusable."""
    from mamut_routing_tools.campaign.descriptors import selection_features

    collinear = [[0.0, 0.0]] + [[float(i), 0.0] for i in range(1, 21)]
    descriptors = compute_descriptors(collinear, [0] + [1] * 20, capacity=4)
    assert not all(np.isfinite(v) for v in selection_features(descriptors))
