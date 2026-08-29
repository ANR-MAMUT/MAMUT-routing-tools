"""The candidate pool: what to consider generating, and what each one would be.

A campaign that wants measured diversity cannot pick parameters and hope. It
enumerates far more candidates than it needs, works out what each one would
*look like* without paying for it, and lets :mod:`~.select` keep the spread.

:class:`CandidateSpec` is a complete, deterministic recipe for one instance --
everything needed to rebuild it byte for byte later, which is what makes the
"propose" and "generate" phases separable.

:func:`evaluate_candidates` is the cheap half of generation: it builds the
customer selection and draws the demands exactly as the real generator will,
then stops. No distance matrix is computed, which is the whole point -- the
Dijkstra passes are what cost hours, and they are only spent on winners.
"""

from __future__ import annotations

import random
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from mamut_routing_tools.campaign.city_profile import CityProfile
from mamut_routing_tools.campaign.descriptors import (
    SELECTION_FEATURES,
    InstanceDescriptors,
    compute_descriptors,
    selection_features,
)
from mamut_routing_tools.family.naming import METHOD_TAGS
from mamut_routing_tools.generation.pois import POI_CATEGORIES
from mamut_routing_tools.generation.select import POI_ATTACH_NEAREST_VERTEX
from mamut_routing_tools.generation.writers import slugify
from mamut_routing_tools.generation.demands import (
    capacity_from_avg_route_size,
    generate_demands,
)
from mamut_routing_tools.generation.pois import is_poi_source_tag
from mamut_routing_tools.generation.single import GenerationRequest, build_generation_selection

# --- the design axes -------------------------------------------------------
# Uchoa et al.'s X-instance vocabulary, which the generator already implements:
# seven demand distributions, seven average-route-size bands, three depot
# placements, three customer placements. Plus this project's own axes: how
# customers are sourced (real amenities, synthetic points, or a mix) and how
# tight the clustering is.

DEMAND_TYPES: tuple[int, ...] = tuple(range(1, 8))
AVG_ROUTE_SIZES: tuple[int, ...] = tuple(range(1, 8))
DEPOT_MODES: tuple[str, ...] = ("random", "center", "corner")
CUSTOMER_MODES: tuple[str, ...] = ("random", "clustered", "random_clustered")
METHODS: tuple[str, ...] = ("poi_categories", "parametric_attach", "hybrid")
CLUSTER_SEEDS: tuple[int, ...] = (2, 4, 8)
CLUSTER_DECAY_METERS: tuple[float, ...] = (400.0, 800.0, 1600.0)
HYBRID_POI_SHARES: tuple[float, ...] = (0.3, 0.5, 0.7)

#: Amenities that are *premises a vehicle serves*, and nothing else.
#:
#: The platform's ``POI_CATEGORIES`` is its full vocabulary and stays as it is --
#: the workbench, the OSM fetcher and the extract auditor all key on it. But a
#: routing benchmark cannot draw customers from all of it, because most of what
#: OSM tags ``amenity`` is street furniture. Measured on the fetched extracts:
#:
#:     Lyon     32 949 amenities, of which bench + waste_basket + recycling
#:              alone are 69.9 %; with parking, drinking_water, toilets, atm,
#:              charging_station, bicycle_rental, taxi and shelter, ~78 %.
#:     Quimper   1 091 amenities, the same three are 41.5 %.
#:
#: Sampling from that pool produces instances whose customers are park benches
#: and bins, which makes "the customers are real places on a real map" a claim
#: rather than a fact. ``amenity=parking`` is the subtle one: in most cities it
#: marks an individual parking *space*, not a car park, so it both floods the
#: pool and misrepresents what it floods it with.
#:
#: Curating costs nothing at runtime: this is a strict subset of
#: ``POI_CATEGORIES``, the catalog is parsed once with the full list, and every
#: consumer filters it afterwards.
CAMPAIGN_POI_CATEGORIES: tuple[str, ...] = (
    # food and drink
    "restaurant", "cafe", "bar", "fast_food", "pub", "biergarten",
    "ice_cream", "food_court", "nightclub",
    # health
    "pharmacy", "hospital", "clinic", "doctors", "dentist", "veterinary",
    # education
    "school", "university", "college", "kindergarten",
    # civic
    "post_office", "police", "fire_station", "townhall", "courthouse", "library",
    # culture
    "theatre", "cinema", "arts_centre", "community_centre", "museum",
    "place_of_worship",
    # commerce and vehicle services
    "bank", "marketplace", "fuel", "car_wash",
)

#: What the curation removes, kept explicit so the choice is reviewable and so a
#: test can assert none of it creeps back in.
EXCLUDED_POI_CATEGORIES: tuple[str, ...] = (
    # street furniture: not a destination
    "bench", "waste_basket", "recycling", "drinking_water", "toilets",
    "shower", "shelter",
    # real objects, but not premises with a delivery address
    "parking", "atm", "charging_station", "bicycle_rental", "taxi",
    "bus_station", "ferry_terminal",
)

#: How a POI binds to the road graph. ``nearest_node`` -- the generator default,
#: kept for byte-parity with the retired Julia pipeline -- demands that a POI's
#: nearest road *node* be a graph vertex in its own right, which is rarely true:
#: it discards ~80% of a city's amenities. ``nearest_vertex`` snaps to the
#: closest routable point within 50 m instead, which is what the customer
#: actually is.
CAMPAIGN_POI_ATTACH_MODE = POI_ATTACH_NEAREST_VERTEX

#: Smallest number of routes a published instance may require.
#:
#: This is the guard v2 did not have, and its absence is why 22 of that set's 110
#: bases were not vehicle routing problems at all -- 19 of them had best known
#: solutions with exactly two routes, five of those shaped like [147, 1].
#:
#: CVRPLIB gets this for free by pairing each route-size vocabulary with a size
#: range: XML100 draws r up to 50 at n = 100, and XL only introduces r up to 200
#: alongside n >= 1000. This repository ported XL's seven bands but kept the main
#: tier at n = 100..1000, so a 50-200 customer route target could land on a
#: hundred-customer instance. Nothing downstream noticed, because a two-route
#: instance is perfectly valid -- it is just not interesting.
#:
#: The rule is applied to the *band*, not to the realized draw: see
#: :func:`admissible_route_size_bands`.
CAMPAIGN_MIN_ROUTES = 6

#: Main tier sizes. Spread over the range the study cares about, with the small
#: end at 100 because below that a modern solver closes the gap instantly and
#: the anytime curve carries no signal.
MAIN_SIZE_GRID: tuple[int, ...] = (100, 150, 200, 300, 400, 500, 700, 1000)
#: Large tier. Kept short on purpose: sidecar bytes grow with n^2 and the
#: pinned-path Dijkstra pass grows faster still.
LARGE_SIZE_GRID: tuple[int, ...] = (2000, 5000)
#: POI-only tier. Every customer is a real amenity here, so the grid has to
#: be finer than the large tier's: a city's capacity is whatever its amenity
#: map happens to hold, and a coarse grid would round most cities down hard.
#:
#: Starts *above* the main ladder's top rung rather than at it. A base is named
#: ``mamut-<city>-n<N>-<tag>``, so a POI tier instance that reused a main-tier
#: city at the same size would collide -- and ``build_base`` returns early on an
#: existing base, so the collision would look like a successful build while
#: silently republishing the main-tier instance.
POI_SIZE_GRID: tuple[int, ...] = (1200, 1500, 2000, 2500, 3000, 4000, 5000)


def stable_seed(*parts: Any) -> int:
    """A seed derived from what the instance *is*, never from a counter.

    Mirrors ``family.family._stable_seed``: re-running the campaign, or
    regenerating one instance on its own, must land on the same draw.
    """
    return zlib.crc32("|".join(str(part) for part in parts).encode("utf-8"))


@dataclass(frozen=True)
class CandidateSpec:
    """A complete, deterministic recipe for one base instance."""

    city: str
    osm_path: Path
    n: int
    method: str
    demand_type: int
    avg_route_size: int
    depot_mode: str
    customer_mode: str
    cluster_seeds: int
    cluster_decay_meters: float
    hybrid_poi_share: float
    poi_attach_mode: str = CAMPAIGN_POI_ATTACH_MODE
    categories: tuple[str, ...] = CAMPAIGN_POI_CATEGORIES
    #: The city's road-network distortion stratum, carried along so the
    #: selector can quota on it without a second lookup.
    distortion_stratum: str | None = None
    tier: str = "main"
    #: Which ladder position this candidate is competing for, when the campaign
    #: assigns sizes rather than drawing them. Empty for grid-drawn tiers.
    rung_key: str = ""

    @property
    def seed(self) -> int:
        return stable_seed(
            self.city,
            self.n,
            self.method,
            self.demand_type,
            self.avg_route_size,
            self.depot_mode,
            self.customer_mode,
            self.cluster_seeds,
            self.cluster_decay_meters,
            self.hybrid_poi_share,
            self.poi_attach_mode,
            # The category list is part of what the instance *is*: change it and
            # the same city, size and method draw a different customer set. Left
            # out of the seed, a curation would silently repopulate every
            # instance while keeping its published name and its recorded seed.
            ",".join(self.categories),
        )

    @property
    def method_tag(self) -> str:
        return METHOD_TAGS[self.method]

    def to_request(self) -> GenerationRequest:
        return GenerationRequest(
            city=self.city,
            osm_path=Path(self.osm_path),
            method=self.method,
            n_customers=self.n,
            seed=self.seed,
            demand_type=self.demand_type,
            avg_route_size=self.avg_route_size,
            depot_mode=self.depot_mode,
            customer_mode=self.customer_mode,
            cluster_seeds=self.cluster_seeds,
            cluster_decay_meters=self.cluster_decay_meters,
            hybrid_poi_share=self.hybrid_poi_share,
            poi_attach_mode=self.poi_attach_mode,
            categories=list(self.categories),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["osm_path"] = str(self.osm_path)
        payload["categories"] = list(self.categories)
        payload["seed"] = self.seed
        payload["method_tag"] = self.method_tag
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateSpec":
        names = {spec_field.name for spec_field in fields(cls)}
        values = {key: value for key, value in payload.items() if key in names}
        values["osm_path"] = Path(values["osm_path"])
        if "categories" in values:
            values["categories"] = tuple(values["categories"])
        return cls(**values)


@dataclass(frozen=True)
class EvaluatedCandidate:
    """A candidate together with what it would look like if generated."""

    spec: CandidateSpec
    descriptors: InstanceDescriptors
    #: customers the pool could actually serve, when short of the request
    resolved_n: int
    #: What the selection *is*, which is not always what was requested: a POI
    #: request an extract cannot fill is topped up with road points, and the
    #: generator reclassifies it as ``hybrid`` (or ``parametric_attach`` when no
    #: amenity could be used at all). The published name and the method quota
    #: both follow this, never ``spec.method`` -- otherwise a file called
    #: ``...-poi`` could hold customers sitting on plain intersections.
    effective_method: str = ""
    #: share of customers that are real amenities rather than road points
    poi_fraction: float = 0.0

    @property
    def method_tag(self) -> str:
        return METHOD_TAGS[self.effective_method or self.spec.method]

    def feature_vector(self) -> list[float]:
        return selection_features(self.descriptors)

    def is_usable(self) -> bool:
        """Servable at the requested size, a genuine VRP, and fully described.

        The size check is not pedantry. A city whose amenities are sparse serves
        a POI request short -- Nairobi's extract has 27 amenities in total -- and
        the generator only warns. Accepting the short instance would put a
        candidate selected for the 701-1000 bucket into the tree at n=198, which
        is exactly the kind of silent drift a designed benchmark must not have.

        Neither is the route-count check. This gate read ``lb_cap >= 2`` in v2 --
        "not a TSP" -- and 22 of the 110 published bases cleared it while still
        being trivially partitioned, 19 of them into exactly two routes. Two is
        not a fleet. The band admissibility rule in :mod:`~.ladder` is what makes
        the guarantee hold by construction; this is the backstop that catches the
        one case it cannot, where ``Q = ceil(r * mean demand)`` rounds up far
        enough to cost a route at the very edge of a band.
        """
        return (
            self.resolved_n >= self.spec.n
            and self.descriptors.lb_cap >= CAMPAIGN_MIN_ROUTES
            and all(isfinite(v) for v in self.feature_vector())
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "descriptors": self.descriptors.to_dict(),
            "resolved_n": self.resolved_n,
            "effective_method": self.effective_method,
            "method_tag": self.method_tag,
            "poi_fraction": self.poi_fraction,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluatedCandidate":
        return cls(
            spec=CandidateSpec.from_dict(payload["spec"]),
            descriptors=InstanceDescriptors(**payload["descriptors"]),
            resolved_n=int(payload["resolved_n"]),
            effective_method=str(payload.get("effective_method") or ""),
            poi_fraction=float(payload.get("poi_fraction", 0.0)),
        )


def _balanced_column(levels: Sequence[Any], length: int, rng: np.random.Generator) -> list[Any]:
    """``length`` draws in which every level appears within one of every other.

    Exactly balanced for any ``length``, not only for multiples of the level
    count.

    A randomized Latin design rather than independent draws: with 7 demand
    types over ~800 candidates, independent sampling would leave the rarest
    level 15% short of the commonest often enough to matter, and the pool has
    to *contain* a balanced design before the selector can honour a quota.
    """
    count = len(levels)
    base, extra = divmod(length, count)
    # Truncating a shuffled over-long list -- the obvious implementation -- is
    # only balanced when ``length`` divides evenly: otherwise the discarded tail
    # can take two copies of one level and none of another, and the column comes
    # out 9/11 rather than 10/11. The remainder is therefore handed to
    # ``extra`` randomly chosen levels instead of falling where it may.
    counts = [base] * count
    for position in rng.permutation(count)[:extra]:
        counts[int(position)] += 1

    column = [level for level, repeats in zip(levels, counts) for _ in range(repeats)]
    rng.shuffle(column)
    return column


def enumerate_for_rungs(
    assignments: Sequence[Any],
    *,
    osm_dir: str | Path = "osmdata",
    per_rung: int = 6,
    base_seed: int = 0,
    tier: str = "main",
    categories: Sequence[str] = CAMPAIGN_POI_CATEGORIES,
    poi_attach_mode: str = CAMPAIGN_POI_ATTACH_MODE,
) -> list[CandidateSpec]:
    """``per_rung`` candidates for each ladder position.

    The counterpart of :func:`enumerate_candidates` for a campaign that has
    already decided *what* each instance is -- city, size and sourcing method
    come from a :class:`~.ladder.RungAssignment` and are not up for negotiation
    here. What is still free is what the instance is *like*: how demand is
    distributed, how tightly the vehicles are loaded, where the depot sits and
    whether the customers cluster. Those are drawn as balanced columns across
    the whole pool exactly as before, so the selector still gets a spread to
    choose from on every axis it can still move.

    Splitting it this way is what stops the two decisions from fighting. Drawn
    together, a size the city could not serve was silently downgraded; drawn in
    order, the size is guaranteed feasible before any descriptor is computed.
    """
    if per_rung < 1:
        raise ValueError("per_rung must be >= 1")
    if not assignments:
        return []

    # A RungAssignment carries the extract's *file name*, not its path: capacity
    # measurements are shared across checkouts and an absolute path would not
    # survive the trip. The directory is supplied here instead.
    root = Path(osm_dir)
    total = per_rung * len(assignments)
    rng = np.random.default_rng(
        stable_seed("rungs", tier, base_seed, len(assignments), per_rung)
    )
    # ``avg_route_size`` is absent from this table on purpose. Which bands keep an
    # instance a genuine VRP depends on ``n``, so the band comes from the rung
    # alongside the city and the size; drawing it here as a balanced column over
    # all seven levels is precisely what put a 50-200 customer route target on a
    # hundred-customer instance in v2.
    columns = {
        "demand_type": _balanced_column(DEMAND_TYPES, total, rng),
        "depot_mode": _balanced_column(DEPOT_MODES, total, rng),
        "customer_mode": _balanced_column(CUSTOMER_MODES, total, rng),
        "cluster_seeds": _balanced_column(CLUSTER_SEEDS, total, rng),
        "cluster_decay_meters": _balanced_column(CLUSTER_DECAY_METERS, total, rng),
        "hybrid_poi_share": _balanced_column(HYBRID_POI_SHARES, total, rng),
    }

    specs: list[CandidateSpec] = []
    for index in range(total):
        rung = assignments[index // per_rung]
        specs.append(
            CandidateSpec(
                city=rung.city,
                osm_path=root / rung.osm_file,
                n=int(rung.n),
                method=str(rung.method),
                demand_type=int(columns["demand_type"][index]),
                avg_route_size=int(rung.avg_route_size),
                depot_mode=str(columns["depot_mode"][index]),
                customer_mode=str(columns["customer_mode"][index]),
                cluster_seeds=int(columns["cluster_seeds"][index]),
                cluster_decay_meters=float(columns["cluster_decay_meters"][index]),
                hybrid_poi_share=float(columns["hybrid_poi_share"][index]),
                poi_attach_mode=poi_attach_mode,
                categories=tuple(categories),
                distortion_stratum=rung.distortion_stratum,
                tier=tier,
                rung_key=rung.rung_key,
            )
        )
    return specs


def _admissible_band(band: int, n: int) -> int:
    """``band`` if an ``n``-customer instance can carry it, else the largest that can.

    Imported lazily: :mod:`~.ladder` imports this module for its constants, so
    reaching the other way at module scope would close the cycle.
    """
    from mamut_routing_tools.campaign.ladder import admissible_route_size_bands

    allowed = admissible_route_size_bands(n)
    return band if band in allowed else max(allowed)


def enumerate_candidates(
    cities: Sequence[tuple[str, Path]],
    profiles: Sequence[CityProfile] = (),
    *,
    per_city: int = 8,
    size_grid: Sequence[int] = MAIN_SIZE_GRID,
    methods: Sequence[str] = METHODS,
    size_ceiling: Mapping[str, int] | None = None,
    tier: str = "main",
    base_seed: int = 0,
) -> list[CandidateSpec]:
    """``per_city`` candidates for every city, balanced on every design axis.

    ``methods`` narrows the sourcing axis. The POI-only tier passes a single
    method because "every customer is a real amenity" is the point of that tier,
    not one option among three.

    ``size_ceiling`` caps ``n`` per city. Sizes are drawn from a grid that knows
    nothing about how many customers a given city can actually serve, so without
    it a POI-only tier proposes n = 5000 in a city holding 1800 amenities and
    discovers the shortfall only after paying for the evaluation. The cap
    replaces an over-large draw with the largest grid size that still fits, and
    drops the candidate when even the smallest does not.

    The route-size band is drawn as a balanced column and then *repaired against
    the size*, because which bands leave a real fleet depends on ``n``: a target
    of 50-200 customers per route is an ordinary instance at n = 4000 and a
    two-route degenerate at n = 272. A draw the size cannot support falls back to
    the largest admissible band, which keeps the column's ordering -- longer
    draws still get longer routes -- without letting it off the end.
    """
    if per_city < 1:
        raise ValueError("per_city must be >= 1")
    if not cities:
        return []
    if not methods:
        raise ValueError("methods must not be empty")

    stratum_of = {profile.city: profile.distortion_stratum for profile in profiles}
    total = per_city * len(cities)
    rng = np.random.default_rng(stable_seed("enumerate", tier, base_seed, len(cities), per_city))

    columns = {
        "n": _balanced_column(size_grid, total, rng),
        "method": _balanced_column(methods, total, rng),
        "demand_type": _balanced_column(DEMAND_TYPES, total, rng),
        "avg_route_size": _balanced_column(AVG_ROUTE_SIZES, total, rng),
        "depot_mode": _balanced_column(DEPOT_MODES, total, rng),
        "customer_mode": _balanced_column(CUSTOMER_MODES, total, rng),
        "cluster_seeds": _balanced_column(CLUSTER_SEEDS, total, rng),
        "cluster_decay_meters": _balanced_column(CLUSTER_DECAY_METERS, total, rng),
        "hybrid_poi_share": _balanced_column(HYBRID_POI_SHARES, total, rng),
    }

    ordered_sizes = sorted(size_grid)
    specs: list[CandidateSpec] = []
    for index in range(total):
        city, osm_path = cities[index // per_city]
        n = int(columns["n"][index])
        if size_ceiling is not None:
            ceiling = int(size_ceiling.get(city, 0))
            if n > ceiling:
                fitting = [size for size in ordered_sizes if size <= ceiling]
                if not fitting:
                    continue
                n = fitting[-1]
        specs.append(
            CandidateSpec(
                city=city,
                osm_path=Path(osm_path),
                n=n,
                method=str(columns["method"][index]),
                demand_type=int(columns["demand_type"][index]),
                avg_route_size=_admissible_band(int(columns["avg_route_size"][index]), n),
                depot_mode=str(columns["depot_mode"][index]),
                customer_mode=str(columns["customer_mode"][index]),
                cluster_seeds=int(columns["cluster_seeds"][index]),
                cluster_decay_meters=float(columns["cluster_decay_meters"][index]),
                hybrid_poi_share=float(columns["hybrid_poi_share"][index]),
                distortion_stratum=stratum_of.get(city),
                tier=tier,
            )
        )
    return specs


def evaluate_candidate(spec: CandidateSpec) -> EvaluatedCandidate:
    """What this candidate would be, short of its distance matrices.

    Rebuilds the customer selection and draws the demands through exactly the
    calls ``generate_single_instance`` makes, from exactly the same seed, so the
    descriptors describe the instance that generation will actually produce.
    """
    selection = build_generation_selection(spec.to_request())
    graph = selection.graph
    coordinates = [graph.node_enu[graph.node_of[v]][:2] for v in selection.vertices]

    customer_ll = list(zip(selection.poi_lats[1:], selection.poi_lons[1:]))
    demand_values, _total, _maximum, route_target = generate_demands(
        random.Random(spec.seed), customer_ll, spec.demand_type, spec.avg_route_size
    )
    capacity = capacity_from_avg_route_size(route_target, demand_values)

    customer_tags = selection.source_tags[1:]
    poi_count = sum(1 for tag in customer_tags if is_poi_source_tag(tag))
    return EvaluatedCandidate(
        spec=spec,
        descriptors=compute_descriptors(coordinates, [0, *demand_values], capacity),
        resolved_n=len(selection.vertices) - 1,
        effective_method=str(selection.params["method"]),
        poi_fraction=poi_count / len(customer_tags) if customer_tags else 0.0,
    )


def _evaluate_city(specs: list[CandidateSpec]) -> list[EvaluatedCandidate]:
    """Every candidate of one city, in one process, on one parsed road graph.

    Grouping by city is not a nicety: the road graph, the POI catalog and the
    POI-to-vertex attachment are all cached per extract, so a city evaluated in
    one go parses its OSM once instead of ``per_city`` times.
    """
    evaluated: list[EvaluatedCandidate] = []
    for spec in specs:
        try:
            evaluated.append(evaluate_candidate(spec))
        except (ValueError, FileNotFoundError) as error:
            # One unservable draw (a pool too small for n, an unreachable
            # depot) must not cost the city its other candidates.
            warnings.warn(f"{spec.city} n={spec.n} {spec.method}: {error}", stacklevel=2)
    return evaluated


def evaluate_candidates(
    specs: Sequence[CandidateSpec],
    *,
    jobs: int = 1,
    drop_unusable: bool = True,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> list[EvaluatedCandidate]:
    """Evaluate a whole pool, one process per city.

    ``drop_unusable=False`` returns everything that evaluated, letting the
    caller apply :meth:`EvaluatedCandidate.is_usable` itself. That is what a
    campaign wants for its saved pool: usability depends on the feature space,
    and re-deciding it must not cost another pass over every OSM extract.
    """
    by_city: dict[str, list[CandidateSpec]] = {}
    for spec in specs:
        by_city.setdefault(spec.city, []).append(spec)
    groups = list(by_city.items())

    evaluated: list[EvaluatedCandidate] = []
    if jobs <= 1:
        for index, (city, city_specs) in enumerate(groups, start=1):
            if on_progress is not None:
                on_progress(city, index, len(groups))
            evaluated.extend(_evaluate_city(city_specs))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(_evaluate_city, city_specs): city for city, city_specs in groups
            }
            for index, (future, city) in enumerate(futures.items(), start=1):
                if on_progress is not None:
                    on_progress(city, index, len(groups))
                evaluated.extend(future.result())

    if not drop_unusable:
        return evaluated

    usable = [candidate for candidate in evaluated if candidate.is_usable()]
    dropped = len(evaluated) - len(usable)
    if dropped:
        short = sum(1 for c in evaluated if c.resolved_n < c.spec.n)
        warnings.warn(
            f"dropped {dropped} candidate(s): {short} could not be served at the requested "
            f"size, {dropped - short} had LB_cap < {CAMPAIGN_MIN_ROUTES} or an "
            "undefined descriptor",
            stacklevel=2,
        )
    return usable


def resolve_cities(osm_dir: str | Path, names: Iterable[str] | None = None) -> list[tuple[str, Path]]:
    """``(city_slug, osm_path)`` for the extracts on disk, slugged like the tree.

    ``Aix-en-Provence.osm`` becomes ``aix_en_provence``, matching the place slugs
    the benchmark tree and the website already use.
    """
    directory = Path(osm_dir)
    wanted = {slugify(name) for name in names} if names else None
    cities: list[tuple[str, Path]] = []
    for path in sorted(directory.glob("*.osm")):
        slug = slugify(path.stem)
        if wanted is None or slug in wanted:
            cities.append((slug, path))
    if wanted is not None:
        missing = wanted - {slug for slug, _ in cities}
        if missing:
            raise FileNotFoundError(f"no OSM extract for: {', '.join(sorted(missing))}")
    return cities
