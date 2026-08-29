"""Which city gets which size, and which sourcing method.

The previous design drew ``n``, ``method`` and the city independently and then
discovered, during generation, that the city could not serve the draw. A POI
request the amenities could not fill was quietly completed with sampled road
points and reclassified as ``hybrid``: the instance still reached its size, so
nothing failed, and the published set ended up with a third of the sourcing it
was designed for.

This module inverts that. Sizes come first, as a *ladder* -- one distinct ``n``
per instance, rising smoothly, the way Uchoa et al.'s X set moves from 101 to
1001 -- and each rung is then matched to a city that can demonstrably supply it.
The matching reads the measured POI capacity from :mod:`~.poi_capacity`, so a
rung asking for 800 real amenities is only ever given to a city that has them.

Two consequences worth stating, because they are the point:

- **A rung that cannot be filled is reported, not silently downgraded.** The
  unfillable list says what each orphan rung needed and what the best remaining
  city offered, which is a precise specification for how many more extracts to
  fetch rather than a vague "get some bigger cities".
- **Size stops being a factor and becomes a covariate.** With one instance per
  ``n`` you can regress solver performance on size; with fourteen instances
  piled on ``n = 100`` you can only compare buckets.

What this module does *not* decide is what the instance is like -- demands,
depot placement, clustering. Those stay with the descriptor-driven selector in
:mod:`~.select`, which now chooses among candidates that share a pinned
``(city, n, method)`` triple.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from mamut_routing_tools.campaign.city_profile import STRATA, CityProfile
from mamut_routing_tools.campaign.design import (
    AVG_ROUTE_SIZES,
    CAMPAIGN_MIN_ROUTES,
    stable_seed,
)
from mamut_routing_tools.campaign.poi_capacity import PoiCapacity
from mamut_routing_tools.generation.demands import avg_route_size_bounds

#: Sourcing mix for the main tier. POI-sourced instances are the platform's
#: distinguishing artifact -- customers that are real places on a real map -- so
#: they are half the set rather than a third of it. The other two are kept
#: because they are genuinely different problem shapes: ``parametric_attach`` can put
#: customers anywhere the road network reaches, including geometries no city's
#: commercial map would ever produce, and ``hybrid`` sits between the two.
DEFAULT_METHOD_WEIGHTS: dict[str, float] = {
    "poi_categories": 0.50,
    "hybrid": 0.25,
    "parametric_attach": 0.25,
}

#: How much amenity slack above ``n`` a city must have before a rung is given to
#: it. At ``capacity == n + 1`` the "selection" is the entire amenity pool, so
#: the seed stops mattering and the instance stops being a draw from anything.
#:
#: Four rather than the 1.25 that merely makes the draw non-degenerate, because
#: the matching is best-fit and best-fit lands every rung on a city that *just*
#: clears the bar. Measured on the real 102-city pool over a 100-rung ladder:
#:
#:     headroom   filled   min POI ratio   corr(log n, log city size)
#:       1.25     100/100       1.25                 -0.40
#:       2.00     100/100       2.01                 -0.43
#:       4.00     100/100       4.00                 -0.07
#:       6.00     100/100       6.00                 +0.26
#:       8.00      97/100         --                   --
#:
#: The last column is the one that decides it. At low headroom the large POI
#: rungs land on small amenity-dense cities while the large road networks go to
#: parametric rungs, so instance size ends up *negatively* correlated with city
#: size -- a confound between the study's covariate and the cities it is
#: measured over. Four zeroes that correlation, keeps every POI instance a
#: genuine sample of at most a quarter of its city's amenities, and still fills
#: the ladder with room to spare.
DEFAULT_HEADROOM = 4.0

#: Road vertices a parametric rung needs per customer. Parametric sampling can
#: reuse no vertex, so it needs the graph to be comfortably larger than the
#: instance; well below the POI headroom because road vertices are abundant.
DEFAULT_VERTEX_FACTOR = 4.0


def admissible_route_size_bands(
    n: int, *, min_routes: int = CAMPAIGN_MIN_ROUTES
) -> list[int]:
    """Route-size bands that keep an ``n``-customer instance a real VRP.

    A band is admissible only if its *whole* interval fits in
    ``[3, n / min_routes]``. Testing the band rather than the realized draw is
    the point: ``r`` is drawn uniformly inside the band at generation time, so
    admitting a band that only partly fits would make soundness depend on the
    draw, which is exactly the coin-flip v2 lost. Full containment means the
    guarantee holds for every seed.

    Band 7 (50-200 customers per route) therefore never appears below n = 1200,
    which independently reproduces CVRPLIB's own convention: XL introduced that
    band together with n >= 1000, and it is only this repository that had it
    reachable from n = 100.
    """
    if min_routes < 1:
        raise ValueError(f"min_routes must be positive, got {min_routes}")
    admissible = [
        band
        for band in AVG_ROUTE_SIZES
        if avg_route_size_bounds(band)[1] * min_routes <= n
    ]
    if not admissible:
        raise ValueError(
            f"no route-size band keeps n={n} above {min_routes} routes; "
            f"the smallest band needs n >= {int(avg_route_size_bounds(AVG_ROUTE_SIZES[0])[1] * min_routes)}"
        )
    return admissible


def assign_route_size_bands(
    sizes: Sequence[int],
    rng: np.random.Generator,
    *,
    min_routes: int = CAMPAIGN_MIN_ROUTES,
) -> list[int]:
    """One route-size band per rung, spread as evenly as admissibility allows.

    The bands available to a rung are a *prefix* of the vocabulary -- a bigger
    instance can carry every band a smaller one can, plus longer ones -- so the
    rungs with the fewest choices are the ones that constrain the outcome.
    Serving them first, each taking the least-used band it is allowed, is what
    lets the scarce long bands land on the large rungs that are the only ones
    able to hold them. Ties are broken with ``rng`` rather than by index, which
    keeps the assignment from correlating band with ``n`` any more than
    admissibility already forces.

    Returns bands in the order of ``sizes``.
    """
    order = sorted(
        range(len(sizes)),
        key=lambda index: (len(admissible_route_size_bands(sizes[index], min_routes=min_routes)), rng.random()),
    )
    used: dict[int, int] = {band: 0 for band in AVG_ROUTE_SIZES}
    bands: list[int | None] = [None] * len(sizes)
    for index in order:
        allowed = admissible_route_size_bands(sizes[index], min_routes=min_routes)
        fewest = min(used[band] for band in allowed)
        tied = [band for band in allowed if used[band] == fewest]
        chosen = int(tied[int(rng.integers(len(tied)))])
        bands[index] = chosen
        used[chosen] += 1
    return _decorrelate_bands(
        sizes, [int(band) for band in bands], min_routes=min_routes  # type: ignore[arg-type]
    )


def band_size_correlation(sizes: Sequence[int], bands: Sequence[int]) -> float:
    """Pearson correlation between ``log n`` and the assigned band."""
    if len(sizes) < 2:
        return 0.0
    logs = np.log(np.asarray(sizes, dtype=float))
    values = np.asarray(bands, dtype=float)
    if logs.std() == 0 or values.std() == 0:
        return 0.0
    return float(np.corrcoef(logs, values)[0, 1])


def _decorrelate_bands(
    sizes: Sequence[int], bands: list[int], *, min_routes: int
) -> list[int]:
    """Swap bands between rungs to pull ``corr(log n, band)`` toward zero.

    Admissibility forces *some* correlation -- band 6 cannot appear below
    n = 300, so the long bands are stuck at the top of the ladder -- but the
    greedy pass leaves more than the arithmetic requires, because whichever
    bands the constrained small rungs did not take pile onto the large ones.

    Only swaps are considered, so the even apportionment computed above is
    preserved exactly; a swap is legal only when each band is admissible at the
    other rung's size. This is the same shape as the size-to-city guard the
    matching already applies, and for the same reason: a design axis that
    correlates with ``n`` is confounded with the study's own covariate.
    """
    allowed = [set(admissible_route_size_bands(n, min_routes=min_routes)) for n in sizes]
    current = abs(band_size_correlation(sizes, bands))
    improved = True
    while improved and current > 1e-9:
        improved = False
        for i in range(len(bands)):
            for j in range(i + 1, len(bands)):
                if bands[i] == bands[j]:
                    continue
                if bands[j] not in allowed[i] or bands[i] not in allowed[j]:
                    continue
                bands[i], bands[j] = bands[j], bands[i]
                candidate = abs(band_size_correlation(sizes, bands))
                if candidate < current - 1e-12:
                    current = candidate
                    improved = True
                else:
                    bands[i], bands[j] = bands[j], bands[i]
    return bands


def size_ladder(k: int, n_min: int = 100, n_max: int = 1000) -> list[int]:
    """``k`` distinct sizes rising smoothly from ``n_min`` to ``n_max``.

    Geometrically spaced, so the ladder spends its resolution where instances
    differ most: the step from 100 to 110 changes a CVRP far more than the step
    from 900 to 910. Rounding collides at the bottom, where consecutive rungs
    are less than a customer apart, so a repair pass walks the ladder and pushes
    each rung at least one past its predecessor. Both endpoints are exact.
    """
    if k < 2:
        raise ValueError("a ladder needs at least two rungs")
    if n_min < 1 or n_max <= n_min:
        raise ValueError(f"need 1 <= n_min < n_max, got {n_min} and {n_max}")
    if n_max - n_min + 1 < k:
        raise ValueError(
            f"cannot fit {k} distinct sizes between {n_min} and {n_max}"
        )

    raw = np.geomspace(n_min, n_max, k)
    rungs: list[int] = []
    for index, value in enumerate(raw):
        rung = int(round(value))
        if rungs:
            rung = max(rung, rungs[-1] + 1)
        # Never overshoot so far that the tail cannot stay strictly increasing.
        rung = min(rung, n_max - (k - 1 - index))
        rungs.append(rung)
    rungs[-1] = n_max
    return rungs


def weighted_column(
    weights: Mapping[str, float], length: int, rng: np.random.Generator
) -> list[str]:
    """``length`` labels in the requested proportions, shuffled.

    Exact rather than sampled: 50/25/25 of 100 is 50, 25 and 25, not "about
    that". Remainders from the largest-remainder apportionment go to the
    heaviest weights first, so a 50 % share is never the one left short.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to something positive")

    exact = {name: length * weight / total for name, weight in weights.items()}
    counts = {name: int(value) for name, value in exact.items()}
    remainder = length - sum(counts.values())
    # Largest remainder, then largest weight, then name -- fully deterministic.
    order = sorted(
        weights,
        key=lambda name: (-(exact[name] - counts[name]), -weights[name], name),
    )
    for name in order[:remainder]:
        counts[name] += 1

    column = [name for name, count in counts.items() for _ in range(count)]
    rng.shuffle(column)
    return column


@dataclass(frozen=True)
class RungAssignment:
    """One ladder position, and the city that will supply it."""

    index: int
    n: int
    method: str
    #: Route-size band, assigned here rather than drawn by the selector because
    #: which bands are *legal* depends on ``n``: see
    #: :func:`admissible_route_size_bands`. An axis whose valid range varies by
    #: rung cannot be a free draw filtered afterwards -- that is how v2 ended up
    #: with 39 of 100 instances in the longest band and 22 of them not VRPs.
    avg_route_size: int
    city: str
    osm_file: str
    distortion_stratum: str | None
    #: What this pairing demanded of the city, and what the city actually has.
    #: Kept so the assignment is auditable without re-deriving the arithmetic.
    required: int
    available: int

    @property
    def rung_key(self) -> str:
        """Stable identifier for quota'ing exactly one instance per rung.

        The route-size band belongs in it. The key is what the candidate-pool
        cache is fingerprinted on, and a plan that differs only in its bands is a
        different design producing different demands and capacities -- but would
        hash identically without this, and silently reuse the other one's pool.
        """
        return f"{self.index:03d}-n{self.n}-{self.method}-b{self.avg_route_size}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "RungAssignment":
        return cls(**{key: record[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class UnfilledRung:
    """A rung no remaining city could supply, and by how much it fell short."""

    index: int
    n: int
    method: str
    required: int
    best_available: int
    best_city: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LadderPlan:
    """The complete size/method/city assignment, plus what it could not fill."""

    assignments: list[RungAssignment] = field(default_factory=list)
    unfilled: list[UnfilledRung] = field(default_factory=list)
    method_weights: dict[str, float] = field(default_factory=dict)
    headroom: float = DEFAULT_HEADROOM
    min_routes: int = CAMPAIGN_MIN_ROUTES

    @property
    def complete(self) -> bool:
        return not self.unfilled

    @property
    def band_size_corr(self) -> float:
        """``corr(log n, band)`` over the plan -- the decorrelation guard's result.

        Reported rather than asserted: admissibility puts a hard floor under it
        (band 6 cannot appear below n = 300), so the honest thing is to publish
        what the assignment achieved, not to pretend the axes are independent.
        """
        return band_size_correlation(
            [item.n for item in self.assignments],
            [item.avg_route_size for item in self.assignments],
        )

    @property
    def guaranteed_min_routes(self) -> float:
        """Fewest routes any assigned rung can produce, over every legal draw."""
        if not self.assignments:
            return float("inf")
        return min(
            item.n / avg_route_size_bounds(item.avg_route_size)[1]
            for item in self.assignments
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mamut-ladder-plan",
            "version": 1,
            "method_weights": self.method_weights,
            "headroom": self.headroom,
            "min_routes": self.min_routes,
            "band_size_corr": self.band_size_corr,
            "guaranteed_min_routes": self.guaranteed_min_routes,
            "assignments": [item.to_dict() for item in self.assignments],
            "unfilled": [item.to_dict() for item in self.unfilled],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LadderPlan":
        return cls(
            assignments=[RungAssignment.from_dict(r) for r in payload["assignments"]],
            unfilled=[UnfilledRung(**r) for r in payload.get("unfilled", [])],
            method_weights=dict(payload.get("method_weights", {})),
            headroom=float(payload.get("headroom", DEFAULT_HEADROOM)),
            min_routes=int(payload.get("min_routes", CAMPAIGN_MIN_ROUTES)),
        )


def required_supply(
    n: int,
    method: str,
    *,
    hybrid_share: float,
    headroom: float,
    vertex_factor: float,
) -> tuple[str, int]:
    """``(resource, amount)`` a rung of this size and method demands.

    ``resource`` is ``"poi"`` or ``"vertices"`` -- the two are not
    interchangeable, and comparing a POI rung against a road-graph size is how
    the old design talked itself into instances it could not build.
    """
    nodes = n + 1  # customers plus the depot
    if method == "poi_categories":
        return "poi", int(np.ceil(nodes * headroom))
    if method == "hybrid":
        return "poi", int(np.ceil(nodes * hybrid_share * headroom))
    if method == "parametric_attach":
        return "vertices", int(np.ceil(nodes * vertex_factor))
    raise ValueError(f"unknown sourcing method {method!r}")


def assign_rungs(
    ladder: Sequence[int],
    capacities: Iterable[PoiCapacity],
    profiles: Iterable[CityProfile] = (),
    *,
    method_weights: Mapping[str, float] | None = None,
    hybrid_share: float = 0.5,
    headroom: float = DEFAULT_HEADROOM,
    vertex_factor: float = DEFAULT_VERTEX_FACTOR,
    min_routes: int = CAMPAIGN_MIN_ROUTES,
    base_seed: int = 0,
) -> LadderPlan:
    """Match every rung to a city that can demonstrably supply it.

    **Cheapest sufficient city, hardest rung first.** Rungs are processed in
    decreasing order of what they demand, and each takes the unused city that
    satisfies it while costing the pool the least *amenity capacity*.

    Amenity capacity is the scarce resource and road vertices are not: every
    city has plenty of road and only some have plenty of shops. So the cost of
    spending a city is its POI capacity, whatever the rung happens to need. For
    a POI or hybrid rung that is ordinary best fit -- the smallest capacity that
    still clears the bar, which keeps Tokyo available for the rungs that need
    Tokyo. For a parametric rung, which consumes no amenities at all, it means
    deliberately spending the amenity-*poorest* city that has enough road.

    That second case is not a refinement, it is the difference between filling
    the ladder and not. Ranking parametric rungs by road vertices instead makes
    a city with 563 000 vertices and 57 amenities the *last* candidate for the
    only kind of rung it can serve, so it is never used -- and the ladder then
    runs out of cities while five amenity-dead extracts sit unspent.

    Ties among equally sufficient cities break toward the distortion stratum
    that is currently most under-represented, so the road-network diversity the
    study depends on survives a matching that is otherwise driven entirely by
    amenity counts.

    One instance per city: a city is removed from the pool once used.
    """
    weights = dict(method_weights or DEFAULT_METHOD_WEIGHTS)
    rng = np.random.default_rng(stable_seed("ladder", base_seed, len(ladder), tuple(sorted(weights))))
    methods = weighted_column(weights, len(ladder), rng)
    bands = assign_route_size_bands(ladder, rng, min_routes=min_routes)

    stratum_of = {profile.city: profile.distortion_stratum for profile in profiles}
    pool = {item.city: item for item in capacities}

    demands: list[tuple[int, int, str, int, str, int]] = []
    for index, (n, method, band) in enumerate(zip(ladder, methods, bands)):
        resource, amount = required_supply(
            n,
            method,
            hybrid_share=hybrid_share,
            headroom=headroom,
            vertex_factor=vertex_factor,
        )
        demands.append((index, n, method, band, resource, amount))

    # Hardest first. Ordering across resources by raw amount is fine: it only
    # decides who picks first, and within a resource the order is exactly right.
    demands.sort(key=lambda item: (-item[5], item[0]))

    def supply(city: PoiCapacity, resource: str) -> int:
        return city.capacity if resource == "poi" else city.num_vertices

    used_strata: dict[str | None, int] = {stratum: 0 for stratum in STRATA}
    assignments: list[RungAssignment] = []
    unfilled: list[UnfilledRung] = []

    for index, n, method, band, resource, amount in demands:
        fitting = [city for city in pool.values() if supply(city, resource) >= amount]
        if not fitting:
            best = max(pool.values(), key=lambda c: supply(c, resource), default=None)
            unfilled.append(
                UnfilledRung(
                    index=index,
                    n=n,
                    method=method,
                    required=amount,
                    best_available=supply(best, resource) if best else 0,
                    best_city=best.city if best else None,
                )
            )
            continue

        chosen = min(
            fitting,
            key=lambda c: (
                c.capacity,                                  # cheapest in the scarce resource
                used_strata.get(stratum_of.get(c.city), 0),  # then rarest stratum
                c.city,                                      # then deterministic
            ),
        )
        del pool[chosen.city]
        stratum = stratum_of.get(chosen.city)
        used_strata[stratum] = used_strata.get(stratum, 0) + 1
        assignments.append(
            RungAssignment(
                index=index,
                n=n,
                method=method,
                avg_route_size=band,
                city=chosen.city,
                osm_file=chosen.osm_file,
                distortion_stratum=stratum,
                required=amount,
                available=supply(chosen, resource),
            )
        )

    assignments.sort(key=lambda item: item.index)
    unfilled.sort(key=lambda item: item.index)
    return LadderPlan(
        assignments=assignments,
        unfilled=unfilled,
        method_weights=weights,
        headroom=headroom,
        min_routes=min_routes,
    )


def save_plan(plan: LadderPlan, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_plan(path: str | Path) -> LadderPlan:
    return LadderPlan.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def plan_summary(plan: LadderPlan) -> dict[str, Any]:
    """Counts a human can scan to see whether the design came out as asked."""
    methods: dict[str, int] = {}
    strata: dict[str, int] = {}
    bands: dict[int, int] = {}
    for item in plan.assignments:
        methods[item.method] = methods.get(item.method, 0) + 1
        key = item.distortion_stratum or "?"
        strata[key] = strata.get(key, 0) + 1
        bands[item.avg_route_size] = bands.get(item.avg_route_size, 0) + 1
    sizes = [item.n for item in plan.assignments]
    return {
        "assigned": len(plan.assignments),
        "unfilled": len(plan.unfilled),
        "methods": methods,
        "strata": strata,
        "route_size_bands": dict(sorted(bands.items())),
        # The two numbers that say whether the capacity design came out sound:
        # the worst case over every legal draw, and how far the route-size axis
        # ended up entangled with the size axis it is meant to be free of.
        "guaranteed_min_routes": round(plan.guaranteed_min_routes, 2),
        "band_size_corr": round(plan.band_size_corr, 3),
        "n_min": min(sizes) if sizes else None,
        "n_max": max(sizes) if sizes else None,
        "cities": len({item.city for item in plan.assignments}),
    }
