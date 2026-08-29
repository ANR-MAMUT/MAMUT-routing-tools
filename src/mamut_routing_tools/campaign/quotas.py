"""The coverage requirements of the Mamut2026 CVRP campaign.

Kept apart from the selector so the *policy* -- what this benchmark promises to
cover -- is one readable block that a reader of the family README can check
against the published coverage table, rather than something buried in a CLI.
"""

from __future__ import annotations

from typing import Sequence

from mamut_routing_tools.campaign.city_profile import STRATA
from mamut_routing_tools.campaign.design import (
    AVG_ROUTE_SIZES,
    CUSTOMER_MODES,
    DEMAND_TYPES,
    DEPOT_MODES,
    METHODS,
    EvaluatedCandidate,
)
from mamut_routing_tools.campaign.select import Quota

#: Size buckets the main tier must spread over. Half-open, upper bound inclusive
#: on the last one.
SIZE_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("100-200", 100, 200),
    ("201-400", 201, 400),
    ("401-700", 401, 700),
    ("701-1000", 701, 1000),
)


def size_bucket(n: int) -> str:
    for name, low, high in SIZE_BUCKETS:
        if low <= n <= high:
            return name
    return f">{SIZE_BUCKETS[-1][2]}"


#: Size buckets for the POI-only large tier. Wider and fewer than the main
#: tier's: above n = 1000 the binding constraint is how many amenities a city
#: actually holds, so there is no point promising a fine size grid that only
#: a handful of cities on Earth could fill.
POI_SIZE_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("1000-1500", 1000, 1500),
    ("1501-2500", 1501, 2500),
    ("2501-5000", 2501, 5000),
)


def poi_size_bucket(n: int) -> str:
    for name, low, high in POI_SIZE_BUCKETS:
        if low <= n <= high:
            return name
    return f"<{POI_SIZE_BUCKETS[0][1]}" if n < POI_SIZE_BUCKETS[0][1] else f">{POI_SIZE_BUCKETS[-1][2]}"


#: An axis is only quota'd once the budget can give each of its levels this
#: many instances. Below it, a coverage floor stops being a floor and becomes a
#: grid: requiring one of each of three levels out of four slots fixes almost
#: the whole selection, and three such axes at once fix it completely, leaving
#: the max-min pass -- the entire reason for selecting rather than enumerating --
#: nothing to choose. Small pilot runs therefore select on spread alone, which
#: is the honest behaviour: four instances cannot cover seven demand types.
MIN_SHARE_FOR_QUOTA = 2.0


def _even_minimum(levels: Sequence[object], k: int, fraction: float) -> dict[object, int]:
    """``fraction`` of an even share of ``k``, per level.

    An even share would be ``k / len(levels)``; demanding all of it would leave
    the selector no freedom at all. ``fraction`` is how much of the even share
    is mandatory, and the result is truncated, so the total requirement
    ``len(levels) * floor(k / len(levels) * fraction)`` never exceeds ``k`` and
    a quota can never be arithmetically unsatisfiable on its own.

    Returns nothing at all when the even share is below
    :data:`MIN_SHARE_FOR_QUOTA` -- see the note there.
    """
    share = k / len(levels)
    if share < MIN_SHARE_FOR_QUOTA:
        return {}
    minimum = int(share * fraction)
    return {level: minimum for level in levels} if minimum > 0 else {}


def _even_maximum(levels: Sequence[object], k: int, slack: float) -> dict[object, int]:
    """An even share of ``k`` per level, times ``slack``, as a ceiling.

    The mirror of :func:`_even_minimum`, and it exists because a floor alone
    does not constrain a max-min selector. Spread is scored on a continuum, so
    once every floor is met the remaining budget goes wherever feature space is
    emptiest -- which is its edges. In v2 the seven route-size levels had a floor
    of 7 each and the longest level took 39 of 100; the levels were not
    *neglected*, they were outbid.

    ``slack`` above 1 leaves the selector real freedom: at 1.6 with seven levels
    and k = 100 a level may take 22 rather than its even 14, but not 39.
    """
    share = k / len(levels)
    if share < MIN_SHARE_FOR_QUOTA:
        return {}
    return {level: max(1, int(share * slack)) for level in levels}


def ladder_tier_quotas(rung_keys: Sequence[str]) -> list[Quota]:
    """Coverage for a campaign whose sizes and sourcing are already assigned.

    Four of the old quotas are gone because the ladder assignment decided them
    before selection began: ``method`` is fixed per rung, ``avg_route_size`` too,
    ``size_bucket`` is covered by construction (one instance per distinct ``n``),
    and ``city`` is already one-per-instance. What replaces them is a single pinned quota --
    exactly one instance per ladder position -- which is what turns "select 100
    good instances" into "select the best instance for each of these 100 slots".

    The axes that remain are the ones the assignment left free. They keep their
    floors so the set stays analysable on each of them, and ``demand_type`` gains
    a *ceiling* as well -- a floor alone does not stop a max-min selector from
    parking its spare budget on one level.
    """
    k = len(rung_keys)
    return [
        # min == max == 1: the selector must fill every rung and may not double
        # up. Quota.cap_for enforces the ceiling hard; _shortfall raises up
        # front if any rung has no usable candidate.
        Quota(
            name="rung",
            key=lambda c: c.spec.rung_key,
            minimum={key: 1 for key in rung_keys},
            maximum={key: 1 for key in rung_keys},
        ),
        Quota(name="city", key=lambda c: c.spec.city, per_level_maximum=1),
        Quota(
            name="distortion_stratum",
            key=lambda c: c.spec.distortion_stratum,
            minimum=_even_minimum(STRATA, k, 0.80),
        ),
        Quota(
            name="demand_type",
            key=lambda c: c.spec.demand_type,
            minimum=_even_minimum(DEMAND_TYPES, k, 0.70),
            maximum=_even_maximum(DEMAND_TYPES, k, 1.60),
        ),
        # No ``avg_route_size`` quota: the ladder assigns the band per rung, so
        # it is already exact rather than merely floored. See
        # ``ladder.assign_route_size_bands``.
        Quota(
            name="depot_mode",
            key=lambda c: c.spec.depot_mode,
            minimum=_even_minimum(DEPOT_MODES, k, 0.75),
        ),
        Quota(
            name="customer_mode",
            key=lambda c: c.spec.customer_mode,
            minimum=_even_minimum(CUSTOMER_MODES, k, 0.75),
        ),
    ]


def main_tier_quotas(k: int) -> list[Quota]:
    """Coverage the ~100-instance main tier must satisfy.

    One instance per city, all three road-distortion strata represented, and
    every level of every design axis present in analysable numbers. The
    fractions are deliberately below an even share: they are a floor that makes
    the set analysable, not a straitjacket that turns selection back into a grid.

    Superseded by :func:`ladder_tier_quotas` for the ladder campaign; kept for
    the grid-drawn tiers, which still draw their sizes and methods at random.
    """
    return [
        Quota(name="city", key=lambda c: c.spec.city, per_level_maximum=1),
        Quota(
            name="distortion_stratum",
            key=lambda c: c.spec.distortion_stratum,
            minimum=_even_minimum(STRATA, k, 0.80),
        ),
        Quota(
            name="demand_type",
            key=lambda c: c.spec.demand_type,
            minimum=_even_minimum(DEMAND_TYPES, k, 0.70),
        ),
        Quota(
            name="avg_route_size",
            key=lambda c: c.spec.avg_route_size,
            minimum=_even_minimum(AVG_ROUTE_SIZES, k, 0.55),
        ),
        Quota(
            name="depot_mode",
            key=lambda c: c.spec.depot_mode,
            minimum=_even_minimum(DEPOT_MODES, k, 0.75),
        ),
        Quota(
            name="customer_mode",
            key=lambda c: c.spec.customer_mode,
            minimum=_even_minimum(CUSTOMER_MODES, k, 0.75),
        ),
        Quota(
            name="method",
            # The realized sourcing, not the requested one: see
            # EvaluatedCandidate.effective_method.
            key=lambda c: c.effective_method or c.spec.method,
            minimum=_even_minimum(METHODS, k, 0.60),
        ),
        Quota(
            name="size_bucket",
            key=lambda c: size_bucket(c.spec.n),
            minimum=_even_minimum([name for name, _, _ in SIZE_BUCKETS], k, 0.60),
        ),
    ]


def large_tier_quotas(k: int) -> list[Quota]:
    """Coverage for the handful of large instances.

    Far weaker, and deliberately so: at this size the budget is measured in
    hours per instance, so the only things worth insisting on are that no city
    appears twice and that the distortion strata are not all the same.
    """
    return [
        Quota(name="city", key=lambda c: c.spec.city, per_level_maximum=1),
        Quota(
            name="distortion_stratum",
            key=lambda c: c.spec.distortion_stratum,
            minimum={stratum: 1 for stratum in STRATA} if k >= len(STRATA) else {},
        ),
    ]


def excluded_cities(selected: Sequence[EvaluatedCandidate]) -> set[str]:
    """Cities already used, so a later tier does not reuse one."""
    return {candidate.spec.city for candidate in selected}


def poi_tier_quotas(k: int) -> list[Quota]:
    """Coverage for the POI-only large tier.

    Three axes only. The sourcing axis is gone -- every instance in this tier is
    ``poi_categories`` by construction, which is the tier's whole point -- and
    the demand and depot axes are left to the max-min pass, because the pool is
    already small: only a handful of cities on Earth hold enough attachable
    amenities to serve thousands of customers, and quota'ing every axis over
    ~10 slots would fix the selection rather than guide it.

    What is worth insisting on is that the tier spans its *size* range. An
    instance set that is nominally "1000 to 5000" but is really nine instances
    at 1100 and one at 5000 does not measure what happens as n grows.

    Route size gets a *ceiling* and no floor. A floor here would be one of the
    quotas that fixes rather than guides -- seven levels over ten slots -- but a
    ceiling costs the selector almost nothing and stops the thing that actually
    happened in v2, where five of these ten instances landed in the longest band
    and ``mamut-athens-n4000-poi`` came out at 154 customers per route. This is
    the tier where the longest band is legitimate, so it is capped rather than
    excluded.
    """
    return [
        Quota(name="city", key=lambda c: c.spec.city, per_level_maximum=1),
        Quota(
            name="distortion_stratum",
            key=lambda c: c.spec.distortion_stratum,
            minimum={stratum: 1 for stratum in STRATA} if k >= len(STRATA) else {},
        ),
        Quota(
            name="poi_size_bucket",
            key=lambda c: poi_size_bucket(c.spec.n),
            minimum=_even_minimum([name for name, _, _ in POI_SIZE_BUCKETS], k, 0.60),
        ),
        Quota(
            name="avg_route_size",
            key=lambda c: c.spec.avg_route_size,
            per_level_maximum=max(2, int(k / len(AVG_ROUTE_SIZES) * 1.60)),
        ),
    ]
