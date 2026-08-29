"""The minimum number of vehicles a load needs, computed rather than estimated.

CVRPLIB names an instance ``X-n101-k25``, and that ``k`` is the exact optimum of
a bin-packing problem over the demands -- not a heuristic's fleet size and not
the continuous bound. It is published as a *lower bound*: the fleet is not fixed
and a solution may use more routes than the name says.

The continuous bound ``ceil(sum(q) / Q)`` is what this repository published
before, and it is genuinely weaker. On the v2 collection it understated the
truth on several bases: ``mamut-metz-n135-poi`` shipped ``num_vehicles_lb = 33``
where the Martello-Toth bound alone already proves 37. A lower bound that is not
tight is not wrong, but it is worth less to anyone using it.

Exactness here is cheap because these demand distributions are kind: capacity is
``ceil(r * mean demand)`` for a route-size target ``r`` of at least three, so
bins hold several items and first-fit-decreasing usually lands on the bound. On
the v2 collection ``L2 == FFD`` closed 104 of 110 bases with no search at all.
The six that did not are handled by a bounded search, and if even that does not
close, the bound is published as a bound and said to be one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

#: How many search nodes a single instance may spend proving its bound before
#: the answer is published as an interval instead. Reached only by instances the
#: cheap bounds leave open, which are a small minority.
DEFAULT_NODE_BUDGET = 200_000


@dataclass(frozen=True)
class BinCount:
    """What is known about the minimum bin count, and how firmly."""

    #: Best proven lower bound. This is the value to publish: it is a valid
    #: lower bound whether or not the search closed.
    lower: int
    #: Best packing actually constructed.
    upper: int
    #: How the bound was established, for the audit trail.
    method: str

    @property
    def proven(self) -> bool:
        return self.lower == self.upper


def continuous_bound(demands: Sequence[int], capacity: int) -> int:
    """``ceil(sum / capacity)`` -- the bound every load satisfies for free."""
    total = sum(demands)
    if total <= 0:
        return 0
    return int(math.ceil(total / capacity))


def martello_toth_l2(demands: Sequence[int], capacity: int) -> int:
    """The L2 bound: count the items too big to share, then bound the rest.

    For a threshold ``a``, items larger than ``capacity - a`` need a bin each and
    can take no item of size ``a`` or more alongside. Items above half capacity
    cannot pair with each other, so each holds its own bin too, leaving a known
    amount of room. Everything from ``a`` up to half capacity must fit in that
    room or open new bins. Maximising over ``a`` gives the bound.
    """
    sizes = sorted((int(q) for q in demands if q > 0), reverse=True)
    if not sizes:
        return 0
    half = capacity / 2
    best = continuous_bound(sizes, capacity)
    thresholds = {0} | {size for size in sizes if size <= half}
    for alpha in thresholds:
        big = [size for size in sizes if size > capacity - alpha]
        medium = [size for size in sizes if capacity - alpha >= size > half]
        small = [size for size in sizes if half >= size >= alpha]
        # Room left over in the bins the medium items already occupy.
        spare = len(medium) * capacity - sum(medium)
        overflow = sum(small) - spare
        extra = int(math.ceil(overflow / capacity)) if overflow > 0 else 0
        best = max(best, len(big) + len(medium) + extra)
    return best


def cardinality_bound(demands: Sequence[int], capacity: int) -> int:
    """How few bins can hold this many items, ignoring the sizes' arithmetic.

    L2 reasons about weight and misses counting. Four thousand demands of 3 in
    bins of 10 weigh 12 000 against 12 000 of capacity, so every weight-based
    bound says 1200 -- but only three of them fit in a bin, so the answer is
    1334. Small-demand instances are exactly where this bites, and
    ``demand_type`` 2 and 3 produce them by design.

    An item larger than ``capacity / (t + 1)`` cannot appear more than ``t``
    times in one bin, so counting such items and dividing by ``t`` bounds the
    fleet. Maximising over ``t`` costs one pass per value and needs no search.
    """
    sizes = sorted((int(q) for q in demands if q > 0), reverse=True)
    if not sizes:
        return 0
    best = 0
    smallest = sizes[-1]
    # Beyond this, the threshold drops under the smallest item and the count
    # stops growing while the divisor keeps rising.
    limit = max(1, capacity // smallest)
    index = 0
    for t in range(1, limit + 1):
        threshold = capacity / (t + 1)
        while index < len(sizes) and sizes[index] > threshold:
            index += 1
        if index:
            best = max(best, int(math.ceil(index / t)))
    return best


def first_fit_decreasing(demands: Sequence[int], capacity: int) -> int:
    """Bins used by FFD -- a constructive upper bound."""
    remaining: list[int] = []
    for size in sorted((int(q) for q in demands if q > 0), reverse=True):
        for index, room in enumerate(remaining):
            if room >= size:
                remaining[index] = room - size
                break
        else:
            remaining.append(capacity - size)
    return len(remaining)


def best_fit_decreasing(demands: Sequence[int], capacity: int) -> int:
    """Bins used by BFD. Beats FFD often enough to be worth the second pass."""
    remaining: list[int] = []
    for size in sorted((int(q) for q in demands if q > 0), reverse=True):
        best_index = -1
        best_room = capacity + 1
        for index, room in enumerate(remaining):
            if size <= room < best_room:
                best_index, best_room = index, room
        if best_index < 0:
            remaining.append(capacity - size)
        else:
            remaining[best_index] -= size
    return len(remaining)


class _BudgetExhausted(Exception):
    """The search spent its node budget without settling the question."""


def _packs_into(sizes: Sequence[int], capacity: int, bins: int, budget: list[int]) -> bool:
    """Can ``sizes`` (descending) fit in ``bins`` bins? Depth-first, budgeted.

    Two prunings carry the search. Placing the largest unplaced item first makes
    every decision the hardest remaining one. And bins with equal remaining room
    are interchangeable, so trying only *distinct* remaining capacities collapses
    the symmetry that makes naive bin packing hopeless -- without it a hundred
    equal demands would be explored a hundred-factorial ways.

    Iterative rather than recursive: the depth of this search is the number of
    customers, and the POI tier reaches 4000 of them, which is four times
    CPython's default stack.
    """
    count = len(sizes)
    remaining = [capacity] * bins
    # Per depth: which slot the item went into, which room sizes have been tried,
    # and where to resume scanning after a backtrack.
    chosen = [-1] * count
    tried: list[set[int]] = [set() for _ in range(count)]
    resume = [0] * count

    depth = 0
    while depth >= 0:
        if depth == count:
            return True
        if budget[0] <= 0:
            raise _BudgetExhausted
        budget[0] -= 1

        size = sizes[depth]
        seen = tried[depth]
        slot = resume[depth]
        placed = False
        while slot < bins:
            room = remaining[slot]
            if room < size or room in seen:
                # Empty bins are interchangeable and occupy the tail, so once one
                # has been tried and rejected, none of the rest can help.
                if room == capacity:
                    break
                slot += 1
                continue
            seen.add(room)
            remaining[slot] = room - size
            chosen[depth] = slot
            resume[depth] = slot + 1
            depth += 1
            if depth < count:
                tried[depth].clear()
                resume[depth] = 0
            placed = True
            break

        if not placed:
            tried[depth].clear()
            resume[depth] = 0
            depth -= 1
            if depth >= 0:
                remaining[chosen[depth]] += sizes[depth]

    return False


def minimum_bins(
    demands: Sequence[int],
    capacity: int,
    *,
    node_budget: int = DEFAULT_NODE_BUDGET,
) -> BinCount:
    """Fewest bins of ``capacity`` that hold ``demands``.

    Returns what was proven, not a guess: when the bounds meet, ``lower`` is the
    optimum; when the search runs out of budget, ``lower`` is still a valid lower
    bound and ``proven`` says so.
    """
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity}")
    sizes = sorted((int(q) for q in demands if q > 0), reverse=True)
    if not sizes:
        return BinCount(lower=0, upper=0, method="empty")
    if sizes[0] > capacity:
        raise ValueError(
            f"demand {sizes[0]} exceeds capacity {capacity}: no packing exists"
        )

    lower = max(
        continuous_bound(sizes, capacity),
        martello_toth_l2(sizes, capacity),
        cardinality_bound(sizes, capacity),
    )
    upper = min(first_fit_decreasing(sizes, capacity), best_fit_decreasing(sizes, capacity))
    if lower >= upper:
        return BinCount(lower=upper, upper=upper, method="bounds-met")

    budget = [int(node_budget)]
    for target in range(lower, upper):
        try:
            if _packs_into(sizes, capacity, target, budget):
                return BinCount(lower=target, upper=target, method="search")
        except _BudgetExhausted:
            return BinCount(lower=lower, upper=upper, method="budget-exhausted")
        lower = target + 1
    return BinCount(lower=upper, upper=upper, method="search")


def minimum_vehicles(demands: Sequence[int], capacity: int) -> int:
    """The published ``k``: the best proven lower bound on the fleet.

    Valid as a lower bound in every case, exact in all but the rare instance
    whose search does not close. Callers that need to know which they got should
    use :func:`minimum_bins`.
    """
    return minimum_bins(demands, capacity).lower
