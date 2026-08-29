"""The exact fleet lower bound published in an instance's name.

``k`` in ``mamut-lyon-n1000-k24-poi`` follows CVRPLIB: it is the minimum number
of bins the demands need, and it is a *lower bound* on a solution's route count,
not a cap. Getting it wrong in the safe direction costs information; getting it
wrong in the unsafe direction would make the name a false claim, so every test
here checks the value against something independent of the implementation.
"""

from __future__ import annotations

import random

import pytest

from mamut_routing_tools.generation.binpacking import (
    BinCount,
    best_fit_decreasing,
    cardinality_bound,
    continuous_bound,
    first_fit_decreasing,
    martello_toth_l2,
    minimum_bins,
    minimum_vehicles,
)


def _brute_force(sizes: list[int], capacity: int) -> int:
    """Exhaustive minimum bin count, for cross-checking small cases."""
    order = sorted(sizes, reverse=True)
    best = [len(order)]

    def recurse(index: int, bins: list[int]) -> None:
        if len(bins) >= best[0]:
            return
        if index == len(order):
            best[0] = len(bins)
            return
        size = order[index]
        seen: set[int] = set()
        for slot, room in enumerate(bins):
            if room >= size and room not in seen:
                seen.add(room)
                bins[slot] -= size
                recurse(index + 1, bins)
                bins[slot] += size
        bins.append(capacity - size)
        recurse(index + 1, bins)
        bins.pop()

    recurse(0, [])
    return best[0]


@pytest.mark.parametrize(
    ("sizes", "capacity", "expected"),
    [
        ([1] * 10, 3, 4),          # 3 + 3 + 3 + 1
        ([5, 5, 5, 5], 10, 2),
        ([3] * 6, 7, 3),           # two per bin; the third unit is wasted
        ([7, 6, 5, 4, 3, 2], 10, 3),
        ([9, 9, 9], 10, 3),        # nothing pairs
        ([10], 10, 1),
    ],
)
def test_known_packings(sizes, capacity, expected) -> None:
    result = minimum_bins(sizes, capacity)
    assert result.lower == expected
    assert result.proven


def test_matches_brute_force_on_random_instances() -> None:
    """The property that matters, checked against an independent search."""
    rng = random.Random(0)
    for _ in range(300):
        capacity = rng.randint(5, 20)
        sizes = [rng.randint(1, capacity) for _ in range(rng.randint(2, 9))]
        result = minimum_bins(sizes, capacity)
        assert result.proven, f"{sizes} / {capacity} should be settled at this size"
        assert result.lower == _brute_force(sizes, capacity)


def test_bounds_always_bracket_the_answer() -> None:
    """Whether or not the search closes, the interval must be honest."""
    rng = random.Random(7)
    for _ in range(200):
        capacity = rng.randint(10, 60)
        sizes = [rng.randint(1, capacity) for _ in range(rng.randint(5, 60))]
        result = minimum_bins(sizes, capacity)
        assert continuous_bound(sizes, capacity) <= result.lower
        assert martello_toth_l2(sizes, capacity) <= result.lower
        assert cardinality_bound(sizes, capacity) <= result.lower
        assert result.lower <= result.upper
        assert result.upper <= min(
            first_fit_decreasing(sizes, capacity), best_fit_decreasing(sizes, capacity)
        )


def test_l2_can_beat_the_continuous_bound() -> None:
    """Why this module exists rather than a one-line ``ceil``.

    Three items of 6 in bins of 10 need three bins -- no two fit together -- but
    ``ceil(18 / 10)`` says two. The published v2 collection carried exactly this
    error: ``mamut-metz-n135-poi`` shipped ``num_vehicles_lb = 33`` where L2
    proves 37.
    """
    assert continuous_bound([6, 6, 6], 10) == 2
    assert martello_toth_l2([6, 6, 6], 10) == 3
    assert minimum_vehicles([6, 6, 6], 10) == 3


def test_an_unproven_result_still_reports_a_valid_lower_bound() -> None:
    """A budget too small to settle the question must not produce a wrong answer."""
    rng = random.Random(3)
    capacity = 100
    sizes = [rng.randint(20, 60) for _ in range(60)]
    exact = minimum_bins(sizes, capacity)
    starved = minimum_bins(sizes, capacity, node_budget=1)
    assert starved.lower <= exact.lower
    assert starved.lower >= continuous_bound(sizes, capacity)


def test_a_demand_larger_than_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="no packing exists"):
        minimum_bins([5, 12], 10)


def test_no_demand_needs_no_vehicle() -> None:
    assert minimum_bins([], 10) == BinCount(lower=0, upper=0, method="empty")
    assert minimum_vehicles([0, 0], 10) == 0


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        minimum_bins([1], 0)


def test_deep_instances_do_not_exhaust_the_stack() -> None:
    """The POI tier reaches n = 4000, four times CPython's default recursion limit."""
    sizes = [3] * 4000
    assert minimum_vehicles(sizes, 10) == 1334


def test_counting_beats_weighing_when_the_demands_are_small() -> None:
    """Why :func:`cardinality_bound` is there.

    Four thousand demands of 3 in bins of 10 weigh exactly 12 000 against 12 000
    of capacity, so every weight-based bound says 1200 and is wrong by 134 bins:
    a bin holds three of them, not three and a third.
    """
    sizes = [3] * 4000
    assert continuous_bound(sizes, 10) == 1200
    assert martello_toth_l2(sizes, 10) == 1200
    assert cardinality_bound(sizes, 10) == 1334
    assert minimum_bins(sizes, 10).proven
