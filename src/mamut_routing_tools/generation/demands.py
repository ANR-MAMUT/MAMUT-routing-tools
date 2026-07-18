"""Demand distributions, route-size bands, and the capacity formula (port of
the workbench generator's seven demand types)."""

from __future__ import annotations

import math
import random

DEMAND_TYPES = tuple(range(1, 8))
AVG_ROUTE_SIZES = tuple(range(1, 8))


def demand_distribution_bounds(demand_type: int) -> tuple[int, int]:
    bounds = {1: (1, 1), 2: (1, 10), 3: (5, 10), 4: (1, 100), 5: (50, 100), 6: (1, 50), 7: (1, 10)}
    try:
        return bounds[demand_type]
    except KeyError:
        raise ValueError(f"Demand distribution out of range: {demand_type}") from None


def avg_route_size_bounds(avg_route_size: int) -> tuple[float, float]:
    bounds = {
        1: (3.0, 5.0),
        2: (5.0, 8.0),
        3: (8.0, 12.0),
        4: (12.0, 16.0),
        5: (16.0, 25.0),
        6: (25.0, 50.0),
        7: (50.0, 200.0),
    }
    try:
        return bounds[avg_route_size]
    except KeyError:
        raise ValueError(f"Average route size out of range: {avg_route_size}") from None


def generate_demands(
    rng: random.Random,
    customer_ll: list[tuple[float, float]],
    demand_type: int,
    avg_route_size: int,
) -> tuple[list[int], int, int, float]:
    """Per-customer demands; returns (demands, total, max, target route size r)."""
    n = len(customer_ll)
    if n < 1:
        raise ValueError("At least one customer is required")

    rlo, rhi = avg_route_size_bounds(avg_route_size)
    r = rng.random() * (rhi - rlo) + rlo

    if demand_type == 1:
        demands = [1] * n
        return demands, n, 1, r

    lo, hi = demand_distribution_bounds(demand_type)
    lat_center = sum(p[0] for p in customer_ll) / n
    lon_center = sum(p[1] for p in customer_ll) / n

    demands = []
    for i in range(n):
        d = rng.randint(lo, hi)
        if demand_type == 6:
            lat, lon = customer_ll[i]
            same_diagonal = (lat < lat_center and lon < lon_center) or (lat >= lat_center and lon >= lon_center)
            d = rng.randint(51, 100) if same_diagonal else rng.randint(1, 50)
        elif demand_type == 7:
            # Julia iterates 1-based: the first ceil-ish 1.5*n/r customers in
            # POSITION get bulky demands, the rest small ones (then shuffled).
            if (i + 1) < (n / r) * 1.5:
                d = rng.randint(50, 100)
            else:
                d = rng.randint(1, 10)
        demands.append(d)

    if demand_type != 6:
        rng.shuffle(demands)

    return demands, sum(demands), max(demands), r


def capacity_from_avg_route_size(r: float, demands: list[int]) -> int:
    total = sum(demands)
    max_demand = max(demands) if demands else 0
    if total == len(demands):
        candidate = math.floor(r)
    else:
        candidate = max(max_demand, math.ceil(r * total / len(demands)))
    if len(demands) < 2:
        return max(max_demand, candidate)
    return min(max(candidate, max_demand), total - 1)
