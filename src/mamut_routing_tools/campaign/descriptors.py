"""Descriptive features of a CVRP instance.

Everywhere else in this project "metric" means an *arc-cost metric variant*
(euclidean / shortest / fastest). This module is the other kind: the statistics
that say what an instance *is like* -- how clustered its customers are, how
uneven its demands are, how far off-centre its depot sits, how many customers a
vehicle can carry. They exist so a campaign can *choose* instances that span the
feature space instead of hoping a parameter grid happens to.

Two blocks, split by what they cost:

:class:`InstanceDescriptors` needs only coordinates, demands and capacity, so it
can be computed for a large candidate pool before committing to any shortest
path computation. That is what makes over-generate-then-select affordable.

:class:`MetricDivergence` needs the three materialized arc-cost matrices and is
therefore computed after generation. It measures how far an instance departs
from the Euclidean assumption -- the independent variable of the metric study.

Degenerate inputs (collinear customers, a constant demand vector) leave some
features undefined; those fields come back as NaN rather than a guessed value,
and the selector drops non-finite candidates.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.stats import kendalltau

#: Neighbours used for the spatial weights of the demand autocorrelation.
MORAN_NEIGHBOURS = 8

#: Off-diagonal entries sampled when a rank correlation would otherwise run over
#: a full n x n matrix. At n=1000 that is already 10^6 pairs per metric pair.
MAX_RANK_PAIRS = 200_000

#: Rows per chunk when averaging pairwise distances. The naive expression
#: ``points[:, None, :] - customers[None, :, :]`` is (n+1) x n x 2 floats --
#: 400 MB at n=5000 -- for a result that is one mean per row.
DISTANCE_ROW_CHUNK = 512


@dataclass(frozen=True)
class InstanceDescriptors:
    """What an instance looks like, from coordinates and demands alone."""

    n: int
    #: sqrt of the customer convex-hull area, in metres. The one feature that
    #: carries scale; reported for context, never selected on (selecting on it
    #: would just re-select city size).
    extent_m: float

    # --- spatial, all scale-free -------------------------------------------
    #: Clark-Evans aggregation index: mean nearest-neighbour distance over the
    #: value expected under complete spatial randomness. < 1 clustered,
    #: ~ 1 random, > 1 regular. The continuous version of Solomon's C/R/RC.
    clark_evans_r: float
    #: Coefficient of variation of the nearest-neighbour distances. Separates
    #: "a few tight clumps in empty space" from "evenly dense".
    nnd_cv: float
    #: std / mean of the customer distances to the customer centroid.
    radial_dispersion: float
    #: Depot offset from the customer centroid, in units of the mean customer
    #: radius. 0 = dead centre; ~1 = as far out as a typical customer.
    depot_centrality: float
    #: Rank of the depot's mean distance-to-customers among all nodes, in
    #: [0, 1]. 0 = the most central node of the instance, 1 = the most remote.
    depot_eccentricity: float

    # --- demand -------------------------------------------------------------
    demand_cv: float
    demand_gini: float
    demand_max_over_mean: float
    #: Moran's I of demand over a k-nearest-neighbour graph: is a big demand
    #: more likely to sit next to another big demand? Near 0 when demands are
    #: drawn independently of position, clearly positive for the spatially
    #: correlated demand distribution (``demand_type=6``).
    demand_moran_i: float

    # --- capacity and route structure ---------------------------------------
    #: ceil(total demand / capacity) -- the bin-packing lower bound on the
    #: fleet, and the same quantity the artifacts store as num_vehicles_lb.
    lb_cap: int
    #: n / lb_cap: the *realized* average route size, as opposed to the band
    #: the generator was asked to target.
    route_size: float
    #: 1 - total demand / (lb_cap * capacity). Near 0 means the lower bound is
    #: only reachable by a near-perfect packing; large means capacity is slack.
    capacity_slack: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricDivergence:
    """How far an instance's road metrics depart from its Euclidean one."""

    #: mean and 90th percentile of shortest-road / euclidean over node pairs.
    #: 1.0 would be an obstacle-free plane; real cities sit well above it.
    detour_mean: float
    detour_p90: float
    #: mean relative gap between opposite directions of the same pair. Zero for
    #: a Euclidean matrix by construction; positive wherever one-ways bite.
    asymmetry_shortest: float
    asymmetry_fastest: float
    #: Kendall rank correlations between the metrics' orderings of node pairs.
    #: ``rank_tau_eucl_fast`` is the study's dose variable: how badly a solver's
    #: Euclidean intuition about "which customer is nearer" is violated.
    rank_tau_eucl_short: float
    rank_tau_eucl_fast: float
    rank_tau_short_fast: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Feature names the selector spreads over. ``extent_m`` and ``lb_cap`` are
#: deliberately absent: the first is pure scale, the second is ``n / route_size``
#: and would double-count the capacity axis.
#: Selection features measured as ratios rather than as differences, and so
#: z-scored in logs -- see :func:`selection_features`. ``capacity_slack`` and the
#: correlation-shaped features are deliberately absent: they are already bounded,
#: and several can be zero or negative.
LOG_SCALED_FEATURES: frozenset[str] = frozenset({"route_size", "demand_max_over_mean"})

SELECTION_FEATURES: tuple[str, ...] = (
    "clark_evans_r",
    "nnd_cv",
    "radial_dispersion",
    "depot_centrality",
    "depot_eccentricity",
    "demand_cv",
    "demand_gini",
    "demand_max_over_mean",
    "demand_moran_i",
    "route_size",
    "capacity_slack",
)


def selection_features(descriptors: InstanceDescriptors) -> list[float]:
    """Where this instance sits in feature space, for the selector.

    Not the same thing as the descriptors themselves. A statistic can be
    formally undefined while the instance still has a perfectly definite
    position on that axis, and conflating the two silently deletes instances:

    ``demand_moran_i`` is undefined when every demand is equal, because there is
    no variance to correlate. But "are big demands next to big demands?" has an
    unambiguous answer for a constant field -- no, there is no such pattern --
    and 0 is exactly where such an instance belongs on the autocorrelation axis.
    Leaving it NaN made every unit-demand instance unusable, which quietly
    removed one of the seven demand distributions from the whole benchmark.

    Only that case is imputed. A NaN from a degenerate *geometry*
    (``clark_evans_r`` on collinear customers) is a genuinely unusable instance
    and stays NaN, so :meth:`EvaluatedCandidate.is_usable` still rejects it.

    Two features are taken in logs, because they are ratios spanning an order of
    magnitude and the selector z-scores whatever it is handed. On a linear scale
    the step from 50 to 200 customers per route counts eleven times the step from
    3 to 16, so a max-min spread objective buys most of its score at one end of
    the axis and parks instances there. In v2 that put 39 of 100 instances in the
    longest route-size band against a quota floor of 7 -- the selector was not
    ignoring the axis, it was being paid to sit on its edge. A log scale prices
    the two ends the same way a reader does, in ratios.
    """
    return [
        math.log(value) if name in LOG_SCALED_FEATURES and value > 0 else value
        for name, value in zip(SELECTION_FEATURES, descriptor_values(descriptors))
    ]


def descriptor_values(descriptors: InstanceDescriptors) -> list[float]:
    """The selection descriptors in their own units, imputation applied.

    Split from :func:`selection_features` because the two answer different
    questions and confusing them misreports the collection. The selector wants
    the transformed space; a *reader* wants customers per route to be a number
    of customers. Reporting the transformed value under the descriptor's own
    name published a route size of "1.097" for a set whose shortest routes hold
    three customers.
    """
    values: list[float] = []
    for name in SELECTION_FEATURES:
        value = getattr(descriptors, name)
        if (
            name == "demand_moran_i"
            and math.isnan(value)
            and descriptors.demand_cv == 0.0
        ):
            value = 0.0
        values.append(float(value))
    return values


def _point_area(points: np.ndarray) -> float:
    """Convex-hull area of a point set, falling back to its bounding box.

    Qhull refuses degenerate input (fewer than three points, or collinear ones).
    A bounding box is the honest answer there; when that is flat too, the point
    set has no area at all and the caller gets 0.0.
    """
    if len(points) >= 3:
        try:
            return float(ConvexHull(points).volume)  # 2-D "volume" is the area
        except QhullError:
            pass
    spans = points.max(axis=0) - points.min(axis=0)
    return float(spans[0] * spans[1])


def _gini(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0.0:
        return float("nan")
    ordered = np.sort(values.astype(float))
    n = len(ordered)
    index = np.arange(1, n + 1, dtype=float)
    return float((2.0 * (index * ordered).sum()) / (n * total) - (n + 1.0) / n)


def _moran_i(points: np.ndarray, values: np.ndarray, neighbours: int) -> float:
    """Moran's I of ``values`` under binary k-nearest-neighbour weights."""
    n = len(points)
    k = min(neighbours, n - 1)
    if k < 1:
        return float("nan")
    deviation = values.astype(float) - values.mean()
    denominator = float((deviation**2).sum())
    if denominator <= 0.0:
        # A constant field has no variance to correlate; "no autocorrelation"
        # would be a fabricated answer, so report undefined.
        return float("nan")
    # k + 1 because the first neighbour of a point is always the point itself
    _, indices = cKDTree(points).query(points, k=k + 1)
    neighbour_indices = indices[:, 1:]
    numerator = float((deviation[:, None] * deviation[neighbour_indices]).sum())
    weight_total = float(n * k)
    return (n / weight_total) * (numerator / denominator)


def _mean_distance_to(sources: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Mean distance from every source to all targets, in row chunks."""
    means = np.empty(len(sources), dtype=float)
    for start in range(0, len(sources), DISTANCE_ROW_CHUNK):
        block = sources[start : start + DISTANCE_ROW_CHUNK]
        means[start : start + len(block)] = np.linalg.norm(
            block[:, None, :] - targets[None, :, :], axis=2
        ).mean(axis=1)
    return means


def compute_descriptors(
    coordinates: Sequence[Sequence[float]],
    demands: Sequence[int],
    capacity: int,
) -> InstanceDescriptors:
    """Descriptors of one instance. Index 0 is the depot, in both sequences."""
    points = np.asarray(coordinates, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("coordinates must be a sequence of (x, y) pairs")
    if len(points) != len(demands):
        raise ValueError("coordinates and demands must have the same length")
    if capacity <= 0:
        raise ValueError("capacity must be positive")

    depot = points[0]
    customers = points[1:]
    n = len(customers)
    if n < 2:
        raise ValueError("at least 2 customers are required")

    quantities = np.asarray(demands[1:], dtype=float)
    if (quantities < 0).any():
        raise ValueError("customer demands must be non-negative")

    # --- spatial -----------------------------------------------------------
    area = _point_area(customers)
    distances, _ = cKDTree(customers).query(customers, k=2)
    nnd = distances[:, 1]
    mean_nnd = float(nnd.mean())
    if area > 0.0:
        expected_nnd = 0.5 * math.sqrt(area / n)
        clark_evans_r = mean_nnd / expected_nnd if expected_nnd > 0.0 else float("nan")
    else:
        clark_evans_r = float("nan")
    nnd_cv = float(nnd.std() / mean_nnd) if mean_nnd > 0.0 else float("nan")

    centroid = customers.mean(axis=0)
    radii = np.linalg.norm(customers - centroid, axis=1)
    mean_radius = float(radii.mean())
    radial_dispersion = float(radii.std() / mean_radius) if mean_radius > 0.0 else float("nan")
    depot_offset = float(np.linalg.norm(depot - centroid))
    depot_centrality = depot_offset / mean_radius if mean_radius > 0.0 else float("nan")

    # Where the depot sits in the instance's own "how remote is this point"
    # ordering. Scale-free and shape-free, unlike a raw distance.
    mean_to_customers = _mean_distance_to(points, customers)
    depot_eccentricity = float((mean_to_customers < mean_to_customers[0]).sum()) / n

    # --- demand ------------------------------------------------------------
    mean_demand = float(quantities.mean())
    if mean_demand > 0.0:
        demand_cv = float(quantities.std() / mean_demand)
        demand_max_over_mean = float(quantities.max() / mean_demand)
    else:
        demand_cv = float("nan")
        demand_max_over_mean = float("nan")
    demand_gini = _gini(quantities)
    demand_moran_i = _moran_i(customers, quantities, MORAN_NEIGHBOURS)

    # --- capacity ----------------------------------------------------------
    total_demand = float(quantities.sum())
    lb_cap = int(math.ceil(total_demand / capacity)) if total_demand > 0 else 0
    route_size = n / lb_cap if lb_cap > 0 else float("nan")
    capacity_slack = 1.0 - total_demand / (lb_cap * capacity) if lb_cap > 0 else float("nan")

    return InstanceDescriptors(
        n=n,
        extent_m=math.sqrt(area),
        clark_evans_r=clark_evans_r,
        nnd_cv=nnd_cv,
        radial_dispersion=radial_dispersion,
        depot_centrality=depot_centrality,
        depot_eccentricity=depot_eccentricity,
        demand_cv=demand_cv,
        demand_gini=demand_gini,
        demand_max_over_mean=demand_max_over_mean,
        demand_moran_i=demand_moran_i,
        lb_cap=lb_cap,
        route_size=route_size,
        capacity_slack=capacity_slack,
    )


def _offdiagonal(matrix: np.ndarray) -> np.ndarray:
    return matrix[~np.eye(len(matrix), dtype=bool)]


def _sampled_tau(left: np.ndarray, right: np.ndarray, rng: np.random.Generator) -> float:
    """Kendall tau over at most :data:`MAX_RANK_PAIRS` of the given entries."""
    if len(left) > MAX_RANK_PAIRS:
        indices = rng.choice(len(left), size=MAX_RANK_PAIRS, replace=False)
        left, right = left[indices], right[indices]
    return float(kendalltau(left, right).correlation)


def _asymmetry(matrix: np.ndarray) -> float:
    """Mean relative gap between the two directions of each node pair."""
    upper = np.triu_indices(len(matrix), k=1)
    forward, backward = matrix[upper], matrix.T[upper]
    average = 0.5 * (forward + backward)
    usable = average > 0.0
    if not usable.any():
        return float("nan")
    return float((np.abs(forward - backward)[usable] / average[usable]).mean())


def compute_divergence(
    euclidean: Sequence[Sequence[float]],
    shortest: Sequence[Sequence[float]],
    fastest: Sequence[Sequence[float]],
    *,
    seed: int = 0,
) -> MetricDivergence:
    """How much the road metrics of one instance disagree with its Euclidean one.

    ``seed`` fixes the pair subsample used for the rank correlations, so the
    reported values are reproducible for a given instance.
    """
    d_eucl = np.asarray(euclidean, dtype=float)
    d_short = np.asarray(shortest, dtype=float)
    d_fast = np.asarray(fastest, dtype=float)
    if not (d_eucl.shape == d_short.shape == d_fast.shape):
        raise ValueError("the three matrices must have the same shape")
    if d_eucl.ndim != 2 or d_eucl.shape[0] != d_eucl.shape[1]:
        raise ValueError("matrices must be square")

    flat_eucl = _offdiagonal(d_eucl)
    flat_short = _offdiagonal(d_short)
    flat_fast = _offdiagonal(d_fast)

    positive = flat_eucl > 0.0
    if positive.any():
        detour = flat_short[positive] / flat_eucl[positive]
        detour_mean = float(detour.mean())
        detour_p90 = float(np.percentile(detour, 90))
    else:
        detour_mean = detour_p90 = float("nan")

    # One generator, drawn from in a fixed order: the three correlations use
    # different subsamples, and re-running with the same seed reproduces them.
    rng = np.random.default_rng(seed)
    return MetricDivergence(
        detour_mean=detour_mean,
        detour_p90=detour_p90,
        asymmetry_shortest=_asymmetry(d_short),
        asymmetry_fastest=_asymmetry(d_fast),
        rank_tau_eucl_short=_sampled_tau(flat_eucl, flat_short, rng),
        rank_tau_eucl_fast=_sampled_tau(flat_eucl, flat_fast, rng),
        rank_tau_short_fast=_sampled_tau(flat_short, flat_fast, rng),
    )
