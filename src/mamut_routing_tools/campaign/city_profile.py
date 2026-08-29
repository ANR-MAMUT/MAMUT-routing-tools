"""How far each city's road network departs from the Euclidean plane.

The metric study asks whether solvers degrade when the Euclidean assumption
breaks. That assumption breaks by different amounts in different cities: a flat
grid city barely bends straight lines, while a city cut by a river, a harbour or
a mountain bends them a lot. If the benchmark is drawn without regard for this,
the study's independent variable is left to chance.

So each city is profiled once, cheaply, before any instance is generated:
a few hundred random node pairs, their road distances against their straight
line distances. The resulting :class:`CityProfile` carries a distortion
stratum, and the campaign requires the selected set to span all three.

This is an *estimate over the whole city*, not a property of any instance. The
exact per-instance figure is :func:`~.descriptors.compute_divergence`, computed
after generation over the instance's own node set.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.stats import kendalltau

from mamut_routing_tools.generation.matrices import metric_csr
from mamut_routing_tools.roadgraph.build import load_road_graph

#: Dijkstra runs per city. Each one is a full single-source pass over the city
#: graph, so this is the term that decides the profiling cost.
DEFAULT_SOURCES = 16
#: Targets sampled from each source's distance row.
DEFAULT_TARGETS = 256

STRATA = ("low", "mid", "high")


@dataclass(frozen=True)
class CityProfile:
    """One city's road-network distortion, estimated over random node pairs."""

    city: str
    osm_file: str
    num_vertices: int
    num_edges: int
    #: mean and 90th percentile of road distance / straight-line distance.
    detour_mean: float
    detour_p90: float
    #: Kendall tau between the straight-line and free-flow-time orderings of the
    #: sampled pairs. Low tau = a solver's Euclidean sense of "nearer" is often
    #: wrong here.
    rank_tau_eucl_fast: float
    #: sampled pairs behind the figures above
    num_pairs: int
    #: "low" / "mid" / "high", assigned by tercile across a whole city set.
    #: ``None`` until :func:`assign_strata` has seen every city.
    distortion_stratum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CityProfile":
        return cls(**payload)


def profile_city(
    osm_path: str | Path,
    city: str,
    *,
    num_sources: int = DEFAULT_SOURCES,
    num_targets: int = DEFAULT_TARGETS,
    seed: int = 0,
) -> CityProfile:
    """Profile one city from its OSM extract.

    Runs ``num_sources`` single-source Dijkstras per metric and samples
    ``num_targets`` reachable targets from each, rather than an all-pairs pass:
    a few thousand pairs already pin the mean detour to well within the spread
    between cities, and an all-pairs pass on a 200k-vertex graph is not a
    "cheap pre-pass" by any reading.
    """
    graph = load_road_graph(osm_path)
    rng = np.random.default_rng(seed)

    vertex_count = graph.vertex_count
    sources = rng.choice(vertex_count, size=min(num_sources, vertex_count), replace=False)
    targets = rng.choice(vertex_count, size=min(num_targets, vertex_count), replace=False)

    road_short = dijkstra(metric_csr(graph, "shortest"), directed=True, indices=sources)[:, targets]
    road_fast = dijkstra(metric_csr(graph, "fastest"), directed=True, indices=sources)[:, targets]

    enu = np.asarray([graph.node_enu[graph.node_of[v]][:2] for v in range(vertex_count)], dtype=float)
    straight = np.linalg.norm(enu[sources][:, None, :] - enu[targets][None, :, :], axis=2)

    # A pair is usable when both metrics reach it and the two nodes are not the
    # same point; unreachable pairs would otherwise poison every statistic.
    usable = np.isfinite(road_short) & np.isfinite(road_fast) & (straight > 0.0)
    if not usable.any():
        raise ValueError(f"{city}: no reachable sampled node pair on the road graph")

    detour = road_short[usable] / straight[usable]
    tau = kendalltau(straight[usable], road_fast[usable]).correlation

    return CityProfile(
        city=city,
        osm_file=Path(osm_path).name,
        num_vertices=vertex_count,
        num_edges=graph.edge_count,
        detour_mean=float(detour.mean()),
        detour_p90=float(np.percentile(detour, 90)),
        rank_tau_eucl_fast=float(tau),
        num_pairs=int(usable.sum()),
    )


def assign_strata(profiles: Sequence[CityProfile]) -> list[CityProfile]:
    """Label each profile low / mid / high by tercile of ``detour_mean``.

    Terciles of the observed set, not fixed thresholds: what counts as a
    high-distortion city is only meaningful relative to the cities on hand, and
    absolute cut-offs would silently empty a stratum on a different city list.
    """
    if not profiles:
        return []
    means = np.asarray([p.detour_mean for p in profiles], dtype=float)
    low_cut, high_cut = np.quantile(means, [1 / 3, 2 / 3])

    labelled: list[CityProfile] = []
    for profile in profiles:
        if profile.detour_mean <= low_cut:
            stratum = "low"
        elif profile.detour_mean <= high_cut:
            stratum = "mid"
        else:
            stratum = "high"
        labelled.append(CityProfile(**{**profile.to_dict(), "distortion_stratum": stratum}))
    return labelled


def _profile_one(task: tuple[str, str, int, int, int]) -> CityProfile:
    """Worker entry point: ``(city, osm_path, num_sources, num_targets, seed)``."""
    city, osm_path, num_sources, num_targets, seed = task
    return profile_city(
        osm_path, city, num_sources=num_sources, num_targets=num_targets, seed=seed
    )


def profile_cities(
    cities: Iterable[tuple[str, Path]],
    *,
    num_sources: int = DEFAULT_SOURCES,
    num_targets: int = DEFAULT_TARGETS,
    seed: int = 0,
    jobs: int = 1,
    on_progress: Callable[[str], None] | None = None,
) -> list[CityProfile]:
    """Profile every city, then assign the distortion strata across the set.

    Each city is an independent OSM parse plus a handful of Dijkstras, so this
    parallelizes perfectly; a worker holds one city's road graph at a time.
    The strata are assigned afterwards over the whole set, which is why the
    results are gathered before anything is labelled.
    """
    tasks = [
        (city, str(osm_path), num_sources, num_targets, seed) for city, osm_path in cities
    ]
    if not tasks:
        return []

    profiles: list[CityProfile] = []
    if jobs <= 1:
        for task in tasks:
            if on_progress is not None:
                on_progress(task[0])
            profiles.append(_profile_one(task))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            pending = {pool.submit(_profile_one, task): task[0] for task in tasks}
            for future in as_completed(pending):
                profile = future.result()
                if on_progress is not None:
                    on_progress(pending[future])
                profiles.append(profile)
        # Deterministic order regardless of which worker finished first.
        profiles.sort(key=lambda profile: profile.city)
    return assign_strata(profiles)


def save_profiles(profiles: Sequence[CityProfile], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "mamut-city-profiles", "version": 1, "cities": [p.to_dict() for p in profiles]}
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_profiles(path: str | Path) -> list[CityProfile]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [CityProfile.from_dict(record) for record in payload["cities"]]
