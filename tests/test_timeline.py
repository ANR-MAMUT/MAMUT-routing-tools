from __future__ import annotations

import random

import pytest

from mamut_routing_lib.checker import SolutionCheckStatus, check_solution
from mamut_routing_lib.models import BenchmarkInstance, BenchmarkSolution
from mamut_routing_tools.timeline import TimelineError, build_static_timeline

ARC_COSTS = [
    [0, 5, 6],
    [5, 0, 3],
    [6, 3, 0],
]


def test_vehicle_waits_for_the_window_then_serves() -> None:
    timeline = build_static_timeline(
        [[1, 2]],
        ARC_COSTS,
        time_windows=[(0, 100), (10, 50), (0, 50)],
        service_times=[0, 2, 2],
    )
    assert timeline == {
        "time_unit": "s",
        "start_rule": "depot_at_zero",
        "routes": [
            {
                "departure": 0,
                "return_arrival": 23.0,
                # Customer 1: arrives at 5, waits until 10, leaves at 12.
                # Customer 2: arrives at 15 and leaves at 17; 6 more back to the depot.
                "stops": [[1, 5.0, 10.0, 12.0], [2, 15.0, 15.0, 17.0]],
            }
        ],
    }


def test_capacity_only_routes_are_clocked_by_the_arc_costs() -> None:
    timeline = build_static_timeline([[2, 1], [1]], ARC_COSTS)
    assert timeline["routes"] == [
        {"departure": 0, "return_arrival": 14.0, "stops": [[2, 6.0, 6.0, 6.0], [1, 9.0, 9.0, 9.0]]},
        {"departure": 0, "return_arrival": 10.0, "stops": [[1, 5.0, 5.0, 5.0]]},
    ]


def test_return_is_clamped_to_the_depot_ready_time_like_the_checker() -> None:
    timeline = build_static_timeline(
        [[1]], ARC_COSTS, time_windows=[(30, 100), (0, 50), (0, 50)], service_times=[0, 0, 0]
    )
    assert timeline["routes"][0]["return_arrival"] == 30.0


@pytest.mark.parametrize(
    "time_windows",
    [
        [(0, 100), (0, 50), (0, 9)],  # customer 2 reached at 10
        [(0, 15), (0, 50), (0, 50)],  # back at the depot at 18
    ],
)
def test_a_missed_window_raises(time_windows: list[tuple[int, int]]) -> None:
    with pytest.raises(TimelineError):
        build_static_timeline([[1, 2]], ARC_COSTS, time_windows=time_windows, service_times=[0, 2, 2])


def _random_instance(rng: random.Random, n: int, slack: int, tighten: bool) -> tuple[BenchmarkInstance, list[list[int]]]:
    points = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n + 1)]
    arc_costs = [[0 if i == j else round(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5) + 1 for j, b in enumerate(points)] for i, a in enumerate(points)]
    service_times = [0] + [rng.randint(0, 10) for _ in range(n)]
    customers = list(range(1, n + 1))
    rng.shuffle(customers)
    routes = [customers[: n // 2], customers[n // 2 :]]
    # Windows around a slow schedule, so some vehicles arrive early and wait.
    time_windows: list[tuple[int, int]] = [(0, 0)] * (n + 1)
    horizon = 0
    for route in routes:
        current, previous = 0, 0
        for customer in route:
            current += arc_costs[previous][customer] + rng.randint(0, 15)
            time_windows[customer] = (current, current + slack)
            current += service_times[customer]
            previous = customer
        horizon = max(horizon, current + arc_costs[previous][0] + slack)
    time_windows[0] = (0, horizon)
    if tighten:
        victim = routes[0][-1]
        ready, _ = time_windows[victim]
        time_windows[victim] = (max(0, ready - 40), max(0, ready - 30))
    instance = BenchmarkInstance(
        instance_name="poryos-n8-timeline",
        instance_origin="OsmCvrpGen",
        benchmark_name="Poryos2026",
        num_customers=n,
        vehicle_capacity=10_000,
        coordinates=points,
        demands=[0] + [1] * n,
        depot=0,
        arc_costs=arc_costs,
        service_times=service_times,
        time_windows=time_windows,
    )
    return instance, routes


@pytest.mark.parametrize("seed", range(40))
def test_timeline_agrees_with_the_checker(seed: int) -> None:
    rng = random.Random(seed)
    instance, routes = _random_instance(rng, n=8, slack=rng.choice([0, 5, 30]), tighten=seed % 4 == 0)
    verdict = check_solution(instance, BenchmarkSolution(instance_name=instance.instance_name, routes=routes, cost=None))
    try:
        timeline = build_static_timeline(
            routes,
            instance.arc_costs,
            depot=instance.depot,
            time_windows=instance.time_windows,
            service_times=instance.service_times,
        )
    except TimelineError:
        assert verdict.status == SolutionCheckStatus.TIME_WINDOW_VIOLATED
        return
    assert verdict.is_valid(), verdict
    # Travel between consecutive events is exactly the checker's routing cost.
    travel = 0.0
    for route in timeline["routes"]:
        previous_departure = route["departure"]
        for _, arrival, service_start, departure in route["stops"]:
            travel += arrival - previous_departure
            assert arrival <= service_start <= departure
            previous_departure = departure
        travel += route["return_arrival"] - previous_departure
    assert travel == pytest.approx(verdict.routing_cost)
