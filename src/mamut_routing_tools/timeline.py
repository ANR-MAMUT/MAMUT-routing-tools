"""Per-stop schedule of a solution: what the route animation plays back.

``build_static_timeline`` replays the canonical checker's time accounting
(``mamut_routing_lib.checker._check_common_route_constraints``) operation for
operation, so an animated vehicle is where the checker says it is:

- every route leaves the depot at t = 0;
- arrival = previous departure + arc cost (the arc cost *is* the travel time);
- a vehicle early for a time window waits until the window opens;
- departure = service start + service time;
- the return is clamped to the depot's ready time and must meet its due date.

Without time windows (CVRP) the arc costs still clock the route: with a
``fastest`` matrix that is free-flow travel time, the only time the published
data defines for a capacity-only instance.

The result is plain JSON, shared by the website's ``timeline.<Objective>.json``
sidecar and the tools GUI ``/timeline`` endpoint::

    {"time_unit": "s", "start_rule": "depot_at_zero",
     "routes": [{"departure": 0, "return_arrival": 4210.5,
                 "stops": [[node, arrival, service_start, departure], ...]}]}
"""

from __future__ import annotations

from collections.abc import Sequence

TIMELINE_START_RULE = "depot_at_zero"


class TimelineError(ValueError):
    """The routes are not time-feasible under the checker's accounting."""


def build_static_timeline(
    routes: Sequence[Sequence[int]],
    arc_costs: Sequence[Sequence[float]],
    *,
    depot: int = 0,
    time_windows: Sequence[Sequence[float]] | None = None,
    service_times: Sequence[float] | None = None,
    decimals: int = 3,
) -> dict:
    """Arrival, service start and departure of every stop, as the checker counts them.

    ``routes`` hold customer indices without the depot, as BKS files store them.
    Times are accumulated exactly (left to right, like the checker) and only
    rounded to ``decimals`` for the output. Raises :class:`TimelineError` when a
    time window is missed, which the checker would reject as well.
    """

    def rounded(value: float) -> float:
        return round(float(value), decimals)

    timeline_routes = []
    for route in routes:
        current_time: float = 0
        previous = depot
        stops = []
        for customer in route:
            arrival = current_time + arc_costs[previous][customer]
            service_start = arrival
            if time_windows is not None:
                ready_time, due_date = time_windows[customer]
                if service_start < ready_time:
                    service_start = ready_time
                if service_start > due_date:
                    raise TimelineError(f"Time window violated for customer {customer} on route: {list(route)}")
            service = service_times[customer] if service_times is not None else 0
            departure = service_start + service
            stops.append([int(customer), rounded(arrival), rounded(service_start), rounded(departure)])
            current_time = departure
            previous = customer
        return_arrival = current_time + arc_costs[previous][depot]
        if time_windows is not None:
            depot_ready_time, depot_due_date = time_windows[depot]
            if return_arrival < depot_ready_time:
                return_arrival = depot_ready_time
            if return_arrival > depot_due_date:
                raise TimelineError(f"Time window violated when returning to depot on route: {list(route)}")
        timeline_routes.append({"departure": 0, "return_arrival": rounded(return_arrival), "stops": stops})
    return {"time_unit": "s", "start_rule": TIMELINE_START_RULE, "routes": timeline_routes}
