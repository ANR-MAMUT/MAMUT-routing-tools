"""Static per-stop schedules for the website and the workbench route animation."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from mamut_routing_lib.exact_types import binary64_operand

TIMELINE_START_RULE = "depot_at_zero"


class TimelineError(ValueError):
    """The requested route clock cannot produce a finite display schedule."""


def _exact_time(value: int | float) -> Fraction:
    return Fraction.from_float(binary64_operand(value))


def build_static_timeline(
    routes: Sequence[Sequence[int]],
    arc_costs: Sequence[Sequence[float]],
    *,
    depot: int = 0,
    time_windows: Sequence[Sequence[float]] | None = None,
    service_times: Sequence[float] | None = None,
    decimals: int | None = None,
) -> dict:
    """Compute per-stop times from exact binary64 operands in the supplied route order.

    VRPTW routes leave at depot opening. A customer may start service anywhere in its closed window, and the return must meet depot closing. A capacity-only instance can use its fastest matrix as an illustrative clock starting at zero. Only traversed arcs are converted. Event times use nearest-binary64 display by default. Callers can request fewer decimal places for display.
    """
    def display(value: Fraction) -> float:
        try:
            projected = float(value)
            return projected if decimals is None else round(projected, decimals)
        except OverflowError as error:
            raise TimelineError("An event time exceeds the binary64 display range.") from error

    start_rule = TIMELINE_START_RULE
    depot_opening = Fraction(0)
    if time_windows is not None:
        start_rule = "depot_opening"
        depot_opening = _exact_time(time_windows[depot][0])
        depot_closing = _exact_time(time_windows[depot][1])
        if service_times is not None and _exact_time(service_times[depot]) != 0:
            raise TimelineError("Depot service time must be zero.")

    timeline_routes = []
    for route in routes:
        current_time = depot_opening
        previous = depot
        stops = []
        for customer in route:
            arrival = current_time + _exact_time(arc_costs[previous][customer])
            service_start = arrival
            if time_windows is not None:
                opening, closing = time_windows[customer]
                service_start = max(arrival, _exact_time(opening))
                if service_start > _exact_time(closing):
                    raise TimelineError(f"Time window violated for customer {customer} on route: {list(route)}")
            service = _exact_time(service_times[customer]) if service_times is not None else 0
            departure = service_start + service
            stops.append([int(customer), display(arrival), display(service_start), display(departure)])
            current_time = departure
            previous = customer
        return_arrival = current_time + _exact_time(arc_costs[previous][depot])
        if time_windows is not None and return_arrival > depot_closing:
            raise TimelineError(f"Time window violated when returning to depot on route: {list(route)}")
        timeline_routes.append({"departure": display(depot_opening), "return_arrival": display(return_arrival), "stops": stops})
    return {"time_unit": "s", "start_rule": start_rule, "routes": timeline_routes}
