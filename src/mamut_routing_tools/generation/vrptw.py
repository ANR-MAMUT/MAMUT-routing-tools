"""VRPTW derivation from generated CVRP instances (port of the workbench
route_centered / reachable_interval time-window synthesis).

Both methods persist capacity-, horizon- (and window-) feasible *anchor
routes* in the derivation manifest; ``derive-td`` certifies the windows
along them under traffic. ``route_centered`` centres each window on the
customer's free-flow visit time along those routes (the family's
``construct_anchor_routes``); before 0.6.0 it used a single capacity-agnostic
nearest-neighbour tour, which ran past the horizon beyond about 80
customers and left most windows equal to the whole feasible interval.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from mamut_routing_tools.family.tw_synthesis import (
    construct_anchor_routes,
    construct_window_anchor_routes,
    validate_static_anchor,
)
from mamut_routing_tools.generation.writers import ParsedCvrpInstance, parse_cvrp_vrp, write_json

TW_METHODS = ("route_centered", "reachable_interval")
#: ``derivation.anchor_policy`` values: how the persisted anchor routes were built.
ANCHOR_POLICIES = {
    "route_centered": "capacity-horizon-nearest-neighbour",
    "reachable_interval": "capacity-window-horizon-nearest-neighbour",
}
DEFAULT_TW_METHOD = "route_centered"
HORIZON_START = 0
HORIZON_END = 86400


def stable_seed(*parts: Any) -> int:
    """Deterministic cross-run seed from arbitrary parts (replaces Julia's
    ``hash(tuple)``; values differ from Julia by design, determinism within
    Python is the contract)."""
    text = "".join(repr(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def nearest_neighbour_route(travel_times: list[list[int]], depot: int = 0) -> list[int]:
    n = len(travel_times)
    if n < 1:
        raise ValueError("Travel-time matrix is empty")
    visited = [False] * n
    visited[depot] = True
    route = [depot]
    current = depot
    while len(route) < n:
        best = -1
        best_cost = None
        for j in range(n):
            if visited[j] or j == current:
                continue
            cost = travel_times[current][j]
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best = j
        if best < 0:
            break
        route.append(best)
        visited[best] = True
        current = best
    return route


def simulate_arrival_times(
    route: list[int],
    travel_times: list[list[int]],
    service_times: list[int],
    horizon_start: int = HORIZON_START,
) -> list[int]:
    n = len(travel_times)
    arrivals = [horizon_start] * n
    if not route:
        return arrivals
    t = horizon_start
    prev = route[0]
    arrivals[prev] = t
    for cur in route[1:]:
        t += travel_times[prev][cur]
        arrivals[cur] = t
        t += service_times[cur]
        prev = cur
    return arrivals


def repair_time_window(
    tw_start: int,
    tw_end: int,
    travel_to: int,
    travel_back: int,
    service: int,
    horizon_start: int,
    horizon_end: int,
) -> tuple[int, int]:
    earliest_feasible = horizon_start + travel_to
    latest_feasible = horizon_end - service - travel_back
    if latest_feasible < earliest_feasible:
        clamped = min(max((earliest_feasible + latest_feasible) // 2, horizon_start), horizon_end)
        return clamped, clamped
    e = max(tw_start, earliest_feasible)
    latest = min(tw_end, latest_feasible)
    if e > latest:
        return earliest_feasible, latest_feasible
    return e, latest


def generate_service_times(
    rng: random.Random,
    n_total: int,
    horizon_start: int,
    horizon_end: int,
    *,
    mean_ratio_target: float = 0.01,
    mean_ratio_std: float = 0.005,
    depot: int = 0,
) -> tuple[list[int], float]:
    horizon = float(horizon_end - horizon_start)
    mean_ratio = _clamp(rng.gauss(0.0, 1.0) * mean_ratio_std + mean_ratio_target, 0.001, 0.2)
    mean_service = horizon * mean_ratio
    upper = max(1, int(mean_service * 2))
    service_times = [0] * n_total
    for i in range(n_total):
        if i == depot:
            continue
        sampled = rng.gauss(0.0, 1.0) * (mean_service / 2.0) + mean_service
        service_times[i] = min(max(round(sampled), 1), upper)
    return service_times, mean_ratio


def generate_tw_route_centered(
    rng: random.Random,
    travel_times: list[list[int]],
    service_times: list[int],
    horizon_start: int,
    horizon_end: int,
    *,
    demands: list[int],
    capacity: int,
    depot: int = 0,
    width_ratio_mean: float = 0.2,
    width_ratio_std: float = 0.08,
) -> tuple[list[tuple[int, int]], list[list[int]], float]:
    """Windows centred on free-flow visits along capacity-and-horizon-feasible anchor routes.

    Returns ``(time_windows, anchor_routes, width_ratio_mean)``. The rng draws
    (one width per customer, in index order) are unchanged, so a seed still
    gives the same widths; only the centres moved to the feasible routes.
    """
    if depot != 0:
        raise ValueError(f"route_centered windows need the depot at node 0, got {depot}")
    n = len(travel_times)
    horizon = float(horizon_end - horizon_start)
    anchor_routes, arrivals = construct_anchor_routes(
        travel_times, demands, capacity, service_times, horizon_start=horizon_start, horizon_end=horizon_end
    )

    time_windows: list[tuple[int, int]] = [(0, 0)] * n
    time_windows[depot] = (horizon_start, horizon_end)
    for i in range(n):
        if i == depot:
            continue
        center = arrivals[i]
        width_ratio = _clamp(rng.gauss(0.0, 1.0) * width_ratio_std + width_ratio_mean, 0.01, 1.0)
        width = max(1, round(horizon * width_ratio))
        e = round(center - width / 2)
        latest = round(center + width / 2)
        time_windows[i] = repair_time_window(
            e, latest, travel_times[depot][i], travel_times[i][depot], service_times[i], horizon_start, horizon_end
        )
    return time_windows, anchor_routes, width_ratio_mean


def generate_tw_reachable_interval(
    rng: random.Random,
    travel_times: list[list[int]],
    service_times: list[int],
    horizon_start: int,
    horizon_end: int,
    *,
    depot: int = 0,
    width_ratio_mean: float = 0.5,
    width_ratio_std: float = 0.2,
) -> tuple[list[tuple[int, int]], float]:
    n = len(travel_times)
    horizon = float(horizon_end - horizon_start)
    time_windows: list[tuple[int, int]] = [(0, 0)] * n
    time_windows[depot] = (horizon_start, horizon_end)
    for i in range(n):
        if i == depot:
            continue
        travel_to = travel_times[depot][i]
        travel_back = travel_times[i][depot]
        service_i = service_times[i]
        earliest = horizon_start + travel_to
        latest = horizon_end - service_i - travel_back
        width_ratio = _clamp(rng.gauss(0.0, 1.0) * width_ratio_std + width_ratio_mean, 0.01, 1.0)
        width = max(1, round(horizon * width_ratio))
        if latest < earliest:
            clamped = min(max(earliest, horizon_start), horizon_end)
            time_windows[i] = (clamped, clamped)
            continue
        center_low = earliest + width // 2
        center_high = latest - width // 2
        if center_low > center_high:
            center = (earliest + latest) // 2
        else:
            center = rng.randint(center_low, center_high)
        e = center - width // 2
        time_windows[i] = repair_time_window(
            e, e + width, travel_to, travel_back, service_i, horizon_start, horizon_end
        )
    return time_windows, width_ratio_mean


def generate_vrptw_fields(
    seed_parts: tuple,
    travel_times: list[list[int]],
    horizon_start: int,
    horizon_end: int,
    tw_method: str,
    *,
    demands: list[int],
    capacity: int,
    depot: int = 0,
) -> tuple[list[int], list[tuple[int, int]], dict[str, Any]]:
    rng = random.Random(stable_seed(*seed_parts))
    n = len(travel_times)
    service_times, mean_service_ratio = generate_service_times(rng, n, horizon_start, horizon_end, depot=depot)

    method = tw_method.strip().lower()
    if method not in TW_METHODS:
        raise ValueError(f"Unsupported TW method '{tw_method}'. Use one of: {', '.join(TW_METHODS)}.")

    if method == "route_centered":
        time_windows, anchor_routes, width_ratio_mean = generate_tw_route_centered(
            rng,
            travel_times,
            service_times,
            horizon_start,
            horizon_end,
            demands=demands,
            capacity=capacity,
            depot=depot,
        )
    else:
        time_windows, width_ratio_mean = generate_tw_reachable_interval(
            rng, travel_times, service_times, horizon_start, horizon_end, depot=depot
        )
        anchor_routes = None

    repaired_count = 0
    for i in range(n):
        if i == depot:
            continue
        e_in, l_in = time_windows[i]
        e_out, l_out = repair_time_window(
            e_in, l_in, travel_times[depot][i], travel_times[i][depot], service_times[i], horizon_start, horizon_end
        )
        if (e_out, l_out) != (e_in, l_in):
            repaired_count += 1
        time_windows[i] = (e_out, l_out)

    if anchor_routes is None:
        anchor_routes = construct_window_anchor_routes(
            travel_times,
            demands,
            capacity,
            service_times,
            time_windows,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
        )
    validate_static_anchor(
        anchor_routes,
        travel_times,
        demands,
        capacity,
        service_times,
        time_windows,
        horizon_start=horizon_start,
        horizon_end=horizon_end,
    )

    stochastic_params: dict[str, Any] = {
        "tw_method": method,
        "horizon_start": horizon_start,
        "horizon_end": horizon_end,
        "mean_service_time_horizon_ratio": mean_service_ratio,
        "time_window_ratio": width_ratio_mean,
        "tw_repaired_count": repaired_count,
        # Feasible anchor routes (capacity, horizon and the final windows),
        # persisted so a downstream TD derivation lifts the windows along
        # them under traffic without recomputing them.
        "anchor_policy": ANCHOR_POLICIES[method],
        "anchor_routes": anchor_routes,
    }
    return service_times, time_windows, stochastic_params


def write_cvrptw_vrp(
    path: str | Path,
    parsed: ParsedCvrpInstance,
    instance_name: str,
    comment: str,
    service_times: list[int],
    time_windows: list[tuple[int, int]],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        cvrptw_text(parsed, instance_name, comment, service_times, time_windows), encoding="utf-8"
    )


def cvrptw_text(
    parsed: ParsedCvrpInstance,
    instance_name: str,
    comment: str,
    service_times: list[int],
    time_windows: list[tuple[int, int]],
) -> str:
    """The CVRPTW ``.vrp`` text :func:`write_cvrptw_vrp` writes."""
    lines = [f"NAME : {instance_name}", "TYPE : CVRPTW"]
    if comment:
        lines.append(f"COMMENT : {comment}")
    lines.extend(
        [
            f"DIMENSION : {parsed.dimension}",
            f"CAPACITY : {parsed.capacity}",
            "EDGE_WEIGHT_TYPE : EXPLICIT",
            "EDGE_WEIGHT_FORMAT : FULL_MATRIX",
            "EDGE_WEIGHT_SECTION",
        ]
    )
    lines.extend(" ".join(str(value) for value in row) for row in parsed.arc_costs)
    lines.append("NODE_COORD_SECTION")
    lines.extend(f"{i + 1} {x} {y}" for i, (x, y) in enumerate(parsed.coordinates))
    lines.append("DEMAND_SECTION")
    lines.extend(f"{i + 1} {demand}" for i, demand in enumerate(parsed.demands))
    lines.append("TIME_WINDOW_SECTION")
    lines.extend(f"{i + 1} {ready} {due}" for i, (ready, due) in enumerate(time_windows))
    lines.append("SERVICE_TIME_SECTION")
    lines.extend(f"{i + 1} {service}" for i, service in enumerate(service_times))
    lines.extend(["DEPOT_SECTION", str(parsed.depot_node_index), "-1", "EOF"])
    return "\n".join(lines) + "\n"


def derive_vrptw_from_cvrp(
    folder: str | Path,
    base: str,
    *,
    tw_method: str = DEFAULT_TW_METHOD,
    horizon_start: int = HORIZON_START,
    horizon_end: int = HORIZON_END,
    place_slug: str = "",
    source_seed: int = 0,
) -> dict[str, Any]:
    """Derive the fastest-metric VRPTW twin next to a generated CVRP base.

    Reads ``<base>_fastest.vrp``, synthesizes service times + time windows on
    the fastest travel times, and writes ``<base>_fastest.cvrptw.vrp`` plus a
    ``<base>_vrptw_manifest.json`` describing the derivation.
    """
    folder = Path(folder)
    fastest_path = folder / f"{base}_fastest.vrp"
    if not fastest_path.is_file():
        raise FileNotFoundError(f"CVRP fastest .vrp not found at {fastest_path}")
    parsed = parse_cvrp_vrp(fastest_path)
    depot_index = parsed.depot_node_index - 1

    seed_parts = (base, place_slug, source_seed, tw_method, horizon_start, horizon_end, "vrptw_workbench_v1")
    service_times, time_windows, stochastic_params = generate_vrptw_fields(
        seed_parts,
        parsed.arc_costs,
        horizon_start,
        horizon_end,
        tw_method,
        demands=[int(value) for value in parsed.demands],
        capacity=int(parsed.capacity),
        depot=depot_index,
    )

    vrptw_name = f"{base}_fastest_vrptw"
    vrptw_filename = f"{base}_fastest.cvrptw.vrp"
    text = cvrptw_text(
        parsed,
        vrptw_name,
        f"VRPTW derived from {base}_fastest ({tw_method})",
        service_times,
        time_windows,
    )
    manifest_path = folder / f"{base}_vrptw_manifest.json"
    twin_path = folder / vrptw_filename
    if twin_path.is_file() and manifest_path.is_file():
        # Deterministic: an identical derivation is left untouched, so its
        # files (and the fingerprints derive-td records) do not churn.
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except ValueError:
            previous = {}
        if twin_path.read_text(encoding="utf-8") == text and previous.get("derivation") == json.loads(
            json.dumps(stochastic_params)
        ):
            return {
                "ok": True,
                "action": "unchanged",
                "vrptw_file": vrptw_filename,
                "manifest": manifest_path.name,
                "derivation": stochastic_params,
            }
    twin_path.write_text(text, encoding="utf-8")
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "base_name": base,
        "source_file": fastest_path.name,
        "vrptw_file": vrptw_filename,
        "derivation": stochastic_params,
    }
    write_json(manifest_path, manifest)
    return {
        "ok": True,
        "action": "derived",
        "vrptw_file": vrptw_filename,
        "manifest": f"{base}_vrptw_manifest.json",
        "derivation": stochastic_params,
    }
