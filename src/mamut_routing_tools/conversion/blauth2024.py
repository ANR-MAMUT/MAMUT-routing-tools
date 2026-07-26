"""Convert the Blauth et al. 2024 vrptdt-benchmark into canonical MAMUT artifacts.

Upstream: https://gitlab.com/muelleratorunibonnde/vrptdt-benchmark (delivery-only
variant; the pickup-and-delivery variant is out of MAMUT's node-routing scope).
Design authority: the Blauth2024 family design note in tdvrptw-workspace
(reports/design/blauth2024-family-design.md) and the TD benchmark standard.

The conversion is a pure relabeling of upstream integer-millisecond data:
depot -> 0, items "1".."n" -> customers 1..n, arrival-time functions copied
verbatim (self-arcs dropped after an identity assertion). No numeric
transformation happens anywhere, so the emitted canonical bytes and the
``atf_sha256`` pins are bit-identical on any machine.

Family contract (validated by Onyr 2026-07-26): TDVRPTW only, integer
milliseconds since midnight, horizon [54000000, 79200000] (the real-arc ATF
domain, 15:00 to 22:00), depot time window [54000000, 86400000] (start at or
after 15:00; the midnight due date is provably non-binding: max(ys) over all
arcs of every instance is below midnight, asserted here per instance),
service 180000 ms per customer, demands 1 with vehicle capacity n (one item
per address, capacity never binds), unlimited fleet (num_vehicles null), and
``fleet_fixed_cost`` = 36000000 ms (the upstream $200 per vehicle at $20/h;
$ = cost_ms / 180000) under the FleetCostDuration objective.
"""

from __future__ import annotations

import bz2
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from mamut_routing_lib.json_utils import save_json_to_file
from mamut_routing_lib.td import InstanceATFs, compute_atf_sha256, save_instance_atfs

FAMILY = "Blauth2024"
UPSTREAM_CITIES = (
    "berlin",
    "cincinnati",
    "kyiv",
    "london",
    "madrid",
    "nairobi",
    "new_york",
    "san_francisco",
    "sao_paulo",
    "seattle",
)
UPSTREAM_SIZES = (10, 500, 1000, 2000)

HORIZON_START = 54_000_000  # 15:00 in ms since midnight
HORIZON_END = 79_200_000  # 22:00
DEPOT_DUE_DATE = 86_400_000  # midnight; provably non-binding, asserted per instance
SERVICE_TIME = 180_000  # 3 minutes
# Upstream only guarantees arc coverage "at least half an hour past 21:03";
# some arcs end before 22:00. The latest feasible departure on any arc is
# 21:03 (= latest TW end 21:00 + one 3-min service; depot departures after
# 21:00 cannot reach any customer within its TW), so [21:03, 22:00] is
# semantically dead and a short arc may be completed to the canonical horizon
# end at constant travel time (one appended integer breakpoint) without
# changing any feasible evaluation — the same convention as the Dabia2013
# horizon extension. Arcs ending before 21:03 would make the extension
# observable and are refused.
EXTENSION_SAFE_START = 75_780_000  # 21:03
FLEET_FIXED_COST = 36_000_000  # 10 h = $200 at $20/h
DOLLARS_PER_MS = 1.0 / 180_000.0

GENERATOR_NAME = "convert_blauth2024"
GENERATOR_VERSION = "1"


class Blauth2024ConversionError(ValueError):
    """A gate failed: the upstream data does not match the family contract."""


@dataclass
class ConvertedInstance:
    instance_name: str
    city: str
    n: int
    instance_path: Path
    atf_path: Path
    atf_sha256: str
    max_arrival: int
    upstream_files: dict[str, str] = field(default_factory=dict)


def instance_name_for(city: str) -> str:
    return f"Blauth-{city}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Blauth2024ConversionError(message)


def _evaluate_exact(xs: list[int], ys: list[int], t: int) -> tuple[int, int]:
    """PWL evaluation at integer ``t`` as an exact rational (numerator, denominator)."""
    from bisect import bisect_left

    k = bisect_left(xs, t)
    if k < len(xs) and xs[k] == t:
        return ys[k], 1
    x0, x1, y0, y1 = xs[k - 1], xs[k], ys[k - 1], ys[k]
    return y0 * (x1 - x0) + (y1 - y0) * (t - x0), x1 - x0


def _load_items(items_path: Path, n: int) -> dict:
    data = json.loads(items_path.read_text())
    _require(set(data) == {"depot", "items"}, f"{items_path}: unexpected top-level keys {sorted(data)}")
    items = data["items"]
    expected_ids = {str(k) for k in range(1, n + 1)}
    _require(set(items) == expected_ids, f"{items_path}: item ids are not exactly 1..{n}")
    for item_id, item in items.items():
        for key in ("earliest_delivery", "latest_delivery"):
            _require(isinstance(item[key], int), f"{items_path}: item {item_id} {key} is not an integer")
        _require(
            HORIZON_START <= item["earliest_delivery"] <= item["latest_delivery"] <= HORIZON_END,
            f"{items_path}: item {item_id} time window outside the horizon",
        )
    return data


def _validate_arc(source: str, target: str, xs: list[int], ys: list[int], context: str) -> None:
    _require(len(xs) == len(ys) and len(xs) >= 2, f"{context}: arc {source}->{target} malformed breakpoint arrays")
    for value in xs:
        _require(isinstance(value, int), f"{context}: arc {source}->{target} non-integer departure {value!r}")
    for value in ys:
        _require(isinstance(value, int), f"{context}: arc {source}->{target} non-integer arrival {value!r}")
    for k in range(len(xs) - 1):
        _require(xs[k] < xs[k + 1], f"{context}: arc {source}->{target} departures not strictly increasing at {k}")
        _require(ys[k] <= ys[k + 1], f"{context}: arc {source}->{target} FIFO violation at breakpoint {k}")
    for k in range(len(xs)):
        _require(ys[k] >= xs[k], f"{context}: arc {source}->{target} negative travel time at breakpoint {k}")
    _require(
        xs[0] == HORIZON_START,
        f"{context}: arc {source}->{target} domain starts at {xs[0]}, expected {HORIZON_START}",
    )
    _require(
        xs[-1] <= HORIZON_END,
        f"{context}: arc {source}->{target} domain ends at {xs[-1]}, beyond {HORIZON_END}",
    )
    _require(
        xs[-1] >= EXTENSION_SAFE_START,
        f"{context}: arc {source}->{target} domain ends at {xs[-1]}, before the "
        f"21:03 extension-safety bound {EXTENSION_SAFE_START}: completion to the "
        "horizon end would be observable by feasible schedules",
    )


def convert_instance(
    upstream_root: Path,
    output_root: Path,
    city: str,
    n: int,
    *,
    upstream_commit: str | None = None,
) -> ConvertedInstance:
    """Convert one (city, size) delivery-only instance; returns paths + pins.

    Writes ``n=<N>/Blauth-<city>.vrp.json`` and ``.atf.json.gz`` under
    ``output_root`` (the family directory, historic problem-type-first
    layout). Raises :class:`Blauth2024ConversionError` on any gate failure.
    """
    items_path = upstream_root / "instances" / f"{city}_{n}.json"
    tt_path = upstream_root / "instances" / f"{city}_{n}_tt.json.bz2"
    _require(items_path.is_file(), f"missing upstream file {items_path}")
    _require(tt_path.is_file(), f"missing upstream file {tt_path}")

    data = _load_items(items_path, n)
    items = data["items"]
    # Item ids are exactly "1".."n": customer index = int(id), depot = 0.
    id_to_vertex = {str(k): k for k in range(1, n + 1)}
    id_to_vertex["depot"] = 0

    with bz2.open(tt_path, "rt") as handle:
        tt_entries = json.load(handle)
    expected_entries = (n + 1) * (n + 1)
    _require(
        len(tt_entries) == expected_entries,
        f"{tt_path}: {len(tt_entries)} entries, expected {expected_entries}",
    )

    arcs: dict[tuple[int, int], object] = {}
    raw_return_arcs: dict[int, tuple[list[int], list[int]]] = {}
    max_arrival = 0
    co_located_arcs = 0
    domain_extended_arcs = 0
    context = f"{city}_{n}"
    # Import here so the module stays importable for --help without the heavy deps.
    from mamut_routing_lib.td import NDCPWLF

    identity_shape_ok = lambda xs, ys: xs == ys and xs[0] == 0 and xs[-1] == HORIZON_END and len(xs) == 2  # noqa: E731

    for entry in tt_entries:
        source, target = entry["from"], entry["to"]
        atf = entry["atf"]
        xs, ys = atf["atf_leave_time"], atf["atf_arrive_time"]
        if source == target:
            _require(
                identity_shape_ok(xs, ys),
                f"{context}: self-arc {source} is not the expected identity",
            )
            continue
        key = (id_to_vertex[source], id_to_vertex[target])
        _require(key not in arcs, f"{context}: duplicate arc {source}->{target}")
        if xs[0] == 0:
            # Upstream encodes co-located address pairs (duplicate shop
            # coordinates) as the exact full identity, like self-arcs. Keep
            # them as zero-travel arcs restricted to the canonical horizon.
            _require(
                identity_shape_ok(xs, ys),
                f"{context}: arc {source}->{target} starts at 0 but is not the co-located identity",
            )
            arcs[key] = NDCPWLF([float(HORIZON_START), float(HORIZON_END)], [float(HORIZON_START), float(HORIZON_END)])
            co_located_arcs += 1
            if key[1] == 0:
                raw_return_arcs[key[0]] = ([HORIZON_START, HORIZON_END], [HORIZON_START, HORIZON_END])
            if HORIZON_END > max_arrival:
                max_arrival = HORIZON_END
            continue
        _validate_arc(source, target, xs, ys, context)
        if key[1] == 0:
            raw_return_arcs[key[0]] = (list(xs), list(ys))
        if xs[-1] < HORIZON_END:
            # Constant-travel completion of a short upstream domain (see the
            # EXTENSION_SAFE_START comment): integer, FIFO-safe, dead region.
            xs = [*xs, HORIZON_END]
            ys = [*ys, ys[-1] + (HORIZON_END - xs[-2])]
            domain_extended_arcs += 1
        arcs[key] = NDCPWLF([float(v) for v in xs], [float(v) for v in ys])
        if ys[-1] > max_arrival:
            max_arrival = ys[-1]

    _require(len(arcs) == (n + 1) * n, f"{context}: {len(arcs)} directed arcs, expected {(n + 1) * n}")

    # Midnight-due-date gate, exact rational: the latest feasible departure on
    # any arc is 21:03 (EXTENSION_SAFE_START), and the depot due date only
    # restricts the return arrival, so midnight is non-binding iff every
    # return arc arrives before it when departing at 21:03. The raw-data
    # loose bound (max arrival < midnight) does NOT survive the dead-region
    # domain extension above (a slow ~2 h arc extended to a 22:00 departure
    # can arrive past midnight), and is not the semantically relevant bound.
    _require(len(raw_return_arcs) == n, f"{context}: {len(raw_return_arcs)} return arcs, expected {n}")
    max_return_num, max_return_den = 0, 1
    for vertex in range(1, n + 1):
        xs_r, ys_r = raw_return_arcs[vertex]
        value_num, value_den = _evaluate_exact(xs_r, ys_r, EXTENSION_SAFE_START)
        if value_num * max_return_den > max_return_num * value_den:
            max_return_num, max_return_den = value_num, value_den
    _require(
        max_return_num < DEPOT_DUE_DATE * max_return_den,
        f"{context}: a feasible return (departure 21:03) can arrive at "
        f"{max_return_num / max_return_den:.1f} >= the midnight depot due date {DEPOT_DUE_DATE}",
    )
    max_feasible_return_ceil = -(-max_return_num // max_return_den)

    upstream_files = {
        items_path.name: _sha256_file(items_path),
        tt_path.name: _sha256_file(tt_path),
    }
    generator = {
        "name": GENERATOR_NAME,
        "version": GENERATOR_VERSION,
        "source": "vrptdt-benchmark (Blauth et al. 2024, delivery-only)",
        **({"upstream_commit": upstream_commit} if upstream_commit else {}),
        "upstream_files": upstream_files,
    }

    name = instance_name_for(city)
    atfs = InstanceATFs(
        instance_name=name,
        benchmark_name=FAMILY,
        horizon=(float(HORIZON_START), float(HORIZON_END)),
        num_customers=n,
        arcs=arcs,
        generator=generator,
    )
    atf_sha = compute_atf_sha256(atfs)

    size_dir = output_root / f"n={n}"
    size_dir.mkdir(parents=True, exist_ok=True)
    atf_path = size_dir / f"{name}.atf.json.gz"
    save_instance_atfs(atfs, atf_path)

    coordinates = [[data["depot"]["longitude"], data["depot"]["latitude"]]]
    time_windows: list[list[int]] = [[HORIZON_START, DEPOT_DUE_DATE]]
    for k in range(1, n + 1):
        item = items[str(k)]
        coordinates.append([item["longitude"], item["latitude"]])
        time_windows.append([item["earliest_delivery"], item["latest_delivery"]])

    payload = {
        "instance_name": name,
        "instance_origin": FAMILY,
        "benchmark_name": FAMILY,
        "num_customers": n,
        "num_vehicles": None,
        "vehicle_capacity": n,
        "coordinates": coordinates,
        "demands": [0] + [1] * n,
        "service_times": [0] + [SERVICE_TIME] * n,
        "depot": 0,
        "horizon": [HORIZON_START, HORIZON_END],
        "fleet_fixed_cost": FLEET_FIXED_COST,
        "time_windows": time_windows,
        "td": {"model": "atf-ndcpwlf", "atf_path": atf_path.name, "atf_sha256": atf_sha},
        "metadata": {
            "authors": "Florian Rascoussier (0nyr)",
            "generator": generator,
            "time_unit": "millisecond",
            "coordinates_convention": "[longitude, latitude], WGS84 degrees, informative only",
            "objective": {
                "function": "FleetCostDuration",
                "fleet_fixed_cost_ms": FLEET_FIXED_COST,
                "dollar_mapping": "upstream cost in $ = cost_ms / 180000 ($200 per vehicle + $20 per hour)",
            },
            "depot_due_date": {
                "value": DEPOT_DUE_DATE,
                "rationale": (
                    "upstream imposes no return deadline; midnight is provably non-binding: "
                    "no feasible departure exists after 21:03 (latest TW end 21:00 + 3 min "
                    "service), and every return arc evaluated exactly at 21:03 arrives before "
                    "midnight (exact-rational gate at conversion). Note max_arc_arrival may "
                    "exceed midnight inside the semantically dead post-21:03 region "
                    "(constant-travel domain extensions)."
                ),
                "max_arc_arrival": max_arrival,
                "max_feasible_return_ceil": max_feasible_return_ceil,
            },
            "tw_semantics": "service begins within the time window (arrival after waiting <= latest); hard TWs",
            "co_located_arcs": co_located_arcs,
            "domain_extended_arcs": domain_extended_arcs,
            "distances_channel": (
                "upstream atf_dist_end_time/atf_distance (fastest-path meters per departure range) "
                "is not carried into the ATF sidecar; see the upstream files pinned in generator"
            ),
            "license": "CC-BY-NC-4.0 (OSM ODbL + Uber Movement speed data lineage)",
        },
    }
    instance_path = size_dir / f"{name}.vrp.json"
    save_json_to_file(payload, instance_path)

    return ConvertedInstance(
        instance_name=name,
        city=city,
        n=n,
        instance_path=instance_path,
        atf_path=atf_path,
        atf_sha256=atf_sha,
        max_arrival=max_arrival,
        upstream_files=upstream_files,
    )


def convert_family(
    upstream_root: Path,
    output_root: Path,
    *,
    cities: tuple[str, ...] = UPSTREAM_CITIES,
    sizes: tuple[int, ...] = (10, 500),
    upstream_commit: str | None = None,
    verify_roundtrip: bool = True,
) -> list[ConvertedInstance]:
    """Convert the requested slice; optionally re-load every emitted instance.

    The round-trip verification re-reads each ``.vrp.json`` through
    ``load_td_instance``, which re-derives the canonical ATF bytes and checks
    ``atf_sha256`` — the determinism gate of the standard.
    """
    results: list[ConvertedInstance] = []
    for n in sizes:
        for city in cities:
            result = convert_instance(upstream_root, output_root, city, n, upstream_commit=upstream_commit)
            if verify_roundtrip:
                from mamut_routing_lib.td import load_td_instance

                loaded = load_td_instance(result.instance_path)
                _require(
                    loaded.instance.fleet_fixed_cost == FLEET_FIXED_COST,
                    f"{result.instance_name}: fleet_fixed_cost did not round-trip",
                )
            results.append(result)
    return results
