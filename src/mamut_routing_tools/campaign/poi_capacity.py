"""How many real amenities a city can actually serve as customers.

The large tier asks for POI-only instances: every customer is a real place on
the map, not a point sampled off the road network. That is the most defensible
kind of instance the platform can build -- the customer set is a fact about the
city rather than an artefact of a sampler -- but it is bounded by something the
size grid knows nothing about. A city can only serve as many POI customers as
it has *distinct road vertices* reachable from its amenities: two cafes on the
same corner collapse into one customer, and an amenity more than
``attach_radius_m`` from any routable point is not a customer at all.

So the ceiling is neither the population nor the amenity count. It is
``len({vertex for poi in catalog if poi attaches})``, which
:func:`~mamut_routing_tools.generation.select.select_customers_poi` already
reports as ``stats["attached"]``. This module measures it directly, per city,
so the tier is sized from what the extracts hold instead of from an assumption
about which cities are "big enough".
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from mamut_routing_tools.generation.pois import POI_CATEGORIES, load_poi_catalog
from mamut_routing_tools.generation.select import (
    DEFAULT_POI_ATTACH_RADIUS_M,
    POI_ATTACH_NEAREST_VERTEX,
    catalog_attachment,
)
from mamut_routing_tools.roadgraph.build import load_road_graph


def categories_digest(categories: Sequence[str]) -> str:
    """A short fingerprint of a category list, order-independent.

    A capacity is only meaningful for the categories it was measured over, and
    curating the list changes every number in the file. Without a fingerprint a
    stale ``poi-capacity.json`` is indistinguishable from a fresh one, and the
    campaign would size instances against amenities it is no longer drawing
    from.
    """
    joined = ",".join(sorted(set(categories)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class PoiCapacity:
    """One city's ceiling on POI-only customers."""

    city: str
    osm_file: str
    num_vertices: int
    #: Amenities of the requested categories present in the extract.
    catalog_size: int
    #: Of those, how many attach to the road graph at all.
    attached_pois: int
    #: The ceiling: distinct graph vertices the whole amenity pool reaches.
    #: Anything above this cannot be built POI-only at any seed.
    capacity: int
    #: ``attached_pois / capacity`` -- how many amenities share a vertex on
    #: average. High values mean a dense high street collapsing into few stops.
    collapse_ratio: float
    #: The attach mode and radius the figures were measured under. A capacity
    #: is meaningless without them: ``nearest_node`` yields roughly a fifth of
    #: what ``nearest_vertex`` does on the same extract.
    attach_mode: str
    attach_radius_m: float
    #: Fingerprint of the category list behind the numbers above. Defaulted so
    #: an older file still loads, but an empty digest never matches a live one,
    #: which is exactly the "re-measure this" signal.
    categories_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "PoiCapacity":
        return cls(**record)

    def supports(self, n: int) -> bool:
        """Whether a POI-only instance of ``n`` customers plus a depot fits."""
        return self.capacity >= n + 1


def measure_capacity(
    osm_path: str | Path,
    city: str,
    *,
    categories: Sequence[str] = POI_CATEGORIES,
    attach_mode: str = POI_ATTACH_NEAREST_VERTEX,
    attach_radius_m: float = DEFAULT_POI_ATTACH_RADIUS_M,
) -> PoiCapacity:
    """Measure one city, reusing the same caches the generator uses.

    Deliberately does not call ``select_customers_poi``: that shuffles the pool
    and builds customer lists to arrive at the same ``attached`` count. The
    ceiling is a property of the attachment map alone, so the map is enough.
    """
    path = Path(osm_path)
    graph = load_road_graph(str(path), only_intersections=True, trim_to_connected=True)
    catalog = load_poi_catalog(str(path))
    attachment = catalog_attachment(graph, str(path), attach_mode, attach_radius_m)

    wanted = set(categories)
    vertices: set[int] = set()
    attached = 0
    catalog_size = 0
    for index, poi in enumerate(catalog):
        if poi.category not in wanted:
            continue
        catalog_size += 1
        hit = attachment[index]
        if hit is None:
            continue
        attached += 1
        vertices.add(hit[0])

    capacity = len(vertices)
    return PoiCapacity(
        city=city,
        osm_file=path.name,
        num_vertices=graph.vertex_count,
        catalog_size=catalog_size,
        attached_pois=attached,
        capacity=capacity,
        collapse_ratio=round(attached / capacity, 3) if capacity else 0.0,
        attach_mode=attach_mode,
        attach_radius_m=attach_radius_m,
        categories_digest=categories_digest(categories),
    )


def _measure_one(task: tuple[str, str, tuple[str, ...], str, float]) -> PoiCapacity:
    city, osm_path, categories, attach_mode, radius = task
    return measure_capacity(
        osm_path,
        city,
        categories=categories,
        attach_mode=attach_mode,
        attach_radius_m=radius,
    )


def measure_cities(
    cities: Iterable[tuple[str, Path]],
    *,
    categories: Sequence[str] = POI_CATEGORIES,
    attach_mode: str = POI_ATTACH_NEAREST_VERTEX,
    attach_radius_m: float = DEFAULT_POI_ATTACH_RADIUS_M,
    jobs: int = 1,
    on_progress: Callable[[str], None] | None = None,
) -> list[PoiCapacity]:
    """Measure every city; one worker holds one extract at a time."""
    cats = tuple(categories)
    tasks = [
        (city, str(osm_path), cats, attach_mode, attach_radius_m)
        for city, osm_path in cities
    ]
    if not tasks:
        return []

    results: list[PoiCapacity] = []
    if jobs <= 1:
        for task in tasks:
            if on_progress is not None:
                on_progress(task[0])
            results.append(_measure_one(task))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            pending = {pool.submit(_measure_one, task): task[0] for task in tasks}
            for future in as_completed(pending):
                result = future.result()
                if on_progress is not None:
                    on_progress(pending[future])
                results.append(result)
    results.sort(key=lambda item: item.city)
    return results


def save_capacities(capacities: Sequence[PoiCapacity], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "mamut-poi-capacity",
        "version": 1,
        "cities": [item.to_dict() for item in capacities],
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_capacities(
    path: str | Path, *, expect_categories: Sequence[str] | None = None
) -> list[PoiCapacity]:
    """Load a capacity file, refusing one measured over other categories.

    ``expect_categories`` turns a silent mis-sizing into a loud failure: every
    number in the file is relative to a category list, so reading it against a
    different one would classify cities by amenities the campaign no longer
    draws from.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    capacities = [PoiCapacity.from_dict(record) for record in payload["cities"]]
    if expect_categories is not None:
        wanted = categories_digest(expect_categories)
        stale = {item.categories_digest for item in capacities} - {wanted}
        if stale:
            raise ValueError(
                f"{path}: measured over categories {sorted(stale)}, "
                f"but this campaign uses {wanted} "
                "-- re-run the poi-capacity phase"
            )
    return capacities


def cities_supporting(
    capacities: Iterable[PoiCapacity], n: int, *, headroom: float = 1.0
) -> list[PoiCapacity]:
    """Cities that can serve ``n`` POI customers, largest capacity first.

    ``headroom`` above 1.0 demands slack: at exactly ``capacity == n + 1`` the
    "selection" is the whole amenity pool, so the seed stops mattering and the
    instance is no longer a draw from anything.
    """
    threshold = (n + 1) * headroom
    fitting = [item for item in capacities if item.capacity >= threshold]
    fitting.sort(key=lambda item: item.capacity, reverse=True)
    return fitting
