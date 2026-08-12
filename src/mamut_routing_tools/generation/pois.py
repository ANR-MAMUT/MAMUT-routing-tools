"""Amenity POI extraction from an OSM XML extract."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple

POI_CATEGORIES = [
    "restaurant",
    "cafe",
    "bar",
    "fast_food",
    "pub",
    "school",
    "university",
    "hospital",
    "clinic",
    "pharmacy",
    "dentist",
    "doctors",
    "veterinary",
    "bank",
    "atm",
    "post_office",
    "police",
    "fire_station",
    "townhall",
    "courthouse",
    "library",
    "theatre",
    "cinema",
    "arts_centre",
    "community_centre",
    "museum",
    "place_of_worship",
    "marketplace",
    "fuel",
    "charging_station",
    "car_wash",
    "parking",
    "bus_station",
    "taxi",
    "bicycle_rental",
    "ferry_terminal",
    "kindergarten",
    "college",
    "nightclub",
    "biergarten",
    "ice_cream",
    "food_court",
    "bench",
    "drinking_water",
    "toilets",
    "shower",
    "shelter",
    "waste_basket",
    "recycling",
]

DEFAULT_CATEGORIES = POI_CATEGORIES[:7]


class NoPoiFoundError(ValueError):
    """The extract holds no amenity of the requested categories.

    A ``ValueError`` still, so callers that only care that the selection failed
    keep working; named so that the hybrid method can tell this apart from a
    genuine bad request and fall back to parametric points instead of failing.
    """


def is_poi_source_tag(tag: str) -> bool:
    """Whether a node's ``source_tag`` means it came from an amenity.

    Covers ``poi`` and ``poi_manual``; parametric tags (``param``,
    ``param_fill``) are not POIs.
    """
    return str(tag).startswith("poi")


class Poi(NamedTuple):
    lat: float
    lon: float
    category: str
    # Defaulted so existing positional unpacking of (lat, lon, category) keeps
    # working; both come straight from the OSM element and may be missing.
    osm_id: int | None = None
    name: str | None = None
    # OSM ids are only unique per element type, so an id alone cannot identify a
    # POI once ways and relations are in play. Defaulted to the historical value
    # so metadata written before ways were supported still reads back correctly.
    osm_type: str = "node"


def poi_ref(osm_type: str | None, osm_id: int | None) -> tuple[str, int] | None:
    """A hashable ``(type, id)`` identity, or ``None`` for an unidentifiable POI."""
    if osm_id is None:
        return None
    return (str(osm_type or "node"), int(osm_id))


def _element_point(element: ET.Element) -> tuple[float, float] | None:
    """Where an OSM element sits: its own position, or its ``<center>``.

    Nodes carry ``lat``/``lon`` directly. Ways and relations -- a shop drawn as
    a building outline, a multipolygon -- have no position of their own, so the
    fetcher asks Overpass for ``out center`` and reads the representative point
    it computes. An element with neither is not usable as a customer location.
    """
    lat = element.get("lat")
    lon = element.get("lon")
    if lat is None or lon is None:
        center = element.find("center")
        if center is None:
            return None
        lat = center.get("lat")
        lon = center.get("lon")
    if lat is None or lon is None:
        return None
    try:
        return float(lat), float(lon)
    except ValueError:
        return None


def find_pois(osm_path: str | Path, categories: list[str] | None = None) -> list[Poi]:
    """Every element tagged with an ``amenity`` in ``categories``, in file order.

    Nodes, ways and relations alike: in a real city a large share of the bars,
    restaurants, schools and car parks are mapped as building outlines rather
    than points, and reading nodes alone silently loses them.
    """
    wanted = set(categories or DEFAULT_CATEGORIES)
    pois: list[Poi] = []
    for _event, element in ET.iterparse(str(osm_path), events=("end",)):
        if element.tag not in ("node", "way", "relation"):
            continue
        point = _element_point(element)
        if point is not None:
            # The whole tag block is read rather than stopping at the amenity,
            # because the display name usually sits after it in file order.
            category: str | None = None
            name: str | None = None
            for tag in element:
                if tag.tag != "tag":
                    continue
                key = tag.get("k")
                if key == "amenity":
                    value = tag.get("v")
                    if value in wanted:
                        category = str(value)
                    else:
                        break
                elif key == "name" and name is None:
                    name = str(tag.get("v") or "") or None
            if category is not None:
                raw_id = element.get("id")
                pois.append(
                    Poi(
                        point[0],
                        point[1],
                        category,
                        int(raw_id) if raw_id is not None else None,
                        name,
                        element.tag,
                    )
                )
        element.clear()
    return pois


# Parsed catalogs keyed by (resolved path, mtime, size). Every consumer of the
# full catalog -- the picker endpoint, which re-queries on each category toggle,
# and manual generation, which resolves picked ids on every preview and generate
# -- shares this, because a 100+ MB extract costs seconds to re-parse. Bounded so
# a long session browsing many cities cannot grow it without limit.
_CATALOG_CACHE: OrderedDict[tuple[str, int, int], list[Poi]] = OrderedDict()
_CATALOG_CACHE_MAX_CITIES = 4


def load_poi_catalog(osm_path: str | Path) -> list[Poi]:
    """Every POI of every known category in an extract, parsed at most once.

    The whole catalog is read rather than a filtered subset, so switching the
    category filter never costs another pass over the file. The returned list is
    shared and must not be mutated by callers.
    """
    path = Path(osm_path)
    stat = path.stat()
    key = (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    cached = _CATALOG_CACHE.get(key)
    if cached is not None:
        _CATALOG_CACHE.move_to_end(key)
        return cached
    pois = find_pois(path, list(POI_CATEGORIES))
    _CATALOG_CACHE[key] = pois
    while len(_CATALOG_CACHE) > _CATALOG_CACHE_MAX_CITIES:
        _CATALOG_CACHE.popitem(last=False)
    return pois
