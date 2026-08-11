"""Amenity POI extraction from an OSM XML extract."""

from __future__ import annotations

import xml.etree.ElementTree as ET
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
    # working; both come straight from the OSM node and may be missing.
    osm_id: int | None = None
    name: str | None = None


def find_pois(osm_path: str | Path, categories: list[str] | None = None) -> list[Poi]:
    """All nodes tagged with an ``amenity`` in ``categories``, in file order."""
    wanted = set(categories or DEFAULT_CATEGORIES)
    pois: list[Poi] = []
    for _event, element in ET.iterparse(str(osm_path), events=("end",)):
        if element.tag != "node":
            continue
        lat = element.get("lat")
        lon = element.get("lon")
        if lat is not None and lon is not None:
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
                        float(lat),
                        float(lon),
                        category,
                        int(raw_id) if raw_id is not None else None,
                        name,
                    )
                )
        element.clear()
    return pois
