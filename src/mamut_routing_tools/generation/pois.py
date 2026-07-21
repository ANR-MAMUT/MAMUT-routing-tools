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


class Poi(NamedTuple):
    lat: float
    lon: float
    category: str


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
            for tag in element:
                if tag.tag == "tag" and tag.get("k") == "amenity" and tag.get("v") in wanted:
                    pois.append(Poi(float(lat), float(lon), str(tag.get("v"))))
                    break
        element.clear()
    return pois
