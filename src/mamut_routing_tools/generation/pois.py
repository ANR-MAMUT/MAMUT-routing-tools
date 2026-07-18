"""Amenity POI extraction from an OSM XML extract."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

DEFAULT_CATEGORIES = ["restaurant", "cafe", "bar", "fast_food", "pub", "school", "university"]


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
