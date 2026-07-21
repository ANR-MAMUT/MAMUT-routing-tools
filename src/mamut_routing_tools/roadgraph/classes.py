"""Road classes shared by graph construction and OSM acquisition."""

from __future__ import annotations

#: OpenStreetMapX-compatible road classes used by the MAMUT road engine.
ROAD_CLASSES: dict[str, int] = {
    "motorway": 1,
    "trunk": 2,
    "primary": 3,
    "secondary": 4,
    "tertiary": 5,
    "unclassified": 6,
    "residential": 6,
    "service": 7,
    "motorway_link": 1,
    "trunk_link": 2,
    "primary_link": 3,
    "secondary_link": 4,
    "tertiary_link": 5,
    "living_street": 8,
    "pedestrian": 8,
    "road": 6,
}
