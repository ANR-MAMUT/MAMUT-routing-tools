"""Streaming OSM XML parsing for the road engine.

Reads only what the road graph needs: node coordinates, ways with their tags
and node lists, and the ``<bounds>`` element. City extracts run to ~150 MB,
so parsing is incremental with element recycling.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OsmWay:
    way_id: int
    nodes: list[int] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class OsmData:
    nodes: dict[int, tuple[float, float]]  # id -> (lat, lon)
    ways: list[OsmWay]
    bounds: tuple[float, float, float, float]  # (min_lat, min_lon, max_lat, max_lon)


def ensure_bounds(osm_path: Path) -> None:
    """Inject a ``<bounds>`` element computed from node coordinates when the
    extract ships without one (some Overpass responses omit it). Matches the
    Julia pipeline's behavior so both read identical files afterwards."""
    with osm_path.open("r", encoding="utf-8") as handle:
        head = handle.read(64 * 1024)
    if re.search(r"<bounds\b", head):
        return
    min_lat = min_lon = float("inf")
    max_lat = max_lon = float("-inf")
    count = 0
    for _, elem in ET.iterparse(osm_path, events=("start",)):
        if elem.tag == "node":
            lat = float(elem.attrib["lat"])
            lon = float(elem.attrib["lon"])
            min_lat, max_lat = min(min_lat, lat), max(max_lat, lat)
            min_lon, max_lon = min(min_lon, lon), max(max_lon, lon)
            count += 1
        elem.clear()
    if count == 0:
        raise ValueError(f"OSM file '{osm_path}' contains no nodes; cannot derive bounds")
    bounds_line = (
        f'  <bounds minlat="{min_lat}" minlon="{min_lon}" maxlat="{max_lat}" maxlon="{max_lon}"/>\n'
    )
    text = osm_path.read_text(encoding="utf-8")
    match = re.search(r"<osm\b[^>]*>\n?", text)
    if match is None:
        raise ValueError(f"OSM file '{osm_path}' has no <osm> root element")
    insert_at = match.end()
    osm_path.write_text(text[:insert_at] + bounds_line + text[insert_at:], encoding="utf-8")


def parse_osm(osm_path: Path) -> OsmData:
    nodes: dict[int, tuple[float, float]] = {}
    ways: list[OsmWay] = []
    bounds: tuple[float, float, float, float] | None = None

    current_way: OsmWay | None = None
    for event, elem in ET.iterparse(osm_path, events=("start", "end")):
        tag = elem.tag
        if event == "start":
            if tag == "way":
                current_way = OsmWay(way_id=int(elem.attrib.get("id", "0")))
            elif tag == "nd" and current_way is not None:
                current_way.nodes.append(int(elem.attrib["ref"]))
            elif tag == "tag" and current_way is not None:
                current_way.tags[elem.attrib.get("k", "")] = elem.attrib.get("v", "")
            elif tag == "bounds":
                bounds = (
                    float(elem.attrib["minlat"]),
                    float(elem.attrib["minlon"]),
                    float(elem.attrib["maxlat"]),
                    float(elem.attrib["maxlon"]),
                )
            continue
        if tag == "node":
            nodes[int(elem.attrib["id"])] = (float(elem.attrib["lat"]), float(elem.attrib["lon"]))
            elem.clear()
        elif tag == "way":
            if current_way is not None:
                ways.append(current_way)
                current_way = None
            elem.clear()
        elif tag == "osm":
            elem.clear()

    if bounds is None:
        raise ValueError(f"OSM file '{osm_path}' has no <bounds> element; run ensure_bounds first")
    return OsmData(nodes=nodes, ways=ways, bounds=bounds)


def _inbounds(point: tuple[float, float], bounds: tuple[float, float, float, float]) -> bool:
    min_lat, min_lon, max_lat, max_lon = bounds
    lat, lon = point
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def _onbounds(point: tuple[float, float], bounds: tuple[float, float, float, float]) -> bool:
    min_lat, min_lon, max_lat, max_lon = bounds
    lat, lon = point
    return lon == min_lon or lon == max_lon or lat == min_lat or lat == max_lat


def _boundary_point(
    p1: tuple[float, float],
    p2: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    min_lat, min_lon, max_lat, max_lon = bounds
    x1, y1 = p1[1], p1[0]  # x = lon, y = lat
    x2, y2 = p2[1], p2[0]

    x, y = float("inf"), float("inf")
    if x1 < min_lon < x2 or x1 > min_lon > x2:
        x = min_lon
        y = y1 + (y2 - y1) * (min_lon - x1) / (x2 - x1)
    elif x1 < max_lon < x2 or x1 > max_lon > x2:
        x = max_lon
        y = y1 + (y2 - y1) * (max_lon - x1) / (x2 - x1)
    if _inbounds((y, x), bounds):
        return (y, x)

    if y1 < min_lat < y2 or y1 > min_lat > y2:
        x = x1 + (x2 - x1) * (min_lat - y1) / (y2 - y1)
        y = min_lat
    elif y1 < max_lat < y2 or y1 > max_lat > y2:
        x = x1 + (x2 - x1) * (max_lat - y1) / (y2 - y1)
        y = max_lat
    if _inbounds((y, x), bounds):
        return (y, x)
    raise ValueError("Failed to find boundary point")


def _crop_way(
    nodes: dict[int, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    way: OsmWay,
    allocate_node_id,
) -> bool:
    """Crop one way to the bounds rectangle (OpenStreetMapX crop! port).

    Returns True when the way should be dropped entirely. Outside nodes at
    an inside/outside transition are replaced by synthetic nodes on the
    boundary (unless the neighbouring inside node already sits on it);
    interior runs of outside nodes are removed."""
    way.nodes = [node for node in way.nodes if node in nodes]
    valid = [_inbounds(nodes[node], bounds) for node in way.nodes]
    inside = sum(valid)
    if inside == 0:
        return True
    if inside == len(valid):
        return False

    leave = [True] * len(way.nodes)
    for index in range(len(way.nodes)):
        if valid[index]:
            continue
        prev_valid = valid[index - 1] if index > 0 else False
        next_valid = valid[index + 1] if index < len(way.nodes) - 1 else False
        if prev_valid:
            if not _onbounds(nodes[way.nodes[index - 1]], bounds):
                point = _boundary_point(nodes[way.nodes[index - 1]], nodes[way.nodes[index]], bounds)
                new_id = allocate_node_id()
                nodes[new_id] = point
                way.nodes[index] = new_id
            else:
                leave[index] = False
        elif next_valid:
            if not _onbounds(nodes[way.nodes[index + 1]], bounds):
                point = _boundary_point(nodes[way.nodes[index]], nodes[way.nodes[index + 1]], bounds)
                new_id = allocate_node_id()
                nodes[new_id] = point
                way.nodes[index] = new_id
            else:
                leave[index] = False
        else:
            leave[index] = False
    way.nodes = [node for node, keep in zip(way.nodes, leave) if keep]
    return False


def crop_to_bounds(osm_data: OsmData) -> None:
    """Crop ways and nodes to the ``<bounds>`` rectangle, as
    ``get_map_data`` does before graph construction. Mutates in place."""
    min_lat, min_lon, max_lat, max_lon = osm_data.bounds
    if min_lon > max_lon:
        raise ValueError("Antimeridian-crossing bounds are not supported")
    next_id = max(osm_data.nodes.keys(), default=0) + 1

    def allocate_node_id() -> int:
        nonlocal next_id
        value = next_id
        next_id += 1
        return value

    osm_data.ways = [
        way for way in osm_data.ways if not _crop_way(osm_data.nodes, osm_data.bounds, way, allocate_node_id)
    ]
    osm_data.nodes = {
        node_id: point for node_id, point in osm_data.nodes.items() if _inbounds(point, osm_data.bounds)
    }
