"""Coverage audit and POI backfill for OSM extracts already on disk.

An extract is a long-lived local asset: cities are fetched once and reused for
months. When the acquisition side learns to fetch something it previously did
not -- amenities mapped as ways and relations, which is a quarter to a third of
the POIs in a real city -- nothing invalidates the files already downloaded, and
generation quietly keeps drawing from the smaller pool.

This module answers "what do my extracts actually contain?" and refills the
missing part in place, without re-downloading the road network.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from mamut_routing_tools.generation.pois import POI_CATEGORIES
from mamut_routing_tools.osm.fetch import ProgressCallback, fetch_tiled_amenities

#: Extracts holding amenities on ways or relations were fetched by a version
#: that asks for them; the rest predate it and can be refilled.
STATUS_COMPLETE = "complete"
STATUS_NODES_ONLY = "nodes_only"
STATUS_NO_AMENITIES = "no_amenities"

_ELEMENT_TYPES = ("node", "way", "relation")


def audit_extract(osm_path: str | Path) -> dict[str, Any]:
    """What one extract holds: element counts, POI counts per type, and bounds.

    Counted in a single streaming pass without building POI objects, because a
    workspace can hold dozens of extracts of 100+ MB each.
    """
    path = Path(osm_path)
    stat = path.stat()
    elements: Counter[str] = Counter()
    pois: Counter[str] = Counter()
    bounds: dict[str, float] | None = None
    for _event, element in ET.iterparse(str(path), events=("end",)):
        if element.tag == "bounds" and bounds is None:
            try:
                bounds = {key: float(element.attrib[key]) for key in
                          ("minlat", "minlon", "maxlat", "maxlon")}
            except (KeyError, ValueError):
                bounds = None
            continue
        if element.tag not in _ELEMENT_TYPES:
            continue
        elements[element.tag] += 1
        for tag in element:
            if tag.tag == "tag" and tag.get("k") == "amenity":
                pois[element.tag] += 1
                break
        element.clear()

    poi_total = sum(pois.values())
    if poi_total == 0:
        status = STATUS_NO_AMENITIES
    elif pois["way"] or pois["relation"]:
        status = STATUS_COMPLETE
    else:
        status = STATUS_NODES_ONLY
    return {
        "city": path.stem,
        "path": str(path),
        "bytes": stat.st_size,
        "modified_at": stat.st_mtime,
        "nodes": elements["node"],
        "ways": elements["way"],
        "relations": elements["relation"],
        "poi_nodes": pois["node"],
        "poi_ways": pois["way"],
        "poi_relations": pois["relation"],
        "poi_total": poi_total,
        "status": status,
        # Nothing can be refilled without knowing the area to query.
        "can_refresh": bounds is not None,
        "bounds": bounds,
    }


def audit_extracts(osm_dir: str | Path) -> list[dict[str, Any]]:
    """Audit every ``*.osm`` in a directory, by city name."""
    root = Path(osm_dir)
    if not root.is_dir():
        return []
    reports: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.osm")):
        try:
            reports.append(audit_extract(path))
        except (OSError, ET.ParseError) as error:
            reports.append(
                {
                    "city": path.stem,
                    "path": str(path),
                    "status": "unreadable",
                    "error": str(error),
                    "can_refresh": False,
                }
            )
    return reports


def refresh_extract_pois(
    osm_path: str | Path,
    *,
    poi_categories: tuple[str, ...] | None = tuple(POI_CATEGORIES),
    progress: ProgressCallback | None = None,
    tile_cache_dir: str | Path | None = None,
    use_tile_cache: bool = True,
) -> dict[str, Any]:
    """Backfill the amenities an extract is missing, in place.

    Only the amenity query is re-run: the road network already in the file is
    left untouched, which is what makes this cheap enough to run over a whole
    workspace. The report pairs the audit before and after so a caller can say
    what each city gained.
    """
    path = Path(osm_path)
    before = audit_extract(path)
    if not before.get("can_refresh"):
        return {
            "city": before["city"],
            "path": str(path),
            "ok": False,
            "error": "extract has no <bounds>; re-fetch the city instead",
            "before": before,
        }
    bounds = before["bounds"]
    summary = fetch_tiled_amenities(
        bounds["minlat"],
        bounds["minlon"],
        bounds["maxlat"],
        bounds["maxlon"],
        path,
        poi_categories=poi_categories,
        progress=progress,
        tile_cache_dir=tile_cache_dir,
        use_tile_cache=use_tile_cache,
    )
    after = audit_extract(path)
    return {
        "city": before["city"],
        "path": str(path),
        "ok": bool(summary.get("ok")),
        "tiles_total": summary.get("tiles_total"),
        "tiles_ok": summary.get("tiles_ok"),
        "failure_count": summary.get("failure_count"),
        "poi_total_before": before["poi_total"],
        "poi_total_after": after["poi_total"],
        "gained": after["poi_total"] - before["poi_total"],
        "gained_ways": after["poi_ways"] - before["poi_ways"],
        "gained_relations": after["poi_relations"] - before["poi_relations"],
        "status_before": before["status"],
        "status_after": after["status"],
    }


def refresh_extracts(
    paths: list[Path],
    *,
    poi_categories: tuple[str, ...] | None = tuple(POI_CATEGORIES),
    on_city: Callable[[int, int, Path], None] | None = None,
    progress: ProgressCallback | None = None,
    tile_cache_dir: str | Path | None = None,
    use_tile_cache: bool = True,
) -> dict[str, Any]:
    """Refresh several extracts, reporting each one rather than aborting.

    One city failing -- a timeout, an Overpass outage -- must not cost the
    others: the run continues and the failure is reported on its own row.
    """
    results: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        if on_city is not None:
            on_city(index, len(paths), path)
        try:
            results.append(
                refresh_extract_pois(
                    path,
                    poi_categories=poi_categories,
                    progress=progress,
                    tile_cache_dir=tile_cache_dir,
                    use_tile_cache=use_tile_cache,
                )
            )
        except Exception as error:  # noqa: BLE001 - one bad city must not end the run
            results.append(
                {"city": Path(path).stem, "path": str(path), "ok": False, "error": str(error)}
            )
    return {
        "ok": all(entry.get("ok") for entry in results),
        "refreshed": sum(1 for entry in results if entry.get("ok")),
        "gained": sum(int(entry.get("gained") or 0) for entry in results),
        "results": results,
    }
