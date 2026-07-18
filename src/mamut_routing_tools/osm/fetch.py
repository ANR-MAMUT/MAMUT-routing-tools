"""OSM city acquisition: Nominatim geocode + Overpass download.

Port of the Julia pipeline's fetch flow: geocode the city bbox (optionally
clamped around the place's geocode point and padded), download roads plus
amenities from Overpass across three public endpoints with retry/backoff,
fall back to a roads-only query, then backfill amenities tile by tile when
the combined query was too heavy. The result is a single ``<city>.osm``
XML extract with an explicit ``<bounds>`` element.

Runs locally on the user's machine: no shared-server rate limiting beyond
being a polite Overpass citizen (sequential requests, exponential backoff,
identifying User-Agent).
"""

from __future__ import annotations

import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

USER_AGENT = "MAMUT-routing-tools/0.1 (OSM city fetch)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


class FetchError(RuntimeError):
    pass


def sanitize_city_filename(city: str) -> str:
    name = city.strip()
    if not name:
        raise FetchError("City name cannot be empty")
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if name in (".", ".."):
        raise FetchError("Invalid city name")
    return name


def _geocode(query: str) -> dict:
    response = httpx.get(
        NOMINATIM_URL,
        params={"q": query, "format": "json", "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=30.0,
    )
    if response.status_code != 200:
        raise FetchError(f"Geocode failed: HTTP {response.status_code}")
    data = response.json()
    if not data:
        raise FetchError(f"No result found for '{query}'")
    return data[0]


def fetch_city_bbox(city: str, country: str = "") -> tuple[float, float, float, float]:
    """(min_lat, min_lon, max_lat, max_lon) of the geocoded place."""
    query = city if not country.strip() else f"{city}, {country}"
    box = _geocode(query)["boundingbox"]
    return float(box[0]), float(box[2]), float(box[1]), float(box[3])


def fetch_city_center(city: str, country: str = "") -> tuple[float, float]:
    """Geocode point of the place itself, NOT the bbox center: administrative
    bboxes can include remote exclaves (Tokyo Metropolis spans Pacific
    islands, so its bbox center lies in open ocean)."""
    query = city if not country.strip() else f"{city}, {country}"
    result = _geocode(query)
    return float(result["lat"]), float(result["lon"])


def _error_snippet(body: str) -> str:
    return re.sub(r"\s+", " ", body)[:180]


def _is_retryable(status: int, body: str) -> bool:
    if status in (408, 429, 500, 502, 503, 504):
        return True
    lowered = body.lower()
    if "dispatcher_client::request_read_and_idx::timeout" in lowered:
        return True
    if "the server is probably too busy" in lowered:
        return True
    return "runtime error" in lowered and "timeout" in lowered


def build_overpass_query(bbox: str, *, include_amenities: bool = True) -> str:
    if include_amenities:
        return (
            "[out:xml][timeout:180][maxsize:1073741824];\n"
            "(\n"
            f'  way["highway"]{bbox};\n'
            f'  node["amenity"]{bbox};\n'
            ") -> .sel;\n"
            "(\n"
            "  .sel;\n"
            "  .sel >;\n"
            ");\n"
            "out body;\n"
        )
    return (
        "[out:xml][timeout:120][maxsize:536870912];\n"
        "(\n"
        f'  way["highway"]{bbox};\n'
        ") -> .sel;\n"
        "(\n"
        "  .sel;\n"
        "  .sel >;\n"
        ");\n"
        "out body;\n"
    )


@dataclass
class OverpassResult:
    body: str | None
    failures: list[str] = field(default_factory=list)


def fetch_overpass_body(
    query: str,
    *,
    attempts_per_endpoint: int = 1,
    read_timeout: float = 140.0,
    backoff_cap: float = 8.0,
    backoff_base: float = 1.5,
) -> OverpassResult:
    attempts_per_endpoint = max(1, attempts_per_endpoint)
    total_attempts = attempts_per_endpoint * len(OVERPASS_ENDPOINTS)
    attempt_index = 0
    failures: list[str] = []
    headers = {"Content-Type": "text/plain; charset=utf-8", "User-Agent": USER_AGENT}

    for endpoint in OVERPASS_ENDPOINTS:
        for _ in range(attempts_per_endpoint):
            attempt_index += 1
            try:
                response = httpx.post(
                    endpoint,
                    content=query,
                    headers=headers,
                    timeout=httpx.Timeout(read_timeout, connect=20.0),
                )
                body = response.text
                if response.status_code == 200:
                    if "<osm" in body:
                        return OverpassResult(body=body, failures=failures)
                    failures.append(f"{endpoint} -> HTTP 200 but response is not OSM XML")
                else:
                    failures.append(f"{endpoint} -> HTTP {response.status_code}: {_error_snippet(body)}")
                    if not _is_retryable(response.status_code, body):
                        return OverpassResult(body=None, failures=failures)
            except httpx.HTTPError as error:
                failures.append(f"{endpoint} -> {error}")
            if attempt_index < total_attempts:
                time.sleep(min(backoff_cap, backoff_base ** (attempt_index - 1)) + random.random() * 0.2)
    return OverpassResult(body=None, failures=failures)


def download_overpass_query(query: str, outpath: Path) -> OverpassResult:
    result = fetch_overpass_body(
        query,
        attempts_per_endpoint=2,
        read_timeout=220.0,
        backoff_cap=20.0,
        backoff_base=1.7,
    )
    if result.body is not None:
        outpath.write_text(result.body, encoding="utf-8")
    return result


def split_range(minimum: float, maximum: float, max_tile_span: float) -> list[tuple[float, float]]:
    span = maximum - minimum
    if span <= 0:
        return [(minimum, maximum)]
    tiles = max(1, math.ceil(span / max_tile_span))
    step = span / tiles
    return [
        (minimum + i * step, maximum if i == tiles - 1 else minimum + (i + 1) * step)
        for i in range(tiles)
    ]


_NODE_BLOCK_PATTERN = re.compile(
    r'(?s)<node\b[^>]*\bid="-?\d+"[^>]*/>|<node\b[^>]*\bid="-?\d+"[^>]*>.*?</node>'
)
_NODE_ID_PATTERN = re.compile(r'\bid="(-?\d+)"')


def merge_nodes_into_osm(osm_path: Path, node_blocks: list[str]) -> int:
    if not node_blocks:
        return 0
    text = osm_path.read_text(encoding="utf-8")
    existing_ids = {int(match) for match in re.findall(r'<node\b[^>]*\bid="(-?\d+)"[^>]*/?>', text)}
    to_add: list[str] = []
    for block in node_blocks:
        id_match = _NODE_ID_PATTERN.search(block)
        if id_match is None:
            continue
        node_id = int(id_match.group(1))
        if node_id not in existing_ids:
            existing_ids.add(node_id)
            to_add.append(block)
    if not to_add:
        return 0
    close_at = text.rfind("</osm>")
    if close_at < 0:
        raise FetchError(f"Invalid OSM file (missing </osm>): {osm_path}")
    merged = "\n" + "\n".join(to_add) + "\n"
    osm_path.write_text(text[:close_at] + merged + text[close_at:], encoding="utf-8")
    return len(to_add)


def fetch_tiled_amenities(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float, outpath: Path
) -> dict:
    lat_tiles = split_range(min_lat, max_lat, 0.03)
    lon_tiles = split_range(min_lon, max_lon, 0.04)
    total_tiles = len(lat_tiles) * len(lon_tiles)
    blocks: list[str] = []
    tiles_ok = 0
    failure_count = 0
    for lat_lo, lat_hi in lat_tiles:
        for lon_lo, lon_hi in lon_tiles:
            bbox = f"({lat_lo},{lon_lo},{lat_hi},{lon_hi})"
            query = (
                "[out:xml][timeout:75][maxsize:268435456];\n"
                "(\n"
                f'  node["amenity"]{bbox};\n'
                ");\n"
                "out body;\n"
            )
            result = fetch_overpass_body(query, attempts_per_endpoint=1, read_timeout=120.0)
            if result.body is None:
                failure_count += len(result.failures)
                continue
            blocks.extend(match.group(0) for match in _NODE_BLOCK_PATTERN.finditer(result.body))
            tiles_ok += 1
    added = merge_nodes_into_osm(outpath, blocks)
    return {
        "ok": tiles_ok > 0,
        "tiles_total": total_tiles,
        "tiles_ok": tiles_ok,
        "amenity_nodes_added": added,
        "failure_count": failure_count,
    }


def ensure_osm_has_bounds(
    osm_path: Path, min_lat: float, min_lon: float, max_lat: float, max_lon: float
) -> None:
    text = osm_path.read_text(encoding="utf-8")
    if re.search(r"<bounds", text):
        return
    match = re.search(r"<osm\b[^>]*>", text)
    if match is None:
        raise FetchError(f"No <osm> tag in {osm_path}")
    bounds_line = (
        f'<bounds minlat="{min_lat}" minlon="{min_lon}" maxlat="{max_lat}" maxlon="{max_lon}"/>'
    )
    osm_path.write_text(
        text[: match.end()] + "\n  " + bounds_line + "\n" + text[match.end() :],
        encoding="utf-8",
    )


def fetch_and_store_city_osm(
    city: str,
    *,
    country: str = "",
    osm_dir: str | Path = "osmdata",
    padding_km: float = 0.0,
    max_radius_km: float = 0.0,
) -> dict:
    safe_city = sanitize_city_filename(city)
    osm_root = Path(osm_dir)
    osm_root.mkdir(parents=True, exist_ok=True)
    outpath = osm_root / f"{safe_city}.osm"

    min_lat, min_lon, max_lat, max_lon = fetch_city_bbox(city, country)
    if max_radius_km > 0:
        # Clamp oversized administrative bboxes to a square around the
        # geocode point of the place.
        center_lat, center_lon = fetch_city_center(city, country)
        clamp_dlat = max_radius_km / 111.0
        clamp_dlon = max_radius_km / max(1e-6, 111.0 * math.cos(math.radians(center_lat)))
        min_lat, max_lat = center_lat - clamp_dlat, center_lat + clamp_dlat
        min_lon, max_lon = center_lon - clamp_dlon, center_lon + clamp_dlon
    if padding_km > 0:
        dlat = padding_km / 111.0
        mean_lat = (min_lat + max_lat) / 2.0
        dlon = padding_km / max(1e-6, 111.0 * math.cos(math.radians(mean_lat)))
        min_lat -= dlat
        max_lat += dlat
        min_lon -= dlon
        max_lon += dlon

    bbox = f"({min_lat},{min_lon},{max_lat},{max_lon})"
    full = download_overpass_query(build_overpass_query(bbox, include_amenities=True), outpath)
    if full.body is not None:
        dataset_mode = "roads_and_amenities"
    else:
        roads = download_overpass_query(build_overpass_query(bbox, include_amenities=False), outpath)
        if roads.body is None:
            raise FetchError(
                "Overpass unavailable for both queries. roads+amenities failures: "
                + (" | ".join(full.failures) or "no details")
                + " || roads-only failures: "
                + (" | ".join(roads.failures) or "no details")
            )
        dataset_mode = "roads_only"

    amenity_tiling = {"ok": False, "tiles_total": 0, "tiles_ok": 0, "amenity_nodes_added": 0, "failure_count": 0}
    if dataset_mode == "roads_only":
        amenity_tiling = fetch_tiled_amenities(min_lat, min_lon, max_lat, max_lon, outpath)
        if amenity_tiling["ok"] and amenity_tiling["amenity_nodes_added"] > 0:
            dataset_mode = "roads_plus_tiled_amenities"

    ensure_osm_has_bounds(outpath, min_lat, min_lon, max_lat, max_lon)

    return {
        "ok": True,
        "city": safe_city,
        "osm_path": str(outpath),
        "dataset_mode": dataset_mode,
        "amenity_tiling": amenity_tiling,
        "warning": (
            "Amenities could not be fetched from Overpass (including tiled fallback); "
            "POI-based generation may have fewer candidates."
            if dataset_mode == "roads_only"
            else ""
        ),
        "bbox": {"minlat": min_lat, "minlon": min_lon, "maxlat": max_lat, "maxlon": max_lon},
    }
