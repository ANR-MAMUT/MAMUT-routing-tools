"""Reliable, purpose-filtered OSM acquisition through Nominatim and Overpass.

Road ways and POI nodes use separate queries. Small extracts download each
directly; large extracts are assembled from persistently cached bounded tiles
in a disk-backed element store. This avoids Overpass recurse memory failures,
unbounded client-side memory growth, and repeated work after an interruption.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import re
import sqlite3
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import httpx

from mamut_routing_tools.generation.pois import DEFAULT_CATEGORIES
from mamut_routing_tools.roadgraph.classes import ROAD_CLASSES

USER_AGENT = "MAMUT-routing-tools/0.1 (OSM city fetch)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

# Administrative areas larger than this are almost always unusable for a city
# road graph (Tokyo Metropolis, for example, includes distant Pacific islands).
MAX_UNCLAMPED_BBOX_SPAN_KM = 250.0

# Direct queries are faster for genuinely small places. Beyond this span,
# tiling avoids server-side recurse memory failures.
LARGE_BBOX_SPAN_KM = 20.0
ROAD_TILE_LAT_SPAN = 0.12
ROAD_TILE_LON_SPAN = 0.15
AMENITY_TILE_LAT_SPAN = 0.03
AMENITY_TILE_LON_SPAN = 0.04
TILE_CACHE_VERSION = "filtered-v1"

FetchProfileName = Literal["road_cache", "generation", "full"]
FETCH_PROFILES = ("road_cache", "generation", "full")

ProgressCallback = Callable[[dict[str, Any]], None]


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class _FetchProfile:
    name: FetchProfileName
    road_classes: tuple[str, ...] | None
    poi_categories: tuple[str, ...] | None
    include_amenities: bool


def _clean_values(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    cleaned = tuple(dict.fromkeys(value.strip() for value in values if value.strip()))
    if not cleaned:
        raise FetchError(f"{label} cannot be empty")
    return cleaned


def _resolve_fetch_profile(
    profile: FetchProfileName | str | None,
    *,
    include_amenities: bool | None,
    poi_categories: Sequence[str] | None,
) -> _FetchProfile:
    if profile is None:
        resolved_name = "road_cache" if include_amenities is False else "generation"
    elif profile in FETCH_PROFILES:
        resolved_name = profile
    else:
        raise FetchError(
            f"Unknown OSM fetch profile {profile!r}; choose: {', '.join(FETCH_PROFILES)}"
        )

    expected_amenities = resolved_name != "road_cache"
    if include_amenities is not None and include_amenities != expected_amenities:
        raise FetchError(
            f"profile={resolved_name!r} conflicts with include_amenities="
            f"{include_amenities!r}"
        )
    if resolved_name == "road_cache":
        if poi_categories:
            raise FetchError("poi_categories cannot be used with profile='road_cache'")
        return _FetchProfile("road_cache", tuple(ROAD_CLASSES), (), False)
    if resolved_name == "full":
        if poi_categories:
            raise FetchError("poi_categories cannot be used with profile='full'")
        return _FetchProfile("full", None, None, True)
    categories = _clean_values(
        poi_categories if poi_categories is not None else DEFAULT_CATEGORIES,
        label="poi_categories",
    )
    return _FetchProfile("generation", tuple(ROAD_CLASSES), categories, True)


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
    """Return ``(min_lat, min_lon, max_lat, max_lon)`` for a place."""
    query = city if not country.strip() else f"{city}, {country}"
    box = _geocode(query)["boundingbox"]
    return float(box[0]), float(box[2]), float(box[1]), float(box[3])


def fetch_city_center(city: str, country: str = "") -> tuple[float, float]:
    """Return the geocode point, rather than the administrative bbox center."""
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
    return "runtime error" in lowered and ("timeout" in lowered or "out of memory" in lowered)


def _overpass_osm_error(
    body: str, query: str, *, require_ways: bool | None = None
) -> str | None:
    """Identify Overpass errors that are returned with HTTP 200 and an OSM root."""
    if not re.search(r"<osm\b", body) or "</osm>" not in body:
        return "response is not complete OSM XML"
    remark = re.search(r"(?is)<remark\b[^>]*>(.*?)</remark>", body)
    if remark is not None:
        detail = _error_snippet(re.sub(r"<[^>]+>", " ", remark.group(1)))
        return f"Overpass error remark: {detail or 'unknown error'}"
    if require_ways is None:
        require_ways = re.search(r'way\s*\["highway"', query) is not None
    if require_ways and not re.search(r"<way\b", body):
        return "road query returned no ways"
    return None


def _overpass_regex(values: Sequence[str]) -> str:
    return "^(" + "|".join(re.escape(value) for value in values) + ")$"


def _road_statement(bbox: str, road_classes: tuple[str, ...] | None) -> str:
    if road_classes is None:
        return f'way["highway"]{bbox};'
    pattern = json.dumps(_overpass_regex(road_classes))
    return f'way["highway"~{pattern}]{bbox};'


def _amenity_statement(
    bbox: str, poi_categories: tuple[str, ...] | None
) -> str:
    if poi_categories is None:
        return f'node["amenity"]{bbox};'
    pattern = json.dumps(_overpass_regex(poi_categories))
    return f'node["amenity"~{pattern}]{bbox};'


def build_road_overpass_query(
    bbox: str, *, road_classes: tuple[str, ...] | None = tuple(ROAD_CLASSES)
) -> str:
    """Fetch only road ways plus skeleton coordinates for referenced nodes."""
    return (
        "[out:xml][timeout:120][maxsize:536870912];\n"
        f"{_road_statement(bbox, road_classes)}\n"
        "out body qt;\n"
        ">;\n"
        "out skel qt;\n"
    )


def build_amenity_overpass_query(
    bbox: str,
    *,
    poi_categories: tuple[str, ...] | None = tuple(DEFAULT_CATEGORIES),
) -> str:
    """Fetch POI nodes only, optionally restricted to selected categories."""
    return (
        "[out:xml][timeout:75][maxsize:268435456];\n"
        f"{_amenity_statement(bbox, poi_categories)}\n"
        "out body qt;\n"
    )


def build_overpass_query(
    bbox: str,
    *,
    include_amenities: bool | None = None,
    profile: FetchProfileName | str | None = None,
    poi_categories: Sequence[str] | None = None,
) -> str:
    """Build the legacy combined query; downloads use split queries internally."""
    resolved = _resolve_fetch_profile(
        profile,
        include_amenities=include_amenities,
        poi_categories=poi_categories,
    )
    query = (
        "[out:xml][timeout:180][maxsize:1073741824];\n"
        f"{_road_statement(bbox, resolved.road_classes)}\n"
        "out body qt;\n"
        ">;\n"
        "out skel qt;\n"
    )
    if resolved.include_amenities:
        query += (
            f"{_amenity_statement(bbox, resolved.poi_categories)}\n"
            "out body qt;\n"
        )
    return query


def _amenity_query(
    bbox: str, poi_categories: tuple[str, ...] | None = tuple(DEFAULT_CATEGORIES)
) -> str:
    """Compatibility wrapper for the former private query builder."""
    return build_amenity_overpass_query(
        bbox, poi_categories=poi_categories
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
    require_ways: bool | None = None,
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
                    error = _overpass_osm_error(body, query, require_ways=require_ways)
                    if error is None:
                        return OverpassResult(body=body, failures=failures)
                    failures.append(f"{endpoint} -> HTTP 200 invalid OSM: {error}")
                else:
                    failures.append(
                        f"{endpoint} -> HTTP {response.status_code}: {_error_snippet(body)}"
                    )
                    if not _is_retryable(response.status_code, body):
                        return OverpassResult(body=None, failures=failures)
            except httpx.HTTPError as error:
                failures.append(f"{endpoint} -> {error}")
            if attempt_index < total_attempts:
                time.sleep(
                    min(backoff_cap, backoff_base ** (attempt_index - 1))
                    + random.random() * 0.2
                )
    return OverpassResult(body=None, failures=failures)


def _valid_cached_body(body: str, query: str) -> bool:
    if _overpass_osm_error(body, query, require_ways=False) is not None:
        return False
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return False
    return root.tag.rsplit("}", 1)[-1] == "osm"


def _tile_cache_path(cache_root: Path, query: str, *, kind: str) -> Path:
    digest = hashlib.sha256(
        f"{TILE_CACHE_VERSION}\0{query}".encode("utf-8")
    ).hexdigest()
    return cache_root / kind / digest[:2] / f"{digest[2:]}.osm.gz"


def _write_cached_body(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
        with gzip.open(temporary, "wt", encoding="utf-8") as compressed:
            compressed.write(body)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fetch_overpass_tile(
    query: str,
    *,
    cache_root: Path | None,
    kind: str,
    read_timeout: float,
) -> tuple[OverpassResult, bool]:
    cache_path = (
        None
        if cache_root is None
        else _tile_cache_path(cache_root, query, kind=kind)
    )
    if cache_path is not None and cache_path.is_file():
        try:
            with gzip.open(cache_path, "rt", encoding="utf-8") as compressed:
                body = compressed.read()
            if _valid_cached_body(body, query):
                return OverpassResult(body=body), True
        except (OSError, UnicodeError):
            pass
        cache_path.unlink(missing_ok=True)

    result = fetch_overpass_body(
        query,
        attempts_per_endpoint=1,
        read_timeout=read_timeout,
        require_ways=False,
    )
    if result.body is not None and cache_path is not None:
        _write_cached_body(cache_path, result.body)
    return result, False


def _resolve_tile_cache_root(
    target: Path,
    *,
    tile_cache_dir: str | Path | None,
    use_tile_cache: bool,
) -> Path | None:
    if not use_tile_cache:
        return None
    if tile_cache_dir is not None:
        return Path(tile_cache_dir).expanduser()
    return target.parent / ".mamut-osm-tile-cache"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            output.write(text)
            temporary = Path(output.name)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_overpass_query(query: str, outpath: Path) -> OverpassResult:
    result = fetch_overpass_body(
        query,
        attempts_per_endpoint=2,
        read_timeout=220.0,
        backoff_cap=20.0,
        backoff_base=1.7,
    )
    if result.body is not None:
        _atomic_write_text(outpath, result.body)
    return result


def split_range(
    minimum: float, maximum: float, max_tile_span: float
) -> list[tuple[float, float]]:
    span = maximum - minimum
    if span <= 0:
        return [(minimum, maximum)]
    tiles = max(1, math.ceil(span / max_tile_span))
    step = span / tiles
    return [
        (
            minimum + i * step,
            maximum if i == tiles - 1 else minimum + (i + 1) * step,
        )
        for i in range(tiles)
    ]


_NODE_BLOCK_PATTERN = re.compile(
    r'(?s)<node\b[^>]*\bid="-?\d+"[^>]*/>|<node\b[^>]*\bid="-?\d+"[^>]*>.*?</node>'
)
_WAY_BLOCK_PATTERN = re.compile(r'(?s)<way\b[^>]*\bid="-?\d+"[^>]*>.*?</way>')
_ELEMENT_ID_PATTERN = re.compile(r'\bid="(-?\d+)"')


def validate_osm_extract(osm_path: str | Path) -> dict[str, Any]:
    """Validate the structure needed by the road-graph parser.

    In particular, an Overpass ``<remark>`` document is rejected even though
    it is well-formed XML and was returned with HTTP 200.
    """
    path = Path(osm_path)
    if not path.is_file():
        raise FetchError(f"OSM extract does not exist: {path}")

    nodes = 0
    ways = 0
    bounds: dict[str, float] | None = None
    try:
        for _event, elem in ET.iterparse(path, events=("end",)):
            tag = elem.tag.rsplit("}", 1)[-1]
            if tag == "remark":
                detail = " ".join("".join(elem.itertext()).split())
                raise FetchError(
                    f"OSM extract contains an Overpass error remark: "
                    f"{detail or 'unknown error'}"
                )
            if tag == "bounds":
                try:
                    bounds = {
                        "minlat": float(elem.attrib["minlat"]),
                        "minlon": float(elem.attrib["minlon"]),
                        "maxlat": float(elem.attrib["maxlat"]),
                        "maxlon": float(elem.attrib["maxlon"]),
                    }
                except (KeyError, ValueError) as error:
                    raise FetchError(f"OSM extract has invalid bounds: {path}") from error
            elif tag == "node":
                nodes += 1
            elif tag == "way":
                ways += 1
            elem.clear()
    except ET.ParseError as error:
        raise FetchError(f"OSM extract is not complete XML: {path}: {error}") from error

    if bounds is None:
        raise FetchError(f"OSM extract has no <bounds> element: {path}")
    if nodes == 0:
        raise FetchError(f"OSM extract contains no nodes: {path}")
    if ways == 0:
        raise FetchError(f"OSM extract contains no ways: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "nodes": nodes,
        "ways": ways,
        "bounds": bounds,
    }


class _OsmElementStore:
    """Disk-backed, ID-deduplicating store used while assembling tile results."""

    def __init__(self, database_path: Path) -> None:
        self.connection = sqlite3.connect(database_path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY, xml TEXT NOT NULL)")
        self.connection.execute("CREATE TABLE ways (id INTEGER PRIMARY KEY, xml TEXT NOT NULL)")

    @staticmethod
    def _element_id(block: str) -> int | None:
        match = _ELEMENT_ID_PATTERN.search(block)
        return None if match is None else int(match.group(1))

    def add_body(self, body: str) -> tuple[int, int]:
        before_nodes, before_ways = self.counts()
        for match in _NODE_BLOCK_PATTERN.finditer(body):
            block = match.group(0)
            element_id = self._element_id(block)
            if element_id is not None:
                # Amenity tiles can replace a tagless copy from a recurse result.
                self.connection.execute(
                    "INSERT OR REPLACE INTO nodes(id, xml) VALUES (?, ?)",
                    (element_id, block),
                )
        for match in _WAY_BLOCK_PATTERN.finditer(body):
            block = match.group(0)
            element_id = self._element_id(block)
            if element_id is not None:
                self.connection.execute(
                    "INSERT OR IGNORE INTO ways(id, xml) VALUES (?, ?)",
                    (element_id, block),
                )
        self.connection.commit()
        after_nodes, after_ways = self.counts()
        return after_nodes - before_nodes, after_ways - before_ways

    def counts(self) -> tuple[int, int]:
        nodes = int(self.connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])
        ways = int(self.connection.execute("SELECT COUNT(*) FROM ways").fetchone()[0])
        return nodes, ways

    def write(
        self,
        outpath: Path,
        *,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
    ) -> None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        temporary = outpath.with_name(f".{outpath.name}.tiled.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as output:
                output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                output.write('<osm version="0.6" generator="MAMUT-routing-tools tiled Overpass">\n')
                output.write(
                    f'  <bounds minlat="{min_lat}" minlon="{min_lon}" '
                    f'maxlat="{max_lat}" maxlon="{max_lon}"/>\n'
                )
                for (xml,) in self.connection.execute("SELECT xml FROM nodes ORDER BY id"):
                    output.write(f"{xml}\n")
                for (xml,) in self.connection.execute("SELECT xml FROM ways ORDER BY id"):
                    output.write(f"{xml}\n")
                output.write("</osm>\n")
            validate_osm_extract(temporary)
            temporary.replace(outpath)
        finally:
            temporary.unlink(missing_ok=True)

    def close(self) -> None:
        self.connection.close()


def merge_nodes_into_osm(osm_path: Path, node_blocks: list[str]) -> int:
    if not node_blocks:
        return 0
    text = osm_path.read_text(encoding="utf-8")
    incoming: dict[int, str] = {}
    for block in node_blocks:
        id_match = _ELEMENT_ID_PATTERN.search(block)
        if id_match is None:
            continue
        incoming[int(id_match.group(1))] = block
    if not incoming:
        return 0

    existing_ids: set[int] = set()

    def replace_existing(match: re.Match[str]) -> str:
        block = match.group(0)
        id_match = _ELEMENT_ID_PATTERN.search(block)
        if id_match is None:
            return block
        node_id = int(id_match.group(1))
        existing_ids.add(node_id)
        # A separately fetched POI body enriches a skeleton road node with its
        # amenity tag while retaining the same ID and coordinates.
        return incoming.get(node_id, block)

    merged_text = _NODE_BLOCK_PATTERN.sub(replace_existing, text)
    to_add = [block for node_id, block in incoming.items() if node_id not in existing_ids]
    if merged_text == text and not to_add:
        return 0
    close_at = text.rfind("</osm>")
    if close_at < 0:
        raise FetchError(f"Invalid OSM file (missing </osm>): {osm_path}")
    if to_add:
        close_at = merged_text.rfind("</osm>")
        merged = "\n" + "\n".join(to_add) + "\n"
        merged_text = merged_text[:close_at] + merged + merged_text[close_at:]
    _atomic_write_text(osm_path, merged_text)
    return len(to_add)


def fetch_tiled_amenities(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    outpath: Path,
    *,
    poi_categories: tuple[str, ...] | None = tuple(DEFAULT_CATEGORIES),
    progress: ProgressCallback | None = None,
    tile_cache_dir: str | Path | None = None,
    use_tile_cache: bool = True,
) -> dict[str, Any]:
    """Backfill amenities into an already downloaded road extract."""
    target = Path(outpath)
    cache_root = _resolve_tile_cache_root(
        target,
        tile_cache_dir=tile_cache_dir,
        use_tile_cache=use_tile_cache,
    )
    lat_tiles = split_range(min_lat, max_lat, AMENITY_TILE_LAT_SPAN)
    lon_tiles = split_range(min_lon, max_lon, AMENITY_TILE_LON_SPAN)
    total_tiles = len(lat_tiles) * len(lon_tiles)
    blocks: list[str] = []
    tiles_ok = 0
    cache_hits = 0
    failure_count = 0
    current = 0
    for lat_lo, lat_hi in lat_tiles:
        for lon_lo, lon_hi in lon_tiles:
            bbox = f"({lat_lo},{lon_lo},{lat_hi},{lon_hi})"
            result, cache_hit = _fetch_overpass_tile(
                _amenity_query(bbox, poi_categories),
                cache_root=cache_root,
                kind="amenities",
                read_timeout=120.0,
            )
            current += 1
            cache_hits += int(cache_hit)
            if result.body is None:
                failure_count += len(result.failures)
            else:
                blocks.extend(
                    match.group(0) for match in _NODE_BLOCK_PATTERN.finditer(result.body)
                )
                tiles_ok += 1
            if progress is not None:
                progress(
                    {
                        "phase": "amenities",
                        "current": current,
                        "total": total_tiles,
                        "tiles_ok": tiles_ok,
                        "cache_hits": cache_hits,
                    }
                )
    added = merge_nodes_into_osm(outpath, blocks)
    return {
        "ok": tiles_ok == total_tiles,
        "tiles_total": total_tiles,
        "tiles_ok": tiles_ok,
        "cache_hits": cache_hits,
        "amenity_nodes_added": added,
        "failure_count": failure_count,
    }


def _fetch_direct_amenities(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    outpath: Path,
    *,
    poi_categories: tuple[str, ...] | None,
    progress: ProgressCallback | None,
    tile_cache_dir: str | Path | None,
    use_tile_cache: bool,
) -> dict[str, Any]:
    cache_root = _resolve_tile_cache_root(
        outpath,
        tile_cache_dir=tile_cache_dir,
        use_tile_cache=use_tile_cache,
    )
    bbox = f"({min_lat},{min_lon},{max_lat},{max_lon})"
    result, cache_hit = _fetch_overpass_tile(
        _amenity_query(bbox, poi_categories),
        cache_root=cache_root,
        kind="amenities",
        read_timeout=120.0,
    )
    ok = result.body is not None
    added = 0
    if result.body is not None:
        added = merge_nodes_into_osm(
            outpath,
            [match.group(0) for match in _NODE_BLOCK_PATTERN.finditer(result.body)],
        )
    if progress is not None:
        progress(
            {
                "phase": "amenities",
                "current": 1,
                "total": 1,
                "tiles_ok": int(ok),
                "cache_hits": int(cache_hit),
            }
        )
    return {
        "ok": ok,
        "tiles_total": 1,
        "tiles_ok": int(ok),
        "cache_hits": int(cache_hit),
        "amenity_nodes_added": added,
        "failure_count": 0 if ok else len(result.failures),
    }


def _empty_amenity_tiling() -> dict[str, Any]:
    return {
        "ok": False,
        "tiles_total": 0,
        "tiles_ok": 0,
        "cache_hits": 0,
        "amenity_nodes_added": 0,
        "failure_count": 0,
    }


def _empty_road_tiling() -> dict[str, Any]:
    return {
        "ok": False,
        "tiles_total": 0,
        "tiles_ok": 0,
        "cache_hits": 0,
        "failure_count": 0,
    }


def fetch_tiled_osm(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    outpath: str | Path,
    *,
    include_amenities: bool | None = None,
    profile: FetchProfileName | str | None = None,
    poi_categories: Sequence[str] | None = None,
    progress: ProgressCallback | None = None,
    tile_cache_dir: str | Path | None = None,
    use_tile_cache: bool = True,
) -> dict[str, Any]:
    """Fetch every road tile, deduplicate it on disk, then atomically publish.

    Road coverage is all-or-nothing. Amenity tile failures are non-fatal
    because amenities do not affect road-cache computation.
    """
    target = Path(outpath)
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved = _resolve_fetch_profile(
        profile,
        include_amenities=include_amenities,
        poi_categories=poi_categories,
    )
    cache_root = _resolve_tile_cache_root(
        target,
        tile_cache_dir=tile_cache_dir,
        use_tile_cache=use_tile_cache,
    )
    lat_tiles = split_range(min_lat, max_lat, ROAD_TILE_LAT_SPAN)
    lon_tiles = split_range(min_lon, max_lon, ROAD_TILE_LON_SPAN)
    road_tiles_total = len(lat_tiles) * len(lon_tiles)
    road_tiles_done = 0
    road_tiles_ok = 0
    road_cache_hits = 0
    road_failures: list[str] = []
    amenity_tiling = _empty_amenity_tiling()

    with tempfile.TemporaryDirectory(prefix=f".{target.name}.tiles.", dir=target.parent) as temp:
        store = _OsmElementStore(Path(temp) / "elements.sqlite3")
        try:
            for lat_lo, lat_hi in lat_tiles:
                for lon_lo, lon_hi in lon_tiles:
                    bbox = f"({lat_lo},{lon_lo},{lat_hi},{lon_hi})"
                    result, cache_hit = _fetch_overpass_tile(
                        build_road_overpass_query(
                            bbox, road_classes=resolved.road_classes
                        ),
                        cache_root=cache_root,
                        kind="roads",
                        read_timeout=180.0,
                    )
                    road_tiles_done += 1
                    road_cache_hits += int(cache_hit)
                    if result.body is None:
                        road_failures.extend(result.failures)
                    else:
                        store.add_body(result.body)
                        road_tiles_ok += 1
                    if progress is not None:
                        progress(
                            {
                                "phase": "roads",
                                "current": road_tiles_done,
                                "total": road_tiles_total,
                                "tiles_ok": road_tiles_ok,
                                "cache_hits": road_cache_hits,
                            }
                        )

            nodes, ways = store.counts()
            road_tiling = {
                "ok": road_tiles_ok == road_tiles_total and nodes > 0 and ways > 0,
                "tiles_total": road_tiles_total,
                "tiles_ok": road_tiles_ok,
                "cache_hits": road_cache_hits,
                "failure_count": len(road_failures),
            }
            if not road_tiling["ok"]:
                if nodes == 0:
                    road_failures.append("tiled road fetch produced no nodes")
                if ways == 0:
                    road_failures.append("tiled road fetch produced no ways")
                return {
                    "ok": False,
                    "profile": resolved.name,
                    "road_tiling": road_tiling,
                    "amenity_tiling": amenity_tiling,
                    "failures": road_failures,
                }

            if resolved.include_amenities:
                amenity_lat_tiles = split_range(
                    min_lat, max_lat, AMENITY_TILE_LAT_SPAN
                )
                amenity_lon_tiles = split_range(
                    min_lon, max_lon, AMENITY_TILE_LON_SPAN
                )
                amenity_total = len(amenity_lat_tiles) * len(amenity_lon_tiles)
                amenity_done = 0
                amenity_ok = 0
                amenity_cache_hits = 0
                amenity_failures = 0
                amenity_nodes_before, _ = store.counts()
                for lat_lo, lat_hi in amenity_lat_tiles:
                    for lon_lo, lon_hi in amenity_lon_tiles:
                        bbox = f"({lat_lo},{lon_lo},{lat_hi},{lon_hi})"
                        result, cache_hit = _fetch_overpass_tile(
                            _amenity_query(bbox, resolved.poi_categories),
                            cache_root=cache_root,
                            kind="amenities",
                            read_timeout=120.0,
                        )
                        amenity_done += 1
                        amenity_cache_hits += int(cache_hit)
                        if result.body is None:
                            amenity_failures += len(result.failures)
                        else:
                            store.add_body(result.body)
                            amenity_ok += 1
                        if progress is not None:
                            progress(
                                {
                                    "phase": "amenities",
                                    "current": amenity_done,
                                    "total": amenity_total,
                                    "tiles_ok": amenity_ok,
                                    "cache_hits": amenity_cache_hits,
                                }
                            )
                amenity_nodes_after, _ = store.counts()
                amenity_tiling = {
                    "ok": amenity_ok == amenity_total,
                    "tiles_total": amenity_total,
                    "tiles_ok": amenity_ok,
                    "cache_hits": amenity_cache_hits,
                    "amenity_nodes_added": amenity_nodes_after
                    - amenity_nodes_before,
                    "failure_count": amenity_failures,
                }

            store.write(
                target,
                min_lat=min_lat,
                min_lon=min_lon,
                max_lat=max_lat,
                max_lon=max_lon,
            )
        finally:
            store.close()

    return {
        "ok": True,
        "profile": resolved.name,
        "road_tiling": road_tiling,
        "amenity_tiling": amenity_tiling,
        "failures": [],
        "validation": validate_osm_extract(target),
    }


def ensure_osm_has_bounds(
    osm_path: Path,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
) -> None:
    text = osm_path.read_text(encoding="utf-8")
    if re.search(r"<bounds", text):
        return
    match = re.search(r"<osm\b[^>]*>", text)
    if match is None:
        raise FetchError(f"No <osm> tag in {osm_path}")
    bounds_line = (
        f'<bounds minlat="{min_lat}" minlon="{min_lon}" '
        f'maxlat="{max_lat}" maxlon="{max_lon}"/>'
    )
    _atomic_write_text(
        osm_path,
        text[: match.end()] + "\n  " + bounds_line + "\n" + text[match.end() :],
    )


def _bbox_span_km(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float
) -> dict[str, float]:
    mean_lat = (min_lat + max_lat) / 2.0
    return {
        "latitude": abs(max_lat - min_lat) * 111.0,
        "longitude": abs(max_lon - min_lon)
        * 111.0
        * max(0.0, math.cos(math.radians(mean_lat))),
    }


def _download_validated(
    query: str,
    target: Path,
    *,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
) -> tuple[OverpassResult, dict[str, Any] | None]:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.download.",
            suffix=".osm",
            delete=False,
        ) as output:
            temporary = Path(output.name)
        result = download_overpass_query(query, temporary)
        if result.body is None:
            return result, None
        ensure_osm_has_bounds(
            temporary, min_lat, min_lon, max_lat, max_lon
        )
        try:
            validate_osm_extract(temporary)
        except FetchError as error:
            result.failures.append(str(error))
            result.body = None
            return result, None
        temporary.replace(target)
        temporary = None
        return result, validate_osm_extract(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_and_store_bbox_osm(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    outpath: str | Path,
    *,
    include_amenities: bool | None = None,
    profile: FetchProfileName | str | None = None,
    poi_categories: Sequence[str] | None = None,
    progress: ProgressCallback | None = None,
    tile_cache_dir: str | Path | None = None,
    use_tile_cache: bool = True,
) -> dict[str, Any]:
    """Fetch an arbitrary bbox, automatically tiling large or failed queries."""
    if min_lat >= max_lat or min_lon >= max_lon:
        raise FetchError("OSM bbox must have positive latitude and longitude spans")

    target = Path(outpath)
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved = _resolve_fetch_profile(
        profile,
        include_amenities=include_amenities,
        poi_categories=poi_categories,
    )
    bbox = {
        "minlat": min_lat,
        "minlon": min_lon,
        "maxlat": max_lat,
        "maxlon": max_lon,
    }
    bbox_text = f"({min_lat},{min_lon},{max_lat},{max_lon})"
    spans = _bbox_span_km(min_lat, min_lon, max_lat, max_lon)
    is_large = max(spans.values()) > LARGE_BBOX_SPAN_KM
    direct_failures: list[str] = []

    if not is_large:
        result, validation = _download_validated(
            build_road_overpass_query(
                bbox_text, road_classes=resolved.road_classes
            ),
            target,
            min_lat=min_lat,
            min_lon=min_lon,
            max_lat=max_lat,
            max_lon=max_lon,
        )
        direct_failures.extend(result.failures)
        if validation is not None:
            amenity_tiling = _empty_amenity_tiling()
            dataset_mode = "roads_only"
            if resolved.include_amenities:
                amenity_tiling = _fetch_direct_amenities(
                    min_lat=min_lat,
                    min_lon=min_lon,
                    max_lat=max_lat,
                    max_lon=max_lon,
                    outpath=target,
                    poi_categories=resolved.poi_categories,
                    progress=progress,
                    tile_cache_dir=tile_cache_dir,
                    use_tile_cache=use_tile_cache,
                )
                if not amenity_tiling["ok"]:
                    direct_amenity_failures = amenity_tiling["failure_count"]
                    amenity_tiling = fetch_tiled_amenities(
                        min_lat,
                        min_lon,
                        max_lat,
                        max_lon,
                        target,
                        poi_categories=resolved.poi_categories,
                        progress=progress,
                        tile_cache_dir=tile_cache_dir,
                        use_tile_cache=use_tile_cache,
                    )
                    amenity_tiling["failure_count"] += direct_amenity_failures
                if amenity_tiling["ok"]:
                    dataset_mode = "roads_and_amenities"
                elif amenity_tiling["tiles_ok"] > 0:
                    dataset_mode = "roads_plus_partial_amenities"
                else:
                    dataset_mode = "roads_only"
                validation = validate_osm_extract(target)
            warning = (
                "Some amenities could not be fetched; POI-based generation may "
                "have fewer candidates."
                if resolved.include_amenities and not amenity_tiling["ok"]
                else ""
            )
            return {
                "ok": True,
                "osm_path": str(target),
                "profile": resolved.name,
                "road_classes": (
                    list(resolved.road_classes)
                    if resolved.road_classes is not None
                    else None
                ),
                "poi_categories": (
                    list(resolved.poi_categories)
                    if resolved.poi_categories is not None
                    else None
                ),
                "dataset_mode": dataset_mode,
                "bbox": bbox,
                "bbox_span_km": spans,
                "road_tiling": _empty_road_tiling(),
                "amenity_tiling": amenity_tiling,
                "validation": validation,
                "warning": warning,
            }

    tiled = fetch_tiled_osm(
        min_lat,
        min_lon,
        max_lat,
        max_lon,
        target,
        profile=resolved.name,
        poi_categories=resolved.poi_categories,
        progress=progress,
        tile_cache_dir=tile_cache_dir,
        use_tile_cache=use_tile_cache,
    )
    if not tiled["ok"]:
        details = tiled.get("failures", []) + direct_failures
        raise FetchError(
            "Overpass could not provide complete road coverage: "
            + (" | ".join(details) or "no details")
        )

    amenity_tiling = tiled["amenity_tiling"]
    if not resolved.include_amenities:
        dataset_mode = "tiled_roads"
    elif amenity_tiling["ok"]:
        dataset_mode = "tiled_roads_and_amenities"
    elif amenity_tiling["amenity_nodes_added"] > 0:
        dataset_mode = "tiled_roads_plus_partial_amenities"
    else:
        dataset_mode = "tiled_roads_only"
    warning = (
        "Some amenities could not be fetched; POI-based generation may have "
        "fewer candidates."
        if resolved.include_amenities and not amenity_tiling["ok"]
        else ""
    )
    return {
        "ok": True,
        "osm_path": str(target),
        "profile": resolved.name,
        "road_classes": (
            list(resolved.road_classes)
            if resolved.road_classes is not None
            else None
        ),
        "poi_categories": (
            list(resolved.poi_categories)
            if resolved.poi_categories is not None
            else None
        ),
        "dataset_mode": dataset_mode,
        "bbox": bbox,
        "bbox_span_km": spans,
        "road_tiling": tiled["road_tiling"],
        "amenity_tiling": amenity_tiling,
        "validation": tiled["validation"],
        "warning": warning,
    }


def fetch_and_store_city_osm(
    city: str,
    *,
    country: str = "",
    osm_dir: str | Path = "osmdata",
    padding_km: float = 0.0,
    max_radius_km: float = 0.0,
    profile: FetchProfileName | str = "generation",
    poi_categories: Sequence[str] | None = None,
    progress: ProgressCallback | None = None,
    tile_cache_dir: str | Path | None = None,
    use_tile_cache: bool = True,
) -> dict[str, Any]:
    safe_city = sanitize_city_filename(city)
    osm_root = Path(osm_dir)
    osm_root.mkdir(parents=True, exist_ok=True)
    outpath = osm_root / f"{safe_city}.osm"

    min_lat, min_lon, max_lat, max_lon = fetch_city_bbox(city, country)
    if max_radius_km <= 0:
        spans = _bbox_span_km(min_lat, min_lon, max_lat, max_lon)
        if max(spans.values()) > MAX_UNCLAMPED_BBOX_SPAN_KM:
            country_hint = f" --country {country!r}" if country.strip() else ""
            raise FetchError(
                f"The geocoded bbox for {city!r} spans about "
                f"{max(spans.values()):.0f} km and is too large for a city extract. "
                f"Select the urban area explicitly, for example: "
                f"mamut-tools osm fetch-city {city!r}{country_hint} --max-radius-km 15"
            )
    else:
        center_lat, center_lon = fetch_city_center(city, country)
        clamp_dlat = max_radius_km / 111.0
        clamp_dlon = max_radius_km / max(
            1e-6, 111.0 * math.cos(math.radians(center_lat))
        )
        min_lat, max_lat = center_lat - clamp_dlat, center_lat + clamp_dlat
        min_lon, max_lon = center_lon - clamp_dlon, center_lon + clamp_dlon

    if padding_km > 0:
        dlat = padding_km / 111.0
        mean_lat = (min_lat + max_lat) / 2.0
        dlon = padding_km / max(
            1e-6, 111.0 * math.cos(math.radians(mean_lat))
        )
        min_lat -= dlat
        max_lat += dlat
        min_lon -= dlon
        max_lon += dlon

    summary = fetch_and_store_bbox_osm(
        min_lat,
        min_lon,
        max_lat,
        max_lon,
        outpath,
        profile=profile,
        poi_categories=poi_categories,
        progress=progress,
        tile_cache_dir=tile_cache_dir,
        use_tile_cache=use_tile_cache,
    )
    return {"city": safe_city, **summary}
