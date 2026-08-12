"""OSM fetch helpers (pure logic, no network)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mamut_routing_tools.osm import fetch as fetch_module
from mamut_routing_tools.osm.fetch import (
    FetchError,
    OverpassResult,
    _is_retryable,
    _tile_cache_path,
    build_amenity_overpass_query,
    build_overpass_query,
    build_road_overpass_query,
    fetch_and_store_bbox_osm,
    fetch_tiled_osm,
    merge_nodes_into_osm,
    sanitize_city_filename,
    split_range,
    validate_osm_extract,
)
from mamut_routing_tools.roadgraph.classes import ROAD_CLASSES
from mamut_routing_tools.roadgraph.osmxml import parse_osm


def test_sanitize_city_filename() -> None:
    assert sanitize_city_filename("  Le  Mans ") == "Le Mans"
    assert sanitize_city_filename("A/B:C*D") == "A_B_C_D"
    with pytest.raises(FetchError):
        sanitize_city_filename("   ")
    with pytest.raises(FetchError):
        sanitize_city_filename("..")


def test_split_range() -> None:
    tiles = split_range(0.0, 0.1, 0.03)
    assert len(tiles) == 4
    assert tiles[0][0] == 0.0
    assert tiles[-1][1] == 0.1
    assert split_range(1.0, 1.0, 0.03) == [(1.0, 1.0)]


def test_overpass_query_shapes() -> None:
    bbox = "(44.99,3.99,45.01,4.01)"
    roads = build_road_overpass_query(bbox)
    assert 'way["highway"~' in roads
    assert bbox in roads
    assert "footway" not in roads
    assert all(road_class in roads for road_class in ROAD_CLASSES)
    assert "out body qt;" in roads
    assert "out skel qt;" in roads
    assert "amenity" not in roads

    pois = build_amenity_overpass_query(bbox)
    # Amenities are mapped as nodes, building outlines and multipolygons alike;
    # out center gives the latter two a point without their geometry.
    assert 'node["amenity"~' in pois
    assert 'way["amenity"~' in pois
    assert 'relation["amenity"~' in pois
    assert "out center qt;" in pois
    assert "restaurant" in pois and "parking" not in pois

    extended_pois = build_amenity_overpass_query(
        bbox,
        poi_categories=("hospital", "library", "charging_station"),
    )
    assert "hospital" in extended_pois
    assert "library" in extended_pois
    assert "charging_station" in extended_pois
    assert "restaurant" not in extended_pois
    assert _tile_cache_path(Path("cache"), pois, kind="amenities") != _tile_cache_path(
        Path("cache"), extended_pois, kind="amenities"
    )

    combined = build_overpass_query(bbox)
    assert 'way["highway"~' in combined
    assert 'node["amenity"~' in combined
    assert 'way["amenity"~' in combined

    full = build_overpass_query(bbox, profile="full")
    assert 'way["highway"]' + bbox in full
    assert 'node["amenity"]' + bbox in full
    assert 'relation["amenity"]' + bbox in full


def test_fetch_profiles_validate_conflicts() -> None:
    bbox = "(44.99,3.99,45.01,4.01)"
    with pytest.raises(FetchError, match="Unknown OSM fetch profile"):
        build_overpass_query(bbox, profile="unknown")
    with pytest.raises(FetchError, match="conflicts"):
        build_overpass_query(
            bbox, profile="road_cache", include_amenities=True
        )
    with pytest.raises(FetchError, match="poi_categories"):
        build_overpass_query(
            bbox, profile="road_cache", poi_categories=["cafe"]
        )


def test_retryable_classification() -> None:
    assert _is_retryable(429, "")
    assert _is_retryable(200, "runtime error: query timeout")
    assert _is_retryable(200, "runtime error: Query ran out of memory in recurse")
    assert not _is_retryable(400, "static error: parse error")


def test_http_200_overpass_remark_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    body = (
        '<?xml version="1.0"?><osm version="0.6">'
        '<remark>runtime error: Query ran out of memory in recurse</remark></osm>'
    )
    monkeypatch.setattr(fetch_module, "OVERPASS_ENDPOINTS", ["https://example.test"])
    monkeypatch.setattr(
        fetch_module.httpx,
        "post",
        lambda *args, **kwargs: SimpleNamespace(status_code=200, text=body),
    )
    monkeypatch.setattr(fetch_module.time, "sleep", lambda _seconds: None)

    result = fetch_module.fetch_overpass_body(
        'way["highway"](1,2,3,4);out body;', attempts_per_endpoint=1
    )

    assert result.body is None
    assert any("out of memory" in failure for failure in result.failures)


def test_validate_osm_rejects_overpass_remark(tmp_path: Path) -> None:
    osm = tmp_path / "failed.osm"
    osm.write_text(
        '<?xml version="1.0"?><osm version="0.6">'
        '<bounds minlat="1" minlon="2" maxlat="3" maxlon="4"/>'
        '<remark>runtime error: timeout</remark></osm>',
        encoding="utf-8",
    )

    with pytest.raises(FetchError, match="error remark"):
        validate_osm_extract(osm)


def test_oversized_city_bbox_requires_explicit_clamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        fetch_module,
        "fetch_city_bbox",
        lambda city, country="": (20.2145811, 135.8536855, 35.8984245, 154.205541),
    )

    with pytest.raises(FetchError, match=r"--max-radius-km 15"):
        fetch_module.fetch_and_store_city_osm("Tokyo", osm_dir=tmp_path)

    assert not (tmp_path / "Tokyo.osm").exists()


def _road_tile(
    nodes: list[tuple[int, float, float]], way_id: int
) -> str:
    node_xml = "".join(
        f'<node id="{node_id}" lat="{lat}" lon="{lon}"/>'
        for node_id, lat, lon in nodes
    )
    refs = "".join(f'<nd ref="{node_id}"/>' for node_id, _lat, _lon in nodes)
    return (
        '<?xml version="1.0"?><osm version="0.6">'
        f"{node_xml}<way id=\"{way_id}\">{refs}"
        '<tag k="highway" v="residential"/></way></osm>'
    )


def test_tiled_fetch_deduplicates_and_writes_parser_compatible_osm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies = iter(
        [
            _road_tile([(1, 45.01, 4.01), (2, 45.06, 4.02)], 10),
            _road_tile([(2, 45.06, 4.02), (3, 45.12, 4.03)], 11),
        ]
    )
    monkeypatch.setattr(
        fetch_module,
        "fetch_overpass_body",
        lambda *args, **kwargs: OverpassResult(body=next(bodies)),
    )
    target = tmp_path / "city.osm"

    summary = fetch_tiled_osm(
        45.0, 4.0, 45.13, 4.1, target, include_amenities=False
    )

    assert summary["ok"] is True
    assert summary["road_tiling"]["tiles_total"] == 2
    assert summary["validation"]["nodes"] == 3
    assert summary["validation"]["ways"] == 2
    text = target.read_text(encoding="utf-8")
    assert text.count('<node id="2"') == 1
    assert text.index("<node") < text.index("<way")
    parsed = parse_osm(target)
    assert sorted(parsed.nodes) == [1, 2, 3]
    assert [way.way_id for way in parsed.ways] == [10, 11]


def test_tiled_fetch_reuses_validated_persistent_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies = iter(
        [
            _road_tile([(1, 45.01, 4.01), (2, 45.06, 4.02)], 10),
            _road_tile([(2, 45.06, 4.02), (3, 45.12, 4.03)], 11),
        ]
    )
    calls = 0

    def first_fetch(*args, **kwargs) -> OverpassResult:
        nonlocal calls
        calls += 1
        return OverpassResult(body=next(bodies))

    monkeypatch.setattr(fetch_module, "fetch_overpass_body", first_fetch)
    cache_dir = tmp_path / "tile-cache"
    first = fetch_tiled_osm(
        45.0,
        4.0,
        45.13,
        4.1,
        tmp_path / "first.osm",
        profile="road_cache",
        tile_cache_dir=cache_dir,
    )
    assert calls == 2
    assert first["road_tiling"]["cache_hits"] == 0

    def unexpected_fetch(*args, **kwargs) -> OverpassResult:
        raise AssertionError("cached tiles should avoid another Overpass request")

    monkeypatch.setattr(fetch_module, "fetch_overpass_body", unexpected_fetch)
    second = fetch_tiled_osm(
        45.0,
        4.0,
        45.13,
        4.1,
        tmp_path / "second.osm",
        profile="road_cache",
        tile_cache_dir=cache_dir,
    )
    assert second["road_tiling"]["cache_hits"] == 2
    assert second["validation"]["ways"] == 2


def test_incomplete_tiled_roads_do_not_replace_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "city.osm"
    target.write_text("existing", encoding="utf-8")
    results = iter(
        [
            OverpassResult(body=_road_tile([(1, 45.01, 4.01)], 10)),
            OverpassResult(body=None, failures=["tile failed"]),
        ]
    )
    monkeypatch.setattr(
        fetch_module,
        "fetch_overpass_body",
        lambda *args, **kwargs: next(results),
    )

    summary = fetch_tiled_osm(
        45.0, 4.0, 45.13, 4.1, target, include_amenities=False
    )

    assert summary["ok"] is False
    assert target.read_text(encoding="utf-8") == "existing"


def test_large_bbox_uses_tiled_fetch_without_monolithic_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "Tokyo.osm"

    def fake_tiled(*args, **kwargs) -> dict:
        target.write_text("mock", encoding="utf-8")
        return {
            "ok": True,
            "road_tiling": {
                "ok": True,
                "tiles_total": 9,
                "tiles_ok": 9,
                "failure_count": 0,
            },
            "amenity_tiling": {
                "ok": False,
                "tiles_total": 0,
                "tiles_ok": 0,
                "amenity_nodes_added": 0,
                "failure_count": 0,
            },
            "validation": {"nodes": 100, "ways": 20},
        }

    monkeypatch.setattr(fetch_module, "fetch_tiled_osm", fake_tiled)
    monkeypatch.setattr(
        fetch_module,
        "download_overpass_query",
        lambda *args, **kwargs: pytest.fail("large bbox used a monolithic query"),
    )

    summary = fetch_and_store_bbox_osm(
        35.55,
        139.60,
        35.81,
        139.92,
        target,
        include_amenities=False,
    )

    assert summary["dataset_mode"] == "tiled_roads"
    assert summary["road_tiling"]["tiles_total"] == 9


def test_small_generation_fetch_splits_roads_and_filtered_pois(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queries: list[str] = []
    road_body = _road_tile(
        [(1, 45.0, 4.0), (2, 45.005, 4.005)], 10
    )
    poi_body = (
        '<?xml version="1.0"?><osm version="0.6">'
        '<node id="2" lat="45.005" lon="4.005">'
        '<tag k="amenity" v="cafe"/></node>'
        '<node id="3" lat="45.006" lon="4.006">'
        '<tag k="amenity" v="restaurant"/></node>'
        # A restaurant drawn as a building outline: no position of its own, so
        # Overpass reports the centre it computed.
        '<way id="77"><center lat="45.007" lon="4.007"/>'
        '<tag k="amenity" v="restaurant"/><tag k="name" v="Chez Way"/></way>'
        "</osm>"
    )

    def fake_fetch(query: str, **kwargs) -> OverpassResult:
        queries.append(query)
        return OverpassResult(
            body=road_body if 'way["highway"' in query else poi_body
        )

    monkeypatch.setattr(fetch_module, "fetch_overpass_body", fake_fetch)
    target = tmp_path / "generation.osm"
    summary = fetch_and_store_bbox_osm(
        44.99,
        3.99,
        45.01,
        4.01,
        target,
        profile="generation",
        poi_categories=["cafe", "restaurant"],
        use_tile_cache=False,
    )

    assert len(queries) == 2
    assert 'way["highway"~' in queries[0]
    assert "amenity" not in queries[0]
    assert 'node["amenity"~' in queries[1]
    assert summary["profile"] == "generation"
    assert summary["dataset_mode"] == "roads_and_amenities"
    assert summary["poi_categories"] == ["cafe", "restaurant"]
    text = target.read_text(encoding="utf-8")
    assert text.count('id="2"') == 1
    assert '<tag k="amenity" v="cafe"/>' in text
    assert 'id="3"' in text
    # The way-mapped restaurant survives into the extract, centre and all, and
    # does not displace the road way it shares the file with.
    assert '<way id="77">' in text and '<center lat="45.007"' in text
    assert '<tag k="name" v="Chez Way"/>' in text
    assert summary["amenity_tiling"]["amenity_ways_added"] == 1


def test_merge_nodes_into_osm(tmp_path: Path) -> None:
    osm = tmp_path / "city.osm"
    osm.write_text(
        '<?xml version="1.0"?>\n<osm version="0.6">\n'
        '  <node id="1" lat="45.0" lon="4.0"/>\n'
        "</osm>\n",
        encoding="utf-8",
    )
    added = merge_nodes_into_osm(
        osm,
        [
            '<node id="1" lat="45.0" lon="4.0"/>',
            '<node id="2" lat="45.1" lon="4.1"><tag k="amenity" v="cafe"/></node>',
        ],
    )
    assert added == 1  # id 1 already present
    text = osm.read_text(encoding="utf-8")
    assert 'id="2"' in text
    assert text.rstrip().endswith("</osm>")


def test_merge_nodes_enriches_existing_skeleton_node(tmp_path: Path) -> None:
    osm = tmp_path / "city.osm"
    osm.write_text(
        '<?xml version="1.0"?><osm version="0.6">'
        '<node id="1" lat="45.0" lon="4.0"/>'
        '<way id="10"><nd ref="1"/><tag k="highway" v="road"/></way>'
        "</osm>",
        encoding="utf-8",
    )

    added = merge_nodes_into_osm(
        osm,
        [
            '<node id="1" lat="45.0" lon="4.0">'
            '<tag k="amenity" v="cafe"/></node>'
        ],
    )

    assert added == 0
    text = osm.read_text(encoding="utf-8")
    assert text.count('id="1"') == 1
    assert 'amenity" v="cafe' in text


def test_audit_extract_separates_point_and_outline_pois(
    tmp_path: Path, fixture_osm_path: Path
) -> None:
    from mamut_routing_tools.osm import audit_extract, audit_extracts

    report = audit_extract(fixture_osm_path)
    assert report["city"] == "Testville"
    assert report["poi_nodes"] == 7 and report["poi_ways"] == 1 and report["poi_relations"] == 1
    # An extract holding way-mapped amenities was fetched by a version that asks
    # for them, so it has nothing to gain from a refresh.
    assert report["status"] == "complete" and report["can_refresh"] is True
    assert report["bounds"]["minlat"] == 44.99

    # The signature of an extract fetched before ways were queried: amenities,
    # but every one of them on a node.
    (tmp_path / "Oldtown.osm").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6">\n'
        '  <bounds minlat="44.99" minlon="3.99" maxlat="45.01" maxlon="4.01"/>\n'
        '<node id="1" lat="45.0" lon="4.0"><tag k="amenity" v="cafe"/></node>\n'
        '<way id="10"><nd ref="1"/><tag k="highway" v="residential"/></way>\n'
        "</osm>\n",
        encoding="utf-8",
    )
    listed = {entry["city"]: entry for entry in audit_extracts(tmp_path)}
    assert listed["Oldtown"]["poi_nodes"] == 1 and listed["Oldtown"]["poi_ways"] == 0
    assert listed["Oldtown"]["status"] == "nodes_only"
    assert listed["Testville"]["status"] == "complete"


def test_audit_reports_an_extract_it_cannot_read(tmp_path: Path) -> None:
    from mamut_routing_tools.osm import audit_extracts

    (tmp_path / "Broken.osm").write_text("<osm><node id=", encoding="utf-8")
    (entry,) = audit_extracts(tmp_path)
    # One unreadable file must not hide the rest of the workspace.
    assert entry["status"] == "unreadable" and entry["can_refresh"] is False


def test_refresh_extract_backfills_only_the_amenity_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mamut_routing_tools.osm import refresh_extract_pois

    target = tmp_path / "Pointville.osm"
    target.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6">\n'
        '  <bounds minlat="44.99" minlon="3.99" maxlat="45.01" maxlon="4.01"/>\n'
        '<node id="1" lat="45.0" lon="4.0"><tag k="amenity" v="cafe"/></node>\n'
        '<node id="2" lat="45.001" lon="4.001"/>\n'
        '<way id="10"><nd ref="1"/><nd ref="2"/><tag k="highway" v="residential"/></way>\n'
        "</osm>\n",
        encoding="utf-8",
    )
    queries: list[str] = []

    def fake_fetch(query: str, **kwargs) -> OverpassResult:
        queries.append(query)
        return OverpassResult(
            body='<?xml version="1.0"?><osm version="0.6">'
            '<way id="88"><center lat="45.002" lon="4.002"/>'
            '<tag k="amenity" v="restaurant"/></way></osm>'
        )

    monkeypatch.setattr(fetch_module, "fetch_overpass_body", fake_fetch)
    report = refresh_extract_pois(target, use_tile_cache=False)

    assert report["ok"] and report["gained_ways"] >= 1
    assert report["status_before"] == "nodes_only" and report["status_after"] == "complete"
    assert report["poi_total_after"] > report["poi_total_before"]
    # Roads are never re-downloaded, and the road way keeps its node refs.
    assert queries and all("highway" not in query for query in queries)
    text = target.read_text(encoding="utf-8")
    assert '<way id="10"><nd ref="1"/>' in text and '<way id="88">' in text


def test_refresh_extract_refuses_a_file_without_bounds(tmp_path: Path) -> None:
    from mamut_routing_tools.osm import refresh_extract_pois

    target = tmp_path / "Boundless.osm"
    target.write_text(
        '<?xml version="1.0"?><osm version="0.6">'
        '<node id="1" lat="45.0" lon="4.0"><tag k="amenity" v="cafe"/></node></osm>',
        encoding="utf-8",
    )
    report = refresh_extract_pois(target)
    assert report["ok"] is False and "bounds" in report["error"]
