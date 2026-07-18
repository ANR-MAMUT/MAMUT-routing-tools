"""OSM fetch helpers (pure logic, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mamut_routing_tools.osm.fetch import (
    FetchError,
    _is_retryable,
    build_overpass_query,
    merge_nodes_into_osm,
    sanitize_city_filename,
    split_range,
)


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
    full = build_overpass_query(bbox, include_amenities=True)
    assert 'way["highway"]' + bbox in full
    assert 'node["amenity"]' + bbox in full
    roads = build_overpass_query(bbox, include_amenities=False)
    assert "amenity" not in roads


def test_retryable_classification() -> None:
    assert _is_retryable(429, "")
    assert _is_retryable(200, "runtime error: query timeout")
    assert not _is_retryable(400, "static error: parse error")


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
