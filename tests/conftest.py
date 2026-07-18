from __future__ import annotations

from pathlib import Path

import pytest

# Synthetic city: a bidirectional cross inside bounds, a oneway spur that
# crosses the east boundary (exercises the crop with a synthetic boundary
# node), a disconnected two-node island (exercises the SCC trim), plus a
# footway and an invisible way that the road filter must drop.
FIXTURE_OSM = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="test">
  <bounds minlat="44.99" minlon="3.99" maxlat="45.01" maxlon="4.01"/>
  <node id="1" lat="45.000" lon="4.000"/>
  <node id="2" lat="45.000" lon="4.005"/>
  <node id="3" lat="45.000" lon="4.008"/>
  <node id="4" lat="45.005" lon="4.005"/>
  <node id="5" lat="44.995" lon="4.005"/>
  <node id="6" lat="45.000" lon="4.020"/>
  <node id="7" lat="45.007" lon="4.001"/>
  <node id="8" lat="45.007" lon="4.002"/>
  <way id="10">
    <nd ref="1"/><nd ref="2"/><nd ref="3"/>
    <tag k="highway" v="residential"/>
  </way>
  <way id="11">
    <nd ref="4"/><nd ref="2"/><nd ref="5"/>
    <tag k="highway" v="primary"/>
  </way>
  <way id="12">
    <nd ref="3"/><nd ref="6"/>
    <tag k="highway" v="service"/>
    <tag k="oneway" v="yes"/>
  </way>
  <way id="13">
    <nd ref="7"/><nd ref="8"/>
    <tag k="highway" v="residential"/>
  </way>
  <way id="14">
    <nd ref="1"/><nd ref="4"/>
    <tag k="highway" v="footway"/>
  </way>
  <way id="15">
    <nd ref="1"/><nd ref="5"/>
    <tag k="highway" v="residential"/>
    <tag k="visible" v="false"/>
  </way>
</osm>
"""


@pytest.fixture()
def fixture_osm_path(tmp_path: Path) -> Path:
    path = tmp_path / "Testville.osm"
    path.write_text(FIXTURE_OSM, encoding="utf-8")
    return path
