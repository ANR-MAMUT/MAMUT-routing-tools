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
  <node id="1" lat="45.000" lon="4.000">
    <tag k="amenity" v="restaurant"/>
    <tag k="name" v="Chez Un"/>
  </node>
  <node id="2" lat="45.000" lon="4.005">
    <tag k="amenity" v="cafe"/>
    <tag k="name" v="Caf&#233; Deux"/>
  </node>
  <node id="3" lat="45.000" lon="4.008">
    <tag k="amenity" v="pharmacy"/>
  </node>
  <node id="4" lat="45.005" lon="4.005">
    <tag k="amenity" v="school"/>
    <tag k="name" v="&#201;cole Quatre"/>
  </node>
  <node id="5" lat="44.995" lon="4.005">
    <tag k="amenity" v="bar"/>
    <tag k="name" v="Bar Cinq"/>
  </node>
  <node id="6" lat="45.000" lon="4.020"/>
  <node id="7" lat="45.007" lon="4.001"/>
  <node id="8" lat="45.007" lon="4.002"/>
  <!-- A bank beside the disconnected island: its nearest ROAD NODE (7) is not a
       graph vertex, so the strict rule discards it even though a real vertex
       stands a few hundred metres away. This is the Westport Inn case. -->
  <node id="12" lat="45.0070" lon="4.0015">
    <tag k="amenity" v="bank"/>
    <tag k="name" v="Banque de l&apos;&#238;le"/>
  </node>
  <!-- An amenity nowhere near a road: exercises the unreachable-pick path. -->
  <node id="9" lat="45.500" lon="4.500">
    <tag k="amenity" v="museum"/>
    <tag k="name" v="Mus&#233;e Lointain"/>
  </node>
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
  <!-- A pub drawn as a building outline: no lat/lon of its own, located by the
       <center> Overpass computes for it. It reuses id 4, which node 4 (the
       school) also has, because OSM ids are only unique per element type. -->
  <way id="4">
    <center lat="45.0056" lon="4.005"/>
    <tag k="amenity" v="pub"/>
    <tag k="name" v="Le Quatre"/>
  </way>
  <relation id="20">
    <center lat="45.0006" lon="4.008"/>
    <tag k="amenity" v="marketplace"/>
    <tag k="name" v="March&#233; Vingt"/>
  </relation>
</osm>
"""


@pytest.fixture()
def fixture_osm_path(tmp_path: Path) -> Path:
    path = tmp_path / "Testville.osm"
    path.write_text(FIXTURE_OSM, encoding="utf-8")
    return path
