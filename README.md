# MAMUT-routing-tools

Local generation tool suite for the [MAMUT-routing](https://github.com/ANR-MAMUT/MAMUT-routing) benchmark project: OSM city acquisition, a road-graph engine, BKS route-geometry materialization, and interactive CVRP/VRPTW instance generation. The public MAMUT-routing website is fully static; everything compute-heavy lives here and runs on your own machine.

Part of the [ANR MAMUT project](https://mamut.roadef.org/).

## Status

Beta. The tool suite is being extracted from the website's former Julia backend; interfaces may change between releases.

## Components

- `mamut-tools roadgraph`: build and inspect drivable road graphs from OSM XML extracts. The construction is a faithful Python port of the OpenStreetMapX.jl pipeline the project previously used (same road classes, oneway rules, intersection segmentation, ENU distances, and strongly-connected trim), so graphs and route geometry stay consistent with previously published data.
- `mamut-tools geometry`: materialize road-following polylines for best-known solutions, in the exact artifact format the MAMUT-routing website consumes.
- Planned: OSM city fetch (Nominatim + Overpass), interactive CVRP/VRPTW generation with a local workbench GUI, and the official time-dependent benchmark campaign pipeline.

## Install

Requires Python >= 3.11 and [uv](https://github.com/astral-sh/uv).

```bash
git clone --recurse-submodules https://github.com/ANR-MAMUT/MAMUT-routing-tools.git
cd MAMUT-routing-tools
uv sync
uv run mamut-tools --help
```

The `MAMUT-routing-lib` contract library is vendored as a git submodule and installed as an editable dependency.

## Quick examples

```bash
# Road-graph statistics for a city extract
uv run mamut-tools roadgraph info path/to/City.osm

# Materialize a route-geometry group plan (website build contract)
uv run mamut-tools geometry materialize-plan plan.json --repo-root path/to/MAMUT-routing --result-dir out/
```
