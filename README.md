# MAMUT-routing-tools

Local generation tool suite for the [MAMUT-routing](https://github.com/ANR-MAMUT/MAMUT-routing) benchmark project: OSM city acquisition, a road-graph engine, BKS route-geometry materialization, and interactive CVRP / VRPTW / time-dependent (TDVRP, TDVRPTW) instance generation. The public MAMUT-routing website is fully static; everything compute-heavy lives here and runs on your own machine.

Part of the [ANR MAMUT project](https://anr.fr/Projet-ANR-22-CE22-0016).

## Status

Beta. All benchmark generation now lives here: the website's former Julia backend has been fully ported to Python and removed. Interfaces may still change between releases.

## Components

- `mamut-tools roadgraph`: build and inspect drivable road graphs from OSM XML extracts. The construction is a faithful Python port of the OpenStreetMapX.jl pipeline the project previously used (same road classes, oneway rules, intersection segmentation, ENU distances, and strongly-connected trim), so graphs and route geometry stay consistent with previously published data.
- `mamut-tools geometry`: materialize road-following polylines for Best-Known Solutions (BKS), in the exact artifact format the MAMUT-routing website consumes.
- `mamut-tools osm fetch-city`: download and structurally validate a purpose-filtered OSM extract for a city by name, using atomic tiled road and POI acquisition plus a persistent tile cache when a single Overpass query would be too large.
- `mamut-tools generate`: per-instance generation on city road graphs — `single` (CVRP), `preview`, `derive-vrptw` (the fastest-metric VRPTW twin), and `derive-td` (the TDVRP + TDVRPTW twins: traffic overlay → arrival-time functions → time-window lift to time-dependent feasibility). Batch family generation (many cities × sizes) is delegated to per-campaign scripts that call the `mamut_routing_tools.family` and `mamut_routing_tools.td` library.
- `mamut-tools solve`: PyVRP solving of generated and benchmark instances via mamut-routing-lib; with the `kayros` extra (`pip install 'mamut-routing-tools[kayros]'`), [KAYROS](https://pypi.org/project/kayros/) solves the time-dependent instances (Duration objective, anytime with exact certification tooling).
- `mamut-tools gui`: a CLI-owned local workbench GUI (loopback server with token security) for fetching cities, previewing, generating, solving, and rendering road-following routes on a map. Long operations run as persistent jobs with real state/logs; solver runs are checker-validated, retained across restarts, and comparable by objective, fleet, loads, route edges, and customer grouping.
The traffic models (`bpr` commuter simulation, `wave` rush-hour dip) and the road-graph time-dependent travel model live in `mamut_routing_tools.td`; the family build engine (base publish, VRPTW derivation, TD-twin materialization) lives in `mamut_routing_tools.family`.

## Install

Requires Python >= 3.11. Two variants:

### Option A — from PyPI (recommended for users)

Published on [PyPI](https://pypi.org/project/mamut-routing-tools/). With [uv](https://github.com/astral-sh/uv), no installation step is needed:

```bash
uvx --from mamut-routing-tools mamut-tools --help
```

Or install it into an environment:

```bash
pip install mamut-routing-tools
# or
uv add mamut-routing-tools
```

### Option B — from source (recommended for contributors)

Clone the repository and use the project environment:

```bash
git clone --recurse-submodules https://github.com/ANR-MAMUT/MAMUT-routing-tools.git
cd MAMUT-routing-tools
uv sync
uv run mamut-tools --help
uv run pytest
```

In both variants the `MAMUT-routing-lib` contract library resolves from PyPI; the vendored submodule checkout exists for contract reference and unreleased-lib development.

## Onboarding: discovering the CLI

Everything in this suite is reachable from the single `mamut-tools` entry point, and **every level of the command tree answers `--help`**. You do not need to hunt through this README for a flag: ask the CLI directly.

```bash
uv run mamut-tools --help              # top level: lists all command groups
uv run mamut-tools gui --help          # a command group: lists its sub-commands
uv run mamut-tools gui start --help    # a sub-command: its options and defaults
```

To find out which build you are actually running, use `--version` (or `-V`):

```bash
uv run mamut-tools --version
# mamut-tools 0.3.3 (/path/to/MAMUT-routing-tools/src/mamut_routing_tools)
```

It prints the version alongside the package location, which tells you whether you are on a PyPI install or an editable source checkout.

The top level lists the command groups (`roadgraph`, `geometry`, `osm`, `generate`, `solve`, `gui`); drilling down one level at a time is the intended way to explore. When in doubt, add `--help` to whatever you just typed.

### Starting and stopping the workbench GUI

The GUI is the friendliest way to fetch a city, generate instances, solve them, and see routes drawn on a map. The CLI owns the server process: `start` launches it as a detached background process and returns immediately, so your shell stays free.

```bash
uv run mamut-tools gui start
```

This prints a URL carrying the access token for that server instance, and opens it in your browser:

```
Workbench GUI running (pid 391337), workspace /path/to/.cache/mamut-tools
http://127.0.0.1:39117/?token=<token>
```

The port is picked automatically and the server binds to loopback only, so it is never reachable from outside your machine. Useful options: `--port <N>` to pin a port, `--no-open` to skip opening the browser (handy over SSH), and `--output-dir <DIR>` to choose the workspace directory holding generated instances.

Check on it or shut it down with:

```bash
uv run mamut-tools gui status   # running? healthy? which URL and workspace?
uv run mamut-tools gui stop     # terminate the background server
```

`gui status` reprints the tokened URL, which is the quickest way to recover it if you lose the browser tab. If you would rather watch the server logs live, `gui run` runs it in the foreground instead (development mode, stop with Ctrl-C).

Generated instances remain under `<workspace>/instances/`. Generation controls include the historical POI amenity selection and random, centered, or excentered depot placement. Hybrid sampling exposes its target POI/parametric proportion; parametric sampling exposes the customer distribution, number of clusters, and clustering radius/decay distance. The GUI keeps its additional durable state separately:

- validated solver runs under `<workspace>/solutions/<instance-id>/`;
- job records under `<workspace>/state/jobs/`;
- append-only job logs under `<workspace>/state/logs/`.

Both instances and solutions remain available after the GUI or machine restarts, until their workspace files are removed. Selecting an instance immediately displays its depot and customer positions without requiring a solve. Select any saved run to render it again—the customer markers then adopt their route colors—or compare two runs with the same objective and metric to inspect cost and route-count deltas, route loads, changed directed edges, and changes to customer grouping. Cancellation is cooperative: queued work stops immediately, while a running solver or matrix calculation stops at its next safe checkpoint.

The GUI fetches every POI category shown in its category picker when acquiring a city. Category checkboxes filter generation only, so changing them later does not require another OSM download. The lower-level `mamut-tools osm fetch-city` command remains configurable through repeated `--poi-category` options.

### If the tool does not behave as documented

You are most likely running a different revision than you think. Start by asking the tool itself, then bring the checkout up to date:

```bash
uv run mamut-tools --version    # which version, and from which directory?
git pull --recurse-submodules   # update the repo AND the vendored submodule
uv sync                         # re-resolve dependencies afterwards
```

The `MAMUT-routing-lib` submodule is a frequent source of confusion: a conflict or a stale checkout there is easy to miss, and it leaves you on old behaviour with no obvious symptom. `git status` in the repository root reports a modified submodule; `git submodule update --init --recursive` puts it back on the pinned commit. Always run `uv sync` after pulling, since the dependency set moves between releases.

## Quick examples

```bash
# Fetch Tokyo's urban area into ./osmdata (the administrative bbox includes
# distant islands, so explicitly clamp it around the geocoded city point)
uv run mamut-tools osm fetch-city Tokyo --country Japan --max-radius-km 15

# Road-cache builds skip POIs and download only road classes used by the engine
uv run mamut-tools osm fetch-city Tokyo --country Japan --max-radius-km 15 --profile road_cache

# Generation defaults to the seven built-in POI categories; override them by
# repeating --poi-category
uv run mamut-tools osm fetch-city Lyon --profile generation \
  --poi-category restaurant --poi-category cafe

# When running inside MAMUT-routing-tools, target the parent site's data folder
uv run mamut-tools osm fetch-city Tokyo --country Japan --max-radius-km 15 --osm-dir ../osmdata

# Verify that an extract has bounds, nodes and ways and contains no error remark
uv run mamut-tools osm validate ../osmdata/Tokyo.osm

# Road-graph statistics for a city extract
uv run mamut-tools roadgraph info path/to/City.osm

# Materialize a route-geometry group plan (website build contract)
uv run mamut-tools geometry materialize-plan plan.json --repo-root path/to/MAMUT-routing --result-dir out/
```

### OSM download profiles

- `generation` (default) downloads only the 16 road classes understood by the road engine, skeleton coordinates for their referenced nodes, and selected POI nodes. Roads and POIs use separate Overpass queries, so a POI failure cannot invalidate complete road data.
- `road_cache` downloads the filtered road network without POIs. This is the profile used by the MAMUT-routing site build.
- `full` retains the broad `highway=*` and `amenity=*` behavior for compatibility.

Successful tile responses are validated and cached under `<osm-dir>/.mamut-osm-tile-cache`. Repeating an interrupted request reuses those tiles; pass `--no-tile-cache` to disable reuse or `--tile-cache-dir` to choose another location.
