# MAMUT-routing-tools

Local generation tool suite for the [MAMUT-routing](https://github.com/ANR-MAMUT/MAMUT-routing) benchmark project: OSM city acquisition, a road-graph engine, BKS route-geometry materialization, and interactive CVRP/VRPTW instance generation. The public MAMUT-routing website is fully static; everything compute-heavy lives here and runs on your own machine.

Part of the [ANR MAMUT project](https://anr.fr/Projet-ANR-22-CE22-0016).

## Status

Beta. The tool suite is being extracted from the website's former Julia backend; interfaces may change between releases.

## Components

- `mamut-tools roadgraph`: build and inspect drivable road graphs from OSM XML extracts. The construction is a faithful Python port of the OpenStreetMapX.jl pipeline the project previously used (same road classes, oneway rules, intersection segmentation, ENU distances, and strongly-connected trim), so graphs and route geometry stay consistent with previously published data.
- `mamut-tools geometry`: materialize road-following polylines for Best-Known Solutions (BKS), in the exact artifact format the MAMUT-routing website consumes.
- `mamut-tools osm fetch-city`: download an OSM extract (roads + amenities) for a city by name, via Nominatim geocoding and Overpass with retry, roads-only fallback, and tiled amenity backfill.
- `mamut-tools generate`: interactive CVRP/VRPTW instance generation on city road graphs (single, bulk, preview, VRPTW derivation), the port of the historical MAMUT workbench generator.
- `mamut-tools solve`: PyVRP solving of generated and benchmark instances via mamut-routing-lib; with the `kayros` extra (`pip install 'mamut-routing-tools[kayros]'`), [KAYROS](https://pypi.org/project/kayros/) solves the time-dependent instances (Duration objective, anytime with exact certification tooling).
- `mamut-tools gui`: a CLI-owned local workbench GUI (loopback server with token security) for fetching cities, previewing, generating, solving, and rendering road-following routes on a map.
- Planned: the official time-dependent benchmark campaign pipeline.

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

### If the tool does not behave as documented

You are most likely running a different revision than you think. Two things to check, in order:

```bash
git pull --recurse-submodules   # update the repo AND the vendored submodule
uv sync                         # re-resolve dependencies afterwards
```

The `MAMUT-routing-lib` submodule is a frequent source of confusion: a conflict or a stale checkout there is easy to miss, and it leaves you on old behaviour with no obvious symptom. `git status` in the repository root reports a modified submodule; `git submodule update --init --recursive` puts it back on the pinned commit. Always run `uv sync` after pulling, since the dependency set moves between releases.

## Quick examples

```bash
# Road-graph statistics for a city extract
uv run mamut-tools roadgraph info path/to/City.osm

# Materialize a route-geometry group plan (website build contract)
uv run mamut-tools geometry materialize-plan plan.json --repo-root path/to/MAMUT-routing --result-dir out/
```
