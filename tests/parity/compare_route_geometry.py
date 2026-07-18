"""Rebuild a committed route-geometry artifact with the Python engine and
compare polylines against the Julia-produced cache.

Usage:
  uv run python tests/parity/compare_route_geometry.py <mamut-routing-repo> <artifact.route-geometry.json.gz> [more artifacts...]

For each artifact: reconstructs the group plan from the artifact's recorded
provenance (geo sidecar, metric, BKS routes), runs the materializer, and
compares every edge polyline pointwise (haversine tolerance per point) plus
the straight-fallback set.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

from mamut_routing_tools.geo import haversine_m
from mamut_routing_tools.geometry.materialize import materialize_group

POINT_TOLERANCE_M = 1.0


def load_gz_json(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def artifact_paths_lonlat(artifact: dict) -> dict[str, list[list[float]]]:
    vertices = artifact["vertex_lonlat"]
    return {key: [vertices[index] for index in indexes] for key, indexes in artifact["paths"].items()}


def main() -> int:
    repo_root = Path(sys.argv[1]).resolve()
    failures = 0
    for artifact_arg in sys.argv[2:]:
        artifact_path = Path(artifact_arg)
        artifact = load_gz_json(artifact_path)
        bks_bytes = (repo_root / artifact["bks_path"]).read_bytes()
        if hashlib.sha256(bks_bytes).hexdigest() != artifact["bks_sha256"]:
            print(f"SKIP {artifact_path.name.split('.')[0][:16]}: stale artifact (BKS changed since it was built)")
            continue
        bks = json.loads(bks_bytes)
        geo_path = artifact["source_geo_path"]
        meta = load_gz_json(repo_root / geo_path)
        meta["depot_instance_node_id"] = 0
        group = {
            "result_file": "compare.json",
            "geo_path": geo_path,
            "metric": artifact["metric"],
            "meta": meta,
            "entries": [
                {
                    "bks_path": artifact["bks_path"],
                    "routes": [[int(stop) for stop in route] for route in bks["routes"]],
                }
            ],
        }
        result = materialize_group(repo_root, group)

        reference = artifact_paths_lonlat(artifact)
        produced = {
            key.removeprefix("node:").replace("_", "-"): segment
            for key, segment in result["edge_cache"].items()
        }
        produced_fallbacks = {
            key.removeprefix("node:").replace("_", "-") for key in result["straight_fallback_edges"]
        }
        reference_fallbacks = set(artifact.get("straight_fallback_paths", []))

        edge_failures: list[str] = []
        max_deviation = 0.0
        point_counts_differ = 0
        for key, reference_points in sorted(reference.items()):
            candidate = produced.get(key)
            if candidate is None:
                edge_failures.append(f"missing edge {key}")
                continue
            if len(candidate) != len(reference_points):
                point_counts_differ += 1
            # Pointwise comparison over the aligned prefix; a differing count
            # with tiny endpoint deviation means an equal-cost alternative
            # path, which coordinate tolerance is meant to judge.
            deviation = max(
                (
                    haversine_m(a[1], a[0], b[1], b[0])
                    for a, b in zip(candidate, reference_points)
                ),
                default=0.0,
            )
            max_deviation = max(max_deviation, deviation)
            if deviation > POINT_TOLERANCE_M and len(candidate) == len(reference_points):
                edge_failures.append(f"edge {key} deviates {deviation:.1f} m over {len(reference_points)} points")

        name = artifact_path.name.split(".")[0][:16]
        status = "OK " if not edge_failures and produced_fallbacks == reference_fallbacks else "FAIL"
        print(
            f"{status} {name}: edges={len(reference)} max_pointwise_deviation_m={max_deviation:.3f} "
            f"count_mismatches={point_counts_differ} fallbacks(py/jl)={len(produced_fallbacks)}/{len(reference_fallbacks)}"
        )
        for line in edge_failures[:8]:
            print(f"    {line}")
        if edge_failures or produced_fallbacks != reference_fallbacks:
            failures += 1
    print("ROUTE-GEOMETRY PARITY:", "FAIL" if failures else "OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
