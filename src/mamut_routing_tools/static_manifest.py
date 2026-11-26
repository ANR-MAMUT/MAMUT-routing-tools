"""Stage a static family edition and certify its incumbent with both exact checkers.

Run with `python -m mamut_routing_tools.static_manifest --help`.
A descriptor and incumbent are read from one family. All publications are
written to a separate staging tree in the same family-relative layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from mamut_routing_lib.artifacts import (
    get_bks_path_for_instance, hydrate_collection_instance, instance_problem_type,
    load_benchmark_instance, load_bks, write_bks_file,
)
from mamut_routing_lib.exact import evaluate_exact
from mamut_routing_lib.exact_store import evaluation_metadata, verify_stored
from mamut_routing_lib.exact_types import STATIC_PROFILE
from mamut_routing_lib.manifests import (
    SPECIFICATION, VerifiedManifest, file_sha256, manifest_bytes, manifest_path_for, matrix_sha256,
)
from mamut_routing_lib.models import ArcCostsDistancesRef, ArcCostsEuclidean
from mamut_routing_lib.objectives import base_objective
from mamut_routing_lib.sidecars import COLLECTION_MARKER_FILENAME, load_collection_marker, validate_relative_sidecar_path


EUCLIDEAN = "mamut_routing_lib.artifacts.resolve_arc_costs"
ROAD_DISTANCES = "mamut_routing_tools.family.materialize_distances"


def _units(family, instance, problem):
    if family == "Sintef2008":
        unit = "Solomon/Homberger benchmark unit"
    elif family == "Dimacs2021":
        unit = "one tenth of a Solomon/Homberger benchmark unit"
    elif family == "Ortec2022":
        unit = "second"
    elif family in ("Mamut2026", "Poryos2026"):
        metric = instance.metric_variant.value
        if metric not in ("euclidean", "shortest", "fastest"):
            raise ValueError(f"Unknown static city metric: {metric}")
        unit = "second" if metric == "fastest" else "metre"
    else:
        raise ValueError(f"No published static units for family {family}")
    units = {"cost": unit}
    if problem == "VRPTW":
        units["time"] = unit
    return units


def _recipe(name, parameters, recipes, root):
    artifacts = recipes.get(name)
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"Missing materializer artifacts for {name}")
    for artifact in artifacts:
        path = root / validate_relative_sidecar_path(artifact["path"])
        if artifact.get("scope") != "file" or file_sha256(path) != artifact["sha256"]:
            raise ValueError(f"Materializer artifact failed verification: {path}")
    return {"callable": name, "parameters": parameters, "artifacts": artifacts}


def _materialization(instance, root, recipes, *, regenerated_distances):
    source = getattr(instance, "arc_costs_source", None)
    sidecars = []
    if regenerated_distances and not isinstance(source, ArcCostsDistancesRef):
        raise ValueError("Road reconstruction requires a distances-sidecar descriptor")
    if source is None:
        materialization = {"kind": "embedded-matrix"}
    elif isinstance(source, ArcCostsEuclidean):
        materialization = {"kind": "regenerated", "recipe": _recipe(
            EUCLIDEAN, {"model": "euclidean", "decimals": source.decimals}, recipes, root,
        )}
    elif isinstance(source, ArcCostsDistancesRef):
        distances = source.distances
        # All released matrix sidecars use canonical JSON. Check the actual
        # decompressed bytes as well as the semantic matrix computed below.
        if file_sha256(root / distances.path, scope="decompressed") != distances.sha256:
            raise ValueError(f"Distance sidecar disagrees with descriptor pin: {distances.path}")
        if regenerated_distances:
            road = Path(distances.path).with_name(instance.instance_name + ".road.json.gz")
            sidecars.append({"path": road.as_posix(), "sha256": file_sha256(root / road), "scope": "file"})
            materialization = {"kind": "regenerated", "recipe": _recipe(
                ROAD_DISTANCES,
                {"metric": instance.metric_variant.value, "decimals": 3,
                 "output_path": distances.path, "output_decompressed_sha256": distances.sha256},
                recipes, root,
            )}
        else:
            sidecars.append({"path": distances.path, "sha256": distances.sha256, "scope": "decompressed"})
            materialization = {"kind": "sidecar-matrix"}
    else:
        raise ValueError(f"Unsupported static materialization: {source!r}")
    return sidecars, materialization


def _source_manifest(path, bks, manifest, encoded, verified):
    """Bind a certified input to its original bytes before issuing a new pin."""
    source = manifest_path_for(path)
    if "standard" not in bks.metadata or not source.is_file():
        return verified
    original = source.read_bytes()
    if original == encoded:
        return verified
    legacy = dict(manifest, cost_is_travel_time=manifest["problem"] == "VRPTW" or manifest["units"]["cost"] == "second")
    if original != manifest_bytes(legacy):
        raise ValueError(f"Source manifest differs from the staged edition: {source}")
    return replace(verified, manifest_sha256=hashlib.sha256(original).hexdigest())


def stage_static(root: Path, descriptor: Path, output: Path, *, recipes: dict,
                 revision: str, regenerated_distances: bool = False) -> dict:
    """Stage the edition manifest/BKS pair after two independent exact route evaluations.

    For an omitted distance matrix, first recover the digest-checked cache in
    the input snapshot using the bundled materialize_distances callable and
    select regenerated_distances. Its committed road graph is the required
    family artifact. Cache files are dispensable for the published edition.
    """
    root, output = root.resolve(), output.resolve()
    if root == output or output.is_relative_to(root):
        raise ValueError("The output staging tree must be outside the source family")
    relative = Path(validate_relative_sidecar_path(descriptor.as_posix()))
    path = root / relative
    if not path.name.endswith(".vrp.json") or not revision:
        raise ValueError("Expected a .vrp.json descriptor and a nonempty edition")
    instance = load_benchmark_instance(path)
    family = instance.benchmark_name.value
    if load_collection_marker(root / COLLECTION_MARKER_FILENAME).family != family:
        raise ValueError(f"Descriptor family disagrees with collection marker: {path}")
    problem = instance_problem_type(instance).value
    if problem not in ("CVRP", "VRPTW"):
        raise ValueError(f"Expected a static descriptor, found {problem}")
    objective = base_objective(instance)
    bks_path = get_bks_path_for_instance(path, objective)
    bks = load_bks(bks_path)
    if bks.instance_name != instance.instance_name or bks.objective_function != objective:
        raise ValueError(f"Mismatched incumbent identity: {bks_path}")
    if "optimality" in bks.metadata:
        raise ValueError(f"Static optimality claim needs a separate target review: {bks_path}")
    raw_sha = file_sha256(bks_path)
    sidecars, materialization = _materialization(instance, root, recipes, regenerated_distances=regenerated_distances)
    hydrated = hydrate_collection_instance(instance, path)
    materialization.update(interpretation="binary64", matrix_sha256=matrix_sha256(hydrated.arc_costs))
    native = evaluate_exact(hydrated, bks.routes, backend="native", objective_function=objective)
    reference = evaluate_exact(hydrated, bks.routes, backend="python", objective_function=objective)
    native_result, reference_result = native.as_dict(), reference.as_dict()
    del native_result["backend"], reference_result["backend"]
    if native_result != reference_result:
        raise ValueError(f"Native and Fraction evaluation disagree: {path}")
    features = ["single-depot", "homogeneous-capacity", "mandatory-visits"]
    if problem == "VRPTW":
        features.append("service-start-windows")
    manifest = {
        "specification": SPECIFICATION, "family": family,
        "benchmark_revision": revision, "instance_name": instance.instance_name,
        "problem": problem, "objective": objective.value, "default_profile": STATIC_PROFILE,
        "units": _units(family, instance, problem), "features": features,
        "instance": {"path": relative.as_posix(), "sha256": file_sha256(path)},
        "sidecars": sidecars, "materialization": materialization,
    }
    encoded = manifest_bytes(manifest)
    manifest_hash = hashlib.sha256(encoded).hexdigest()
    verified = VerifiedManifest(SPECIFICATION, revision, manifest_hash, objective, hydrated)
    verify_stored(bks, native, _source_manifest(path, bks, manifest, encoded, verified))
    metadata = dict(bks.metadata)
    metadata.pop("exact_evaluation", None)
    metadata.update(evaluation_metadata(native, verified))
    if "validated_cost" in metadata:
        metadata["validated_cost"] = native.display_cost
    for key in ("validated_num_routes", "validated_num_vehicles"):
        if key in metadata:
            metadata[key] = len(native.routes)
    current = bks.model_copy(update={"routes": [list(r) for r in native.routes],
                                     "cost": native.display_cost, "metadata": metadata})
    verify_stored(current, reference, verified)
    manifest_path = output / manifest_path_for(relative)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(encoded)
    output_bks = output / bks_path.relative_to(root)
    write_bks_file(current, output_bks)
    return {
        "family": family, "instance": relative.as_posix(), "bks": bks_path.relative_to(root).as_posix(),
        "instance_sha256": manifest["instance"]["sha256"], "incumbent_sha256": raw_sha,
        "manifest_sha256": manifest_hash, "certified_bks_sha256": file_sha256(output_bks),
        "matrix_sha256": materialization["matrix_sha256"], "materialization": materialization["kind"],
        "num_routes": len(native.routes), "authoritative_cost": native_result["authoritative_cost"],
        "display_changed": bks.cost != native.display_cost, "routes_sha256": metadata["standard"]["routes_sha256"],
        "native_backend": native.backend, "reference_backend": reference.backend,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family_root", type=Path)
    parser.add_argument("descriptor", type=Path, help="Descriptor path relative to family root")
    parser.add_argument("--output", type=Path, required=True, help="Separate, writable staging family root")
    parser.add_argument("--recipes", type=Path, help="JSON mapping from materializer callable to artifact references")
    parser.add_argument("--edition", default="r1")
    parser.add_argument("--regenerated-distances", action="store_true",
                        help="Publish the road recipe for an omitted, recovered distance matrix")
    args = parser.parse_args()
    recipes = json.loads(args.recipes.read_bytes()) if args.recipes is not None else {}
    result = stage_static(args.family_root, args.descriptor, args.output, recipes=recipes,
                          revision=args.edition, regenerated_distances=args.regenerated_distances)
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
