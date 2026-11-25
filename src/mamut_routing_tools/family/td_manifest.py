"""Stage a release manifest and a dual-checked exact TD incumbent outside the input tree."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from mamut_routing_lib.artifacts import get_bks_path_for_instance, instance_problem_type, load_bks, write_bks_file
from mamut_routing_lib.exact_store import read_result_fraction, standard_metadata
from mamut_routing_lib.exact_td import prepare_td
from mamut_routing_lib.exact_types import TD_PROFILE
from mamut_routing_lib.json_utils import strict_json_loads
from mamut_routing_lib.manifests import SPECIFICATION, file_sha256, manifest_bytes, manifest_path_for
from mamut_routing_lib.models import BenchmarkBKS
from mamut_routing_lib.objectives import base_objective
from mamut_routing_lib.sidecars import validate_relative_sidecar_path
from mamut_routing_lib.td.artifacts import load_td_benchmark_instance
from mamut_routing_lib.td.manifest_stream import stream_td_materialization
from mamut_routing_lib.td.reprice import route_arcs

IGP = "mamut_routing_lib.td.igp.materialize_instance_atfs"
ROAD = "mamut_routing_lib.td.roadgraph.materialize_instance_atfs_roadgraph"
BLAUTH = "mamut_routing_tools.conversion.blauth2024.convert_instance"
EXTERNAL = ".mamut/external/vrptdt-benchmark"
DATASET_URL = "https://bonndata.uni-bonn.de/api/access/datafile/5896"
BLAUTH_LOCATOR = {
    "path": ".mamut/artifacts/5422d97e42fbe65ca63877aeb5158eb9745e2e462eb70f68ea843520f0501a66/Blauth2024-bonndata-v1.json",
    "scope": "file",
    "sha256": "5422d97e42fbe65ca63877aeb5158eb9745e2e462eb70f68ea843520f0501a66",
}


def _recipe(root, catalog, name, parameters):
    records = catalog.get(name)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Missing recipe archive for {name}")
    for ref in records:
        path = root / validate_relative_sidecar_path(ref["path"])
        if ref.get("scope") != "file" or file_sha256(path) != ref["sha256"]:
            raise ValueError(f"Recipe archive does not match {ref['path']}")
    return {"callable": name, "parameters": parameters, "artifacts": records}


def _manifest(root, relative, instance, proof, catalog, upstream_root):
    td = instance.td
    family = instance.benchmark_name.value
    objective = base_objective(instance)
    material = {"kind": "stored-atf", "interpretation": "binary64", "atf_stream_sha256": proof["atf_stream_sha256"],
                "canonical_atf_sha256": proof["canonical_atf_sha256"]}
    if material["canonical_atf_sha256"] != td.atf_sha256:
        raise ValueError("Complete canonical ATF digest differs from descriptor")
    sidecars = list(proof["sidecars"])
    recipe = None
    if td.model == "igp-profile":
        recipe = _recipe(root, catalog, IGP, {"time_periods": td.time_periods, "speeds": td.speeds,
                                             "categories_path": td.categories_path})
    elif td.model == "road-graph":
        recipe = _recipe(root, catalog, ROAD, {"sample_step": td.sample_step, "simplify_tolerance": td.simplify_tolerance,
                                              "graph_path": td.graph.path, "traffic_path": td.traffic.path})
    elif td.model == "atf-ndcpwlf" and not sidecars:
        if family != "Blauth2024" or upstream_root is None:
            raise ValueError("Missing stored ATF and no Blauth source directory supplied")
        raw = instance.metadata["generator"]
        locator_path = root / BLAUTH_LOCATOR["path"]
        if file_sha256(locator_path) != BLAUTH_LOCATOR["sha256"]:
            raise ValueError("BonnData source locator differs from its archived digest")
        locator = strict_json_loads(locator_path.read_bytes())
        if locator["archive"]["url"] != DATASET_URL:
            raise ValueError("Unexpected BonnData archive source")
        members = []
        for name, expected in sorted(raw["upstream_files"].items()):
            member = "instances/" + validate_relative_sidecar_path(name)
            if file_sha256(upstream_root / member) != expected:
                raise ValueError(f"Blauth source member differs from upstream pin: {name}")
            archived = locator["members"][name]
            if archived["sha256"] != expected or archived["member"] != f"vrptdt-benchmark-main/{member}":
                raise ValueError(f"BonnData index disagrees with descriptor or member path: {name}")
            members.append({"path": member, "sha256": expected, "scope": "file",
                            "uri": DATASET_URL, "archive_member": archived["member"]})
        recipe = _recipe(root, catalog, BLAUTH, {
            "city": instance.instance_name.removeprefix("Blauth-"), "n": instance.num_customers,
            "upstream_root": EXTERNAL, "upstream_files": members,
            "upstream_commit_from": "metadata.generator.upstream_commit", "output_root": ".",
            "output_path": td.atf_path, "output_decompressed_sha256": td.atf_sha256,
        })
        if BLAUTH_LOCATOR not in recipe["artifacts"]:
            recipe["artifacts"] = [*recipe["artifacts"], BLAUTH_LOCATOR]
    if recipe is not None:
        material.update(kind="regenerated", recipe=recipe)
    units = {"Blauth2024": "millisecond", "Poryos2026": "second", "Rifki2020": "second"}
    unit = units.get(family, f"{family} benchmark time unit")
    problem = instance_problem_type(instance).value
    features = ["single-depot", "homogeneous-capacity", "mandatory-visits", "time-dependent-travel"]
    if problem == "TDVRPTW":
        features.append("service-start-windows")
    if objective.value == "FleetCostDuration":
        features.append("fixed-vehicle-charge")
    return {"specification": SPECIFICATION, "family": family, "benchmark_revision": "r1",
            "instance_name": instance.instance_name, "problem": problem, "objective": objective.value,
            "default_profile": TD_PROFILE, "features": features, "units": {"cost": unit, "time": unit},
            "instance": {"path": relative.as_posix(), "sha256": proof["instance_sha256"]},
            "sidecars": sidecars, "materialization": material}


def _display(rational):
    value = read_result_fraction(rational)
    try:
        return float(value)
    except OverflowError:
        return None


def _write_stage(root, relative, output, proof, recipes, upstream_root):
    """Write a trusted internal dual-backend checkpoint after verifying its input identities."""
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("A release must be staged outside its input collection")
    descriptor = root / validate_relative_sidecar_path(relative.as_posix())
    instance = load_td_benchmark_instance(descriptor)
    bks_path = get_bks_path_for_instance(descriptor, base_objective(instance))
    if file_sha256(descriptor) != proof["instance_sha256"] or file_sha256(bks_path) != proof["incumbent_sha256"]:
        raise ValueError("Input descriptor or incumbent changed during certification")
    bks = load_bks(bks_path)
    if bks.instance_name != instance.instance_name or bks.objective_function != base_objective(instance):
        raise ValueError("Incumbent belongs to a different instance or objective")
    manifest = _manifest(root, relative, instance, proof, recipes, upstream_root)
    encoded = manifest_bytes(manifest)
    digest = sha256(encoded).hexdigest()
    standard = deepcopy(proof["standard"])
    if (standard["objective"] != bks.objective_function.value or standard["profile"] != TD_PROFILE
            or proof["display_cost"] != _display(standard["authoritative_cost"])):
        raise ValueError("TD evaluation has the wrong objective, display or arithmetic profile")
    standard.update(specification=SPECIFICATION, benchmark_revision="r1", manifest_sha256=digest)
    metadata = deepcopy(bks.metadata)
    for key in ("standard", "exact_evaluation"):
        if key in metadata:
            raise ValueError("Expected a pre-r1 incumbent, found an existing exact record")
    metadata.pop("repriced", None)
    if "optimality" in metadata:
        raise ValueError("Incumbent carries an optimality claim. Stage the unclaimed record and record "
                         "a solver claim under the TD profile afterwards.")
    witnesses = standard["route_evaluations"]
    if len(witnesses) != len(bks.routes) or len(bks.routes) != standard["num_routes"]:
        raise ValueError("Incomplete per-route witness vector")
    metadata.update(standard=standard, validated_cost=proof["display_cost"], validated_num_routes=len(bks.routes),
                    validated_by="mamut-routing-lib", route_durations=[_display(w["duration"]) for w in witnesses],
                    route_departure_times=[_display(w["departure_time"]) for w in witnesses])
    current = BenchmarkBKS(**{**bks.model_dump(), "routes": sorted(bks.routes), "cost": proof["display_cost"],
                              "metadata": metadata})
    path = output / manifest_path_for(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    output_bks = output / bks_path.relative_to(root)
    write_bks_file(current, output_bks)
    return {"manifest_sha256": digest, "certified_bks_sha256": file_sha256(output_bks),
            "routes": len(current.routes)}


def stage_td(root: Path, descriptor: Path, output: Path, *, recipes: dict, upstream_root: Path | None = None,
             atf_cache_root: Path | None = None):
    """Verify all arcs, compare both exact backends, and stage one manifest and incumbent."""
    path = root / validate_relative_sidecar_path(descriptor.as_posix())
    instance_sha = file_sha256(path)
    instance = load_td_benchmark_instance(path)
    objective = base_objective(instance)
    bks_path = get_bks_path_for_instance(path, objective)
    bks_sha = file_sha256(bks_path)
    bks = load_bks(bks_path)
    checked = stream_td_materialization(path, root, route_arcs(bks.routes), atf_cache_root=atf_cache_root)
    native = prepare_td(instance, checked.operands, backend="native", objective_function=objective).evaluate(bks.routes)
    reference = prepare_td(instance, checked.operands, backend="python", objective_function=objective).evaluate(bks.routes)
    if (native.routes, native.cost, native.route_evaluations) != (reference.routes, reference.cost, reference.route_evaluations):
        raise ValueError("Exact TD backends disagree")
    proof = dict(instance_sha256=instance_sha, incumbent_sha256=bks_sha,
                 atf_stream_sha256=checked.atf_stream_sha256, canonical_atf_sha256=checked.canonical_sha256,
                 sidecars=checked.sidecars, standard=standard_metadata(native), display_cost=native.display_cost)
    return _write_stage(root, descriptor, output, proof, recipes, upstream_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family_root", type=Path)
    parser.add_argument("descriptor", type=Path, help="Path relative to the family root")
    parser.add_argument("output", type=Path, help="Isolated staging tree")
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, help="Blauth source directory containing instances/")
    parser.add_argument("--atf-cache-root", type=Path, help="Directory with checked regenerated ATFs")
    args = parser.parse_args()
    result = stage_td(args.family_root, args.descriptor, args.output, recipes=json.loads(args.recipes.read_bytes()),
                      upstream_root=args.upstream_root, atf_cache_root=args.atf_cache_root)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
