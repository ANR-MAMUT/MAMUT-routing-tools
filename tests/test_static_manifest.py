import hashlib
import json
from pathlib import Path

import pytest

from mamut_routing_lib.manifests import manifest_bytes
from mamut_routing_tools.static_manifest import stage_static


def test_stages_repriced_incumbent_and_complete_embedded_matrix(tmp_path):
    root, stage = tmp_path / "input", tmp_path / "stage"
    root.mkdir()
    (root / "mamut-collection.json").write_text(json.dumps({
        "format": "mamut-collection", "format_version": 1, "family": "Sintef2008", "layout_version": 2,
    }))
    desc = Path("VRPTW/example.vrp.json")
    (root / desc).parent.mkdir()
    (root / desc).write_text(json.dumps({
        "instance_name": "example", "instance_origin": "Solomon1987", "benchmark_name": "Sintef2008",
        "num_customers": 2, "num_vehicles": 2, "vehicle_capacity": 1,
        "coordinates": [[0, 0], [1, 0], [2, 0]], "demands": [0, 1, 1],
        "arc_costs": [[0, 0.1, 0.2], [0.1, 0, 0.3], [0.2, 0.3, 0]],
        "service_times": [0, 0, 0], "time_windows": [[0, 5], [0, 5], [0, 5]],
    }))
    incumbent = desc.with_name("example.bks.HierarchicalVehicleCost.json")
    (root / incumbent).write_text(json.dumps({
        "instance_name": "example", "routes": [[2], [1]], "cost": 0.6,
        "objective_function": "HierarchicalVehicleCost",
        "metadata": {"authors": "fixture", "validated_cost": 0.6, "validated_num_vehicles": 2},
    }))
    original = (root / incumbent).read_bytes()
    receipt = stage_static(root, desc, stage, recipes={}, revision="r1")
    manifest = stage / desc.with_name("example.manifest.json")
    published = json.loads(manifest.read_bytes())
    assert "cost_is_travel_time" not in published
    assert published["units"] == {
        "cost": "Solomon/Homberger benchmark unit",
        "time": "Solomon/Homberger benchmark unit",
    }
    record = json.loads((stage / incumbent).read_bytes())
    standard = record["metadata"]["standard"]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == standard["manifest_sha256"]
    assert standard["authoritative_cost"] == {"numerator": "10808639105689191", "denominator": "18014398509481984"}
    assert standard["objective"] == "HierarchicalVehicleCost"
    assert record["routes"] == [[1], [2]]
    assert record["cost"] == record["metadata"]["validated_cost"] == 0.6000000000000001
    assert receipt["display_changed"]
    assert (root / incumbent).read_bytes() == original
    with pytest.raises(ValueError, match="outside"):
        stage_static(root, desc, root, recipes={}, revision="r1")

    # Reissue the old, byte-bound pair and replay the resulting pair.
    prior = stage / "previous"
    prior.mkdir()
    prior_manifest = (root / desc).with_name(manifest.name)
    old = manifest_bytes(dict(published, cost_is_travel_time=True))
    prior_manifest.write_bytes(old)
    standard["manifest_sha256"] = hashlib.sha256(old).hexdigest()
    (root / incumbent).write_text(json.dumps(record))
    reissued = stage_static(root, desc, prior, recipes={}, revision="r1")
    assert reissued["manifest_sha256"] == receipt["manifest_sha256"]
    assert (prior / incumbent).read_bytes() == (stage / incumbent).read_bytes()

    # A mismatched claim and a different, even correctly pinned, unit both fail.
    standard["manifest_sha256"] = reissued["manifest_sha256"]
    (root / incumbent).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="manifest_sha256"):
        stage_static(root, desc, prior, recipes={}, revision="r1")
    changed_unit = manifest_bytes(dict(published, cost_is_travel_time=True, units={"cost": "metre"}))
    prior_manifest.write_bytes(changed_unit)
    standard["manifest_sha256"] = hashlib.sha256(changed_unit).hexdigest()
    (root / incumbent).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Source manifest"):
        stage_static(root, desc, prior, recipes={}, revision="r1")
    prior_manifest.write_bytes(manifest.read_bytes())
    (root / incumbent).write_bytes((stage / incumbent).read_bytes())
    assert stage_static(root, desc, prior, recipes={}, revision="r1")["manifest_sha256"] == receipt["manifest_sha256"]
    (root / incumbent).write_bytes(original)

    # The incumbent traverses neither 1 -> 2 nor 2 -> 1. The entire matrix
    # still identifies the edition, even when the exact solution cost is equal.
    changed = json.loads((root / desc).read_bytes())
    changed["arc_costs"][1][2] = 1
    (root / desc).write_text(json.dumps(changed))
    updated = stage_static(root, desc, stage, recipes={}, revision="r2")
    assert updated["authoritative_cost"] == receipt["authoritative_cost"]
    assert updated["matrix_sha256"] != receipt["matrix_sha256"]
