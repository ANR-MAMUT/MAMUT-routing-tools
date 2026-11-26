import hashlib
import json
from pathlib import Path

import pytest

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

    # The incumbent traverses neither 1 -> 2 nor 2 -> 1. The entire matrix
    # still identifies the edition, even when the exact solution cost is equal.
    changed = json.loads((root / desc).read_bytes())
    changed["arc_costs"][1][2] = 1
    (root / desc).write_text(json.dumps(changed))
    updated = stage_static(root, desc, stage, recipes={}, revision="r2")
    assert updated["authoritative_cost"] == receipt["authoritative_cost"]
    assert updated["matrix_sha256"] != receipt["matrix_sha256"]
