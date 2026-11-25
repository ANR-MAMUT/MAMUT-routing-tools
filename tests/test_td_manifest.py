import hashlib
import json

import pytest

from mamut_routing_lib.td.artifacts import InstanceATFs, compute_atf_sha256, save_instance_atfs
from mamut_routing_lib.td.pwlf import NDCPWLF
from mamut_routing_lib.manifests import manifest_path_for
from mamut_routing_tools.family.td_manifest import stage_td


def _toy_family(tmp_path, metadata):
    root = tmp_path / "Dabia2013"
    root.mkdir()
    desc = root / "toy.vrp.json"
    arc = NDCPWLF([0.0, 10.0], [1.0, 11.0])
    atfs = InstanceATFs("toy", "Dabia2013", (0.0, 10.0), 1, {(0, 1): arc, (1, 0): arc})
    save_instance_atfs(atfs, root / "toy.atf.json")
    desc.write_text(json.dumps({"instance_name": "toy", "benchmark_name": "Dabia2013", "instance_origin": "Solomon1987",
                               "num_customers": 1, "num_vehicles": 1, "vehicle_capacity": 1, "demands": [0, 1],
                               "coordinates": [[0, 0], [0, 1]], "depot": 0, "service_times": [0, 0], "horizon": [0, 10],
                               "td": {"model": "atf-ndcpwlf", "atf_path": "toy.atf.json", "atf_sha256": compute_atf_sha256(atfs)}}))
    record = {"instance_name": "toy", "objective_function": "Duration", "routes": [[1]], "cost": 2, "metadata": metadata}
    path = root / "toy.bks.Duration.json"
    path.write_text(json.dumps(record))
    return root, desc, path, record


def test_td_staging_writes_manifest_and_tied_departure(tmp_path):
    pytest.importorskip("mamut_routing_lib._td_exact")
    root, desc, path, record = _toy_family(tmp_path, {"authors": "Test"})
    out = tmp_path / "release"
    result = stage_td(root, desc.relative_to(root), out, recipes={})
    saved = json.loads((out / path.name).read_bytes())
    manifest = (out / manifest_path_for(desc.name)).read_bytes()
    assert saved["cost"] == 2 and result["routes"] == 1
    std = saved["metadata"]["standard"]
    assert std["manifest_sha256"] == hashlib.sha256(manifest).hexdigest() == result["manifest_sha256"]
    assert std["route_evaluations"] == [{"duration": {"numerator": "2", "denominator": "1"},
                                          "departure_time": {"numerator": "0", "denominator": "1"}}]
    assert "optimality" not in saved["metadata"]
    assert json.loads(path.read_bytes()) == record


def test_td_staging_refuses_an_incumbent_with_a_claim(tmp_path):
    pytest.importorskip("mamut_routing_lib._td_exact")
    claim = {"status": "solver-reported-optimal", "target_profile": "td-binary64-exact-v1", "solver": "test solver"}
    root, desc, path, record = _toy_family(tmp_path, {"authors": "Test", "optimality": claim})
    out = tmp_path / "release"
    with pytest.raises(ValueError, match="optimality claim"):
        stage_td(root, desc.relative_to(root), out, recipes={})
    assert not (out / path.name).exists()
    assert json.loads(path.read_bytes()) == record
