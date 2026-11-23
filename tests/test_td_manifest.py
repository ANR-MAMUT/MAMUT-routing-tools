import hashlib
import json

import pytest

from mamut_routing_lib.td.artifacts import InstanceATFs, compute_atf_sha256, save_instance_atfs
from mamut_routing_lib.td.pwlf import NDCPWLF
from mamut_routing_lib.manifests import manifest_path_for
from mamut_routing_tools.family.td_manifest import stage_td


def test_td_staging_preserves_original_solver_target_and_tied_departure(tmp_path):
    pytest.importorskip("mamut_routing_lib._td_exact")
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
    historical = {"instance_name": "toy", "objective_function": "Duration", "routes": [[1]], "cost": 2,
                  "metadata": {"authors": "Test", "optimality": {"proven": True, "prover": "test solver",
                               "certificate": "historical certificate", "date": "2026-08-06", "campaign": "test",
                               "proven_optimum": 1.9999999999999998, "dual_bound": 1.9999999999999998}}}
    original = json.dumps(historical).encode()
    historical["metadata"]["optimality"]["proven_optimum"] = 2.0
    path = root / "toy.bks.Duration.json"
    path.write_text(json.dumps(historical))
    out = tmp_path / "release"
    stage_td(root, desc.relative_to(root), out, recipes={}, original_claim=original, original_claim_target="td-fold/1")
    saved = json.loads((out / path.name).read_bytes())
    manifest = (out / manifest_path_for(desc.name)).read_bytes()
    assert saved["cost"] == 2
    std = saved["metadata"]["standard"]
    assert std["manifest_sha256"] == hashlib.sha256(manifest).hexdigest()
    assert std["route_evaluations"] == [{"duration": {"numerator": "2", "denominator": "1"},
                                          "departure_time": {"numerator": "0", "denominator": "1"}}]
    claim = saved["metadata"]["optimality"]
    assert claim["target_profile"] == "td-fold/1"
    assert claim["reported_primal"] == claim["reported_dual"] == 1.9999999999999998
    assert (out / claim["evidence"]["path"]).read_bytes() == original
    assert json.loads(path.read_bytes()) == historical
