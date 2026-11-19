import json
import numpy as np
from scipy.ndimage import label

from scripts.generate_curved_render_case import build_case, generate, VESSEL, TARGET, MEMBRANE
from resectionlab.imaging import load_case, read_case_artifacts
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_engine import SurgicalEngine


def test_generated_curved_source_topology_and_replay(tmp_path):
    case, truth = build_case()
    masks = case.compartments
    assert all(ref.provenance == "simulated" for ref in case.source_refs)
    assert np.max(np.sum(list(masks.values()), axis=0)) == 1
    assert truth["membraneMaximumCellsPerColumn"] == 1
    assert truth["layers"][MEMBRANE]["faceConnectedComponents"] == 1
    assert truth["layers"][TARGET]["faceConnectedComponents"] == 2
    cc, count = label(masks[VESSEL])
    assert count == 1
    for point in [truth["vessel"]["junctionRasMm"], *(b["tipRasMm"] for b in truth["vessel"]["branches"])]:
        voxel = np.rint((np.linalg.inv(case.affine) @ [*point, 1.0])[:3]).astype(int)
        assert cc[tuple(voxel)] == 1
    path = generate(tmp_path)
    assert generate(tmp_path) == path
    saved = load_case(path)
    assert saved.semantic_hash == case.semantic_hash
    record = read_case_artifacts(path)["surgical_toolbox"]
    branch = record["branchId"]
    engine = SurgicalEngine.restore(from_case(saved, record["mapping"]), record["branches"][branch])
    report = json.loads((tmp_path / "geometry-and-actions.json").read_text())
    assert engine.state_id == report["endpointStateId"]
    assert report["rejection"]["reasonCode"] == "WITHDRAW_BEFORE_EXCHANGE"
    assert report["rejection"]["stateId"] == record["timeline"][branch][1]["stateAfter"]
    assert (
        0
        < report["endpointLedger"]["originalTarget"]["removedMm3"]
        < report["endpointLedger"]["originalTarget"]["sourceVolumeMm3"]
    )
