"""Tiny mutation checks for the portable audit's independent provenance guard."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("portable_auditor", SCRIPTS / "audit_pat16_pat20_portable_proposals.py")
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def fixture():
    directory = "artifacts/frozen/r1"
    report = {"variants": {v: {"inference": {"artifact_hashes": {
        f"{v}_mask.nii.gz": "a" * 64, f"{v}_distance_mm.nii.gz": "b" * 64}}} for v in ("main", "nocsf")}}
    data = json.dumps(report).encode()
    portable = {
        "extraction_report_utf8": data.decode(), "extraction_report_sha256": sha256(data).hexdigest(),
        "selected_repetition": "r1", "mask_arrays_embedded": True, "predicted_distance_arrays_embedded": False,
        "brain_reviewed": False, "cortical_access_permitted": False, "clinical_deficit_probability": None,
        "historical_binary_attestation": {key: False for key in (
            "historical_installed_package_binary_hashes_predeclared", "historical_interpreter_binary_hash_predeclared",
            "unrecorded_runs_excluded_by_attestation")},
        "independent_extraction_audits": [{"path": p, "sha256": h} for p, h in auditor.EXTRACTION_AUDITS.items()],
        "external_extraction_artifacts": [{"variant": variant, "artifact_kind": kind,
            "path": f'{directory}/{variant}_{"mask" if mask else "distance_mm"}.nii.gz',
            "sha256": ("a" if mask else "b") * 64, "native_array_embedded_as_proposal": mask,
            "embedded_as_original_file": False}
            for variant in ("main", "nocsf") for kind, mask in (("source_mask", True), ("predicted_signed_distance", False))],
    }
    return portable, data, directory


def test_complete_portable_inventory_is_read_only():
    portable, data, directory = fixture()
    original = deepcopy(portable)
    assert auditor.check_portable_artifacts(portable, data, directory) == json.loads(data)
    assert portable == original


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(extraction_report_utf8=p["extraction_report_utf8"] + " "),
    lambda p: p.update(extraction_report_sha256="0" * 64),
    lambda p: p.update(selected_repetition="r2"),
    lambda p: p.update(predicted_distance_arrays_embedded=True),
    lambda p: p.update(brain_reviewed=True),
    lambda p: p.update(cortical_access_permitted=True),
    lambda p: p.update(clinical_deficit_probability=0.01),
    lambda p: p["historical_binary_attestation"].update(historical_interpreter_binary_hash_predeclared=True),
    lambda p: p["external_extraction_artifacts"].pop(),
    lambda p: p["external_extraction_artifacts"].__setitem__(1, deepcopy(p["external_extraction_artifacts"][0])),
    lambda p: p["external_extraction_artifacts"][0].update(path="artifacts/other/mask.nii.gz"),
    lambda p: p["external_extraction_artifacts"][0].update(sha256="c" * 64),
    lambda p: p["external_extraction_artifacts"][1].update(native_array_embedded_as_proposal=True),
    lambda p: p["independent_extraction_audits"][0].update(sha256="0" * 64),
])
def test_inaccurate_portable_claim_rejected(mutation):
    portable, data, directory = fixture()
    mutation(portable)
    with pytest.raises(ValueError):
        auditor.check_portable_artifacts(portable, data, directory)
