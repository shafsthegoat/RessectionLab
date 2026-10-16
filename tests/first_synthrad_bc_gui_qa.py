"""Opt-in independent native-GUI save audit for exactly two SynthRAD cases.

Requires parent-supplied saved-file pins and recorded view expectations. No model
module, inference, GUI control, source rewrite or previous audit rewrite occurs.
Saved copies must live outside the source preparation/output directory.

Pins format: {"subjects": {"1BB033": {"reviewed_sha256": "...",
  "expected": {"selected_modality": "CT", "series": {"CT": {
    "layer_opacity": {"skull": 0.65}, "visible_layers": {"skull": true}}}}},
  "1BC077": {...}}}. CT and T1 are the permitted series labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from io import BytesIO
import json
from pathlib import Path
import resource
import sys
from zipfile import ZipFile

import psutil

from resectionlab.core import thaw_json
from resectionlab.workspace_bundle import load_workspace
from resectionlab.workspace_imaging import source_registration
from first_release_gui_save import same_case_arrays, archive_originals

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"outputs/poc-synthrad-bc-source-v1"
SAVED = ROOT/"outputs/poc-first-qa/synthrad-bc-ui/saved"
SOURCE_PINS = {
    "1BB033": "6da3934b794ad13945ea8d1443a70c46625a75c9b8a7cf2a800d2830c906dd82",
    "1BC077": "171101168d1244d509fc64249cd9039a74aa9a35f35535dddc753307e80f905d",
}
FROZEN = {
    "artifacts/poc-synthrad-bc-integration-v1/independent-qa.json": "90eadfb9fd95d4c64778a7356791c49da8735bee8b21562ebe70f3f84d617346",
    "artifacts/poc-synthrad-bc-preparation-v1/receipt.json": "50ba6725e4c118f8f13f7acfcc27359be55028f13e948cccb5bb30b9333ebc35",
    "artifacts/patient-corpus-ten-person-v1/independent-qa.json": "2d28dcde7d6e7ff3a432ae6fe81e3cab7fa082e6ab5f05b5ecfbcd90c685951d",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stamp(path):
    s = Path(path).stat()
    return {"sha256": sha(path), "bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def source_members_exact(source, reviewed):
    """Independent nested archive check, separate from canonical case loading."""
    with ZipFile(source) as left, ZipFile(reviewed) as right:
        a, b = (json.loads(z.read("workspace.json")) for z in (left, right))
        assert len(a["images"]) == len(b["images"]) == 1
        old, new = a["images"][0], b["images"][0]
        for field in ("annotationKind", "modality", "scope", "sourceCaseHash", "sourceFrameHash"):
            assert old["source"][field] == new["source"][field]
        with ZipFile(BytesIO(left.read(old["member"]))) as x, ZipFile(BytesIO(right.read(new["member"]))) as y:
            assert set(x.namelist()) == set(y.namelist())
            for name in x.namelist():
                assert x.read(name) == y.read(name), "Native MR nested content changed"
        assert a["episodeReplay"] == b["episodeReplay"]
        return {"all_native_MR_nested_member_bytes_exact": True, "episode_replay_unchanged": True}


def audit(sid, pin):
    source = BASE/sid/("SynthRAD2023-"+sid+".ressectionlab")
    reviewed = SAVED/("SynthRAD2023-"+sid+"-reviewed.ressectionlab")
    assert source.is_file() and reviewed.is_file()
    before = {str(p.relative_to(ROOT)): stamp(p) for p in (source, reviewed)}
    assert before[str(source.relative_to(ROOT))]["sha256"] == SOURCE_PINS[sid]
    assert before[str(reviewed.relative_to(ROOT))]["sha256"] == pin["reviewed_sha256"]
    original, original_artifacts, old = load_workspace(source)
    actual, actual_artifacts, saved = load_workspace(reviewed)
    arrays = same_case_arrays(original, actual)
    assert set(actual.compartments) == set(actual.source_compartments) == {"brain", "skull"}
    assert set(actual.metadata["anatomy_layers"]) == {"brain", "skull"}
    assert all(layer["provenance"] == "estimated" for layer in actual.metadata["anatomy_layers"].values())
    assert len(old["sources"]) == len(saved["sources"]) == 1
    prior_display, current_display = old["sources"][0], saved["sources"][0]
    mr = same_case_arrays(prior_display.case, current_display.case)
    assert current_display.modality == "T1" and current_display.annotation_kind == "none"
    assert not current_display.case.compartments and not current_display.case.source_compartments
    for case in (actual, current_display.case):
        assert case.metadata["split"]["role"] == "TRAIN" and case.metadata["split"]["training_admitted"] is False
        assert case.metadata["training_admitted"] is False and case.metadata["is_synthetic"] is False
        assert set(case.metadata["missing_anatomy"]) == {"tumor", "vessels", "ventricles"}
    old_mapping = source_registration(original, prior_display.case)
    mapping = source_registration(actual, current_display.case)
    assert old_mapping == mapping and mapping[1]["overlayPermitted"] is False
    originals = archive_originals(source, reviewed)
    assert originals["count"] == 5
    archive_checks = source_members_exact(source, reviewed)
    old_artifacts, artifacts = thaw_json(original_artifacts), thaw_json(actual_artifacts)
    assert {k: v for k, v in old_artifacts.items() if k != "workspace"} == {k: v for k, v in artifacts.items() if k != "workspace"}
    assert "patient_rehearsal" not in artifacts

    # Match parent-recorded controls rather than infer intent from the save.
    imaging = saved["imagingState"]
    series_id = current_display.identity(actual)
    ids = {"CT": "primary", "T1": series_id}
    selected = imaging["selectedSeriesId"]
    selected_modality = "CT" if selected in (None, "primary") else "T1" if selected == series_id else None
    expected = pin["expected"]
    assert selected_modality == expected["selected_modality"]
    assert set(expected.get("series", {})) <= set(ids)
    checked = {}
    for modality, want in expected.get("series", {}).items():
        state = imaging["states"][ids[modality]]
        for name, value in want.get("layer_opacity", {}).items():
            assert state["viewer"]["layerOpacity"][name] == value
        for name, value in want.get("visible_layers", {}).items():
            assert state["visibleLayers"][name] is value
        if "volume_rendering_enabled" in want:
            assert state["viewer"]["volumeRendering"]["enabled"] is want["volume_rendering_enabled"]
        if want.get("no_layers"):
            assert state["visibleLayers"] == {}
            assert not state.get("viewer", {}).get("layerOpacity", {})
        if "cursor" in want:
            assert state["cursor"] == want["cursor"]
        checked[modality] = want
    assert before == {str(p.relative_to(ROOT)): stamp(p) for p in (source, reviewed)}
    return {"status": "pass", "subject_id": sid, "inputs": before, "case_hash": actual.semantic_hash,
        "source_CT_arrays_and_metadata": arrays, "native_MR_arrays_and_metadata": mr,
        "source_references_model_provenance_and_role_metadata_exact": True,
        "originals": originals, "archive_checks": archive_checks, "registration_metadata_exact": True,
        "fusion_permitted": False, "selected_modality": selected_modality, "recorded_view_expectations_verified": checked,
        "saved_imaging_state": imaging, "saved_workspace_preferences": artifacts.get("workspace", {}),
        "source_and_reviewed_files_unchanged": True, "training_admitted": False,
        "clinical_validation": False, "GUI_observation": "Parent performed native GUI actions; this independent audit verifies the saved artifacts only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--released-pins", type=Path, required=True)
    args = parser.parse_args()
    pins = json.loads(args.released_pins.read_text())
    assert set(pins["subjects"]) == set(SOURCE_PINS)
    destination = ROOT/"artifacts/poc-synthrad-bc-integration-v1/gui-save-independent-qa.json"
    assert not destination.exists(), "Frozen GUI audit already exists; choose a new audit version"
    source_output_bytes = sum(p.stat().st_size for p in BASE.rglob("*") if p.is_file())
    roots = [BASE, *[ROOT/"outputs/poc-models"/("SynthRAD2023-"+sid) for sid in SOURCE_PINS]]
    measured_bytes = sum(p.stat().st_size for folder in roots for p in folder.rglob("*") if p.is_file())
    assert source_output_bytes == 148745103 and measured_bytes == 149452038
    before = {name: stamp(ROOT/name) for name in FROZEN}
    for name, expected in FROZEN.items():
        assert before[name]["sha256"] == expected
    cases = []
    for sid in SOURCE_PINS:
        assert psutil.virtual_memory().available >= 1024**3
        cases.append(audit(sid, pins["subjects"][sid]))
        gc.collect()
        assert psutil.Process().memory_info().rss < 1024**3
        print(sid+" GUI save preservation pass", flush=True)
    assert before == {name: stamp(ROOT/name) for name in FROZEN}
    assert not any(name in sys.modules for name in ("torch", "totalsegmentator", "resectionlab.poc_models"))
    report = {"schema": "independent-synthrad-bc-gui-save-audit-v1", "status": "pass", "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": cases, "frozen_audits_unchanged": before, "auditor_sha256": sha(Path(__file__)),
        "pins_sha256": sha(args.released_pins), "model_imports_or_inference": False,
        "source_output_directory_bytes_unchanged": source_output_bytes,
        "source_and_model_output_bytes_unchanged": measured_bytes,
        "own_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "auditor_adjustments": ["The frozen149452038-byte denominator includes the source directory plus two model-output directories; the source directory alone is148745103 bytes. Initial GUI audit assertion omitted the two model directories; corrected before any bundle decoding."],
        "limitations": ["Preservation and persistence checks do not establish model accuracy or anatomical completeness.",
            "Native GUI interaction and fresh-process visual checks were observed separately by the parent."]}
    content = json.dumps(report, indent=2)+"\n";assert "/Users/" not in content and "file://" not in content
    assert sum(p.stat().st_size for p in BASE.rglob("*") if p.is_file()) == source_output_bytes
    assert sum(p.stat().st_size for folder in roots for p in folder.rglob("*") if p.is_file()) == measured_bytes
    with destination.open("x") as output:
        output.write(content)
    print(json.dumps({"status": "pass", "artifact": str(destination.relative_to(ROOT)), "sha256": sha(destination)}))


if __name__ == "__main__":
    main()
