"""View-only registered evidence must not become current patient function."""
from dataclasses import replace
import json
from pathlib import Path
import threading

import nibabel as nib
import numpy as np
import pytest

from resectionlab.core import SourceRef, array_digest, semantic_digest
from resectionlab.desktop_bridge import BridgeRuntime, _Request
from resectionlab.imaging import create_synthetic_case, save_case
from resectionlab.prior_proposals import RegisteredPriorProposal
from resectionlab.structural_evidence import structural_frame_hash
from resectionlab.worlds import WorldGenerator, WorldGeneratorConfig, generate_partitions


def fixture_case():
    case = create_synthetic_case((24, 24, 24))
    refs = tuple(SourceRef(source_id=name, uri=f"synthetic://{name}", sha256=digit * 64,
                           native_frame="RAS+", provenance="simulated")
                 for name, digit in (("structural", "1"), ("registration_t1", "2"), ("lesion", "3")))
    return replace(case, source_refs=refs)


def proposal(case, identity="motor-preview"):
    coverage = np.zeros(case.mri.shape, bool)
    coverage[:12] = True
    values = np.zeros(case.mri.shape, np.float32)
    values[1:8] = .25
    ras = np.diag([-1., -1., 1., 1.]) @ case.affine if case.frame == "LPS+" else case.affine
    return RegisteredPriorProposal(
        proposal_id=identity, map_id="motor_functional_concordance", component="motor",
        map_kind="functional_concordance", data=values, sampling_coverage=coverage,
        affine_ras_mm=ras, source=SourceRef(source_id="fixture_population_atlas",
            uri="synthetic://atlas", sha256="4" * 64, provenance="prior", native_frame="FSL_MNI152_RAS_mm"),
        source_prior_hash="5" * 64, source_prior_shape=case.mri.shape, source_prior_affine_ras_mm=ras,
        source_image_hash=array_digest(case.mri), source_frame_hash=structural_frame_hash(case),
        source_file_sha256="1" * 64, registration_image_sha256="2" * 64,
        registration_lesion_sha256="3" * 64,
        registration_compartments_hash=semantic_digest({name: array_digest(mask) for name, mask in case.compartments.items()}),
        source_case_planning_hash=case.planning_hash, registration_case_hash=case.semantic_hash,
        registration_hash="6" * 64, mni_ras_to_patient_ras_mm=np.eye(4),
        registration_method="synthetic_known_transform", registration_version="fixture-v1",
        registration_report_sha256="7" * 64, registration_inventory_sha256="8" * 64,
        preview_sha256="9" * 64, prior_manifest_sha256="a" * 64,
        template_identity="fixture", template_sha256="b" * 64)


def call(runtime, events, identity, op, args):
    runtime.submit({"id": identity, "op": op, "args": args})
    assert runtime.wait_idle(20)
    return [item for item in events if item.get("id") == identity][-1]


def test_empty_identity_and_registered_proposals_leave_sampled_worlds_unchanged():
    case = fixture_case()
    assert "prior_proposals" not in case.to_manifest()
    assert replace(case, prior_proposals={}).semantic_hash == case.semantic_hash
    item = proposal(case)
    revised = case.revised(prior_proposals={item.proposal_id: item})
    assert revised.semantic_hash != case.semantic_hash == item.registration_case_hash
    assert revised.planning_hash == case.planning_hash
    config = WorldGeneratorConfig(translation_scale_mm=(1., 2., 3.), rotation_scale_deg=(1., 1., 1.))
    before = generate_partitions(case.semantic_hash, config, 17, planning_hash=case.planning_hash)
    after = generate_partitions(revised.semantic_hash, config, 17, planning_hash=revised.planning_hash)
    for role in ("optimization", "selection", "final_evaluation", "stress"):
        original, updated = getattr(before, role), getattr(after, role)
        assert original.seeds == updated.seeds
        sampler = WorldGenerator(original.generator)
        for index in range(len(original.seeds)):
            np.testing.assert_array_equal(sampler.sample(original, index).anatomy_transform_mm,
                                          sampler.sample(updated, index).anatomy_transform_mm)


@pytest.mark.parametrize("change", ["image", "affine", "labels", "source"])
def test_attached_proposal_rejects_changed_source_anatomy(change):
    case = fixture_case()
    item = proposal(case)
    case = case.revised(prior_proposals={item.proposal_id: item})
    if change == "image":
        image = case.mri.copy(); image[0, 0, 0] += 1
        edits = {"mri": image}
    elif change == "affine":
        affine = case.affine.copy(); affine[0, 3] += 1
        edits = {"affine": affine}
    elif change == "labels":
        labels = dict(case.compartments)
        names = list(labels); labels[names[0]], labels[names[1]] = labels[names[1]], labels[names[0]]
        edits = {"compartments": labels}
    else:
        edits = {"source_refs": (replace(case.source_refs[0], sha256="c" * 64), *case.source_refs[1:])}
    with pytest.raises(ValueError, match="different case inputs|source file|lesion"):
        case.revised(**edits)


@pytest.mark.filterwarnings("ignore:Setting the strides:DeprecationWarning")
def test_coverage_layout_mutation_invalidates_case_and_cannot_change_unknown_to_zero():
    case = fixture_case(); item = proposal(case)
    enriched = case.revised(prior_proposals={item.proposal_id: item})
    item.sampling_coverage.strides = tuple(reversed(item.sampling_coverage.strides))
    with pytest.raises(ValueError, match="array metadata"):
        _ = enriched.semantic_hash
    with pytest.raises(ValueError, match="array metadata"):
        _ = enriched.planning_hash


def test_prior_payload_is_separate_read_only_ras_evidence_and_roundtrips(tmp_path):
    case = replace(fixture_case(), frame="LPS+"); item = proposal(case)
    other = replace(item, proposal_id="a-other", map_id="semantics_functional_concordance", component="semantics")
    enriched = case.revised(prior_proposals={item.proposal_id: item, other.proposal_id: other})
    events = []; runtime = BridgeRuntime(tmp_path / "transfer", events.append)
    try:
        payload = runtime.session._install_case(enriched, {}, _Request("install"))
        layer_order = [layer["proposalId"] for layer in payload["priorProposals"]]
        assert layer_order == sorted(layer_order)
        layer = next(layer for layer in payload["priorProposals"] if layer["proposalId"] == item.proposal_id)
        assert layer["frame"] == "RAS+" and payload["frame"] == "LPS+"
        np.testing.assert_array_equal(layer["affine"], np.diag([-1., -1., 1., 1.]) @ case.affine)
        assert layer["viewOnly"] and not layer["planningEligible"] and not layer["patientSpecificFunction"]
        assert layer["reviewStatus"] == "alignment_review_required" and layer["clinicalDeficitProbability"] is None
        values = np.fromfile(layer["data"]["path"], dtype="<f4").reshape(case.mri.shape)
        coverage = np.fromfile(layer["samplingCoverage"]["path"], dtype="uint8").reshape(case.mri.shape)
        np.testing.assert_array_equal(values, item.data)
        np.testing.assert_array_equal(coverage, item.sampling_coverage)
        assert values[0, 0, 0] == values[-1, 0, 0] == 0 and coverage[0, 0, 0] != coverage[-1, 0, 0]
        assert payload["planningHash"] == case.planning_hash and payload["brainMask"] is not None
        destination = tmp_path / "evidence.ressectionlab"
        saved = call(runtime, events, "save", "saveCase", {"caseHash": enriched.semantic_hash, "path": str(destination)})
        assert saved["event"] == "result", saved
    finally:
        runtime.close()
    events = []; runtime = BridgeRuntime(tmp_path / "fresh", events.append)
    try:
        restored = call(runtime, events, "load", "loadCase", {"path": str(destination)})
        assert restored["event"] == "result", restored
        assert [layer["proposalId"] for layer in restored["result"]["priorProposals"]] == layer_order
        layer = next(layer for layer in restored["result"]["priorProposals"] if layer["proposalId"] == item.proposal_id)
        assert layer["evidenceHash"] == item.evidence_hash
        assert layer["registrationCaseHash"] == case.semantic_hash != restored["result"]["caseHash"]
    finally:
        runtime.close()


def import_paths(tmp_path, count):
    directory = tmp_path / "registered"; directory.mkdir()
    cache = tmp_path / "cache"; cache.mkdir()
    (directory / "functional_overlay_inventory.json").write_text(json.dumps([{"map_id": str(n)} for n in range(count)]))
    (directory / "registration_comparison.json").write_text("{}")
    manifest = tmp_path / "manifest.json"; manifest.write_text("{}")
    image = tmp_path / "image.nii"
    source = nib.Nifti1Image(np.zeros((24, 24, 24), np.float32), np.eye(4))
    source.header.set_xyzt_units("mm")
    nib.save(source, image)
    return {"registrationDirectory": str(directory), "sourceCacheDirectory": str(cache),
            "priorManifestPath": str(manifest), "sourceImagePath": str(image),
            "registrationImagePath": str(image), "lesionPath": str(image)}


@pytest.mark.parametrize("bound", ["count", "bytes"])
def test_import_limits_precede_registration_or_transfer(tmp_path, monkeypatch, bound):
    case = fixture_case(); events = []; runtime = BridgeRuntime(tmp_path / "transfer", events.append)
    runtime.session._install_case(case, {}, _Request("install"))
    def forbidden(*args, **kwargs):
        raise AssertionError("Resource refusal must precede numerical import")
    monkeypatch.setattr("resectionlab.prior_proposals.import_registered_prior_proposals", forbidden)
    if bound == "bytes":
        monkeypatch.setattr("resectionlab.desktop_bridge.MAX_CASE_BYTES", runtime.session._case_array_bytes(case) + 1)
    try:
        result = call(runtime, events, "import", "importPriorProposals",
                      {"caseHash": case.semantic_hash, **import_paths(tmp_path, 17 if bound == "count" else 1)})
        assert result["error"]["code"] == "CASE_SIZE_LIMIT"
        assert list(runtime.session.cases) == [case.semantic_hash]
    finally:
        runtime.close()


def test_cancelled_prior_install_cleans_orphan_arrays_and_preserves_case(tmp_path, monkeypatch):
    case = fixture_case(); item = proposal(case)
    enriched = case.revised(prior_proposals={item.proposal_id: item})
    events = []; runtime = BridgeRuntime(tmp_path / "transfer", events.append)
    runtime.session._install_case(case, {}, _Request("install"))
    original_paths = {path.name for path in runtime.session.transfers.root.iterdir()}
    entered, release = threading.Event(), threading.Event()
    def importer(*args, **kwargs):
        return enriched
    monkeypatch.setattr("resectionlab.prior_proposals.import_registered_prior_proposals", importer)
    transfer = runtime.session.transfers.array
    def paused_transfer(value, dtype):
        result = transfer(value, dtype)
        if value is item.data:
            entered.set(); assert release.wait(5)
        return result
    monkeypatch.setattr(runtime.session.transfers, "array", paused_transfer)
    try:
        runtime.submit({"id": "import", "op": "importPriorProposals", "args": {"caseHash": case.semantic_hash, **import_paths(tmp_path, 1)}})
        assert entered.wait(3)
        runtime.submit({"id": "cancel", "op": "cancel", "args": {"requestId": "import"}})
        release.set(); assert runtime.wait_idle(5)
        assert [value for value in events if value.get("id") == "import"][-1]["event"] == "cancelled"
        assert list(runtime.session.cases) == [case.semantic_hash]
        assert {path.name for path in runtime.session.transfers.root.iterdir()} == original_paths
    finally:
        release.set(); runtime.close()
