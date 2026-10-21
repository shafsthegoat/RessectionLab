"""Generated software controls; no real image/model execution in tests."""
import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("poc_ixi_vessel", Path(__file__).resolve().parents[1]/"scripts/poc_ixi265_vesselfm.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_request_contains_one_angiogram_and_no_annotation_input(tmp_path):
    weights = tmp_path/"fixture-weights"; weights.write_bytes(b"test")
    request = runner.make_request(tmp_path/"acquired-MRA.nii.gz", weights, tmp_path/"output")
    assert set(request) == {"image", "case", "model_id", "modality", "device", "output", "weights", "parameters"}
    assert request["modality"] == "MRA" and request["parameters"]["postprocessing"] is False
    assert not any("reference" in str(path) for path, _ in runner.required_paths().values())


def test_checksum_refuses_mutated_prediction(tmp_path):
    path = tmp_path/"saved-mask"; path.write_bytes(b"original")
    digest = runner.sha(path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Pinned artifact mismatch"):
        runner.verify(path, digest)


def test_native_score_threshold_and_frame_are_bound(tmp_path):
    reference = nib.Nifti1Image(np.zeros((7, 8, 9), np.float32), np.diag([1., 2., 3., 1.]))
    reference.set_qform(reference.affine, 1); reference.set_sform(reference.affine, 1)
    score = np.zeros(reference.shape, np.float32); score[3, 4, 5] = .75
    mask_path = tmp_path/"mask.nii.gz"; score_path = tmp_path/"score.nii.gz"
    runner.save_native(score > .5, reference, mask_path, binary=True)
    runner.save_native(score, reference, score_path)
    assert runner.validate_native(mask_path, score_path, reference)["foreground_voxels"] == 1
    np.testing.assert_array_equal(nib.load(mask_path).get_qform(), reference.get_qform())
    score[3, 4, 5] = .25
    runner.save_native(score, reference, score_path)
    with pytest.raises(ValueError, match="threshold"):
        runner.validate_native(mask_path, score_path, reference)
    shifted = nib.Nifti1Image(np.zeros(reference.shape), reference.affine+np.array([[0,0,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]))
    with pytest.raises(ValueError, match="geometry"):
        runner.validate_native(mask_path, score_path, shifted)


def test_estimated_mask_transfer_obeys_physical_sampling_direction_and_coverage():
    sitk = pytest.importorskip("SimpleITK")
    from resectionlab.preprocessing import RAS_TO_LPS
    source = np.zeros((11, 12, 13), np.float32); source[6, 5, 7] = 1
    coverage = np.zeros_like(source); coverage[1:-1, 1:-1, 1:-1] = 1
    affine = np.array([[0., -2., 0., 15.], [1.5, 0., 0., -8.], [0., 0., 3., 6.], [0., 0., 0., 1.]])
    translation = affine[:3, 0]*2
    transform = sitk.TranslationTransform(3, tuple(RAS_TO_LPS[:3, :3]@translation))
    mapped = runner.map_to_target(source, affine, source.shape, affine, transform, nearest=True)
    domain = runner.map_to_target(coverage, affine, source.shape, affine, transform, nearest=True)
    assert np.argwhere(mapped > 0).tolist() == [[4, 5, 7]]
    assert not domain[-2:].any()
    assert not ((mapped > 0) & (domain == 0)).any()


def test_candidate_preserves_parent_arrays_originals_and_reference_sources(tmp_path):
    from resectionlab.core import CaseData, SourceRef
    from resectionlab.workspace_bundle import OriginalFile, load_workspace, save_workspace, view_state
    from resectionlab.workspace_imaging import DisplaySource
    output = tmp_path/"estimates"; output.mkdir()
    shape = (9, 10, 11)
    affine = np.diag([1., 1.2, 1.4, 1.])
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    label = np.zeros(shape, bool); label[2:5, 3:6, 4:7] = True
    metadata = {"patient_group": "IXI:265", "split": {"role": "TRAIN", "training_admitted": False},
                "workspace_registration": {"kind": "estimated-unreviewed", "overlayPermitted": False}}
    primary = CaseData(runner.CASE, values, {}, affine,
        (SourceRef("structural", "generated:T1", sha256=runner.T1_SHA, provenance="observed"),),
        metadata=metadata, brain_mask=values > 100)
    sources = []
    for modality, digest in [("MRA", runner.MRA_SHA), ("T1", runner.T1_SHA)]:
        case = CaseData(runner.CASE, values, {"source_vessels": label}, affine,
            (SourceRef("structural", "generated:"+modality, sha256=digest, provenance="observed"),
             SourceRef("supplied_target_annotation", "generated:source-reference", sha256="a"*64, provenance="observed")),
            metadata=metadata)
        sources.append(DisplaySource(case, modality, "source-provided"))
    originals = []
    for i in range(16):
        path = tmp_path/f"original-{i}"; path.write_bytes(f"generated original {i}".encode())
        originals.append(OriginalFile.from_path(path, role="metadata"))
    host = tmp_path/"host.ressectionlab"
    save_workspace(primary, host, sources=sources, imaging=view_state(primary, sources),
                   episode=None, selection=None, artifacts={}, originals=originals)
    prediction = np.zeros(shape, np.uint8); prediction[6, 7, 8] = 1
    reference = nib.Nifti1Image(values, affine)
    for name, array in [("vessels_mask.nii.gz", prediction), ("vessels_mask_on_T1_estimated.nii.gz", prediction),
                        ("native_sampling_coverage.nii.gz", np.ones(shape, np.uint8)), ("T1_sampling_coverage.nii.gz", np.ones(shape, np.uint8))]:
        runner.save_native(array, reference, output/name, binary=True)
    runner.write_json(output/"inference.json", {"status": "succeeded", "model_id": "vesselfm-base", "input": {"sha256": runner.MRA_SHA},
        "identity": {"weights_sha256": runner.WEIGHT_SHA}, "cache_key": "generated-only", "training_lineage": {"test_fixture": True},
        "model": {"name": "generated-test-model", "version": "test-only", "license": "test-only"},
        "outputs": [{"path": str(output/"vessels_mask.nii.gz"), "sha256": runner.sha(output/"vessels_mask.nii.gz")}]})
    runner.write_json(output/"registered.json", {"inference_sha256": runner.sha(output/"inference.json"),
        "registration_sha256": runner.REGISTRATION_SHA, "anatomical_registration_accepted": False,
        "outputs": [{"path": str(output/name), "sha256": runner.sha(output/name)} for name in
                    ["vessels_mask_on_T1_estimated.nii.gz", "native_sampling_coverage.nii.gz", "T1_sampling_coverage.nii.gz"]]})
    for name in ("prepared.json", "visual-qc.json"):
        runner.write_json(output/name, {"generated_test": True})
    host_digest = runner.sha(host)
    report = runner.package(output, host_bundle=host, host_digest=host_digest)
    assert runner.sha(host) == host_digest
    assert report["fusion_permitted"] == [False, False, False]
    assert report["preserved_parent_originals"] == 16 and report["total_originals"] == 24
    loaded, _, restored = load_workspace(report["bundle"])
    assert loaded.semantic_hash == primary.semantic_hash and loaded.planning_hash == primary.planning_hash
    assert not loaded.compartments and not loaded.critical_evidence and not loaded.prior_proposals
    assert restored["originals"][:16] == originals
    assert [item.annotation_kind for item in restored["sources"]] == ["source-provided", "source-provided", "estimated"]
    np.testing.assert_array_equal(restored["sources"][0].case.compartments["source_vessels"], label)
    estimated = restored["sources"][2].case
    np.testing.assert_array_equal(estimated.compartments["estimated_vessels"], prediction > 0)
    assert estimated.metadata["anatomy_layers"]["estimated_vessels"]["policy_use"] == "excluded"
    assert estimated.source_refs[-1].provenance == "estimated"
    with pytest.raises(ValueError, match="already exists"):
        runner.package(output, host_bundle=host, host_digest=host_digest)
