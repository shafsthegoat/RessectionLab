"""Generated controls for direction, partial coverage and input rejection."""
import gzip
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import map_coordinates

spec = importlib.util.spec_from_file_location("poc_ixi265", Path(__file__).resolve().parents[1]/"scripts/poc_ixi265_registration.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_hash_mismatch_rejects_before_nifti_or_gzip_decode(tmp_path, monkeypatch):
    path = tmp_path/"untrusted.gz"
    path.write_bytes(b"not an image")
    monkeypatch.setattr(runner.nib, "load", lambda *_: pytest.fail("decoded before hash validation"))
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.checked_source(path, "0"*64)


def test_complete_payload_and_crc_are_checked(tmp_path):
    path = tmp_path/"valid.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((7, 8, 9), np.float32), np.eye(4)), path)
    image, values, count = runner.checked_source(path, runner.sha(path))
    assert values.shape == (7, 8, 9) and count == 352+7*8*9*4
    payload = gzip.decompress(path.read_bytes())
    path.write_bytes(gzip.compress(payload+b"extra"))
    with pytest.raises(ValueError, match="payload length"):
        runner.checked_source(path, runner.sha(path))


def test_binary_validation_does_not_sort_and_rejects_nonbinary(monkeypatch):
    monkeypatch.setattr(runner.np, "unique", lambda *_: pytest.fail("binary validation must not sort"))
    assert runner.nonempty_binary(np.array([0., 1., 0.]))
    assert not runner.nonempty_binary(np.zeros(3))
    assert not runner.nonempty_binary(np.array([0., .5, 1.]))
    assert not runner.nonempty_binary(np.array([0., np.nan, 1.]))


def test_ras_sampling_direction_and_inverse_with_oblique_physical_grid():
    sitk = pytest.importorskip("SimpleITK")
    from resectionlab.preprocessing import RAS_TO_LPS, sitk_transform_to_ras
    moving = np.zeros((12, 13, 14), np.float32)
    moving[7, 6, 8] = 1
    affine = np.array([[0., -2., 0., 20.], [1.5, 0., 0., -10.], [0., 0., 3., 7.], [0., 0., 0., 1.]])
    # Input index x=7 must appear at output index x=5: output->input adds +2 index-x.
    translation_ras = affine[:3, 0]*2
    transform = sitk.TranslationTransform(3, tuple(RAS_TO_LPS[:3, :3]@translation_ras))
    result = runner.resample(moving, affine, moving, affine, transform, label=True)
    assert np.argwhere(result == 1).tolist() == [[5, 6, 8]]
    matrix = sitk_transform_to_ras(transform)
    point = nib.affines.apply_affine(affine, [5, 6, 8])
    np.testing.assert_allclose(nib.affines.apply_affine(matrix, point), nib.affines.apply_affine(affine, [7, 6, 8]))
    np.testing.assert_allclose(nib.affines.apply_affine(np.linalg.inv(matrix), nib.affines.apply_affine(matrix, point)), point)


def test_partial_slab_coverage_excludes_unknown_area_and_border():
    sitk = pytest.importorskip("SimpleITK")
    source = runner.interior_coverage((10, 12, 6))
    source_affine = np.eye(4)
    source_affine[2, 3] = 7
    target = np.zeros((10, 12, 24), np.float32)
    result = runner.resample(source, source_affine, target, np.eye(4), sitk.Transform(3, sitk.sitkIdentity), label=True)
    assert not result[:, :, :8].any() and not result[:, :, 12:].any()
    assert result[4, 5, 8:12].tolist() == [1, 1, 1, 1]
    assert not result[0].any()


def test_rigid_control_recovers_injected_translation_without_accepting_anatomy():
    pytest.importorskip("SimpleITK")
    grid = np.indices((42, 42, 42), dtype=float)
    moving = np.zeros((42, 42, 42), np.float32)
    for center, sigma, strength in [((20, 21, 22), (8, 9, 10), .4), ((14, 16, 25), (3, 4, 5), .9),
                                    ((29, 25, 14), (4, 3, 5), .7), ((24, 12, 17), (3, 4, 3), .8)]:
        moving += strength*np.exp(-sum(((grid[d]-center[d])/sigma[d])**2 for d in range(3))/2)
    affine = np.diag([2., 2., 2., 1.])
    affine[:3, 3] = -42
    translation = np.array([3., -2., 1.])
    fixed = map_coordinates(moving, grid+translation[:, None, None, None]/2, order=1)
    _, report = runner.rigid_candidate(fixed, affine, moving, affine, fixed > .06, timeout=30)
    matrix = np.asarray(report["T1_RAS_mm_to_MRA_RAS_mm"])
    np.testing.assert_allclose(matrix[:3, 3], translation, atol=.65)
    assert report["matched_objective"]["after"] > report["matched_objective"]["before"]
    assert report["anatomical_registration_accepted"] is False
    assert report["registration_reviewed"] is False
    np.testing.assert_allclose(matrix@report["MRA_RAS_mm_to_T1_RAS_mm"], np.eye(4), atol=1e-10)


def test_canonical_package_keeps_reference_out_of_primary_and_preserves_originals(tmp_path, monkeypatch):
    from resectionlab.workspace_bundle import load_workspace
    output = tmp_path/"output"
    for name in ("originals", "registration", "synthstrip"):
        (output/name).mkdir(parents=True)
    image = np.arange(12**3, dtype=np.float32).reshape((12,)*3)
    label = np.zeros_like(image, np.uint8)
    label[5:7, 5:7, 5:7] = 1
    brain = np.zeros_like(label)
    brain[2:-2, 2:-2, 2:-2] = 1
    affine = np.diag([1., 1.2, 1.4, 1.])
    def save(values, path):
        result = nib.Nifti1Image(values, affine)
        result.header.set_xyzt_units("mm")
        result.set_qform(affine, 1)
        result.set_sform(affine, 1)
        nib.save(result, path)
    local = {}
    for key, name, values in [("T1", "t1.nii.gz", image), ("MRA", "mra.nii.gz", image+1), ("reference", "reference.nii.gz", label)]:
        path = output/"originals"/name
        save(values, path)
        local[key] = (name, runner.sha(path))
    monkeypatch.setattr(runner, "ALLOWLIST", local)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "OLD", tmp_path)
    brain_path = output/"synthstrip/brain_mask.nii.gz"
    save(brain, brain_path)
    save(label, output/"registration/source_vessels_on_T1_candidate.nii.gz")
    weight = tmp_path/"generated-test-weight"
    weight.write_bytes(b"explicit generated fixture; not a medical model")
    runner.write_json(output/"synthstrip/inference.json", {
        "schema": "resectionlab-poc-inference-v1", "status": "succeeded", "case": runner.CASE,
        "input": {"sha256": local["T1"][1]}, "outputs": [{"path": str(brain_path), "sha256": runner.sha(brain_path)}],
        "model": {"name": "generated fixture", "version": "test-only", "license": "test-only"},
        "weights": [{"path": str(weight), "sha256": runner.sha(weight), "bytes": weight.stat().st_size}], "cache_key": "generated-test-key"})
    runner.write_json(output/"registration/registration.json", {"anatomical_registration_accepted": False})
    runner.write_json(output/"source-receipt.json", {"generated_test": True})
    for relative in ("artifacts/ixi-t1-mra-vessel-byte-intake-v1/source-rights-and-head-proofs.json",
                     "build/ixi-vessel-source-preparation-v1/ixi-page.html", "build/ixi-vessel-source-preparation-v1/zenodo-record.json"):
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated software-control fixture")
    runner.package(output)
    primary, _, restored = load_workspace(output/(runner.CASE+".ressectionlab"))
    assert not primary.compartments and not primary.critical_evidence and not primary.prior_proposals
    assert primary.brain_mask.any()
    assert primary.metadata["patient_group"] == "IXI:265"
    assert primary.metadata["split"]["role"] == "TRAIN"
    assert primary.metadata["split"]["training_admitted"] is False
    assert len(restored["sources"]) == 2
    assert all(set(source.case.compartments) == {"source_vessels"} for source in restored["sources"])
    receipt = json.loads((output/"bundle-receipt.json").read_text())
    assert receipt["fusion_permitted"] == [False, False]
    assert receipt["anatomical_registration_accepted"] is False
    saved_hashes = {original.sha256 for original in restored["originals"]}
    assert {digest for _, digest in local.values()} <= saved_hashes
