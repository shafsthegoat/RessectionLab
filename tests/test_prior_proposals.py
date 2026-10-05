"""Registered previews remain exact source-bound view-only proposals."""
from dataclasses import replace
from io import BytesIO
import json
import warnings
import importlib.util
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import nibabel as nib
import numpy as np
import pytest

from resectionlab.anatomy import load_functional_prior, register_prior
from resectionlab.core import CaseData, SourceRef
from resectionlab.imaging import ImagingError, file_sha256, load_case, save_case
from resectionlab.prior_proposals import RegisteredPriorProposal, import_registered_prior_proposals


def save_image(path, values, affine=None):
    image = nib.Nifti1Image(np.asarray(values, dtype=np.float32), np.eye(4) if affine is None else affine)
    image.header.set_xyzt_units("mm")
    image.set_qform(image.affine, code=1)
    image.set_sform(image.affine, code=1)
    nib.save(image, path)


@pytest.fixture
def registered_fixture(tmp_path):
    shape = (8, 9, 10)
    x, y, z = np.indices(shape)
    display = (10 + 2 * x + y + z).astype(np.float32)
    registration = (10 + x + y + 3 * z).astype(np.float32)
    labels = np.zeros(shape, np.float32)
    labels[2:4, 3:5, 4:6] = 1
    labels[4:6, 3:5, 4:6] = 2
    image_path, reg_image_path, lesion_path = [tmp_path / name for name in ("display.nii", "registration.nii", "lesion.nii")]
    for path, data in ((image_path, display), (reg_image_path, registration), (lesion_path, labels)):
        save_image(path, data)
    case = CaseData("analytic_prior_fixture", display, {"enhancing": labels == 1, "edema": labels == 2}, np.eye(4),
                    (SourceRef("structural", image_path.as_uri(), file_sha256(image_path)),
                     SourceRef("annotation", lesion_path.as_uri(), file_sha256(lesion_path))),
                    metadata={"label_conventions": {"1": "enhancing", "2": "edema"},
                              "source_files": [{"sha256": file_sha256(reg_image_path), "role": "structural_mri"}]})
    cache, directory = tmp_path / "cache", tmp_path / "registration"
    cache.mkdir(); directory.mkdir()
    prior_path = cache / "analytic_prior.nii"
    save_image(prior_path, (x + y + z) / (sum(shape) - 3))
    specification = {"map_id": "motor_functional_concordance", "component": "motor", "map_kind": "functional_concordance",
                     "record_id": "synthetic-unit-test", "archive_member": "analytic_prior.nii", "cache_file": prior_path.name,
                     "sha256": file_sha256(prior_path), "license": "synthetic_fixture_only", "template_frame": "FSL_MNI152_RAS_mm"}
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "evidence_type": "prior", "maps": [specification]}))
    prior = load_functional_prior(prior_path, specification)
    transform = np.eye(4); transform[0, 3] = 1.5
    registered = register_prior(prior, target_shape=shape, target_affine=np.eye(4), case_hash=case.semantic_hash,
                                mni_ras_to_patient_ras_mm=transform, method="synthetic_known_transform", version="test-v1")
    snapshot = directory / "implementation_snapshot.py"
    snapshot.write_text("# Synthetic known-transform test fixture, no actual registration execution\n")
    report = {"schema": "resectionlab.prior_registration/1", "version": "test-v1",
              "implementation_sha256": file_sha256(snapshot), "case_hash": case.semantic_hash,
              "patient_sha256": file_sha256(reg_image_path), "lesion_sha256": file_sha256(lesion_path),
              "template_sha256": "a" * 64, "template_identity": "synthetic_unit_test_grid",
              "patient_shape": list(shape), "patient_affine_ras_mm": np.eye(4).tolist(),
              "qc_state": "alignment_review_required", "patient_specific_function": False,
              "clinical_deficit_probability": None, "limitations": ["synthetic_unit_test_only"],
              "candidates": [{"candidate_id": "known_translation", "status": "candidate_requires_alignment_review",
                              "qc_state": "alignment_review_required", "mni_ras_to_patient_ras_mm": transform.tolist()}]}
    (directory / "registration_comparison.json").write_text(json.dumps(report))
    (directory / "functional_overlay_inventory.json").write_text(json.dumps([registered.inventory_item()]))
    preview = directory / "motor_functional_concordance_review_preview.npz"
    np.savez_compressed(preview, values=registered.data, sampling_coverage=registered.sampling_coverage,
                        affine_ras_mm=registered.affine_ras_mm)
    kwargs = {"registration_directory": directory, "source_cache_directory": cache,
              "prior_manifest_path": manifest_path, "source_image_path": image_path,
              "registration_image_path": reg_image_path, "lesion_path": lesion_path}
    return case, kwargs, preview


def test_real_import_path_binds_distinct_display_registration_and_lesion_bytes(registered_fixture):
    case, kwargs, _ = registered_fixture
    imported = import_registered_prior_proposals(case, **kwargs)
    proposal = next(iter(imported.prior_proposals.values()))
    assert imported.planning_hash == case.planning_hash and imported.semantic_hash != case.semantic_hash
    assert proposal.registration_case_hash == case.semantic_hash
    assert proposal.source_file_sha256 != proposal.registration_image_sha256
    assert proposal.review_status == "alignment_review_required" and proposal.view_only
    assert not proposal.planning_eligible and not proposal.patient_specific_function
    assert proposal.to_manifest()["clinical_deficit_probability"] is None
    assert proposal.sampling_coverage.mean() < 1
    assert not proposal.data[~proposal.sampling_coverage].any()
    assert not proposal.data.flags.writeable and not proposal.sampling_coverage.flags.writeable
    assert import_registered_prior_proposals(imported, **kwargs) is imported
    assert not case.prior_proposals and case.brain_mask is None


def test_bundle_roundtrip_preserves_registered_data_coverage_and_gates(registered_fixture, tmp_path):
    case, kwargs, _ = registered_fixture
    imported = import_registered_prior_proposals(case, **kwargs)
    path = save_case(imported, tmp_path / "case.ressectionlab")
    restored = load_case(path)
    assert restored.semantic_hash == imported.semantic_hash and restored.planning_hash == case.planning_hash
    first, second = next(iter(imported.prior_proposals.values())), next(iter(restored.prior_proposals.values()))
    assert first.to_manifest() == second.to_manifest()
    assert np.array_equal(first.data, second.data) and np.array_equal(first.sampling_coverage, second.sampling_coverage)
    with ZipFile(path) as archive:
        with np.load(BytesIO(archive.read("arrays.npz")), allow_pickle=False) as arrays:
            assert not any("template" in name or "source_prior" in name for name in arrays.files)
            assert {"prior_values_0", "prior_coverage_0"}.issubset(arrays.files)


def test_empty_proposals_preserve_existing_case_manifest_and_identity(registered_fixture, tmp_path):
    case, _, _ = registered_fixture
    assert "prior_proposals" not in case.to_manifest()
    assert replace(case, prior_proposals={}).semantic_hash == case.semantic_hash
    path = save_case(case, tmp_path / "empty.ressectionlab")
    with ZipFile(path) as archive:
        assert "prior_proposals" not in json.loads(archive.read("manifest.json"))
    assert load_case(path).semantic_hash == case.semantic_hash


@pytest.mark.parametrize("field", ["review_status", "planning_eligible", "patient_specific_function", "clinical_deficit_probability", "data_hash", "sampling_coverage_hash", "registration_hash"])
def test_forged_saved_manifest_is_rejected(registered_fixture, field):
    case, kwargs, _ = registered_fixture
    proposal = next(iter(import_registered_prior_proposals(case, **kwargs).prior_proposals.values()))
    manifest = proposal.to_manifest()
    manifest[field] = {"review_status": "alignment_accepted", "planning_eligible": True,
                       "patient_specific_function": True, "clinical_deficit_probability": 0.05}.get(field, "sha256:" + "0" * 64)
    with pytest.raises(ValueError):
        RegisteredPriorProposal.from_manifest(manifest, data=proposal.data, sampling_coverage=proposal.sampling_coverage)


@pytest.mark.parametrize("part", ["values", "sampling_coverage", "affine_ras_mm"])
def test_preview_tampering_fails_reconstruction_check(registered_fixture, part):
    case, kwargs, preview = registered_fixture
    with np.load(preview, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if part == "values":
        arrays[part][4, 4, 4] += 0.01
    elif part == "sampling_coverage":
        arrays[part][4, 4, 4] = False
    else:
        arrays[part][0, 3] += 1
    np.savez_compressed(preview, **arrays)
    with pytest.raises(ValueError, match="Saved prior preview differs"):
        import_registered_prior_proposals(case, **kwargs)


def test_active_label_swap_with_unchanged_union_is_rejected(registered_fixture):
    case, kwargs, _ = registered_fixture
    changed = case.revised(compartments={"enhancing": case.compartments["edema"], "edema": case.compartments["enhancing"]})
    assert np.array_equal(np.logical_or.reduce(list(case.compartments.values())), np.logical_or.reduce(list(changed.compartments.values())))
    with pytest.raises(ValueError, match="compartment labels differ"):
        import_registered_prior_proposals(changed, **kwargs)
    imported = import_registered_prior_proposals(case, **kwargs)
    with pytest.raises(ValueError, match="different case inputs|exact lesion labels"):
        imported.revised(compartments=changed.compartments)


@pytest.mark.parametrize("part", ["source_image_path", "registration_image_path", "lesion_path"])
def test_changed_source_bytes_are_rejected(registered_fixture, part):
    case, kwargs, _ = registered_fixture
    path = kwargs[part]
    image = nib.load(path)
    values = image.get_fdata().copy(); values[0, 0, 0] += 1
    save_image(path, values)
    with pytest.raises(ValueError, match="source hashes"):
        import_registered_prior_proposals(case, **kwargs)


def test_cancel_is_atomic_and_source_cache_traversal_is_rejected(registered_fixture):
    case, kwargs, _ = registered_fixture
    with pytest.raises(ValueError, match="PRIOR_IMPORT_CANCELLED"):
        import_registered_prior_proposals(case, **kwargs, cancelled=lambda: True)
    assert not case.prior_proposals
    path = kwargs["prior_manifest_path"]
    manifest = json.loads(path.read_text()); manifest["maps"][0]["cache_file"] = "../display.nii"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        import_registered_prior_proposals(case, **kwargs)


@pytest.mark.parametrize("field", ["data", "sampling_coverage", "affine_ras_mm", "source_prior_affine_ras_mm", "mni_ras_to_patient_ras_mm"])
def test_array_metadata_mutation_cannot_retain_saved_identity(registered_fixture, field):
    case, kwargs, _ = registered_fixture
    proposal = next(iter(import_registered_prior_proposals(case, **kwargs).prior_proposals.values()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        getattr(proposal, field).shape = (-1,)
    with pytest.raises(ValueError, match="array metadata changed"):
        proposal.to_manifest()


def test_saved_bundle_cannot_promote_review_by_metadata_edit(registered_fixture, tmp_path):
    case, kwargs, _ = registered_fixture
    imported = import_registered_prior_proposals(case, **kwargs)
    path = save_case(imported, tmp_path / "case.ressectionlab")
    with ZipFile(path) as archive:
        payload, manifest = archive.read("arrays.npz"), json.loads(archive.read("manifest.json"))
    next(iter(manifest["prior_proposals"].values()))["manifest"]["planning_eligible"] = True
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest)); archive.writestr("arrays.npz", payload)
    with pytest.raises(ImagingError, match="mandatory review gates"):
        load_case(path)


@pytest.mark.parametrize("defect", ["transform", "accepted", "source", "duplicate"])
def test_inventory_cannot_substitute_transform_source_or_review(registered_fixture, defect):
    case, kwargs, _ = registered_fixture
    path = kwargs["registration_directory"] / "functional_overlay_inventory.json"
    inventory = json.loads(path.read_text())
    if defect == "transform":
        inventory[0]["mni_ras_to_patient_ras_mm"][0][3] += 1
    elif defect == "accepted":
        inventory[0]["qc_state"] = "alignment_accepted"
    elif defect == "source":
        inventory[0]["source"]["sha256"] = "0" * 64
    else:
        inventory.append(inventory[0])
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError):
        import_registered_prior_proposals(case, **kwargs)


def test_small_affine_scale_drift_accumulates_across_physical_grid(registered_fixture):
    case, kwargs, _ = registered_fixture
    proposal = next(iter(import_registered_prior_proposals(case, **kwargs).prior_proposals.values()))
    drift = proposal.affine_ras_mm.copy(); drift[2, 2] += 0.005
    # Matrix difference looks small; displacement reaches 0.045 mm at the far face.
    changed = replace(proposal, affine_ras_mm=drift)
    with pytest.raises(ValueError, match="physical frame"):
        replace(case, prior_proposals={changed.proposal_id: changed})


@pytest.mark.parametrize("kind", ["existing_file", "dangling_symlink"])
def test_cli_preserves_existing_output_before_import(registered_fixture, tmp_path, monkeypatch, kind):
    _, kwargs, _ = registered_fixture
    script = Path(__file__).resolve().parents[1] / "scripts/attach_prior_proposals.py"
    spec = importlib.util.spec_from_file_location("attach_prior_proposals", script)
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    destination = tmp_path / "keep.ressectionlab"
    if kind == "existing_file":
        destination.write_bytes(b"existing immutable artifact")
    else:
        destination.symlink_to(tmp_path / "absent.ressectionlab")
    arguments = [str(script), "--case", str(tmp_path / "case.ressectionlab"), "--output", str(destination)]
    for option, key in (("registration-directory", "registration_directory"), ("source-cache-directory", "source_cache_directory"),
                        ("prior-manifest", "prior_manifest_path"), ("source-image", "source_image_path"),
                        ("registration-image", "registration_image_path"), ("lesion", "lesion_path")):
        arguments.extend(["--" + option, str(kwargs[key])])
    monkeypatch.setattr("sys.argv", arguments)
    monkeypatch.setattr(cli, "load_case", lambda _: pytest.fail("Existing output must fail before input loading"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    if kind == "existing_file":
        assert destination.read_bytes() == b"existing immutable artifact"
    else:
        assert destination.is_symlink() and not destination.exists()
