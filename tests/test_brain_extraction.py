"""Extraction contracts; synthetic fixtures do not measure clinical accuracy."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import nibabel as nib
import numpy as np
import pytest

import resectionlab.brain_extraction as be
from resectionlab.imaging import load_fractional_annotation_case


def save_image(path, data, affine=None):
    image = nib.Nifti1Image(data, np.eye(4) if affine is None else affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
    return path


def test_assets_missing_and_modified_are_named_failures_without_network(tmp_path):
    with pytest.raises(be.BrainExtractionError, match="MODEL_ASSET_MISSING"):
        be.ensure_model_assets(tmp_path, model="nocsf")
    (tmp_path / "mri_synthstrip").write_text("altered checkpoint runner")
    with pytest.raises(be.BrainExtractionError, match="MODEL_ASSET_HASH_MISMATCH"):
        be.ensure_model_assets(tmp_path, model="nocsf")
    with pytest.raises(be.BrainExtractionError, match="UNSUPPORTED_EXTRACTION_MODEL"):
        be.ensure_model_assets(tmp_path, model="invented")


def test_mps_adapter_requires_exact_device_block_and_marks_modification():
    device_block = """        if not torch.cuda.is_available():
            sf.system.fatal('-g flag provided but CUDA is not available')
        device = torch.device('cuda')
        device_name = 'GPU'"""
    original = "# upstream\n" + device_block + "\n# untouched model and resampling"
    adapted = be.mps_runner_source(original, 4 * 1024**3)
    assert adapted.startswith("# MODIFIED by RessectionLab")
    assert "set_per_process_memory_fraction" in adapted
    assert adapted.endswith("# untouched model and resampling")
    assert "device = torch.device('mps')" in adapted
    with pytest.raises(be.BrainExtractionError, match="UPSTREAM_ADAPTER_MISMATCH"):
        be.mps_runner_source("unexpected upstream revision", 4 * 1024**3)


def test_mask_native_frame_and_binary_contract(tmp_path):
    affine = np.diag([2., 3., 4., 1.])
    affine[:3, 3] = [14, -27, 10]
    t1 = save_image(tmp_path / "t1.nii.gz", np.ones((8, 9, 10), np.float32), affine)
    mask_path = save_image(tmp_path / "mask.nii.gz", np.ones((8, 9, 10), np.uint8), affine)
    mask, qc = be.validate_mask(mask_path, t1)
    assert mask.dtype == bool
    np.testing.assert_allclose(qc["affine_ras_mm"], affine)
    shifted = affine.copy()
    shifted[0, 3] += 2
    save_image(mask_path, mask.astype(np.uint8), shifted)
    with pytest.raises(be.BrainExtractionError, match="EXTRACTION_FRAME_MISMATCH"):
        be.validate_mask(mask_path, t1)
    save_image(mask_path, np.ones(mask.shape, np.float32) * 0.5, affine)
    with pytest.raises(be.BrainExtractionError, match="EXTRACTION_MASK_NONBINARY"):
        be.validate_mask(mask_path, t1)
    save_image(mask_path, np.zeros(mask.shape, np.uint8), affine)
    with pytest.raises(be.BrainExtractionError, match="EMPTY_EXTRACTION_MASK"):
        be.validate_mask(mask_path, t1)


@pytest.mark.parametrize("validator,code", [(be.validate_mask, "EXTRACTION_FRAME_MISMATCH"),
                                            (be.validate_distance_map, "EXTRACTION_DISTANCE_FRAME_MISMATCH")])
@pytest.mark.parametrize("scale_drift,acceptable", [(0.001, False), (0.00001, True)])
def test_full_image_corner_drift_bounds_mask_and_distance_frames(tmp_path, validator, code, scale_drift, acceptable):
    shape = (256, 4, 4)
    image = save_image(tmp_path / "t1.nii.gz", np.ones(shape, np.float32))
    drifted = np.eye(4)
    drifted[0, 0] += scale_drift
    derived = save_image(tmp_path / "derived.nii.gz", np.ones(shape, np.uint8), drifted)
    # Entrywise tolerance would accept both, but 0.001 scale drift is 0.255 mm
    # at the far voxel centre. The allowed tolerance is 0.01 mm over the grid.
    assert np.allclose(drifted, np.eye(4), atol=.01, rtol=1e-5)
    if acceptable:
        result = validator(derived, image)
        qc = result[1] if isinstance(result, tuple) else result["native_geometry"]
        assert 0 < qc["maximum_native_corner_displacement_mm"] < .01
    else:
        with pytest.raises(be.BrainExtractionError, match=code):
            validator(derived, image)


def test_fractional_opposite_axis_tumor_is_reindexed_before_extraction_qc(tmp_path):
    shape = (12, 14, 16)
    image = save_image(tmp_path / "t1.nii.gz", np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
    annotation = np.zeros(shape, np.float32)
    annotation[2:4, 5:8, 6:9] = 0.8
    annotation[6:8, 5:8, 6:9] = 0.3
    reversed_affine = np.eye(4)
    reversed_affine[0, 0], reversed_affine[0, 3] = -1, shape[0] - 1
    source = save_image(tmp_path / "annotation.nii.gz", annotation[::-1], reversed_affine)
    case = load_fractional_annotation_case(image, source, threshold=0.5,
                                           annotation_interpretation="fractional intensity, not probability")
    tumor = next(iter(case.compartments.values()))
    np.testing.assert_array_equal(tumor, annotation >= 0.5)
    mask = np.zeros(shape, bool)
    mask[1:6, 3:10, 4:12] = True
    qc = be.extraction_qc(mask, np.eye(4), tumor=tumor)
    assert qc["source_annotation_inclusion_fraction"] == 1
    assert qc["brain_reviewed"] is False
    assert qc["cortical_access_permitted"] is False


def test_qc_uses_physical_volume_and_flags_missing_annotation_support():
    mask = np.zeros((12, 14, 16), bool)
    mask[1:7, 3:10, 4:12] = True
    tumor = np.zeros_like(mask)
    tumor[6:8, 5:8, 6:9] = True
    qc = be.extraction_qc(mask, np.diag([2., 3., 4., 1.]), tumor=tumor, baseline=mask.copy())
    assert qc["mask_volume_ml"] == pytest.approx(mask.sum() * 24 / 1000)
    assert qc["source_annotation_inclusion_fraction"] == 0.5
    assert "SOURCE_ANNOTATION_EXTENDS_OUTSIDE_EXTRACTION" in qc["flags"]
    assert qc["baseline_dice_agreement"] == 1
    assert qc["clinical_deficit_probability"] is None
    assert qc["cortex_localized"] is False
    mask[0, 0, 0] = True
    qc = be.extraction_qc(mask, np.eye(4))
    assert "MULTIPLE_EXTRACTION_COMPONENTS" in qc["flags"]
    assert "EXTRACTION_TOUCHES_INPUT_FOV_BOUNDARY" in qc["flags"]


def test_one_omitted_annotation_voxel_remains_visible_below_two_percent():
    tumor = np.zeros((20, 20, 20), bool)
    tumor[5:15, 5:15, 5:15] = True
    mask = tumor.copy()
    mask[5, 5, 5] = False
    qc = be.extraction_qc(mask, np.eye(4), tumor=tumor)
    assert qc["source_annotation_inclusion_fraction"] == .999
    assert qc["source_annotation_outside_voxels"] == 1
    assert "SOURCE_ANNOTATION_EXTENDS_OUTSIDE_EXTRACTION" in qc["flags"]


@pytest.mark.parametrize("affine", [np.zeros((4, 4)), np.eye(3), np.full((4, 4), np.nan)])
def test_qc_rejects_invalid_affine(affine):
    with pytest.raises(be.BrainExtractionError, match="INVALID_QC_AFFINE"):
        be.extraction_qc(np.ones((8, 8, 8), bool), affine)


def test_intensity_baseline_removes_thin_external_shell_but_stays_unreviewed():
    xyz = np.indices((48, 48, 48)) - 24
    distance = np.sqrt((xyz**2).sum(axis=0))
    image = np.zeros(distance.shape, np.float32)
    image[distance < 13] = 100
    image[(distance > 19) & (distance < 21)] = 80
    image[(distance >= 13) & (distance < 17)] = 15
    core, report = be.intensity_core_baseline(image, np.ones(3))
    assert core[24, 24, 24]
    assert not core[distance > 14].any()
    assert report["reviewed"] is False
    assert report["whole_brain_coverage"] == "unknown"
    with pytest.raises(be.BrainExtractionError, match="INVALID_BASELINE_SPACING"):
        be.intensity_core_baseline(image, np.array([1, 0, 1]))


def fake_assets(tmp_path, monkeypatch, script):
    cache = tmp_path / "cache"
    cache.mkdir()
    assets = {}
    for name, body in [("mri_synthstrip", script), ("LICENSE_FreeSurfer.txt", "synthetic fixture"),
                       ("synthstrip.nocsf.1.pt", "synthetic fixture")]:
        (cache / name).write_text(body)
        assets[name] = ("https://example.invalid/unused", sha256(body.encode()).hexdigest())
    monkeypatch.setattr(be, "ASSETS", assets)
    original_run = subprocess.run
    def run(command, **kwargs):
        if len(command) > 1 and command[1] == "-c":
            return subprocess.CompletedProcess(command, 0, json.dumps({"fixture": True}), "")
        return original_run(command, **kwargs)
    monkeypatch.setattr(be.subprocess, "run", run)
    return cache


def test_actual_child_timeout_is_recorded_and_does_not_grant_review(tmp_path, monkeypatch):
    image = save_image(tmp_path / "t1.nii.gz", np.ones((8, 8, 8), np.float32))
    cache = fake_assets(tmp_path, monkeypatch, "import time\ntime.sleep(20)\n")
    output = tmp_path / "result"
    with pytest.raises(be.BrainExtractionError, match="EXTRACTION_TIME_BUDGET_EXCEEDED"):
        be.run_synthstrip(image, cache, output, timeout_seconds=1)
    record = json.loads((output / "nocsf_inference_record.json").read_text())
    assert record["exit_code"] != 0
    assert record["brain_reviewed"] is False
    assert record["cortical_access_permitted"] is False
    assert record["elapsed_seconds"] < 10
    assert not (output / "nocsf_mask.nii.gz").exists()


@pytest.mark.parametrize("mode,expected", [
    ("frame", "EXTRACTION_DISTANCE_FRAME_MISMATCH"),
    ("shape", "EXTRACTION_DISTANCE_FRAME_MISMATCH"),
    ("nonfinite", "EXTRACTION_DISTANCE_NONFINITE"),
    ("valid", None),
])
def test_child_success_requires_distance_map_geometry_and_values(tmp_path, monkeypatch, mode, expected):
    image = save_image(tmp_path / "t1.nii.gz", np.ones((8, 8, 8), np.float32))
    script = f'''import argparse
import nibabel as nib
import numpy as np
p=argparse.ArgumentParser()
for name in ("image", "mask", "sdt"):
    p.add_argument("--"+name)
args,_=p.parse_known_args()
source=nib.load(args.image)
mask=nib.Nifti1Image(np.ones(source.shape,np.uint8),source.affine)
mask.header.set_xyzt_units("mm")
nib.save(mask,args.mask)
distance=np.linspace(-4,10,np.prod(source.shape),dtype=np.float32).reshape(source.shape)
affine=source.affine.copy()
if {mode!r} == "frame": affine[0,3] += 5
if {mode!r} == "shape": distance=distance[:-1]
if {mode!r} == "nonfinite": distance[1,1,1]=np.nan
output=nib.Nifti1Image(distance,affine)
output.header.set_xyzt_units("mm")
nib.save(output,args.sdt)
'''
    cache = fake_assets(tmp_path, monkeypatch, script)
    output = tmp_path / "result"
    if expected:
        with pytest.raises(be.BrainExtractionError, match=expected):
            be.run_synthstrip(image, cache, output)
    else:
        be.run_synthstrip(image, cache, output)
    record = json.loads((output / "nocsf_inference_record.json").read_text())
    assert record["exit_code"] == 0
    assert record["failure"] == expected
    assert record["brain_reviewed"] is False
    if expected:
        assert "artifact_hashes" not in record
    else:
        assert record["distance_map_qc"]["finite_values"] is True
        assert record["distance_map_qc"]["minimum_mm"] == -4
        assert record["distance_map_qc"]["maximum_mm"] == 10
        assert len(record["artifact_hashes"]) == 2


def test_existing_output_is_never_reused_after_failed_rerun(tmp_path, monkeypatch):
    image = save_image(tmp_path / "t1.nii.gz", np.ones((8, 8, 8), np.float32))
    cache = fake_assets(tmp_path, monkeypatch, "raise RuntimeError('must not execute')")
    output = tmp_path / "result"
    output.mkdir()
    existing = save_image(output / "nocsf_mask.nii.gz", np.ones((8, 8, 8), np.uint8))
    digest = sha256(existing.read_bytes()).hexdigest()
    with pytest.raises(be.BrainExtractionError, match="EXTRACTION_OUTPUT_EXISTS"):
        be.run_synthstrip(image, cache, output)
    assert sha256(existing.read_bytes()).hexdigest() == digest


def test_cli_refuses_existing_experiment_before_overwriting_baseline(tmp_path, monkeypatch):
    output = tmp_path / "experiment"
    output.mkdir()
    marker = output / "intensity_core_baseline.nii.gz"
    marker.write_bytes(b"earlier artifact")
    monkeypatch.setattr(be.sys, "argv", ["brain_extraction", "--t1", "unused-input.nii.gz",
                                        "--cache", str(tmp_path), "--output", str(output)])
    with pytest.raises(be.BrainExtractionError, match="EXTRACTION_OUTPUT_EXISTS"):
        be.main()
    assert marker.read_bytes() == b"earlier artifact"
