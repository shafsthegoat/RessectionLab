"""Generated-tested, scan-only spatial adapter prototype; never opens labels.

This optional research module makes ANTs' image resampling and point-transform
directions explicit for separately released scan-preparation experiments.
It never loads a model; callers control patient processing. All values are
research estimates.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO


class AdapterAbstain(ValueError):
    """A scan pair must not enter the estimator or planner."""


class MissingModality(AdapterAbstain):
    pass


class SourceMismatch(AdapterAbstain):
    pass


class FrameMismatch(AdapterAbstain):
    pass


class AtlasShapeFailure(AdapterAbstain):
    pass


class CoverageFailure(AdapterAbstain):
    pass


class MaskNotQualified(AdapterAbstain):
    pass


class RegistrationFailure(AdapterAbstain):
    pass


class ModelContractFailure(AdapterAbstain):
    pass


@dataclass(frozen=True)
class Scan:
    modality: str
    path: Path
    expected_sha256: str
    availability_basis: str  # exact "preoperative_source"; not a fabricated time
    acquisition_time_utc: str | None = None


@dataclass(frozen=True)
class QualifiedMask:
    """Scan-derived mask, with an external QC decision already recorded."""

    path: Path
    expected_sha256: str
    source_sha256: str
    frame: str  # "t1c" or "flair"; no labels or outcome maps are accepted
    qc_status: str
    limitations: str = ""


@dataclass(frozen=True)
class PreparedVolume:
    path: Path
    shape_xyz: tuple[int, int, int]
    affine_ras_mm: np.ndarray
    sha256: str


@dataclass(frozen=True)
class ImageTransform:
    """Affine file used to resample moving image into fixed image space.

    For ANTs' affine files, applying the same matrix to a physical *point*
    maps fixed -> moving; image output moves moving -> fixed. Native return
    therefore uses ``whichtoinvert=[True]`` and reverses step order.
    """

    path: Path
    sha256: str
    moving_frame: str
    fixed_frame: str
    generation_receipt_path: Path | None = None
    generation_receipt_sha256: str | None = None


@dataclass(frozen=True)
class PreparedPair:
    """Geometry-only scan preparation; not accepted anatomical QC or inference."""

    t1c_atlas: PreparedVolume
    flair_atlas: PreparedVolume
    atlas: PreparedVolume
    atlas_mask: PreparedVolume
    t1c_native: PreparedVolume
    flair_native: PreparedVolume
    flair_to_t1c: ImageTransform
    t1c_to_atlas: ImageTransform
    atlas_common_fov_fraction: float  # full-atlas FOV statistic; mask support is checked at 100% separately
    qc_status: str = "geometry_only_requires_local_anatomy_review"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_model_metadata_only(model_dir: Path, extracted_receipt: Path) -> dict:
    """Verify pinned nnU-Net metadata and bytes; do not unpickle or run a model."""
    receipt = json.loads(Path(extracted_receipt).read_text())
    expected_files = {"dataset.json", "plans.json", "fold_all/checkpoint_final.pth"}
    if set(receipt.get("files", {})) != expected_files:
        raise ModelContractFailure("pinned extraction manifest has unexpected members")
    for relative in sorted(expected_files):
        candidate = Path(model_dir) / relative
        if not candidate.is_file() or sha256_file(candidate) != receipt["files"][relative]["sha256"]:
            raise ModelContractFailure(f"pinned model file absent or changed: {relative}")
    dataset = json.loads((Path(model_dir) / "dataset.json").read_text())
    plans = json.loads((Path(model_dir) / "plans.json").read_text())
    config = plans.get("configurations", {}).get("3d_fullres", {})
    required = (
        dataset.get("channel_names") == {"0": "t1c", "1": "t2f"}
        and dataset.get("regions_class_order") == [2, 1, 3]
        and dataset.get("labels", {}).get("whole tumor") == [1, 2, 3]
        and plans.get("image_reader_writer") == "SimpleITKIO"
        and plans.get("transpose_forward") == [0, 1, 2]
        and config.get("spacing") == [1.0, 1.0, 1.0]
        and config.get("patch_size") == [128, 128, 128]
        and config.get("normalization_schemes") == ["ZScoreNormalization"] * 2
        and config.get("use_mask_for_norm") == [True, True]
        and config.get("preprocessor_name") == "DefaultPreprocessor"
    )
    if not required:
        raise ModelContractFailure("nnU-Net channel, label, reader or preprocessing contract changed")
    return {"dataset": dataset, "plans": plans,
            "checkpoint_sha256": receipt["files"]["fold_all/checkpoint_final.pth"]["sha256"]}


def manual_predictor_from_preverified_network(network, checked_metadata: dict,
                                              *, checkpoint_sha256: str,
                                              strict_state_verified: bool, device):
    """Attach licensed nnU-Net inference to an *already* safely verified network.

    The separate resource-supervised loader must use ``weights_only=True`` with
    scoped safe globals, compare every checkpoint key/shape/value, and restore
    the allowlist. This function intentionally cannot read a checkpoint or call
    nnU-Net's default folder initializer, which uses unsafe unrestricted load.
    It performs no forward pass and has not been patient-tested.
    """
    if (not strict_state_verified or checkpoint_sha256 != checked_metadata.get("checkpoint_sha256")
            or os.getenv("nnUNet_compile", "").lower() in {"1", "true", "t"}):
        raise ModelContractFailure("verified strict state, exact checkpoint and no compile are required")
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    if not isinstance(device, torch.device) or device.type not in {"cpu", "mps"}:
        raise ModelContractFailure("explicit CPU or MPS torch device is required")
    manager = PlansManager(checked_metadata["plans"])
    config = manager.get_configuration("3d_fullres")
    if manager.get_label_manager(checked_metadata["dataset"]).num_segmentation_heads != 3:
        raise ModelContractFailure("expected three nnU-Net region heads")
    network.eval()
    predictor = nnUNetPredictor(device=device, use_mirroring=False,
                                perform_everything_on_device=False,
                                verbose=False, allow_tqdm=False)
    # state_dict tensors share the loaded network's storage; no second full
    # checkpoint copy is retained solely for nnU-Net's fold iteration.
    predictor.manual_initialization(network, manager, config, [network.state_dict()],
                                    checked_metadata["dataset"], "nnUNetTrainer", None)
    return predictor


def _require_hash(path: Path, expected: str) -> str:
    if not Path(path).is_file() or len(expected) != 64:
        raise SourceMismatch(f"missing source or SHA-256 receipt: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise SourceMismatch(f"source SHA-256 mismatch: {path}")
    return actual


def limited_mask_from_existing_synthstrip_record(t1c: Scan, mask_path: Path,
                                                  record_path: Path,
                                                  *, expected_record_sha256: str,
                                                  limitations: str) -> QualifiedMask:
    """Bind an existing scan-only T1 SynthStrip output to its saved run record.

    This verifies provenance for a *diagnostic* mask, not anatomical adequacy.
    It reads only the record and hashes files; no patient voxel array is opened.
    """
    if t1c.modality != "t1c" or not limitations.strip():
        raise MaskNotQualified("exact T1c role and documented limitations are required")
    _require_hash(t1c.path, t1c.expected_sha256)
    _require_hash(record_path, expected_record_sha256)
    try:
        record = json.loads(Path(record_path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MaskNotQualified("missing or invalid SynthStrip generation record") from error
    model = record.get("model", {})
    hashes = record.get("artifact_hashes", {})
    expected_mask = hashes.get("main_mask.nii.gz")
    if (record.get("input_sha256") != t1c.expected_sha256
            or model.get("name") != "SynthStrip"
            or record.get("provenance") != "estimated"
            or record.get("exit_code") != 0
            or record.get("failure") not in (None, "")
            or record.get("brain_reviewed") is not False
            or record.get("cortical_access_permitted") is not False
            or not isinstance(expected_mask, str)
            or Path(mask_path).name != "main_mask.nii.gz"):
        raise MaskNotQualified("generation record does not bind a limited scan-derived T1 mask")
    _require_hash(mask_path, expected_mask)
    return QualifiedMask(Path(mask_path), expected_mask, t1c.expected_sha256,
                         "t1c", "documented_limited_scan_derived", limitations)


def _sitk_affine_ras(path: Path) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3:
        raise FrameMismatch("SimpleITK did not read exactly three spatial axes")
    linear_lps = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3) @ np.diag(image.GetSpacing())
    result = np.eye(4)
    result[:3, :3] = np.diag([-1.0, -1.0, 1.0]) @ linear_lps
    result[:3, 3] = np.diag([-1.0, -1.0, 1.0]) @ np.asarray(image.GetOrigin())
    return result


def _ants_affine_ras(path: Path) -> np.ndarray:
    image = ants.image_read(str(path))
    if image.dimension != 3:
        raise FrameMismatch("ANTs did not read exactly three spatial axes")
    result = np.eye(4)
    lps_to_ras = np.diag([-1.0, -1.0, 1.0])
    result[:3, :3] = lps_to_ras @ np.asarray(image.direction, dtype=float) @ np.diag(image.spacing)
    result[:3, 3] = lps_to_ras @ np.asarray(image.origin, dtype=float)
    return result


def read_volume(path: Path, expected_sha256: str, *, atlas: bool = False,
                working_dir: Path | None = None) -> PreparedVolume:
    """Verify SHA, dimensionality, finite voxels and nibabel/SITK RAS frame.

    Only a singleton fourth axis is removable from the pinned atlas; the
    original affine is copied unchanged. Patient scans must already be 3-D.
    """
    path = Path(path)
    source_sha = _require_hash(path, expected_sha256)
    image = nib.load(str(path))
    original_affine = np.asarray(image.affine, dtype=float).copy()
    data = np.asarray(image.dataobj)
    if data.ndim == 4 and atlas and data.shape[3] == 1:
        if working_dir is None:
            raise AtlasShapeFailure("atlas singleton axis needs a private working directory")
        working_dir.mkdir(parents=True, exist_ok=True)
        squeezed = working_dir / "atlas-squeezed-3d.nii.gz"
        nib.save(nib.Nifti1Image(data[..., 0], image.affine, header=image.header), str(squeezed))
        path = squeezed
        image = nib.load(str(path))
        data = np.asarray(image.dataobj)
    if data.ndim != 3:
        error = AtlasShapeFailure if atlas else FrameMismatch
        raise error(f"expected exactly 3-D image; got {tuple(data.shape)}")
    if not np.isfinite(data).all():
        raise FrameMismatch("scan contains nonfinite intensities")
    affine = np.asarray(image.affine, dtype=float)
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-8:
        raise FrameMismatch("invalid physical affine")
    if image.header.get_xyzt_units()[0] != "mm":
        raise FrameMismatch("spatial units must be explicit millimeters")
    sitk_affine = _sitk_affine_ras(path)
    if not np.allclose(affine, sitk_affine, atol=1e-4, rtol=0):
        raise FrameMismatch("nibabel RAS and SimpleITK LPS physical frames disagree")
    ants_affine = _ants_affine_ras(path)
    if not np.allclose(affine, ants_affine, atol=1e-4, rtol=0):
        raise FrameMismatch("nibabel RAS and ANTs LPS physical frames disagree")
    if atlas and source_sha != sha256_file(path) and not np.array_equal(affine, original_affine):
        raise FrameMismatch("atlas squeeze changed spatial affine")
    return PreparedVolume(path, tuple(int(x) for x in data.shape), affine, sha256_file(path))


def require_support_map_frame(map_path: Path, map_sha256: str,
                              reference_path: Path, reference_sha256: str) -> PreparedVolume:
    """Require exact saved bytes and one unambiguous physical grid before use.

    A missing qform is acceptable when the sform is coded. This checks frame
    integrity only; support-bit validity, anatomical QC, and planner admission
    remain separate decisions.
    """
    volumes = []
    for name, path, expected_hash in (
        ("support map", Path(map_path), map_sha256),
        ("reference", Path(reference_path), reference_sha256),
    ):
        _require_hash(path, expected_hash)
        image = nib.load(str(path))
        sform, sform_code = image.header.get_sform(coded=True)
        if sform_code == 0 or sform is None:
            raise FrameMismatch(f"{name} requires a coded sform")
        qform, qform_code = image.header.get_qform(coded=True)
        if qform_code != 0 and not np.allclose(qform, sform, atol=1e-4, rtol=0):
            raise FrameMismatch(f"{name} has conflicting coded sform and qform")
        volume = read_volume(path, expected_hash)
        if volume.sha256 != expected_hash:
            raise SourceMismatch(f"{name} changed during physical-frame verification")
        if not np.allclose(volume.affine_ras_mm, sform, atol=1e-4, rtol=0):
            raise FrameMismatch(f"{name} coded sform differs from physical reader frame")
        volumes.append(volume)
    support, reference = volumes
    if (support.shape_xyz != reference.shape_xyz
            or not np.allclose(support.affine_ras_mm, reference.affine_ras_mm,
                               atol=1e-4, rtol=0)):
        raise FrameMismatch("support map and reference have different physical grids")
    return support


def validate_scan_inputs(t1c: Scan | None, flair: Scan | None,
                         mask: QualifiedMask | None,
                         *, diagnostic_only: bool = False) -> None:
    if t1c is None or flair is None or t1c.modality != "t1c" or flair.modality != "t2f":
        raise MissingModality("require original T1c channel 0 and FLAIR/T2f channel 1")
    if t1c.availability_basis != "preoperative_source" or flair.availability_basis != "preoperative_source":
        raise SourceMismatch("both scans need a verified preoperative availability basis")
    if mask is None:
        raise MaskNotQualified("a scan-derived mask is required")
    qualified = mask.qc_status == "qualified_scan_derived"
    limited = (mask.qc_status == "documented_limited_scan_derived"
               and bool(mask.limitations.strip()) and diagnostic_only)
    if not qualified and not limited:
        raise MaskNotQualified("mask scope is not approved for the declared use")
    expected_source = t1c.expected_sha256 if mask.frame == "t1c" else flair.expected_sha256 if mask.frame == "flair" else None
    if mask.source_sha256 != expected_source:
        raise MaskNotQualified("mask source/frame lineage does not match the permitted scans")


def assert_same_frame(first: PreparedVolume, second: PreparedVolume) -> None:
    if first.shape_xyz != second.shape_xyz or not np.allclose(first.affine_ras_mm, second.affine_ras_mm,
                                                              atol=1e-4, rtol=0):
        raise FrameMismatch("equal array shape is insufficient; physical frames must match")


def register_rigid_scan_only(fixed: PreparedVolume, moving: PreparedVolume,
                             *, moving_frame: str, fixed_frame: str,
                             working_dir: Path, seed: int = 17) -> ImageTransform:
    """Optional ANTs rigid registration of two scans, never of annotations.

    The caller must freeze these parameters and separately review local QC.
    The Case4 DEVELOPMENT preparation has exercised this path; numerical
    consistency does not establish anatomical accuracy.
    """
    working_dir.mkdir(parents=True, exist_ok=True)
    result = ants.registration(
        fixed=ants.image_read(str(fixed.path)), moving=ants.image_read(str(moving.path)),
        type_of_transform="Rigid", aff_metric="mattes", aff_sampling=32,
        random_seed=seed, outprefix=str(working_dir / "rigid-"),
    )
    forward = result.get("fwdtransforms", [])
    if len(forward) != 1 or Path(forward[0]).suffix != ".mat":
        raise RegistrationFailure("rigid registration did not produce exactly one affine")
    target = working_dir / f"{moving_frame}-image-to-{fixed_frame}.mat"
    shutil.copy2(forward[0], target)
    transform_sha = sha256_file(target)
    receipt = {
        "schema_version": 1,
        "provenance": "estimated_from_scans_only",
        "algorithm": "ANTsPyx rigid image registration",
        "ants_version": ants.__version__,
        "type_of_transform": "Rigid",
        "aff_metric": "mattes",
        "aff_sampling": 32,
        "random_seed": seed,
        "moving_frame": moving_frame,
        "fixed_frame": fixed_frame,
        "moving_scan_sha256": moving.sha256,
        "fixed_scan_sha256": fixed.sha256,
        "forward_image_transform_sha256": transform_sha,
        "inverse_image_uses_whichtoinvert": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "qc_status": "estimated_unreviewed",
    }
    receipt_path = working_dir / f"{moving_frame}-to-{fixed_frame}-registration.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return ImageTransform(target, transform_sha, moving_frame, fixed_frame,
                          receipt_path, sha256_file(receipt_path))


def verify_scan_registration_receipt(step: ImageTransform,
                                     *, moving: PreparedVolume,
                                     fixed: PreparedVolume) -> dict:
    """Verify exact inputs and output of a registration generated by this API."""
    if step.generation_receipt_path is None or step.generation_receipt_sha256 is None:
        raise RegistrationFailure("scan-only generation receipt is required")
    _require_hash(step.generation_receipt_path, step.generation_receipt_sha256)
    _require_hash(step.path, step.sha256)
    receipt = json.loads(step.generation_receipt_path.read_text())
    expected = {
        "provenance": "estimated_from_scans_only",
        "algorithm": "ANTsPyx rigid image registration",
        "type_of_transform": "Rigid",
        "moving_frame": step.moving_frame,
        "fixed_frame": step.fixed_frame,
        "moving_scan_sha256": moving.sha256,
        "fixed_scan_sha256": fixed.sha256,
        "forward_image_transform_sha256": step.sha256,
        "inverse_image_uses_whichtoinvert": True,
        "qc_status": "estimated_unreviewed",
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RegistrationFailure("registration receipt disagrees with exact scan/transform bytes")
    return receipt


def resample_image(moving: PreparedVolume, fixed: PreparedVolume, transform: ImageTransform,
                   *, source_frame: str, target_frame: str,
                   inverse: bool = False, label: bool = False) -> ants.ANTsImage:
    """ANTs pull resampling with explicit image-space inverse flag."""
    expected = ((transform.fixed_frame, transform.moving_frame) if inverse else
                (transform.moving_frame, transform.fixed_frame))
    if (source_frame, target_frame) != expected:
        raise RegistrationFailure("resampling frame order contradicts transform direction")
    _require_hash(transform.path, transform.sha256)
    warped = ants.apply_transforms(
        fixed=ants.image_read(str(fixed.path)), moving=ants.image_read(str(moving.path)),
        transformlist=[str(transform.path)], whichtoinvert=[inverse],
        interpolator="nearestNeighbor" if label else "linear", defaultvalue=0,
    )
    if tuple(warped.shape) != fixed.shape_xyz:
        raise FrameMismatch("ANTs output shape differs from fixed reference")
    return warped


def assert_mask_coverage(mask: np.ndarray, *coverage: np.ndarray) -> None:
    if mask.ndim != 3 or not np.any(mask):
        raise MaskNotQualified("qualified scan-derived mask is empty or not 3-D")
    if any(c.shape != mask.shape or not np.all(c[mask]) for c in coverage):
        raise CoverageFailure("one source modality does not cover all qualified mask voxels")


def _save_checked(image: ants.ANTsImage, path: Path,
                  reference: PreparedVolume) -> PreparedVolume:
    ants.image_write(image, str(path))
    result = read_volume(path, sha256_file(path))
    assert_same_frame(result, reference)
    return result


def _prepare_pair_with_fixed_affines(t1c: Scan, flair: Scan, mask: QualifiedMask,
                                     atlas_path: Path, atlas_sha256: str,
                                     flair_to_t1c: ImageTransform,
                                     t1c_to_atlas: ImageTransform,
                                     private_dir: Path, *,
                                     diagnostic_only: bool = False,
                                     require_registration_receipts: bool = True) -> PreparedPair:
    """Internal geometry helper; unverified transforms only in generated tests.

    Supplied transforms must be scan-only products; this function never opens
    an annotation or chooses a planning target. Local registration/skull-strip
    anatomy QC remains a separate admission gate after geometric checks.
    """
    validate_scan_inputs(t1c, flair, mask, diagnostic_only=diagnostic_only)
    if ((flair_to_t1c.moving_frame, flair_to_t1c.fixed_frame) != ("flair", "t1c") or
            (t1c_to_atlas.moving_frame, t1c_to_atlas.fixed_frame) != ("t1c", "atlas")):
        raise RegistrationFailure("required scan-only transform chain is flair -> t1c -> atlas")
    private_dir = Path(private_dir)
    private_dir.mkdir(parents=True, exist_ok=True)
    t1c_native = read_volume(t1c.path, t1c.expected_sha256)
    flair_native = read_volume(flair.path, flair.expected_sha256)
    atlas = read_volume(atlas_path, atlas_sha256, atlas=True, working_dir=private_dir)
    if require_registration_receipts:
        verify_scan_registration_receipt(flair_to_t1c, moving=flair_native, fixed=t1c_native)
        verify_scan_registration_receipt(t1c_to_atlas, moving=t1c_native, fixed=atlas)
    source_for_mask = t1c_native if mask.frame == "t1c" else flair_native
    native_mask = read_volume(mask.path, mask.expected_sha256)
    assert_same_frame(source_for_mask, native_mask)
    mask_data = np.asarray(nib.load(str(native_mask.path)).dataobj)
    if not np.isin(mask_data, [0, 1]).all() or not mask_data.any():
        raise MaskNotQualified("qualified mask must be nonempty and binary")

    flair_t1c_image = resample_image(flair_native, t1c_native, flair_to_t1c,
                                     source_frame="flair", target_frame="t1c")
    flair_t1c = _save_checked(flair_t1c_image, private_dir / "flair-in-t1c.nii.gz", t1c_native)
    if mask.frame == "flair":
        t1c_mask_image = resample_image(native_mask, t1c_native, flair_to_t1c,
                                        source_frame="flair", target_frame="t1c", label=True)
        t1c_mask = _save_checked(t1c_mask_image, private_dir / "mask-in-t1c.nii.gz", t1c_native)
    else:
        t1c_mask = native_mask
    mask_atlas_image = resample_image(t1c_mask, atlas, t1c_to_atlas,
                                      source_frame="t1c", target_frame="atlas", label=True)
    atlas_mask = _save_checked(mask_atlas_image, private_dir / "qualified-mask-atlas.nii.gz", atlas)
    atlas_mask_bool = mask_atlas_image.numpy() > 0.5

    # Coverage is transformed from explicit source support, not inferred from
    # an intensity zero (which might be a genuine scan value).
    coverage_paths = []
    for label, source in (("t1c", t1c_native), ("flair", flair_native)):
        ones_path = private_dir / f"{label}-native-coverage.nii.gz"
        ones = nib.Nifti1Image(np.ones(source.shape_xyz, dtype=np.uint8), source.affine_ras_mm)
        ones.header.set_xyzt_units("mm")
        nib.save(ones, str(ones_path))
        coverage_paths.append(read_volume(ones_path, sha256_file(ones_path)))
    flair_coverage_t1c = resample_image(coverage_paths[1], t1c_native, flair_to_t1c,
                                        source_frame="flair", target_frame="t1c", label=True)
    flair_coverage_t1c_path = private_dir / "flair-coverage-in-t1c.nii.gz"
    flair_coverage_t1c_vol = _save_checked(flair_coverage_t1c, flair_coverage_t1c_path, t1c_native)
    t1c_coverage_atlas = resample_image(coverage_paths[0], atlas, t1c_to_atlas,
                                        source_frame="t1c", target_frame="atlas", label=True).numpy() > 0.5
    flair_coverage_atlas = resample_image(flair_coverage_t1c_vol, atlas, t1c_to_atlas,
                                          source_frame="t1c", target_frame="atlas", label=True).numpy() > 0.5
    assert_mask_coverage(atlas_mask_bool, t1c_coverage_atlas, flair_coverage_atlas)

    t1c_atlas_image = resample_image(t1c_native, atlas, t1c_to_atlas,
                                     source_frame="t1c", target_frame="atlas")
    flair_atlas_image = resample_image(flair_t1c, atlas, t1c_to_atlas,
                                       source_frame="t1c", target_frame="atlas")
    t1c_atlas = _save_checked(t1c_atlas_image * mask_atlas_image,
                              private_dir / "input_0000.nii.gz", atlas)
    flair_atlas = _save_checked(flair_atlas_image * mask_atlas_image,
                                private_dir / "input_0001.nii.gz", atlas)
    read_ordered_nnunet_channels(t1c_atlas, flair_atlas)
    status = ("diagnostic_forward_only_mask_limitations_retained" if diagnostic_only else
              "geometry_only_requires_local_anatomy_review")
    return PreparedPair(t1c_atlas, flair_atlas, atlas, atlas_mask,
                        t1c_native, flair_native, flair_to_t1c, t1c_to_atlas,
                        float(np.mean(t1c_coverage_atlas & flair_coverage_atlas)), status)


def prepare_registered_diagnostic_pair(t1c: Scan, flair: Scan, *,
                                       synthstrip_mask_path: Path,
                                       synthstrip_record_path: Path,
                                       synthstrip_record_sha256: str,
                                       mask_limitations: str,
                                       atlas_path: Path, atlas_sha256: str,
                                       private_dir: Path, seed: int = 17) -> PreparedPair:
    """Source-only T1/FLAIR/SRI24 registration with internally generated receipts.

    This is a limited diagnostic preparation path. The record is tied to T1c
    bytes; both transforms are calculated from the two scans and atlas here.
    No caller-provided transform or annotation path is accepted. The return
    remains unreviewed and is not a direct planner admission.
    """
    mask = limited_mask_from_existing_synthstrip_record(
        t1c, synthstrip_mask_path, synthstrip_record_path,
        expected_record_sha256=synthstrip_record_sha256,
        limitations=mask_limitations)
    validate_scan_inputs(t1c, flair, mask, diagnostic_only=True)
    private_dir = Path(private_dir)
    private_dir.mkdir(parents=True, exist_ok=True)
    t1c_native = read_volume(t1c.path, t1c.expected_sha256)
    flair_native = read_volume(flair.path, flair.expected_sha256)
    atlas = read_volume(atlas_path, atlas_sha256, atlas=True, working_dir=private_dir)
    flair_to_t1c = register_rigid_scan_only(
        t1c_native, flair_native, moving_frame="flair", fixed_frame="t1c",
        working_dir=private_dir / "flair-to-t1c", seed=seed)
    t1c_to_atlas = register_rigid_scan_only(
        atlas, t1c_native, moving_frame="t1c", fixed_frame="atlas",
        working_dir=private_dir / "t1c-to-atlas", seed=seed)
    return _prepare_pair_with_fixed_affines(
        t1c, flair, mask, atlas_path, atlas_sha256, flair_to_t1c,
        t1c_to_atlas, private_dir / "prepared", diagnostic_only=True,
        require_registration_receipts=True)


def map_t1_support_to_native_flair(prepared: PreparedPair, t1_mask_path: Path,
                                   t1_mask_sha256: str, *, source_t1c_sha256: str,
                                   output_path: Path) -> PreparedVolume:
    """Resample known T1-native support into FLAIR frame before same-grid use.

    This is separate from a model's native-FLAIR WT estimate. A caller still
    needs bounded-ROI, lineage and coverage review before a research planner.
    """
    if source_t1c_sha256 != prepared.t1c_native.sha256:
        raise SourceMismatch("T1 support lineage does not match the prepared T1c scan")
    t1_mask = read_volume(t1_mask_path, t1_mask_sha256)
    assert_same_frame(t1_mask, prepared.t1c_native)
    data = np.asarray(nib.load(str(t1_mask.path)).dataobj)
    if not np.isin(data, [0, 1]).all() or not np.any(data):
        raise MaskNotQualified("T1-native support must be nonempty and binary")
    warped = resample_image(t1_mask, prepared.flair_native, prepared.flair_to_t1c,
                            source_frame="t1c", target_frame="flair", inverse=True, label=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = _save_checked(warped, output_path, prepared.flair_native)
    returned = np.asarray(nib.load(str(result.path)).dataobj)
    if not np.isin(returned, [0, 1]).all() or not np.any(returned):
        raise CoverageFailure("T1 support does not survive native-FLAIR mapping")
    return result


def map_record_bound_t1_support_to_flair(prepared: PreparedPair, t1c: Scan, *,
                                         synthstrip_mask_path: Path,
                                         synthstrip_record_path: Path,
                                         synthstrip_record_sha256: str,
                                         limitations: str,
                                         output_path: Path) -> PreparedVolume:
    """Narrow Case4-style T1 support mapping with existing-record binding."""
    verified = limited_mask_from_existing_synthstrip_record(
        t1c, synthstrip_mask_path, synthstrip_record_path,
        expected_record_sha256=synthstrip_record_sha256,
        limitations=limitations)
    return map_t1_support_to_native_flair(
        prepared, verified.path, verified.expected_sha256,
        source_t1c_sha256=verified.source_sha256, output_path=output_path)


def read_ordered_nnunet_channels(t1c_atlas: PreparedVolume,
                                 flair_atlas: PreparedVolume) -> tuple[np.ndarray, dict]:
    """Use licensed nnU-Net 2.5.2 IO; output shape is [T1c, FLAIR, Z, Y, X]."""
    assert_same_frame(t1c_atlas, flair_atlas)
    data, props = SimpleITKIO().read_images([str(t1c_atlas.path), str(flair_atlas.path)])
    if data.shape != (2, *reversed(t1c_atlas.shape_xyz)):
        raise FrameMismatch("nnU-Net channel/axis order differs from declared [C,Z,Y,X]")
    expected_spacing_zyx = np.linalg.norm(t1c_atlas.affine_ras_mm[:3, :3], axis=0)[::-1]
    if not np.allclose(props["spacing"], expected_spacing_zyx, atol=1e-4, rtol=0):
        raise FrameMismatch("nnU-Net ZYX spacing differs from physical XYZ spacing")
    return data, props


def whole_tumor_from_labels(labels: np.ndarray) -> np.ndarray:
    """Convert nnU-Net exported labels only; three logits are region heads."""
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise FrameMismatch("expected an exported integer 3-D label map")
    if not np.isin(labels, [0, 1, 2, 3]).all():
        raise FrameMismatch("unexpected exported label value")
    return np.isin(labels, [1, 2, 3]).astype(np.uint8)
