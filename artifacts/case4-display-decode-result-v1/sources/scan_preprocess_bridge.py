"""Scan-only nnU-Net preparation and display decoding, with no network runner.

Every output is an unreviewed diagnostic candidate, never a planner or
evaluation admission. Patient use requires separately verified input and
model-output receipts.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from resectionlab.scan_support_contract import diagnostic_display_state, unreviewed_input_domain
from resectionlab.scan_target_adapter import (
    CoverageFailure, FrameMismatch, ModelContractFailure, SourceMismatch,
    assert_same_frame, read_ordered_nnunet_channels, read_volume,
    require_support_map_frame, sha256_file, whole_tumor_from_labels,
)

PATCH_ZYX = (128, 128, 128)
PINNED_NNUNET_VERSION = "2.5.2"
PINNED_PLANS_SHA256 = "e85abbe6a41f4e5e0164d53dd3a297baf10e06cd5f05c1d46a22d2d869d64ad5"
PINNED_DATASET_SHA256 = "3a7c7c1fd5eb420243a25496bae2c2fb9f210d7910363b9361aaebc69e615a3f"
PINNED_RESAMPLING = {
    "resampling_fn_data": "resample_data_or_seg_to_shape",
    "resampling_fn_data_kwargs": {"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": None},
    "resampling_fn_seg": "resample_data_or_seg_to_shape",
    "resampling_fn_seg_kwargs": {"is_seg": True, "order": 1, "order_z": 0, "force_separate_z": None},
    "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
    "resampling_fn_probabilities_kwargs": {"is_seg": False, "order": 1, "order_z": 0,
                                               "force_separate_z": None},
}


def _exact_json(path: Path, expected_sha256: str) -> dict:
    path = Path(path)
    if (not path.is_file() or path.is_symlink() or len(expected_sha256) != 64 or
            sha256_file(path) != expected_sha256):
        raise SourceMismatch("exact plans/dataset metadata changed")
    result = json.loads(path.read_text())
    if sha256_file(path) != expected_sha256:
        raise SourceMismatch("metadata changed during read")
    return result


def _checked_plan(plans_path, plans_sha256, dataset_path, dataset_sha256):
    try:
        installed_version = version("nnunetv2")
    except PackageNotFoundError as exc:
        raise ModelContractFailure("pinned nnU-Net runtime is absent") from exc
    if installed_version != PINNED_NNUNET_VERSION:
        raise ModelContractFailure("nnU-Net runtime version differs from pinned 2.5.2")
    if plans_sha256 != PINNED_PLANS_SHA256 or dataset_sha256 != PINNED_DATASET_SHA256:
        raise ModelContractFailure("metadata does not match independently pinned trained-model bytes")
    plans = _exact_json(plans_path, plans_sha256)
    dataset = _exact_json(dataset_path, dataset_sha256)
    config = plans.get("configurations", {}).get("3d_fullres", {})
    if not (dataset.get("channel_names") == {"0": "t1c", "1": "t2f"}
            and dataset.get("regions_class_order") == [2, 1, 3]
            and dataset.get("labels", {}).get("whole tumor") == [1, 2, 3]
            and plans.get("image_reader_writer") == "SimpleITKIO"
            and plans.get("transpose_forward") == [0, 1, 2]
            and plans.get("transpose_backward") == [0, 1, 2]
            and config.get("spacing") == [1.0, 1.0, 1.0]
            and config.get("patch_size") == list(PATCH_ZYX)
            and config.get("normalization_schemes") == ["ZScoreNormalization"] * 2
            and config.get("use_mask_for_norm") == [True, True]
            and config.get("preprocessor_name") == "DefaultPreprocessor"
            and all(config.get(key) == value for key, value in PINNED_RESAMPLING.items())):
        raise ModelContractFailure("pinned t1c+t2f preprocessing plan changed")
    manager = PlansManager(plans)
    return manager, manager.get_configuration("3d_fullres"), dataset


def _binary_patch(value, name):
    array = np.asarray(value)
    if (array.shape != PATCH_ZYX or array.dtype.kind not in "biuf"
            or not np.isfinite(array).all() or not np.isin(array, (0, 1)).all()):
        raise CoverageFailure(f"{name} must be a finite binary 128³ ZYX patch")
    return array.astype(np.uint8, copy=False)


@dataclass(frozen=True)
class PatchGeometry:
    """Exact crop/resample/pad map; all indices are ZYX until final XYZ transpose."""

    atlas_shape_xyz: tuple[int, int, int]
    atlas_affine_ras_mm: tuple[tuple[float, ...], ...]
    shape_before_cropping_zyx: tuple[int, int, int]
    crop_bbox_zyx: tuple[tuple[int, int], ...]
    shape_after_cropping_zyx: tuple[int, int, int]
    preprocessed_shape_zyx: tuple[int, int, int]
    original_spacing_zyx_mm: tuple[float, float, float]
    target_spacing_zyx_mm: tuple[float, float, float]
    patch_start_zyx: tuple[int, int, int]
    plans_sha256: str
    dataset_sha256: str
    t1c_sha256: str
    flair_sha256: str
    support_map_sha256: str

    def _insert_patch(self, patch: np.ndarray) -> np.ndarray:
        """Place CZYX patch content on its preprocessed grid, dropping padding."""
        data = np.zeros((patch.shape[0], *self.preprocessed_shape_zyx), patch.dtype)
        source_slices = []
        target_slices = []
        for start, size in zip(self.patch_start_zyx, self.preprocessed_shape_zyx):
            lo, hi = max(0, start), min(size, start + 128)
            if hi <= lo:
                raise CoverageFailure("patch has no preprocessed-grid intersection")
            target_slices.append(slice(lo, hi))
            source_slices.append(slice(lo - start, hi - start))
        data[(slice(None), *target_slices)] = patch[(slice(None), *source_slices)]
        return data

    def _uncrop_xyz(self, cropped: np.ndarray) -> np.ndarray:
        if cropped.shape != self.shape_after_cropping_zyx:
            raise CoverageFailure("inverse probability spacing did not restore cropped shape")
        original = np.zeros(self.shape_before_cropping_zyx, cropped.dtype)
        bbox_slices = tuple(slice(lo, hi) for lo, hi in self.crop_bbox_zyx)
        if original[bbox_slices].shape != cropped.shape:
            raise CoverageFailure("inverse crop bounds changed")
        original[bbox_slices] = cropped
        atlas_xyz = original.transpose(2, 1, 0)
        if atlas_xyz.shape != self.atlas_shape_xyz:
            raise CoverageFailure("inverse ZYX→XYZ shape changed")
        return atlas_xyz

    def map_patch_coverage_to_atlas(self, patch_binary, configuration) -> np.ndarray:
        """Conservative geometry/output support only; this never decodes labels.

        A voxel is covered only if the nnU-Net probability-resampling stencil
        lies entirely inside the supplied output coverage. This uses the same
        pinned resampling function as logits, not segmentation interpolation.
        """
        patch = _binary_patch(patch_binary, "patch output coverage")
        full = self._insert_patch(patch[None].astype(np.float32, copy=False))
        # Resampling both indicators avoids calling a value rounded to 1.0
        # "fully supported" when a small missing-data interpolation weight was
        # lost to float32 rounding. A false unknown is preferable to a false 0.
        indicators = np.concatenate((full, 1.0 - full), axis=0)
        resampled = configuration.resampling_fn_probabilities(
            indicators, self.shape_after_cropping_zyx,
            self.target_spacing_zyx_mm, self.original_spacing_zyx_mm)
        if isinstance(resampled, torch.Tensor):
            resampled = resampled.cpu().numpy()
        if (resampled.shape != (2, *self.shape_after_cropping_zyx)
                or not np.isfinite(resampled).all()):
            raise CoverageFailure("inverse probability-coverage resampling is invalid")
        supported = (resampled[0] >= 1.0) & (resampled[1] <= 0.0)
        return self._uncrop_xyz(supported).astype(bool)

    def map_patch_logits_to_atlas(self, logits_czyx, configuration, label_manager) -> np.ndarray:
        """Decode supplied logits using the pinned nnU-Net export order.

        Output outside separate coverage is meaningless and must not be shown.
        Neutral zero-filled unobserved voxels only serve as a resampling carrier.
        """
        logits = np.asarray(logits_czyx)
        if (logits.shape != (3, *PATCH_ZYX) or logits.dtype.kind != "f"
                or not np.isfinite(logits).all()):
            raise ModelContractFailure("supplied logits must be finite float [3,128³] CZYX")
        full = self._insert_patch(logits.astype(np.float32, copy=False))
        resampled = configuration.resampling_fn_probabilities(
            full, self.shape_after_cropping_zyx,
            self.target_spacing_zyx_mm, self.original_spacing_zyx_mm)
        if isinstance(resampled, torch.Tensor):
            resampled = resampled.cpu().numpy()
        if (resampled.shape != (3, *self.shape_after_cropping_zyx)
                or not np.isfinite(resampled).all()):
            raise ModelContractFailure("inverse logit resampling is invalid")
        probabilities = label_manager.apply_inference_nonlin(resampled)
        labels = label_manager.convert_probabilities_to_segmentation(probabilities)
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        return self._uncrop_xyz(np.asarray(labels, dtype=np.uint8))

    def display_only_metadata(self) -> dict:
        """JSON-safe typed display contract; no prediction exists at this stage."""
        return {
            "schema": "unreviewed_scan_candidate_patch_v2",
            "display_kind": "candidate_segmentation_unreviewed",
            "coordinate_frame": "atlas_RAS_mm_coded_sform",
            "shape_xyz": list(self.atlas_shape_xyz),
            "affine_ras_mm": [list(row) for row in self.atlas_affine_ras_mm],
            "channel_order": ["t1c", "t2f"],
            "patch_shape_zyx": list(PATCH_ZYX),
            "patch_start_preprocessed_zyx": list(self.patch_start_zyx),
            "source_sha256": {"t1c": self.t1c_sha256, "flair": self.flair_sha256,
                              "support_map": self.support_map_sha256,
                              "plans": self.plans_sha256, "dataset": self.dataset_sha256},
            "prediction_coverage_status": "unavailable_until_verified_output_receipt",
            "decoding_semantics": "pinned_nnunet_probability_resample_then_regions",
            "uncertainties": ["estimated_mask_anatomy_unreviewed",
                              "registration_anatomy_unreviewed",
                              "model_training_overlap_unknown",
                              "fixed_patch_not_whole_volume"],
            "unknown_state": -1,
            "negative_state": 0,
            "positive_state": 1,
            "anatomical_qc": "unreviewed",
            "planning_eligible": False,
            "evaluation_eligible": False,
        }


@dataclass(frozen=True)
class DiagnosticDisplayOutput:
    """Unreviewed display states; never an estimate for planning or evaluation."""

    state_xyz: np.ndarray  # -1 unknown, 0 negative, 1 candidate positive
    prediction_coverage_xyz: np.ndarray  # conservative actual-output support ∩ bit 7
    positives_excluded_outside_coverage: int
    metadata: dict


@dataclass(frozen=True)
class PreparedDiagnosticPatch:
    tensor_czyx: np.ndarray  # [2,128,128,128] float32; not a saved/model output
    geometry: PatchGeometry
    support_bits_xyz: np.ndarray  # exact grid, unreviewed 0/1/2/3/7 codes
    configuration: object  # pinned nnU-Net ConfigurationManager
    label_manager: object  # pinned nnU-Net LabelManager, region order from dataset

    def candidate_domain_xyz(self) -> np.ndarray:
        return unreviewed_input_domain(self.support_bits_xyz)

    def prospective_domain_xyz(self) -> np.ndarray:
        """Model-call eligibility only; this is NOT observed output coverage."""
        mapped = self.geometry.map_patch_coverage_to_atlas(
            np.ones(PATCH_ZYX, np.uint8), self.configuration)
        return mapped & self.candidate_domain_xyz()

    def decode_supplied_logits_for_display(self, logits_czyx,
                                           output_coverage_zyx) -> DiagnosticDisplayOutput:
        """Map supplied patch logits, preserving unknown where output is absent.

        This is an algorithmic adapter, not a verified model-output receipt. A
        separate runner must authenticate prediction bytes before patient use.
        """
        actual = self.geometry.map_patch_coverage_to_atlas(
            output_coverage_zyx, self.configuration) & self.candidate_domain_xyz()
        labels = self.geometry.map_patch_logits_to_atlas(
            logits_czyx, self.configuration, self.label_manager)
        candidate = whole_tumor_from_labels(labels)
        state, excluded = diagnostic_display_state(candidate, self.support_bits_xyz, actual)
        metadata = self.geometry.display_only_metadata()
        metadata["prediction_coverage_status"] = "supplied_output_unverified"
        metadata["prediction_coverage_voxels"] = int(actual.sum())
        metadata["positives_excluded_outside_coverage"] = excluded
        return DiagnosticDisplayOutput(state, actual, excluded, metadata)


def prepare_scan_only_patch(*, t1c_path: Path, t1c_sha256: str,
                            flair_path: Path, flair_sha256: str,
                            support_map_path: Path, support_map_sha256: str,
                            plans_path: Path, plans_sha256: str,
                            dataset_path: Path, dataset_sha256: str) -> PreparedDiagnosticPatch:
    """Use pinned nnU-Net preprocessing on exact scan-only channels, no checkpoint.

    The fixed center patch is derived from the nonzero-cropped scans alone.
    It does not use a tumor annotation, private reference or model proposal.
    """
    manager, configuration, dataset = _checked_plan(
        plans_path, plans_sha256, dataset_path, dataset_sha256)
    t1c = read_volume(t1c_path, t1c_sha256)
    flair = read_volume(flair_path, flair_sha256)
    assert_same_frame(t1c, flair)
    support = require_support_map_frame(support_map_path, support_map_sha256,
                                        t1c_path, t1c_sha256)
    assert_same_frame(support, t1c)
    bits = np.asarray(nib.load(str(support_map_path)).dataobj)
    domain = unreviewed_input_domain(bits)
    if bits.shape != t1c.shape_xyz or not domain.any():
        raise CoverageFailure("support bitfield is empty or off the declared atlas grid")
    if sha256_file(Path(support_map_path)) != support_map_sha256:
        raise SourceMismatch("support bits changed after frame verification")
    raw_czyx, properties = read_ordered_nnunet_channels(t1c, flair)
    if sha256_file(Path(t1c_path)) != t1c_sha256 or sha256_file(Path(flair_path)) != flair_sha256:
        raise SourceMismatch("source channels changed during nnU-Net read")
    if not np.isfinite(raw_czyx).all():
        raise FrameMismatch("nonfinite ordered channel")
    nonzero_xyz = np.any(raw_czyx != 0, axis=0).transpose(2, 1, 0)
    if np.any(nonzero_xyz & ~domain) or any(not np.any(raw_czyx[c]) for c in (0, 1)):
        raise CoverageFailure("scan-only channels are not confined to nonempty bit-7 support")
    props = copy.deepcopy(properties)
    processed, synthetic_nonzero_seg = DefaultPreprocessor(verbose=False).run_case_npy(
        raw_czyx, None, props, manager, configuration, dataset)
    if (processed.ndim != 4 or processed.shape[0] != 2 or
            not np.isfinite(processed).all() or synthetic_nonzero_seg is None):
        raise ModelContractFailure("pinned preprocessor returned unexpected data")
    shape = tuple(int(x) for x in processed.shape[1:])
    starts = tuple((size - 128) // 2 for size in shape)
    patch = np.zeros((2, *PATCH_ZYX), np.float32)
    for_axis_src = []
    for_axis_dest = []
    for start, size in zip(starts, shape):
        lo, hi = max(0, start), min(size, start + 128)
        for_axis_src.append(slice(lo, hi))
        for_axis_dest.append(slice(lo - start, hi - start))
    patch[(slice(None), *for_axis_dest)] = processed[(slice(None), *for_axis_src)]
    bbox = tuple(tuple(int(value) for value in pair) for pair in props["bbox_used_for_cropping"])
    original_shape = tuple(int(x) for x in props["shape_before_cropping"])
    crop_shape = tuple(int(x) for x in props["shape_after_cropping_and_before_resampling"])
    if (len(bbox) != 3 or len(original_shape) != 3 or len(crop_shape) != 3 or
            any(hi <= lo or lo < 0 or hi > dim for (lo, hi), dim in zip(bbox, original_shape)) or
            tuple(hi - lo for lo, hi in bbox) != crop_shape or
            original_shape != tuple(reversed(t1c.shape_xyz))):
        raise ModelContractFailure("preprocessor crop/axis receipt is inconsistent")
    geometry = PatchGeometry(
        t1c.shape_xyz, tuple(tuple(float(v) for v in row) for row in t1c.affine_ras_mm),
        original_shape, bbox, crop_shape, shape,
        tuple(float(x) for x in properties["spacing"]),
        tuple(float(x) for x in configuration.spacing), starts,
        plans_sha256, dataset_sha256, t1c_sha256, flair_sha256, support_map_sha256)
    result = PreparedDiagnosticPatch(patch, geometry, bits.astype(np.uint8, copy=True),
                                     configuration, manager.get_label_manager(dataset))
    if not result.prospective_domain_xyz().any():
        raise CoverageFailure("fixed public patch does not intersect candidate diagnostic support")
    return result
