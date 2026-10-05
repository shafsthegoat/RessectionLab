#!/usr/bin/env python3
"""Six fixed TRAIN rows; four initial tasks, view preparation only, no learning.

Patient execution requires a separately frozen archive/release receipt. The
existing native task constructs its original inventory once; alternative views
never become task inputs and cannot change the physical candidate catalog.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_real_training_cases as prep
from preflight_real_spatial_policy import peak_rss_bytes, read_declaration, sha256, supervise_worker, write_json

VERSION = "fixed-training-observation-coverage-v2"
RELEASE_VERSION = "fixed-training-observation-coverage-release-v2"
SUBJECTS = ("sub-PAT05", "sub-PAT16", "sub-PAT20", "sub-PAT22", "sub-PAT25", "sub-PAT28")
BLOCKED = frozenset({"sub-PAT16", "sub-PAT20"})
SAVED = "artifacts/remaining-training-frozen-spatial-float64-v1"
MANIFEST = "manifests/experiments/training-observation-coverage-v2.json"
PAT05_GRID_RECEIPT = "artifacts/pat05-real-geometric-learning-v1/receipt.json"
PAT05_GRID_RECEIPT_SHA256 = "fdc575e6695a7f949f65de93e7165f7833b514fcd193094a24ac3e0f94e7ee77"
PAT05_GRID_POINTER = "/initial_task_metrics/native_grid_reconciliation"
PAT05_GRID_SHA256 = "a9b1a2f220e2a0137e16aa31b5e59146ea0c54d8d3b5c2fd7242764d2fb4005a"
PAT05_ORIGINAL_GRID_KEYS = frozenset({"derived_affine_ras_mm", "maximum_corner_displacement_mm",
    "method", "native_and_actor_physical_grid", "original_affine_ras_mm",
    "proposal_and_crop_indices_basis", "resampled", "shape"})
PAT05_COMPLETE_GRID_KEYS = PAT05_ORIGINAL_GRID_KEYS | frozenset({"corner_domain",
    "derived_affine_hash", "derived_spacing_mm", "handedness_preserved",
    "maximum_allowed_corner_displacement_mm", "maximum_allowed_gram_error", "origin_preserved",
    "original_affine_hash", "original_spacing_mm", "source_axis_gram_max_error",
    "source_image_hash", "support_hash"})
ANCHOR_SHA256 = {
    "sub-PAT05": "d69c9b6cf1f0f329d698c3857bca13efaa4a4a9a7b424208711affd4ac8480f4",
    "sub-PAT16": "3743dd12460cf598089664bd243fba883300c2505d477df0ec4aadd2bb596608",
    "sub-PAT20": "d5b72a815e4b2b6aeeedfedbc0687e74843c21f832cab184a4e523f011037312",
    "sub-PAT22": "2dbf6be558d5e7b370f43751480dad7b87bf98dfaae62d0c5f2cd0a5d7e6e1b6",
    "sub-PAT25": "f4c6269f551f52fbe359c61d408030af81d4b233ac4ad11f98a87228f81dc70c",
    "sub-PAT28": "b4fdb77d181808c494a81cbae0004a633be12bf59f84f139e0c83095339cb081",
}
SETTINGS = {"whole_worker_envelope_seconds": 180., "cooperative_seconds": 170.,
    "max_wall_seconds": 174., "max_rss_bytes": 6 * 1024**3, "cpu_threads": 1,
    "max_initial_previews_per_case": 78, "max_total_previews": 312,
    "newly_attempted_cases": 4, "retained_cohort_denominator": 6,
    "local_shape": [64, 64, 64], "coarse_shape": [64, 64, 64], "ray_samples": 5,
    "extent_roundtrip_atol_mm": 1e-8, "coarse_mass_rtol": 1e-6,
    "coarse_mass_small_volume_atol_mm3": 1e-6,
    "executed_transitions": 0, "policy_forwards": 0, "optimizer_updates": 0}


def _canonical(value):
    """Exact JSON values/types, insensitive only to mapping insertion order."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def pat05_complete_grid_binding(original):
    """Authenticate saved small metadata only, before any patient decode/preview.

    Keep the historical member's eight fields intact. This separate binding pins
    all twenty executed grid fields, including image/support lineage and bounds.
    The matrix-only affine hash is not the composite structural-frame hash.
    """
    with (ROOT / PAT05_GRID_RECEIPT).open("rb") as stream:
        raw = stream.read(2 * 1024**2 + 1)
    if len(raw) > 2 * 1024**2 or hashlib.sha256(raw).hexdigest() != PAT05_GRID_RECEIPT_SHA256:
        raise ValueError("Original complete PAT05 grid receipt bytes changed")
    receipt = json.loads(raw)
    try:
        grid = receipt
        for key in PAT05_GRID_POINTER.split("/")[1:]:
            grid = grid[key]
    except (KeyError, TypeError) as error:
        raise ValueError("Original complete PAT05 grid JSON pointer is absent") from error
    subset = original["member"]["expected_native_grid_binding"]
    if (not isinstance(grid, dict) or set(grid) != PAT05_COMPLETE_GRID_KEYS
            or not isinstance(subset, dict) or set(subset) != PAT05_ORIGINAL_GRID_KEYS
            or hashlib.sha256(_canonical(grid)).hexdigest() != PAT05_GRID_SHA256
            or _canonical({key: grid[key] for key in subset}) != _canonical(subset)):
        raise ValueError("Complete PAT05 grid or original eight-field projection changed")
    acknowledgement = original["member"]["research_support_acknowledgment"]
    if (grid["source_image_hash"] != acknowledgement["source_image_hash"]
            or grid["support_hash"] != acknowledgement["mask_hash"]):
        raise ValueError("Complete PAT05 grid source image or support differs from its acknowledgment")
    return {"schema": "pat05-complete-native-grid-binding-v1",
        "receipt": {"path": PAT05_GRID_RECEIPT, "sha256": PAT05_GRID_RECEIPT_SHA256,
                    "json_pointer": PAT05_GRID_POINTER},
        "grid_sha256": PAT05_GRID_SHA256, "complete_grid": grid}


class PAT05TaskBindingMismatch(ValueError):
    """Small expected/actual task metadata retained even when preparation fails."""

    def __init__(self, details):
        self.details = json.loads(_canonical(details))
        super().__init__("PAT05 physical task or complete native grid differs: "
                         + ", ".join(self.details["mismatched_fields"]))


def verify_pat05_task_binding(task, original, binding):
    """Full-field equality; never downgrade to a subset or numeric tolerance."""
    from resectionlab.core import thaw_json
    if (set(binding) != {"schema", "receipt", "grid_sha256", "complete_grid"}
            or binding["schema"] != "pat05-complete-native-grid-binding-v1"
            or binding["receipt"] != {"path": PAT05_GRID_RECEIPT, "sha256": PAT05_GRID_RECEIPT_SHA256,
                                      "json_pointer": PAT05_GRID_POINTER}
            or binding["grid_sha256"] != PAT05_GRID_SHA256
            or set(binding["complete_grid"]) != PAT05_COMPLETE_GRID_KEYS
            or hashlib.sha256(_canonical(binding["complete_grid"])).hexdigest() != PAT05_GRID_SHA256):
        raise ValueError("Expected complete PAT05 grid binding is not the pinned historical contract")
    actual_grid = thaw_json(task.case._grid_record)
    expected_grid = binding["complete_grid"]
    expected = {"native_grid": expected_grid, "objective": original["objective"],
                "max_steps": original["settings"]["max_steps"]}
    actual = {"native_grid": actual_grid, "objective": asdict(task.reward_spec),
              "max_steps": task.max_steps}
    missing = sorted(set(expected_grid) - set(actual_grid))
    extra = sorted(set(actual_grid) - set(expected_grid))
    changed = sorted(key for key in set(expected_grid) & set(actual_grid)
                     if _canonical(actual_grid[key]) != _canonical(expected_grid[key]))
    mismatches = ["native_grid." + key for key in sorted(set(missing + extra + changed))]
    mismatches += [key for key in ("objective", "max_steps") if _canonical(actual[key]) != _canonical(expected[key])]
    if mismatches:
        raise PAT05TaskBindingMismatch({"schema": "pat05-task-binding-mismatch-v1",
            "grid_anchor": {key: value for key, value in binding.items() if key != "complete_grid"},
            "mismatched_fields": mismatches, "missing_grid_fields": missing,
            "extra_grid_fields": extra, "changed_grid_fields": changed,
            "expected": expected, "actual": actual})


def source_inventory():
    paths = [Path(__file__), ROOT / "scripts/prepare_real_training_cases.py",
             ROOT / "scripts/preflight_real_spatial_policy.py",
             *(ROOT / "src/resectionlab").rglob("*.py")]
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def original_records():
    """Pinned small JSON only; no image, checkpoint or SELECT bundle read."""
    cohort = prep.read_development_cohort(ROOT / prep.COHORT_PATH)
    records = {}
    for subject in SUBJECTS:
        prep.require_development_role(cohort, subject, role="TRAIN")
        path = ROOT / (prep.COMMON_PATH if subject == "sub-PAT05" else f"{SAVED}/{subject}/preparation.json")
        if path.exists():
            raw = path.read_bytes()
        else:
            with tarfile.open(ROOT / SAVED / "completed-run.tar.gz") as archive:
                stream = archive.extractfile(f"{subject}/preparation.json")
                if stream is None:
                    raise ValueError("Missing fixed original preparation receipt")
                raw = stream.read()
        if hashlib.sha256(raw).hexdigest() != ANCHOR_SHA256[subject]:
            raise ValueError("Original preparation/task anchor changed: " + subject)
        row = json.loads(raw)
        if subject == "sub-PAT05":
            if row["member"]["subject"] != subject or row["member"]["role"] != "TRAIN":
                raise ValueError("Original PAT05 TRAIN identity changed")
            # Include this pinned dependency in the worker's existing before/
            # after original-record closure. The source member stays unchanged.
            row = {**row, "complete_native_grid_binding": pat05_complete_grid_binding(row)}
        elif (row.get("subject") != subject or row.get("role") != "TRAIN"
              or prep.binding_hash(row["binding"]) != row.get("binding_hash")
              or row.get("status") != ("blocked_support_conflict" if subject in BLOCKED else "prepared")):
            raise ValueError("Original TRAIN preparation status/binding changed")
        records[subject] = row
    return records


def declaration():
    from resectionlab.spatial_multiscale_views import PREPROCESSING_VERSION
    originals = original_records()
    common = prep.common_task_definition()
    members = {}
    for subject, row in originals.items():
        member = row["member"] if subject == "sub-PAT05" else row["binding"]["member"]
        if subject != "sub-PAT05" and row["binding"]["common_task"] != common:
            raise ValueError("Original tasks do not share the fixed physical definition")
        members[subject] = {"member": member, "original_record_sha256": ANCHOR_SHA256[subject],
            "historical_status": "prepared" if subject == "sub-PAT05" else row["status"],
            "new_task_permitted": subject not in BLOCKED}
        if subject == "sub-PAT05":
            members[subject]["complete_native_grid_binding"] = pat05_complete_grid_binding(row)
    return {"version": VERSION, "declared_at": datetime.now(timezone.utc).isoformat(),
        "subjects": list(SUBJECTS), "settings": SETTINGS, "source_sha256": source_inventory(),
        "cohort_path": prep.COHORT_PATH, "cohort_sha256": prep.COHORT_SHA256,
        "members": members, "common_task": common,
        "representations": ["legacy_access_crop", "target_local", "whole_source", "target_local_plus_whole_source"],
        "preprocessing_version": PREPROCESSING_VERSION, "new_view_value_dtype": "float64",
        "scope": "Four new zero-learning initial observations; two historical support blocks; six fixed TRAIN rows",
        "input_track": "annotation_assisted", "target_rule": "covered positive permitted nominal bounding box; no reference/reward/access selection",
        "normalization": "reuse frozen support-percentile limits; normalize source before either view; source MRI unchanged",
        "geometry": "unchanged native-grid task, original access/tools/proposals/objective/horizon; alternative views never enter task",
        "ray_convention": "current five entry-to-tip samples; separate continuous shaft centerline at approach/deepest poses; center-domain and fullcell-domain distinct",
        "success": "bounded correct representation and unchanged task; no requirement for improved coverage or return",
        "prohibited": ["cuts", "search", "policy_forward", "learning", "checkpoint_load", "access_selection", "support_or_target_edit", "automatic_retry"],
        "clinical_deficit_probability": None, "clinical_use_permitted": False}


def validate(record):
    if (record.get("version") != VERSION or record.get("subjects") != list(SUBJECTS)
            or record.get("settings") != SETTINGS):
        raise ValueError("Only the fixed ordered six-TRAIN zero-learning diagnostic is permitted")
    expected = declaration()
    expected["declared_at"] = record.get("declared_at")
    if expected != record:
        raise ValueError("Frozen source, original patient/task binding or diagnostic settings changed")
    if datetime.fromisoformat(record["declared_at"]).utcoffset() is None:
        raise ValueError("Prospective declaration needs an aware timestamp")
    return original_records()


def validate_release(release, manifest_sha256):
    """An archive/release identity must exist before any worker image read."""
    commit = release.get("source_commit", "")
    if (release.get("version") != RELEASE_VERSION or release.get("authorized") is not True
            or release.get("manifest_sha256") != manifest_sha256
            or release.get("runtime_root") != str(ROOT.resolve())
            or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit)):
        raise ValueError("Exact prospective source archive and execution release required")
    archive = Path(release.get("source_archive", ""))
    if not archive.is_file() or sha256(archive) != release.get("source_archive_sha256"):
        raise ValueError("Released immutable source archive changed or is absent")
    # Prove the actual numerical files equal archive entries, not merely a path.
    with tarfile.open(archive) as stream:
        if stream.pax_headers.get("comment") != commit:
            raise ValueError("Released archive commit differs from declared source commit")
        if hashlib.sha256(stream.extractfile(MANIFEST).read()).hexdigest() != manifest_sha256:
            raise ValueError("Released archive manifest differs from declared byte identity")
        for path, digest in source_inventory().items():
            entry = stream.getmember(path)
            if not entry.isfile():
                raise ValueError("Numerical archive entry is not a regular file")
            if hashlib.sha256(stream.extractfile(entry).read()).hexdigest() != digest:
                raise ValueError("Runtime numerical source differs from released archive")


def load_initial(subject, originals, record, check):
    """Only four fixed successful TRAIN sources can reach an image decoder."""
    if subject not in SUBJECTS or subject in BLOCKED:
        raise ValueError("Only original successful TRAIN preparations may be decoded")
    check()
    if subject != "sub-PAT05":
        return prep.load_prepared_training_case(originals[subject], cancelled=lambda: _cancelled(check))
    row, member = originals[subject], record["members"][subject]["member"]
    if _canonical(member) != _canonical(row["member"]):
        raise ValueError("Original PAT05 member declaration changed")
    binding = pat05_complete_grid_binding(row)
    if _canonical(record["members"][subject].get("complete_native_grid_binding")) != _canonical(binding):
        raise ValueError("Declared complete PAT05 grid binding differs from the authenticated historical record")
    cohort = prep.read_development_cohort(ROOT / prep.COHORT_PATH)
    prep.require_development_role(cohort, subject, role="TRAIN")
    path = ROOT / member["case_bundle"]
    if sha256(path) != member["case_bundle_sha256"]:
        raise ValueError("Original PAT05 bundle bytes changed")
    check()
    case = prep._decode_case(path)
    source = case.metadata.get("source_collection", {})
    if (case.case_id != "BTC-ds001226-sub-PAT05-preop" or case.semantic_hash != member["case_semantic_hash"]
            or case.planning_hash != member["research_support_acknowledgment"]["planning_hash"]
            or case.metadata.get("is_synthetic")
            or any(source.get(key) != cohort["source"][key] for key in ("accession", "release", "git_commit"))
            or not any(ref.source_id == "structural" and ref.provenance == "observed" for ref in case.source_refs)):
        raise ValueError("Original PAT05 source identity changed")
    check()
    task = prep._construct_task(case, member, record["common_task"], lambda: _cancelled(check))
    verify_pat05_task_binding(task, row, binding)
    return task


def _cancelled(check):
    try:
        check()
        return False
    except TimeoutError:
        return True


class InitialOnlyViolation(RuntimeError):
    """Fatal observer failure: CPython disables profiling after a callback raises."""


class InitialInventoryGuard:
    """Count initial previews and forbid commits/steps/policy/search calls."""
    def __init__(self, check):
        self.check, self.total, self.case_calls = check, 0, 0
        self.subject, self.seen, self.active = None, set(), False

    def begin_case(self, subject):
        self.check()
        if self.active and sys.getprofile() != self._observe:
            raise InitialOnlyViolation("Initial-only observer was disabled; cannot continue")
        if subject not in SUBJECTS or subject in BLOCKED or subject in self.seen or self.total >= SETTINGS["max_total_previews"]:
            raise ValueError("No preview allocation remains for this case")
        self.subject, self.case_calls = subject, 0
        self.seen.add(subject)

    def __enter__(self):
        from resectionlab.native_resection import NativeResectionEngine
        from resectionlab.native_spatial_task import NativeSpatialTask
        if sys.getprofile() is not None or sys.gettrace() is not None:
            raise RuntimeError("Initial-only diagnostic requires an exclusive call observer")
        self.preview = NativeResectionEngine.preview_stroke.__code__
        self.forbidden = {NativeResectionEngine.commit_preview.__code__, NativeSpatialTask.step.__code__}
        sys.setprofile(self._observe)
        self.active = True
        return self

    def _observe(self, frame, event, result):
        if event != "call":
            return
        code = frame.f_code
        module = frame.f_globals.get("__name__", "")
        if (code in self.forbidden or (module == "resectionlab.spatial_policy"
                and code.co_name in {"__init__", "forward", "gradient_step", "reinforce_loss", "behavior_cloning_loss"})
                or (module == "resectionlab.observed_search" and code.co_name != "<module>")
                or (module.startswith("torch.optim.") and code.co_name in {"__init__", "step"})):
            raise InitialOnlyViolation("Policy, learning, search and committed transitions are prohibited")
        if code is self.preview:
            self.check()
            if (self.subject is None or self.case_calls >= SETTINGS["max_initial_previews_per_case"]
                    or self.total >= SETTINGS["max_total_previews"]):
                raise InitialOnlyViolation("Initial preview cap reached before execution")
            self.case_calls += 1
            self.total += 1

    def __exit__(self, *args):
        sys.setprofile(None)
        self.active = False


def full_source_from_task(task, observation):
    """Extract allowed fields only; private target/reference metrics are unused."""
    import numpy as np
    from resectionlab.spatial_multiscale_views import PermittedSourceChannel, PermittedVolumeSource
    from resectionlab.spatial_observations import CHANNEL_NAMES
    case = task.case
    intensity = case.structural_intensity
    if case.intensity_normalization != "raw":
        lower, upper = case._normalization_record["lower"], case._normalization_record["upper"]
        intensity = np.clip((intensity.astype(np.float64) - lower) / (upper - lower), 0., 1.).astype(np.float32)
    arrays = {"structural_intensity": intensity, "nominal_tissue": case.observed_support,
        "nominal_target": case.nominal_target, "observed_cavity": task._engine.removed_mask}
    channels = {name: PermittedSourceChannel(arrays.get(name), **dict(observation.channel_provenance[name]))
                if arrays.get(name) is not None else PermittedSourceChannel() for name in CHANNEL_NAMES}
    return PermittedVolumeSource(channels, case._native_affine_ras_mm, case.track, case.source_hash)


def _corners(shape, affine, *, cells):
    import numpy as np
    bounds = [(-.5, n - .5) if cells else (0., n - 1.) for n in shape]
    voxels = np.asarray(list(product(*bounds)))
    return voxels @ np.asarray(affine)[:3, :3].T + np.asarray(affine)[:3, 3]


def _interval(first, last, low, high):
    """Closed box intersection in segment parameter space; None means no hit."""
    lo, hi = 0., 1.
    for position, delta, a, b in zip(first, last - first, low, high):
        if delta == 0:
            if not a <= position <= b:
                return None
        else:
            left, right = sorted(((a - position) / delta, (b - position) / delta))
            lo, hi = max(lo, left), min(hi, right)
            if hi < lo:
                return None
    return [float(lo), float(hi)]


def _union_length(intervals):
    spans = sorted(interval for interval in intervals if interval is not None)
    if not spans:
        return 0.
    start, end = spans[0]
    total = 0.
    for first, last in spans[1:]:
        if first > end:
            total += end - start
            start, end = first, last
        else:
            end = max(end, last)
    return float(total + end - start)


def segment_visibility(first, last, affine, shape, coverage):
    """Five policy-convention samples plus distinct continuous box intervals.

    Fractional channel coverage is interpolated, never promoted to known/true.
    Full-cell extents report geometric representation extent, not interpolation
    availability or finite-radius tool clearance.
    """
    import numpy as np
    from resectionlab.spatial_policy import GRID_ROUNDOFF_BAND, world_to_sample_grid
    first, last, shape = np.asarray(first, float), np.asarray(last, float), np.asarray(shape)
    affine, coverage = np.asarray(affine, float), np.asarray(coverage, float)
    points = first + np.linspace(0., 1., SETTINGS["ray_samples"])[:, None] * (last - first)
    grid = world_to_sample_grid(points, affine, shape)
    inside = (np.abs(grid) <= 1).all(axis=1)
    voxel = (grid[:, ::-1] + 1.) * (shape - 1) / 2.
    coverage_samples = np.zeros((len(points), coverage.shape[0]), float)
    for i in np.flatnonzero(inside):
        lower = np.floor(voxel[i]).astype(int)
        upper = np.minimum(lower + 1, shape - 1)
        fraction = voxel[i] - lower
        for choice in product((0, 1), repeat=3):
            index = tuple(np.where(choice, upper, lower))
            weight = float(np.prod(np.where(choice, fraction, 1. - fraction)))
            coverage_samples[i] += weight * coverage[(slice(None), *index)]
    inv = np.linalg.inv(affine)
    native = points @ inv[:3, :3].T + inv[:3, 3]
    center_interval = _interval(grid[0], grid[-1], [-1.] * 3, [1.] * 3)
    cell_grid = 2. * (native + .5) / shape - 1.
    cell_grid = np.where(np.abs(np.abs(cell_grid) - 1.) <= GRID_ROUNDOFF_BAND, np.sign(cell_grid), cell_grid)
    cell_interval = _interval(cell_grid[0], cell_grid[-1], [-1.] * 3, [1.] * 3)
    cell_inside = (np.abs(cell_grid) <= 1.).all(axis=1)
    return {"sample_points_ras_mm": points.tolist(), "sample_inside_center_domain": inside.tolist(),
        "sample_inside_fullcell_extent": cell_inside.tolist(),
        "sample_channel_coverage_fraction": coverage_samples.tolist(),
        "center_domain_interval": center_interval, "fullcell_extent_interval": cell_interval,
        "continuous_center_domain_fraction": _union_length([center_interval]),
        "continuous_fullcell_extent_fraction": _union_length([cell_interval])}


def view_description(view):
    import numpy as np
    view.assert_intact()
    shape, affine = view.image_channels.shape[1:], np.asarray(view.affine_ras_mm)
    coverage = np.asarray(view.coverage if hasattr(view, "coverage") else view.coverage_fraction)
    corners = _corners(shape, affine, cells=True)
    inv = np.linalg.inv(affine)
    roundtrip = (corners @ inv[:3, :3].T + inv[:3, 3]) @ affine[:3, :3].T + affine[:3, 3]
    error = float(np.max(np.linalg.norm(corners - roundtrip, axis=1)))
    if error > SETTINGS["extent_roundtrip_atol_mm"]:
        raise ValueError("View physical coordinate roundtrip exceeds declared tolerance")
    return {"shape": list(shape), "value_dtype": str(view.image_channels.dtype), "affine_ras_mm": affine.tolist(),
        "center_corners_ras_mm": _corners(shape, affine, cells=False).tolist(),
        "fullcell_corners_ras_mm": corners.tolist(), "roundtrip_max_error_mm": error,
        "channel_available": view.channel_available.tolist(),
        "coverage_fraction_range": [[float(x.min()), float(x.max())] for x in coverage],
        "nominal_mass_mm3": float(np.sum(view.image_channels[2], dtype=np.float64) * abs(np.linalg.det(affine[:3, :3]))),
        "nominal_positive_cells": int(np.count_nonzero(view.image_channels[2] > 0)),
        "fingerprint": view.fingerprint,
        "preprocessing_version": getattr(view, "preprocessing_version", "legacy-access-crop-unmodified"),
        "preprocessing_hash": getattr(view, "preprocessing_hash", None)}


def compare_views(observation, views, inventory, tools, source_shape, source_affine, nominal_target):
    import numpy as np
    from resectionlab.core import thaw_json
    mapping = {"legacy_access_crop": observation, "target_local": views.target_local, "whole_source": views.whole_source}
    details = {name: view_description(view) for name, view in mapping.items()}
    source_corners = _corners(source_shape, source_affine, cells=True)
    global_corners = np.asarray(details["whole_source"]["fullcell_corners_ras_mm"])
    extent_error = float(np.max(np.linalg.norm(global_corners - source_corners, axis=1)))
    source_mass = float(np.asarray(nominal_target).sum(dtype=np.float64) * abs(np.linalg.det(source_affine[:3, :3])))
    mass_error = abs(details["whole_source"]["nominal_mass_mm3"] - source_mass)
    mass_limit = SETTINGS["coarse_mass_small_volume_atol_mm3"] if source_mass < 1. else SETTINGS["coarse_mass_rtol"] * source_mass
    if extent_error > SETTINGS["extent_roundtrip_atol_mm"] or mass_error > mass_limit:
        raise ValueError("Global physical extent or nominal mass exceeds prospective error gate")
    for name, info in details.items():
        info.update(nominal_mass_fraction=info["nominal_mass_mm3"] / source_mass if source_mass else None,
            nominal_mass_omitted_mm3=max(0., source_mass - info["nominal_mass_mm3"]))
    tool_map, rows = {tool.tool_id: tool for tool in tools}, []
    for proposal in inventory["emitted"]:
        entry, tip = np.asarray(proposal["entry_mm"]), np.asarray(proposal["tip_mm"])
        axis = (tip - entry) / np.linalg.norm(tip - entry)
        tool = tool_map[proposal["tool_id"]]
        segments = {"entry_to_tip": (entry, tip),
            "approach_shaft_centerline": (entry - tool.working_length_mm * axis, entry - tool.tip_length_mm * axis),
            "deepest_shaft_centerline": (tip - tool.working_length_mm * axis, tip - tool.tip_length_mm * axis)}
        visibility = {}
        for segment_name, (first, last) in segments.items():
            individual = {name: segment_visibility(first, last, view.affine_ras_mm, view.image_channels.shape[1:],
                view.coverage if hasattr(view, "coverage") else view.coverage_fraction) for name, view in mapping.items()}
            a, b = individual["target_local"], individual["whole_source"]
            individual["target_local_plus_whole_source"] = {
                "sample_inside_center_domain": np.logical_or(a["sample_inside_center_domain"], b["sample_inside_center_domain"]).tolist(),
                "sample_inside_fullcell_extent": np.logical_or(a["sample_inside_fullcell_extent"], b["sample_inside_fullcell_extent"]).tolist(),
                "sample_channel_coverage_fraction_max_across_views": np.maximum(a["sample_channel_coverage_fraction"], b["sample_channel_coverage_fraction"]).tolist(),
                "continuous_center_domain_fraction": _union_length([a["center_domain_interval"], b["center_domain_interval"]]),
                "continuous_fullcell_extent_fraction": _union_length([a["fullcell_extent_interval"], b["fullcell_extent_interval"]]),
                "combination": "geometric union and maximum observed coverage; no feature fusion or independence assumption"}
            visibility[segment_name] = individual
        rows.append({"action_id": proposal["action_id"], "tool_id": tool.tool_id,
            "accepted": proposal["feasible"], "reason": proposal["reason"], "visibility": visibility})
    summaries = {}
    for group, selected in (("emitted", rows), ("accepted", [x for x in rows if x["accepted"]])):
        summaries[group] = {"action_count": len(selected), "views": {}}
        for name in (*mapping, "target_local_plus_whole_source"):
            segments = {}
            for segment in ("entry_to_tip", "approach_shaft_centerline", "deepest_shaft_centerline"):
                samples = [x["visibility"][segment][name] for x in selected]
                segments[segment] = {"mean_center_sample_fraction": None if not samples else float(np.mean([
                    np.mean(x["sample_inside_center_domain"]) for x in samples])),
                    **{f"mean_{domain}_fraction": None if not samples else float(np.mean([
                        x[f"continuous_{domain}_fraction"] for x in samples]))
                       for domain in ("center_domain", "fullcell_extent")}}
            summaries[group]["views"][name] = segments
    return {"views": details, "visibility_summary": summaries, "preparation_report": thaw_json(views.report),
        "source_shape": list(source_shape), "source_affine_ras_mm": source_affine.tolist(),
        "source_fullcell_corners_ras_mm": source_corners.tolist(), "source_nominal_mass_mm3": source_mass,
        "global_extent_max_error_mm": extent_error, "coarse_nominal_mass_absolute_error_mm3": mass_error,
        "coarse_nominal_mass_relative_error": mass_error / source_mass if source_mass else None,
        "coarse_nominal_mass_allowed_absolute_error_mm3": mass_limit,
        "emitted_actions": len(rows), "accepted_actions": sum(x["accepted"] for x in rows), "actions": rows,
        "interpretation": "Visibility of permitted representations only; no clearance, deformation, full-removal or clinical-accuracy claim. Coarse target cells are volume averages, not native target-cell counts."}


def task_invariants(task):
    import numpy as np
    from resectionlab.core import array_digest, semantic_digest
    task._assert_frozen()
    engine = task._engine
    if (engine.revision or engine.history or task._steps or task._history
            or np.any(engine.removed_mask) or np.any(engine.contact_mask)):
        raise RuntimeError("Representation diagnostic requires untouched initial procedure state")
    inventory, observation = task.candidate_inventory(), task.observation()
    return {"physical_inventory_hash": semantic_digest([
        {key: row[key] for key in ("tool_id", "entry_mm", "tip_mm", "feasible", "reason")} for row in inventory["emitted"]]),
        "inventory_hash": semantic_digest(inventory), "observation_hash": observation.fingerprint,
        "source_hash": task.case.source_hash, "decision_model_hash": task.decision_model_hash,
        "cavity_state_hash": engine.state_hash,
        "state_array_hashes": {name: array_digest(getattr(engine, name)) for name in
            ("remaining_mask", "removed_mask", "contact_mask", "connected_free_mask")}}


def inspect_task(task, check=lambda: None):
    from resectionlab.spatial_multiscale_views import prepare_multiscale_views
    from resectionlab.spatial_policy_diagnostics import runtime_proposal_coverage
    check()
    before = task_invariants(task)
    observation, inventory = task.observation(), task.candidate_inventory()
    runtime_proposal_coverage(task.case, inventory, observation, native_affine=task.case._native_affine_ras_mm)
    start = time.perf_counter()
    source = full_source_from_task(task, observation)
    source_seconds = time.perf_counter() - start
    check()
    start = time.perf_counter()
    views = prepare_multiscale_views(source, local_shape=tuple(SETTINGS["local_shape"]), coarse_shape=tuple(SETTINGS["coarse_shape"]))
    view_seconds = time.perf_counter() - start
    check()
    start = time.perf_counter()
    comparison = compare_views(observation, views, inventory, task.case.tools,
        task.case.structural_intensity.shape, task.case._native_affine_ras_mm, source.channels["nominal_target"].data)
    comparison["original_source_affine_ras_mm"] = task.case.affine_ras_mm.tolist()
    comparison["physical_frame_basis"] = "native reconciliation affine for views and rays; original source index frame retained separately"
    diagnostic_seconds = time.perf_counter() - start
    after = task_invariants(task)
    if before != after:
        raise RuntimeError("View-only construction changed task, inventory or legacy observation")
    check()
    return {"status": "complete", "task_before": before, "task_after": after,
        "physical_catalog_unchanged": True, "executed_transitions": 0, "policy_forwards": 0,
        "optimizer_updates": 0, "source_preparation_seconds": source_seconds,
        "view_construction_seconds": view_seconds, "coverage_diagnostic_seconds": diagnostic_seconds,
        "initial_inventory": inventory, "coverage": comparison, "peak_rss_bytes": peak_rss_bytes()}


def loaded_source_paths():
    paths = {}
    for name, module in tuple(sys.modules.items()):
        if name == "resectionlab" or name.startswith("resectionlab."):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to((ROOT / "src/resectionlab").resolve()):
                raise ValueError("Numerical import escaped the released source root")
            paths[name] = str(path)
    return paths


def blank_report():
    return {"version": VERSION, "status": "preparing", "settings": SETTINGS,
        "cohort_denominator": 6, "new_task_attempts": 0, "new_representation_completions": 0,
        "historical_support_blocks": 2, "policy_forwards": 0, "optimizer_updates": 0, "executed_transitions": 0,
        "clinical_deficit_probability": None, "subjects": {subject: {
            "status": "historical_support_block" if subject in BLOCKED else "not_attempted",
            "new_task_attempted": False, "representation": None} for subject in SUBJECTS}}


def worker(record, output, *, manifest_sha256, release):
    started = time.perf_counter()
    report = blank_report()
    def check():
        if time.perf_counter() - started >= SETTINGS["cooperative_seconds"] or peak_rss_bytes() > SETTINGS["max_rss_bytes"]:
            raise TimeoutError("Whole-worker wall or RSS limit reached")
    def save():
        report.update(elapsed_seconds=time.perf_counter() - started, peak_rss_bytes=peak_rss_bytes())
        write_json(output / "receipt.json", report)
    save()
    try:
        validate_release(release, manifest_sha256)
        originals = validate(record)
        report["initial_import_paths"] = loaded_source_paths()
        check()
        for subject in BLOCKED:
            report["subjects"][subject].update(historical_preparation_sha256=ANCHOR_SHA256[subject],
                historical_coverage=originals[subject]["coverage"],
                reason="Retained original source-bound target/support conflict; no new decode or representation attempted")
        save()
        report["bundle_bytes_verified_before"] = {}
        for subject in SUBJECTS:
            member = record["members"][subject]["member"]
            if sha256(ROOT / member["case_bundle"]) != member["case_bundle_sha256"]:
                raise ValueError("Original bundle changed before task construction: " + subject)
            report["bundle_bytes_verified_before"][subject] = True
            check()
        save()
        with InitialInventoryGuard(check) as guard:
            for subject in SUBJECTS:
                if subject in BLOCKED:
                    continue
                guard.begin_case(subject)
                row = report["subjects"][subject]
                row.update(status="preparing", new_task_attempted=True)
                report["new_task_attempts"] += 1
                save()
                task = None
                try:
                    case_start = time.perf_counter()
                    task = load_initial(subject, originals, record, check)
                    row["initial_preparation_seconds"] = time.perf_counter() - case_start
                    row["initial_previews"] = guard.case_calls
                    # Prevent a view helper from silently requesting more previews.
                    guard.subject = None
                    row["representation"] = inspect_task(task, check)
                    row["status"] = "complete"
                    report["new_representation_completions"] += 1
                except BaseException as error:
                    row.update(status="failed", initial_previews=guard.case_calls,
                        failure={"type": type(error).__name__, "message": str(error)})
                    if isinstance(error, PAT05TaskBindingMismatch):
                        row["failure"]["binding_mismatch"] = error.details
                    if isinstance(error, (InitialOnlyViolation, TimeoutError, KeyboardInterrupt, SystemExit)):
                        raise
                finally:
                    report["total_previews"] = guard.total
                    save()
                    del task
                    gc.collect()
                check()
        closures = {subject: sha256(ROOT / record["members"][subject]["member"]["case_bundle"]) ==
                    record["members"][subject]["member"]["case_bundle_sha256"] for subject in SUBJECTS}
        report["bundle_bytes_unchanged"] = closures
        report["sources_unchanged"] = source_inventory() == record["source_sha256"]
        report["original_records_unchanged"] = original_records() == originals
        report["final_import_paths"] = loaded_source_paths()
        if not all(closures.values()) or not report["sources_unchanged"] or not report["original_records_unchanged"]:
            raise ValueError("Original source bytes or metadata changed during diagnostic")
        report["status"] = "complete" if report["new_representation_completions"] == 4 else "incomplete"
        check()
        save()
    except BaseException as error:
        report.update(status="incomplete", failure={"type": type(error).__name__, "message": str(error)})
        save()
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declare", action="store_true")
    parser.add_argument("--manifest", type=Path, default=ROOT / MANIFEST)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--release", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--expected-release-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.declare:
        if args.manifest.exists():
            raise ValueError("Preserve existing declaration; do not overwrite")
        write_json(args.manifest, declaration())
        return
    record, digest, raw = read_declaration(args.manifest, args.expected_sha256)
    if not args.execute and not args.worker:
        validate(record)
        print("Metadata/source validation passed; no patient data read")
        return
    if args.output is None or args.release is None:
        parser.error("Execution requires output and a separately frozen release receipt")
    release, release_digest, release_raw = read_declaration(args.release, args.expected_release_sha256)
    if args.worker:
        if args.expected_sha256 is None or args.expected_release_sha256 is None:
            raise ValueError("Worker requires parent-bound declaration and release byte identities")
        result = worker(record, args.output, manifest_sha256=digest, release=release)
        if result["status"] != "complete":
            raise SystemExit(1)
        return
    validate(record)
    validate_release(release, digest)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "declaration-input.json").write_bytes(raw)
    (args.output / "release-input.json").write_bytes(release_raw)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--manifest",
        str(args.output / "declaration-input.json"), "--expected-sha256", digest,
        "--release", str(args.output / "release-input.json"), "--expected-release-sha256", release_digest,
        "--output", str(args.output)]
    result = supervise_worker(command, args.output, SETTINGS, digest)
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
