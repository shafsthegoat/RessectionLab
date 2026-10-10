"""ReMIND MR/SEG geometry and explicit public-grid crop QC.

Inputs are an exact hash-bound case manifest and the existing family-role cohort.
Header projection precedes a separately invoked crop conversion. The source
converter is reused; no label admission, resampling or registration is implicit.
Original executed first-case scripts and evidence remain immutable snapshots.
"""
from __future__ import annotations
import gc
import hashlib
import io
import itertools
import json
from pathlib import Path
import resource
import struct
import sys
import time
import types
import zlib

ROOT = Path(__file__).resolve().parents[2]
CORE_SHA = "34e6ceb9560344a88226489b2dd8ca6bfb90cb87c52d2ff3184d393293df4ef6"
COHORT_SHA = "326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05"
MAX_RSS = 2*1024**3
MAX_CORNER_MM = .001
PUBLIC_SERIES_KINDS = frozenset(("structural_t1ce", "whole_tumor", "cerebrum"))
PUBLIC_SUBJECTS_BY_ROLE = {"TRAIN": ("ReMIND-008", "ReMIND-010", "ReMIND-020", "ReMIND-025"),
                           "SELECT": ("ReMIND-013", "ReMIND-037")}


def need(value, reason):
    if not value: raise ValueError(reason)


def digest(raw): return hashlib.sha256(raw).hexdigest()


def save(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream: stream.write(raw)
    return digest(raw)


def source_module(path, expected):
    raw = Path(path).read_bytes(); need(digest(raw) == expected, "converter_source_changed")
    value = types.ModuleType(Path(path).stem); value.__file__ = str(path)
    exec(compile(raw, str(path), "exec", dont_inherit=True), value.__dict__)
    return value


def value(ds, name):
    item = getattr(ds, name, None)
    if item is None: return None
    if isinstance(item, (str, int, float)): return str(item)
    try: return [str(x) for x in item]
    except TypeError: return str(item)


def timing(ds):
    return {name: value(ds, name) for name in ("StudyDescription", "StudyDate", "StudyTime", "SeriesDescription", "SeriesDate", "SeriesTime", "AcquisitionDateTime", "AcquisitionDate", "AcquisitionTime", "ContentDate", "ContentTime")}


def read_object(obj, *, pixels, accounting):
    import pydicom
    path = Path(obj["path"])
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink(), "exact_source_path")
    before = path.stat(); size = obj["bytes"]
    need(before.st_size == size and 0 < size <= 64*1024**2, "object_extent")
    with path.open("rb", buffering=0) as stream:
        raw = stream.read(size)
    accounting["source_file_returned_bytes"] += len(raw)
    after = path.stat()
    need(len(raw) == size and (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "source_changed_or_short")
    need(digest(raw) == obj["sha256"], "source_sha256")
    need(not obj.get("expected_md5") or hashlib.md5(raw).hexdigest() == obj["expected_md5"], "source_md5")
    accounting["objects_verified"] += 1
    ds = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=not pixels, force=False)
    return ds


def identity(ds, case, series):
    expected = {"PatientID": case["patient_id"], "SeriesInstanceUID": series["SeriesInstanceUID"],
                "StudyInstanceUID": series["StudyInstanceUID"], "Modality": series["Modality"]}
    need(all(str(getattr(ds, key, "")) == str(v) for key, v in expected.items()), "patient_or_series_identity")


def geometry(datasets, core):
    import numpy as np
    first = datasets[0]
    if str(first.Modality) == "MR":
        orientation, spacing = first.ImageOrientationPatient, first.PixelSpacing
        for ds in datasets:
            need(ds.Rows == first.Rows and ds.Columns == first.Columns and int(getattr(ds, "NumberOfFrames", 1)) == 1 and
                 np.allclose(ds.ImageOrientationPatient, orientation, rtol=0, atol=1e-6) and
                 np.allclose(ds.PixelSpacing, spacing, rtol=0, atol=1e-6), "MR_classic_uniform_grid")
        positions = [ds.ImagePositionPatient for ds in datasets]
    else:
        need(len(datasets) == 1 and str(first.SegmentationType) == "BINARY" and len(first.SegmentSequence) == 1, "single_binary_SEG_required")
        shared = first.SharedFunctionalGroupsSequence[0]
        orientation, spacing = shared.PlaneOrientationSequence[0].ImageOrientationPatient, shared.PixelMeasuresSequence[0].PixelSpacing
        for frame in first.PerFrameFunctionalGroupsSequence:
            if hasattr(frame, "PlaneOrientationSequence"):
                need(len(frame.PlaneOrientationSequence) == 1 and
                     np.allclose(frame.PlaneOrientationSequence[0].ImageOrientationPatient, orientation, rtol=0, atol=1e-6),
                     "SEG_per_frame_orientation_conflict")
            if hasattr(frame, "PixelMeasuresSequence"):
                need(len(frame.PixelMeasuresSequence) == 1 and
                     np.allclose(frame.PixelMeasuresSequence[0].PixelSpacing, spacing, rtol=0, atol=1e-6),
                     "SEG_per_frame_spacing_conflict")
        positions = [f.PlanePositionSequence[0].ImagePositionPatient for f in first.PerFrameFunctionalGroupsSequence]
        need(len(positions) == int(first.NumberOfFrames), "SEG_frame_count")
        need(all(f.SegmentIdentificationSequence[0].ReferencedSegmentNumber == first.SegmentSequence[0].SegmentNumber
                 for f in first.PerFrameFunctionalGroupsSequence), "SEG_mixed_segment")
    order, affine, residual = core.regular_affine(orientation, spacing, positions)
    frames = {str(getattr(ds, "FrameOfReferenceUID", "")) for ds in datasets}
    need(len(frames) == 1 and "" not in frames, "one_present_frame_required")
    import itertools
    shape = [int(first.Columns), int(first.Rows), len(positions)]
    corners = np.asarray(list(itertools.product(*[(-.5, n-.5) for n in shape])))
    world = (np.c_[corners, np.ones(8)] @ affine.T)[:, :3]
    return {"shape_xyz": [int(first.Columns), int(first.Rows), len(positions)], "affine_xyz_to_ras_mm": affine.tolist(),
            "frame_of_reference_uid": next(iter(frames)), "slice_position_max_residual_mm": residual,
            "raw_orientation": [str(x) for x in orientation], "raw_pixel_spacing": [str(x) for x in spacing],
            "raw_positions": [[str(x) for x in p] for p in positions], "sorted_source_indices": order.tolist(),
            "slice_step_mm": float(np.linalg.norm(affine[:3, 2])),
            "native_grid_cell_bounds_ras_mm": {"minimum": world.min(0).tolist(), "maximum": world.max(0).tolist()},
            "bounds_are_anatomical_coverage": False}


def ancestry(ds):
    segments = []
    for item in ds.SegmentSequence:
        codes = []
        for key in ("SegmentedPropertyCategoryCodeSequence", "SegmentedPropertyTypeCodeSequence", "AnatomicRegionSequence"):
            codes.extend({"sequence": key, **{n: value(code, n) for n in ("CodeValue", "CodingSchemeDesignator", "CodeMeaning")}} for code in getattr(item, key, []))
        segments.append({"number": int(item.SegmentNumber), "label": value(item, "SegmentLabel"), "description": value(item, "SegmentDescription"),
                         "algorithm_type": value(item, "SegmentAlgorithmType"), "algorithm_name": value(item, "SegmentAlgorithmName"), "coded_semantics": codes})
    return {"segments": segments, "referenced_sop_instance_uids": sorted({str(e.value) for e in ds.iterall() if e.keyword == "ReferencedSOPInstanceUID"}),
            "source_annotation_accuracy": "unreviewed", "automatic_annotation_is_independent_ground_truth": False}


def pairing(mr, seg):
    import numpy as np
    same = mr["geometry"]["frame_of_reference_uid"] == seg["geometry"]["frame_of_reference_uid"]
    refs = set(seg["ancestry"]["referenced_sop_instance_uids"]); sops = set(mr["sop_instance_uids"])
    return {"frame_uid_matches": same, "explicit_SOP_refs_present": bool(refs), "all_explicit_refs_in_selected_MR": bool(refs) and refs <= sops,
            "unmatched_explicit_SOP_refs": sorted(refs - sops), "source_description": seg["source_description"],
            "MR_index_to_SEG_index": (np.linalg.inv(seg["geometry"]["affine_xyz_to_ras_mm"]) @ np.asarray(mr["geometry"]["affine_xyz_to_ras_mm"])).tolist() if same else None,
            "registration_performed": False, "identity_cross_frame_assumed": False, "expert_alignment_review": False,
            "correspondence": "shared_frame_plus_source_description" if same and not refs else "explicit_refs_and_frame_match" if same and refs <= sops else "unresolved"}


def validate_public_scope(case, *, public_only=False):
    """Public mode is an exact three-series read boundary, never a filter."""
    if public_only:
        need(case["patient_id"] in PUBLIC_SUBJECTS_BY_ROLE.get(case["role"], ()), "frozen_public_pilot_role")
        kinds = [s["kind"] for s in case["series"]]
        need(len(kinds) == 3 and set(kinds) == PUBLIC_SERIES_KINDS, "exact_public_series_only")
    else:
        need(case["role"] == "TRAIN", "TRAIN_case_required")


def project(case, core, accounting, *, public_only=False):
    validate_public_scope(case, public_only=public_only)
    need(len({s["kind"] for s in case["series"]}) == len(case["series"]) and
         {s["kind"] for s in case["series"]} >= (PUBLIC_SERIES_KINDS if public_only else
             {"structural_t1ce", "whole_tumor", "ventricles"}), "case_series_kinds")
    rows = []
    for series in case["series"]:
        need(series["kind"] in ("structural_t1ce", "whole_tumor", "ventricles", "cerebrum"), "undeclared_series_kind")
        need(series["Modality"] == ("MR" if series["kind"] == "structural_t1ce" else "SEG"), "kind_modality")
        headers = [read_object(obj, pixels=False, accounting=accounting) for obj in series["objects"]]
        for ds in headers: identity(ds, case, series)
        sops = [str(ds.SOPInstanceUID) for ds in headers]; need(len(set(sops)) == len(sops), "duplicate_source_SOP")
        row = {"kind": series["kind"], "modality": series["Modality"], "series_instance_uid": series["SeriesInstanceUID"],
               "source_description": series.get("source_description"), "annotation_provenance": series.get("annotation_provenance"),
               "sop_instance_uids": sops, "geometry": geometry(headers, core), "timing_raw": [timing(ds) for ds in headers],
               "timing_interpretation": "Retain deidentified source timestamps; Preop label alone is not annotation availability proof.",
               "source_objects": series["objects"]}
        if series["Modality"] == "SEG": row["ancestry"] = ancestry(headers[0])
        rows.append(row)
    mr = next(x for x in rows if x["kind"] == "structural_t1ce")
    for row in rows:
        if row["modality"] == "SEG": row["alignment"] = pairing(mr, row)
    voxels = int(__import__("math").prod(mr["geometry"]["shape_xyz"]))
    source_bytes = sum(o["bytes"] for s in case["series"] for o in s["objects"])
    # Conservative declared estimate, not a measured process peak.
    estimate = source_bytes + voxels*16 + 256*1024**2
    return {"series": rows, "MR_voxels": voxels, "MR_float32_bytes": voxels*4,
            "estimated_native_peak_bytes": estimate, "native_peak_estimate_not_measurement": True,
            "within_existing_32million_voxel_case_cap": voxels <= 32000000, "downsampling_performed": False}


def validate_role(case, cohort, *, public_only=False):
    """Preserve original family roles; SELECT has no optimization admission."""
    members = [row for row in cohort["members"] if row["subject"] == case["patient_id"]]
    allowed = (case["patient_id"] in PUBLIC_SUBJECTS_BY_ROLE.get(case["role"], ())) if public_only else case["role"] == "TRAIN"
    need(len(members) == 1 and allowed and case["role"] == members[0]["role"] and
         case["patient_group"] == members[0]["patient_group"], "immutable_public_pilot_role" if public_only else "immutable_TRAIN_role")


def validate_case(case, repository_root, *, public_only=False):
    repository_root = Path(repository_root).resolve()
    if public_only: validate_public_scope(case, public_only=True)
    parents = case["parent_bindings"]
    parent = parents.get("source_binding", parents.get("first_train_source_binding_v2"))
    need(isinstance(parent, dict), "source_binding_required")
    path = Path(parent["path"])
    if not path.is_absolute(): path = repository_root / path
    raw = path.read_bytes(); need(digest(raw) == parent["sha256"], "source_binding_digest")
    binding = json.loads(raw)
    cohort_raw = (repository_root / "manifests/experiments/remind-component-cohort-v1.json").read_bytes()
    need(digest(cohort_raw) == COHORT_SHA, "cohort_digest")
    validate_role(case, json.loads(cohort_raw), public_only=public_only)
    need(all(binding[k] == case[k] for k in ("patient_id", "patient_group", "role")), "binding_role")
    count = sum(len(s["objects"]) for s in case["series"])
    need(count == binding["total_objects"] and sum(o["bytes"] for s in case["series"] for o in s["objects"]) == binding["total_bytes"], "exact_object_extent")
    declared = {str(repository_root / o["path"]): {k: o[k] for k in ("bytes", "sha256", "expected_md5", "series_uuid")} for o in binding["objects"]}
    nested = {o["path"]: {**{k: o[k] for k in ("bytes", "sha256", "expected_md5")}, "series_uuid": s["series_uuid"]} for s in case["series"] for o in s["objects"]}
    need(len(nested) == len(declared) == len(binding["objects"]) == count and nested == declared, "exact_nested_object_binding")
    source_series = {s["crdc_series_uuid"]: s for s in binding["source_series"]}
    need(len(source_series) == len(binding["source_series"]) == len(case["series"]) and
         {s["series_uuid"] for s in case["series"]} == set(source_series), "exact_source_series")
    for series in case["series"]:
        original = source_series[series["series_uuid"]]
        need(original["PatientID"] == case["patient_id"] and
             all(series[key] == original[key] for key in ("SeriesInstanceUID", "StudyInstanceUID", "Modality")) and
             series.get("source_description") == original["SeriesDescription"] and
             len(series["objects"]) == int(original["instanceCount"]), "source_series_identity")
    return {"source_binding_sha256": parent["sha256"], "cohort_sha256": COHORT_SHA, "role": case["role"], "patient_group": case["patient_group"]}


def validate_header_binding(headers, case, case_sha256, *, public_only=False):
    """Reject a different case, source set or unresolved frame before decoding."""
    validate_public_scope(case, public_only=public_only)
    if public_only: need(headers.get("public_only") is True, "public_header_scope_required")
    need(headers["case_sha256"] == case_sha256 and headers["status"] == "header_geometry_and_ancestry_projected" and
         headers["reused_converter_sha256"] == CORE_SHA, "completed_header_case")
    rows = {s["kind"]: s for s in headers["series"]}
    need(len(rows) == len(headers["series"]) == len(case["series"]) and
         set(rows) == {s["kind"] for s in case["series"]} == (PUBLIC_SERIES_KINDS if public_only else
             {"structural_t1ce", "whole_tumor", "ventricles", "cerebrum"}), "crop_series_kinds")
    for series in case["series"]:
        row = rows[series["kind"]]
        need(row["source_objects"] == series["objects"] and row["series_instance_uid"] == series["SeriesInstanceUID"] and
             row["modality"] == series["Modality"], "header_source_binding")
        if series["Modality"] == "SEG":
            link = pairing(rows["structural_t1ce"], row)
            need(link["frame_uid_matches"] and not link["unmatched_explicit_SOP_refs"], "SEG_correspondence_unresolved")
    return rows


def run_headers(args):
    repository_root = Path(args.repository_root or ROOT).resolve()
    public_only = bool(getattr(args, "public_only", False))
    started = time.monotonic(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    result = {"status": "failed", "phase": args.phase, "role": None if public_only else "TRAIN", "training_admitted": False,
              "planner_inputs_admitted": False, "actor_inputs_admitted": False, "anatomy_and_coverage_review": "unreviewed",
              "source_file_returned_bytes": 0, "objects_verified": 0, "pixel_decode_calls": 0}
    try:
        raw = args.case.read_bytes(); need(digest(raw) == args.case_sha256, "case_manifest_digest")
        case = json.loads(raw); result["case_sha256"] = args.case_sha256; result.update(patient_id=case["patient_id"], patient_group=case["patient_group"])
        result["immutable_role_binding"] = validate_case(case, repository_root, public_only=public_only)
        result["role"] = case["role"]
        if public_only: result.update(public_only=True, optimizer_updates_performed=0, private_reference_loaded=False)
        result["case_uncertainties"] = case.get("case_uncertainties", [])
        core = source_module(repository_root / "scripts/convert_remind_development.py", CORE_SHA)
        result.update(project(case, core, result, public_only=public_only))
        result.update(status="header_geometry_and_ancestry_projected", native_conversion_performed=False,
                      missing_anatomy=["patient-specific vessels", "functional anatomy"],
                      target_meaning="Whole-tumor source annotation; not a prescribed resection target.",
                      negative_label_semantics="Outside source SEG grid is unknown. Inside-grid zero is source-label absence, not evidence of all critical anatomy absent.")
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        result.update(elapsed_seconds=time.monotonic()-started,
                      peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024)),
                      executing_script_sha256=digest(Path(__file__).read_bytes()), reused_converter_sha256=CORE_SHA)
        save(output / "result.json", result)
    print(json.dumps({k: result.get(k) for k in ("status", "objects_verified", "pixel_decode_calls", "MR_voxels", "estimated_native_peak_bytes", "elapsed_seconds", "peak_RSS_bytes")}))
    return 0 if result["status"] != "failed" else 1


def peak(): return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024))


def guard(started):
    need(peak() <= MAX_RSS and time.monotonic()-started < 110, "resource_bound")


def corners(shape):
    import numpy as np
    return np.asarray(list(itertools.product(*[(-.5, int(n)-.5) for n in shape])))


def corner_error(a, b, shape):
    import numpy as np
    points = np.c_[corners(shape), np.ones(8)]
    return float(np.linalg.norm((points @ (np.asarray(a)-np.asarray(b)).T)[:, :3], axis=1).max())


def grid_quality(affine, shape):
    import numpy as np
    a = np.asarray(affine, dtype=float); basis = a[:3, :3]; spacing = np.linalg.norm(basis, axis=0)
    directions = basis/spacing; u, _, vt = np.linalg.svd(directions); polar = a.copy(); polar[:3, :3] = (u@vt)*spacing
    return {"spacing_mm": spacing.tolist(), "max_normalized_Gram_error": float(np.abs(directions.T@directions-np.eye(3)).max()),
            "polar_corner_difference_mm": corner_error(a, polar, shape), "polar_transform_applied": False}


def integer_map(source_affine, source_shape, target_affine):
    """Declare a near-coincident index translation; retain its physical error."""
    import numpy as np
    source, target = np.asarray(source_affine), np.asarray(target_affine)
    mapping = np.linalg.inv(target) @ source
    rounded = np.eye(4); rounded[:3, 3] = np.rint(mapping[:3, 3])
    error = corner_error(source, target @ rounded, source_shape)
    need(np.allclose(mapping[:3, :3], np.eye(3), rtol=0, atol=2e-6) and error <= MAX_CORNER_MM, "not_nearly_coincident_integer_grid")
    return rounded[:3, 3].astype(int), {"source_index_to_target_index": mapping.tolist(), "declared_integer_translation": rounded[:3, 3].astype(int).tolist(),
                                      "maximum_world_corner_difference_mm": error, "bound_mm": MAX_CORNER_MM,
                                      "interpolation": "none; explicit nearly coincident integer reindex", "registration_estimated": False}


def raw_mr_samples(ds):
    import numpy as np
    need(str(ds.file_meta.TransferSyntaxUID) in ("1.2.840.10008.1.2", "1.2.840.10008.1.2.1") and
         int(ds.BitsAllocated) == 16 and 1 <= int(ds.BitsStored) <= 16 and int(ds.HighBit) == int(ds.BitsStored)-1 and
         int(ds.PixelRepresentation) in (0, 1) and int(ds.SamplesPerPixel) == 1 and str(ds.PhotometricInterpretation) == "MONOCHROME2", "MR_encoding")
    need(type(ds.PixelData) is bytes and len(ds.PixelData) == int(ds.Rows)*int(ds.Columns)*2, "MR_payload_bytes")
    bits = int(ds.BitsStored); words = np.frombuffer(ds.PixelData, dtype="<u2").astype(np.int32) & ((1 << bits)-1)
    if int(ds.PixelRepresentation): words = words - ((words & (1 << (bits-1))) != 0)*(1 << bits)
    return words.reshape(int(ds.Rows), int(ds.Columns)).astype(np.float32)*float(getattr(ds, "RescaleSlope", 1))+float(getattr(ds, "RescaleIntercept", 0))


def raw_seg_samples(ds):
    import numpy as np
    need(str(ds.file_meta.TransferSyntaxUID) in ("1.2.840.10008.1.2", "1.2.840.10008.1.2.1") and int(ds.BitsAllocated) == 1 and
         int(ds.BitsStored) == 1 and int(ds.HighBit) == 0 and int(ds.PixelRepresentation) == 0 and type(ds.PixelData) is bytes, "SEG_bit_encoding")
    count = int(ds.NumberOfFrames)*int(ds.Rows)*int(ds.Columns)
    need((count+7)//8 <= len(ds.PixelData) <= (count+7)//8+1, "SEG_bit_extent")
    return np.unpackbits(np.frombuffer(ds.PixelData, dtype=np.uint8), bitorder="little", count=count).reshape(int(ds.NumberOfFrames), int(ds.Rows), int(ds.Columns))


def place(data, offset, shape):
    import numpy as np
    need(all(0 <= int(o) and int(o)+n <= t for o, n, t in zip(offset, data.shape, shape)), "source_domain_outside_public_crop")
    slices = tuple(slice(int(o), int(o)+n) for o, n in zip(offset, data.shape))
    values = np.zeros(shape, np.uint8); domain = np.zeros(shape, np.uint8)
    values[slices] = data; domain[slices] = 1
    return values, domain


def file_sha(path):
    hashed = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""): hashed.update(block)
    return hashed.hexdigest()


def png(path, rgb):
    import numpy as np
    a = np.asarray(rgb, dtype=np.uint8); h, w, _ = a.shape
    def chunk(kind, data): return struct.pack(">I", len(data))+kind+data+struct.pack(">I", zlib.crc32(kind+data)&0xffffffff)
    raw = b"".join(b"\0"+a[row].tobytes() for row in range(h))
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))+chunk(b"IDAT", zlib.compress(raw))+chunk(b"IEND", b""))


def overlays(image, masks, output, sample_point, *, public_only=False):
    import numpy as np
    sample = image[::4, ::4, ::4]; lo, hi = np.percentile(sample, [1, 99.5]); need(hi > lo, "degenerate_display_window")
    records = []
    for axis in range(3):
        index = int(sample_point[axis]); plane = np.take(image, index, axis=axis).T
        gray = np.clip((plane-lo)/(hi-lo), 0, 1); rgb = np.repeat(gray[:, :, None], 3, axis=2)
        for kind, color in (("cerebrum", [0, 1, 0]), ("whole_tumor", [1, .15, 0]), ("ventricles", [0, .4, 1])):
            if public_only and kind == "ventricles": continue
            mask = np.take(masks[kind], index, axis=axis).T.astype(bool)
            if kind == "cerebrum": mask &= ~(np.roll(mask, 1, 0)&np.roll(mask, -1, 0)&np.roll(mask, 1, 1)&np.roll(mask, -1, 1))
            rgb[mask] = .35*rgb[mask]+.65*np.asarray(color)
        prefix = "public-qc" if public_only else "evaluation-only"
        path = output/f"{prefix}-axis{axis}-index{index}.png"; png(path, rgb*255)
        records.append({"file": path.name, "axis": axis, "index": index, "orientation": "array-plane transpose; use retained RAS affine, not a clinical radiological convention", "available_to_actor": False})
    return {"images": records, "window": [float(lo), float(hi)], "selection": "source whole-tumor grid centre for QC only", "colors": {"cerebrum_boundary": "green", "whole_tumor": "red", **({} if public_only else {"ventricles": "blue"})}}


def public_target_support_relation(target, support):
    """Public source consistency; never clip the target or repair the support."""
    import numpy as np
    total = int(np.count_nonzero(target))
    outside = int(np.count_nonzero((target != 0) & (support == 0)))
    return {"whole_tumor_positive_voxels": total, "whole_tumor_positive_outside_public_support": outside,
            "outside_fraction": outside/total if total else None, "target_clipped": False, "support_filled_or_modified": False,
            "task_interpretation": "Source whole-tumor annotation is not prescribed resection. Any support-limited target or expanded removable domain must be an explicit task condition; source arrays remain unchanged."}


def mr_sampling_crop(mr_geometry, public_geometry):
    """Choose native MRI cells intersecting the public support grid world bounds."""
    import numpy as np
    mr = np.asarray(mr_geometry["affine_xyz_to_ras_mm"], dtype=float)
    public = np.asarray(public_geometry["affine_xyz_to_ras_mm"], dtype=float)
    full = np.asarray(mr_geometry["shape_xyz"], dtype=int)
    mapped = (np.c_[corners(public_geometry["shape_xyz"]), np.ones(8)] @ (np.linalg.inv(mr) @ public).T)[:, :3]
    low = np.floor(mapped.min(0)+.5).astype(int)
    high = np.ceil(mapped.max(0)+.5).astype(int)
    start, stop = np.maximum(low, 0), np.minimum(high, full)
    clipped_to_mr = bool(np.any(start != low) or np.any(stop != high))
    # MRI samples keep their native indices. Only this explicit derived planning
    # affine receives the sub-micron orthogonal roundoff adjustment.
    spacing = np.linalg.norm(mr[:3, :3], axis=0)
    u, _, vt = np.linalg.svd(mr[:3, :3]/spacing)
    inset = 0
    while True:
        shape = stop-start
        need(np.all(shape >= 3), "no_contained_public_MR_crop")
        native = mr.copy(); native[:3, 3] = (mr @ np.r_[start, 1])[:3]
        derived = native.copy(); derived[:3, :3] = (u@vt)*spacing
        source_corners = (np.c_[corners(shape), np.ones(8)] @ (np.linalg.inv(public) @ derived).T)[:, :3]
        if np.all(source_corners >= -.5) and np.all(source_corners < np.asarray(public_geometry["shape_xyz"])-.5): break
        # A deliberately conservative box; no anatomy-positive extent, target,
        # or private reference influences this uniform one-cell inset.
        start += 1; stop -= 1; inset += 1
    need(int(np.prod(shape)) <= 32000000, "public_MR_crop_extent")
    error = corner_error(native, derived, shape)
    need(error <= MAX_CORNER_MM, "MR_orthogonal_roundoff_exceeds_bound")
    return start, tuple(int(n) for n in shape), native, derived, {
        "selection": "public cerebrum grid world bounds intersect MRI, then uniformly inset until every output cell corner is inside public source coverage",
        "uses_private_labels": False, "uses_label_positive_extent": False,
        "public_cell_bounds_in_MR_index": {"minimum": mapped.min(0).tolist(), "maximum": mapped.max(0).tolist()},
        "unclipped_MR_start": low.tolist(), "unclipped_MR_stop": high.tolist(),
        "clipped_to_MR_coverage": clipped_to_mr,
        "uniform_inset_MR_cells_per_face": inset, "all_output_cell_corners_inside_public_source_domain": True,
        "original_MR_sampling_mm": spacing.tolist(), "image_interpolation": "none",
        "planning_affine_method": "explicit polar orthogonal roundoff projection; exact source crop affine retained",
        "maximum_world_corner_difference_mm": error, "bound_mm": MAX_CORNER_MM}


def resample_binary_nn(source, source_affine, target_affine, target_shape, checkpoint=None):
    """Nearest source voxel centre, with half-open native cell-domain coverage.

    Process one target XY plane at a time. Ties go toward the positive source
    index (floor(x+.5)); no label-positive or private information sets the grid.
    """
    import numpy as np
    need(source.ndim == 3 and source.dtype == np.uint8 and bool(np.all(source <= 1)), "binary_source_required")
    transform = np.linalg.inv(np.asarray(source_affine)) @ np.asarray(target_affine)
    x, y = np.indices(target_shape[:2], dtype=np.float64)
    base = transform[:3, 0, None, None]*x + transform[:3, 1, None, None]*y + transform[:3, 3, None, None]
    values = np.zeros(target_shape, np.uint8); domain = np.zeros(target_shape, np.uint8)
    bounds = np.asarray(source.shape)[:, None, None]
    for z in range(target_shape[2]):
        if checkpoint: checkpoint()
        coordinates = base + transform[:3, 2, None, None]*z
        indices = np.floor(coordinates+.5).astype(np.int64)
        valid = np.all((coordinates >= -.5) & (coordinates < bounds-.5), axis=0)
        domain[:, :, z] = valid
        points = indices[:, valid]
        values[:, :, z][valid] = source[points[0], points[1], points[2]]
    # Cropping losses and sampling losses differ. Count source-positive centres
    # outside output cell coverage separately; NN may still miss tiny structures.
    reverse = np.linalg.inv(np.asarray(target_affine)) @ np.asarray(source_affine)
    outside = 0
    for z in range(source.shape[2]):
        if checkpoint: checkpoint()
        sx, sy = np.nonzero(source[:, :, z])
        mapped = reverse[:3, 0, None]*sx + reverse[:3, 1, None]*sy + (reverse[:3, 2]*z+reverse[:3, 3])[:, None]
        inside = np.all((mapped >= -.5) & (mapped < np.asarray(target_shape)[:, None]-.5), axis=0)
        outside += int(np.count_nonzero(~inside))
    return values, domain, {"method": "nearest source centre; floor(index+.5), half-open source cell domain",
        "target_index_to_source_index": transform.tolist(), "source_index_to_target_index": reverse.tolist(),
        "source_positive_centres_outside_target_grid": outside,
        "source_positive_voxels": int(source.sum()), "resampled_positive_voxels": int(values.sum()),
        "resampled_source_domain_voxels": int(domain.sum()), "outside_source_domain": "unknown",
        "source_grid_unchanged": True, "resampling_can_omit_subvoxel_labels": True,
        "source_positive_volume_mm3": float(source.sum()*abs(np.linalg.det(np.asarray(source_affine)[:3, :3]))),
        "resampled_positive_volume_mm3": float(values.sum()*abs(np.linalg.det(np.asarray(target_affine)[:3, :3])))}


def run_crop(args):
    repository_root = Path(args.repository_root or ROOT).resolve()
    public_only = bool(getattr(args, "public_only", False))
    import numpy as np
    started = time.monotonic(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "role": None if public_only else "TRAIN", "source_file_returned_bytes": 0, "objects_verified": 0,
              "training_admitted": False, "actor_inputs_admitted": False, "clinical_validity": "unreviewed", "artifacts": {}}
    try:
        raw = args.case.read_bytes(); need(digest(raw) == args.case_sha256, "exact_case"); case = json.loads(raw)
        report["immutable_role_binding"] = validate_case(case, repository_root, public_only=public_only)
        report.update(patient_id=case["patient_id"], patient_group=case["patient_group"], role=case["role"])
        if public_only: report.update(public_only=True, optimizer_updates_performed=0, private_reference_loaded=False)
        raw = args.headers.read_bytes(); need(digest(raw) == args.headers_sha256, "exact_header_snapshot"); headers = json.loads(raw)
        by_kind = validate_header_binding(headers, case, args.case_sha256, public_only=public_only)
        series = {s["kind"]: s for s in case["series"]}
        core = source_module(repository_root / "scripts/convert_remind_development.py", CORE_SHA)
        mr = by_kind["structural_t1ce"]; public = by_kind["cerebrum"]
        resample_labels = args.phase == "crop-mr"
        mr_affine = np.asarray(mr["geometry"]["affine_xyz_to_ras_mm"])
        if resample_labels:
            start, shape, native_crop_affine, derived, crop_map = mr_sampling_crop(mr["geometry"], public["geometry"])
        else:
            shape = tuple(public["geometry"]["shape_xyz"]); need(np.prod(shape) <= 32000000, "planning_voxel_cap")
            derived = np.asarray(public["geometry"]["affine_xyz_to_ras_mm"])
            start, crop_map = integer_map(derived, shape, mr_affine)
            need(all(0 <= int(o) and int(o)+n <= full for o, n, full in zip(start, shape, mr["geometry"]["shape_xyz"])), "public_crop_outside_MR")
            native_crop_affine = mr_affine.copy(); native_crop_affine[:3, 3] = (mr_affine @ np.r_[start, 1])[:3]
        need(corner_error(native_crop_affine, derived, shape) <= MAX_CORNER_MM, "derived_grid_world_difference")
        selected_indices = mr["geometry"]["sorted_source_indices"][int(start[2]):int(start[2])+shape[2]]
        estimated_peak = sum(series["structural_t1ce"]["objects"][i]["bytes"] for i in selected_indices) + int(np.prod(mr["geometry"]["shape_xyz"][:2]))*shape[2]*16 + 256*1024**2
        report.update(estimated_crop_peak_bytes=estimated_peak, native_peak_estimate_not_measurement=True)
        need(estimated_peak <= MAX_RSS, "estimated_crop_resource_bound")
        datasets = []
        for i in selected_indices:
            guard(started); ds = read_object(series["structural_t1ce"]["objects"][i], pixels=True, accounting=report)
            identity(ds, case, series["structural_t1ce"]); need(str(ds.SOPInstanceUID) == mr["sop_instance_uids"][i], "MR_header_SOP_binding")
            ds.pixel_array_options(index=None, raw=True, decoding_plugin="", use_v2_backend=False); datasets.append(ds)
        data, fitted, details = core.convert_mr(datasets)
        expected_subset = mr_affine.copy(); expected_subset[:3, 3] = (mr_affine @ [0, 0, int(start[2]), 1])[:3]
        fitted_error = corner_error(fitted, expected_subset, data.shape); need(fitted_error <= MAX_CORNER_MM, "subset_fit_vs_full_header_grid")
        for z, ds in enumerate(datasets): need(np.array_equal(data[:, :, z].T, raw_mr_samples(ds)), "independent_MR_source_sample_mismatch")
        crop = data[int(start[0]):int(start[0])+shape[0], int(start[1]):int(start[1])+shape[1], :].copy()
        del ds, datasets, data; gc.collect(); guard(started)
        np.save(output/"MR_native_crop.npy", crop, allow_pickle=False)
        report.update(header_snapshot_sha256=args.headers_sha256, case_sha256=args.case_sha256, public_crop_start_MR=start.tolist(), shape_xyz=list(shape),
            MR_native_crop_affine_ras_mm=native_crop_affine.tolist(), planning_derived_affine_ras_mm=derived.tolist(),
            planning_grid_source="original MRI sampling cropped using public cerebrum world bounds; explicit bounded orthogonal roundoff" if resample_labels else "public automatic cerebrum SEG native grid", crop_selection_uses_private_ventricle_labels=False,
            labels_resampled=resample_labels,
            reindex_policy={"source_MR_samples_preserved": True, "image_interpolation": "none", "original_MR_affine_overwritten": False,
                "maximum_crop_world_difference_mm": corner_error(native_crop_affine, derived, shape), "maximum_bound_mm": MAX_CORNER_MM,
                "subset_converter_fit_difference_mm": fitted_error, "source_MR_grid": grid_quality(native_crop_affine, shape), "derived_grid": grid_quality(derived, shape)},
            MR_pixel_source_objects=len(selected_indices), MR_header_only_objects_outside_selected_z=len(series["structural_t1ce"]["objects"])-len(selected_indices),
            full_MR_native_array_written=False, selected_MR_pixels_match_independent_raw_bytes=True, source_cropping_map=crop_map,
            case_uncertainties=case.get("case_uncertainties", []), annotations={})
        masks = {}; tumor_point = None
        label_kinds = ("cerebrum", "whole_tumor") if public_only else ("cerebrum", "whole_tumor", "ventricles")
        for kind in label_kinds:
            guard(started); source = series[kind]; saved = by_kind[kind]
            need(saved["geometry"]["frame_of_reference_uid"] == mr["geometry"]["frame_of_reference_uid"], "SEG_cross_frame")
            ds = read_object(source["objects"][0], pixels=True, accounting=report); identity(ds, case, source)
            need(str(ds.SOPInstanceUID) == saved["sop_instance_uids"][0], "SEG_header_SOP_binding")
            ds.pixel_array_options(index=None, raw=True, decoding_plugin="", use_v2_backend=False)
            native, affine, detail = core.convert_seg(ds)
            expected = raw_seg_samples(ds)[saved["geometry"]["sorted_source_indices"]].transpose(2, 1, 0)
            need(np.array_equal(native, expected), "independent_SEG_bit_sample_mismatch")
            need(corner_error(affine, saved["geometry"]["affine_xyz_to_ras_mm"], native.shape) < 1e-9, "SEG_saved_geometry")
            if resample_labels:
                placed, domain, placement = resample_binary_nn(native, affine, derived, shape, lambda: guard(started))
                if kind == "cerebrum": need(bool(np.all(domain)), "public_support_source_domain_incomplete")
            else:
                offset, placement = integer_map(affine, native.shape, derived)
                placed, domain = place(native, offset, shape)
            masks[kind] = placed
            np.save(output/f"{kind}_source_label.npy", placed, allow_pickle=False)
            if kind != "cerebrum" or resample_labels: np.save(output/f"{kind}_source_grid_domain.npy", domain, allow_pickle=False)
            report["annotations"][kind] = {"source_native_affine_ras_mm": affine.tolist(), "source_native_shape": list(native.shape),
                "source_native_geometry_retained": True, "placement": placement, "source_positive_voxels": int(native.sum()),
                "placed_positive_voxels": int(placed.sum()), "native_grid_domain_voxels": int(native.size),
                "placed_source_domain_voxels": int(domain.sum()),
                "outside_native_grid": "unknown; zero padded mask must be paired with domain", "source_samples_equal": True,
                "ancestry": saved["ancestry"], "correspondence": saved["alignment"], "annotation_accuracy": "unreviewed"}
            if kind == "whole_tumor":
                if resample_labels:
                    centre = (np.linalg.inv(derived) @ affine @ np.r_[(np.asarray(native.shape)-1)/2, 1])[:3]
                    tumor_point = np.clip(np.floor(centre+.5).astype(int), 0, np.asarray(shape)-1)
                else: tumor_point = offset+(np.asarray(native.shape)-1)//2
            del native, expected, domain, ds; gc.collect()
        if not public_only:
            report["public_support_private_ventricle_relation"] = {"ventricle_positive_voxels": int(masks["ventricles"].sum()),
                "ventricle_positive_in_public_support_zero": int(np.count_nonzero(masks["ventricles"] & (masks["cerebrum"] == 0))),
                "ventricle_positive_in_public_support_one": int(np.count_nonzero(masks["ventricles"] & masks["cerebrum"])),
                "public_support_zero_voxels": int(np.count_nonzero(masks["cerebrum"] == 0)),
                "support_filled_or_modified": False, "information_boundary": "The automatic cerebrum mask is explicitly supplied public estimated support. Its holes/zeros may reveal anatomy; retain and declare this cue, never fill or hide it to manufacture difficulty."}
        report["public_target_support_relation"] = public_target_support_relation(masks["whole_tumor"], masks["cerebrum"])
        if not public_only:
            report["evaluation_only_target_ventricle_relation"] = {"overlap_positive_voxels": int(np.count_nonzero(masks["whole_tumor"] & masks["ventricles"])),
                "interpretation": "Source-reference overlap can contradict simultaneous perfect target-removal and zero-contact hard goals; it is not clinical truth."}
        report["public_overlays" if public_only else "evaluation_only_overlays"] = overlays(crop, masks, output, tumor_point, public_only=public_only)
        for path in sorted(output.iterdir()):
            if path.suffix in (".npy", ".png"): report["artifacts"][path.name] = {"bytes": path.stat().st_size, "sha256": file_sha(path)}
        guard(started)
        report.update(status="public_crop_source_samples_and_label_grids_converted_anatomy_unreviewed",
            explicit_source_SOP_links_present={kind: by_kind[kind]["alignment"]["explicit_SOP_refs_present"] for kind in label_kinds},
            whole_tumor_is_prescribed_resection_target=False, anatomical_coverage="Only source annotation domains; other anatomy remains unknown", registration_performed=False,
            declared_conditions={"support": "supplied automatic Brainlab cerebrum estimate", "whole_tumor": "manual source annotation; any task target use must be explicit", **({} if public_only else {"ventricles": "automatic source reference, not independent manual truth"}), "vessels_and_function": "unknown"})
    except BaseException as error: report.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        report.update(elapsed_seconds=time.monotonic()-started, peak_RSS_bytes=peak(), executing_script_sha256=file_sha(__file__), reused_converter_sha256=CORE_SHA,
                      RSS_cap_bytes=MAX_RSS, output_bytes_before_receipt=sum(p.stat().st_size for p in output.iterdir() if p.is_file()))
        save(output/"conversion-result.json", report)
    print(json.dumps({k: report.get(k) for k in ("status", "objects_verified", "source_file_returned_bytes", "shape_xyz", "elapsed_seconds", "peak_RSS_bytes")}))
    return 0 if report["status"] != "failed" else 1
