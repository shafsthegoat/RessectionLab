"""Local imaging intake and reviewable rigid registration."""

from pathlib import Path
import json
import re
import numpy as np
from .patient_anatomy import affine, world, reconstruction_lock, file_hash


MRI_SEQUENCES = {"T1", "T1CE", "T2", "FLAIR", "MRA", "TOF"}
CT_SEQUENCES = {"CT", "CTA"}
IMAGING_MODALITIES = MRI_SEQUENCES | CT_SEQUENCES
OPERATIVE_LABELS = {
    "tumor": ("tumor", "target", "#c56df1"),
    "brain": ("brain", "context", "#91c4b6"),
    "bone": ("skull", "context", "#d8e7f8"),
    "skull": ("skull", "context", "#d8e7f8"),
    "vessels": ("vessels", "critical", "#ef5679"),
    "vessel": ("vessels", "critical", "#ef5679"),
    "context": ("context", "context", "#97bdcf"),
}


def _validated_role_map(role_map, present):
    if not isinstance(role_map, dict):
        raise ValueError("roleMap must map numeric label values to operative roles")
    result = {}
    for raw_value, raw_spec in role_map.items():
        if isinstance(raw_value, bool) or not str(raw_value).isdigit():
            raise ValueError("roleMap keys must be positive integer label values")
        value = int(raw_value)
        if value < 1 or value > 65535 or value in result:
            raise ValueError("roleMap contains a duplicate or out-of-range label value")
        if isinstance(raw_spec, str):
            spec = {"kind": raw_spec}
        elif isinstance(raw_spec, dict) and set(raw_spec) <= {"kind", "label", "color"}:
            spec = raw_spec
        else:
            raise ValueError(f"Invalid roleMap entry for label {value}")
        key = spec.get("kind")
        if not isinstance(key, str) or key not in OPERATIVE_LABELS:
            raise ValueError(f"Unsupported operative role {key!r} for label {value}")
        kind, role, color = OPERATIVE_LABELS[key]
        display_label = spec.get("label")
        if display_label is not None and (not isinstance(display_label, str) or not display_label.strip() or len(display_label) > 128):
            raise ValueError(f"Invalid roleMap label for value {value}")
        if "color" in spec:
            color = spec["color"]
            if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                raise ValueError(f"Invalid roleMap color for value {value}")
        result[value] = {"kind": kind, "role": role, "color": color, "label": display_label}
    if set(result) != present:
        missing = sorted(present - set(result))
        extra = sorted(set(result) - present)
        raise ValueError(f"roleMap must cover every nonzero label exactly; missing={missing}, absent={extra}")
    return result


def imaging_modality(modality, *, dicom_modality=None):
    """Require a semantic sequence; DICOM's MR is only an acquisition family."""
    source = str(dicom_modality).strip().upper() if dicom_modality is not None else None
    if source is not None and source not in {"MR", "CT"}:
        raise ValueError(f"Unsupported DICOM Modality {source!r}; import an MR or CT image series")
    if modality is None and source == "CT":
        return "CT"
    value = str(modality).strip().upper() if modality is not None else ""
    if value not in IMAGING_MODALITIES:
        raise ValueError(
            f"Unsupported imaging sequence {modality!r}; choose T1, T1CE, T2, FLAIR, CT, MRA, TOF or CTA explicitly"
        )
    if source == "MR" and value not in MRI_SEQUENCES:
        raise ValueError(f"DICOM MR cannot be imported as {value}; choose an MRI sequence")
    if source == "CT" and value not in CT_SEQUENCES:
        raise ValueError(f"DICOM CT cannot be imported as {value}; choose CT or CTA")
    return value


def import_nifti_series(collection, path, *, series_id, modality, timepoint="preoperative", patient_id=None):
    """Add a typed NIfTI source without guessing sequence from its filename."""
    return collection.add_series(
        path, series_id=series_id, modality=imaging_modality(modality),
        timepoint=timepoint, patient_id=patient_id,
    )


def import_dicom(collection, directory, *, modality=None, sequence_by_series=None, timepoint="preoperative"):
    import pydicom
    import SimpleITK as sitk

    directory = Path(directory)
    ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)) or []
    if not ids:
        raise ValueError("No readable DICOM image series")
    if sequence_by_series is not None and not isinstance(sequence_by_series, dict):
        raise ValueError("sequenceBySeries must map DICOM SeriesInstanceUID to an explicit sequence")
    if modality is not None and sequence_by_series:
        raise ValueError("Use either modality for one DICOM series or sequenceBySeries for multiple series")
    if modality is not None and len(ids) != 1:
        raise ValueError("Multiple DICOM series require sequenceBySeries keyed by SeriesInstanceUID")
    if sequence_by_series is not None and set(sequence_by_series) != set(ids):
        raise ValueError("sequenceBySeries must identify every DICOM SeriesInstanceUID exactly once")
    prepared = []
    for uid in ids:
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", uid):
            raise ValueError("Invalid DICOM SeriesInstanceUID")
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), uid)
        headers = [pydicom.dcmread(p, stop_before_pixels=True) for p in files]
        if not headers:
            raise ValueError(f"DICOM series {uid} contains no readable images")
        patients = {str(getattr(h, "PatientID", "")) for h in headers}
        if (
            len(patients) != 1
            or not next(iter(patients))
            or next(iter(patients)) != collection.document["patient_id"]
        ):
            raise ValueError(
                "DICOM patient ID must match the collection; explicit deidentified association required"
            )
        source_modalities = {str(getattr(h, "Modality", "")).upper() for h in headers}
        if len(source_modalities) != 1:
            raise ValueError(f"DICOM series {uid} mixes acquisition modalities")
        source_modality = next(iter(source_modalities))
        if any(str(getattr(h, "SeriesInstanceUID", uid)) != uid for h in headers):
            raise ValueError(f"DICOM series {uid} contains a mismatched SeriesInstanceUID")
        if len({str(getattr(h, "FrameOfReferenceUID", "")) for h in headers}) != 1:
            raise ValueError(f"DICOM series {uid} mixes frames of reference")
        if any(row["id"] == uid for row in collection.document["series"]):
            raise ValueError(f"DICOM series {uid} is already imported")
        sequence = imaging_modality(
            sequence_by_series[uid] if sequence_by_series is not None else modality,
            dicom_modality=source_modality,
        )
        prepared.append((uid, files, headers[0], source_modality, sequence))
    imported = []
    for uid, files, header, source_modality, sequence in prepared:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        image = reader.Execute()
        if image.GetDimension() != 3:
            raise ValueError("Select a single 3D DICOM series")
        converted = collection.root / "dicom-converted"
        converted.mkdir(exist_ok=True)
        out = converted / (uid + ".nii.gz")
        sitk.WriteImage(image, str(out))
        row = collection.add_series(
            out,
            series_id=uid,
            modality=sequence,
            timepoint=timepoint,
        )
        row["label"] = str(getattr(header, "SeriesDescription", "")) or f"{row['modality']} series {len(imported) + 1}"
        row["dicom_modality"] = source_modality
        row["dicom_originals"] = [collection.original(p) for p in files]
        row["acquisition_datetime"] = (
            str(getattr(header, "AcquisitionDateTime", "")) or None
        )
        row["dicom_frame_uid"] = (
            str(getattr(header, "FrameOfReferenceUID", "")) or None
        )
        imported.append(row)
    return imported


def rigid_registration(collection, moving_id, *, cancelled=lambda: False):
    import SimpleITK as sitk

    moving = collection.series(moving_id)
    fixed = collection.series(collection.document["reference_series"])
    if moving["timepoint"] != fixed["timepoint"]:
        raise ValueError(
            "Different timepoints require separate registration review; automatic rigid fusion withheld"
        )
    for row in [moving, fixed]:
        if (
            file_hash(collection.path(row["original"]["path"]))
            != row["original"]["sha256"]
        ):
            raise ValueError("Original source changed before registration")
    with reconstruction_lock():
        f = sitk.ReadImage(
            str(collection.path(fixed["original"]["path"])), sitk.sitkFloat32
        )
        m = sitk.ReadImage(
            str(collection.path(moving["original"]["path"])), sitk.sitkFloat32
        )
        initial = sitk.CenteredTransformInitializer(
            f,
            m,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(50)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(0.02, 20261014)
        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetOptimizerAsGradientDescent(1.0, 100, 1e-6, 10)
        registration.SetOptimizerScalesFromPhysicalShift()
        registration.SetShrinkFactorsPerLevel([4, 2, 1])
        registration.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration.SetInitialTransform(initial, inPlace=False)
        registration.AddCommand(
            sitk.sitkIterationEvent,
            lambda: registration.StopRegistration() if cancelled() else None,
        )
        result = registration.Execute(f, m)
        if cancelled():
            raise InterruptedError("Registration cancelled")
        # ITK transform samples fixed -> moving in LPS. Store moving -> fixed RAS.
        origin = np.array(result.TransformPoint((0.0, 0.0, 0.0)))
        matrix = np.eye(4)
        matrix[:3, 3] = origin
        for axis in range(3):
            matrix[:3, axis] = (
                np.array(result.TransformPoint(tuple(np.eye(3)[axis]))) - origin
            )
        lps = np.diag([-1.0, -1.0, 1.0, 1.0])
        matrix = lps @ np.linalg.inv(matrix) @ lps
        collection.set_registration(
            moving_id,
            matrix,
            status="unreviewed",
            method="Mattes MI rigid; independent landmarks pending",
        )
        return {
            "matrix": matrix.tolist(),
            "metric": float(registration.GetMetricValue()),
            "status": "unreviewed",
        }


def review_registration(
    collection, series_id, moving_points, reference_points, reviewer
):
    if (
        not reviewer.strip()
        or len(moving_points) < 3
        or len(moving_points) != len(reference_points)
    ):
        raise ValueError(
            "Reviewer and at least three paired anatomical landmarks required"
        )
    m = np.asarray(moving_points, float)
    r = np.asarray(reference_points, float)
    if (
        m.shape != r.shape
        or m.ndim != 2
        or m.shape[1] != 3
        or not np.isfinite(m).all()
        or not np.isfinite(r).all()
    ):
        raise ValueError("Invalid landmarks")
    if (
        np.linalg.matrix_rank(m - m.mean(axis=0)) < 2
        or np.linalg.matrix_rank(r - r.mean(axis=0)) < 2
    ):
        raise ValueError("Landmarks must not be coincident or collinear")
    t = collection.document["transforms"][series_id]
    errors = np.linalg.norm(world(affine(t["matrix"]), m) - r, axis=1)
    if errors.max() > 2.0:
        raise ValueError("Landmark residual exceeds 2 mm; improve alignment before accepting review")
    collection.set_registration(
        series_id,
        t["matrix"],
        status="reviewed",
        method=t["method"],
        uncertainty_mm=float(errors.max()),
        landmarks={
            "moving": m.tolist(),
            "reference": r.tolist(),
            "errors_mm": errors.tolist(),
            "reviewer": reviewer,
            "meaning": "sampled landmark residuals, not a whole-image error bound",
        },
    )
    for key in ("direction", "reference_series", "parent_series", "source_to_parent", "algorithm_version", "input_key", "quality", "transform_digest"):
        if key in t:
            collection.document["transforms"][series_id][key] = t[key]
    return errors.tolist()


def import_anatomy_derivative(
    collection, path, *, source_series, kind, label, role="context", role_map=None
):
    """Explicit source-associated labels, functional scores or tractography.

    Coordinates are preserved; source registration must independently be reviewed.
    No tract or functional data is inferred from structural MRI.
    """
    import nibabel as nib
    import uuid
    from .patient_anatomy import MAX_ARRAY

    c = collection
    c.series(source_series)
    p = Path(path)
    identity = str(uuid.uuid4())
    if role_map is not None and kind != "labels":
        raise ValueError("roleMap applies only to categorical labels")
    if kind == "tractography":
        original = c.original(p)
        tractogram = nib.streamlines.load(p, lazy_load=False).tractogram
        tractogram = tractogram.to_world()
        streamlines = tractogram.streamlines
        if streamlines._data.nbytes > MAX_ARRAY:
            raise ValueError("Tractography exceeds local geometry limit")
        points = np.asarray(streamlines._data, dtype=np.float32)
        offsets = np.concatenate(
            [
                np.array([0], np.uint32),
                np.cumsum([len(s) for s in streamlines], dtype=np.uint32),
            ]
        )
        item = {
            "id": identity,
            "kind": kind,
            "label": label,
            "source_series": source_series,
            "original": original,
            "points_ras_mm": c.put_array(points),
            "offsets": c.put_array(offsets),
            "review_status": "unreviewed source registration and tractography",
            "role": "context",
        }
        c.document.setdefault("tractography", []).append(item)
        return item
    im = nib.load(p)
    if len(im.shape) != 3 or np.prod(im.shape) * 4 > MAX_ARRAY:
        raise ValueError("Select a bounded single-volume derivative")
    units = im.header.get_xyzt_units()[0]
    if units not in {"mm", "unknown"}:
        raise ValueError("Imported anatomy requires millimeter geometry")
    values = im.get_fdata(dtype=np.float32)
    if kind in {"motor", "language", "functional"}:
        # Continuous maps are retained as scores, without inventing a calibrated
        # probability or automatically turning positive scores into constraints.
        item = c.add_layer(
            np.zeros(im.shape, np.uint8),
            im.affine,
            layer_id=identity,
            source_series=source_series,
            kind=kind,
            role="context",
            color="#f7bc65",
            label=label,
            provenance={
                "source": c.original(p),
                "threshold": "unset; explicitly review score meaning before labeling",
            },
        )
        item["score"] = c.put_array(values)
        item["score_meaning"] = (
            "supplied functional values; units and threshold require review"
        )
        item["units_status"] = units
        return item
    if (
        not np.array_equal(values, np.rint(values))
        or np.any(values < 0)
        or values.max() > 65535
    ):
        raise ValueError("Categorical derivative must contain integer labels")
    present = {int(value) for value in np.unique(values) if value != 0}
    mapping = _validated_role_map(role_map, present) if role_map is not None else None
    original = c.original(p)
    ref = c.put_array(values.astype(np.uint16))
    items = []
    for value in sorted(present):
        specification = mapping[value] if mapping is not None else None
        row = {
            "id": identity + "-" + str(value),
            "patient_id": c.document["patient_id"],
            "source_series": source_series,
            "label": specification["label"] if specification and specification["label"] else f"{label} {value}",
            "kind": specification["kind"] if specification else kind,
            "role": specification["role"] if specification else role,
            "color": specification["color"] if specification else "#97bdcf",
            "affine": affine(im.affine).tolist(),
            "units_status": units,
            "data": ref,
            "label_value": value,
            "revision": 0,
            "review_status": "unreviewed imported annotation",
            "coverage": "supplied annotation field; negative voxels unassessed",
            "provenance": {"source": original, "source_label_value": value,
                           "kind": "observed_source_annotation"},
        }
        c.document["layers"].append(row)
        items.append(row)
    return items
