"""Local imaging intake and reviewable rigid registration."""

from pathlib import Path
import json
import numpy as np
from .patient_anatomy import affine, world, reconstruction_lock, file_hash


def import_dicom(collection, directory, *, modality=None, timepoint="preoperative"):
    import pydicom
    import SimpleITK as sitk

    directory = Path(directory)
    ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)) or []
    if not ids:
        raise ValueError("No readable DICOM image series")
    imported = []
    for uid in ids:
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), uid)
        headers = [pydicom.dcmread(p, stop_before_pixels=True) for p in files]
        patients = {str(getattr(h, "PatientID", "")) for h in headers}
        if (
            len(patients) != 1
            or not next(iter(patients))
            or next(iter(patients)) != collection.document["patient_id"]
        ):
            raise ValueError(
                "DICOM patient ID must match the collection; explicit deidentified association required"
            )
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
            modality=modality or str(headers[0].Modality),
            timepoint=timepoint,
        )
        row["dicom_originals"] = [collection.original(p) for p in files]
        row["acquisition_datetime"] = (
            str(getattr(headers[0], "AcquisitionDateTime", "")) or None
        )
        row["dicom_frame_uid"] = (
            str(getattr(headers[0], "FrameOfReferenceUID", "")) or None
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
    return errors.tolist()


def import_anatomy_derivative(
    collection, path, *, source_series, kind, label, role="context"
):
    """Explicit source-associated labels, functional scores or tractography.

    Coordinates are preserved; source registration must independently be reviewed.
    No tract or functional data is inferred from structural MRI.
    """
    import nibabel as nib
    import uuid
    from .patient_anatomy import MAX_ARRAY

    c = collection
    source = c.series(source_series)
    p = Path(path)
    identity = str(uuid.uuid4())
    original = c.original(p)
    if kind == "tractography":
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
                "source": original,
                "threshold": "unset; explicitly review score meaning before labeling",
            },
        )
        item["score"] = c.put_array(values)
        item["score_meaning"] = (
            "supplied functional values; units and threshold require review"
        )
        return item
    if (
        not np.array_equal(values, np.rint(values))
        or np.any(values < 0)
        or values.max() > 65535
    ):
        raise ValueError("Categorical derivative must contain integer labels")
    ref = c.put_array(values.astype(np.uint16))
    items = []
    for value in np.unique(values):
        if value == 0:
            continue
        row = {
            "id": identity + "-" + str(int(value)),
            "patient_id": c.document["patient_id"],
            "source_series": source_series,
            "label": f"{label} {int(value)}",
            "kind": kind,
            "role": role,
            "color": "#97bdcf",
            "affine": affine(im.affine).tolist(),
            "data": ref,
            "label_value": int(value),
            "revision": 0,
            "review_status": "unreviewed imported annotation",
            "coverage": "supplied annotation field; negative voxels unassessed",
            "provenance": {"source": original},
        }
        c.document["layers"].append(row)
        items.append(row)
    return items
