"""Native-grid, same-patient anatomy. Display derivatives never replace evidence.

Packages are portable directories: content-addressed arrays, original files,
immutable revision manifests, and a small atomic current-manifest pointer.
No code or arbitrary paths in a package are executed.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

SCHEMA = "patient-anatomy-v1"
MAX_ARRAY = 512 * 1024**2
ROLES = {"context", "target", "critical"}
NIFTI_FORM_TOLERANCE_MM = 0.01


def source_metadata_phrases(row):
    original = row.get("original", {})
    values = [row.get("id"), row.get("modality"), row.get("label"), row.get("source_kind"),
              original.get("name") or Path(original.get("path") or "").name]
    keys = ("ImageType", "SeriesDescription", "ProtocolName", "SequenceName", "DerivationDescription",
            "image_type", "dicom_image_type", "series_description", "protocol_name", "derivation_description")
    for record in (row, row.get("metadata") or {}, row.get("dicom") or {}):
        values.extend(record.get(key) for key in keys)
    return " ".join(str(value).lower() for value in values if value)


def projection_or_localizer(row):
    """A projected or scout image can be inspected, but cannot place anatomy in 3D."""
    if row.get("projection_or_localizer") is True:
        return True
    phrases = source_metadata_phrases(row).replace("_", " ")
    return bool(re.search(r"\b(?:(?:tof)?mips?|minip|projections?|locali[sz]er|scout)\b|(?:maximum|minimum).intensity", phrases))


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def affine(value):
    a = np.asarray(value, dtype=float)
    if (
        a.shape != (4, 4)
        or not np.isfinite(a).all()
        or not np.array_equal(a[3], [0, 0, 0, 1])
        or abs(np.linalg.det(a[:3, :3])) < 1e-10
    ):
        raise ValueError("Invalid voxel-to-RAS millimeter affine")
    return a


def world(a, points):
    return np.asarray(points) @ a[:3, :3].T + a[:3, 3]


def nifti_geometry(image):
    """Record active native transforms and the affine used to display the image."""
    units, time_units = image.header.get_xyzt_units()
    result = {"spatial_units": units, "temporal_units": time_units}
    forms = {}
    invalid = []
    for name in ("qform", "sform"):
        code = int(image.header[f"{name}_code"])
        result[f"{name}_code"] = code
        if code:
            try:
                forms[name] = affine(getattr(image, f"get_{name}")())
            except (ValueError, np.linalg.LinAlgError):
                invalid.append(name)
        result[name] = forms[name].tolist() if name in forms else None
    result["display_basis"] = ("coded sform" if result["sform_code"] else
                               "coded qform" if result["qform_code"] else "uncoded header fallback")
    if invalid:
        result.update(frame_status="invalid", invalid_forms=invalid)
    elif not forms:
        result["frame_status"] = "uncoded"
    else:
        result["frame_status"] = "coded"
        if len(forms) == 2 and units == "mm":
            shape = (*image.shape, 1, 1, 1)[:3]
            corners = list(product(*[(0, count - 1) for count in shape]))
            disagreement = float(np.linalg.norm(world(forms["qform"], corners) - world(forms["sform"], corners), axis=1).max())
            if np.isfinite(disagreement):
                result["maximum_qform_sform_corner_difference_mm"] = disagreement
                if disagreement > NIFTI_FORM_TOLERANCE_MM:
                    result["frame_status"] = "conflicted"
            else:
                result["frame_status"] = "invalid"
    return result


def nifti_geometry_issue(geometry, *, allow_header_conflict=False):
    units = geometry["spatial_units"]
    if units != "mm":
        return f"Geometry requires declared millimeter units (NIfTI declares {units})"
    status = geometry["frame_status"]
    if status == "uncoded":
        return "NIfTI source has no coded qform or sform"
    if status == "invalid":
        return "NIfTI source has an invalid coded qform or sform"
    if status == "conflicted" and not allow_header_conflict:
        return "Unresolved NIfTI qform/sform disagreement"
    return None


@contextmanager
def reconstruction_lock(root=None):
    """All reconstruction adapters share one nonblocking local job slot."""
    root = Path(root or Path.home() / ".cache/resectionlab")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "reconstruction.lock").open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise ValueError("Another local reconstruction is running") from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class PatientImagingCollection:
    def __init__(self, directory, document=None):
        self.root = Path(directory).resolve()
        manifest = self.root / "patient-anatomy.json"
        self.document = (
            deepcopy(document)
            if document is not None
            else json.loads(manifest.read_text())
        )
        d = self.document
        if (
            d.get("schema") != SCHEMA
            or not isinstance(d.get("patient_id"), str)
            or not d["patient_id"]
        ):
            raise ValueError("Unsupported patient anatomy collection")
        sources = self.source_records()
        if len(sources) > 64 or len(d.get("layers", [])) > 512:
            raise ValueError("Collection exceeds series/layer limit")
        if "snapshot_id" in d and d["snapshot_id"] != self.identity():
            raise ValueError("Anatomy manifest fingerprint differs")
        ids = set()
        rows = sources + d["layers"]
        for row in rows:
            if row["id"] in ids or row.get("patient_id") != d["patient_id"]:
                raise ValueError("Duplicate identity or cross-patient anatomy")
            ids.add(row["id"])
            if row.get("storage") != "source-asset" or "affine" in row:
                affine(row["affine"])
            if row.get("role", "context") not in ROLES:
                raise ValueError("Unknown anatomical role")
        self._verified = set()
        self._native_headers = {}
        self._base_snapshot = (d.get("snapshot_id") or self.identity()) if manifest.is_file() else None
        self.published_reference = d.get("reference_series") if self._base_snapshot else None

    @classmethod
    def create(cls, directory, patient_id, *, timepoint=None):
        if timepoint is not None and timepoint not in {"preoperative", "postoperative", "source-session-unknown"}:
            raise ValueError("Choose a preoperative, postoperative or unspecified study timepoint")
        root = Path(directory)
        if (root / "patient-anatomy.json").exists():
            raise ValueError("Patient collection already exists")
        root.mkdir(parents=True, exist_ok=True)
        return cls(
            root,
            {
                "schema": SCHEMA,
                "patient_id": patient_id,
                **({"study_timepoint": timepoint} if timepoint else {}),
                "revision": 0,
                "series": [],
                "layers": [],
                "transforms": {},
                "history": [],
                "missing": [],
            },
        )

    def identity(self):
        return digest({k: v for k, v in self.document.items() if k != "snapshot_id"})

    def source_records(self):
        """Scalar scans and locally imported raw sources, excluding legacy inventory notes."""
        return self.document["series"] + [
            row for row in self.document.get("source_assets", []) if row.get("storage") == "source-asset"
        ]

    def path(self, relative):
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("Package requires relative members")
        p = (self.root / relative).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError("Package member escapes patient directory")
        return p

    def put_array(self, data):
        data = np.ascontiguousarray(data)
        if (
            data.dtype.hasobject
            or data.nbytes > MAX_ARRAY
            or not np.isfinite(data).all()
        ):
            raise ValueError("Nonfinite, object or oversized anatomy array")
        h = hashlib.sha256(memoryview(data).cast("B")).hexdigest()
        key = f"arrays/{h}-{data.dtype.name}-" + "x".join(map(str, data.shape)) + ".npy"
        p = self.path(key)
        p.parent.mkdir(exist_ok=True)
        if not p.exists():
            with tempfile.NamedTemporaryFile(dir=p.parent, delete=False) as f:
                np.save(f, data, allow_pickle=False)
                tmp = Path(f.name)
            os.replace(tmp, p)
        return {
            "path": key,
            "sha256": file_hash(p),
            "shape": list(data.shape),
            "dtype": data.dtype.name,
        }

    def array(self, ref):
        p = self.path(ref["path"])
        if p.stat().st_size > MAX_ARRAY + 4096:
            raise ValueError("Oversized anatomy member")
        key = (str(p), p.stat().st_mtime_ns, ref["sha256"])
        if key not in self._verified:
            if file_hash(p) != ref["sha256"]:
                raise ValueError("Anatomy array checksum differs")
            self._verified.add(key)
        a = np.load(p, mmap_mode="r", allow_pickle=False)
        if list(a.shape) != ref["shape"] or a.dtype.name != ref["dtype"]:
            raise ValueError("Anatomy array layout differs")
        return a

    def original(self, path):
        path = Path(path)
        h = file_hash(path)
        name = f"originals/{h}/{path.name}"
        out = self.path(name)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            with tempfile.NamedTemporaryFile(dir=out.parent, delete=False) as f:
                temporary = Path(f.name)
            try:
                shutil.copyfile(path, temporary)
                if file_hash(temporary) != h:
                    raise ValueError("Original source changed during import")
                os.replace(temporary, out)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            key = (str(out), out.stat().st_mtime_ns, h)
            if key not in self._verified and file_hash(out) != h:
                raise ValueError("Stored original source checksum differs")
            self._verified.add(key)
        return {"path": name, "sha256": h, "name": path.name}

    def verify_source_association(self, timepoint, patient_id=None):
        if timepoint is not None and (not isinstance(timepoint, str) or not timepoint.strip()):
            raise ValueError("An explicit acquisition/timepoint is required")
        sources = self.source_records()
        study = self.document.get("study_timepoint") or (sources[0]["timepoint"] if sources else None)
        if (study is not None or sources) and timepoint != study:
            raise ValueError("Pre-op and post-op scans belong in separate patient study files. Create a new study for this timepoint.")
        if patient_id is not None and patient_id != self.document["patient_id"]:
            raise ValueError("Cannot merge different patients")

    def verify_new_source_id(self, series_id):
        if not isinstance(series_id, str) or not series_id.strip():
            raise ValueError("An explicit series identity is required")
        if any(
            x["id"] == series_id
            for x in self.source_records() + self.document["layers"]
        ):
            raise ValueError("Duplicate series identity")

    def add_series(
        self, path, *, series_id, modality, timepoint="preoperative", patient_id=None,
        original=None,
    ):
        self.verify_source_association(timepoint, patient_id)
        self.verify_new_source_id(series_id)
        original = self.original(path) if original is None else original
        im = nib.load(self.path(original["path"]))
        if len(im.shape) != 3 or np.prod(im.shape) * 4 > MAX_ARRAY:
            raise ValueError("Select a bounded 3D anatomical volume")
        a = affine(im.affine)
        geometry = nifti_geometry(im)
        units = geometry["spatial_units"]
        if units not in {"mm", "unknown"}:
            raise ValueError("Import requires millimeter geometry")
        row = {
            "id": series_id,
            "patient_id": self.document["patient_id"],
            "modality": modality,
            "timepoint": timepoint,
            "affine": a.tolist(),
            "original": original,
            "data": self.put_array(im.get_fdata(dtype=np.float32)),
            "acquisition_datetime": None,
            "grid_spacing_mm": nib.affines.voxel_sizes(a).tolist(),
            "acquisition_resolution": "not inferred from supplied grid",
            "units_status": units,
            "native_geometry": geometry,
        }
        self.document["series"].append(row)
        if len(self.document["series"]) == 1:
            self.document["reference_series"] = series_id
            self.document["transforms"][series_id] = {
                "matrix": np.eye(4).tolist(),
                "status": "reference",
                "uncertainty_mm": None,
            }
        return row

    def preview_pyramid(self, series_id, levels=(2, 4, 8)):
        """Separate linear display previews; native intensity and geometry retained."""
        row = self.series(series_id)
        a = affine(row["affine"])
        data = self.array(row["data"])
        result = []
        for factor in levels:
            if factor not in {2, 4, 8, 16}:
                raise ValueError("Unsupported preview scale")
            shape = tuple(int(np.ceil(n / factor)) for n in data.shape)
            preview_affine = a @ np.diag([factor, factor, factor, 1.0])
            preview = resample(data, a, shape, preview_affine, order=1)
            result.append(
                {
                    "factor": factor,
                    "data": self.put_array(preview),
                    "affine": preview_affine.tolist(),
                    "use": "display preview only",
                }
            )
        row["preview_pyramid"] = result
        return result

    def series(self, key):
        return next(x for x in self.document["series"] if x["id"] == key)

    def layer(self, key):
        return next(x for x in self.document["layers"] if x["id"] == key)

    def add_layer(
        self,
        data,
        native_affine,
        *,
        layer_id,
        source_series,
        kind,
        role,
        color,
        label=None,
        probability=None,
        label_value=1,
        provenance=None,
    ):
        source = self.series(source_series)
        if role not in ROLES or any(
            x["id"] == layer_id
            for x in self.document["series"] + self.document["layers"]
        ):
            raise ValueError("Invalid or duplicate anatomical layer")
        data = np.asarray(data)
        if (
            data.ndim != 3
            or not np.issubdtype(data.dtype, np.integer)
            or np.any(data < 0)
            or data.max() > 65535
        ):
            raise ValueError("Expected a nonnegative categorical label volume")
        row = {
            "id": layer_id,
            "patient_id": source["patient_id"],
            "source_series": source_series,
            "label": label or layer_id,
            "kind": kind,
            "role": role,
            "color": color,
            "affine": affine(native_affine).tolist(),
            "data": self.put_array(
                data.astype(np.uint16 if data.max() > 255 else np.uint8)
            ),
            "label_value": int(label_value),
            "review_status": "unreviewed",
            "provenance": provenance or {},
            "coverage": "source image field; anatomical sensitivity unassessed",
            "revision": 0,
        }
        if probability is not None:
            p = np.asarray(probability, dtype=np.float32)
            if p.shape != data.shape or np.any((p < 0) | (p > 1)):
                raise ValueError(
                    "Probability grid must match segmentation and lie in [0,1]"
                )
            row["probability"] = self.put_array(p)
            row["score_meaning"] = "uncalibrated model score"
        self.document["layers"].append(row)
        return row

    def mask(self, row):
        return self.array(row["data"]) == row["label_value"]

    def require_native_geometry(self, row, *, allow_header_conflict=False, allow_legacy_annotation_grid=False):
        """Separate native viewing from permission to interpret a grid in RAS mm."""
        units = row.get("units_status")
        if units is not None and units != "mm":
            raise ValueError(f"Geometry for {row['id']} requires declared millimeter units")
        is_series = "source_series" not in row
        provenance = row.get("provenance", {})
        original = row.get("original") if is_series else provenance.get("source")
        has_original = isinstance(original, dict) and isinstance(original.get("path"), str)
        supplied = provenance.get("kind") in {"observed_source_annotation", "supplied_annotation"}
        functional = row.get("kind") in {"functional", "motor", "language"} and "score" in row
        # In-memory model/synthetic layers declare their affine through the
        # RAS-mm add_layer contract. Supplied volumes have a native header.
        if not is_series and "native_geometry" not in row and not (has_original and (supplied or functional)):
            return
        if not has_original:
            raise ValueError(f"Original geometry for {row['id']} is unavailable")
        path = self.path(original["path"])
        try:
            stat = path.stat()
            key = (path, stat.st_mtime_ns, stat.st_size)
            if key not in self._native_headers:
                image = nib.load(path)
                self._native_headers[key] = (nifti_geometry(image), image.shape, affine(image.affine))
            geometry, shape, matrix = self._native_headers[key]
        except (OSError, ValueError, nib.filebasedimages.ImageFileError) as error:
            raise ValueError(f"Cannot read original native geometry for {row['id']}") from error
        problem = nifti_geometry_issue(geometry, allow_header_conflict=allow_header_conflict)
        if (problem and allow_legacy_annotation_grid and not is_series and "native_geometry" not in row
                and provenance.get("kind") == "observed_source_annotation"
                and geometry["spatial_units"] == "unknown" and geometry["frame_status"] == "coded"):
            source = self.series(row["source_series"])
            # The older research decoder explicitly supported this exact
            # source-bound affine and shape case. A shape match alone is
            # insufficient, and alternate input frames must agree.
            forms_agree = all(geometry[name] is None or np.allclose(geometry[name], matrix, rtol=0, atol=1e-5)
                              for name in ("qform", "sform"))
            if (forms_agree and tuple(source["data"]["shape"]) == shape
                    and np.array_equal(source["affine"], matrix) and np.array_equal(row["affine"], matrix)):
                self.require_native_geometry(source)
                identity = (str(path), stat.st_mtime_ns, original["sha256"])
                if identity not in self._verified:
                    if file_hash(path) != original["sha256"]:
                        raise ValueError(f"Original annotation identity changed: {row['id']}")
                    self._verified.add(identity)
                problem = None
        if problem:
            raise ValueError(f"{problem}: {row['id']}")
        if (tuple(row["data"]["shape"]) != shape
                or not np.allclose(matrix, row["affine"], rtol=0, atol=1e-5)
                or (row.get("native_geometry") is not None and row["native_geometry"] != geometry)):
            raise ValueError(f"Original native geometry differs from imported grid: {row['id']}")

    def patient_affine(self, row):
        sid = row.get("source_series", row["id"])
        if sid != row["id"]:
            self.require_native_geometry(row)
        return self.registration_matrix(sid, require_review=True, require_geometry=True) @ affine(row["affine"])

    def registration_matrix(self, sid, seen=(), *, require_review=False, require_geometry=False):
        """Resolve a source-bound graph, optionally requiring reviewed metric geometry."""
        if sid in seen or len(seen) >= 64:
            raise ValueError('Cyclic registration graph')
        t = self.document['transforms'].get(sid)
        if require_review and (not t or t.get("status") not in {"reference", "reviewed", "source-provided"}):
            raise ValueError(f"Registration for {sid} needs review")
        if not t:
            raise ValueError('Missing source registration')
        m = affine(t['matrix'])
        if not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-5) or np.linalg.det(m[:3, :3]) < 0:
            raise ValueError('Registration requires a rigid source transform')
        source, reference = self.series(sid), self.series(self.document['reference_series'])
        if source["timepoint"] != reference["timepoint"]:
            raise ValueError("Registration requires the same study timepoint")
        if require_geometry:
            if projection_or_localizer(source) or projection_or_localizer(reference):
                raise ValueError("Projection or localizer supports independent source review only")
            self.require_native_geometry(source)
            if not seen and source is not reference:
                self.require_native_geometry(reference)
        if t['status'] == 'reference':
            if sid != reference['id'] or not np.allclose(m, np.eye(4)):
                raise ValueError('Invalid reference registration')
        elif (t.get('source_sha256') != source['original']['sha256'] or
              t.get('reference_sha256') != reference['original']['sha256']):
            raise ValueError('Registration source identity changed; repeat alignment')
        if t.get('parent_series'):
            parent = self.registration_matrix(t['parent_series'], (*seen, sid), require_review=require_review,
                                              require_geometry=require_geometry)
            if t.get('reference_series') != reference['id'] or not np.allclose(parent @ affine(t['source_to_parent']), m, atol=1e-6):
                raise ValueError('Registration parent changed; repeat alignment')
        if t.get('transform_digest') and t['transform_digest'] != digest({'matrix': t['matrix'], 'relative': t.get('source_to_parent')}):
            raise ValueError('Registration quality no longer matches its transform')
        return m

    def set_registration(
        self, series_id, matrix, *, status, method, uncertainty_mm=None, landmarks=None
    ):
        self.series(series_id)
        if status not in {"unreviewed", "reviewed", "source-provided"}:
            raise ValueError("Invalid registration status")
        m = affine(matrix)
        if (
            not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-5)
            or np.linalg.det(m[:3, :3]) < 0
        ):
            raise ValueError("This importer accepts rigid transforms only")
        if uncertainty_mm is not None and (
            not np.isfinite(uncertainty_mm) or uncertainty_mm < 0
        ):
            raise ValueError("Invalid registration uncertainty")
        self.document["transforms"][series_id] = {
            "matrix": m.tolist(),
            "status": status,
            "method": method,
            "uncertainty_mm": uncertainty_mm,
            "landmarks": landmarks or [],
            "source_sha256": self.series(series_id)["original"]["sha256"],
            "reference_sha256": self.series(self.document["reference_series"])[
                "original"
            ]["sha256"],
        }

    def save(self, event="save", *, author="local researcher", expected=None):
        # Validate proposed edits/imports before replacing the current manifest.
        PatientImagingCollection(
            self.root, {k: v for k, v in self.document.items() if k != "snapshot_id"}
        )
        self.root.mkdir(exist_ok=True)
        with (self.root / ".write.lock").open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            current = self.root / "patient-anatomy.json"
            published = json.loads(current.read_text()) if current.exists() else None
            current_id = (published.get("snapshot_id") or digest(published)) if published is not None else None
            if current_id != self._base_snapshot or (expected is not None and current_id != expected):
                raise ValueError("Patient changed; reload before committing")
            document = dict(self.document)
            document["revision"] += 1
            document["history"] = [
                *document["history"],
                {
                    "event": event,
                    "author": author,
                    "at": datetime.now(timezone.utc).isoformat(),
                },
            ]
            document.pop("snapshot_id", None)
            document["snapshot_id"] = digest(document)
            text = json.dumps(document, indent=2, allow_nan=False) + "\n"
            rev = self.root / "revisions"
            rev.mkdir(exist_ok=True)
            for destination in (rev / (document["snapshot_id"] + ".json"), current):
                with tempfile.NamedTemporaryFile(dir=destination.parent, mode="w", delete=False) as tmp:
                    temporary = Path(tmp.name)
                    try:
                        tmp.write(text)
                        tmp.flush()
                        os.fsync(tmp.fileno())
                        os.replace(temporary, destination)
                    finally:
                        temporary.unlink(missing_ok=True)
            self.document.update(document)
            self._base_snapshot = document["snapshot_id"]
            self.published_reference = document.get("reference_series")
        return self._base_snapshot

    def edit(
        self, layer_id, operation, *, points=(), radius_mm=2.0, threshold=0.5, mask=None
    ):
        row = self.layer(layer_id)
        data = (
            self.mask(row).copy()
            if mask is None
            else np.asarray(mask, dtype=bool).copy()
        )
        if data.shape != tuple(row["data"]["shape"]):
            raise ValueError("Edit grid differs")
        a = self.patient_affine(row)
        inverse = np.linalg.inv(a)
        if operation in {"paint", "erase"}:
            if not 0 < radius_mm <= 50 or len(points) > 10000:
                raise ValueError("Brush radius or stroke exceeds limit")
            for point in points:
                if np.asarray(point).shape != (3,) or not np.isfinite(point).all():
                    raise ValueError("Invalid brush point")
                centre = world(inverse, point)
                extent = radius_mm * np.linalg.norm(inverse[:3, :3], axis=1)
                lo = np.maximum(0, np.floor(centre - extent).astype(int))
                hi = np.minimum(data.shape, np.ceil(centre + extent).astype(int) + 1)
                if np.any(hi <= lo):
                    continue
                indices = np.indices(hi - lo).reshape(3, -1).T + lo
                selected = (
                    np.linalg.norm(world(a, indices) - point, axis=1) <= radius_mm
                )
                chosen = indices[selected]
                data[tuple(chosen.T)] = operation == "paint"
        elif operation == "threshold":
            if "probability" not in row or not 0 < threshold < 1:
                raise ValueError("A retained probability field is required")
            data = self.array(row["probability"]) >= threshold
        elif operation == "refine-vessels":
            from .anatomy_vessels import refine_layer
            data, _ = refine_layer(self, row, data, threshold=threshold,
                                   radius_mm=radius_mm)
        elif operation in {"component", "seed"}:
            if len(points) != 1:
                raise ValueError("One source-linked seed required")
            index = np.floor(world(inverse, points[0]) + 0.5).astype(int)
            if np.any(index < 0) or np.any(index >= data.shape):
                raise ValueError("Seed outside layer")
            candidate = data
            if operation == "seed":
                if "probability" not in row or not 0 < threshold < 1:
                    raise ValueError("Seeded growth requires retained model scores")
                candidate = self.array(row["probability"]) >= threshold
            components, _ = ndi.label(candidate, structure=np.ones((3, 3, 3)))
            label = components[tuple(index)]
            if not label:
                raise ValueError("Seed is outside the selected structure")
            data = components == label
        else:
            raise ValueError("Unsupported correction operation")
        return data

    def commit_edit(self, layer_id, mask, *, author, expected, operations):
        if not author.strip():
            raise ValueError("Correction author is required")
        current = json.loads((self.root / "patient-anatomy.json").read_text())
        if current["snapshot_id"] != expected:
            raise ValueError("Patient changed; reload before committing")
        row = self.layer(layer_id)
        if tuple(mask.shape) != tuple(row["data"]["shape"]):
            raise ValueError("Correction grid changed")
        row.setdefault("original_data", deepcopy(row["data"]))
        row.setdefault("original_label_value", row["label_value"])
        row["data"] = self.put_array(np.asarray(mask, dtype=np.uint8))
        row["label_value"] = 1
        row["revision"] += 1
        row["review_status"] = "edited; anatomical review pending"
        row["edits"] = [
            *row.get("edits", []),
            {"author": author, "operations": operations},
        ]
        row.pop("mesh", None)
        return self.save(
            "commit correction: " + layer_id, author=author, expected=expected
        )


@dataclass(frozen=True)
class PatientAnatomySnapshot:
    collection: PatientImagingCollection
    snapshot_id: str

    @classmethod
    def capture(cls, collection):
        frozen = PatientImagingCollection(collection.root, collection.document)
        return cls(frozen, frozen.identity())

    def _check(self):
        if self.collection.identity() != self.snapshot_id:
            raise ValueError("Immutable anatomy snapshot was modified")

    def sample(self, point):
        self._check()
        if np.asarray(point).shape != (3,) or not np.isfinite(point).all():
            raise ValueError("A finite physical point is required")
        from .anatomy_evidence import inference_bounds
        result = {}
        for row in self.collection.document["layers"]:
            try:
                p = world(np.linalg.inv(self.collection.patient_affine(row)), point)
            except ValueError:
                result[row["id"]] = {
                    "known": False,
                    "reason": "registration_unreviewed",
                }
                continue
            shape = np.array(row["data"]["shape"])
            inside = bool(np.all(p >= -0.5) and np.all(p < shape - 0.5))
            coverage = row.get("coverage_fields", {})
            bounds = inference_bounds(row)
            index = np.floor(p + 0.5).astype(int)
            if bounds is not None:
                inside = inside and bool(np.all(index >= bounds[0]) and np.all(index <= bounds[1]))
            slices = coverage.get("supported_axial_indices")
            if slices is not None:
                inside = inside and int(index[2]) in slices
            if row.get("availability") == "empty_output":
                result[row["id"]] = {"known": False, "occupied": None, "reason": "empty_output"}
                continue
            value = (
                None
                if not inside
                else bool(
                    self.collection.array(row["data"])[
                        tuple(np.floor(p + 0.5).astype(int))
                    ]
                    == row["label_value"]
                )
            )
            result[row["id"]] = {
                "known": inside,
                "occupied": value,
                "role": row["role"],
                "review_status": row["review_status"],
                "negative_meaning": "not labelled; absence of anatomy is not established",
            }
            if not inside:
                result[row["id"]]["reason"] = "outside_source_coverage"
            if inside and row.get("score"):
                result[row["id"]].update(occupied=None, known=False, reason="score_without_threshold")
                result[row["id"]]["score"] = float(
                    self.collection.array(row["score"])[
                        tuple(np.floor(p + 0.5).astype(int))
                    ]
                )
                result[row["id"]]["score_meaning"] = row.get("score_meaning")
        return {
            "snapshot_id": self.snapshot_id,
            "layers": result,
            "missing": self.collection.document["missing"],
        }

    def vessel_geometry(self, layer_id="vessels", point=None):
        self._check()
        from .anatomy_geometry import inspect_vessel_geometry

        return inspect_vessel_geometry(self.collection, layer_id, point)

    def clearance(self, start, end, radius_mm=1.0):
        self._check()
        from .anatomy_planning import route_clearance

        return route_clearance(self, start, end, radius_mm)

    def distance(self, layer_id, point):
        """Conservative distance to labelled voxel cells; missing anatomy is unknown."""
        self._check()
        from scipy.spatial import cKDTree

        from .anatomy_evidence import is_siam_context
        row = self.collection.layer(layer_id)
        if is_siam_context(row):
            return {"snapshot_id": self.snapshot_id, "available": False, "reason": "SIAM tissue estimate is context only"}
        state = self.sample(point)["layers"][layer_id]
        if not state["known"]:
            return {
                "snapshot_id": self.snapshot_id,
                "available": False,
                "reason": state.get("reason", "outside_source_coverage"),
            }
        a = self.collection.patient_affine(row)
        indices = np.argwhere(self.collection.mask(row))
        if not len(indices):
            return {
                "snapshot_id": self.snapshot_id,
                "available": False,
                "reason": "no labelled structure",
            }
        distance = float(cKDTree(world(a, indices)).query(point)[0])
        bound = distance - 0.5 * sum(np.linalg.norm(a[:3, i]) for i in range(3))
        return {
            "snapshot_id": self.snapshot_id,
            "available": True,
            "distance_lower_bound_mm": bound,
            "meaning": "distance to labelled voxel cells only; negative means contact cannot be excluded",
        }

    def simulation_grid(self, reference_id=None, *, removed=None):
        """Explicit NN target and conservative obstacle derivative; never a display mesh."""
        self._check()
        c = self.collection
        reference = c.series(reference_id or c.document["reference_series"])
        target_affine = c.patient_affine(reference)
        shape = tuple(reference["data"]["shape"])
        targets = {}
        critical = {}
        unknown = []
        errors = {}
        from .anatomy_evidence import vascular_evidence_supported
        for row in c.document["layers"]:
            if (not vascular_evidence_supported(c, row) or row.get("availability") == "empty_output"
                    or row.get("provenance", {}).get("evidence_use") == "context_only"):
                unknown.append(row["id"])
                continue
            if row["role"] == "context":
                continue
            try:
                source_affine = c.patient_affine(row)
            except ValueError:
                unknown.append(row["id"])
                continue
            source = c.mask(row)
            native_volume = float(
                np.count_nonzero(source) * abs(np.linalg.det(source_affine[:3, :3]))
            )
            same_grid = source.shape == shape and np.allclose(
                source_affine, target_affine
            )
            errors[row["id"]] = {
                "source_data_sha256": row["data"]["sha256"],
                "source_affine": source_affine.tolist(),
                "reference_series": reference["id"],
                "method": "identity" if same_grid else "nearest-neighbour",
                "native_foreground_mm3": native_volume,
                "boundary_error_mm": None,
                "boundary_error_meaning": "not a validated boundary error; volume change and sampling bounds recorded separately",
                "voxel_center_quantization_bound_mm": 0.0
                if same_grid
                else 0.5 * sum(np.linalg.norm(target_affine[:3, i]) for i in range(3)),
            }
            if not native_volume:
                unknown.append(row["id"])
                continue
            if row["role"] == "critical" and not np.allclose(
                source_affine, target_affine
            ):
                axes = source_affine[:3, :3]
                spacing = np.linalg.norm(axes, axis=0)
                if not np.allclose(
                    (axes / spacing).T @ (axes / spacing), np.eye(3), atol=1e-5
                ):
                    raise ValueError(
                        "Sheared critical grid needs an explicit orthogonal derivative"
                    )
                margin = 0.5 * (
                    np.linalg.norm(spacing)
                    + np.linalg.norm(nib.affines.voxel_sizes(target_affine))
                )
                source = ndi.distance_transform_edt(~source, sampling=spacing) <= margin
                errors[row["id"]].update(
                    method="conservative-dilation-then-nearest-neighbour",
                    conservative_dilation_mm=float(margin),
                )
            output = (
                resample(
                    source.astype(np.uint8),
                    source_affine,
                    shape,
                    target_affine,
                    order=0,
                )
                > 0
            )
            if row["role"] == "target":
                if removed is not None:
                    output &= ~np.asarray(removed, dtype=bool)
                targets[row["id"]] = output
            else:
                critical[row["id"]] = output
            output_volume = float(
                np.count_nonzero(output) * abs(np.linalg.det(target_affine[:3, :3]))
            )
            errors[row["id"]].update(
                derived_foreground_mm3=output_volume,
                volume_change_mm3=output_volume - native_volume,
                derivative_sha256=digest(
                    {
                        "array": hashlib.sha256(
                            np.ascontiguousarray(output).tobytes()
                        ).hexdigest(),
                        "affine": target_affine.tolist(),
                    }
                ),
            )
        return {
            "snapshot_id": self.snapshot_id,
            "affine": target_affine,
            "shape": shape,
            "targets": targets,
            "critical": critical,
            "unknown": [*c.document["missing"], *unknown],
            "resampling": errors,
            "meaning": "geometric research derivative; no tissue mechanics or clinical safety claim",
        }


def resample(data, source_affine, shape, target_affine, *, order=0):
    transform = np.linalg.inv(affine(source_affine)) @ affine(target_affine)
    if tuple(data.shape) == tuple(shape) and np.allclose(
        transform, np.eye(4), atol=1e-7
    ):
        return np.asarray(data)
    output = np.empty(shape, dtype=np.float32 if order else data.dtype)
    # Slab the output to bound temporary interpolation memory.
    for start in range(0, shape[0], 16):
        stop = min(start + 16, shape[0])
        offset = transform[:3, 3] + transform[:3, 0] * start
        ndi.affine_transform(
            data,
            transform[:3, :3],
            offset,
            output_shape=(stop - start, *shape[1:]),
            output=output[start:stop],
            order=order,
            mode="constant",
            cval=0,
            prefilter=False,
        )
    return output
