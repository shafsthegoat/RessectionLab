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
        self.document = (
            deepcopy(document)
            if document is not None
            else json.loads((self.root / "patient-anatomy.json").read_text())
        )
        d = self.document
        if (
            d.get("schema") != SCHEMA
            or not isinstance(d.get("patient_id"), str)
            or not d["patient_id"]
        ):
            raise ValueError("Unsupported patient anatomy collection")
        if len(d.get("series", [])) > 64 or len(d.get("layers", [])) > 512:
            raise ValueError("Collection exceeds series/layer limit")
        if "snapshot_id" in d and d["snapshot_id"] != self.identity():
            raise ValueError("Anatomy manifest fingerprint differs")
        ids = set()
        for row in d["series"] + d["layers"]:
            if row["id"] in ids or row.get("patient_id") != d["patient_id"]:
                raise ValueError("Duplicate identity or cross-patient anatomy")
            ids.add(row["id"])
            affine(row["affine"])
            if row.get("role", "context") not in ROLES:
                raise ValueError("Unknown anatomical role")
        self._verified = set()

    @classmethod
    def create(cls, directory, patient_id, *, timepoint=None):
        if timepoint is not None and timepoint not in {"preoperative", "postoperative"}:
            raise ValueError("Choose a preoperative or postoperative study")
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
            shutil.copyfile(path, out)
        return {"path": name, "sha256": h, "name": path.name}

    def add_series(
        self, path, *, series_id, modality, timepoint="preoperative", patient_id=None
    ):
        if not isinstance(timepoint, str) or not timepoint.strip():
            raise ValueError("An explicit acquisition/timepoint is required")
        study = self.document.get("study_timepoint")
        if study is None and self.document["series"]:
            study = self.document["series"][0]["timepoint"]
        if study is not None and timepoint != study:
            raise ValueError("Pre-op and post-op scans belong in separate patient study files. Create a new study for this timepoint.")
        if patient_id is not None and patient_id != self.document["patient_id"]:
            raise ValueError("Cannot merge different patients")
        if any(
            x["id"] == series_id
            for x in self.document["series"] + self.document["layers"]
        ):
            raise ValueError("Duplicate series identity")
        im = nib.load(path)
        if len(im.shape) != 3 or np.prod(im.shape) * 4 > MAX_ARRAY:
            raise ValueError("Select a bounded 3D anatomical volume")
        a = affine(im.affine)
        units = im.header.get_xyzt_units()[0]
        if units not in {"mm", "unknown"}:
            raise ValueError("Import requires millimeter geometry")
        row = {
            "id": series_id,
            "patient_id": self.document["patient_id"],
            "modality": modality,
            "timepoint": timepoint,
            "affine": a.tolist(),
            "original": self.original(path),
            "data": self.put_array(im.get_fdata(dtype=np.float32)),
            "acquisition_datetime": None,
            "grid_spacing_mm": nib.affines.voxel_sizes(a).tolist(),
            "acquisition_resolution": "not inferred from supplied grid",
            "units_status": units,
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

    def patient_affine(self, row):
        sid = row.get("source_series", row["id"])
        t = self.document["transforms"].get(sid)
        if t is None or t["status"] not in {"reference", "reviewed", "source-provided"}:
            raise ValueError(f"Registration for {sid} needs review")
        return self.registration_matrix(sid) @ affine(row["affine"])

    def registration_matrix(self, sid, seen=()):
        """Resolve and validate a source-bound graph, independent of review state."""
        if sid in seen or len(seen) >= 64:
            raise ValueError('Cyclic registration graph')
        t = self.document['transforms'].get(sid)
        if not t:
            raise ValueError('Missing source registration')
        m = affine(t['matrix'])
        source, reference = self.series(sid), self.series(self.document['reference_series'])
        if t['status'] == 'reference':
            if sid != reference['id'] or not np.allclose(m, np.eye(4)):
                raise ValueError('Invalid reference registration')
        elif (t.get('source_sha256') != source['original']['sha256'] or
              t.get('reference_sha256') != reference['original']['sha256']):
            raise ValueError('Registration source identity changed; repeat alignment')
        if t.get('parent_series'):
            parent = self.registration_matrix(t['parent_series'], (*seen, sid))
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
            if expected is not None and (
                not current.exists()
                or json.loads(current.read_text())["snapshot_id"] != expected
            ):
                raise ValueError("Patient changed; reload before committing")
            self.document["revision"] += 1
            self.document["history"].append(
                {
                    "event": event,
                    "author": author,
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )
            self.document["snapshot_id"] = self.identity()
            text = json.dumps(self.document, indent=2, allow_nan=False) + "\n"
            rev = self.root / "revisions"
            rev.mkdir(exist_ok=True)
            (rev / (self.document["snapshot_id"] + ".json")).write_text(text)
            with tempfile.NamedTemporaryFile(
                dir=self.root, mode="w", delete=False
            ) as tmp:
                tmp.write(text)
                name = tmp.name
            os.replace(name, current)
        return self.document["snapshot_id"]

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
            if inside and row.get("score"):
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

        state = self.sample(point)["layers"][layer_id]
        if not state["known"]:
            return {
                "snapshot_id": self.snapshot_id,
                "available": False,
                "reason": state.get("reason", "outside_source_coverage"),
            }
        row = self.collection.layer(layer_id)
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
        for row in c.document["layers"]:
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
