"""Bounded positive-to-background entry faces in patient RAS geometry."""
import numpy as np

from .anatomy_cortical_access import HEMISPHERES, cortical_role, qualified_cortical_source
from .anatomy_evidence import inference_bounds
from .patient_anatomy import world

# Space directions in patient RAS, independent of voxel ordering or the editor entry.
DIRECTIONS = {"Superior": (0, 0, 1), "Right-superior": (1, 0, .61), "Left-superior": (-1, 0, .61),
              "Anterior-superior": (0, 1, .61), "Posterior-superior": (0, -1, .61)}
MAX_DEPTH_MM = 200
MAX_FACES_PER_AXIS = 2048
ORIGIN = "native-source-entry-windows"


def _source_kind(row):
    if qualified_cortical_source(row) and cortical_role(row) in HEMISPHERES.values():
        return "hemisphere"
    if row["kind"] in {"skull", "bone"}:
        return "skull"
    if row["kind"] == "brain":
        return "brain"
    return None


def _lattice_faces(origin, vector, shape, depth):
    """Ray distances at native voxel faces, bounded independently of source grid size."""
    distances = [0., depth]
    for coord, step, size in zip(origin, vector, shape):
        if abs(step) < 1e-12:
            continue
        end = coord + depth * step
        lo = max(0, int(np.ceil(min(coord, end) + .5)))
        hi = min(size, int(np.floor(max(coord, end) + .5)))
        if hi - lo >= MAX_FACES_PER_AXIS:
            raise ValueError("Entry-source lattice exceeds the bounded ray budget")
        distances.extend((np.arange(lo, hi + 1) - .5 - coord) / step)
    distances = np.sort(np.clip(distances, 0, depth))
    return distances[np.r_[True, np.diff(distances) > 1.e-8]]


def _entry(row, source, data, inverse, source_from_label, target, direction, depth):
    origin, vector = world(inverse, target), inverse[:3, :3] @ direction
    faces = _lattice_faces(origin, vector, data.shape, depth)
    points = origin + ((faces[1:] + faces[:-1]) * .5)[:, None] * vector
    cells = np.floor(points + .5).astype(int)
    lower, upper = inference_bounds(row) or (np.zeros(3), np.asarray(data.shape) - 1)
    known = np.all((cells >= lower) & (cells <= upper), axis=1)
    for samples in (cells, points):
        native = world(source_from_label, samples)
        known &= np.all((native >= -.5) & (native < np.asarray(source["data"]["shape"]) - .5), axis=1)
    slices = row.get("coverage_fields", {}).get("supported_axial_indices")
    if slices is not None:
        if not isinstance(slices, list) or any(type(z) is not int or not 0 <= z < data.shape[2] for z in slices):
            raise ValueError("Invalid entry-source slice coverage")
        known &= np.isin(cells[:, 2], slices)
    positive = np.zeros(len(cells), bool)
    positive[known] = data[tuple(cells[known].T)] == row["label_value"]
    occupied = np.flatnonzero(positive)
    if not len(occupied) or (_source_kind(row) == "hemisphere" and not positive[0]):
        return None
    i = occupied[-1]
    # A labelled field edge or the search distance limit is not evidence
    # of an exterior surface. Require a separate observed exterior cell.
    if i + 1 == len(cells) or not known[i + 1] or np.count_nonzero(cells[i] != cells[i + 1]) != 1:
        return None
    return target + faces[i + 1] * direction, cells[i].tolist()


def entry_windows(geometry, base):
    """Use at most three aligned sources per tissue class and one point per axis."""
    c = geometry.c
    target_row = next((r for r in geometry.layers if r["id"] == base.get("targetLayerId")), None)
    if not target_row:
        raise ValueError("Select a source-bound lesion label for distinct entries")
    timepoint = c.series(target_row["source_series"])["timepoint"]
    target = np.asarray(base["target"])
    depth = min(base["constraints"].get("maxLengthMm", MAX_DEPTH_MM), MAX_DEPTH_MM)
    candidates, issues = {}, []
    for kind in ("hemisphere", "skull", "brain"):
        rows = sorted((row for row in geometry.layers if _source_kind(row) == kind),
                      key=lambda row: (row["source_series"] != geometry.reference["id"], row["id"]))
        inspected = 0
        for row in rows:
            if row["id"] in geometry.context_reasons:
                issues.append(f"{row['label']}: {geometry.context_reasons[row['id']]}")
                continue
            source = c.series(row["source_series"])
            if (source["timepoint"] != timepoint or row.get("availability") == "empty_output"
                    or row.get("evidence_status") == "empty-output"):
                continue
            if inspected >= 3 or len(candidates) == len(DIRECTIONS):
                break
            try:
                layer_frame = c.patient_affine(row)
                source_from_label = np.linalg.solve(c.patient_affine(source), layer_frame)
                inverse = np.linalg.inv(layer_frame)
                data = c.array(row["data"])
                inspected += 1
                for name, axis in DIRECTIONS.items():
                    geometry.check()
                    if name in candidates:
                        continue
                    direction = np.asarray(axis) / np.linalg.norm(axis)
                    found = _entry(row, source, data, inverse, source_from_label, target, direction, depth)
                    if found is None:
                        continue
                    entry, voxel = found
                    support = {"direction": name, "kind": kind, "layerId": row["id"], "label": row["label"],
                               "sourceSeries": source["id"], "sourceSha256": source["original"]["sha256"],
                               "dataSha256": row["data"]["sha256"], "revision": row.get("revision", 0),
                               "labelValue": row["label_value"], "voxel": voxel,
                               "model": row.get("provenance", {}).get("model"), "reviewStatus": row.get("review_status")}
                    candidates[name] = (entry, support)
            except InterruptedError:
                raise
            except (ValueError, OSError) as error:
                issues.append(f"{row['label']}: {error}")
    for name in DIRECTIONS:
        if name not in candidates:
            issues.append(f"{name}: no observed exterior face within supported skull or brain coverage")
    return list(candidates.values()), issues


def verify_window(geometry, route):
    """Reassess a saved surface route only if its original labelled face still supports it."""
    support = route.get("tradeoff", {}).get("entrySupport", {})
    windows, _ = entry_windows(geometry, route)
    if not any(current == support and np.allclose(entry, route["entry"], atol=1e-6, rtol=0)
               for entry, current in windows):
        raise ValueError("The saved entry surface changed. Suggest distinct entries again for the current anatomy.")
