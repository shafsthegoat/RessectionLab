"""Source-score-supported vascular margins, separate from display fairing.

This deliberately cannot repair missing connections: additions must be simple
points under foreground-26/background-6 digital topology. Original positives,
including isolated small components, are retained without selection by size.
"""

import numpy as np
from scipy import ndimage as ndi
from .patient_anatomy import affine, digest, file_hash

VERSION = "source-score-simple-margin-v1"
FOREGROUND = np.ones((3, 3, 3), bool)
BACKGROUND = ndi.generate_binary_structure(3, 1)
BACKGROUND_NEIGHBORHOOD = ndi.generate_binary_structure(3, 2)


def fragmentation(mask):
    labels, count = ndi.label(mask, structure=FOREGROUND)
    sizes = np.bincount(labels.ravel())[1:]
    return {"voxels": int(np.count_nonzero(mask)), "components_26": int(count),
            "largest_component_voxels": int(sizes.max(initial=0)),
            "singleton_components": int(np.count_nonzero(sizes == 1))}


def refine_margin(mask, probability, native_affine, *, threshold=.35,
                  radius_mm=.8, support_mask=None, cancelled=lambda: False):
    """Recover supported margins with bounded motion and unchanged topology.

    A fixed shell around the input limits total growth, rather than repeatedly
    dilating the result. Sequential high-score-first admission prevents competing
    components from meeting. The local tests also reject branch shortcuts inside
    one component. Scores are uncalibrated estimates, never validated confidence.
    """
    original = np.asarray(mask, dtype=bool)
    support = original if support_mask is None else np.asarray(support_mask, dtype=bool)
    p = np.asarray(probability)
    if (original.ndim != 3 or support.shape != original.shape or p.shape != original.shape or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1))):
        raise ValueError("Finite source probabilities on the exact mask grid are required")
    if not np.isfinite([threshold, radius_mm]).all() or not 0 < threshold < .5 or not 0 < radius_mm <= 2:
        raise ValueError("Vessel margin requires a score in (0,.5) and radius in (0,2] mm")
    a = affine(native_affine)
    spacing = np.linalg.norm(a[:3, :3], axis=0)
    if not np.allclose((a[:3, :3]/spacing).T @ (a[:3, :3]/spacing), np.eye(3), atol=1e-5):
        raise ValueError("Vessel margin requires an orthogonal source grid")
    extent = np.floor(radius_mm / spacing).astype(int)
    if np.prod(2*extent+1) > 4096:
        raise ValueError("Vessel margin exceeds source-grid neighborhood budget")
    offsets = np.indices(tuple(2*extent+1)).transpose(1, 2, 3, 0)-extent
    shell = np.sum((offsets*spacing)**2, axis=-1) <= radius_mm**2 + 1e-12
    candidates = ndi.binary_dilation(support, structure=shell) & ~original & (p >= threshold)
    coordinates = np.argwhere(candidates)
    if len(coordinates) > 1_000_000:
        raise ValueError("Vessel margin exceeds bounded candidate budget")
    order = np.argsort(-p[tuple(coordinates.T)], kind="stable")
    padded = np.pad(original, 1)
    accepted = 0
    # For foreground-26/background-6 topology, background paths are measured in
    # N18 (faces and edges), excluding corners. At least one background face must
    # survive: an edge alone cannot preserve a face-connected background cavity.
    # Requiring all N18 background cells connected is deliberately conservative.
    for step, index in enumerate(order):
        if step % 2048 == 0 and cancelled():
            raise InterruptedError("Vessel margin cancelled")
        x, y, z = coordinates[index]
        cube = padded[x:x+3, y:y+3, z:z+3]
        if ndi.label(cube, structure=FOREGROUND)[1] != 1:
            continue
        background = ~cube & BACKGROUND_NEIGHBORHOOD
        background[1, 1, 1] = False
        if not np.any(background & BACKGROUND) or ndi.label(background, structure=BACKGROUND)[1] != 1:
            continue
        padded[x+1, y+1, z+1] = True
        accepted += 1
    result = padded[1:-1, 1:-1, 1:-1].copy()
    before, after = fragmentation(original), fragmentation(result)
    if after["components_26"] != before["components_26"] or np.any(original & ~result):
        raise RuntimeError("Vessel margin topology invariant failed")
    report = {"version": VERSION, "threshold": float(threshold), "radius_mm": float(radius_mm),
              "candidate_voxels": len(coordinates), "added_voxels": accepted,
              "withheld_voxels": len(coordinates)-accepted,
              "removed_voxels": 0, "before": before, "after": after,
              "meaning": "source-score-supported estimate; no gap bridging; scores uncalibrated; anatomical review pending",
              "display_smoothing": False, "clinical_validation": False}
    return result, report


def refine_layer(collection, row, mask=None, *, threshold=.35, radius_mm=.8,
                 cancelled=lambda: False):
    if row["kind"] not in {"vessels", "artery", "vein", "sinus"} or "probability" not in row:
        raise ValueError("A vascular layer with retained source scores is required")
    # patient_affine validates source-bound registration; content-addressed array
    # reads verify both the segmentation and score hashes.
    a = collection.patient_affine(row)
    source = collection.series(row["source_series"])["original"]
    if file_hash(collection.path(source["path"])) != source["sha256"]:
        raise ValueError("Acquired source image changed before vessel refinement")
    support_ref = row.get("original_data", row["data"])
    support = collection.array(support_ref) == row.get("original_label_value", row["label_value"])
    result, report = refine_margin(collection.mask(row) if mask is None else mask,
                                   collection.array(row["probability"]), a,
                                   threshold=threshold, radius_mm=radius_mm,
                                   support_mask=support,
                                   cancelled=cancelled)
    report.update(source_series=row["source_series"], layer_id=row["id"],
                  source_image_sha256=source["sha256"],
                  input_data_sha256=row["data"]["sha256"],
                  margin_support_data=support_ref,
                  probability_sha256=row["probability"]["sha256"],
                  input_revision=row["revision"], snapshot_id=collection.identity(),
                  parameters_id=digest({"version": VERSION, "threshold": threshold, "radius_mm": radius_mm}))
    return result, report
