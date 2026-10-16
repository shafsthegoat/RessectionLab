"""Annotation-bound geometry evaluation; absent annotations yield no accuracy score."""

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from .patient_anatomy import affine, world


def evaluate(
    predicted, reference, source_affine, *, tolerance_mm=0.5, annotation_coverage=None
):
    if reference is None:
        return {
            "status": "visual_review_required",
            "accuracy": None,
            "reason": "No reference annotation supplied",
        }
    a = affine(source_affine)
    p = np.array(predicted, bool, copy=True)
    r = np.array(reference, bool, copy=True)
    if p.shape != r.shape or p.ndim != 3 or not 0 <= tolerance_mm <= 10:
        raise ValueError("Matching annotation grids and finite tolerance required")
    coverage = (
        np.ones(r.shape, bool)
        if annotation_coverage is None
        else np.asarray(annotation_coverage, bool)
    )
    if coverage.shape != r.shape:
        raise ValueError("Coverage grid differs")
    p &= coverage
    r &= coverage

    def coords(mask):
        return world(a, np.argwhere(mask))

    def distances(first, second):
        x, y = coords(first), coords(second)
        return np.full(len(x), np.inf) if not len(y) else cKDTree(y).query(x)[0]

    ps = p & ~ndi.binary_erosion(p)
    rs = r & ~ndi.binary_erosion(r)
    dp, dr = distances(ps, rs), distances(rs, ps)
    skeleton_p = skeletonize(p)
    skeleton_r = skeletonize(r)
    cp, cr = distances(skeleton_p, skeleton_r), distances(skeleton_r, skeleton_p)
    result = {
        "status": "annotation_comparison",
        "tolerance_mm": tolerance_mm,
        "boundary_sampling": "boundary voxel centers; resolution-limited",
        "boundary_mean_mm": float(np.mean(np.concatenate([dp, dr])))
        if len(dp) and len(dr)
        else None,
        "boundary_hd95_mm": float(np.percentile(np.concatenate([dp, dr]), 95))
        if len(dp) and len(dr)
        else None,
        "centerline_precision": float(np.mean(cp <= tolerance_mm)) if len(cp) else None,
        "centerline_recall": float(np.mean(cr <= tolerance_mm)) if len(cr) else None,
        "predicted_components_26": int(ndi.label(p, np.ones((3, 3, 3)))[1]),
        "reference_components_26": int(ndi.label(r, np.ones((3, 3, 3)))[1]),
        "review_requirement": "annotation provenance and field coverage determine interpretation",
    }
    spacing = np.linalg.norm(a[:3, :3], axis=0)
    if np.allclose(
        (a[:3, :3] / spacing).T @ (a[:3, :3] / spacing), np.eye(3), atol=1e-5
    ):
        diameter = 2 * ndi.distance_transform_edt(r, sampling=spacing)[skeleton_r]
        bins = [0, 1, 2, 4, 8, float("inf")]
        result["diameter_bins"] = []
        for low, high in zip(bins, bins[1:]):
            select = (diameter >= low) & (diameter < high)
            result["diameter_bins"].append(
                {
                    "min_mm": low,
                    "max_mm": None if np.isinf(high) else high,
                    "reference_points": int(select.sum()),
                    "centerline_recall": float(np.mean(cr[select] <= tolerance_mm))
                    if select.any()
                    else None,
                }
            )
    # Reference branch segments are skeleton paths between junction neighborhoods.
    degree = (
        ndi.convolve(
            skeleton_r.astype(np.uint8), np.ones((3, 3, 3), np.uint8), mode="constant"
        )
        - skeleton_r
    )
    paths, count = ndi.label(skeleton_r & (degree <= 2), np.ones((3, 3, 3)))
    supported = np.zeros(r.shape, bool)
    supported[skeleton_r] = cr <= tolerance_mm
    branches = []
    for index in range(1, count + 1):
        branch = paths == index
        if branch.sum() >= 3:
            branches.append(bool(np.all(supported[branch])))
    result["reference_branch_segments"] = len(branches)
    result["complete_branch_segment_recall"] = (
        float(np.mean(branches)) if branches else None
    )
    return result
