"""Deterministic engineering lesions in physical units, with explicit limits.

These compartments and intensities are synthetic scenarios, not glioma biology,
segmentation ground truth for acquired disease, or calibrated mass effect.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

RECIPE = "physical-irregular-compartments-v2"


def geometry(brain, affine, *, seed, variant=0, placement_support=None):
    brain = np.asarray(brain, bool)
    affine = np.asarray(affine, float)
    if brain.ndim != 3 or not brain.any() or affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("A finite physical grid and nonempty host brain estimate are required")
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    directions = affine[:3, :3] / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-4):
        raise ValueError("This recipe supports orthogonal native grids, including oblique rotations")
    rng = np.random.default_rng(seed)
    support=brain if placement_support is None else np.asarray(placement_support,bool)
    if support.shape!=brain.shape or np.any(support & ~brain):raise ValueError('Placement support must be a subset of the declared brain estimate')
    requested_radii = np.array([6., 8., 10.]) * (1. + .15 * (variant % 3))
    distance = distance_transform_edt(support, sampling=spacing)
    # One host-level scale preserves the requested size differences. Scaling
    # each variant to the maximum fitting radius would collapse their sizes
    # and force every center into the same deepest support cell.
    fit_scale=min(1.,(float(distance.max())-1.6)/(13.*1.6))
    if fit_scale<.5:raise ValueError('No placement support for the minimum engineering lesion dimensions')
    radii=requested_radii*fit_scale
    margin = float(radii.max() * 1.6 + 1.5)
    candidates = np.argwhere(distance >= margin)
    if not len(candidates):
        raise ValueError("Host estimate has no interior supporting the untruncated recipe")
    # Alternating hemispheric/anteroposterior placements in actual RAS coordinates;
    # these normalized locations do not establish a named lobe or tissue type.
    world = candidates @ affine[:3, :3].T + affine[:3, 3]
    fractions = [(0.25, .65, .65), (.75, .65, .5), (.3, .35, .45), (.7, .4, .7)]
    desired = world.min(0) + np.ptp(world, axis=0) * fractions[variant % len(fractions)]
    center = candidates[np.argmin(np.sum((world - desired) ** 2, axis=1))]
    center_ras = affine[:3, :3] @ center + affine[:3, 3]
    # Allocate only the lesion ROI. A sphere enclosing the largest axis bounds
    # oblique ellipsoids, so the crop cannot clip a compartment.
    half = np.ceil((radii.max() * 1.7 + 2.) / spacing).astype(int)
    low = np.maximum(center - half, 0)
    high = np.minimum(center + half + 1, brain.shape)
    grids = np.meshgrid(*(np.arange(a, b) - c for a, b, c in zip(low, high, center)), indexing="ij")
    delta = np.stack(grids, -1) @ affine[:3, :3].T
    # A seeded rigid orientation changes elongation without stretching the host.
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    local = delta @ rotation / radii
    radial = np.linalg.norm(local, axis=-1)
    phase = rng.uniform(-np.pi, np.pi, 3)
    perturbation = 1. + .09 * np.sin(local[..., 0] * 2. + phase[0]) * np.cos(local[..., 1] * 2.5 + phase[1]) * np.sin(local[..., 2] * 1.7 + phase[2])
    q = radial / perturbation
    roi = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    compartments = {}
    for name, mask in {"synthetic_core": q <= .65, "synthetic_rim": (q > .65) & (q <= 1.),
                       "synthetic_edema": (q > 1.) & (q <= 1.5)}.items():
        full = np.zeros(brain.shape, bool)
        full[roi] = mask
        if not full.any() or np.any(full & ~support):
            raise ValueError("Synthetic compartment is empty or escapes the host estimate; no clipping repair")
        if any(np.take(full, [0, n-1], axis=a).any() for a, n in enumerate(brain.shape)):
            raise ValueError("Synthetic compartment touches image boundary")
        compartments[name] = full
    return compartments, {"recipe": RECIPE, "seed": int(seed), "variant": int(variant),
        "center_voxel": center.tolist(), "center_ras_mm": center_ras.tolist(), "radii_mm": radii.tolist(),
        "requested_radii_mm":requested_radii.tolist(),"support_fit_scale":fit_scale,
        "rotation_ras": rotation.tolist(), "phase": phase.tolist(), "placement": "normalized RAS bounds of admitted interior; no lobe inference",
        "brain_margin_mm": margin, "outside_brain_voxels": 0, "donors": [],
        "clinical_parameter_distribution": False, "tissue_class_constraint": "brain-envelope estimate only" if placement_support is None else "conservative high-T1-signal support estimate; not validated parenchyma or GM/WM/CSF classification"}


def appearance(baseline, brain, compartments, *, seed, modality="T1"):
    if modality not in {"T1", "T2", "PD"}:
        raise ValueError("Only explicitly supported acquired structural sequences may be synthesized")
    image = np.asarray(baseline, np.float32)
    brain = np.asarray(brain, bool)
    values = image[brain]
    low, high = np.percentile(values, [5, 95])
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("Host has no usable intensity range")
    union = np.logical_or.reduce(list(compartments.values()))
    support = gaussian_filter(union.astype(np.float32), .7)
    # Seeded smooth texture is shared spatially across sequences. Sequence-specific
    # levels describe this engineering observation model, not acquired contrast.
    rng = np.random.default_rng(seed)
    noise = gaussian_filter(rng.normal(size=image.shape).astype(np.float32), 1.3)
    noise /= max(float(noise.std()), 1e-8)
    levels = {"T1": (.20, .42, .52), "T2": (.8, .7, .88), "PD": (.65, .6, .72)}[modality]
    signal = image.copy()
    for (name, mask), level in zip(compartments.items(), levels):
        signal[mask] = low + (high - low) * (level + .035 * noise[mask])
    mixed = image * (1. - support) + signal * support
    # The smoothing support blends only existing compartment signal; pixels
    # outside it remain exactly acquired. No synthetic T1ce/FLAIR is invented.
    if not np.isfinite(mixed).all():
        raise ValueError("Nonfinite synthetic observation")
    return mixed.astype(np.float32), {"modality": modality, "model": "shared compartments with seeded smooth intensity texture and soft boundary blend",
        "levels_fraction_of_host_p05_p95": list(levels), "host_percentiles": [float(low), float(high)],
        "seed": int(seed), "modified_voxels": int(np.count_nonzero(mixed != image)), "clinical_appearance_validated": False}


def mild_deformation(image, brain, affine, compartments, *, center_ras_mm, amplitude_mm=1.):
    """One backwards pull field for image and labels; zero at brain boundary.

    No measured skull, vessels, tracts or functional structures are available to
    this function. Callers must withhold those components unless transformed by
    this exact field. This approximation is deliberately small and uncalibrated.
    """
    if not 0 < amplitude_mm <= 2.:
        raise ValueError("Initial deformation amplitude must be in (0,2] mm")
    brain = np.asarray(brain, bool)
    affine = np.asarray(affine, float)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    grid = np.indices(brain.shape, dtype=np.float32)
    world = np.einsum("ij,j...->i...", affine[:3, :3].astype(np.float32), grid) + affine[:3, 3, None, None, None].astype(np.float32)
    delta = world - np.asarray(center_ras_mm, np.float32)[:, None, None, None]
    radius = np.sqrt(np.sum(delta ** 2, axis=0))
    distance = distance_transform_edt(brain, sampling=spacing)
    t = np.clip((distance - float(spacing.max())) / 12., 0, 1)
    taper = t * t * (3 - 2 * t)
    scale = amplitude_mm * np.exp(-radius**2 / (2 * 22.**2)) * taper / np.maximum(radius, 8.)
    backward_world = -delta * scale
    backward_voxel = np.einsum("ij,j...->i...", np.linalg.inv(affine[:3, :3]).astype(np.float32), backward_world.astype(np.float32))
    pull = grid + backward_voxel
    # Compute determinant of d(source coordinates)/d(output coordinates). This
    # dimensionless determinant is invariant under the fixed affine basis.
    gradients = [[np.gradient(backward_voxel[i], axis=j) for j in range(3)] for i in range(3)]
    a,b,c = gradients[0]; d,e,f = gradients[1]; g,h,k = gradients[2]
    a=a+1; e=e+1; k=k+1
    jacobian = a*(e*k-f*h)-b*(d*k-f*g)+c*(d*h-e*g)
    minimum, maximum = float(jacobian.min()), float(jacobian.max())
    if minimum < .2 or maximum > 5.:
        raise ValueError(f"Deformation Jacobian outside prospective engineering range: {minimum}, {maximum}")
    # Numerically invert the same map by fixed point iteration, independently of
    # image sampling; residual is evaluated only in moving tissue, away from FOV.
    inverse = grid.copy()
    for _ in range(12):
        inverse = grid - np.stack([map_coordinates(v, inverse, order=1, mode="nearest", prefilter=False) for v in backward_voxel])
    composed = inverse + np.stack([map_coordinates(v, inverse, order=1, mode="nearest", prefilter=False) for v in backward_voxel]) - grid
    residual_world = np.einsum("ij,j...->i...", affine[:3, :3].astype(np.float32), composed)
    residual = float(np.sqrt(np.sum(residual_world**2, axis=0))[brain].max())
    if residual > .5 * float(spacing.min()):
        raise ValueError("Forward/inverse deformation residual exceeds half a native voxel")
    moved_brain = map_coordinates(brain.astype(np.uint8), pull, order=0, mode="constant", cval=0, prefilter=False) > 0
    moved = {name: map_coordinates(mask.astype(np.uint8), pull, order=0, mode="constant", cval=0, prefilter=False) > 0
             for name, mask in compartments.items()}
    if any(not mask.any() or np.any(mask & ~moved_brain) for mask in moved.values()):
        raise ValueError("Transformed compartment escapes transformed brain support")
    output = map_coordinates(np.asarray(image, np.float32), pull, order=1, mode="constant", cval=0, prefilter=False)
    return output, moved_brain, moved, backward_world.astype(np.float32), {
        "model": "bounded smooth radial backward displacement with fixed brain-envelope boundary; uncalibrated engineering approximation",
        "amplitude_parameter_mm": float(amplitude_mm), "jacobian_min": minimum, "jacobian_max": maximum,
        "inverse_residual_max_mm": residual, "fixed_boundary_max_mm": float(np.linalg.norm(backward_world[:, distance <= spacing.max()], axis=0).max()),
        "field_direction": "output RAS mm to baseline RAS mm: source=output+backward_displacement",
        "interpolation": {"image": "linear", "labels": "nearest"}, "skull": "missing; no measured skull moved",
        "withheld_components": ["MRA baseline", "vessels", "tracts", "functional anatomy"], "physiological_validation": False}
