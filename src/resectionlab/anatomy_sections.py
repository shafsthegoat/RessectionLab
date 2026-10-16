"""Physical-coordinate oblique and registration checkerboard inspection."""

import numpy as np
from scipy import ndimage as ndi
from .patient_anatomy import affine, world


def section(
    collection,
    *,
    series_id,
    center,
    normal,
    spacing_mm=0.5,
    size=512,
    moving_id=None,
    checkerboard=True,
    layers=(),
):
    c = collection
    row = c.series(series_id)
    if (
        not isinstance(size, int)
        or not 16 <= size <= 1024
        or not 0.1 <= spacing_mm <= 5
    ):
        raise ValueError("Section sampling limit exceeded")
    centre = np.asarray(center, float)
    n = np.asarray(normal, float)
    if (
        centre.shape != (3,)
        or n.shape != (3,)
        or not np.isfinite([*centre, *n]).all()
        or np.linalg.norm(n) < 1e-6
    ):
        raise ValueError("Invalid section plane")
    n = n / np.linalg.norm(n)
    anchor = np.eye(3)[np.argmin(abs(n))]
    u = np.cross(n, anchor)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    y, x = np.indices((size, size))
    points = (
        centre
        + (x[..., None] - (size - 1) / 2) * spacing_mm * u
        + (y[..., None] - (size - 1) / 2) * spacing_mm * v
    )
    try:
        a = c.patient_affine(row)
        frame = "patient"
    except ValueError:
        a = affine(row["affine"])
        frame = "independent source"

    def sample(data, transform, order):
        indices = world(np.linalg.inv(transform), points).reshape(-1, 3).T
        valid = np.all(indices >= 0, axis=0) & np.all(
            indices <= np.array(data.shape)[:, None] - 1, axis=0
        )
        values = ndi.map_coordinates(
            data, indices, order=order, mode="constant", cval=0, prefilter=False
        ).reshape(size, size)
        return values, valid.reshape(size, size)

    def image(source, transform):
        data = c.array(source["data"])
        values, covered = sample(data, transform, 1)
        finite = np.asarray(data).ravel()[:: max(1, data.size // 20000)]
        positive = finite[finite != 0]
        lo, hi = np.percentile(positive if len(positive) else [0, 1], [0.5, 99.5])
        gray = np.clip((values - lo) / max(hi - lo, 1e-6), 0, 1)
        return np.repeat(gray[..., None], 3, axis=2), covered

    rgb, coverage = image(row, a)
    moving_status = None
    if moving_id:
        if frame != "patient":
            raise ValueError("Checkerboard reference must be in the patient frame")
        moving = c.series(moving_id)
        t = c.document["transforms"].get(moving_id)
        if not t:
            raise ValueError(
                "Estimate a source-to-patient transform before checkerboard review"
            )
        second, covered = image(moving, affine(t["matrix"]) @ affine(moving["affine"]))
        moving_status = t["status"]
        choose = ((x // 32 + y // 32) % 2).astype(bool)
        if checkerboard:
            rgb[choose] = second[choose]
            coverage[choose] = covered[choose]
        else:
            rgb = 0.5 * (rgb + second)
            coverage &= covered
    for name in layers[:8]:
        layer = c.layer(name)
        try:
            transform = (
                c.patient_affine(layer)
                if frame == "patient"
                else affine(layer["affine"])
            )
        except ValueError:
            continue
        if frame != "patient" and layer["source_series"] != series_id:
            continue
        values, covered = sample(c.array(layer["data"]), transform, 0)
        if layer.get("score"):
            scores = c.array(layer["score"])
            score, score_covered = sample(scores, transform, 1)
            scale = float(np.max(np.abs(scores)))
            if scale > 0:
                alpha = np.clip(np.abs(score) / scale, 0, 1)[..., None] * 0.65
                tint = np.where(
                    (score >= 0)[..., None],
                    np.array([1.0, 0.35, 0.05]),
                    np.array([0.05, 0.4, 1.0]),
                )
                alpha *= score_covered[..., None]
                rgb = rgb * (1 - alpha) + tint * alpha
        inside = (values == layer["label_value"]) & covered
        contour = inside & ~ndi.binary_erosion(inside)
        color = layer["color"].lstrip("#")
        rgb[contour] = [int(color[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    rgb[~coverage] = [0.12, 0.06, 0.16]
    rgba = np.concatenate(
        [np.rint(rgb * 255).astype(np.uint8), np.full((size, size, 1), 255, np.uint8)],
        axis=2,
    )
    return {
        "pixels": rgba,
        "center": centre.tolist(),
        "u": u.tolist(),
        "v": v.tolist(),
        "spacingMm": spacing_mm,
        "size": size,
        "frame": frame,
        "movingStatus": moving_status,
        "coverageMeaning": "purple pixels are outside sampled image support",
        "snapshotId": c.identity(),
    }
