#!/usr/bin/env python3
"""Display the acquired pilot's central native planes without registration.

Axis flips/permutation preserve acquired samples; there is no resampling,
segmentation or generated anatomy. This is display QC, not clinical review.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from acquire_lausanne_pilot import DATA, RESULT, ROOT, preserve, json_bytes, sha256_file


def main() -> None:
    report = json.loads(RESULT.read_bytes())
    output = ROOT / "outputs/lausanne-original-pilot-v1/midplanes.png"
    receipt = ROOT / "artifacts/lausanne-original-pilot-v1/display.json"
    if output.exists() or receipt.exists():
        raise ValueError("Preserve existing display output and receipt")
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained", facecolor="#111827")
    records = []
    for row, item in enumerate(report["files"][:2]):
        source = DATA / item["path"]
        if sha256_file(source) != item["sha256"]:
            raise ValueError("Acquired image bytes changed")
        image = nib.as_closest_canonical(nib.load(source))
        if int(np.prod(image.shape)) * 4 > 512 * 1024**2:
            raise ValueError("Image exceeds local display memory bound")
        array = image.get_fdata(dtype=np.float32)
        spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
        positive = array[array > 0]
        low, high = np.percentile(positive, [1, 99])
        del positive
        midpoint = [length // 2 for length in array.shape]
        label = "T1" if row == 0 else "TOF angiography"
        for dimension, name in enumerate(["R", "A", "S"]):
            plane = np.take(array, midpoint[dimension], axis=dimension).T
            in_plane = [axis for axis in range(3) if axis != dimension]
            extent = [0, plane.shape[1] * spacing[in_plane[0]], 0, plane.shape[0] * spacing[in_plane[1]]]
            axis = axes[row, dimension]
            axis.imshow(plane, origin="lower", cmap="gray", vmin=low, vmax=high, extent=extent, aspect="equal")
            axis.set_title(f"{label} · perpendicular to canonical {name} axis", color="white", fontsize=10)
            axis.set_xlabel("mm along voxel axis", color="#d1d5db", fontsize=8)
            axis.set_ylabel("mm along voxel axis", color="#d1d5db", fontsize=8)
            axis.tick_params(colors="#d1d5db", labelsize=7)
        records.append({"path": item["path"], "sha256": item["sha256"],
                        "canonical_midpoint_indices": midpoint, "canonical_affine": image.affine.tolist(),
                        "window": [float(low), float(high)], "window_rule": "1st/99th percentile of positive acquired voxels",
                        "canonical_spacing_mm": spacing.tolist()})
        del array
    figure.suptitle("Lausanne sub-000 · original acquired T1 / TOF\nCentral voxel planes; obliquity retained. Coverage and interscan registration unassessed.",
                    color="white", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140, facecolor=figure.get_facecolor())
    plt.close(figure)
    preserve(receipt, json_bytes({"source_acquisition_sha256": sha256_file(RESULT),
        "script_sha256": sha256_file(Path(__file__)), "output": str(output.relative_to(ROOT)),
        "output_sha256": sha256_file(output), "files": records,
        "interpretation": "Display integrity only; no anatomical, vessel-label or registration acceptance"}))
    print(output)


if __name__ == "__main__":
    main()
