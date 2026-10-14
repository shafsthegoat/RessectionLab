"""Compare inference settings and thresholds without claiming anatomical accuracy."""

import argparse, json
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from resectionlab.patient_anatomy import file_hash

p = argparse.ArgumentParser()
p.add_argument("--gaussian", required=True)
p.add_argument("--constant", required=True)
p.add_argument("--baseline", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
records = {}
for name, path in [
    ("native_gaussian", a.gaussian),
    ("native_constant", a.constant),
    ("coarse_baseline", a.baseline),
]:
    r = json.loads(Path(path).read_text())
    prob = r.get("probability")
    mask = r["outputs"][0]
    im = nib.load(prob["path"] if prob else mask["path"])
    values = im.get_fdata(dtype=np.float32)
    spacing = nib.affines.voxel_sizes(im.affine)
    stats = []
    for t in [0.3, 0.5, 0.7] if prob else [0.5]:
        binary = values >= t
        components, n = ndi.label(binary, np.ones((3, 3, 3)))
        sizes = np.bincount(components.ravel())[1:]
        stats.append(
            {
                "threshold": t,
                "voxels": int(binary.sum()),
                "volume_ml": float(
                    binary.sum() * abs(np.linalg.det(im.affine[:3, :3])) / 1000
                ),
                "components_26": int(n),
                "largest_component_voxels": int(sizes.max(initial=0)),
            }
        )
    records[name] = {
        "receipt": str(Path(path).resolve()),
        "receipt_sha256": file_hash(path),
        "source_shape": r["input"]["shape"],
        "grid_mm": spacing.tolist(),
        "elapsed_seconds": r["elapsed_seconds"],
        "measurements": stats,
    }
Path(a.output).write_text(
    json.dumps(
        {
            "interpretation": "Threshold and blending comparison only. No reference vessel annotations; recall, precision and accuracy remain unmeasured. No brain ROI clipping or tumor subtraction in native outputs.",
            "runs": records,
        },
        indent=2,
    )
    + "\n"
)
print(a.output)
