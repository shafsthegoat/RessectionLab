#!/usr/bin/env python3
"""Render a source-image/mask contact sheet from a completed model record."""
from pathlib import Path
import argparse
import json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    if record["status"] != "succeeded":
        raise ValueError("Only completed inference can be displayed")
    source = nib.as_closest_canonical(nib.load(record["input"]["path"]))
    values = source.get_fdata(dtype=np.float32)
    columns = len(record["outputs"])+1
    fig, axes = plt.subplots(3, columns, figsize=(5*columns, 12), facecolor="#09121b")
    if record["input"]["modality"] == "CT":
        low, high = 0, 1500
    else:
        low, high = np.percentile(values[values>0], [1,99])
    masks = [nib.as_closest_canonical(nib.load(x["path"])).get_fdata(dtype=np.float32) for x in record["outputs"]]
    center = [int(x//2) for x in source.shape]
    for row, axis in enumerate([2,1,0]):
        source_slice = np.take(values, center[axis], axis=axis).T
        remaining = [i for i in range(3) if i != axis]
        aspect = source.header.get_zooms()[remaining[1]]/source.header.get_zooms()[remaining[0]]
        for col in range(columns):
            ax = axes[row,col]
            ax.imshow(source_slice, cmap="gray", origin="lower", vmin=low, vmax=high, aspect=aspect)
            if col:
                overlay = np.take(masks[col-1], center[axis], axis=axis).T
                rgba = np.zeros(overlay.shape+(4,), dtype=np.float32)
                rgba[...,1:3] = 1
                rgba[...,3] = overlay*0.55
                ax.imshow(rgba, origin="lower", aspect=aspect)
            ax.set_title((record["outputs"][col-1]["name"] if col else "Original scan")+f" · {['Axial', 'Coronal', 'Sagittal'][row]}", color="white")
            ax.axis("off")
    fig.suptitle(f"{record['case']} · {record['model']['name']}\nActual local inference · unreviewed segmentation · native geometry preserved", color="white", fontsize=15)
    fig.tight_layout(rect=(0,0,1,0.94))
    output = args.record.parent / "source-mask-qc.png"
    fig.savefig(output, dpi=120, facecolor=fig.get_facecolor())
    print(output)


if __name__ == "__main__":
    main()
