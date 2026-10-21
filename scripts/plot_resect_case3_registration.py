#!/usr/bin/env python3
"""Plot the fixed Case3 comparison from its coordinate-free scalar summary."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = args.summary.read_bytes()
    result = json.loads(payload)
    if (result.get("status") != "scored_all_full_tag_points"
            or result.get("case_id") != "RESECT:Case3"
            or result.get("role") != "TRAIN"
            or result.get("excluded_point_indices") != []):
        raise ValueError("Complete fixed Case3 TRAIN summary required")
    rows = result["rows"]
    indices = np.array([row["point_index"] for row in rows])
    identity = np.array([row["identity_TRE_mm"] for row in rows], dtype=float)
    rigid = np.array([row["rigid_TRE_mm"] for row in rows], dtype=float)
    if (len(rows) != result["point_count"]
            or indices.tolist() != list(range(1, len(rows) + 1))
            or not all(row["retained_in_all_statistics"] for row in rows)
            or not np.isfinite(np.r_[identity, rigid]).all()):
        raise ValueError("Every finite landmark must be retained in source order")
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
    colors = ("#345e8f", "#b34b30")
    ax = axes[0]
    ax.plot(indices, identity, "o-", color=colors[0], label="Identity (unchanged scans)", lw=1.5, ms=4)
    ax.plot(indices, rigid, "s-", color=colors[1], label="Image-only rigid fit", lw=1.5, ms=4)
    ax.set(xlabel="Published landmark row", ylabel="Target-registration error (mm)",
           title=f"All {len(rows)} paired landmarks", xticks=indices, ylim=(0, None))
    ax.grid(axis="y", alpha=.18)
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax = axes[1]
    names = ("RMS", "Median", "Maximum")
    x = np.arange(len(names))
    for offset, method, color, label in ((-.18, "identity", colors[0], "Identity"),
                                         (.18, "rigid", colors[1], "Rigid")):
        values = [result[method][name + "_mm"] for name in ("RMS", "median", "max")]
        bars = ax.bar(x + offset, values, width=.34, color=color, label=label)
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3, fontsize=9)
    ax.set(xticks=x, xticklabels=names, ylabel="Error (mm)", title="Within-case summary")
    ax.set_ylim(0, max(result["identity"]["max_mm"], result["rigid"]["max_mm"]) * 1.22)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    delta = result["primary_delta_RMS_mm_rigid_minus_identity"]
    fig.suptitle("RESECT Case3: fixed image-only alignment comparison", x=.06, ha="left", fontsize=15)
    fig.text(.06, .035,
             f"One TRAIN person; every landmark retained. Rigid − identity RMS: {delta:+.3f} mm.\n"
             "Landmarks were not used to fit the transform. No population or clinical-validity claim.",
             fontsize=9, color="#444444", va="bottom")
    fig.tight_layout(rect=(.035, .13, .99, .92), w_pad=2.2)
    names_out = ("alignment-comparison.png", "alignment-comparison.svg")
    for name in names_out:
        output_path = args.output / name
        fig.savefig(output_path, dpi=180, facecolor="white")
        if output_path.suffix == ".svg":
            output_path.write_text("\n".join(line.rstrip() for line in output_path.read_text().splitlines()) + "\n")
    plt.close(fig)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    record = {"schema": "resect-case3-registration-scalar-figure-v1",
              "summary_path": str(args.summary), "summary_sha256": hashlib.sha256(payload).hexdigest(),
              "plotter_sha256": sha(Path(__file__)), "matplotlib_version": matplotlib.__version__,
              "numpy_version": np.__version__, "point_count": len(rows),
              "coordinate_payloads_read": False, "image_arrays_read": False,
              "all_source_rows_retained": True,
              "outputs": {name: sha(args.output / name) for name in names_out}}
    (args.output / "figure-record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "plotted_coordinate_free_summary", "output": str(args.output)}))


if __name__ == "__main__":
    main()
