"""Plot saved scalar results only; does not run a model or inspect patient data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
report = json.loads((HERE / "run-01/report.json").read_text())
rows = [row for row in report["rows"] if row["spacing_mm"] == .25]
labels = [row["row_id"].replace("-0.25mm", "").replace("half-phase", "phase")
          .replace("-", " ").title() for row in rows]
gain = [100 * row["partial_evacuation"]["lower_bound_gain_mm3"] /
        row["partial_evacuation"]["gap_before_mm3"] for row in rows]
old_error = [100 * (1 - row["native"]["removed_physical_target_mm3"] /
                   row["exact_continuous_target_cap_mm3"]) for row in rows]
new_error = [100 * (1 - row["partial_evacuation"]["physical_target_evacuated_mm3"] /
                   row["exact_continuous_target_cap_mm3"]) for row in rows]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={"width_ratios": [1, 1.35]})
fig.subplots_adjust(left=.18, right=.98, bottom=.28, top=.73, wspace=.24)
fig.suptitle("Partial removal is represented more accurately", x=.05, ha="left",
             y=.97, fontsize=19, fontweight="bold", color="#153b46")
fig.text(.05, .865, "Fixed generated cuts on 0.25 mm grids · same 13/18 completed sequences · unchanged rewards",
         fontsize=11, color="#45636c")
y = np.arange(len(rows))
axes[0].barh(y, gain, color="#188780", height=.63)
axes[0].set_yticks(y, labels)
axes[0].invert_yaxis()
axes[0].set_xlim(0, 48)
axes[0].set_title("Smaller unresolved volume interval", loc="left", fontsize=11, pad=14)
axes[0].set_xlabel("Reduction in U − L (%)")
for pos, value in zip(y, gain):
    axes[0].text(value+.8, pos, f"{value:.1f}%", va="center", fontsize=9)
axes[1].barh(y-.17, old_error, height=.32, color="#a4b0b5", label="Whole cells only")
axes[1].barh(y+.17, new_error, height=.32, color="#188780", label="Certified subcells included")
axes[1].set_yticks(y, [""]*len(rows))
axes[1].invert_yaxis()
axes[1].set_xlim(0, 90)
axes[1].set_title("Target-volume undercount remains", loc="left", fontsize=11, pad=14)
axes[1].set_xlabel("Shortfall against ideal continuous target cap (%)")
for pos, before, after in zip(y, old_error, new_error):
    axes[1].text(before+.9, pos-.17, f"{before:.1f}", va="center", fontsize=9)
    axes[1].text(after+.9, pos+.17, f"{after:.1f}", va="center", fontsize=9)
handles, legend_labels = axes[1].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(.58, .165),
           frameon=False, fontsize=9)
for ax in axes:
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e4eaec", linewidth=.7)
fig.text(.05, .06,
         "Independent analytic geometry reference; idealized cut-and-evacuation assumption.\n"
         "U is the fixed planned-contact upper bound; L is certified physical volume. No patient or cutting-force validation.",
         fontsize=9, color="#45636c", linespacing=1.6)
fig.savefig(HERE / "volume-comparison.png", dpi=160, facecolor="white")
