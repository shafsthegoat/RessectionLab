"""Plot recorded results; restore raw episodes from completed-run.tar.gz first."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

folder = Path(__file__).resolve().parent
result = json.loads((folder / "receipt.json").read_text())
bc = json.loads((folder.parent / "pat05-real-geometric-imitation-v1" / "receipt.json").read_text())
rl = json.loads((folder.parent / "pat05-real-geometric-learning-v1" / "receipt.json").read_text())
control, augmented = [result["branches"][key]["latest"]["return"] for key in ("control", "augmented")]
values = [bc["latest_policy"]["return"], control, augmented, rl["episodes"]["greedy_search"]["return"]]
labels = ["Prior imitation\n8 updates", "Original only\n+8 updates", "Visited states\n+8 updates", "Greedy\nsearch"]
colors = ["#718096", "#C2A16A", "#196D91", "#207C63"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
axes[0].bar(np.arange(4), values, color=colors)
axes[0].set(xticks=np.arange(4), xticklabels=labels, ylabel="Geometric return (fixed research weights)", ylim=(0, 480))
axes[0].set_title("Matched continuation: +51.40 with added examples", loc="left", weight="bold", fontsize=11)
for index, value in enumerate(values):
    axes[0].text(index, value + 9, f"{value:.2f}", ha="center")
for key, offset, color, label in [("control", -.18, colors[1], "Original examples"),
                                   ("augmented", .18, colors[2], "Added visited states")]:
    episode = json.loads((folder / (key + "_latest.json")).read_text())
    axes[1].bar(np.arange(3) + offset, [d["reward"] for d in episode["decisions"]], .36, color=color, label=label)
axes[1].axhline(0, color="#999999", linewidth=.7)
axes[1].set(xticks=np.arange(3), xticklabels=["First action", "Second action", "Third action"],
            ylabel="Incremental geometric return", ylim=(-20, 180))
axes[1].set_title("Improvement comes after the common first action", loc="left", weight="bold", fontsize=11)
axes[1].legend(frameon=False, loc="upper right")
for axis in axes:
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=.15)
fig.suptitle("PAT05 TRAIN · same initial weights and Adam state · 48 training forwards per branch", weight="bold", fontsize=12)
fig.supxlabel("One deterministic pair; no repeatability, patient-transfer, RL-improvement or clinical-fidelity claim.", fontsize=9)
fig.savefig(folder / "matched-comparison.png", dpi=170)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained", sharey=True)
for axis, key, label, color in zip(axes, ("control", "augmented"),
                                  ("Original examples · 3 unique states", "Added visited states · 5 unique states"),
                                  (colors[1], colors[2])):
    branch = result["branches"][key]
    # Logged update losses are pre-step; append the separate post-update readout.
    losses = [row["loss"] for row in branch["curve"]] + [branch["latest_training_loss"]["loss"]]
    axis.plot(range(9), losses, "o-", color=color, markersize=4)
    axis.set(xticks=[0, 2, 4, 6, 8], xlabel="Additional optimizer steps completed")
    axis.set_title(label, loc="left", fontsize=10, weight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=.15)
axes[0].set_ylabel("Mean cross-entropy on each branch's examples")
fig.suptitle("Matched eight-update training traces · PAT05 TRAIN", weight="bold")
fig.supxlabel("Losses use different example sets; independent rollout outcomes determine route quality.", fontsize=8)
fig.savefig(folder / "learning-curves.png", dpi=170)
