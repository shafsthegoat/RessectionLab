"""Plot the completed receipt; no simulation or patient data loading."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
episodes = json.loads((folder / "receipt.json").read_text())["episodes"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
labels = ["Initial policy", "After 2 RL updates", "Random 1", "Random 2", "Random 3", "Greedy search"]
keys = ["initial_policy", "latest_policy", "random_legal_0", "random_legal_1", "random_legal_2", "greedy_search"]
values = [episodes[key]["return"] for key in keys]
colors = ["#718096", "#196D91", "#B3BAC5", "#B3BAC5", "#B3BAC5", "#207C63"]
axis = axes[0]
bars = axis.barh(labels[::-1], values[::-1], color=colors[::-1])
axis.axvline(0, color="#a0a0a0", lw=.8)
axis.set(xlabel="Geometric return (fixed research weights)", xlim=(-70, 485))
axis.set_title("Independent episode comparison", loc="left", weight="bold")
for bar, value in zip(bars, values[::-1]):
    axis.text(max(0, value) + 6, bar.get_y() + bar.get_height() / 2,
              f"{value:.2f}", va="center", ha="left", fontsize=9)
axis = axes[1]
training = [episodes[f"optimization_{update}_{index}"]["return"]
            for update in range(2) for index in range(2)]
axis.plot(range(1, 5), training, "o-", color="#196D91", label="Stochastic training episode")
axis.axvline(2.5, color="#999999", lw=1, ls=":")
axis.axhline(episodes["greedy_search"]["return"], color="#207C63", ls="--", label="Greedy search")
axis.axhline(episodes["latest_policy"]["return"], color="#718096", ls=":", label="Initial and latest argmax")
axis.set(xticks=[1, 2, 3, 4], xlabel="On-policy training episode (two per update)", ylabel="Geometric return")
axis.set_title("Two-update learning trace", loc="left", weight="bold")
axis.legend(frameon=False, fontsize=8, loc="center right")
axis.grid(axis="y", alpha=.15)
fig.suptitle("PAT05 TRAIN · annotation-assisted geometric planning · seed 11", fontsize=13, weight="bold")
fig.supxlabel("No deterministic improvement; one training anatomy. No transfer, physical-fidelity or clinical-accuracy claim.", fontsize=9)
for axis in axes:
    axis.spines[["top", "right"]].set_visible(False)
fig.savefig(folder / "learning-summary.png", dpi=170)
