"""Compare recorded outcomes; restore raw episode files from completed-run.tar.gz first."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

folder = Path(__file__).resolve().parent
rl = json.loads((folder.parent / "pat05-real-geometric-learning-v1" / "receipt.json").read_text())
bc = json.loads((folder / "receipt.json").read_text())
rows = [rl["episodes"][key]["independent_evaluation"]["outcomes"]
        for key in ("initial_policy", "latest_policy")]
rows += [bc["latest_policy"]["independent_evaluation"]["outcomes"],
         rl["episodes"]["greedy_search"]["independent_evaluation"]["outcomes"]]
latest = json.loads((folder / "latest_policy.json").read_text())
teacher = json.loads((folder / "teacher_replay.json").read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
x = np.arange(4)
axes[0].bar(x - .18, [r["target_removed_mm3"] for r in rows], .36, label="Annotated target", color="#207C63")
axes[0].bar(x + .18, [r["normal_removed_mm3"] for r in rows], .36, label="Other tissue", color="#C2A16A")
axes[0].set(xticks=x, xticklabels=["Initial", "2 RL\nupdates", "8 imitation\nupdates", "Greedy\nsearch"],
            ylabel="Simulated removed volume (mm³)", ylim=(0, 630))
axes[0].set_title("Task outcomes, not clinical accuracy", loc="left", weight="bold")
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
for index, row in enumerate(rows):
    axes[0].text(index, max(row["target_removed_mm3"], row["normal_removed_mm3"]) + 15,
                 f'Return {row["total_reward"]:.1f}', ha="center", fontsize=8)
x = np.arange(3)
axes[1].bar(x - .18, [d["reward"] for d in latest["decisions"]], .36, color="#196D91", label="After imitation")
axes[1].bar(x + .18, [d["reward"] for d in teacher["decisions"]], .36, color="#207C63", label="Greedy search")
axes[1].set(xticks=x, xticklabels=["First action", "Second action", "Third action"], ylabel="Incremental geometric return", ylim=(-20, 200))
axes[1].axhline(0, color="#999999", linewidth=.7)
axes[1].set_title("The remaining gap is after the first action", loc="left", weight="bold", fontsize=11)
axes[1].legend(frameon=False, loc="upper right", fontsize=9)
for axis in axes:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=.15)
    axis.set_axisbelow(True)
fig.suptitle("PAT05 TRAIN · real anatomy · annotation-assisted geometric planning", weight="bold", fontsize=13)
fig.supxlabel("One patient and one seed. Imitation improvement does not establish RL gains, transfer or physical fidelity.", fontsize=9)
fig.savefig(folder / "learning-comparison.png", dpi=170)
