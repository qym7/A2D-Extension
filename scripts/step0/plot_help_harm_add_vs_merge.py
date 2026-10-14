"""Addition vs convex merge: per-item gains/losses relative to Dream-v0-Instruct, split by AR correctness.

Reads technical_details/figs/help_harm_add_vs_merge.json (8-state counts, computed 2026-10-09 from
dream_study/step2_instruct/outputs/official_eval samples) and writes the PNG next to it.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path(__file__).resolve().parents[2] / "technical_details" / "figs"
data = json.loads((FIGS / "help_harm_add_vs_merge.json").read_text())

INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEG = {  # (key, side, color, legend)
    "own_loss": (-1, "#4a3aa7", "Lost, AR also right (D=1,A=1,M=0)"),
    "harm_inh": (-1, "#eb6834", "AR-harm inherited (D=1,A=0,M=0)"),
    "help_inh": (+1, "#2a78d6", "AR-help inherited (D=0,A=1,M=1)"),
    "synergy": (+1, "#1baf7a", "New solve, AR also wrong (D=0,A=0,M=1)"),
}
ROWS = [("add", "Addition\n$D_p+1.1\\,\\tau_A$"), ("merge", "Convex merge\n$D_b+0.4\\tau_D+0.6\\tau_A$")]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": BASE, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharex=True)
for ax, (bench, modes) in zip(axes, data.items()):
    for y, (mode, label) in enumerate(ROWS):
        r = modes[mode]
        left = right = 0
        for key in ["harm_inh", "own_loss"]:  # losses grow leftwards from 0
            side, color, _ = SEG[key]
            v = r[key]
            ax.barh(y, -v, left=-left, height=0.55, color=color, edgecolor="#fcfcfb", linewidth=2)
            ax.text(-left - v / 2, y, str(v), ha="center", va="center", color="white", fontsize=9, weight="bold")
            left += v
        for key in ["help_inh", "synergy"]:
            side, color, _ = SEG[key]
            v = r[key]
            ax.barh(y, v, left=right, height=0.55, color=color, edgecolor="#fcfcfb", linewidth=2)
            ax.text(right + v / 2, y, str(v), ha="center", va="center", color="white", fontsize=9, weight="bold")
            right += v
        net = right - left
        ax.text(right + 3, y, f"net {net:+d}", va="center", color=INK2, fontsize=9)
    pools = modes["add"]
    ax.set_title(f"{bench}\nAR-help pool {pools['help_pool']} · AR-harm pool {pools['harm_pool']}",
                 fontsize=10, color=INK, loc="left")
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(ROWS)), [l for _, l in ROWS])
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.set_xlabel("← items lost vs Dream-Instruct      items gained vs Dream-Instruct →")
axes[0].set_xlim(-60, 140)
handles = [plt.Rectangle((0, 0), 1, 1, color=SEG[k][1]) for k in ["harm_inh", "own_loss", "help_inh", "synergy"]]
fig.legend(handles, [SEG[k][2] for k in ["harm_inh", "own_loss", "help_inh", "synergy"]],
           loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Relative to Dream-v0-Instruct: addition gains more AR-help without inheriting more AR-harm",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.patch.set_facecolor("#fcfcfb")
out = FIGS / "help_harm_add_vs_merge.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(out)
