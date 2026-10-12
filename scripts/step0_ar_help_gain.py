"""How much more AR-help does the AR-help-best beta get than the accuracy-best beta, and what does it cost?

Reads technical_details/figs/ar_help_vs_acc_peak.json (per-point help/harm/acc, built by ar_help_vs_acc_peak.py).
Rule (user, 2026-10-09): the AR-help-best beta is chosen only on the 0.1 grid (betas that are multiples of 0.1;
0.05-step points such as 0.05/0.15/0.25 are not eligible). The accuracy-best beta is the reported best on all points.
Ties in AR-help -> the smallest tied beta. "still rising" = AR-help-best beta is the largest 0.1-grid beta run.
Writes technical_details/figs/ar_help_gain.{json,png,md}.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path(__file__).resolve().parents[1] / "technical_details" / "figs"
S = json.loads((FIGS / "ar_help_vs_acc_peak.json").read_text())["series"]
on_grid = lambda b: abs(round(b * 10) - b * 10) < 1e-6
UNRELIABLE = {"Nemotron-8B · mbpp"}  # first-code-block regrade moves the parent 62 -> 86

rows = []
for name, r in S.items():
    for mode in ["merge", "addition"]:
        pts = [p for p in r[mode] if p["beta"] > 0]
        grid = [p for p in pts if on_grid(p["beta"])]
        if len(grid) < 3:
            continue
        acc_best = max(pts, key=lambda p: (p["acc"], -p["beta"]))
        mh = max(p["help"] for p in grid)
        help_best = min((p for p in grid if p["help"] == mh), key=lambda p: p["beta"])
        tied = [p["beta"] for p in grid if p["help"] == mh]
        pool = r["help_pool"]
        rows.append(dict(
            series=name, mode=mode, pool=pool, n_grid=len(grid), grid_max=max(p["beta"] for p in grid),
            beta_acc=acc_best["beta"], acc_at_acc=acc_best["acc"], help_at_acc=acc_best["help"],
            beta_help=help_best["beta"], acc_at_help=help_best["acc"], help_at_help=help_best["help"],
            d_help=help_best["help"] - acc_best["help"],
            d_help_pp=round(100 * (help_best["help"] - acc_best["help"]) / pool, 1),
            d_acc=round(help_best["acc"] - acc_best["acc"], 1),
            d_harm=help_best["harm"] - acc_best["harm"],
            still_rising=help_best["beta"] == max(p["beta"] for p in grid),
            tied=tied if len(tied) > 1 else None,
            unreliable=name in UNRELIABLE))
(FIGS / "ar_help_gain.json").write_text(json.dumps(rows, indent=1))

# ---------------- markdown table
L = ["| series | mode | acc-best β (acc, AR-help) | AR-help-best β (acc, AR-help) | ΔAR-help | Δacc | ΔAR-harm | peak? |",
     "|---|---|---|---|---:|---:|---:|---|"]
for x in rows:
    peak = "**still rising ↗**" if x["still_rising"] else "peak found"
    if x["tied"]:
        peak += f" (tie {x['tied']})"
    if x["unreliable"]:
        peak += " ⚠ grading"
    L.append(f"| {x['series']} | {x['mode']} | {x['beta_acc']} ({x['acc_at_acc']}, {x['help_at_acc']}/{x['pool']}) | "
             f"{x['beta_help']} ({x['acc_at_help']}, {x['help_at_help']}/{x['pool']}) | "
             f"+{x['d_help']} (+{x['d_help_pp']} pp) | {x['d_acc']:+.1f} | {x['d_harm']:+d} | {peak} |")
print("\n".join(L))
(FIGS / "ar_help_gain_table.md").write_text("\n".join(L) + "\n")

# ---------------- plot: dumbbell, one row per line; x = AR-help % of pool
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
ACC, HELP = "#898781", "#2a78d6"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": BASE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": INK})
fig, ax = plt.subplots(figsize=(11.5, 0.42 * len(rows) + 1.6))
for y, x in enumerate(rows):
    a, h = 100 * x["help_at_acc"] / x["pool"], 100 * x["help_at_help"] / x["pool"]
    ax.plot([a, h], [y, y], color=BASE, linewidth=3, zorder=1, solid_capstyle="round")
    ax.scatter([a], [y], s=70, color=ACC, edgecolors=SURF, linewidths=1.5, zorder=3)
    ax.scatter([h], [y], s=90, marker=">" if x["still_rising"] else "o", color=HELP, edgecolors=SURF, linewidths=1.5, zorder=4)
    tag = f"+{x['d_help_pp']} pp · acc {x['d_acc']:+.1f}"
    if x["still_rising"]:
        tag += "  ↗ still rising"
    if x["unreliable"]:
        tag += "  ⚠"
    ax.text(max(a, h) + 1.5, y, tag, va="center", fontsize=8, color=INK2)
    ax.text(-1.5, y, f"β {x['beta_acc']} → {x['beta_help']}", va="center", ha="right", fontsize=8, color=MUTED)
ax.set_yticks(range(len(rows)), [f"{x['series']} · {x['mode']}" for x in rows], fontsize=8.5)
ax.tick_params(axis="y", length=0, pad=62)
ax.invert_yaxis()
ax.set_xlim(0, 112)
ax.set_xlabel("AR-help inherited (% of the AR-help pool)")
ax.grid(axis="x", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
h = [plt.Line2D([], [], marker="o", color=ACC, linestyle="none", markersize=8, label="at accuracy-best β"),
     plt.Line2D([], [], marker="o", color=HELP, linestyle="none", markersize=8, label="at AR-help-best β (0.1 grid)"),
     plt.Line2D([], [], marker=">", color=HELP, linestyle="none", markersize=8, label="AR-help-best = largest β tried → still rising, peak not confirmed")]
fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.55, 1.0))
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
fig.tight_layout(rect=(0, 0, 1, 0.95))
p = FIGS / "ar_help_gain.png"
fig.savefig(p, dpi=170, bbox_inches="tight", facecolor=SURF)
print(p)
