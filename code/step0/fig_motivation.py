"""Motivation schematic for idea 1 ("over distillation") on the merge line M(a,b) = D_b + a·τ_D + b·τ_A, a+b=1.
Story (top row, left to right): merging more of the AR vector peaks at the best merge (a); pushing past it costs accuracy
only because the over merge stops finishing its answers (b), while every answer it does finish is more often right (c).
So the over merge knows more per token but cannot end. Bottom row (d): the gate — the best merge decodes and keeps its
own token where it is confident, the over merge's token where it is unsure — keeps the finish rate and inherits the
precision, and lands above both ends (the red star in a–c).
Numbers: MATH-500 all 500, lockstep @1024 (step0_setup_baselines.md; gate = M(.4,.6) ⇐ M(0,1) @0.8).
Font: STIXGeneral (Times-compatible; no TeX / Times on this host). Zero GPU.
Writes technical_details/step0_figs/motivation_merge_line.{svg,png,pdf}."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 10, "axes.titlesize": 10.5, "svg.fonttype": "none"})
OUT = Path(__file__).resolve().parents[2] / "technical_details" / "step0_figs"
OUT.mkdir(parents=True, exist_ok=True)

# --- data (MATH-500, all 500) ---
B = [0.0, 0.6, 1.0]
NAMES = ["Instruct\n$M(1,0)$", "best merge\n$M(.4,.6)$", "over merge\n$M(0,1)$"]
ACC = [39.8, 46.6, 45.6]; FIN = [75, 80, 74]; PREC = [53.4, 58.4, 61.3]
GATE = dict(acc=50.0, fin=78, prec=63.9)
# theta sweep of M(.4,.6) <= M(0,1), all 500 (0.3 / 0.7 from the 9/30 B300 run, the rest H200); handed-over share from the last-300 gate counts
TH = [0.0, 0.3, 0.6, 0.7, 0.8]; TH_ACC = [46.6, 47.6, 48.0, 50.4, 50.0]; TH_RATE = {0.6: "15%", 0.8: "24%"}
TEACHER_ACC = 45.6

C_LINE = "#3B5B92"; C_GATE = "#B23A3A"; C_TEA_LIGHT = "#D98C8C"; C_STU_LIGHT = "#DCE4F0"

fig = plt.figure(figsize=(11.5, 7.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.7, wspace=0.40, top=0.82, bottom=0.06, left=0.07, right=0.98)

def panel(ax, y, gy, title, ylabel, ylim, unit="", gate_note="", side=1):
    ax.plot(B, y, "-o", color=C_LINE, lw=1.8, ms=6.5, zorder=3)
    for b, v in zip(B, y):
        ax.annotate(f"{v:g}{unit}", (b, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9.5, color=C_LINE)
    ax.plot([0.6], [gy], marker="*", ms=17, color=C_GATE, zorder=4, ls="none")
    ha = "left" if side > 0 else "right"
    ax.annotate(f"{gy:g}{unit}", (0.6, gy), textcoords="offset points", xytext=(11 * side, -2), ha=ha, fontsize=10, color=C_GATE, fontweight="bold")
    if gate_note:
        ax.annotate(gate_note, (0.6, gy), textcoords="offset points", xytext=(11 * side, -14), ha=ha, fontsize=8.5, color=C_GATE)
    ax.set_xticks(B); ax.set_xticklabels(NAMES, fontsize=9.5)
    ax.set_xlim(-0.18, 1.18); ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_xlabel(r"weight $b$ of the AR vector $\tau_A$ along the merge line", fontsize=9, color="#444444")
    ax.grid(axis="y", color="#e6e6e6", lw=0.8); ax.spines[["top", "right"]].set_visible(False)

ax1 = fig.add_subplot(gs[0, 0]); panel(ax1, ACC, GATE["acc"], "(a) Accuracy peaks at the best merge,\n      and drops past it …", "MATH-500 accuracy (%)", (34, 55), gate_note="+3.4 over the best merge")
ax2 = fig.add_subplot(gs[0, 1]); panel(ax2, FIN, GATE["fin"], "(b) … because the over merge stops\n      finishing its answers …", "answers finished within 1024 tokens (%)", (68, 86), "%", gate_note="finish rate kept")
ax3 = fig.add_subplot(gs[0, 2]); panel(ax3, PREC, GATE["prec"], "(c) … although each finished answer\n      is more often right.", "accuracy on finished answers (%)", (48, 69), gate_note="precision inherited", side=-1)
ax3.add_patch(FancyArrowPatch((0.97, PREC[2] + 0.9), (0.66, GATE["prec"] + 0.9), arrowstyle="-|>", mutation_scale=13, color=C_GATE, lw=1.3, connectionstyle="arc3,rad=-0.3"))

fig.legend(handles=[plt.Line2D([], [], color=C_LINE, marker="o", lw=1.8, label="one model on the merge line (1× decode)"),
                    plt.Line2D([], [], color=C_GATE, marker="*", ms=13, ls="none", label=r"gate $M(.4,.6) \Leftarrow M(0,1)$ @ $\theta$=0.8: best merge decodes, over merge supplies its unsure tokens (2× decode)")],
           loc="lower center", bbox_to_anchor=(0.5, 0.875), fontsize=9.5, frameon=False, ncol=1)

# --- (d) theta sweep ---
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(TH, TH_ACC, "-o", color=C_GATE, lw=1.8, ms=6, zorder=3)
ax4.plot([0.8], [50.0], marker="*", ms=17, color=C_GATE, zorder=4, ls="none")
for t, v in zip(TH, TH_ACC):
    ax4.annotate(f"{v:g}", (t, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color=C_GATE)
ax4.axhline(ACC[1], color=C_LINE, ls=":", lw=1.2); ax4.text(0.98, ACC[1] + 0.25, "best merge alone 46.6", fontsize=8, color=C_LINE, va="bottom", ha="right")
ax4.axhline(TEACHER_ACC, color=C_LINE, ls="--", lw=1.2); ax4.text(0.98, TEACHER_ACC - 0.3, "over merge alone 45.6", fontsize=8, color=C_LINE, va="top", ha="right")
ax4.set_xlim(-0.05, 1.0); ax4.set_ylim(43.5, 52.8); ax4.set_xticks([0, 0.3, 0.6, 0.7, 0.8])
ax4.set_xlabel(r"threshold $\theta$ (hand over where $p<\theta$)", fontsize=9, color="#444444"); ax4.set_ylabel("MATH-500 accuracy (%)", fontsize=9.5)
ax4.set_title(r"(d) Threshold sweep: above both ends from $\theta$ = 0.3," + "\n      flat between 0.7 and 0.8", loc="left", fontsize=10.5)
ax4.grid(axis="y", color="#e6e6e6", lw=0.8); ax4.spines[["top", "right"]].set_visible(False)

# --- (e) token strip ---
axs = fig.add_subplot(gs[1, 1:]); axs.set_axis_off()
conf = [0.99, 0.97, 0.95, 0.40, 0.98, 0.99, 0.60, 0.97, 0.99, 0.30, 0.98, 0.96, 0.99, 0.70, 0.98, 0.99, 0.50, 0.97]
theta = 0.8; n = len(conf); w = 1.0
for i, c in enumerate(conf):
    low = c < theta
    axs.add_patch(Rectangle((i * w, 0.0), 0.9 * w, 0.55, facecolor=(C_GATE if low else C_STU_LIGHT), edgecolor="white", lw=1))
    axs.add_patch(Rectangle((i * w, 0.75), 0.9 * w, c * 0.9, facecolor=(C_TEA_LIGHT if low else C_LINE), edgecolor="none"))
    axs.text(i * w + 0.45 * w, 0.27, "T" if low else "S", ha="center", va="center", fontsize=9.5, color=("white" if low else "#2b3a55"), fontweight="bold")
axs.plot([0, n * w - 0.1], [0.75 + theta * 0.9] * 2, ls="--", color="#888888", lw=1)
axs.text(n * w + 0.2, 0.75 + theta * 0.9, r"$\theta = 0.8$", va="center", fontsize=10, color="#444444")
axs.text(-0.3, 0.75 + 0.45, "best merge's\nconfidence $p$", ha="right", va="center", fontsize=9.5, color="#444444")
axs.text(-0.3, 0.27, "token written", ha="right", va="center", fontsize=9.5, color="#444444")
axs.text(n * w + 0.2, 0.27, "S = best merge's token\nT = over merge's token", va="center", fontsize=9, color="#444444")
axs.set_xlim(-5.5, n * w + 6.5); axs.set_ylim(-0.1, 2.0)
axs.set_title("(e) The gate. The best merge decodes left to right and keeps its own token where it is confident ($p \\geq \\theta$);\n"
              "where it is unsure ($p < \\theta$) the over merge's token is written instead. At $\\theta$ = 0.8 about 24% of the tokens\n"
              "are handed over and about 3% actually change — enough to inherit the precision without losing the finish rate.",
              fontsize=9.6, loc="left")

fig.suptitle("Over distillation: the over-merged model knows more per token but cannot finish — borrow its tokens only where the best merge is unsure",
             fontsize=12.5, y=0.975)
for ext in ("svg", "png", "pdf"):
    fig.savefig(OUT / f"motivation_merge_line.{ext}", dpi=170, bbox_inches="tight")
print("wrote", OUT / "motivation_merge_line.svg")
