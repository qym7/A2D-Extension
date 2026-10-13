"""Motivation schematic for idea 1 ("over distillation") on the merge line M(a,b) = D_b + a·τ_D + b·τ_A, a+b=1.
Top row: along the line the accuracy peaks at the best merge, but its two factors move in opposite directions —
precision on finished problems keeps rising past the peak (the over merge knows more per token), finish rate falls
(it cannot end its answers). The gate M(.4,.6) ⇐ M(0,1) takes the over merge's tokens only where the best merge is
unsure, keeps the finish rate and inherits the precision. Bottom row: the token-level mechanism as a strip.
Numbers: MATH-500 all 500, lockstep @1024 (step0_setup_baselines.md; gate = @0.8, H200 first 200 + last 300).
Zero GPU. Writes technical_details/step0_figs/motivation_merge_line.{svg,png}."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "technical_details" / "step0_figs"
OUT.mkdir(parents=True, exist_ok=True)

# --- data (all 500) ---
B = [0.0, 0.6, 1.0]                       # τ_A weight b: Ins, best merge, over merge
NAMES = ["Ins\nM(1,0)", "best merge\nM(.4,.6)", "over merge\nM(0,1)"]
ACC = [39.8, 46.6, 45.6]
FIN = [75, 80, 74]
PREC = [53.4, 58.4, 61.3]
GATE = dict(acc=50.0, fin=78, prec=63.9, label="M(.4,.6) ⇐ M(0,1) @0.8")

C_LINE = "#4C72B0"; C_GATE = "#C44E52"; C_STU = "#4C72B0"; C_TEA = "#C44E52"; C_GREY = "#999999"

fig = plt.figure(figsize=(11.5, 6.8))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.7], hspace=0.7, wspace=0.38, top=0.84, bottom=0.06)

def panel(ax, y, gy, title, ylim, unit=""):
    ax.plot(B, y, "-o", color=C_LINE, lw=2, ms=7, zorder=3)
    for b, v in zip(B, y):
        ax.annotate(f"{v:g}{unit}", (b, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color=C_LINE)
    ax.plot([0.6], [gy], marker="*", ms=16, color=C_GATE, zorder=4, ls="none")
    ax.annotate(f"{gy:g}{unit}", (0.6, gy), textcoords="offset points", xytext=(10, 2), ha="left", fontsize=9, color=C_GATE, fontweight="bold")
    ax.set_xticks(B); ax.set_xticklabels(NAMES, fontsize=8.5)
    ax.set_xlim(-0.15, 1.15); ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="y", color="#e5e5e5", lw=0.8); ax.spines[["top", "right"]].set_visible(False)

ax1 = fig.add_subplot(gs[0, 0]); panel(ax1, ACC, GATE["acc"], "(a) accuracy: peaks at the best merge", (34, 54))
ax2 = fig.add_subplot(gs[0, 1]); panel(ax2, FIN, GATE["fin"], "(b) finish rate: the over merge\ncannot end its answers", (68, 86), "%")
ax3 = fig.add_subplot(gs[0, 2]); panel(ax3, PREC, GATE["prec"], "(c) precision on finished answers:\nkeeps rising past the peak", (48, 68))
ax1.set_ylabel("MATH-500 acc (%)", fontsize=9); ax2.set_ylabel("answers with \\boxed (%)", fontsize=9); ax3.set_ylabel("acc | finished (%)", fontsize=9)
for ax in (ax1, ax2, ax3):
    ax.set_xlabel("τ_A weight b along the merge line", fontsize=8.5, color="#555555")
# gate arrow in panel (c): over merge's precision brought back to the best merge's position
ax3.add_patch(FancyArrowPatch((1.0, PREC[2]), (0.66, GATE["prec"] - 0.4), arrowstyle="-|>", mutation_scale=14, color=C_GATE, lw=1.4, connectionstyle="arc3,rad=0.25"))
ax1.legend(handles=[plt.Line2D([], [], color=C_LINE, marker="o", lw=2, label="single model on the merge line (1×)"),
                    plt.Line2D([], [], color=C_GATE, marker="*", ms=12, ls="none", label="gate: best merge decodes, over merge fills low-confidence tokens (2×)")],
           loc="lower center", bbox_to_anchor=(0.5, 0.905), bbox_transform=fig.transFigure, fontsize=9, frameon=False, ncol=2)

# --- bottom: token strip schematic ---
axs = fig.add_subplot(gs[1, :]); axs.set_axis_off()
conf = [0.99, 0.97, 0.95, 0.4, 0.98, 0.99, 0.6, 0.97, 0.99, 0.3, 0.98, 0.96, 0.99, 0.7, 0.98, 0.99, 0.5, 0.97]
theta = 0.8; n = len(conf); w = 1.0; h0 = 0.0
for i, c in enumerate(conf):
    low = c < theta
    axs.add_patch(Rectangle((i * w, h0), w * 0.9, 0.55, facecolor=(C_TEA if low else "#dbe4f0"), edgecolor="white", lw=1))
    axs.add_patch(Rectangle((i * w, 0.75), w * 0.9, c * 0.9, facecolor=(C_TEA if low else C_STU), edgecolor="none", alpha=0.85))
    axs.text(i * w + 0.45 * w, 0.27, "T" if low else "S", ha="center", va="center", fontsize=8.5, color=("white" if low else "#2b3a55"), fontweight="bold")
axs.plot([0, n * w], [0.75 + theta * 0.9] * 2, ls="--", color=C_GREY, lw=1)
axs.text(n * w + 0.2, 0.75 + theta * 0.9, f"θ = {theta}", va="center", fontsize=9, color="#555555")
axs.text(-0.3, 0.75 + 0.45, "student's raw\nconfidence p", ha="right", va="center", fontsize=8.5, color="#555555")
axs.text(-0.3, 0.27, "token written", ha="right", va="center", fontsize=8.5, color="#555555")
axs.set_xlim(-8, n * w + 8); axs.set_ylim(-0.1, 1.85)
axs.set_title("(d) one decode, left to right; the student (best merge) picks every position and writes its own token S where p ≥ θ,\n"
              "the over merge's token T where the student is unsure (p < θ): ≈ 24 % of tokens handed over at θ = 0.8, ≈ 3 % actually change",
              fontsize=9.5, loc="center")

fig.suptitle("Over distillation: the over merge knows more per token but cannot finish; gate its tokens into the best merge",
             fontsize=12, y=0.985)
for ext in ("svg", "png"):
    fig.savefig(OUT / f"motivation_merge_line.{ext}", dpi=160, bbox_inches="tight")
print("wrote", OUT / "motivation_merge_line.svg")
