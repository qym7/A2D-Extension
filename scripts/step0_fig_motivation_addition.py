"""Motivation schematic for idea 1 on the ADDITION line (the main line, user 2026-10-13): A(β) = D_b + τ_D + β·τ_A,
student = Ins = A(0), teacher = over addition A(2.0).
Top row: more AR vector raises accuracy up to A(2.0) and collapses at A(3.0) (a); the gain is entirely finish rate (b),
the per-answer precision does not move (c). Bottom row: why the student's unsure positions are the right place to ask
(d: A2D's teacher-forced measurement — the AR vector changes the model's tokens almost only where the model is unsure);
the threshold sweep (e); the gate as a token strip (f).
Numbers: MATH-500 last 300 for a–c (A(3.0) exists only there); θ sweep on the last 300 with the matched-dose random control;
(d) from technical_details/step1_confidence_channel.md §1 (A2D E3, GSM8K 100 problems, teacher-forced, raw softmax confidence).
Font: STIXGeneral (Times-compatible). Zero GPU. Writes technical_details/step0_figs/motivation_addition_line.{svg,pdf,png}."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 10, "svg.fonttype": "none"})
OUT = Path(__file__).resolve().parents[1] / "technical_details" / "step0_figs"

# --- a–c: addition line, MATH-500 last 300 ---
BETA = [0.0, 2.0, 3.0]
NAMES = ["Instruct\n$A(0)$", "over addition\n$A(2.0)$", "$A(3.0)$\n(collapsed)"]
ACC = [36.7, 46.7, 34.0]; FIN = [76, 94, 72]; PREC = [48.5, 49.5, 47.0]
GATE = dict(acc=49.0, fin=93, prec=52.9)          # Ins <= A(2.0) @0.8, last 300
# --- d: A2D E3 (teacher-forced, GSM8K): share of the model's tokens the AR vector changes, by the model's raw confidence
BINS = ["0–.1", ".1–.2", ".2–.3", ".3–.4", ".4–.5", ".5–.6", ".6–.7", ".7–.8", ".8–.9", ".9–1"]
CHANGE = [23.4, 17.9, 17.1, 17.2, 16.5, 11.5, 5.0, 1.8, 0.4, 0.01]
SHARE = [1.1, 2.5, 3.1, 3.9, 4.7, 5.1, 4.7, 5.7, 7.7, 61.4]
# --- e: theta sweep of Ins <= A(2.0), last 300 (same subset as a-c); unconditional random control at the matched dose
TH = [0.0, 0.6, 0.8]; TH_ACC = [36.7, 49.3, 49.0]; TEACHER300 = 46.7
TH_RATE = {0.6: "5%", 0.8: "14%"}
RND = {0.6: 38.3, 0.8: 38.0}

C_LINE = "#3B5B92"; C_GATE = "#B23A3A"; C_TEA_LIGHT = "#D98C8C"; C_STU_LIGHT = "#DCE4F0"; C_GREY = "#777777"

fig = plt.figure(figsize=(12, 8.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.62, wspace=0.42, top=0.82, bottom=0.07, left=0.065, right=0.985)

def panel(ax, y, gy, title, ylabel, ylim, unit="", note="", side=1, note_xy=None):
    ax.plot(BETA, y, "-o", color=C_LINE, lw=1.8, ms=6.5, zorder=3)
    for b, v in zip(BETA, y):
        ax.annotate(f"{v:g}{unit}", (b, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9.5, color=C_LINE)
    ax.plot([2.0], [gy], marker="*", ms=17, color=C_GATE, zorder=4, ls="none")
    ha = "left" if side > 0 else "right"
    ax.annotate(f"{gy:g}{unit}", (2.0, gy), textcoords="offset points", xytext=(11 * side, 4), ha=ha, fontsize=10, color=C_GATE, fontweight="bold")
    if note:
        ax.annotate(note, (2.0, gy), textcoords="offset points", xytext=note_xy or (11 * side, -8), ha=ha, fontsize=8.5, color=C_GATE, va="top")
    ax.set_xticks(BETA); ax.set_xticklabels(NAMES, fontsize=9.5)
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", fontsize=10.5); ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_xlabel(r"weight $\beta$ of the AR vector $\tau_A$ added to Instruct", fontsize=9, color="#444444")
    ax.grid(axis="y", color="#e6e6e6", lw=0.8); ax.spines[["top", "right"]].set_visible(False)

ax1 = fig.add_subplot(gs[0, 0]); panel(ax1, ACC, GATE["acc"], "(a) Adding the AR vector: accuracy rises\n      to $A(2.0)$ and collapses at $A(3.0)$ …", "MATH-500 accuracy, last 300 (%)", (30, 54), note="+12 over Instruct,\n+2.3 over $A(2.0)$ (n.s.)", side=-1, note_xy=(-14, -8))
ax2 = fig.add_subplot(gs[0, 1]); panel(ax2, FIN, GATE["fin"], "(b) … the whole gain is finish rate:\n      the over-added model ends its answers …", "answers finished within 1024 tokens (%)", (66, 100), "%", note="finish rate\nof the teacher", side=-1, note_xy=(-14, -4))
ax3 = fig.add_subplot(gs[0, 2]); panel(ax3, PREC, GATE["prec"], "(c) … but per finished answer it is no more\n      precise than the student. The gate is.", "accuracy on finished answers (%)", (44, 56), note="above both: student's confident\ntokens + teacher's unsure ones", side=-1, note_xy=(-14, -4))

fig.legend(handles=[plt.Line2D([], [], color=C_LINE, marker="o", lw=1.8, label="one model on the addition line (1× decode)"),
                    plt.Line2D([], [], color=C_GATE, marker="*", ms=13, ls="none", label=r"gate Ins $\Leftarrow A(2.0)$ @ $\theta$=0.8: Instruct decodes, $A(2.0)$ supplies the tokens where Instruct is unsure (2× decode)")],
           loc="lower center", bbox_to_anchor=(0.5, 0.868), fontsize=9.5, frameon=False, ncol=1)

# --- (d) why the unsure positions ---
ax4 = fig.add_subplot(gs[1, 0])
x = range(len(BINS))
ax4.bar(x, CHANGE, color=C_GATE, width=0.7)
for i, (c, s) in enumerate(zip(CHANGE, SHARE)):
    ax4.text(i, c + 0.5, f"{c:g}" if c >= 0.1 else "0.01", ha="center", fontsize=7.5, color=C_GATE)
    ax4.text(i, -3.2, f"{s:g}%", ha="center", fontsize=7, color=C_GREY)
ax4.text(-0.6, -5.6, "share of all tokens:", fontsize=7, color=C_GREY, ha="left")
ax4.set_xticks(list(x)); ax4.set_xticklabels(BINS, fontsize=7.5, rotation=0)
ax4.set_ylim(-6.5, 28); ax4.set_yticks([0, 10, 20])
ax4.set_xlabel("the model's own confidence $p$ in its token", fontsize=9, color="#444444")
ax4.set_ylabel("tokens the AR vector changes (%)", fontsize=9.5)
ax4.set_title("(d) Why ask where the student is unsure: the AR\n      vector rewrites a token almost only when the model\n      is unsure about it (A2D, teacher-forced, GSM8K)", loc="left", fontsize=10.5)
ax4.grid(axis="y", color="#e6e6e6", lw=0.8); ax4.spines[["top", "right"]].set_visible(False)
ax4.text(9.3, 22, "96% of the changes fall\nat $p<0.7$; at $p\\geq0.9$\n(61% of all tokens)\n1 change in 15,000", ha="right", va="top", fontsize=8.5, color="#333333")

# --- (e) theta sweep ---
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(TH, TH_ACC, "-o", color=C_GATE, lw=1.8, ms=6, zorder=3)
ax5.plot([0.8], [49.0], marker="*", ms=17, color=C_GATE, zorder=4, ls="none")
ax5.plot(list(RND), list(RND.values()), marker="o", mfc="white", mec=C_GREY, color=C_GREY, ms=6, lw=1.2, ls="--", zorder=3)
for t, v in RND.items():
    ax5.annotate(f"{v:g}", (t, v), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8.5, color=C_GREY)
ax5.text(0.7, 40.6, "same dose, random positions", ha="center", fontsize=8, color=C_GREY)
for t, v in zip(TH, TH_ACC):
    ax5.annotate(f"{v:g}", (t, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color=C_GATE)
ax5.axhline(TH_ACC[0], color=C_LINE, ls=":", lw=1.2); ax5.text(0.98, TH_ACC[0] - 0.4, "Instruct alone 36.7", fontsize=8, color=C_LINE, va="top", ha="right")
ax5.axhline(TEACHER300, color=C_LINE, ls="--", lw=1.2); ax5.text(0.02, TEACHER300 + 0.3, "$A(2.0)$ alone 46.7", fontsize=8, color=C_LINE, va="bottom", ha="left")
ax5.set_xlim(-0.05, 1.0); ax5.set_ylim(33, 53); ax5.set_xticks(TH); ax5.set_xticklabels(["0\n(student)", "0.6\n(5%)", "0.8\n(14%)"], fontsize=9)
ax5.set_xlabel(r"threshold $\theta$ (hand over where $p<\theta$); in brackets: share of tokens handed over", fontsize=8.5, color="#444444"); ax5.set_ylabel("MATH-500 accuracy, last 300 (%)", fontsize=9.5)
ax5.set_title("(e) Threshold: the teacher's level is reached by\n      $\\theta$ = 0.6–0.8 with 5–14% of the tokens; the same\n      dose at random positions stays at the student", loc="left", fontsize=10.5)
ax5.grid(axis="y", color="#e6e6e6", lw=0.8); ax5.spines[["top", "right"]].set_visible(False)

# --- (f) token strip ---
axs = fig.add_subplot(gs[1, 2]); axs.set_axis_off()
conf = [0.99, 0.97, 0.40, 0.98, 0.99, 0.60, 0.97, 0.99, 0.30, 0.98, 0.96, 0.70, 0.98]
theta = 0.8; n = len(conf); w = 1.0
for i, c in enumerate(conf):
    low = c < theta
    axs.add_patch(Rectangle((i * w, 0.0), 0.9 * w, 0.55, facecolor=(C_GATE if low else C_STU_LIGHT), edgecolor="white", lw=1))
    axs.add_patch(Rectangle((i * w, 0.75), 0.9 * w, c * 0.9, facecolor=(C_TEA_LIGHT if low else C_LINE), edgecolor="none"))
    axs.text(i * w + 0.45 * w, 0.27, "T" if low else "S", ha="center", va="center", fontsize=9.5, color=("white" if low else "#2b3a55"), fontweight="bold")
axs.plot([0, n * w - 0.1], [0.75 + theta * 0.9] * 2, ls="--", color="#888888", lw=1)
axs.text(n * w + 0.2, 0.75 + theta * 0.9, r"$\theta$", va="center", fontsize=10, color="#444444")
axs.text(-0.3, 0.75 + 0.45, "Instruct's\nconfidence $p$", ha="right", va="center", fontsize=9, color="#444444")
axs.text(-0.3, 0.27, "token written", ha="right", va="center", fontsize=9, color="#444444")
axs.text(0, -0.35, "S = Instruct's token      T = $A(2.0)$'s token", fontsize=8.5, color="#444444", va="top")
axs.set_xlim(-5.5, n * w + 1.5); axs.set_ylim(-0.8, 1.9)
axs.set_title("(f) The gate: Instruct picks every position left to\n      right and writes its own token where $p\\geq\\theta$;\n      where $p<\\theta$ the teacher's token is written", loc="left", fontsize=10.5)

fig.suptitle("Over distillation on the addition line: the over-added model finishes but is not more precise —\n"
             "keep the student's tokens where it is confident and take the teacher's only where it is unsure",
             fontsize=12.5, y=1.0)
for ext in ("svg", "png", "pdf"):
    fig.savefig(OUT / f"motivation_addition_line.{ext}", dpi=170, bbox_inches="tight")
print("wrote", OUT / "motivation_addition_line.svg")
