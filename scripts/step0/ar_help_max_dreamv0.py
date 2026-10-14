"""Dream-v0: where in the (tau_D, tau_A) plane is AR-help inheritance highest, and is it the accuracy peak?

Zero GPU. Every same-protocol point on disk:
IFEval = lm-eval, 1280 tokens / 1280 steps, T0.1, top_p 0.9, entropy, n=541 (prompt-strict per item);
MATH-500 = lm-eval minerva, plain prompt, 512 tokens / 512 steps, n=500.
D = Dream-v0-Instruct, A = Qwen2.5-7B-Instruct (AR-native). AR-help inherited = D=0, A=1, M=1.
Writes technical_details/figs/ar_help_max_dreamv0.{json,png}.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OE = Path("/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step2_instruct/outputs/official_eval")
FIGS = Path(__file__).resolve().parents[2] / "technical_details" / "figs"
SKIP = ("steps256", "b300ref", "-h200", "smoke", "val0406", "flatctrl", "concise")


def coef(model):
    m = model.split("checkpoints__")[-1]
    if m == "Dream-org__Dream-v0-Instruct-7B":
        return (1.0, 0.0)
    for pat, f in [(r"dreambase_plus_([\d.]+)vdream_([\d.]+)vqwen", lambda r: (float(r[1]), float(r[2]))),
                   (r"dreambase_plus_([\d.]+)vdream_ar0", lambda r: (float(r[1]), 0.0)),
                   (r"dreambase_plus_([\d.]+)vqwenonly", lambda r: (0.0, float(r[1]))),
                   (r"dream-ins-04a_D-06a_A", lambda r: (0.4, 0.6)),
                   (r"dream-ins-(\d+)aD-(\d+)aA", lambda r: tuple(int(s) / (10 if len(s) == 2 else 100) for s in (r[1], r[2])))]:
        r = re.fullmatch(pat, m)
        if r:
            return f(r)
    return None


def collect(pattern, key, n_full, ar_run):
    pts = {}
    for f in sorted(OE.glob(f"*/*/{pattern}")):
        rel = str(f.relative_to(OE))
        c = coef(f.parent.name)
        if c is None or any(s in rel for s in SKIP):
            continue
        rows = [json.loads(l) for l in open(f)]
        if len(rows) != n_full:
            continue
        pts[c] = (rel, {r["doc_id"]: int(bool(r[key])) for r in rows})  # later file of a duplicate wins
    ar = sorted((OE / ar_run).glob(f"*/{pattern}"))[-1]
    A = {json.loads(l)["doc_id"]: int(bool(json.loads(l)[key])) for l in open(ar)}
    D = pts[(1.0, 0.0)][1]
    pool_h = [i for i in D if not D[i] and A[i]]
    pool_x = [i for i in D if D[i] and not A[i]]
    out = []
    for (a, b), (rel, M) in sorted(pts.items()):
        out.append(dict(tau_D=a, tau_A=b, run=rel.split("/")[0], acc=round(100 * sum(M.values()) / len(M), 1),
                        help=sum(M[i] for i in pool_h), harm=sum(1 - M[i] for i in pool_x),
                        help_pool=len(pool_h), harm_pool=len(pool_x)))
    return out


R = {"IFEval": collect("samples_ifeval_*.jsonl", "prompt_level_strict_acc", 541, "ifeval_qwen_inst_ar"),
     "MATH-500": collect("samples_math_500_*.jsonl", "exact_match", 500, "mathins_m500-ar-plain")}
(FIGS / "ar_help_max_dreamv0.json").write_text(json.dumps(R, indent=1))
for k, pts in R.items():
    top = sorted(pts, key=lambda p: (-p["help"], -p["acc"]))[:5]
    best = max(pts, key=lambda p: p["acc"])
    print(k, len(pts), "points; top AR-help:", [(p["tau_D"], p["tau_A"], p["help"], p["acc"]) for p in top],
          "| best acc:", (best["tau_D"], best["tau_A"], best["help"], best["acc"]))

# ---------------- plot
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUES = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": BASE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1]))


def plane(ax, pts, val, label, fmt):
    xs, ys, vs = [p["tau_A"] for p in pts], [p["tau_D"] for p in pts], [p[val] for p in pts]
    sc = ax.scatter(xs, ys, c=vs, cmap=BLUES, s=130, edgecolors=SURF, linewidths=1.5, zorder=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(label, color=INK2)
    cb.outline.set_visible(False)
    ax.plot([0, 1], [1, 0], color=MUTED, linestyle="--", linewidth=1, zorder=1)
    ax.text(0.52, 0.52, "convex line τ_D+τ_A=1", color=MUTED, fontsize=8, rotation=-35)
    hp = max(pts, key=lambda p: (p["help"], p["acc"]))
    ap = max(pts, key=lambda p: p["acc"])
    ax.scatter([hp["tau_A"]], [hp["tau_D"]], marker="*", s=420, color="#eb6834", edgecolors=INK, linewidths=0.8, zorder=5)
    ax.scatter([ap["tau_A"]], [ap["tau_D"]], s=330, facecolors="none", edgecolors=INK, linewidths=1.8, zorder=4)
    par = next(p for p in pts if p["tau_D"] == 1.0 and p["tau_A"] == 0.0)
    ax.annotate("Dream-Instruct", (0, 1), xytext=(0.05, 1.07), fontsize=8, color=INK2)
    for p in pts:
        if p in (hp, ap):
            ax.annotate(fmt(p), (p["tau_A"], p["tau_D"]), xytext=(-55, 15),
                        textcoords="offset points", fontsize=8, color=INK, weight="bold")
    ax.set_xlabel("τ_A coefficient (AR vector)")
    ax.set_ylabel("τ_D coefficient (diffusion post-training)")
    ax.set_xlim(-0.08, 1.75)
    ax.set_ylim(-0.08, 1.15)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


ifv = R["IFEval"]
pool = ifv[0]["help_pool"]
plane(axes[0], ifv, "help", f"AR-help inherited (of {pool})",
      lambda p: f"({p['tau_D']},{p['tau_A']}): help {p['help']}, acc {p['acc']}")
axes[0].set_title(f"IFEval: AR-help inherited ({len(ifv)} points)\n★ = max AR-help   ○ = max accuracy", loc="left", fontsize=10)
plane(axes[1], ifv, "acc", "IFEval prompt-strict acc",
      lambda p: f"({p['tau_D']},{p['tau_A']}): help {p['help']}, acc {p['acc']}")
axes[1].set_title("IFEval: accuracy, same points", loc="left", fontsize=10)

# panel 3: AR-help vs AR-harm, both benchmarks
ax = axes[2]
for name, pts, color, off in [("IFEval", ifv, "#2a78d6", (-60, -42)), ("MATH-500", R["MATH-500"], "#eb6834", (-150, 40))]:
    hp = pts[0]["help_pool"]
    xs = [100 * p["help"] / hp for p in pts]
    ys = [100 * p["harm"] / p["harm_pool"] for p in pts]
    ax.scatter(xs, ys, s=[3 * (p["acc"] - 30) for p in pts], color=color, alpha=0.85, edgecolors=SURF,
               linewidths=1.2, label=f"{name} (help pool {hp}, harm pool {pts[0]['harm_pool']})", zorder=3)
    top = max(pts, key=lambda p: (p["help"], p["acc"]))
    ax.annotate(f"{name} max help ({top['tau_D']},{top['tau_A']})\nacc {top['acc']}", (100 * top["help"] / hp, 100 * top["harm"] / top["harm_pool"]),
                xytext=off, textcoords="offset points", fontsize=8, color=color,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_xlabel("% of AR-help pool inherited (D=0,A=1 → M=1)")
ax.set_ylabel("% of AR-harm pool inherited (D=1,A=0 → M=0)")
ax.set_title("Help vs harm per merge point\n(marker size ∝ accuracy)", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.grid(color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.patch.set_facecolor(SURF)
for a in axes:
    a.set_facecolor(SURF)
fig.tight_layout()
p = FIGS / "ar_help_max_dreamv0.png"
fig.savefig(p, dpi=170, bbox_inches="tight", facecolor=SURF)
print(p)
