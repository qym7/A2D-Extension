"""Per family: is the beta that maximises AR-help the same beta that maximises accuracy?

Zero GPU. Sweep file map: technical_details/figs/sweeps_file_map.json (built by build_sweeps_file_map.py, 2026-10-09);
Dream-v0 IFEval comes from technical_details/figs/ar_help_max_dreamv0.json (convex line tau_D+tau_A=1, addition row tau_D=1).
AR-help inherited = parent wrong, AR right, point right; AR-harm inherited = parent right, AR wrong, point wrong.
Writes technical_details/figs/ar_help_vs_acc_peak.{json,png}. Series with < 3 points are skipped.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path(__file__).resolve().parents[1] / "technical_details" / "figs"
S = json.loads((FIGS / "sweeps_file_map.json").read_text())


def evalplus(p, plus=False):
    ev = json.load(open(p))["eval"]
    return {t: int(s[0]["base_status"] == "pass" and (not plus or s[0]["plus_status"] == "pass")) for t, s in ev.items()}


def load(loader, p):
    if loader == "evalplus":
        return evalplus(p)
    if loader == "evalplus_plus":
        return evalplus(p, True)
    if loader == "reasoner":
        return {r["i"]: int(bool(r["ok"])) for r in json.load(open(p))["per_problem"]}
    if loader.startswith("reasoner_ar"):
        return {i: int(q["frac"] >= 0.5) for i, q in enumerate(json.load(open(p))["results"][0]["per_q"])}
    if loader.startswith("nemo_"):
        bench = Path(p).name
        d = Path(p)
        files = list(d.glob("eval/eval-results/*/output-rs0.jsonl")) or list(d.glob("*/output-rs0.jsonl")) or list(d.glob("output-rs0.jsonl"))
        rows = [json.loads(l) for l in open(files[0])]
        if "follow_all_instructions" in loader:
            return {r["key"]: int(bool(r["strict_eval"]["follow_all_instructions"])) for r in rows}
        field, key = {"nemo_symbolic_correct_unique_id": ("symbolic_correct", "unique_id"),
                      "nemo_is_correct_task_id": ("is_correct", "task_id"),
                      "nemo_symbolic_correct_problem": ("symbolic_correct", "problem")}[loader]
        return {r[key]: int(bool(r[field])) for r in rows}
    raise ValueError(loader)


out = {}
for name, s in S.items():
    if len(s["merge"]) < 3 and len(s["addition"]) < 3:
        continue
    D, A = load(s["loader"], s["parent"]), load(s["ar_loader"], s["ar"])
    ids = set(D) & set(A)
    hp = [i for i in ids if not D[i] and A[i]]
    xp = [i for i in ids if D[i] and not A[i]]
    rec = dict(help_pool=len(hp), harm_pool=len(xp), parent_acc=round(100 * sum(D[i] for i in ids) / len(ids), 1))
    for mode in ["merge", "addition"]:
        pts = []
        for p in s[mode]:
            M = load(s["loader"], p["path"])
            assert set(ids) <= set(M), (name, p["path"])
            pts.append(dict(beta=p["beta"], acc=round(100 * sum(M[i] for i in ids) / len(ids), 1),
                            help=sum(M[i] for i in hp), harm=sum(1 - M[i] for i in xp)))
        rec[mode] = sorted(pts, key=lambda q: q["beta"])
    out[name] = rec

dv = json.loads((FIGS / "ar_help_max_dreamv0.json").read_text())["IFEval"]
cv = [p for p in dv if abs(p["tau_D"] + p["tau_A"] - 1) < 1e-9]
ad = [p for p in dv if p["tau_D"] == 1.0]
out = {"Dream-v0 · IFEval": dict(help_pool=dv[0]["help_pool"], harm_pool=dv[0]["harm_pool"], parent_acc=56.6,
                                 merge=[dict(beta=p["tau_A"], acc=p["acc"], help=p["help"], harm=p["harm"]) for p in sorted(cv, key=lambda p: p["tau_A"])],
                                 addition=[dict(beta=p["tau_A"], acc=p["acc"], help=p["help"], harm=p["harm"]) for p in sorted(ad, key=lambda p: p["tau_A"])]),
       **out}


def argmax_set(pts, k):
    m = max(p[k] for p in pts)
    return [p["beta"] for p in pts if p[k] == m], m


summary = []
for name, r in out.items():
    for mode in ["merge", "addition"]:
        pts = [p for p in r[mode] if p["beta"] > 0]
        if len(pts) < 3:
            continue
        bh, mh = argmax_set(pts, "help")
        ba, ma = argmax_set(pts, "acc")
        summary.append(dict(series=name, mode=mode, betas=[p["beta"] for p in pts], beta_help=bh, help=mh,
                            help_pool=r["help_pool"], beta_acc=ba, acc=ma,
                            help_at_acc=max(p["help"] for p in pts if p["beta"] in ba)))
(FIGS / "ar_help_vs_acc_peak.json").write_text(json.dumps(dict(series=out, summary=summary), indent=1))
for x in summary:
    print(f"{x['series']:30s} {x['mode']:8s} grid {min(x['betas'])}-{max(x['betas'])} ({len(x['betas'])})  "
          f"help-best β={x['beta_help']} ({x['help']}/{x['help_pool']})  acc-best β={x['beta_acc']} ({x['acc']}, help {x['help_at_acc']})")

# ---------------- plot: one panel per series, AR-help (% of pool) vs beta; ▼ marks the accuracy-best beta
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COL = {"merge": "#2a78d6", "addition": "#eb6834"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": BASE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED})
names = list(out)
nc = 3
nr = (len(names) + nc - 1) // nc
fig, axes = plt.subplots(nr, nc, figsize=(13, 3.2 * nr))
for ax, name in zip(axes.flat, names):
    r = out[name]
    for mode in ["merge", "addition"]:
        pts = [p for p in r[mode] if p["beta"] > 0]
        if len(pts) < 2:
            continue
        xs = [p["beta"] for p in pts]
        ys = [100 * p["help"] / r["help_pool"] for p in pts]
        ax.plot(xs, ys, color=COL[mode], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURF, markeredgewidth=1.5, label=mode)
        ba = max(pts, key=lambda p: (p["acc"], -p["beta"]))
        ax.scatter([ba["beta"]], [100 * ba["help"] / r["help_pool"]], marker="v", s=170, color=COL[mode],
                   edgecolors=INK, linewidths=1.2, zorder=5)
    ax.set_title(f"{name}\nparent {r['parent_acc']} · AR-help pool {r['help_pool']}", loc="left", fontsize=9)
    ax.set_xlabel("β  (AR vector coefficient)")
    ax.set_ylabel("AR-help inherited (% of pool)")
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_facecolor(SURF)
for ax in list(axes.flat)[len(names):]:
    ax.axis("off")
h = [plt.Line2D([], [], color=COL[m], marker="o", linewidth=2, label=f"{m}  (D_b+(1−β)τ_D+βτ_A)" if m == "merge" else f"{m}  (D_p+βτ_A)") for m in COL]
h.append(plt.Line2D([], [], color=MUTED, marker="v", markeredgecolor=INK, linestyle="none", markersize=10, label="= β with the best accuracy on that line"))
fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
fig.suptitle("Where does AR-help peak, and is it where accuracy peaks?", x=0.01, y=1.035, ha="left", fontsize=12)
fig.patch.set_facecolor(SURF)
fig.tight_layout(rect=(0, 0, 1, 0.97))
p = FIGS / "ar_help_vs_acc_peak.png"
fig.savefig(p, dpi=160, bbox_inches="tight", facecolor=SURF)
print(p)
