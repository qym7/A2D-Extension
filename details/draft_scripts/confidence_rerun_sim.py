"""Simulate 'run host; if its confidence is low, rerun with a more-merged donor' from saved trajectories. Zero GPU.

Trajectories (dream_study/step2_instruct/outputs/official_eval/<run>/traj/lm_eval.traj.jsonl) store, per item, the
probability of the token committed at every step (sel_prob) and the final ids. key = sha1(prompt)[:16].
Host confidence = mean log sel_prob over committed non-EOS tokens (also: 10th percentile).
For each rerun rate r the lowest-confidence r of items take the donor's answer. Compared with
(a) a random router at the same r (1000 draws -> mean and one-sided p), (b) the oracle, and
(c) a 2-fold cross-validated choice of r (pick r on one half, score on the other) = what you'd get without peeking.
Writes technical_details/figs/confidence_rerun_sim.{json,png}.
"""
import glob
import hashlib
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OE = Path("/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step2_instruct/outputs/official_eval")
FIGS = Path(__file__).resolve().parents[2] / "results" / "figs"
EOS = {151643, 151645}
RATES = [round(0.05 * i, 2) for i in range(21)]


def correct(run, key):
    f = sorted(OE.glob(f"{run}/*/samples_*.jsonl"))[-1]
    out = {}
    for l in open(f):
        r = json.loads(l)
        a = r["arguments"]
        ctx = a["gen_args_0"]["arg_0"] if isinstance(a, dict) else a[0][0]
        out[hashlib.sha1(ctx.encode()).hexdigest()[:16]] = int(bool(r[key]))
    return out


def confidence(run):
    conf = {}
    for l in open(OE / run / "traj/lm_eval.traj.jsonl"):
        t = json.loads(l)
        ids = t["final_ids"]
        keep = [i for i, pos in enumerate(t["committed_pos"]) if 0 <= pos < len(ids) and ids[pos] not in EOS]
        lp = [math.log(max(t["sel_prob"][i], 1e-12)) for i in keep]
        ent = [t["entropy"][i] for i in keep] or t["entropy"]
        if not lp:
            lp = [math.log(max(p, 1e-12)) for p in t["sel_prob"]]
        s = sorted(lp)
        conf[t["key"]] = dict(mean=sum(lp) / len(lp), p10=s[len(s) // 10], min=s[0],
                              negent=-sum(ent) / len(ent), negent_max=-max(ent))
    return conf


def route(H, Dn, order, r):
    k = round(r * len(order))
    swap = set(order[:k])
    return sum(Dn[i] if i in swap else H[i] for i in order) / len(order)


PAIRS = [  # (benchmark, metric key, host run, donor run, label)
    ("IFEval", "prompt_level_strict_acc", "ifeval_ins-10aD-07aA", "ifeval_ins-10aD-11aA", "host (1.0,0.7) → donor (1.0,1.1)"),
    ("IFEval", "prompt_level_strict_acc", "ifeval_ins-10aD-11aA", "ifeval_ins-10aD-14aA", "host (1.0,1.1) → donor (1.0,1.4)"),
    ("IFEval", "prompt_level_strict_acc", "ifeval_ins-10aD-11aA", "ifeval_ins-10aD-16aA", "host (1.0,1.1) → donor (1.0,1.6)"),
    ("MATH-500", "exact_match", "mathins_m500-dream-ins-09aD-00aA-plain", "mathins_m500-dream-ins-09aD-11aA-plain", "host (0.9,0.0) → donor (0.9,1.1)"),
    ("MATH-500", "exact_match", "mathins_m500-dream-ins-095aD-055aA-plain", "mathins_m500-dream-ins-09aD-11aA-plain", "host (0.95,0.55) → donor (0.9,1.1)"),
]
rng = random.Random(0)
res = []
for bench, key, h, d, label in PAIRS:
    H, Dn, C = correct(h, key), correct(d, key), confidence(h)
    ids = sorted(set(H) & set(Dn) & set(C))
    n = len(ids)
    rec = dict(bench=bench, label=label, n=n, host=100 * sum(H[i] for i in ids) / n, donor=100 * sum(Dn[i] for i in ids) / n,
               oracle=100 * sum(H[i] or Dn[i] for i in ids) / n, curves={})
    for sig in ["mean", "p10", "min", "negent", "negent_max"]:
        order = sorted(ids, key=lambda i: C[i][sig])  # least confident first
        curve = []
        for r in RATES:
            acc = 100 * route(H, Dn, order, r)
            rand = [100 * route(H, Dn, rng.sample(ids, n), r) for _ in range(1000)]
            mu = sum(rand) / len(rand)
            p = sum(x >= acc for x in rand) / len(rand)
            curve.append(dict(r=r, acc=round(acc, 2), random=round(mu, 2), p=p))
        # 2-fold CV of the rerun rate
        cv = []
        for _ in range(200):
            sh = ids[:]
            rng.shuffle(sh)
            A, B = sh[: n // 2], sh[n // 2:]
            tot = 0
            for tr, te in [(A, B), (B, A)]:
                otr = sorted(tr, key=lambda i: C[i][sig])
                best_r = max(RATES, key=lambda r: (route(H, Dn, otr, r), -r))
                ote = sorted(te, key=lambda i: C[i][sig])
                tot += route(H, Dn, ote, best_r) * len(te)
            cv.append(100 * tot / n)
        rec["curves"][sig] = dict(curve=curve, cv_acc=round(sum(cv) / len(cv), 2))
    res.append(rec)
    best = max(rec["curves"]["mean"]["curve"], key=lambda c: c["acc"])
    print(f"{bench:8s} {label:36s} host {rec['host']:.1f} donor {rec['donor']:.1f} oracle {rec['oracle']:.1f} | "
          f"best r={best['r']} acc {best['acc']} (random {best['random']}, p={best['p']:.3f}) | "
          "CV " + " ".join(f"{k} {v['cv_acc']}" for k, v in rec["curves"].items()))
(FIGS / "confidence_rerun_sim.json").write_text(json.dumps(res, indent=1))

INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": BASE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, len(res), figsize=(3.6 * len(res), 3.6), sharex=True)
for ax, rec in zip(axes, res):
    c = rec["curves"]["mean"]["curve"]
    xs = [100 * q["r"] for q in c]
    ax.plot(xs, [q["random"] for q in c], color=MUTED, linewidth=2, linestyle="--", label="random rerun")
    ax.plot(xs, [q["acc"] for q in c], color="#2a78d6", linewidth=2, marker="o", markersize=4, label="rerun lowest-confidence")
    ax.axhline(rec["host"], color=BASE, linewidth=1)
    ax.set_title(f"{rec['bench']}: {rec['label']}\nhost {rec['host']:.1f} · donor {rec['donor']:.1f} · oracle {rec['oracle']:.1f}",
                 loc="left", fontsize=8.5)
    ax.set_xlabel("% of items rerun (= extra compute)")
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_facecolor(SURF)
axes[0].set_ylabel("accuracy")
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.patch.set_facecolor(SURF)
fig.tight_layout()
fig.savefig(FIGS / "confidence_rerun_sim.png", dpi=160, bbox_inches="tight", facecolor=SURF)
print(FIGS / "confidence_rerun_sim.png")
