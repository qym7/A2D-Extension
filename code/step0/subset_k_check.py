"""Random k-problem subset vs full set: does the trend hold? Zero GPU. Generalises subset50_check.py.
Pool A: 12 systems done on MATH-500 ids 0..199 (@1024). Pool B: systems done on all 500 (@1024).
For 2000 random subsets: Spearman, pairwise order agreement (by size of the full-set gap), top kept.
Also: how often a subset reproduces "M(.4,.6)<=M(0,1)@0.7 beats M(.4,.6)" (full 500: +19, p=0.09).
"""
import json, random, sys
from pathlib import Path
L = Path("/mnt/lts4/scratch/home/yqin/DllmAR/new_ideas/idea1_selective_distill/outputs")
K = int(sys.argv[1]) if len(sys.argv) > 1 else 100
def tx(p):
    f = L / p / "texts.jsonl"
    return {r["id"]: int(r["ok"]) for r in map(json.loads, open(f))} if f.exists() else {}
S = {"Ins": "step0/lockstep1024/math500/parent-free", "M(.4,.6)": "step0/lockstep1024/math500/merge-free", "M(0,1)": "step0/lockstep1024/math500/l10-free",
     "A(1.1)": "step0/lockstep1024/math500/add11-free", "A(2.0)": "step0/lockstep1024/math500/add20-free",
     "M(.4,.6)⇐M(0,1)@0.3": "step1/gated1024/math500/e1b-mrg10-t03", "M(.4,.6)⇐M(0,1)@0.7": "step1/gated1024/math500/e1b-mrg10-t07"}
for T, d in [("A(1.1)", "step1/gated_ins_add-a10-b11_1024"), ("A(2.0)", "step1/gated_ins_add-a10-b20_1024"), ("M(.4,.6)", "step1/gated_ins_mrg0406_1024")]:
    for t in ["t02", "t04", "t06"]:
        S[f"Ins⇐{T}@0.{t[2]}"] = f"{d}/math500/{t}"
R = {k: tx(p) for k, p in S.items()}
def spear(a, b):
    def rk(x):
        o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]: j += 1
            for q in range(i, j + 1): r[o[q]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rk(a), rk(b); n = len(a); ma, mb = sum(ra) / n, sum(rb) / n
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** .5
q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
for pool, N in [("A: ids 0..199", 200), ("B: ids 0..499", 500)]:
    names = [k for k, v in R.items() if all(i in v for i in range(N))]
    full = {k: 100 * sum(R[k][i] for i in range(N)) / N for k in names}
    rng = random.Random(1); rh, ag, top = [], [], 0; gap = {b: [0, 0] for b in [(0, 3), (3, 6), (6, 10), (10, 100)]}
    for _ in range(2000):
        ids = rng.sample(range(N), K); sub = {k: 100 * sum(R[k][i] for i in ids) / K for k in names}
        rh.append(spear([full[k] for k in names], [sub[k] for k in names]))
        pr = [(a, b) for a in names for b in names if a < b and full[a] != full[b]]
        ag.append(sum((full[a] - full[b]) * (sub[a] - sub[b]) > 0 for a, b in pr) / len(pr))
        top += max(names, key=sub.get) == max(names, key=full.get)
        for a, b in pr:
            for (lo, hi), c in gap.items():
                if lo <= abs(full[a] - full[b]) < hi: c[0] += (full[a] - full[b]) * (sub[a] - sub[b]) > 0; c[1] += 1
    print(f"\n## k={K} from pool {pool}: {len(names)} systems ({', '.join(names)})")
    print(f"Spearman median {q(rh,.5):.2f} (5–95%: {q(rh,.05):.2f}–{q(rh,.95):.2f}); pairwise agreement median {100*q(ag,.5):.0f}% "
          f"(5–95%: {100*q(ag,.05):.0f}–{100*q(ag,.95):.0f}%); top kept {100*top/2000:.0f}%")
    print("order kept by full-set gap: " + "; ".join(f"{lo}-{hi}pp {100*c[0]/c[1]:.0f}% ({c[1]//2000} pairs)" for (lo, hi), c in gap.items() if c[1]))
    if N == 500:
        g, m = R["M(.4,.6)⇐M(0,1)@0.7"], R["M(.4,.6)"]; rng = random.Random(2); win = sig = 0
        for _ in range(2000):
            ids = rng.sample(range(500), K); d = sum(g[i] - m[i] for i in ids); win += d > 0
        print(f"M(.4,.6)⇐M(0,1)@0.7 > M(.4,.6) on a random {K}-subset: {100*win/2000:.0f}% of subsets (full 500: 252 vs 233)")
